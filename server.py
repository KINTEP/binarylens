"""
BinaryLens API Server v1.3
==========================
Run:  python server.py
Open: http://localhost:5000

Requirements:
    pip install flask flask-cors numpy gunicorn

Changes in v1.3:
    - Lazy model loading (fixes OOM on 512MB free tier)
    - Only loads consensus.pkl + 4grams on startup (~50MB)
    - Full models loaded on-demand only (for corpus/add)
    - Single gunicorn worker to minimise RAM
"""

import os, struct, pickle, hashlib, json
import math, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Framework signatures ──────────────────────────────────────
SIGS = {
    'MSVC / CRT':           (-1.747, 0.150),
    'InnoSetup / MSVC':     (-1.849, 0.150),
    'NSIS Installer':       (-2.183, 0.284),
    'WiX Bootstrapper':     (-2.369, 0.201),
    'Custom Bootstrapper':  (-2.493, 0.150),
    'NSIS Compressed':      (-2.496, 0.335),
    'Packed / Encrypted':   (-5.796, 0.300),
    'Driver / CFG binary':  (-6.898, 0.300),
    'MSVC / CRT x64':       (-6.157, 0.080),
}

SECTION_HINTS = {
    'CPADinfo': 'Chrome packer',
    '.wixburn': 'WiX installer',
    '.ndata':   'NSIS installer',
    '.itext':   'InnoSetup',
    '.gfids':   'CFG enabled',
    '.giats':   'CFG IAT table',
    '.tls':     'TLS callbacks',
    '.pdata':   'x64 binary',
}


def classify_binary(lp, sections, fn_count=0, sigs=None):
    if sigs is None:
        sigs = SIGS
    sec_set = set(sections)

    if 'CPADinfo' in sec_set:  return 'Packed / Encrypted'
    if '.wixburn' in sec_set:  return 'WiX Bootstrapper'
    if '.gfids'   in sec_set:  return 'Driver / CFG binary'
    if '.pdata'   in sec_set:  return 'MSVC / CRT x64'

    if '.itext' in sec_set:
        if '.didata' in sec_set and '.reloc' not in sec_set:
            return 'MSVC / CRT'
        return 'InnoSetup / MSVC'

    if '.ndata' in sec_set:
        return 'NSIS Compressed' if fn_count == 0 else 'NSIS Installer'

    if fn_count >= 100:
        return 'Custom Bootstrapper'

    if 0 < fn_count < 100:
        minimal = {'.text', '.rdata', '.data',
                   '.rsrc', '.reloc', '.tls', '.bss'}
        if sec_set <= minimal:
            return 'WiX Bootstrapper'

    if lp < -7.5 and '.gfids' not in sec_set:
        return 'Packed / Encrypted'

    best, best_d = 'Unknown', float('inf')
    for fw, (mu, sigma) in sigs.items():
        d = abs(lp - mu) / sigma
        if d < best_d:
            best_d = d
            best   = fw

    if best == 'Driver / CFG binary' and '.gfids' not in sec_set:
        best = 'Packed / Encrypted'

    return best


# ── CorpusDB (lazy loading) ───────────────────────────────────

class CorpusDB:
    def __init__(self, path='corpus'):
        self.path      = Path(path)
        self.index     = {}
        self.models    = {}   # loaded on-demand only
        self.fg        = {}   # loaded at startup (small)
        self.consensus = None
        self._load()

    def _load(self):
        self.path.mkdir(exist_ok=True)
        (self.path / 'models').mkdir(exist_ok=True)
        (self.path / '4grams').mkdir(exist_ok=True)

        idx = self.path / 'index.json'
        if idx.exists():
            self.index = json.loads(idx.read_text())
            print(f"  Index: {len(self.index)} binaries")

        con = self.path / 'consensus.pkl'
        if con.exists():
            self.consensus = pickle.loads(con.read_bytes())
            ctx = len(self.consensus['p3'])
            print(f"  Consensus: {ctx} contexts")

    def load_4grams_only(self):
        """Load only 4gram sets — small, needed for similarity."""
        loaded = 0
        for sha in self.index:
            fp = self.path / '4grams' / f'{sha}.pkl'
            if fp.exists() and sha not in self.fg:
                try:
                    self.fg[sha] = pickle.loads(fp.read_bytes())
                    loaded += 1
                except Exception:
                    pass
        print(f"  4grams: {loaded} loaded")
        return loaded

    def load_model(self, sha):
        """Load a single model on demand."""
        if sha in self.models:
            return self.models[sha]
        mp = self.path / 'models' / f'{sha}.pkl'
        if mp.exists():
            self.models[sha] = pickle.loads(mp.read_bytes())
            return self.models[sha]
        return None

    def save(self):
        (self.path / 'index.json').write_text(
            json.dumps(self.index, indent=2))
        if self.consensus:
            (self.path / 'consensus.pkl').write_bytes(
                pickle.dumps(self.consensus))

    def contains(self, sha):
        return sha in self.index

    def add(self, sha, meta, model, fg):
        self.index[sha]  = meta
        self.models[sha] = model
        self.fg[sha]     = fg
        (self.path / 'models' / f'{sha}.pkl').write_bytes(
            pickle.dumps(model))
        (self.path / '4grams' / f'{sha}.pkl').write_bytes(
            pickle.dumps(fg))

    def rebuild_consensus(self, exclude_fw=None):
        """Rebuild consensus — loads models on demand."""
        if exclude_fw is None:
            exclude_fw = {'Packed / Encrypted',
                          'Driver / CFG binary',
                          'MSVC / CRT x64'}
        fw_groups = defaultdict(list)
        for sha, meta in self.index.items():
            fw   = meta.get('label') or meta.get('framework', '?')
            arch = meta.get('arch', 'x86')
            if fw in exclude_fw or arch != 'x86':
                continue
            fw_groups[fw].append(sha)

        if not fw_groups:
            return None

        total_w = len(fw_groups)
        c3 = defaultdict(Counter)
        c2 = defaultdict(Counter)
        c1 = defaultdict(Counter)

        for fw, shas in fw_groups.items():
            fw_w  = 1.0 / total_w
            bin_w = fw_w / len(shas)
            for sha in shas:
                m = self.load_model(sha)
                if not m:
                    continue
                for ctx, d in m['p3'].items():
                    for b, p in d.items():
                        c3[ctx][b] += p * bin_w
                for ctx, d in m['p2'].items():
                    for b, p in d.items():
                        c2[ctx][b] += p * bin_w
                for ctx, d in m['p1'].items():
                    for b, p in d.items():
                        c1[ctx][b] += p * bin_w

        def norm(c):
            return {ctx: {k: v / sum(d.values())
                          for k, v in d.items()}
                    for ctx, d in c.items()}

        self.consensus = {
            'p3': norm(c3),
            'p2': norm(c2),
            'p1': norm(c1),
        }
        # Free model RAM after rebuild
        self.models.clear()
        return self.consensus


# ── Analysis helpers ──────────────────────────────────────────

def build_models(data):
    def build(n):
        counts = defaultdict(Counter)
        for i in range(len(data) - n):
            ctx = ",".join(map(str, data[i:i+n]))
            counts[ctx][data[i+n]] += 1
        return {ctx: {k: v / sum(c.values()) for k, v in c.items()}
                for ctx, c in counts.items()}
    p1r = defaultdict(Counter)
    for i in range(len(data) - 1):
        p1r[data[i]][data[i+1]] += 1
    p1 = {b: {k: v / sum(c.values()) for k, v in c.items()}
          for b, c in p1r.items()}
    return {'p3': build(3), 'p2': build(2), 'p1': p1}


def build_4grams(data):
    return set(tuple(data[i:i+4]) for i in range(len(data) - 3))


def score_logp(block, c3, c2, c1):
    lps = []
    for i in range(3, len(block)):
        b  = block[i]
        k3 = ",".join(map(str, block[i-3:i]))
        k2 = ",".join(map(str, block[i-2:i]))
        c  = block[i-1]
        if   k3 in c3 and b in c3[k3]: p = c3[k3][b]
        elif k2 in c2 and b in c2[k2]: p = c2[k2][b]
        elif c  in c1 and b in c1[c]:  p = c1[c][b]
        else:                          p = 1 / 256
        lps.append(math.log2(max(p, 1e-10)))
    return float(np.mean(lps)) if lps else 0.0


def parse_pe(raw):
    try:
        if raw[:2] != b'MZ':
            return None
        pe_off  = struct.unpack_from('<I', raw, 0x3C)[0]
        if raw[pe_off:pe_off+4] != b'PE\x00\x00':
            return None
        machine = struct.unpack_from('<H', raw, pe_off+4)[0]
        nsec    = struct.unpack_from('<H', raw, pe_off+6)[0]
        optsz   = struct.unpack_from('<H', raw, pe_off+20)[0]
        magic   = struct.unpack_from('<H', raw, pe_off+24)[0]
        sec_off = pe_off + 24 + optsz
        arch    = {0x014C:'x86', 0x8664:'x64',
                   0x01C0:'ARM', 0xAA64:'ARM64'}.get(
                       machine, f'0x{machine:04X}')
        pe_type = {0x010B:'PE32', 0x020B:'PE32+'}.get(magic, 'unknown')

        sec_names = []
        text_off = text_size = best_exec_off = best_exec_size = 0

        for i in range(nsec):
            s     = sec_off + i * 40
            sname = raw[s:s+8].rstrip(b'\x00').decode('ascii', 'replace')
            rsz   = struct.unpack_from('<I', raw, s+16)[0]
            roff  = struct.unpack_from('<I', raw, s+20)[0]
            chars = struct.unpack_from('<I', raw, s+36)[0]
            sec_names.append(sname)
            if sname == '.text':
                text_off  = roff
                text_size = rsz
            if (chars & 0x20000000) and rsz > best_exec_size:
                best_exec_size = rsz
                best_exec_off  = roff

        if text_size == 0 and best_exec_size > 0:
            text_off  = best_exec_off
            text_size = best_exec_size

        return arch, pe_type, sec_names, text_off, text_size
    except Exception:
        return None


def analyse_text(data, sec_names, arch, pe_type,
                 filename, consensus, corpus_fg):
    t0 = time.time()

    hints  = [v for k, v in SECTION_HINTS.items() if k in sec_names]
    is_x64 = (pe_type == 'PE32+')

    c3 = consensus['p3']
    c2 = consensus['p2']
    c1 = consensus['p1']

    window, step = 256, 64
    win_scores   = []
    for off in range(0, len(data) - window, step):
        block = data[off:off+window]
        if block.count(0) / window > 0.3:
            continue
        if max(Counter(block).values()) / window > 0.4:
            continue
        lp = score_logp(list(block), c3, c2, c1)
        win_scores.append({'offset': off, 'logp': round(lp, 4)})

    lps     = [w['logp'] for w in win_scores]
    mean_lp = float(np.mean(lps)) if lps else 0.0
    std_lp  = float(np.std(lps))  if lps else 1.0
    std_lp  = std_lp or 1.0
    thresh  = mean_lp - 1.5 * std_lp

    for w in win_scores:
        w['z']    = round((w['logp'] - mean_lp) / std_lp, 3)
        w['flag'] = bool(w['logp'] < thresh)

    fn_count = sum(
        1 for i in range(len(data) - 2)
        if data[i] == 0x55 and data[i+1] == 0x8B
        and data[i+2] == 0xEC
    )
    fw = classify_binary(mean_lp, sec_names, fn_count)

    fg   = build_4grams(data)
    sims = {name: round(len(fg & cfg) / max(len(fg), 1) * 100, 1)
            for name, cfg in corpus_fg.items()}
    sims = dict(sorted(sims.items(), key=lambda x: -x[1]))

    n_anom   = sum(1 for w in win_scores if w['flag'])
    anom_pct = round(100 * n_anom / max(len(win_scores), 1), 2)
    max_sim  = max(sims.values()) if sims else 0

    risk, reasons = 0, []
    if mean_lp < -5.0:
        risk += 2; reasons.append('Very low entropy (possible packing)')
    elif mean_lp < -3.0:
        risk += 1; reasons.append('Below normal logP range')
    if anom_pct > 20:
        risk += 2; reasons.append(f'{anom_pct:.0f}% anomalous windows')
    elif anom_pct > 10:
        risk += 1; reasons.append(f'{anom_pct:.0f}% anomalous windows')
    if max_sim < 5:
        risk += 2; reasons.append('No corpus match (<5%)')
    elif max_sim < 15:
        risk += 1; reasons.append(f'Weak corpus match ({max_sim:.0f}%)')
    if fn_count == 0 and not is_x64:
        risk += 1; reasons.append('No function prologues detected')
    if is_x64:
        risk = max(risk - 1, 0)
        reasons.append('x64 binary — x86 model applied')

    risk_labels = ['LOW','LOW','MODERATE','MODERATE','HIGH','CRITICAL']
    risk_str    = risk_labels[min(risk, 5)]

    return {
        'filename':    filename,
        'arch':        arch,
        'pe_type':     pe_type,
        'sections':    sec_names,
        'hints':       hints,
        'is_x64':      is_x64,
        'text_size':   len(data),
        'functions':   fn_count,
        'framework':   fw,
        'mean_logp':   round(mean_lp, 4),
        'std_logp':    round(std_lp,  4),
        'anomaly_pct': anom_pct,
        'risk':        risk_str,
        'risk_score':  risk,
        'risk_reasons':reasons,
        'similarity':  sims,
        'nearest':     next(iter(sims), 'none'),
        'windows':     win_scores,
        'elapsed_sec': round(time.time() - t0, 3),
        'corpus_size': len(corpus_fg),
    }


# ── Boot — lazy loading ───────────────────────────────────────
print("BinaryLens v1.3 starting...")
db = CorpusDB('corpus')

if db.consensus is None:
    print("  No consensus found — rebuilding (this takes a moment)...")
    db.rebuild_consensus()
    db.save()

# Load ONLY 4grams — ~50MB total, not 200MB+ for full models
db.load_4grams_only()

# Build similarity index from loaded 4grams
corpus_fg = {
    db.index[sha]['name']: db.fg[sha]
    for sha in db.fg
    if sha in db.index
}

n_ctx = len(db.consensus['p3']) if db.consensus else 0
print(f"Ready: {len(db.index)} binaries | "
      f"{n_ctx} contexts | "
      f"{len(corpus_fg)} similarity entries")


# ── Routes ────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/health')
def health():
    return jsonify({
        'status':             'ok',
        'corpus_size':        len(db.index),
        'consensus_contexts': n_ctx,
        'version':            '1.3',
    })


@app.route('/analyse', methods=['POST'])
def analyse():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f   = request.files['file']
    raw = f.read()
    if len(raw) < 64:
        return jsonify({'error': 'File too small'}), 400
    pe = parse_pe(raw)
    if not pe:
        return jsonify({'error': 'Not a valid PE file'}), 400
    arch, pe_type, sec_names, text_off, text_size = pe
    if not text_size:
        return jsonify({'error': 'No executable section found'}), 400
    data   = raw[text_off: text_off + min(text_size, 200_000)]
    result = analyse_text(data, sec_names, arch, pe_type,
                          f.filename, db.consensus, corpus_fg)
    return jsonify(result)


@app.route('/batch', methods=['POST'])
def batch():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files'}), 400
    results = []
    for f in files:
        raw = f.read()
        pe  = parse_pe(raw)
        if not pe:
            results.append({'filename': f.filename,
                            'error': 'Not a PE file'})
            continue
        arch, pe_type, sec_names, text_off, text_size = pe
        if not text_size:
            results.append({'filename': f.filename,
                            'error': 'No executable section'})
            continue
        data = raw[text_off: text_off + min(text_size, 200_000)]
        r    = analyse_text(data, sec_names, arch, pe_type,
                            f.filename, db.consensus, corpus_fg)
        results.append(r)
    results.sort(key=lambda x: -x.get('risk_score', 0))
    return jsonify({'count': len(results), 'results': results})


@app.route('/corpus')
def corpus_info():
    fw_counts = Counter(
        m.get('label', m['framework'])
        for m in db.index.values()
    )
    return jsonify({
        'total':      len(db.index),
        'frameworks': dict(fw_counts),
        'signatures': {k: {'mean': v[0], 'std': v[1]}
                       for k, v in SIGS.items()},
        'consensus_contexts': n_ctx,
    })


@app.route('/corpus/add', methods=['POST'])
def corpus_add():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f     = request.files['file']
    label = request.form.get('label')
    raw   = f.read()
    sha   = hashlib.sha256(raw).hexdigest()[:16]

    if db.contains(sha):
        return jsonify({'sha256': sha, 'added': False,
                        'reason': 'already in corpus'})

    pe = parse_pe(raw)
    if not pe:
        return jsonify({'error': 'Not a valid PE'}), 400
    arch, pe_type, sec_names, text_off, text_size = pe
    if not text_size:
        return jsonify({'error': 'No executable section'}), 400

    data     = raw[text_off: text_off + min(text_size, 200_000)]
    model    = build_models(data)
    fg       = build_4grams(data)
    fn_count = sum(1 for i in range(len(data) - 2)
                   if data[i] == 0x55 and data[i+1] == 0x8B
                   and data[i+2] == 0xEC)

    lp = 0.0
    if db.consensus:
        c3, c2, c1 = (db.consensus['p3'],
                      db.consensus['p2'],
                      db.consensus['p1'])
        lps = []
        for off in range(0, len(data) - 256, 64):
            block = data[off:off+256]
            if block.count(0) / 256 > 0.3: continue
            if max(Counter(block).values()) / 256 > 0.4: continue
            lps.append(score_logp(list(block), c3, c2, c1))
        if lps:
            lp = float(np.mean(lps))

    hints = [v for k, v in SECTION_HINTS.items() if k in sec_names]
    fw    = label or classify_binary(lp, sec_names, fn_count)

    meta = {
        'name': f.filename, 'path': f.filename,
        'size': len(raw),   'text_size': len(data),
        'arch': arch,       'pe_type': pe_type,
        'sections': sec_names, 'hints': hints,
        'functions': fn_count, 'mean_lp': round(lp, 4),
        'framework': fw,    'label': fw, 'sha256': sha,
    }
    db.add(sha, meta, model, fg)
    corpus_fg[f.filename] = fg
    # Rebuild consensus using lazy model loading
    db.rebuild_consensus()
    db.save()

    return jsonify({'sha256': sha, 'added': True,
                    'framework': fw,
                    'corpus_size': len(db.index)})


@app.route('/compare', methods=['POST'])
def compare():
    body = request.get_json() or {}
    a, b = body.get('sha256_a'), body.get('sha256_b')
    if not a or not b:
        return jsonify({'error': 'Need sha256_a and sha256_b'}), 400
    if a not in db.fg or b not in db.fg:
        return jsonify({'error': 'SHA not in corpus'}), 404
    sim = len(db.fg[a] & db.fg[b]) / max(len(db.fg[a]), 1)
    return jsonify({
        'sha256_a': a, 'name_a': db.index[a]['name'],
        'sha256_b': b, 'name_b': db.index[b]['name'],
        'similarity': round(sim * 100, 2),
    })


@app.route('/signatures')
def signatures():
    return jsonify({k: {'mean': v[0], 'std': v[1], 'label': k}
                    for k, v in SIGS.items()})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*52}")
    print(f"  BinaryLens  v1.3")
    print(f"{'='*52}")
    print(f"  Corpus:    {len(db.index)} binaries")
    print(f"  Consensus: {n_ctx} contexts")
    print(f"  Open:      http://localhost:{port}")
    print(f"{'='*52}\n")
    app.run(debug=False, host='0.0.0.0', port=port)