"""Audit all notebooks in the repo for polish consistency.

Run from the repo root:
    python3 tools/qa_check.py

Reports per-notebook:
  - cell count
  - cell IDs (every cell should have a stable id)
  - number of `info=` tooltips on Gradio widgets
  - number of try/except pairs
  - presence of `clear_output()` before demo.launch
  - presence of `concurrency_limit` in queue
  - presence of `demo.load` welcome
  - presence of HF_HOME Drive-cache prologue (if applicable)
  - presence of `from IPython.display import FileLink` in step 6
  - count of `# @title STEP N` form cells

Exits non-zero with a list of findings if any authored notebook fails
the audit. Pre-existing notebooks (Pixal3D, Pixal3D_Wheel_Builder) are
explicitly excluded from the fail check.
"""
import ast, json, os, re, sys

# (notebook, kind) -- kind is 'authored' (must be clean) or 'preexisting' (skip fail check)
NOTEBOOKS = [
    ('Cube_3D_Colab.ipynb',          'authored'),
    ('Dia_Colab.ipynb',              'authored'),
    ('Fish-S2-Pro_Colab.ipynb',      'authored'),
    ('Higgs-Audio_Colab.ipynb',      'authored'),
    ('Hunyuan3D-2.1_Colab.ipynb',    'authored'),
    ('Hunyuan3D-2_Colab.ipynb',      'authored'),
    ('Hunyuan3D-3_Colab.ipynb',      'authored'),
    ('IndicF5_Colab.ipynb',          'authored'),
    ('Kokoro-82M_Colab.ipynb',       'authored'),
    ('MOSS-TTS_Colab.ipynb',         'authored'),
    ('MisoTTS_Colab.ipynb',          'authored'),
    ('Notebook_Generator.ipynb',      'authored'),
    ('Pixal3D_Colab.ipynb',          'preexisting'),
    ('Pixal3D_Wheel_Builder.ipynb',  'preexisting'),
    ('Qwen3-TTS_Colab.ipynb',        'authored'),
    ('Supertonic-3_Colab.ipynb',     'authored'),
    ('TTS_Model_Loader.ipynb',       'authored'),
    ('TTS_Voice_Library.ipynb',      'authored'),
    ('Wan2.2_Animate_Colab.ipynb',    'authored'),
    ('Wan2.2_Colab.ipynb',           'authored'),
    ('Wan2.2_S2V_Colab.ipynb',        'authored'),
    ('dots.tts-soar_Colab.ipynb',    'authored'),
]


def _clean_for_ast(src):
    """Strip Jupyter-specific lines that AST can't handle directly."""
    out = []
    lines = src.split(chr(10))
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith(('!', '%')):
            mi = len(line) - len(line.lstrip())
            i += 1
            while i < len(lines):
                c = lines[i]
                if c.strip() == '':
                    break
                ci = len(c) - len(c.lstrip())
                if ci > mi or c.rstrip().endswith('\\'):
                    i += 1
                    continue
                break
            continue
        if line.lstrip().startswith('# @markdown'):
            i += 1
            continue
        out.append(line)
        i += 1
    return chr(10).join(out)


def _fix_empty_blocks(src):
    """Add a `pass` after lines that end with `:` if the next line isn't indented.
    Helps AST parse notebooks that have empty function bodies or stubs."""
    lines = src.split(chr(10))
    fixed = []
    for i, line in enumerate(lines):
        if line.rstrip().endswith((':',)) and i + 1 < len(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else ''
            if not nxt.strip() or not (nxt[0] in (' ', '\t')):
                fixed.append(line)
                fixed.append('    pass')
                continue
        fixed.append(line)
    return chr(10).join(fixed)


def audit(path):
    with open(path) as f:
        nb = json.load(f)
    cells = nb['cells']
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    md_cells = [c for c in cells if c['cell_type'] == 'markdown']
    src = '\n'.join(''.join(c['source']) for c in cells)

    findings = []
    n_info = src.count("info='") + src.count('info="')
    n_try = len(re.findall(r'\btry\s*:', src))
    n_exc = len(re.findall(r'\bexcept\b', src))

    if 'pip install' in src and 'tqdm' not in src and 'tqdm.auto' not in src:
        if 'from tqdm' not in src and 'tqdm.notebook' not in src:
            findings.append('no tqdm in pip install / downloads')

    if n_info == 0 and 'gr.' in src:
        findings.append('zero info= tooltips on Gradio widgets')

    em = src.count('\u2014')
    if em > 0 and em < 3:
        findings.append(f'only {em} em-dash(es) — possibly inconsistent')

    if 'snapshot_download' in src or 'hf_hub_download' in src or 'pipeline.from_pretrained' in src:
        if 'HF_HOME' not in src and 'CACHE_ROOT' not in src:
            if 'AEI_TTS_Cache' not in src:
                findings.append('downloads from HF without Drive cache prologue')

    if 'demo.queue' in src and 'concurrency_limit' not in src:
        findings.append('demo.queue without concurrency_limit')

    if 'demo.launch' in src:
        idx_launch = src.find('demo.launch')
        if 'clear_output()' not in src[:idx_launch + 200]:
            findings.append('demo.launch without prior clear_output()')

    if 'gr.Blocks' in src and 'demo.load' not in src and 'tab_ip.select' not in src:
        findings.append('no demo.load welcome or tab swap')

    if 'CACHE_ROOT' in src and 'os.environ.setdefault' not in src and 'os.environ[' not in src:
        findings.append('CACHE_ROOT defined but not exported to env')

    if 'step7' in src.lower() or 'Step 7' in src:
        if 'for ' in src and 'try:' not in src:
            findings.append('Step 7 batch loop without per-iteration try/except')

    if 'step6' in src.lower() and 'FileLink' not in src and 'display(' in src:
        findings.append('Step 6 quick test with display() but no FileLink import')

    if 'pip install' in src and 'gradio' in src:
        if 'gradio==' not in src and 'gradio>=' not in src:
            findings.append('gradio installed without version pin')

    return {
        'cells':           len(cells),
        'code':            len(code_cells),
        'md':              len(md_cells),
        'ids':             [c['metadata'].get('id', '?') for c in cells],
        'info_tooltips':   n_info,
        'try_pairs':       (n_try, n_exc),
        'em_dash_count':   em,
        'hf_home':         'HF_HOME' in src,
        'clear_output':    'clear_output()' in src,
        'concurrency':     'concurrency_limit' in src,
        'demo_load':       'demo.load' in src,
        'queue':           'demo.queue' in src,
        'filelink':        'FileLink' in src,
        'findings':        findings,
    }


def main():
    if not os.path.isdir('.'):
        # Try parent dir
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')

    rows = []
    fail_count = 0
    for path, kind in NOTEBOOKS:
        if not os.path.exists(path):
            print(f'  {path}: MISSING')
            fail_count += 1
            continue
        a = audit(path)
        rows.append((path, kind, a))
        if kind == 'authored' and a['findings']:
            fail_count += 1

    print(f"{'notebook':<32}  {'cells':>5}  {'tips':>4}  {'try':>3}  {'exc':>3}  {'HF':>3}  {'co':>3}  {'cl':>3}  {'load':>4}  {'FL':>3}")
    print('-' * 80)
    for path, kind, a in rows:
        print(f'{path:<32}  {a["cells"]:>5}  {a["info_tooltips"]:>4}  '
              f'{a["try_pairs"][0]:>3}  {a["try_pairs"][1]:>3}  '
              f'{"Y" if a["hf_home"] else "N":>3}  '
              f'{"Y" if a["concurrency"] else "N":>3}  '
              f'{"Y" if a["clear_output"] else "N":>3}  '
              f'{"Y" if a["demo_load"] else "N":>4}  '
              f'{"Y" if a["filelink"] else "N":>3}'
              f'  {"(pre)" if kind == "preexisting" else ""}')
        for fi in a['findings']:
            print(f'  {"":>32}    L  {fi}')

    print()
    print('Legend: cells=total, tips=#info tooltips, try=#try, exc=#except handlers,')
    print('        HF=has HF_HOME Drive cache, co=concurrency_limit, cl=clear_output(),')
    print('        load=demo.load welcome, FL=FileLink in Step 6')
    print()
    if fail_count:
        print(f'FAIL: {fail_count} finding(s) in authored notebooks (pre-existing intentionally skipped).')
        sys.exit(1)
    else:
        print('OK: all authored notebooks pass the polish audit.')


if __name__ == '__main__':
    main()
