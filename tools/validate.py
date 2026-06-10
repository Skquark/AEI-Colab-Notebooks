"""Validate that all notebook cells parse as valid Python.

Faster, narrower check than qa_check.py — just AST-parses every code cell
and exits 0 on success, non-zero on failure. Designed to be wired into
CI / pre-commit hooks.

Run from the repo root:
    python3 tools/validate.py
"""
import ast, json, os, sys


def _clean(src):
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


def _fix_empty(src):
    """Insert 'pass' after any control-flow line (ending in ':') that has no
    indented body. A 'body' is defined as: a non-blank line below the control
    line that is indented strictly more than the smallest-indent non-blank
    line in the multi-line header (the 'def' / 'class' / 'if' / etc. line).
    Blank lines between control lines and their bodies are allowed."""
    lines = src.split(chr(10))
    fixed = []
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped.endswith((':',)) and i + 1 < len(lines):
            # Walk back through blank lines and equal-or-greater-indent
            # continuation lines (multi-line signatures) to find the effective
            # header indent. Continue scanning through equal-indent lines
            # (they're signature continuations) and stop only at a less-indent
            # non-blank line (the actual header) or the start of the source.
            header_indent = len(line) - len(line.lstrip())
            for j in range(i - 1, -1, -1):
                prev = lines[j]
                if not prev.strip():
                    continue
                prev_indent = len(prev) - len(prev.lstrip())
                if prev_indent > header_indent:
                    continue
                if prev_indent < header_indent:
                    header_indent = prev_indent
                # equal indent => signature continuation, keep walking
            has_body = False
            for j in range(i + 1, len(lines)):
                nxt = lines[j]
                if not nxt.strip():
                    continue
                nxt_indent = len(nxt) - len(nxt.lstrip())
                if nxt_indent > header_indent:
                    has_body = True
                break
            if not has_body:
                fixed.append(line)
                fixed.append(' ' * (header_indent + 4) + 'pass')
                continue
        fixed.append(line)
    return chr(10).join(fixed)


def validate(path):
    with open(path) as f:
        nb = json.load(f)
    cells = nb['cells']
    ids = [c['metadata'].get('id') for c in cells]
    # Standard 9-cell layout: view-in-github, header, step1..step7
    # Some notebooks legitimately have a different count (e.g. TTS_Model_Loader has 7,
    # TTS_Voice_Library has 10, Pixal3D_Wheel_Builder has 12). For those we only enforce
    # that the IDs follow a stepN-name pattern, not the exact 9 IDs.
    standard_9 = ['view-in-github', 'header', 'step1-install', 'step2-cache', 'step3-core',
                  'step4-ui', 'step5-keepalive', 'step6-quicktest', 'step7-batch']
    errors = []
    if len(cells) == 9:
        # Strict check: full 9-cell pattern
        if ids != standard_9:
            errors.append(f'9-cell notebook IDs: {ids} != {standard_9}')
    else:
        # Lenient check: every cell has a non-empty id, and the first 2 are standard
        if ids[0] != 'view-in-github':
            errors.append(f'first cell id is {ids[0]!r}, expected "view-in-github"')
        if ids[1] != 'header':
            errors.append(f'second cell id is {ids[1]!r}, expected "header"')
        for i, cid in enumerate(ids):
            if not cid or cid == '?':
                errors.append(f'cell {i} missing id')
    for i, c in enumerate(cells):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        try:
            ast.parse(_fix_empty(_clean(src)))
        except SyntaxError as e:
            errors.append(f'cell {i} ({c["metadata"].get("id", "?")}) SyntaxError at line {e.lineno}: {e.msg}')
    return errors


def main():
    if not os.path.isdir('.'):
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
    notebooks = sorted(f for f in os.listdir('.') if f.endswith('.ipynb'))
    if not notebooks:
        print('No notebooks found.')
        sys.exit(1)
    all_errors = {}
    for nb_path in notebooks:
        errs = validate(nb_path)
        if errs:
            all_errors[nb_path] = errs
    if all_errors:
        print(f'FAIL: {len(all_errors)} notebook(s) failed validation:')
        for path, errs in all_errors.items():
            print(f'  {path}:')
            for e in errs:
                print(f'    {e}')
        sys.exit(1)
    else:
        print(f'OK: all {len(notebooks)} notebook(s) parse cleanly.')


if __name__ == '__main__':
    main()
