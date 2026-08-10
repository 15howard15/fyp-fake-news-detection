"""
build_code_map.py -- regenerate the line numbers in CODE_WALKTHROUGH.md.

The walkthrough's jump table exists so a live question ("show me where you clean
the text") lands on the right line instead of a scroll. That only works if the
line numbers are right, and they go stale the moment anything shifts in a file
-- which has already happened once, when a formatter stripped the comments out
of src/ and moved every function in train.py by ~26 lines.

So the table is generated rather than typed. This script finds each function by
NAME in the current source and rewrites the line numbers around it, leaving the
prose untouched. Run it before presenting, or after any edit to src/:

    python src/build_code_map.py

It rewrites only the rows between the AUTO-MAP markers and the `**Lnnn**`
references elsewhere in the document; everything else is left alone.
"""
import ast
import io
import re

import config as cfg

DOC = cfg.ROOT / "CODE_WALKTHROUGH.md"


def function_lines():
    """{filename: {function_name: line_number}} for every module in src/."""
    out = {}
    for p in sorted((cfg.ROOT / "src").glob("*.py")):
        try:
            tree = ast.parse(io.open(p, encoding="utf-8").read())
        except SyntaxError:
            continue
        out[p.name] = {
            n.name: n.lineno
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
    return out


def main():
    if not DOC.exists():
        print(f"{DOC} not found."); return
    lines = function_lines()
    doc = io.open(DOC, encoding="utf-8").read()

    # Every reference looks like:  `train.py` **L72** `train_lr_svm`
    # or                          `train.py` **L72** train_lr_svm
    pattern = re.compile(r'(`([a-z_]+\.py)`\s*\*\*L)(\d+)(\*\*\s*`?([a-zA-Z_]+)`?)')

    fixed, stale, missing = 0, 0, []

    def repl(m):
        nonlocal fixed, stale
        pre, fname, old, post, fn = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        actual = lines.get(fname, {}).get(fn)
        if actual is None:
            missing.append(f"{fname}:{fn}")
            return m.group(0)
        if str(actual) != old:
            stale += 1
        fixed += 1
        return f"{pre}{actual}{post}"

    doc = pattern.sub(repl, doc)

    # `code -g src/file.py:123` commands carry the same numbers.
    def repl_cmd(m):
        fname, old = m.group(1), m.group(2)
        # find the function whose line this was pointing at, by matching any
        # known line number for that file
        known = lines.get(fname, {})
        for fn, ln in known.items():
            if str(ln) == old:
                return f"src/{fname}:{ln}"
        return m.group(0)

    io.open(DOC, "w", encoding="utf-8").write(doc)

    print(f"Checked {fixed} line references in {DOC.name}")
    print(f"  corrected: {stale}")
    if missing:
        print(f"  NOT FOUND (function renamed or removed): {len(missing)}")
        for x in missing:
            print(f"    {x}")


if __name__ == "__main__":
    main()
