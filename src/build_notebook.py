"""Convert the percent-format script (sepsis_early_warning.py) into a Jupyter notebook.

Usage:
    python src/build_notebook.py            # build sepsis_early_warning.ipynb (no outputs)
    python src/build_notebook.py --execute  # build and run it top to bottom, saving outputs
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "sepsis_early_warning.py"
OUT = ROOT / "sepsis_early_warning.ipynb"


def parse_cells(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


nb = nbformat.v4.new_notebook()
nb.cells = parse_cells(SRC.read_text(encoding="utf-8"))
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

if "--execute" in sys.argv:
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()

nbformat.write(nb, OUT)
print(f"Wrote {OUT} ({len(nb.cells)} cells)")
