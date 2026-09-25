"""Batch-driver wrapper for the CTA curation figures.

The figure logic lives in the package (``pirlygenes.cta_curation_plots``) and is
also exposed as ``pirlygenes plot cta-curation``. This wrapper just rides the
shared analyses run layout so the figures land in the timestamped run dir
alongside every other family and participate in
``regenerate_plots.py --promote-docs``.

    python analyses/cta_curation_figures.py            # -> analyses/outputs/run_*/
    python analyses/cta_curation_figures.py --out-dir <run> --run-name cta_curation
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from _run_layout import add_layout_args, resolve_dirs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_layout_args(ap)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.figure_environment import check_figure_environment
    from pirlygenes.cta_curation_plots import render
    from pirlygenes.version import __version__
    from pypdf import PdfWriter

    expected = check_figure_environment()
    _, figdir = resolve_dirs(args, Path(__file__).resolve().parent / "outputs")
    result = render(out_dir=figdir)
    check_figure_environment(expected)
    combined = figdir / "pirlygenes-cta-curation-figures.pdf"
    writer = PdfWriter()
    for key, path in result["paths"].items():
        writer.append(str(path.with_suffix(".pdf")), outline_item=key.replace("_", " ").title())
    writer.add_metadata({"/Title": "CTA paper provenance and curation"})
    writer.write(str(combined))
    writer.close()
    manifest_path = figdir / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pirlygenes_version"] = __version__
    manifest["outputs"][combined.name] = hashlib.sha256(combined.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    index = figdir / "index.md"
    index.write_text(index.read_text().replace(
        "# CTA source intake and curation\n",
        f"# CTA source intake and curation\n\n[All CTA figures, vector PDF]({combined.name})\n", 1,
    ))
    print(f"CTA curation figures from {result['n_genes']} evidence rows -> {figdir}")
    for path in result["paths"].values():
        print(f"  wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
