#!/usr/bin/env python
"""Regenerate **all** analyses plots into ONE timestamped run dir, organised by
plot family in subfolders:

    analyses/outputs/run_<YYYYMMDD-HHMMSS>/
        apd1_causal_model/          aggregate aPD1 causal-factor figures
            cta_vs_apd1_by_exclusion/
        apd1_response/              ORR bars, ORR-vs-TMB
        cta_<metric>_vs_<axis>/     CTA burden vs TMB / aPD1 / incidence / mortality
        cta_addressable/  cta_covering_set/  cta_expression_heatmaps/
        ...

This replaces the old per-script ``outputs/apd1_causal_factors/run_*`` layout
(which buried five scripts' output under one script's name) — every family now
lands in the same timestamped run dir.

    python analyses/regenerate_plots.py

Scripts that fail are reported but don't abort the batch.
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
DOCS = HERE.parent / "docs"

# Families whose output doubles as committed documentation assets: --promote-docs
# copies these into docs/ so the figures embedded in docs/*.md stay in sync.
# {run-subdir: glob}
DOCS_PROMOTE = {"cta_curation": "cta-*.png"}

# aPD1 causal-factor batch — driven by the APD1_RUN_DIR env var, all into one
# subfolder of the run (they already group cta_vs_apd1_by_exclusion/ themselves).
APD1_BATCH = ["exclusion_vs_apd1", "apd1_causal_factors", "apd1_mechanism_screen",
              "apd1_landscape", "apd1_exclusion_scatters"]

# argparse + _run_layout scripts. (script, extra_args, target):
#   target "groups" -> writes its own family subdirs straight under the run dir
#   target "<name>" -> a flat-writing script gets its own named subdir
LAYOUT = [
    ("cta_patient_counts", [], "groups"),
    ("cta_expression_heatmaps", [], "cta_expression_heatmaps"),
    ("cta_curation_figures", [], "cta_curation"),
    ("apd1_response_plots", [], "apd1_response"),
    ("ici_landscape", [], "ici_landscape"),
    ("apd1_ici_factor_contributions", [], "apd1_ici_contributions"),
    ("inhibitor_candidates_vs_ici", [], "inhibitor_candidates"),
    ("suppressor_genes_vs_apd1", [], "suppressor_genes"),
    ("antigen_or_suppression_score", [], "antigen_or_suppression"),
    ("placental_immune_privilege", [], "placental_immune_privilege"),
]


def _run(cmd, env=None):
    return subprocess.run(cmd, cwd=HERE, env=env).returncode == 0


def _promote_docs(run: Path) -> None:
    """Copy curation-figure families from the run into docs/ so the committed
    figures embedded in docs/*.md stay current."""
    for subdir, pattern in DOCS_PROMOTE.items():
        for src in sorted((run / subdir).glob(pattern)):
            shutil.copy2(src, DOCS / src.name)
            print(f"  promoted {src.name} -> docs/", flush=True)


def _build_combined_pdf(run: Path) -> Path | None:
    """Merge vector originals, with a paginated TOC, captions and bookmarks.

    Prefer vector PDF siblings. Legacy PNG-only figures are embedded losslessly
    at their native resolution, without the old 1000-pixel downsampling.
    """
    from io import BytesIO
    from itertools import groupby
    from math import ceil

    from pypdf import PdfReader, PdfWriter, PageObject, Transformation
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    from PIL import Image

    def family_of(p):
        return str(p.parent.relative_to(run))

    pngs = sorted(run.rglob("*.png"), key=lambda p: (family_of(p), p.name))
    if not pngs:
        return None
    groups = [(fam, list(it)) for fam, it in groupby(pngs, key=family_of)]
    toc_pages = ceil(len(groups) / 32)
    family_start, next_page = {}, toc_pages
    for fam, figures in groups:
        family_start[fam] = next_page
        next_page += len(figures) + 1

    def text_page(width, height, lines):
        stream = BytesIO()
        canvas = Canvas(stream, pagesize=(width, height))
        for x, y, size, text in lines:
            canvas.setFont("Helvetica", size)
            canvas.drawString(x, y, text)
        canvas.save()
        stream.seek(0)
        return PdfReader(stream).pages[0]

    writer = PdfWriter()
    for offset in range(0, len(groups), 32):
        lines = [(36, 755, 18, f"Pirlygenes figures - {run.name}"),
                 (36, 727, 10, f"{len(pngs)} figures; vector originals where available; PNGs at native resolution")]
        for i, (fam, figures) in enumerate(groups[offset:offset + 32]):
            lines.append((36, 696 - i * 20, 9,
                          f"p. {family_start[fam] + 1}   {fam} ({len(figures)})"))
        writer.add_page(text_page(720, 792, lines))
    for fam, figures in groups:
        writer.add_outline_item(fam, len(writer.pages))
        writer.add_page(text_page(720, 450, [
            (36, 250, 18, fam), (36, 215, 12, f"{len(figures)} figures")]))
        for png in figures:
            vector = png.with_suffix(".pdf")
            if vector.exists():
                source = PdfReader(vector).pages[0]
            else:
                stream = BytesIO()
                with Image.open(png) as image:
                    dpi = image.info.get("dpi", (300, 300))
                    width = image.width * 72 / dpi[0]
                    height = image.height * 72 / dpi[1]
                    canvas = Canvas(stream, pagesize=(width, height))
                    canvas.drawImage(ImageReader(image), 0, 0, width, height, mask="auto")
                    canvas.save()
                stream.seek(0)
                source = PdfReader(stream).pages[0]
            if source.rotation:
                source.transfer_rotation_to_content()
            width, height = float(source.mediabox.width), float(source.mediabox.height)
            caption = str(png.relative_to(run))
            page = writer.add_page(PageObject.create_blank_page(
                width=width, height=height + 24))
            page.merge_transformed_page(source, Transformation().translate(
                -float(source.mediabox.left), -float(source.mediabox.bottom)))
            size = min(9, (width - 20) / max(len(caption) * .55, 1))
            page.merge_page(text_page(width, height + 24,
                                     [(10, height + 8, size, caption)]))
    out = run / "pirlygenes-all-figures.pdf"
    with out.open("wb") as stream:
        writer.write(stream)
    writer.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate all analyses plots.")
    ap.add_argument(
        "--promote-docs", action="store_true",
        help="after the run, copy curation figures into docs/ (overwrites the "
             "committed doc figures embedded in docs/cta-curation.md)")
    ap.add_argument(
        "--no-pdf", action="store_true",
        help="skip building the combined all-figures.pdf (faster iterative runs)")
    opts = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = OUTPUTS / f"run_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []

    print(f"regenerating all plots -> {run}\n")
    env = {**os.environ, "APD1_RUN_DIR": str(run / "apd1_causal_model")}
    for s in APD1_BATCH:
        print(f"  apd1_causal_model: {s} ...", flush=True)
        (ok if _run([sys.executable, f"{s}.py"], env=env) else failed).append(s)

    for s, extra, target in LAYOUT:
        print(f"  {target}: {s} ...", flush=True)
        if target == "groups":          # writes its own group subdirs under run/
            args = ["--out-dir", str(run), "--no-timestamp"]
        else:                           # flat-writing -> its own named subdir
            args = ["--out-dir", str(run), "--run-name", target]
        (ok if _run([sys.executable, f"{s}.py", *args, *extra]) else failed).append(s)

    # cta_addressable_burden consumes cta_patient_counts' tables (written above by
    # the "groups" entry into the run root) and has its own --run-dir/--fig-dir
    # CLI, so it can't ride the LAYOUT loop. Read tables from the run root, write
    # its plots into their own family subdir.
    print("  cta_addressable: cta_addressable_burden ...", flush=True)
    addr_ok = _run([sys.executable, "cta_addressable_burden.py",
                    "--run-dir", str(run), "--fig-dir", str(run / "cta_addressable")])
    (ok if addr_ok else failed).append("cta_addressable_burden")

    # Share one collapsed reference frame between both statistics. The compute
    # and rendering live behind `pirlygenes plot covering-set` in the package.
    print("  cta_covering_set: pirlygenes.coverage.render_covering_set ...",
          flush=True)
    try:
        from pirlygenes import coverage
        from pirlygenes.expression.accessors import cancer_reference_expression
        df = cancer_reference_expression(collapse_cdna_identical=True)
        for stat in ("q3", "median"):
            res = coverage.render_covering_set(
                stat=stat, out_dir=run / "cta_covering_set", df=df)
            print(f"    {stat}: {res['n_coverable']}/{res['n_cancer_types']} "
                  f"cancer types coverable", flush=True)
            if res["unmapped_codes"]:
                print(f"    {stat}: no burden category for "
                      f"{len(res['unmapped_codes'])} code(s): "
                      f"{', '.join(sorted(res['unmapped_codes']))}", flush=True)
        ok.append("cta_covering_set")
    except Exception as exc:  # noqa: BLE001 - one figure must not abort the batch
        print(f"    FAILED: {exc}", flush=True)
        failed.append("cta_covering_set")

    if opts.promote_docs:
        _promote_docs(run)

    pngs = sorted(run.rglob("*.png"))
    pdf = None if opts.no_pdf else _build_combined_pdf(run)
    print(f"\nrun -> {run}")
    print(f"  {len(pngs)} PNGs across {len({p.parent for p in pngs})} subfolders")
    if pdf is not None:
        print(f"  all-figures.pdf: {pdf} ({len(pngs)} pages)")
    print(f"  ok: {ok}")
    if failed:
        print(f"  FAILED: {failed}")
    for d in sorted({p.parent.relative_to(run) for p in pngs}):
        print(f"    {d}/")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
