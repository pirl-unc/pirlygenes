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
    ("cta_covering_set", [], "cta_covering_set"),
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


# max page dimension (px) in the combined PDF — bounds file size and memory;
# the run dir keeps the full-resolution PNGs for detailed inspection.
_PDF_MAX_PX = 1000
_PDF_PAGE_W = 1000


def _pdf_font(size):
    """A TrueType font at ``size`` — DejaVu Sans (bundled with matplotlib) for
    reliable sizing, falling back to Pillow's default."""
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # noqa: BLE001
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _build_combined_pdf(run: Path) -> Path | None:
    """Bundle every PNG in the run into one organized ``all-figures.pdf``:

      * a **table of contents** page (each section -> start page + figure count),
      * a **section divider** page before each family,
      * one captioned figure per page (run-relative path),
      * **PDF bookmarks** (one per section) when ``pypdf`` is available.

    Sections = the run's family subdirs, ordered by path. Pillow does the image
    embedding (fast), capped at ``_PDF_MAX_PX`` so per-run cost stays bounded;
    full-resolution PNGs remain in the run dir. Resilient: any failure is
    reported but never aborts the batch."""
    if not list(run.rglob("*.png")):
        return None
    try:
        from itertools import groupby

        from PIL import Image, ImageDraw

        def family_of(p):
            return str(p.parent.relative_to(run))

        def disp(fam):  # display label for the run root
            return "(root)" if fam == "." else fam

        # Sort by (family, name) — NOT full path — so every family's files are
        # contiguous; a plain path sort interleaves a dir's files with its
        # subdirs and splits one family into duplicate groupby runs.
        pngs = sorted(run.rglob("*.png"), key=lambda p: (family_of(p), p.name))
        groups = [(fam, list(it)) for fam, it in groupby(pngs, key=family_of)]

        def _caption_page(img, caption):
            cap_h = 20
            page = Image.new("RGB", (img.width, img.height + cap_h), "white")
            page.paste(img, (0, cap_h))
            ImageDraw.Draw(page).text((6, 4), caption, fill="black",
                                      font=_pdf_font(14))
            return page

        # body = [divider, figs...] per section; track each divider's body index
        body, family_start = [], {}
        for fam, figs in groups:
            family_start[fam] = len(body)
            div = Image.new("RGB", (_PDF_PAGE_W, 640), "white")
            dd = ImageDraw.Draw(div)
            dd.text((48, 280), disp(fam), fill="black", font=_pdf_font(38))
            dd.text((48, 344), f"{len(figs)} figure(s)", fill="#555",
                    font=_pdf_font(22))
            body.append(div)
            for f in figs:
                im = Image.open(f).convert("RGB")
                im.thumbnail((_PDF_MAX_PX, _PDF_MAX_PX))
                body.append(_caption_page(im, str(f.relative_to(run))))

        # ToC page goes first, so a section divider at body index s is page s+2.
        toc = Image.new("RGB", (_PDF_PAGE_W, max(640, 130 + 26 * len(groups))),
                        "white")
        tt = ImageDraw.Draw(toc)
        tt.text((48, 28), f"All figures — {run.name}", fill="black",
                font=_pdf_font(30))
        tt.text((48, 72), f"{len(pngs)} figures · {len(groups)} sections",
                fill="#555", font=_pdf_font(18))
        y = 120
        for fam, figs in groups:
            tt.text((48, y), disp(fam), fill="black", font=_pdf_font(15))
            tt.text((_PDF_PAGE_W - 150, y), f"p.{family_start[fam] + 2}  ({len(figs)})",
                    fill="#555", font=_pdf_font(15))
            y += 26

        pages = [toc] + body
        out = run / "all-figures.pdf"
        pages[0].save(out, save_all=True, append_images=pages[1:])

        # bookmarks (one per section -> its divider page); a bonus when pypdf
        # is installed, never required.
        try:
            from pypdf import PdfReader, PdfWriter
            reader, writer = PdfReader(out), PdfWriter()
            for p in reader.pages:
                writer.add_page(p)
            for fam, _figs in groups:
                writer.add_outline_item(disp(fam), family_start[fam] + 1)
            with open(out, "wb") as fh:
                writer.write(fh)
        except Exception:  # noqa: BLE001 — ToC + dividers already navigate
            pass
        return out
    except Exception as exc:  # noqa: BLE001 — never fail the batch over the PDF
        print(f"  WARNING: could not build all-figures.pdf: {exc}", flush=True)
        return None


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
