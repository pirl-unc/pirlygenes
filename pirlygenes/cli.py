"""Pirlygenes CLI — cohort-level reference-data operations.

Hosts subcommands that act on the curated cohort-level reference
data: ``downloads`` (manage the local cache of per-source raw
quantifications), ``build`` (regenerate per-gene-per-cohort summary
statistics from per-sample data), and ``plot`` (cohort-level
visualisations of gene sets and reference matrices).

Per-sample analysis (``analyze`` and siblings) lives in
``pirl-trufflepig``. This CLI keeps the migration-pointer message for
those subcommand names so users with old wrapper scripts get a clear
redirect rather than a confusing argparse error.

This CLI is a thin wrapper around the same Python API that external
consumers (trufflepig, ad-hoc notebooks) use directly:

    from pirlygenes import downloads
    downloads.cache_root()
    downloads.load_registry()
    downloads.collect_cache_usage()

Every subcommand here delegates to a function in the
:mod:`pirlygenes.downloads` (and, when they land, ``pirlygenes.builders``
/ ``pirlygenes.plot``) modules — no behavior is CLI-only.

Pattern matches ``trufflepig/cli.py``: stdlib argparse + per-command
``cmd_*`` handlers + a dispatch dict in :func:`main`.

See ``docs/expression-data-refresh-plan.md`` for the multi-session
roadmap that drives this CLI's surface.
"""

from __future__ import annotations

import argparse
import sys

from . import data_inventory, downloads
from .version import __version__


_ANALYSIS_MOVED_MESSAGE = """\
pirlygenes no longer ships analysis subcommands as of v5.0.0.

`analyze`, `compare-analyze`, `plot-expression`, and
`plot-cancer-cohorts` (per-sample analysis) moved to `pirl-trufflepig`:

    pip install pirl-trufflepig
    trufflepig run --sample expr.tsv --workspace out --cancer-type BLCA
    trufflepig compare --workspace out/long --inputs out/A,out/B
    trufflepig data
    trufflepig cancers

See https://github.com/pirl-unc/trufflepig for the full migration.

Cohort-level subcommands (which DO live in pirlygenes) are:

    pirlygenes downloads list
    pirlygenes downloads cache-dir
    pirlygenes data list
    pirlygenes build <source-id>
    pirlygenes plot <...>

The pirlygenes Python data API is unchanged — `from pirlygenes import
gene_sets_cancer, load_dataset, gene_ids, gene_names, gene_families`
still works.
"""

_NOT_IMPLEMENTED_MESSAGE = (
    "{subcommand!r} is scaffolded but not implemented in this release. "
    "See docs/expression-data-refresh-plan.md milestone {milestone}."
)


_ANALYSIS_SUBCOMMANDS = frozenset({
    "analyze",
    "compare-analyze",
    "plot-expression",
    "plot-cancer-cohorts",
})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pirlygenes",
        description="Pirlygenes cohort-level reference-data CLI.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"pirlygenes {__version__}",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    downloads_parser = subparsers.add_parser(
        "downloads",
        help="Manage the local cache of per-source expression downloads.",
    )
    downloads_sub = downloads_parser.add_subparsers(dest="downloads_action")
    downloads_sub.add_parser(
        "list",
        help="List registered sources grouped by category, sorted by on-disk size.",
    )
    downloads_sub.add_parser(
        "cache-dir",
        help="Print the active cache root (honors PIRLYGENES_CACHE env var).",
    )
    fetch_parser = downloads_sub.add_parser(
        "fetch",
        help="Download raw data for a source (NotImplemented; see plan milestone 5).",
    )
    fetch_parser.add_argument(
        "source_id", help="Source id from expression_sources.yaml"
    )
    prune_parser = downloads_sub.add_parser(
        "prune",
        help="Cap cache disk usage (NotImplemented; see plan milestone 5).",
    )
    prune_parser.add_argument(
        "--max-gb",
        type=float,
        required=True,
        help="Cap total cache size at this many GB.",
    )

    data_parser = subparsers.add_parser(
        "data",
        help="Inspect bundled cohort-level reference data.",
    )
    data_sub = data_parser.add_subparsers(dest="data_action")
    data_sub.add_parser(
        "list",
        help="Summarize cancer-reference-expression bundled rows by cohort.",
    )

    build_parser = subparsers.add_parser(
        "build",
        help="Rebuild per-gene-per-cohort summaries by source-id or cancer-code.",
    )
    build_parser.add_argument(
        "source_id",
        help="Source id (e.g. 'cgci-blgsp', 'treehouse-polya-25-01') or "
             "cancer_code (e.g. 'BL', 'EWS'). Special: 'list' prints "
             "all known source ids; 'all' runs every builder (slow!).",
    )
    build_parser.add_argument(
        "build_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to the underlying builder.",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Cohort-level plots (NotImplemented; see plan milestone 7).",
    )
    plot_parser.add_argument(
        "target", help="What to plot (e.g. a gene symbol or panel name)."
    )

    for name in sorted(_ANALYSIS_SUBCOMMANDS):
        moved = subparsers.add_parser(name, help="Moved to pirl-trufflepig.")
        moved.add_argument(
            "remainder",
            nargs=argparse.REMAINDER,
            help=argparse.SUPPRESS,
        )

    return parser


def cmd_downloads_list(_args: argparse.Namespace) -> int:
    usages = downloads.collect_cache_usage()
    sys.stdout.write(downloads.render_list(usages) + "\n")
    return 0


def cmd_downloads_cache_dir(_args: argparse.Namespace) -> int:
    sys.stdout.write(str(downloads.cache_root()) + "\n")
    return 0


def cmd_downloads_fetch(_args: argparse.Namespace) -> int:
    sys.stderr.write(
        _NOT_IMPLEMENTED_MESSAGE.format(
            subcommand="downloads fetch", milestone=5
        )
        + "\n"
    )
    return 2


def cmd_downloads_prune(_args: argparse.Namespace) -> int:
    sys.stderr.write(
        _NOT_IMPLEMENTED_MESSAGE.format(
            subcommand="downloads prune", milestone=5
        )
        + "\n"
    )
    return 2


def cmd_data_list(_args: argparse.Namespace) -> int:
    snapshot = data_inventory.summarize_inventory()
    sys.stdout.write(data_inventory.render_inventory(snapshot) + "\n")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Dispatch to scripts/build_*.py per the YAML registry's `builder` field.

    Lookup order:
      1. exact source_id match (lowercase)
      2. cancer_code membership in any registered source's cancer_codes
      3. 'list' / 'all' meta-commands

    Standard --summary-output / --samples-output / --cache-dir defaults
    are inferred from the source's id. Extra args passed via REMAINDER.
    """
    import subprocess
    from pathlib import Path

    sources = downloads.load_registry()
    requested = args.source_id

    if requested == "list":
        for s in sorted(sources, key=lambda x: x.id):
            codes = ",".join(s.cancer_codes) or "-"
            builder = s.builder or "(no builder)"
            sys.stdout.write(f"  {s.id:32}  {codes:40}  {builder}\n")
        return 0

    if requested == "all":
        sys.stderr.write(
            "build all: not yet implemented (would run every source's "
            "builder; expect hours of GDC/GEO/Treehouse downloads). "
            "For now, invoke per-source.\n"
        )
        return 2

    # Exact id match, else cancer_code lookup
    src = next((s for s in sources if s.id == requested), None)
    if src is None:
        candidates = [s for s in sources if requested.upper() in {c.upper() for c in s.cancer_codes}]
        if not candidates:
            sys.stderr.write(
                f"no source matches {requested!r}. Run "
                "`pirlygenes build list` to see all source ids.\n"
            )
            return 2
        if len(candidates) > 1:
            sys.stderr.write(
                f"cancer code {requested!r} is covered by multiple sources: "
                f"{[c.id for c in candidates]}. Pick one of those source-ids.\n"
            )
            return 2
        src = candidates[0]

    if not src.builder:
        sys.stderr.write(
            f"source {src.id!r} has no `builder` field in the YAML registry.\n"
        )
        return 2

    builder_path = Path(src.builder)
    if not builder_path.exists():
        sys.stderr.write(
            f"builder script not found: {builder_path}\n"
        )
        return 2

    # Conventional defaults; each script honors at least --summary-output.
    summary_out = "pirlygenes/data/cancer-reference-expression"
    samples_out = "pirlygenes/data/cancer-reference-expression-samples.csv.gz"
    cache_dir = str(downloads.source_cache_dir(src.id, category=src.category))

    cmd = [sys.executable, str(builder_path)]

    # Heuristic arg-passing based on what each builder accepts.
    # Sweep scripts: --summary-output and --refresh-cache.
    # Build scripts: --summary-output, --samples-output, and --cache-dir.
    builder_name = builder_path.name
    if "sweep_treehouse" in builder_name:
        cmd += ["--summary-output", summary_out]
    elif builder_name in {"build_geo_matrix.py", "build_gpl570_microarray.py"}:
        # YAML-driven builders that look up their config block by source id.
        cmd += [
            "--source-id", src.id,
            "--summary-output", summary_out,
            "--samples-output", samples_out,
            "--cache-dir", cache_dir,
        ]
    elif builder_name == "build_target_subprojects.py":
        cmd += [
            "--summary-output", summary_out,
            "--samples-output", samples_out,
            "--cache-root", str(downloads.cache_root() / "expression"),
        ]
    elif builder_name == "build_cllmap_reference_expression.py":
        # CLLMAP needs --input pointing at the downloaded TPM file
        input_file = Path(cache_dir) / "cllmap_rnaseq_tpms_full.tsv.gz"
        if not input_file.exists():
            sys.stderr.write(
                f"CLLMAP input not yet downloaded: {input_file}\n"
                f"Download from {src.url!r} first.\n"
            )
            return 2
        cmd += [
            "--input", str(input_file),
            "--summary-output", summary_out,
            "--samples-output", samples_out,
        ]
    else:
        cmd += [
            "--summary-output", summary_out,
            "--samples-output", samples_out,
            "--cache-dir", cache_dir,
        ]

    # Pass through any extra user-supplied args
    extra = list(args.build_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd += extra

    sys.stdout.write(f"running: {' '.join(cmd)}\n")
    sys.stdout.flush()
    result = subprocess.run(cmd)
    return int(result.returncode)


def cmd_plot(_args: argparse.Namespace) -> int:
    sys.stderr.write(
        _NOT_IMPLEMENTED_MESSAGE.format(subcommand="plot", milestone=7) + "\n"
    )
    return 2


def cmd_analysis_moved(_args: argparse.Namespace) -> int:
    sys.stderr.write(_ANALYSIS_MOVED_MESSAGE)
    return 2


_DOWNLOADS_DISPATCH = {
    "list": cmd_downloads_list,
    "cache-dir": cmd_downloads_cache_dir,
    "fetch": cmd_downloads_fetch,
    "prune": cmd_downloads_prune,
}


_DATA_DISPATCH = {
    "list": cmd_data_list,
}


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    if raw and raw[0] in _ANALYSIS_SUBCOMMANDS:
        return cmd_analysis_moved(None)

    parser = _build_parser()
    args = parser.parse_args(argv)

    subcommand = args.subcommand
    if subcommand is None:
        parser.print_help()
        return 0

    if subcommand in _ANALYSIS_SUBCOMMANDS:
        return cmd_analysis_moved(args)

    if subcommand == "downloads":
        handler = _DOWNLOADS_DISPATCH.get(args.downloads_action)
        if handler is None:
            sys.stderr.write(
                "usage: pirlygenes downloads {list,cache-dir,fetch,prune}\n"
            )
            return 2
        return handler(args)

    if subcommand == "data":
        handler = _DATA_DISPATCH.get(args.data_action)
        if handler is None:
            sys.stderr.write("usage: pirlygenes data {list}\n")
            return 2
        return handler(args)

    dispatch = {
        "build": cmd_build,
        "plot": cmd_plot,
    }
    handler = dispatch.get(subcommand)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
