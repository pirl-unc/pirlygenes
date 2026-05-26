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
        help="Rebuild per-gene-per-cohort summaries (NotImplemented; see plan milestone 2).",
    )
    build_parser.add_argument(
        "source_id",
        help="Source id from expression_sources.yaml, or 'all'.",
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


def cmd_build(_args: argparse.Namespace) -> int:
    sys.stderr.write(
        _NOT_IMPLEMENTED_MESSAGE.format(subcommand="build", milestone=2) + "\n"
    )
    return 2


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
