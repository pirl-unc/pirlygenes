"""Pirlygenes CLI — cohort-level reference-data inspection.

``downloads`` inspects and fetches oncoref-owned source matrices, ``build``
identifies oncoref as the regeneration owner, and ``data`` inspects the
delegated summaries plus pirlygenes' curated compatibility artifacts.

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

Every expression-ingestion operation delegates to oncoref. Pirlygenes contains
no source builder or shard writer.

Pattern matches ``trufflepig/cli.py``: stdlib argparse + per-command
``cmd_*`` handlers + a dispatch dict in :func:`main`.

See ``docs/expression-data-refresh-plan.md`` for the multi-session
roadmap that drives this CLI's surface.
"""

from __future__ import annotations

import argparse
import sys

from . import data_bundle, data_inventory, downloads
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "pirlygenes — curated cancer reference-expression data and the\n"
            "cohort-level tools to inspect and plot it.\n\n"
            "Oncoref owns expression ingestion and source matrices:\n"
            "  downloads   inspect/fetch oncoref per-sample matrices\n"
            "  build       identify the upstream build owner\n"
            "  data        inspect packaged/delegated reference data\n"
        ),
        epilog=(
            "Examples:\n"
            "  pirlygenes data list                 # what cohorts/genes/samples are packaged\n"
            "  pirlygenes data sources NET_PANCREAS       # which sources feed a cancer code\n"
            "  pirlygenes data status               # is the data bundle downloaded?\n"
            "  pirlygenes build list                # source ids and build owners\n"
            "\n"
            "The Python data API is unchanged: `from pirlygenes import\n"
            "gene_sets_cancer, gene_ids, gene_names, gene_families`.\n"
        ),
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"pirlygenes {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", metavar="<command>", title="commands",
    )

    downloads_parser = subparsers.add_parser(
        "downloads",
        help="Inspect or fetch oncoref-owned per-sample matrices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "The oncoref source-matrix cache consumed by patient-level\n"
            "pirlygenes analyses. Use `pirlygenes data` for summary artifacts."
        ),
        epilog="Example:\n  pirlygenes downloads list      # registered sources by on-disk size\n",
    )
    downloads_sub = downloads_parser.add_subparsers(
        dest="downloads_action", metavar="<action>", title="actions",
    )
    downloads_sub.add_parser(
        "list",
        help="List registered sources grouped by category, sorted by on-disk size.",
    )
    downloads_sub.add_parser(
        "cache-dir",
        help=(
            "Print the oncoref matrix cache root "
            "(honors CANCERDATA_SOURCE_MATRICES)."
        ),
    )
    fetch_parser = downloads_sub.add_parser(
        "fetch",
        help="Fetch every published matrix for a source ID or cancer code.",
    )
    fetch_parser.add_argument(
        "source_id", help="Source id from oncoref's expression registry"
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
        help="Inspect + manage the packaged reference data (the build outputs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Cancer reference-expression summaries delegated to oncoref, plus\n"
            "pirlygenes' small curated panels and purpose-specific derived views."
        ),
        epilog=(
            "Examples:\n"
            "  pirlygenes data list                 # cohorts, genes, samples at a glance\n"
            "  pirlygenes data sources NET_PANCREAS       # the source(s) feeding one cancer code\n"
            "  pirlygenes data status               # is the bundle downloaded locally?\n"
        ),
    )
    data_sub = data_parser.add_subparsers(
        dest="data_action", metavar="<action>", title="actions",
    )
    list_parser = data_sub.add_parser(
        "list",
        help="Overview of every cohort — samples, genes measured, and "
             "quantification method (downloads the bundle if not local yet).",
    )
    list_parser.add_argument(
        "--sort", choices=["name", "samples"], default="name",
        help="Order source cohorts by id (default) or by sample count.",
    )
    list_parser.add_argument(
        "--code", metavar="CANCER_CODE",
        help="Show only the source cohort(s) feeding this cancer code.",
    )
    list_parser.add_argument(
        "--flat", action="store_true",
        help="One flat table of every cohort sorted by sample count, with "
             "assay / quantification / reference as columns.",
    )
    data_sub.add_parser(
        "status",
        help="Report which downloadable bundle paths are present in "
             "the local cache for this package version.",
    )
    sources_parser = data_sub.add_parser(
        "sources",
        help="Show the expression source(s) feeding each cancer code "
             "(samples, gene count, quantification method).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "For each cancer code, list its source cohort(s) with sample count,\n"
            "genes measured, and native quantification. When a code has more\n"
            "than one source they are kept SEPARATE (different assays/scales)\n"
            "and never merged — you pick or compare them explicitly."
        ),
        epilog=(
            "Examples:\n"
            "  pirlygenes data sources NET_PANCREAS   # one code (here: 2 sources)\n"
            "  pirlygenes data sources --multi  # only codes with >1 source\n"
        ),
    )
    sources_parser.add_argument(
        "code", nargs="?",
        help="Restrict to one cancer code (e.g. NET_PANCREAS). Omit to list "
             "every code; --multi to show only codes with >1 source.",
    )
    sources_parser.add_argument(
        "--multi", action="store_true",
        help="Only show cancer codes that have more than one source.",
    )
    data_sub.add_parser(
        "cache-dir",
        help="Print the on-disk cache dir for the downloaded data "
             "bundle (override via PIRLYGENES_BUNDLED_DATA).",
    )
    data_sub.add_parser(
        "fetch",
        help="Explicitly download the data bundle from the GitHub "
             "Release matching the installed version.",
    )
    prune_parser = data_sub.add_parser(
        "prune",
        help="Delete stale v<old-version>/ bundled-data cache dirs "
             "left behind by previous installs; keeps the current "
             "version's dir by default.",
    )
    prune_parser.add_argument(
        "--yes", action="store_true",
        help="Actually delete (default is dry-run that just lists).",
    )
    prune_parser.add_argument(
        "--include-current", action="store_true",
        help="Also delete the current version's cache dir (forces a "
             "re-fetch on next data access).",
    )

    build_parser = subparsers.add_parser(
        "build",
        help="Identify the oncoref build owner (pirlygenes is read-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Expression regeneration is owned by oncoref.expression_builders.\n"
            "This compatibility command lists sources and redirects rebuilds;\n"
            "it never writes a second pirlygenes artifact."
        ),
        epilog=(
            "Examples:\n"
            "  pirlygenes build list                # show source ids and owners\n"
            "  pirlygenes build gse98894-midnet     # show the upstream owner\n"
        ),
    )
    build_parser.add_argument(
        "source_id",
        metavar="<source-id|cancer-code|list|all>",
        help="A source id (e.g. 'cgci-blgsp', 'treehouse-polya-25-01') or a "
             "cancer code (e.g. 'BL', 'SARC_EWS'). Use 'list' to print all source "
             "ids. 'all' reports the single upstream owner.",
    )
    build_parser.add_argument(
        "build_args",
        nargs=argparse.REMAINDER,
        help="Deprecated compatibility arguments; no local builder is run.",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Cohort-level plots over the reference data.",
        description=(
            "Cohort-level plots over the packaged reference data.\n\n"
            "actions:\n"
            "  patient-coverage   per-cohort patient coverage of a gene set\n"
            "  cta-curation       CTA panel curation figures (source/filter/HPA)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plot_sub = plot_parser.add_subparsers(
        dest="plot_action", metavar="<action>", title="actions",
    )
    pc = plot_sub.add_parser(
        "patient-coverage",
        help="Per-cohort patient coverage of a gene set (counts CSV + plots).",
        description=(
            "For each cancer cohort with cached per-sample data, count how many\n"
            "patients express each gene of a panel above TPM thresholds, and\n"
            "compute greedy co-occurrence-aware coverage. Writes a counts CSV +\n"
            "a stacked coverage bar + a coverage-curve small-multiples."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pirlygenes plot patient-coverage --gene-set cta\n"
            "  pirlygenes plot patient-coverage --gene-set lineage:PRAD --cohort PRAD\n"
            "  pirlygenes plot patient-coverage --gene-set ./my_symbols.csv\n"
        ),
    )
    pc.add_argument(
        "--gene-set", required=True,
        help=("panel to score: cta | surfaceome | mito | housekeeping | "
              "therapy:<type> | lineage:<code> | a path to a CSV of symbols/ENSG ids"),
    )
    pc.add_argument(
        "--source", default="treehouse-polya-25-01",
        help="expression source id with cached per-sample data (default: %(default)s)",
    )
    pc.add_argument(
        "--threshold", type=int, default=25,
        help="TPM cutoff for the coverage plots (default: %(default)s)",
    )
    pc.add_argument(
        "--cohort", action="append", default=None, metavar="CODE",
        help="restrict to specific cancer-type code(s); repeatable (default: all)",
    )
    pc.add_argument(
        "--out", default="coverage_out",
        help="output directory for the CSV + PNGs (default: %(default)s)",
    )

    cur = plot_sub.add_parser(
        "cta-curation",
        help="CTA curation figures (source overlap, filter funnel/outcome, HPA).",
        description=(
            "Rebuild the five CTA-curation documentation figures from the\n"
            "packaged CTA evidence table (tsarina.CTA_detailed_evidence):\n"
            "source venn, filter funnel/outcome, deflated reproductive-fraction\n"
            "distribution, and protein-reliability-vs-RNA tiers. These are the\n"
            "figures embedded in docs/cta-curation.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  pirlygenes plot cta-curation --out cta_curation_out\n",
    )
    cur.add_argument(
        "--out", default="cta_curation_out",
        help="output directory for the PNGs (default: %(default)s)",
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
    from oncoref import source_matrices

    sys.stdout.write(str(source_matrices.cache_dir()) + "\n")
    return 0


def cmd_downloads_fetch(args: argparse.Namespace) -> int:
    from oncoref import source_matrices
    from . import cohorts

    requested = str(args.source_id)
    try:
        # The owner resolver accepts canonical codes, case variants, and public
        # cancer aliases (for example PANNET -> NET_PANCREAS). Resolve those
        # before interpreting the same token as a source registry ID.
        info = source_matrices.cohort_info(requested)
    except source_matrices.SourceMatrixError:
        codes = sorted(cohorts.cohorts_for_source(requested))
    else:
        codes = [str(info["cancer_code"])]
    if not codes:
        sys.stderr.write(
            f"no oncoref source matrices match {requested!r}. "
            "Use `pirlygenes build list` for source IDs or "
            "`oncoref.source_matrices.available_cohorts()` for cancer codes.\n"
        )
        return 2
    for code in codes:
        try:
            source_matrices.fetch(code)
        except source_matrices.SourceMatrixError as err:
            sys.stderr.write(
                f"failed to fetch oncoref source matrix {code!r}: {err}\n"
            )
            return 2
    noun = "matrix" if len(codes) == 1 else "matrices"
    sys.stdout.write(
        f"fetched {len(codes)} oncoref source {noun}: {', '.join(codes)}\n"
    )
    return 0


def cmd_downloads_prune(_args: argparse.Namespace) -> int:
    sys.stderr.write(
        _NOT_IMPLEMENTED_MESSAGE.format(
            subcommand="downloads prune", milestone=5
        )
        + "\n"
    )
    return 2


def cmd_data_list(args: argparse.Namespace) -> int:
    snapshot = data_inventory.summarize_inventory()
    sys.stdout.write(
        data_inventory.render_inventory(
            snapshot,
            sort_by=getattr(args, "sort", "name"),
            code_filter=getattr(args, "code", None),
            flat=getattr(args, "flat", False),
        )
        + "\n"
    )
    return 0


def cmd_data_sources(args: argparse.Namespace) -> int:
    """List the expression source(s) per cancer code, with their native unit,
    sample count, and gene count, so multi-source cohorts are visible.

    Semantics: when a cancer code has more than one source, the shards are
    kept SEPARATE (different assays / quantification scales) and are NOT
    averaged together. Consumers select or compare them explicitly — e.g.
    the CTA heatmaps pick the most-gene-rich source per code. A microarray
    TPM-proxy is not comparable in absolute magnitude to RNA-seq TPM; see
    docs/recount3-integration.md and the `normalization` column.
    """
    import pirlygenes.expression.accessors as accessors

    df = accessors.cancer_reference_expression()
    native_unit = {
        s.source_cohort: s.unit
        for s in downloads.load_registry()
        if s.source_cohort and s.unit
    }
    grouped = (
        df.groupby(["cancer_code", "source_cohort"])
        .agg(
            n_samples=("n_samples", "first"),
            n_genes=("Ensembl_Gene_ID", "nunique"),
            pipeline=("processing_pipeline", "first"),
        )
        .reset_index()
    )

    all_codes = sorted(grouped["cancer_code"].astype(str).unique())
    if args.code:
        want = args.code.upper()
        if want not in set(all_codes):
            sys.stderr.write(f"no cancer code {want!r} in the reference data.\n")
            return 2
        codes = [want]
    else:
        codes = all_codes

    shown = 0
    for code in codes:
        sub = grouped[grouped["cancer_code"] == code].sort_values(
            "n_genes", ascending=False
        )
        if args.multi and len(sub) < 2:
            continue
        shown += 1
        tag = "  (multi-source — kept separate, not merged)" if len(sub) > 1 else ""
        sys.stdout.write(f"\n{code}{tag}\n")
        for _, r in sub.iterrows():
            unit = native_unit.get(r["source_cohort"]) or (
                data_inventory.native_unit_from_pipeline(str(r["pipeline"]))
            )
            sys.stdout.write(
                f"    {r['source_cohort']:34} n={int(r['n_samples']):<4} "
                f"genes={int(r['n_genes']):<6} {unit}\n"
            )
    if shown == 0:
        sys.stdout.write("no matching cancer codes.\n")
    return 0


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:6.1f} {u}"
        f /= 1024
    return f"{f:.1f} TB"


def cmd_data_status(_args: argparse.Namespace) -> int:
    snap = data_bundle.status()
    sys.stdout.write(f"pirlygenes data bundle for v{snap['data_version']}\n")
    sys.stdout.write(f"  cache dir   : {snap['cache_dir']}\n")
    sys.stdout.write(f"  release URL : {snap['release_url']}\n")
    sys.stdout.write(f"  all local?  : {snap['all_local']}\n")
    sys.stdout.write("  items:\n")
    for name, info in snap["items"].items():
        mark = "✓" if info["present"] else "✗"
        size = _format_bytes(info["size_bytes"]) if info["present"] else "       "
        sys.stdout.write(f"    {mark}  {size}  {name}\n")
    if not snap["all_local"]:
        sys.stdout.write(
            "\nRun `pirlygenes data fetch` to download missing items.\n"
        )
    return 0


def cmd_data_cache_dir(_args: argparse.Namespace) -> int:
    sys.stdout.write(str(data_bundle.cache_dir()) + "\n")
    return 0


def cmd_data_fetch(_args: argparse.Namespace) -> int:
    try:
        data_bundle.fetch(verbose=True)
        return 0
    except Exception as exc:
        sys.stderr.write(f"pirlygenes data fetch failed: {exc}\n")
        return 1


def cmd_data_prune(args: argparse.Namespace) -> int:
    versions = data_bundle.list_cache_versions()
    if not versions:
        sys.stdout.write(
            f"pirlygenes: no cache dirs under "
            f"{data_bundle.cache_root()}; nothing to prune.\n"
        )
        return 0
    keep_current = not getattr(args, "include_current", False)
    dry_run = not getattr(args, "yes", False)
    candidates = data_bundle.prune_cache(
        keep_current=keep_current, dry_run=True,
    )
    sys.stdout.write(
        f"pirlygenes: bundled-data cache at {data_bundle.cache_root()}\n"
    )
    for entry in versions:
        marker = "(current)" if entry["is_current"] else ""
        will_delete = entry in candidates
        action = "DELETE" if will_delete else "keep  "
        size_mb = entry["size_bytes"] / 1e6
        sys.stdout.write(
            f"  {action}  {entry['version']:<10s}  "
            f"{size_mb:7.1f} MB  {marker}\n"
        )
    if not candidates:
        sys.stdout.write("nothing to prune.\n")
        return 0
    total_mb = sum(c["size_bytes"] for c in candidates) / 1e6
    if dry_run:
        sys.stdout.write(
            f"\ndry run — would free {total_mb:.1f} MB across "
            f"{len(candidates)} dir(s). Re-run with --yes to delete.\n"
        )
        return 0
    data_bundle.prune_cache(keep_current=keep_current, dry_run=False)
    sys.stdout.write(
        f"\ndeleted {len(candidates)} cache dir(s), freed {total_mb:.1f} MB.\n"
    )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Preserve source discovery while redirecting every rebuild to oncoref."""
    from . import cohorts

    sources = downloads.load_registry()
    requested = args.source_id

    if requested == "list":
        for s in sorted(sources, key=lambda x: x.id):
            codes = ",".join(s.cancer_codes) or "-"
            sys.stdout.write(
                f"  {s.id:32}  {codes:40}  (oncoref-owned)\n"
            )
        return 0

    if requested == "all":
        sys.stderr.write(
            "pirlygenes is a read-only oncoref consumer; all expression "
            "rebuilds belong to oncoref.expression_builders.\n"
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

    published = sorted(cohorts.cohorts_for_source(src.id))
    if published:
        sys.stderr.write(
            f"source {src.id!r} is built and published by oncoref; use "
            "oncoref.expression_builders for regeneration or "
            "`pirlygenes downloads fetch "
            f"{src.id}` to fetch its selected published matrices "
            f"({', '.join(published)}).\n"
        )
    else:
        sys.stderr.write(
            f"source {src.id!r} is owned by oncoref, but no published "
            "source matrix currently matches it; use "
            "oncoref.expression_builders for regeneration and track the "
            "owner's source-matrix registry for publication.\n"
        )
    return 2


def cmd_plot_patient_coverage(args: argparse.Namespace) -> int:
    from . import coverage

    try:
        result = coverage.render(
            args.gene_set, source_id=args.source, codes=args.cohort,
            threshold=args.threshold, out_dir=args.out,
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    if result["n_cohorts"] == 0:
        sys.stderr.write(
            f"no cohorts with cached per-sample data for source "
            f"'{args.source}' (and gene set '{result['label']}'). "
            f"Run `pirlygenes downloads fetch {args.source}` first.\n"
        )
        return 2
    sys.stdout.write(
        f"{result['label']}: {result['n_cohorts']} cohorts "
        f"(> {args.threshold} TPM)\n"
    )
    for kind, path in result["paths"].items():
        sys.stdout.write(f"  {kind}: {path}\n")
    return 0


def cmd_plot_cta_curation(args: argparse.Namespace) -> int:
    from . import cta_curation_plots

    try:
        result = cta_curation_plots.render(out_dir=args.out)
    except Exception as exc:  # noqa: BLE001 — tsarina/matplotlib failure -> clean exit
        sys.stderr.write(f"error: could not render CTA curation figures: {exc}\n")
        return 2
    sys.stdout.write(
        f"CTA curation figures ({result['n_genes']} evidence rows):\n")
    for kind, path in result["paths"].items():
        sys.stdout.write(f"  {kind}: {path}\n")
    return 0


_PLOT_DISPATCH = {
    "patient-coverage": cmd_plot_patient_coverage,
    "cta-curation": cmd_plot_cta_curation,
}


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
    "sources": cmd_data_sources,
    "status": cmd_data_status,
    "cache-dir": cmd_data_cache_dir,
    "fetch": cmd_data_fetch,
    "prune": cmd_data_prune,
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
            sys.stderr.write(
                "usage: pirlygenes data {list,status,cache-dir,fetch,prune}\n"
            )
            return 2
        return handler(args)

    if subcommand == "plot":
        handler = _PLOT_DISPATCH.get(args.plot_action)
        if handler is None:
            sys.stderr.write(
                "usage: pirlygenes plot {patient-coverage,cta-curation}\n")
            return 2
        return handler(args)

    dispatch = {
        "build": cmd_build,
    }
    handler = dispatch.get(subcommand)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
