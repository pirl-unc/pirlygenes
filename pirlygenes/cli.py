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
            "cohort-level tools to inspect, build, and plot it.\n\n"
            "The data lifecycle has three stages:\n"
            "  downloads   raw per-source quantifications (the build INPUTS)\n"
            "  build       turn those into per-gene-per-cohort summaries\n"
            "  data        inspect the packaged reference data (the OUTPUTS)\n"
        ),
        epilog=(
            "Examples:\n"
            "  pirlygenes data list                 # what cohorts/genes/samples are packaged\n"
            "  pirlygenes data sources NET_PANCREAS       # which sources feed a cancer code\n"
            "  pirlygenes data status               # is the data bundle downloaded?\n"
            "  pirlygenes build list                # all buildable source ids\n"
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
        help="Manage the local cache of raw per-source quantifications (build inputs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "The local cache of RAW per-source data that builders read from —\n"
            "the inputs to `pirlygenes build`. (For the packaged OUTPUT data,\n"
            "use `pirlygenes data` instead.)"
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
        help="Inspect + manage the packaged reference data (the build outputs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "The packaged cancer reference-expression data: per-gene-per-cohort\n"
            "summaries plus the small bundled panels. Heavy summaries download\n"
            "on first use from the version-pinned GitHub release."
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
        help="Rebuild per-gene-per-cohort summaries from a source's raw data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Regenerate the packaged reference summaries for one source (or one\n"
            "cancer code) from its raw quantifications, writing the updated\n"
            "per-source shard. Downloads inputs first if needed."
        ),
        epilog=(
            "Examples:\n"
            "  pirlygenes build list                # show every buildable source id\n"
            "  pirlygenes build gse98894-midnet     # rebuild one source\n"
            "  pirlygenes build BL                  # rebuild whatever feeds a cancer code\n"
        ),
    )
    build_parser.add_argument(
        "source_id",
        metavar="<source-id|cancer-code|list|all>",
        help="A source id (e.g. 'cgci-blgsp', 'treehouse-polya-25-01') or a "
             "cancer code (e.g. 'BL', 'SARC_EWS'). Use 'list' to print all source "
             "ids, or 'all' to run every builder (slow!).",
    )
    build_parser.add_argument(
        "build_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to the underlying builder.",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Cohort-level plots over the reference data.",
        description=(
            "Cohort-level plots over the packaged reference data.\n\n"
            "actions:\n"
            "  patient-coverage   per-cohort patient coverage of a gene set\n"
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


def _builder_accepted_flags(builder_path) -> set[str]:
    """Return the set of long ``--flags`` a builder script's argparse accepts.

    Discovered by running the script with ``--help`` and scraping the usage
    text. Cached per path. Lets the dispatcher pass only flags a given builder
    actually declares (they share a conventional vocabulary but not uniformly).
    """
    import re
    import subprocess

    cached = _builder_accepted_flags._cache.get(str(builder_path))
    if cached is not None:
        return cached
    flags: set[str] = set()
    try:
        proc = subprocess.run(
            [sys.executable, str(builder_path), "--help"],
            capture_output=True, text=True, timeout=120,
        )
        flags = set(re.findall(r"(--[a-zA-Z][a-zA-Z0-9-]+)", proc.stdout))
    except Exception:
        # On any failure, fall back to the conventional trio so behavior
        # matches the historical uniform-args dispatch.
        flags = {"--summary-output", "--samples-output", "--cache-dir"}
    _builder_accepted_flags._cache[str(builder_path)] = flags
    return flags


_builder_accepted_flags._cache = {}


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
    elif builder_name == "build_recount3_source.py":
        # recount3 builder takes the source id positionally and self-fetches
        # the SRP gene-sums from S3 (no local download cache needed).
        cmd += [src.id, "--summary-output", summary_out]
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
            # Fetch from the GitHub source-data mirror (copy of the Broad TSV),
            # falling back to the Broad upstream — so the build is reproducible.
            from .builders import source_data_mirror
            try:
                fetched = source_data_mirror.fetch(
                    "cllmap_rnaseq_tpms_full.tsv.gz", upstream_url=src.url or "",
                )
                input_file = fetched
            except Exception as exc:
                sys.stderr.write(
                    f"CLLMAP input not available from mirror or upstream "
                    f"({src.url!r}): {exc}\n"
                )
                return 2
        cmd += [
            "--input", str(input_file),
            "--summary-output", summary_out,
            "--samples-output", samples_out,
        ]
    else:
        # Generic per-sample builders. They share a small, heterogeneous set of
        # conventional flags (--summary-output / --samples-output / --cache-dir)
        # but not every script declares all three. Introspect the script's
        # accepted flags via --help and pass only the supported subset, so a
        # builder that omits e.g. --samples-output isn't handed an arg it would
        # reject. Source-specific selection flags come from the YAML
        # `builder_args` (e.g. ["--only", "lps"]).
        accepted = _builder_accepted_flags(builder_path)
        candidates = {
            "--summary-output": summary_out,
            "--samples-output": samples_out,
            "--cache-dir": cache_dir,
        }
        for flag, value in candidates.items():
            if flag in accepted:
                cmd += [flag, value]

    # Source-specific fixed args declared in the registry (e.g. --only lps).
    cmd += list(src.builder_args)

    # Pass through any extra user-supplied args
    extra = list(args.build_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd += extra

    sys.stdout.write(f"running: {' '.join(cmd)}\n")
    sys.stdout.flush()
    result = subprocess.run(cmd)
    return int(result.returncode)


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
            f"Run `pirlygenes downloads fetch {args.source}` / "
            f"`pirlygenes build {args.source}` first.\n"
        )
        return 2
    sys.stdout.write(
        f"{result['label']}: {result['n_cohorts']} cohorts "
        f"(> {args.threshold} TPM)\n"
    )
    for kind, path in result["paths"].items():
        sys.stdout.write(f"  {kind}: {path}\n")
    return 0


_PLOT_DISPATCH = {
    "patient-coverage": cmd_plot_patient_coverage,
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
            sys.stderr.write("usage: pirlygenes plot {patient-coverage}\n")
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
