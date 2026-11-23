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
import contextlib
import json
import os
import sys
from pathlib import Path

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
    dl_cache_dir = downloads_sub.add_parser(
        "cache-dir",
        aliases=["path"],
        help=(
            "Print the oncoref matrix cache root "
            "(honors CANCERDATA_SOURCE_MATRICES)."
        ),
    )
    dl_cache_dir.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
    )
    fetch_parser = downloads_sub.add_parser(
        "fetch",
        help="Fetch every published matrix for a source ID or cancer code.",
    )
    fetch_parser.add_argument(
        "source_id", help="Source id from oncoref's expression registry"
    )
    fetch_parser.add_argument(
        "--offline", action="store_true",
        help="Disallow network downloads (equivalent to DATACACHE_OFFLINE=1).",
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
    dirs_parser = data_sub.add_parser(
        "dirs",
        help="Show pirlygenes cache locations, sizes, and offline status (read-only).",
    )
    dirs_parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
    )
    status_parser = data_sub.add_parser(
        "status",
        help="Report which downloadable bundle paths are present in "
             "the local cache for this package version.",
    )
    status_parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
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
    data_cache_dir = data_sub.add_parser(
        "cache-dir",
        aliases=["path"],
        help="Print the on-disk cache dir for the downloaded data "
             "bundle (override via PIRLYGENES_BUNDLED_DATA).",
    )
    data_cache_dir.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
    )
    data_fetch_parser = data_sub.add_parser(
        "fetch",
        help="Explicitly download the data bundle from the GitHub "
             "Release matching the installed version.",
    )
    data_fetch_parser.add_argument(
        "--offline", action="store_true",
        help="Disallow network downloads (equivalent to DATACACHE_OFFLINE=1).",
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
    prune_parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
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
            "  covering-set       cumulative patient + cancer-type coverage of\n"
            "                     a greedy panel, on one chart\n"
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
            "patients express each gene of a panel above clean-TPM or within-\n"
            "sample percentile thresholds, and compute greedy co-occurrence-\n"
            "aware coverage. Auto mode uses oncoref source-scale metadata to\n"
            "avoid comparing microarray proxies as absolute TPM. Writes a\n"
            "counts CSV + a stacked coverage bar + coverage-curve panels."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pirlygenes plot patient-coverage --gene-set cta\n"
            "  pirlygenes plot patient-coverage --gene-set cta --source all\n"
            "  pirlygenes plot patient-coverage --gene-set lineage:PRAD --cohort PRAD\n"
            "  pirlygenes plot patient-coverage --gene-set cta "
            "--threshold-mode percentile --threshold 95\n"
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
        "--threshold-mode",
        choices=("auto", "tpm", "percentile"),
        default="auto",
        help=("threshold contract (default: auto; clean TPM only when every "
              "selected source is owner-marked comparable, otherwise percentile)"),
    )
    pc.add_argument(
        "--threshold", type=int, default=None,
        help="plot cutoff (default: 25 for TPM mode, 95 for percentile mode)",
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
            "Rebuild CTA source overlaps, funnels and evidence figures from the\n"
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

    cs = plot_sub.add_parser(
        "covering-set",
        help="Cumulative patient + cancer-type coverage of a greedy gene panel.",
        description=(
            "Greedy weighted set cover over the cancer-type registry: how few\n"
            "targets it takes before most cancer PATIENTS — and most cancer\n"
            "TYPES — have at least one panel member expressed at an actionable\n"
            "level. Both lines go on one axes, because they disagree: a panel\n"
            "tuned for patient count chases the common carcinomas while one\n"
            "tuned for type count picks up the rare entities.\n\n"
            "Patient weighting comes from the curated disease-burden table\n"
            "(cancer_burden / burden_category), not a hand-maintained incidence\n"
            "dict. A cancer type with no burden category carries zero patient\n"
            "weight and is named on the figure rather than dropped."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  pirlygenes plot covering-set\n"
            "  pirlygenes plot covering-set --stat median --threshold 10\n"
            "  pirlygenes plot covering-set --gene-set cta --out covset_out\n"
        ),
    )
    cs.add_argument(
        "--gene-set", default="cta",
        help="gene set to build the panel from (default: %(default)s)")
    # Literals, not coverage.* — importing the coverage module here would pull
    # pandas/numpy into every `pirlygenes` invocation, including `--help`. The
    # stat names are pinned against coverage.COVERING_SET_STATS by a test, and
    # --threshold defers to coverage.ACTIONABLE_TPM when left unset.
    cs.add_argument(
        "--stat", default="q3", choices=("median", "q3"),
        help="per-cohort statistic a target must clear. CTAs are subset "
             "antigens, so q3 (a target in >=25%% of patients) is the "
             "clinically relevant bar; median hides them. (default: %(default)s)")
    cs.add_argument(
        "--threshold", type=float, default=None,
        help="clean TPM above which a target counts as actionable "
             "(default: the packaged actionable-TPM bar)")
    cs.add_argument(
        "--n-genes", type=int, default=25,
        help="panel size to plot (default: %(default)s; 0 plots the full cover)")
    cs.add_argument(
        "--metric", default="us_incidence_pct",
        help="burden column used for patient weighting (default: %(default)s)")
    cs.add_argument(
        "--cohort", action="append", dest="codes", default=None,
        help="restrict to these cancer types (repeatable)")
    cs.add_argument(
        "--out", default="covering_set_out",
        help="output directory (default: %(default)s)")

    for name in sorted(_ANALYSIS_SUBCOMMANDS):
        moved = subparsers.add_parser(name, help="Moved to pirl-trufflepig.")
        moved.add_argument(
            "remainder",
            nargs=argparse.REMAINDER,
            help=argparse.SUPPRESS,
        )

    return parser


@contextlib.contextmanager
def _offline_override(offline: bool):
    if not offline:
        yield
        return
    prev = os.environ.get("DATACACHE_OFFLINE")
    os.environ["DATACACHE_OFFLINE"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("DATACACHE_OFFLINE", None)
        else:
            os.environ["DATACACHE_OFFLINE"] = prev


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def cmd_downloads_list(_args: argparse.Namespace) -> int:
    usages = downloads.collect_cache_usage()
    sys.stdout.write(downloads.render_list(usages) + "\n")
    return 0


def cmd_downloads_cache_dir(args: argparse.Namespace) -> int:
    from oncoref import source_matrices

    path = str(source_matrices.cache_dir())
    if getattr(args, "json", False):
        sys.stdout.write(json.dumps({"path": path}, indent=2) + "\n")
    else:
        sys.stdout.write(path + "\n")
    return 0


def cmd_downloads_fetch(args: argparse.Namespace) -> int:
    import datacache
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
    offline = bool(getattr(args, "offline", False))
    with _offline_override(offline):
        for code in codes:
            try:
                source_matrices.fetch(code)
            except (source_matrices.SourceMatrixError, datacache.OfflineError) as err:
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


def cmd_data_dirs(args: argparse.Namespace) -> int:
    import datacache

    loc = downloads.resolve_cache_location()
    cache_path = Path(loc.path)
    cache_bytes = _dir_size_bytes(cache_path)

    bundle_path = data_bundle.cache_dir()
    bundle_exists = bundle_path.exists()
    bundle_bytes = _dir_size_bytes(bundle_path)
    bundle_env = "PIRLYGENES_BUNDLED_DATA" if os.environ.get("PIRLYGENES_BUNDLED_DATA") else None

    oncoref_info: dict[str, object] | None = None
    try:
        from oncoref import source_matrices

        mat_path = Path(source_matrices.cache_dir())
        oncoref_info = {
            "path": str(mat_path),
            "env_var": (
                "CANCERDATA_SOURCE_MATRICES"
                if os.environ.get("CANCERDATA_SOURCE_MATRICES")
                else None
            ),
            "exists": mat_path.exists(),
            "size_bytes": _dir_size_bytes(mat_path),
        }
    except Exception:  # noqa: BLE001 — optional/unavailable oncoref inspection
        oncoref_info = None

    offline_src = datacache.offline_source()
    payload: dict[str, object] = {
        "package": "pirlygenes",
        "version": __version__,
        "data_version": data_bundle.DATA_VERSION,
        "offline": offline_src is not None,
        "offline_source": offline_src,
        "cache_root": {
            "path": str(cache_path),
            "origin": loc.origin,
            "detail": loc.detail,
            "exists": cache_path.exists(),
            "size_bytes": cache_bytes,
        },
        "bundled_data": {
            "path": str(bundle_path),
            "env_var": bundle_env,
            "exists": bundle_exists,
            "size_bytes": bundle_bytes,
            "all_local": data_bundle.is_local(),
        },
        "oncoref_matrices": oncoref_info,
    }
    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0

    origin_text = f"{loc.origin} ({loc.detail})" if loc.detail else loc.origin
    sys.stdout.write(f"pirlygenes cache locations (v{__version__}, data v{data_bundle.DATA_VERSION})\n")
    sys.stdout.write(
        f"  cache_root       : {cache_path} "
        f"[{origin_text}; {datacache.format_bytes(cache_bytes)}]\n"
    )
    bundle_state = "ready" if data_bundle.is_local() else ("partial" if bundle_exists else "not created")
    sys.stdout.write(
        f"  bundled_data     : {bundle_path} "
        f"[{bundle_state}; {datacache.format_bytes(bundle_bytes)}]\n"
    )
    if oncoref_info is not None:
        mat_state = (
            datacache.format_bytes(int(oncoref_info["size_bytes"]))
            if oncoref_info["exists"]
            else "not created"
        )
        sys.stdout.write(
            f"  oncoref_matrices : {oncoref_info['path']} [{mat_state}]\n"
        )
    if offline_src:
        sys.stdout.write(f"  offline          : enabled ({offline_src})\n")
    else:
        sys.stdout.write("  offline          : disabled\n")
    return 0


def cmd_data_status(args: argparse.Namespace) -> int:
    snap = data_bundle.status()
    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(snap, indent=2) + "\n")
        return 0
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


def cmd_data_cache_dir(args: argparse.Namespace) -> int:
    path = str(data_bundle.cache_dir())
    if getattr(args, "json", False):
        sys.stdout.write(
            json.dumps(
                {"path": path, "data_dir": str(data_bundle.data_dir())},
                indent=2,
            )
            + "\n"
        )
    else:
        sys.stdout.write(path + "\n")
    return 0


def cmd_data_fetch(args: argparse.Namespace) -> int:
    offline = True if getattr(args, "offline", False) else None
    try:
        data_bundle.fetch(verbose=True, offline=offline)
        return 0
    except Exception as exc:
        sys.stderr.write(f"pirlygenes data fetch failed: {exc}\n")
        return 1


def _serialize_version_entry(entry: dict) -> dict[str, object]:
    return {
        "version": entry["version"],
        "path": str(entry["path"]),
        "size_bytes": int(entry["size_bytes"]),
        "is_current": bool(entry["is_current"]),
    }


def cmd_data_prune(args: argparse.Namespace) -> int:
    as_json = bool(getattr(args, "json", False))
    keep_current = not getattr(args, "include_current", False)
    dry_run = not getattr(args, "yes", False)
    versions = data_bundle.list_cache_versions()
    if not versions:
        if as_json:
            sys.stdout.write(
                json.dumps(
                    {
                        "cache_root": str(data_bundle.cache_root()),
                        "dry_run": dry_run,
                        "keep_current": keep_current,
                        "versions": [],
                        "candidates": [],
                        "deleted": [],
                        "reclaimable_bytes": 0,
                        "freed_bytes": 0,
                    },
                    indent=2,
                )
                + "\n"
            )
            return 0
        sys.stdout.write(
            f"pirlygenes: no cache dirs under "
            f"{data_bundle.cache_root()}; nothing to prune.\n"
        )
        return 0
    candidates = data_bundle.prune_cache(
        keep_current=keep_current, dry_run=True,
    )
    reclaimable_bytes = sum(int(c["size_bytes"]) for c in candidates)
    deleted: list[dict] = []
    if not dry_run and candidates:
        deleted = data_bundle.prune_cache(
            keep_current=keep_current, dry_run=False,
        )
    freed_bytes = sum(int(c["size_bytes"]) for c in deleted)

    if as_json:
        sys.stdout.write(
            json.dumps(
                {
                    "cache_root": str(data_bundle.cache_root()),
                    "dry_run": dry_run,
                    "keep_current": keep_current,
                    "versions": [_serialize_version_entry(e) for e in versions],
                    "candidates": [_serialize_version_entry(e) for e in candidates],
                    "deleted": [_serialize_version_entry(e) for e in deleted],
                    "reclaimable_bytes": reclaimable_bytes,
                    "freed_bytes": freed_bytes,
                },
                indent=2,
            )
            + "\n"
        )
        return 0

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
    total_mb = reclaimable_bytes / 1e6
    if dry_run:
        sys.stdout.write(
            f"\ndry run — would free {total_mb:.1f} MB across "
            f"{len(candidates)} dir(s). Re-run with --yes to delete.\n"
        )
        return 0
    sys.stdout.write(
        f"\ndeleted {len(deleted)} cache dir(s), freed {freed_bytes / 1e6:.1f} MB.\n"
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
            threshold=args.threshold, threshold_mode=args.threshold_mode,
            out_dir=args.out,
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
        f"({result['threshold_label']}; mode={result['threshold_mode']})\n"
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


def cmd_plot_covering_set(args: argparse.Namespace) -> int:
    from . import coverage

    threshold = (coverage.ACTIONABLE_TPM if args.threshold is None
                 else args.threshold)
    try:
        result = coverage.render_covering_set(
            args.gene_set, stat=args.stat, threshold=threshold,
            codes=args.codes, n_genes=args.n_genes, metric=args.metric,
            out_dir=args.out)
    except Exception as exc:  # noqa: BLE001 — data/matplotlib failure -> clean exit
        sys.stderr.write(f"error: could not render the covering set: {exc}\n")
        return 2
    sys.stdout.write(
        f"{result['label']} covering set ({result['stat']}, "
        f"> {result['threshold']:g} clean TPM): "
        f"{result['n_coverable']}/{result['n_cancer_types']} cancer types "
        f"reachable by at least one target\n")
    if result["unmapped_codes"]:
        sys.stdout.write(
            f"  warning: no curated burden category (zero patient weight) for "
            f"{len(result['unmapped_codes'])}: "
            f"{', '.join(sorted(result['unmapped_codes']))}\n")
    for kind, path in result["paths"].items():
        sys.stdout.write(f"  {kind}: {path}\n")
    return 0


_PLOT_DISPATCH = {
    "patient-coverage": cmd_plot_patient_coverage,
    "cta-curation": cmd_plot_cta_curation,
    "covering-set": cmd_plot_covering_set,
}


def cmd_analysis_moved(_args: argparse.Namespace) -> int:
    sys.stderr.write(_ANALYSIS_MOVED_MESSAGE)
    return 2


_DOWNLOADS_DISPATCH = {
    "list": cmd_downloads_list,
    "cache-dir": cmd_downloads_cache_dir,
    "path": cmd_downloads_cache_dir,
    "fetch": cmd_downloads_fetch,
    "prune": cmd_downloads_prune,
}


_DATA_DISPATCH = {
    "dirs": cmd_data_dirs,
    "list": cmd_data_list,
    "sources": cmd_data_sources,
    "status": cmd_data_status,
    "cache-dir": cmd_data_cache_dir,
    "path": cmd_data_cache_dir,
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
                "usage: pirlygenes downloads {list,cache-dir,path,fetch,prune}\n"
            )
            return 2
        return handler(args)

    if subcommand == "data":
        handler = _DATA_DISPATCH.get(args.data_action)
        if handler is None:
            sys.stderr.write(
                "usage: pirlygenes data {dirs,list,sources,status,cache-dir,path,fetch,prune}\n"
            )
            return 2
        return handler(args)

    if subcommand == "plot":
        handler = _PLOT_DISPATCH.get(args.plot_action)
        if handler is None:
            sys.stderr.write(
                "usage: pirlygenes plot {patient-coverage,covering-set,cta-curation}\n")
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
