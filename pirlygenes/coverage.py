# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cohort-level *patient* coverage of a gene set.

The delegated cohort summaries store per-cohort percentiles only, so they can't
answer "how many *patients* in this cohort express gene X above 50 TPM" or
"as I add genes to a panel, how many *new* patients do I pick up". Those are
cohort-level questions that need oncoref's published per-sample source matrices.
This module reads those through the pirlygenes compatibility adapter,
restricted to a named gene set, and computes:

  * per (cohort × gene × threshold) patient counts and percentages, and
  * greedy co-occurrence-aware coverage — as genes are added in the order that
    maximises *new* distinct patients over threshold, the cumulative fraction
    of patients with >=1 panel gene over threshold (a patient expressing
    several panel genes is counted once).

Coverage has two explicit threshold contracts:

``auto`` (the default)
    Uses absolute clean TPM only when oncoref marks every selected source as
    linearly TPM-comparable. Otherwise it uses within-sample percentile rank,
    which is robust to quantifier/platform scale differences.
``tpm`` / ``percentile``
    Force one contract. TPM is rejected for a source that oncoref explicitly
    labels as a microarray TPM proxy; percentile ranks are computed with
    :func:`oncoref.percentile_rank` over each sample's full biological
    transcriptome before the requested panel is selected.

It is the engine behind ``pirlygenes plot patient-coverage`` and generalises
the CTA-specific analysis in ``analyses/cta_patient_counts.py`` to any panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import cohorts as _cohorts
from . import gene_sets_cancer as gsc

DEFAULT_SOURCE = "treehouse-polya-25-01"
DEFAULT_TPM_THRESHOLDS = (25, 50, 100, 200)
DEFAULT_PERCENTILE_THRESHOLDS = (90, 95)

# Historical import compatibility. New code should select the mode-specific
# default through :func:`patient_coverage`.
DEFAULT_THRESHOLDS = DEFAULT_TPM_THRESHOLDS


@dataclass(frozen=True)
class Threshold:
    """One coverage cutoff.

    ``kind`` is ``"tpm"`` or ``"percentile"`` (``"pctile"`` remains an
    accepted analysis-script alias). Absolute TPM uses the historical strict
    ``>`` comparison; percentile mode uses ``>=`` because a p95 call means a
    gene ranks at or above the 95th percentile within that sample.
    """

    kind: str
    value: int

    def __post_init__(self):
        kind = str(self.kind).strip().lower()
        if kind == "pctile":
            kind = "percentile"
        if kind not in {"tpm", "percentile"}:
            raise ValueError("threshold kind must be 'tpm' or 'percentile'")
        numeric = float(self.value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("coverage threshold values must be finite integers")
        value = int(numeric)
        if kind == "tpm" and value < 0:
            raise ValueError("TPM thresholds must be non-negative")
        if kind == "percentile" and not 0 < value <= 100:
            raise ValueError("percentile thresholds must be in (0, 100]")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "value", value)

    @property
    def slug(self) -> str:
        return f"t{self.value}" if self.kind == "tpm" else f"p{self.value}"

    @property
    def count_suffix(self) -> str:
        return f"gt{self.value}" if self.kind == "tpm" else f"p{self.value}"

    @property
    def xlabel(self) -> str:
        return (
            f"> {self.value} clean TPM"
            if self.kind == "tpm"
            else f"≥ {self.value}th within-sample percentile"
        )

    @property
    def label(self) -> str:
        """Compatibility alias used by analysis scripts."""
        return self.xlabel

    @property
    def count_col(self) -> str:
        return f"n_{self.count_suffix}"

    def cutoff(self, cols, pctile_cutoffs=None):
        """Scalar TPM or a per-sample percentile cutoff vector.

        This preserves the precomputed-cutoff interface used by the historical
        CTA analysis while the public coverage path ranks through oncoref.
        Samples without a percentile cutoff receive ``+inf``.
        """
        if self.kind == "tpm":
            return self.value
        col = f"p{self.value}"
        series = pctile_cutoffs[col] if pctile_cutoffs is not None else None
        return np.array([
            series.get(c, np.inf) if series is not None else np.inf
            for c in cols
        ], dtype=float)

    def compare(self, values, cutoff=None):
        """Return the boolean hit mask for ``values``."""
        threshold = self.value if cutoff is None else cutoff
        values = np.asarray(values)
        if self.kind == "tpm":
            return values > threshold
        return values >= threshold

    def hits(self, values, cols=(), pctile_cutoffs=None):
        """Compare raw values using this threshold's scalar/vector cutoff."""
        return self.compare(values, self.cutoff(cols, pctile_cutoffs))


def _available(source_id):
    """Cohorts with cached per-sample matrices for ``source_id``, or — when
    ``source_id == "all"`` — across every registered per-sample source (#275).
    Cross-source is safe because each cohort resolves to its own oncoref
    source-matrix artifact."""
    if source_id == "all":
        return _cohorts.all_available_cohorts()
    return _cohorts.available_cohorts(source_id)


def _selected_cohorts(source_id, codes=None):
    available = _available(source_id)
    if codes:
        wanted = {gsc.resolve_cancer_type(code) for code in codes}
        available = {
            code: cohort
            for code, cohort in available.items()
            if code in wanted
        }
    return available


def _coverage_source_metadata(available):
    """Owner scale/provenance metadata for selected matrix codes.

    The compact oncoref availability table is deliberately used instead of
    inferring comparability from source IDs or pipeline-name substrings. Custom
    compatibility cohorts that are not in oncoref receive explicit ``unknown``
    metadata and therefore make ``auto`` choose percentile mode.
    """
    from oncoref import (
        cancer_reference_expression_availability,
        source_matrices,
    )

    registry = source_matrices.registry()
    registry_rows = {
        str(row["cancer_code"]): row for _, row in registry.iterrows()
    }
    known_codes = [code for code in available if code in registry_rows]
    owner = pd.DataFrame()
    if known_codes:
        owner = cancer_reference_expression_availability(
            cancer_types=known_codes,
            normalize="tpm_clean",
            sample_qc="all",
            reference_source="summary_rows_all",
            all_sources=True,
        )

    out = {}
    for code, cohort in available.items():
        registry_row = registry_rows.get(code)
        source_cohort = (
            str(registry_row["source_cohort"])
            if registry_row is not None
            else cohort.source_id
        )
        if owner.empty:
            match = owner
        else:
            match = owner.loc[
                owner["cancer_code"].astype(str).eq(code)
                & owner["source_cohort"].astype(str).eq(source_cohort)
            ]
        if match.empty and not owner.empty:
            match = owner.loc[
                owner["cancer_code"].astype(str).eq(code)
            ]
        row = match.iloc[0] if not match.empty else None
        comparable = None
        if row is not None and not pd.isna(row.get("linear_tpm_comparable")):
            comparable = bool(row["linear_tpm_comparable"])
        out[code] = {
            "source_cohort": source_cohort,
            "source_type": (
                str(row.get("source_type", "unknown"))
                if row is not None else "unknown"
            ),
            "source_scale_class": (
                str(row.get("source_scale_class", "unknown"))
                if row is not None else "unknown"
            ),
            "linear_tpm_comparable": comparable,
            "normalization": "tpm_clean",
        }
    return out


def _resolve_threshold_mode(requested, metadata):
    mode = str(requested).strip().lower()
    if mode == "pctile":
        mode = "percentile"
    if mode not in {"auto", "tpm", "percentile"}:
        raise ValueError(
            "threshold_mode must be one of 'auto', 'tpm', or 'percentile'"
        )
    if mode == "auto":
        if metadata and all(
            item["linear_tpm_comparable"] is True
            for item in metadata.values()
        ):
            return "tpm"
        return "percentile"
    if mode == "tpm":
        incompatible = sorted(
            code for code, item in metadata.items()
            if item["linear_tpm_comparable"] is False
        )
        if incompatible:
            joined = ", ".join(incompatible)
            raise ValueError(
                "absolute TPM coverage is invalid for non-comparable source "
                f"scale(s): {joined}; use threshold_mode='percentile'"
            )
    return mode


def _threshold_values(mode, thresholds):
    if thresholds is None:
        thresholds = (
            DEFAULT_TPM_THRESHOLDS
            if mode == "tpm"
            else DEFAULT_PERCENTILE_THRESHOLDS
        )
    values = []
    for value in thresholds:
        normalized = Threshold(mode, value).value
        if normalized not in values:
            values.append(normalized)
    return tuple(values)


# --- gene-set resolution ---------------------------------------------------

# A panel is an **ENSG set**. Gene symbols are only ever used to *look up* an
# Ensembl gene id (resolved here, at the boundary) — never as a join/comparison
# key downstream. Every cohort-matrix match below is on Ensembl_Gene_ID alone.
def resolve_gene_set(name: str):
    """Resolve a ``--gene-set`` token to ``(label, ensgs)`` — a set of
    unversioned Ensembl gene ids.

    Supported tokens::

        cta | surfaceome | mito | housekeeping
        therapy:<type>     e.g. therapy:adc, therapy:car-t, therapy:radioligand
        lineage:<code>     per-cancer-type lineage panel (e.g. lineage:PRAD)
        <path>             a CSV/TXT with Symbol and/or Ensembl_Gene_ID column(s),
                           or a single first column of symbols / ENSG ids

    Any symbol-only input is resolved to an ENSG via
    :func:`pirlygenes.gene_ids.find_gene_id_by_name_from_ensembl`; symbols that
    don't resolve are dropped (never silently matched by name downstream).
    """
    token = str(name).strip()
    low = token.lower()
    if low == "cta":
        return "CTA", set(gsc.CTA_gene_ids())
    if low in ("surfaceome", "cancer-surfaceome"):
        return "cancer-surfaceome", set(gsc.cancer_surfaceome_gene_ids())
    if low in ("mito", "mitochondrial"):
        return "mitochondrial", set(gsc.mitochondrial_gene_ids())
    if low == "housekeeping":
        return "housekeeping", set(gsc.housekeeping_gene_ids())
    if low.startswith("therapy:"):
        t = token.split(":", 1)[1]
        return f"therapy:{t}", set(gsc.therapy_target_gene_ids(t))
    if low.startswith("lineage:"):
        code = gsc.resolve_cancer_type(token.split(":", 1)[1])
        # A named coverage panel is a positive gene set. Directional ``low``
        # rows are contrastive evidence and must not be counted as expressed
        # lineage support.
        df = gsc.lineage_genes_df(code, direction="high")
        ensgs = set(df["Ensembl_Gene_ID"].dropna().astype(str).str.split(".").str[0])
        # lineage panels are ENSG-backed; symbols are display-only, not joined.
        return f"lineage:{code}", ensgs
    p = Path(token).expanduser()
    if p.exists():
        return _gene_set_from_file(p)
    raise ValueError(
        f"Unknown --gene-set {name!r}. Use one of: cta, surfaceome, mito, "
        "housekeeping, therapy:<type>, lineage:<code>, or a path to a CSV of "
        "symbols/ENSG ids."
    )


def _symbols_to_ensgs(symbols) -> set:
    """Resolve a set of gene symbols to unversioned ENSGs (symbol used only for
    this lookup; unresolved symbols are dropped)."""
    from .gene_ids import find_gene_id_by_name_from_ensembl, strip_version
    out = set()
    for s in symbols:
        gid = find_gene_id_by_name_from_ensembl(str(s))
        if gid:
            out.add(strip_version(gid))
    return out


def _gene_set_from_file(path: Path):
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    ensgs, symbols = set(), set()
    if "ensembl_gene_id" in cols:
        ensgs |= set(raw[cols["ensembl_gene_id"]].dropna().astype(str)
                     .str.split(".").str[0])
    if "symbol" in cols:
        symbols |= set(raw[cols["symbol"]].dropna().astype(str).str.upper())
    if not ensgs and not symbols:  # bare first column: classify each token
        for v in raw[raw.columns[0]].dropna().astype(str):
            v = v.strip()
            (ensgs.add(v.split(".")[0]) if v.upper().startswith("ENSG")
             else symbols.add(v.upper()))
    # Resolve any symbol-only entries to ENSG up front, so matching is ENSG-only.
    ensgs |= _symbols_to_ensgs(symbols)
    return path.name, ensgs


# --- per-sample access + counting ------------------------------------------

def cohort_matrix(cohort, ensgs=None, *, percentile_rank=False) -> pd.DataFrame:
    """Per-sample clean-TPM matrix for ``cohort``, restricted to panel rows.

    Matching is on the unversioned Ensembl gene id only — symbols are never a
    join key. Owner cohorts are normalized by
    :func:`oncoref.per_sample_expression`; custom compatibility cohorts retain
    the historical reader path. With ``percentile_rank=True``, oncoref ranks
    every gene within each sample *before* selecting panel rows, so the result
    is not the misleading rank within the panel alone.

    Returns an ENSG-indexed, sample-columned DataFrame. A ``{ensg: symbol}``
    display map is stashed in ``df.attrs['symbols']`` so downstream rendering
    can label rows without joining on the symbol.
    """
    import oncoref
    from oncoref import source_matrices

    owner_codes = set(source_matrices.registry()["cancer_code"].astype(str))
    if cohort.code in owner_codes:
        df = oncoref.per_sample_expression(
            cohort.code,
            normalize="tpm_clean",
            auto_fetch=False,
            sample_qc="all",
        )
    else:
        df = _cohorts.read_per_sample(cohort)
    sample_cols = _cohorts.sample_columns(df)
    if percentile_rank:
        df = oncoref.percentile_rank(df, value_cols=sample_cols)
    ensgs = ensgs or set()
    ensg_col = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    mask = ensg_col.isin(ensgs) if ensgs else pd.Series(False, index=df.index)
    sub = df.loc[mask].copy()
    sub["Ensembl_Gene_ID"] = ensg_col[mask]
    symbol_map = {}
    if "Symbol" in sub.columns:
        symbol_map = dict(zip(sub["Ensembl_Gene_ID"], sub["Symbol"].astype(str)))
    out = sub.set_index("Ensembl_Gene_ID")[sample_cols]
    out.attrs["symbols"] = symbol_map
    return out


def greedy_coverage(mat: pd.DataFrame, threshold, *, inclusive=False):
    """Greedily order genes by marginal new patients at ``threshold``.

    ``threshold`` may be a scalar or a sample-aligned vector. ``inclusive`` is
    used for percentile ranks (at-or-above pN); absolute TPM retains the
    historical strict-greater-than contract. Returns
    ``(ordered_row_positions, cumulative_fraction, n_samples)``.
    """
    arr = mat.to_numpy()
    n = arr.shape[1]
    if n == 0 or arr.shape[0] == 0:
        return [], [], n
    hit = arr >= threshold if inclusive else arr > threshold
    covered = np.zeros(n, dtype=bool)
    order, cum, remaining = [], [], set(range(arr.shape[0]))
    while remaining:
        best, best_gain = None, 0
        for i in remaining:
            gain = int((hit[i] & ~covered).sum())
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None or best_gain <= 0:
            break
        covered |= hit[best]
        order.append(best)
        cum.append(covered.sum() / n)
        remaining.discard(best)
    return order, cum, n


def _coverage_frame(
    ensgs,
    available,
    metadata,
    mode,
    threshold_values,
    *,
    greedy_threshold=None,
):
    """Compute counts and optional greedy curves in one matrix pass.

    Keeping the loop here prevents :func:`render` from loading every source
    matrix twice. Matrices are discarded cohort-by-cohort, bounding memory even
    for ``source_id="all"``.
    """
    thresholds = [Threshold(mode, value) for value in threshold_values]
    rows = []
    per = []
    for code, cohort in available.items():
        mat = cohort_matrix(
            cohort,
            ensgs,
            percentile_rank=(mode == "percentile"),
        )
        n = mat.shape[1]
        if n == 0:
            continue
        source = metadata[code]
        symbols = mat.attrs.get("symbols", {})
        for ensg, vals in zip(mat.index, mat.to_numpy()):
            rec = {
                "cancer_code": code,
                "source_cohort": source["source_cohort"],
                "source_type": source["source_type"],
                "source_scale_class": source["source_scale_class"],
                "linear_tpm_comparable": source["linear_tpm_comparable"],
                "normalization": source["normalization"],
                "threshold_mode": mode,
                "n_samples": n,
                "Ensembl_Gene_ID": ensg,
                "Symbol": symbols.get(ensg, ""),
            }
            any_hit = False
            for threshold in thresholds:
                count = int(threshold.compare(vals).sum())
                rec[f"n_{threshold.count_suffix}"] = count
                rec[f"pct_{threshold.count_suffix}"] = round(
                    100 * count / n, 2
                )
                any_hit = any_hit or count > 0
            if any_hit:
                rows.append(rec)

        if greedy_threshold is not None:
            order, cumulative, _ = greedy_coverage(
                mat,
                greedy_threshold.value,
                inclusive=(mode == "percentile"),
            )
            if cumulative:
                names = [
                    symbols.get(mat.index[index]) or mat.index[index]
                    for index in order
                ]
                per.append((code, n, cumulative, names))

    cols = [
        "cancer_code",
        "source_cohort",
        "source_type",
        "source_scale_class",
        "linear_tpm_comparable",
        "normalization",
        "threshold_mode",
        "n_samples",
        "Ensembl_Gene_ID",
        "Symbol",
    ] + [
        f"{prefix}_{threshold.count_suffix}"
        for threshold in thresholds
        for prefix in ("n", "pct")
    ]
    out = pd.DataFrame(rows, columns=cols)
    out.attrs.update({
        "threshold_mode": mode,
        "thresholds": tuple(threshold.value for threshold in thresholds),
        "source_metadata": metadata,
    })
    return out, per


def patient_coverage(
    gene_set: str,
    source_id: str = DEFAULT_SOURCE,
    codes=None,
    thresholds=None,
    *,
    threshold_mode="auto",
) -> pd.DataFrame:
    """Per-cohort/per-gene patient coverage under one threshold contract.

    ``threshold_mode="auto"`` uses owner ``linear_tpm_comparable`` metadata:
    clean TPM for wholly comparable sources, otherwise within-sample
    percentile rank. Explicit ``"tpm"`` is rejected for sources oncoref marks
    non-comparable; ``"percentile"`` is always platform-safe. TPM output uses
    ``n_gt25``/``pct_gt25``-style columns; percentile output uses
    ``n_p90``/``pct_p90``. Only genes with at least one hit are retained.

    ``codes`` optionally restricts to specific cancer types (resolved through
    :func:`gene_sets_cancer.resolve_cancer_type`); default is every cohort with
    a cached per-sample matrix for ``source_id``.
    """
    _label, ensgs = resolve_gene_set(gene_set)
    avail = _selected_cohorts(source_id, codes)
    metadata = _coverage_source_metadata(avail)
    mode = _resolve_threshold_mode(threshold_mode, metadata)
    threshold_values = _threshold_values(mode, thresholds)
    return _coverage_frame(
        ensgs,
        avail,
        metadata,
        mode,
        threshold_values,
    )[0]


# --- rendering (CLI) -------------------------------------------------------

_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#469990", "#9A6324", "#800000", "#808000",
    "#000075", "#e6beff", "#aaffc3", "#ffd8b1", "#a9a9a9", "#fabed4",
]


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")


def render(
    gene_set: str,
    source_id: str = DEFAULT_SOURCE,
    codes=None,
    threshold=None,
    thresholds=None,
    out_dir="coverage_out",
    *,
    threshold_mode="auto",
) -> dict:
    """Compute patient coverage for ``gene_set`` and write a counts CSV plus two
    figures (a per-CTA-style stacked coverage bar and a coverage-curve
    small-multiples) into ``out_dir``. Returns a dict of written paths + the
    counts DataFrame. ``threshold_mode`` follows :func:`patient_coverage`;
    ``threshold`` is the plotted cutoff and ``thresholds`` are tabulated in the
    CSV. Mode-appropriate defaults are 25 clean TPM or p95.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label, ensgs = resolve_gene_set(gene_set)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = _slug(label)

    avail = _selected_cohorts(source_id, codes)
    metadata = _coverage_source_metadata(avail)
    mode = _resolve_threshold_mode(threshold_mode, metadata)
    requested_plot_value = threshold if threshold is not None else (
        25 if mode == "tpm" else 95
    )
    plot_threshold = Threshold(mode, requested_plot_value)
    plot_value = plot_threshold.value
    table_thresholds = list(_threshold_values(mode, thresholds))
    if plot_value not in table_thresholds:
        table_thresholds.append(plot_value)
    counts, per = _coverage_frame(
        ensgs,
        avail,
        metadata,
        mode,
        table_thresholds,
        greedy_threshold=plot_threshold,
    )
    csv_path = out / f"{slug}_patient_counts.csv"
    counts.sort_values(["cancer_code", plot_threshold.count_col],
                       ascending=[True, False]).to_csv(csv_path, index=False)

    per.sort(key=lambda t: t[2][-1])  # ascending plateau -> broadest at top

    paths = {"counts_csv": str(csv_path)}
    if per:
        paths["stacked_bar"] = str(_stacked_bar(
            per,
            label,
            plot_threshold,
            out / f"{slug}_stacked_coverage_{plot_threshold.slug}.png",
            plt,
        ))
        paths["coverage_curves"] = str(_coverage_curves(
            per,
            label,
            plot_threshold,
            out / f"{slug}_coverage_curves_{plot_threshold.slug}.png",
            plt,
        ))
    return {
        "paths": paths,
        "counts": counts,
        "label": label,
        "n_cohorts": len(per),
        "threshold_mode": mode,
        "threshold": plot_value,
        "threshold_label": plot_threshold.xlabel,
    }


def _gene_color_map(genes_ordered):
    seen, colors = [], {}
    for g in genes_ordered:
        if g not in colors:
            colors[g] = _PALETTE[len(seen) % len(_PALETTE)]
            seen.append(g)
    return colors


def _stacked_bar(per, label, threshold, path, plt):
    """Horizontal stacked bar: each cohort's greedy plateau split into each
    gene's marginal new-patient contribution (segments sum to the plateau)."""
    from collections import Counter
    tot = Counter()
    for _code, _n, cum, names in per:
        prev = 0.0
        for nm, c in zip(names, cum):
            tot[nm] += (c - prev) * 100
            prev = c
    color = _gene_color_map([g for g, _ in tot.most_common()])

    fig, ax = plt.subplots(figsize=(13, max(6, len(per) * 0.28)))
    labels = []
    for y, (code, n, cum, names) in enumerate(per):
        labels.append(f"{gsc.format_cancer_code_label(code)}  (n={n})")
        left, prev = 0.0, 0.0
        for j, (nm, c) in enumerate(zip(names, cum)):
            marg = (c - prev) * 100
            prev = c
            if marg <= 0:
                continue
            ax.barh(y, marg, left=left, color=color.get(nm, "#cccccc"),
                    edgecolor="white", linewidth=0.3)
            if (marg >= 3.0 or j == 0) and marg >= 1.5:
                ax.text(left + marg / 2, y, nm, va="center", ha="center",
                        fontsize=4.5, clip_on=True)
            left += marg
    ax.set_yticks(range(len(per)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of patients with ≥1 {label} gene {threshold.xlabel} "
                  "(stacked by each gene's marginal new-patient share, greedy)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"{label} coverage by cancer type, split by gene "
                 f"({threshold.xlabel}, {len(per)} cohorts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _coverage_curves(per, label, threshold, path, plt):
    """Small-multiples of each cohort's greedy coverage curve (sorted by
    plateau, broadest first)."""
    ordered = sorted(per, key=lambda t: t[2][-1], reverse=True)
    ncol = 6
    nrow = (len(ordered) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 1.9),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, (code, n, cum, names) in zip(axes, ordered):
        xs = range(1, len(cum) + 1)
        ax.plot(xs, [c * 100 for c in cum], color="#b5179e", lw=1.2)
        ax.fill_between(xs, [c * 100 for c in cum], alpha=0.15, color="#b5179e")
        for x, (nm, c) in enumerate(zip(names[:3], cum[:3]), start=1):
            ax.annotate(nm, (x, c * 100), fontsize=4, rotation=45,
                        textcoords="offset points", xytext=(1, 2))
        ax.set_title(f"{gsc.format_cancer_code_label(code)} (n={n}) "
                     f"{cum[-1]*100:.0f}%", fontsize=7)
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=5)
        ax.grid(alpha=0.25)
    for ax in axes[len(ordered):]:
        ax.axis("off")
    fig.suptitle(f"{label} panel coverage by cancer type — distinct patients "
                 f"with ≥1 gene {threshold.xlabel} (sorted by plateau)",
                 fontsize=11)
    fig.supxlabel("# genes added (greedy)", fontsize=8)
    fig.supylabel("% patients covered", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
