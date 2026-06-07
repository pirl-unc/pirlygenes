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

The packaged cohort summaries store per-cohort percentiles only, so they can't
answer "how many *patients* in this cohort express gene X above 50 TPM" or
"as I add genes to a panel, how many *new* patients do I pick up". Those are
cohort-level questions that need the per-sample matrices the builders cache
(``<source>/derived/<stem>_per_sample_tpm.parquet``, linear TPM). This module
reads those, restricted to a named gene set, and computes:

  * per (cohort × gene × threshold) patient counts and percentages, and
  * greedy co-occurrence-aware coverage — as genes are added in the order that
    maximises *new* distinct patients over threshold, the cumulative fraction
    of patients with >=1 panel gene over threshold (a patient expressing
    several panel genes is counted once).

It is the engine behind ``pirlygenes plot patient-coverage`` and generalises
the CTA-specific analysis in ``analyses/cta_patient_counts.py`` to any panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import cohorts as _cohorts
from . import gene_sets_cancer as gsc

DEFAULT_SOURCE = "treehouse-polya-25-01"
DEFAULT_THRESHOLDS = (25, 50, 100, 200)


def _available(source_id):
    """Cohorts with cached per-sample matrices for ``source_id``, or — when
    ``source_id == "all"`` — across every registered per-sample source (#275).
    Cross-source is safe because each cohort carries its own ``source_id`` and
    :func:`cohort_matrix` reads that cohort's own parquet."""
    if source_id == "all":
        return _cohorts.all_available_cohorts()
    return _cohorts.available_cohorts(source_id)

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
        df = gsc.lineage_genes_df(code)
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

def cohort_matrix(cohort, ensgs=None) -> pd.DataFrame:
    """Per-sample TPM matrix for ``cohort``, restricted to the panel rows.

    Matching is on the unversioned Ensembl gene id only — symbols are never a
    join key. Returns an ENSG-indexed, sample-columned DataFrame (linear TPM);
    a ``{ensg: symbol}`` display map is stashed in ``df.attrs['symbols']`` so
    downstream rendering can label rows without ever joining on the symbol.
    """
    path = _cohorts.parquet_path(cohort)
    if not path.exists():
        raise FileNotFoundError(
            f"no per-sample parquet for {cohort.code} at {path} — run "
            f"`pirlygenes build {cohort.source_id}` or `downloads fetch` first")
    df = pd.read_parquet(path)
    ensgs = ensgs or set()
    ensg_col = df["Ensembl_Gene_ID"].astype(str).str.split(".").str[0]
    mask = ensg_col.isin(ensgs) if ensgs else pd.Series(False, index=df.index)
    sub = df.loc[mask].copy()
    sub["Ensembl_Gene_ID"] = ensg_col[mask]
    sample_cols = [c for c in sub.columns if c not in ("Ensembl_Gene_ID", "Symbol")]
    symbol_map = {}
    if "Symbol" in sub.columns:
        symbol_map = dict(zip(sub["Ensembl_Gene_ID"], sub["Symbol"].astype(str)))
    out = sub.set_index("Ensembl_Gene_ID")[sample_cols]
    out.attrs["symbols"] = symbol_map
    return out


def greedy_coverage(mat: pd.DataFrame, threshold: float):
    """Greedily order genes by marginal NEW patients (>threshold).

    Returns ``(ordered_row_positions, cumulative_fraction, n_samples)``."""
    arr = mat.to_numpy()
    n = arr.shape[1]
    if n == 0 or arr.shape[0] == 0:
        return [], [], n
    hit = arr > threshold
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


def patient_coverage(gene_set: str, source_id: str = DEFAULT_SOURCE,
                     codes=None, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    """Long table: for every (cohort × gene) the number and % of patients with
    TPM above each threshold. Only genes with at least one hit are kept.

    ``codes`` optionally restricts to specific cancer types (resolved through
    :func:`gene_sets_cancer.resolve_cancer_type`); default is every cohort with
    a cached per-sample matrix for ``source_id``.
    """
    _label, ensgs = resolve_gene_set(gene_set)
    avail = _available(source_id)
    if codes:
        want = {gsc.resolve_cancer_type(c) for c in codes}
        avail = {k: v for k, v in avail.items() if k in want}
    rows = []
    for code, cohort in avail.items():
        mat = cohort_matrix(cohort, ensgs)
        n = mat.shape[1]
        if n == 0:
            continue
        symbols = mat.attrs.get("symbols", {})
        for ensg, vals in zip(mat.index, mat.to_numpy()):
            rec = {"cancer_code": code, "n_samples": n,
                   "Ensembl_Gene_ID": ensg, "Symbol": symbols.get(ensg, "")}
            any_hit = False
            for t in thresholds:
                k = int((vals > t).sum())
                rec[f"n_gt{t}"] = k
                rec[f"pct_gt{t}"] = round(100 * k / n, 2)
                any_hit = any_hit or k > 0
            if any_hit:
                rows.append(rec)
    cols = (["cancer_code", "n_samples", "Ensembl_Gene_ID", "Symbol"]
            + [f"{p}_gt{t}" for t in thresholds for p in ("n", "pct")])
    return pd.DataFrame(rows, columns=cols)


# --- rendering (CLI) -------------------------------------------------------

_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#469990", "#9A6324", "#800000", "#808000",
    "#000075", "#e6beff", "#aaffc3", "#ffd8b1", "#a9a9a9", "#fabed4",
]


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")


def render(gene_set: str, source_id: str = DEFAULT_SOURCE, codes=None,
           threshold: int = 25, thresholds=DEFAULT_THRESHOLDS,
           out_dir="coverage_out") -> dict:
    """Compute patient coverage for ``gene_set`` and write a counts CSV plus two
    figures (a per-CTA-style stacked coverage bar and a coverage-curve
    small-multiples) into ``out_dir``. Returns a dict of written paths + the
    counts DataFrame. ``threshold`` is the TPM cutoff used for the two plots;
    ``thresholds`` are the cutoffs tabulated in the CSV.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label, ensgs = resolve_gene_set(gene_set)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = _slug(label)

    counts = patient_coverage(gene_set, source_id, codes, thresholds)
    csv_path = out / f"{slug}_patient_counts.csv"
    counts.sort_values(["cancer_code", f"n_gt{threshold}"],
                       ascending=[True, False]).to_csv(csv_path, index=False)

    # Per-cohort greedy coverage (reuse the loaded matrices once). Greedy order
    # is computed on ENSG-indexed rows; symbols are mapped in only for display.
    avail = _available(source_id)
    if codes:
        want = {gsc.resolve_cancer_type(c) for c in codes}
        avail = {k: v for k, v in avail.items() if k in want}
    per = []  # (code, n, cum, gene_display_names_in_greedy_order)
    for code, cohort in avail.items():
        mat = cohort_matrix(cohort, ensgs)
        order, cum, n = greedy_coverage(mat, threshold)
        if cum:
            symbols = mat.attrs.get("symbols", {})
            names = [symbols.get(mat.index[i]) or mat.index[i] for i in order]
            per.append((code, n, cum, names))
    per.sort(key=lambda t: t[2][-1])  # ascending plateau -> broadest at top

    paths = {"counts_csv": str(csv_path)}
    if per:
        paths["stacked_bar"] = str(_stacked_bar(
            per, label, threshold, out / f"{slug}_stacked_coverage_t{threshold}.png",
            plt))
        paths["coverage_curves"] = str(_coverage_curves(
            per, label, threshold,
            out / f"{slug}_coverage_curves_t{threshold}.png", plt))
    return {"paths": paths, "counts": counts, "label": label,
            "n_cohorts": len(per)}


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
            if marg >= 3.0 or j == 0:
                if marg >= 1.5:
                    ax.text(left + marg / 2, y, nm, va="center", ha="center",
                            fontsize=4.5, clip_on=True)
            left += marg
    ax.set_yticks(range(len(per)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of patients with ≥1 {label} gene > {threshold} TPM "
                  "(stacked by each gene's marginal new-patient share, greedy)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(f"{label} coverage by cancer type, split by gene "
                 f"(> {threshold} TPM, {len(per)} cohorts)", fontsize=11)
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
                 f"with ≥1 gene > {threshold} TPM (sorted by plateau)",
                 fontsize=11)
    fig.supxlabel("# genes added (greedy)", fontsize=8)
    fig.supylabel("% patients covered", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
