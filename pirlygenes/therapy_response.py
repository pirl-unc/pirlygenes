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

"""Therapy-response gene signatures (issue #57).

Curated per-axis panels that let a single RNA-seq sample be scored for
the signaling state of each axis (AR, ER, HER2, MAPK/EGFR, NE
differentiation, EMT, hypoxia, IFN response) and — crucially —
annotated against the cancer-type cohort median so suppressed states
surface as divergence below cohort rather than just raw low expression.

Motivation: expression alone carries strong signal about what therapy
a tumor has been exposed to. A PRAD sample with KLK2/KLK3/TMPRSS2/NKX3-1
suppressed and FOLH1 (PSMA) elevated + NE markers rising is almost
certainly ADT-treated, possibly with early lineage plasticity. The
scoring module surfaces that chain explicitly, so readers see

    **AR signaling suppressed** (z=-2.1 vs cohort; KLK3 0.12x cohort,
    FOLH1 4.3x cohort, NKX3-1 0.08x cohort) — pattern consistent with
    ADT exposure

rather than having to reconstruct it from per-gene TPM tables.

Five-step attribution flow placement: this is a **step 3/4** layer.
Step 1 (SampleContext) covers library prep / preservation; step 2
handles coarse TME decomposition; step 3 refines TME subtype / state;
step 4 adjusts tumor values before claiming. Therapy-response adds
to step 4 by explaining *why* a tumor gene is high or low independent
of TME mis-attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .load_dataset import get_data


# ── Data access ───────────────────────────────────────────────────────────


def load_therapy_signatures():
    """Return ``{therapy_class: {"up": [Gene, ...], "down": [Gene, ...]}}``.

    Each ``Gene`` is a ``dict`` carrying symbol, ensembl_gene_id,
    mechanism, cancer_context (semicolon-separated TCGA codes or
    ``pan_cancer``), strength (``canonical``/``supported``/``emerging``),
    and refs string.  Callers use the mechanism / refs fields to
    annotate therapy targets in the report.
    """
    df = get_data("therapy-response-signatures")
    out: dict[str, dict[str, list[dict]]] = {}
    for _, row in df.iterrows():
        cls = str(row["therapy_class"])
        direction = str(row["direction"]).strip()
        rec = {
            "symbol": str(row["symbol"]),
            "ensembl_gene_id": str(row["ensembl_gene_id"]),
            "mechanism": str(row.get("mechanism", "")),
            "cancer_context": str(row.get("cancer_context", "")),
            "strength": str(row.get("strength", "")),
            "refs": str(row.get("refs", "")),
        }
        cls_entry = out.setdefault(cls, {"up": [], "down": []})
        cls_entry.setdefault(direction, []).append(rec)
    return out


def _sigs_for_cancer(all_sigs, cancer_code):
    """Return only the therapy classes where at least one gene applies
    to ``cancer_code`` (either ``pan_cancer`` or an explicit match).
    """
    result = {}
    for cls, directions in all_sigs.items():
        kept_up = [g for g in directions.get("up", []) if _applies(g, cancer_code)]
        kept_down = [g for g in directions.get("down", []) if _applies(g, cancer_code)]
        if kept_up or kept_down:
            result[cls] = {"up": kept_up, "down": kept_down}
    return result


def _applies(gene_rec, cancer_code) -> bool:
    ctx = str(gene_rec.get("cancer_context") or "").strip()
    if not ctx or ctx == "pan_cancer":
        return True
    codes = {part.strip().upper() for part in ctx.split(";") if part.strip()}
    return str(cancer_code).upper() in codes


# ── Per-axis scoring ──────────────────────────────────────────────────────


@dataclass
class TherapyAxisScore:
    """Single-sample signaling-state readout for one therapy axis."""

    therapy_class: str
    state: str  # "up" / "down" / "mixed" / "indeterminate"
    up_geomean_fold: Optional[float] = None
    down_geomean_fold: Optional[float] = None
    up_genes_measured: int = 0
    down_genes_measured: int = 0
    per_gene: list[dict] = field(default_factory=list)
    message: str = ""


def _cohort_median_for_symbol(symbol, cancer_code, ref_flat):
    """Return the TCGA cohort median (FPKM) for a symbol in a cancer
    type, or None if the cohort column is missing or the symbol is
    absent from the reference universe."""
    col = f"FPKM_{cancer_code}"
    if col not in ref_flat.columns:
        return None
    sub = ref_flat[ref_flat["Symbol"] == symbol]
    if sub.empty:
        return None
    val = float(sub.iloc[0][col])
    if val != val:  # NaN check without numpy
        return None
    return val


def score_therapy_signatures(
    sample_tpm_by_symbol,
    cancer_type,
):
    """Score each applicable therapy axis for a sample.

    For every axis relevant to ``cancer_type``:

    - For each ``up`` gene, compute sample/cohort fold change
      (1.0 ⇒ equals cohort; >1 ⇒ elevated; <1 ⇒ suppressed).
    - Geomean the fold changes across ``up`` genes → ``up_geomean_fold``.
    - Same for ``down`` genes → ``down_geomean_fold``.
    - State is ``up``, ``down``, ``mixed`` or ``indeterminate`` based on
      which sides diverge ≥2× from cohort.

    Returns ``{therapy_class: TherapyAxisScore}``.

    ``sample_tpm_by_symbol`` is the dict produced by
    ``tumor_purity._build_sample_tpm_by_symbol`` or the
    ``sample_context._build_tpm_by_symbol`` fallback.
    """
    import numpy as np

    from .gene_sets_cancer import pan_cancer_expression

    all_sigs = load_therapy_signatures()
    applicable = _sigs_for_cancer(all_sigs, cancer_type)
    if not applicable:
        return {}

    ref_flat = pan_cancer_expression().drop_duplicates(subset="Symbol")

    out = {}
    for cls, directions in applicable.items():
        per_gene = []
        up_folds = []
        down_folds = []
        for direction, genes in (("up", directions["up"]), ("down", directions["down"])):
            for g in genes:
                sym = g["symbol"]
                sample_tpm = sample_tpm_by_symbol.get(sym)
                cohort_med = _cohort_median_for_symbol(sym, cancer_type, ref_flat)
                if sample_tpm is None or cohort_med is None:
                    continue
                # Cohort-referenced fold change. Pseudocount of 0.5 TPM
                # keeps the ratio well-defined when either side is zero.
                fold = (float(sample_tpm) + 0.5) / (float(cohort_med) + 0.5)
                per_gene.append({
                    "symbol": sym,
                    "direction": direction,
                    "sample_tpm": round(float(sample_tpm), 2),
                    "cohort_median": round(float(cohort_med), 2),
                    "fold_vs_cohort": round(fold, 3),
                    "mechanism": g["mechanism"],
                    "strength": g["strength"],
                })
                if direction == "up":
                    up_folds.append(fold)
                else:
                    down_folds.append(fold)

        up_geo = float(np.exp(np.mean(np.log(up_folds)))) if up_folds else None
        down_geo = float(np.exp(np.mean(np.log(down_folds)))) if down_folds else None

        # Interpretive state. Thresholds:
        #   >2× cohort on the "up" side  → axis is *active*
        #   <0.5× cohort on the "up" side → axis is *suppressed*
        # (the "down" genes are already expected to move opposite to
        # signaling, so they confirm rather than drive the call.)
        if up_geo is None and down_geo is None:
            state = "indeterminate"
            message = "No cohort-resolvable genes on either side"
        elif up_geo is not None and up_geo >= 2.0 and (down_geo is None or down_geo < 1.5):
            state = "up"
            message = (
                f"Active signaling: up-panel geomean {up_geo:.2f}× cohort"
                + (f", down-panel {down_geo:.2f}× cohort" if down_geo else "")
            )
        elif up_geo is not None and up_geo <= 0.5 and (down_geo is None or down_geo >= 1.5):
            state = "down"
            message = (
                f"Suppressed: up-panel geomean {up_geo:.2f}× cohort"
                + (
                    f", down-panel {down_geo:.2f}× cohort (therapy-response genes elevated)"
                    if down_geo and down_geo >= 1.5 else ""
                )
            )
        elif up_geo is not None and up_geo > 1.5 and down_geo is not None and down_geo > 1.5:
            state = "mixed"
            message = (
                f"Mixed: up-panel {up_geo:.2f}× and down-panel {down_geo:.2f}× "
                "both elevated"
            )
        elif up_geo is not None and 0.5 < up_geo < 2.0:
            state = "indeterminate"
            message = (
                f"Near-cohort baseline: up-panel {up_geo:.2f}× cohort"
                + (f", down-panel {down_geo:.2f}× cohort" if down_geo else "")
            )
        else:
            state = "indeterminate"
            message = "Mixed / weak signal"

        out[cls] = TherapyAxisScore(
            therapy_class=cls,
            state=state,
            up_geomean_fold=round(up_geo, 3) if up_geo is not None else None,
            down_geomean_fold=round(down_geo, 3) if down_geo is not None else None,
            up_genes_measured=len(up_folds),
            down_genes_measured=len(down_folds),
            per_gene=per_gene,
            message=message,
        )
    return out


def symbol_therapy_annotations(all_sigs, cancer_code):
    """Return ``{symbol: [mechanism_string, ...]}`` for every gene
    applicable to ``cancer_code``, used to annotate therapy targets in
    the report so the reader sees *why* a gene's observed expression
    might be where it is.
    """
    out: dict[str, list[str]] = {}
    for cls, directions in all_sigs.items():
        for direction, genes in directions.items():
            for g in genes:
                if not _applies(g, cancer_code):
                    continue
                ann = f"{cls.replace('_', ' ')} {direction}: {g['mechanism']}"
                out.setdefault(g["symbol"], []).append(ann)
    return out
