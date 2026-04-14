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

"""Sample-context inference — library prep, preservation, degradation.

First stage of the unified attribution flow (see README "Attribution
flow"). Runs before cancer-type inference and produces a ``SampleContext``
that every downstream step consumes as a **base-layer of expression
expectations**: which genes are expected to be over- or under-represented
for artifactual reasons, and how to compensate.

The detection uses signals that are tissue-independent, so this module
can run on a raw expression table without knowing what cancer or normal
tissue the sample is from:

- **Library prep** — replication-dependent histone fraction and MT
  ribosomal-RNA presence. Histone mRNAs and MT rRNAs are *not*
  polyadenylated and therefore disappear under poly-A capture but
  survive in total-RNA / ribosomal-depletion libraries.
- **Preservation (FFPE vs fresh)** — long/short matched-gene-pair ratio
  (from ``data/degradation-gene-pairs.csv``). Long transcripts degrade
  faster than short ones in FFPE.
- **Missing-MT** — complete absence of mitochondrial genes is an
  artifact (pipeline filter, or exome-capture) rather than biology;
  degrades confidence in all MT-based signals.

What downstream steps use the context for:

- Marker selection: downweight long-transcript markers under FFPE; skip
  the extended-housekeeping exclusion universe from tumor-specific
  marker computations.
- Purity estimator: widen confidence intervals proportional to
  degradation severity.
- Decomposition: adjust fit weights for sample_context-biased genes.
- Tumor-value adjustment: subtract TME-explainable expression from
  therapy-target columns (prevents FN1/COL1A1 low-purity inflation).
- Reporting: show the context chain so users see *why* a number was
  adjusted, not just the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ── Reference gene patterns ──────────────────────────────────────────────
#
# Replication-dependent histones (H1-*, H2AC*, H2BC*, H3C*, H4C*) are the
# canonical family of non-polyadenylated mRNAs. Their fraction collapses
# under poly-A selection and remains meaningful under total-RNA /
# ribo-depletion, making them the primary library-prep indicator when
# combined with MT rRNAs.
#
# We match by symbol prefix to cover both HGNC 2020+ names (H2AC1,
# H2BC1, H3C1, H4C1) and the older HIST* synonyms that some annotations
# still emit (HIST1H2BC, etc.). A sample's histone signal is then
# ``sum(TPM for histone symbols) / sum(TPM for all expressed symbols)``.
_HISTONE_SYMBOL_PREFIXES: tuple[str, ...] = (
    "H1-", "H2AC", "H2BC", "H3C", "H4C",
    "HIST1H", "HIST2H", "HIST3H", "HIST4H",
)

# MT ribosomal RNAs — non-polyadenylated. Present in total / ribo-dep,
# absent in poly-A and typically in exome capture. Combined with the
# histone fraction, this disambiguates all four common libraries.
_MT_RRNA_SYMBOLS: frozenset[str] = frozenset({"MT-RNR1", "MT-RNR2"})

# MT protein-coding mRNAs — polyadenylated. Present in every library
# that retains mitochondrial transcripts (total, ribo-dep, poly-A).
# Absence alongside absence of MT rRNAs indicates either exome capture
# (no MT coverage) or an upstream filter that stripped MT contigs.
_MT_MRNA_SYMBOLS: frozenset[str] = frozenset({
    "MT-CO1", "MT-CO2", "MT-CO3",
    "MT-ND1", "MT-ND2", "MT-ND3", "MT-ND4", "MT-ND4L", "MT-ND5", "MT-ND6",
    "MT-CYB", "MT-ATP6", "MT-ATP8",
})


# ── Thresholds ────────────────────────────────────────────────────────────
# Calibrated from ENCODE total-RNA vs poly-A replicates and a TCGA
# poly-A-capture snapshot. Values are conservative — the inference
# returns ``unknown`` rather than guess when signals conflict.

_THRESHOLDS = {
    # Fraction of total expressed TPM that is replication-dependent
    # histone mRNA.  Poly-A libraries: <<0.1%. Total/ribo-dep: 0.3–2%.
    "histone_fraction_polyA_ceiling": 0.002,   # below: likely poly-A
    "histone_fraction_total_floor":   0.005,   # above: likely total / ribo-dep

    # MT rRNA fraction — fraction of all MT TPM attributed to MT-RNR1/2.
    # Total RNA: MT rRNAs dominate MT signal (>70% of MT TPM).
    # Ribo-depleted: rRNAs removed; MT rRNAs usually <10% of MT TPM.
    # Poly-A: rRNAs absent.
    "mt_rrna_fraction_of_mt_depleted_ceiling": 0.10,
    "mt_rrna_fraction_of_mt_total_floor":      0.40,

    # Overall MT expression — expressed as fraction of all sample TPM.
    # <0.1% flagged as "MT missing" (either filtered or exome capture).
    "mt_fraction_suspicious_floor": 0.001,

    # Tempus xT / Twist RNA Exome and similar *hybrid capture / exome-
    # capture RNA* protocols drop MT probes entirely, so MT-total is
    # effectively zero (<0.05%) and MT-rRNA absolutely zero — much
    # stronger than the ``mt_fraction_suspicious_floor``. Used to route
    # to ``exome_capture`` even when histones are in a "limbo" band
    # (0.1-0.5% — not pure poly-A, not true total RNA).
    "mt_fraction_exome_ceiling": 0.0005,

    # Gene-length degradation index (long/short observed/expected
    # median ratio). Calibrated in sample_quality.QUALITY_THRESHOLDS;
    # kept consistent here.
    "degradation_pair_moderate": 0.30,
    "degradation_pair_severe":   0.20,
    # Upper bound on the index: values >> 1.0 indicate systematic
    # over-representation of long transcripts — a dead giveaway for
    # exon capture enrichment (long genes have more probes), NOT fresh
    # RNA. Without this bound the naive "anything > 0.55 is fresh"
    # path mislabels capture-enriched FFPE samples as fresh_frozen.
    "degradation_pair_biased_upper": 1.40,
}


# ── Data class ────────────────────────────────────────────────────────────

@dataclass
class SampleContext:
    """Base-layer of expression expectations for a sample.

    Every downstream step reads this to know which genes will be over-
    or under-represented for *artifactual* reasons, independent of
    biology.  The attribution flow is:

        raw TPM  →  [SampleContext inference]  →  sample_context +
                    expression_priors  →  coarse TME decomposition  →
                    fine TME subtype dissection  →  tumor-value
                    adjustment  →  conservative tumor-specific core

    Fields are all additive: none replace or override existing
    functionality; they *inform* it.  A ``None`` or ``unknown`` field is
    a signal that the context was inconclusive and downstream should
    behave as if the sample has default expectations.
    """

    # Library preparation method inferred from histone + MT rRNA signals.
    # One of: "poly_a", "ribo_depleted", "total_rna", "exome_capture",
    # "unknown". ``poly_a`` is the most common for RNA-seq of clinical
    # samples; ``total_rna`` and ``ribo_depleted`` together cover most
    # research cohorts and recent ribo-depleted clinical protocols.
    library_prep: str = "unknown"
    library_prep_confidence: float = 0.0  # 0.0 (guess) to 1.0 (strong)

    # Preservation / degradation status. ``fresh_frozen`` is the default
    # expectation for a non-degraded sample; ``ffpe`` flags formalin-
    # fixed paraffin-embedded RNA; ``degraded`` catches partial
    # degradation from slow processing or thaw cycles.
    preservation: str = "unknown"          # "fresh_frozen" | "ffpe" | "degraded" | "unknown"
    preservation_confidence: float = 0.0

    # Severity of degradation from the gene-length pair index.
    degradation_severity: str = "none"     # "none" | "mild" | "moderate" | "severe"
    degradation_index: Optional[float] = None  # long/short observed/expected median

    # Whether mitochondrial genes are missing from the quant table.
    # If True, degradation signal cannot be assessed from MT fold and
    # is unreliable.
    missing_mt: bool = False

    # Diagnostic signals used by the inference, retained so the report
    # can explain *why* a call was made. Do not rely on these keys from
    # external code; they are reporting-only.
    signals: dict = field(default_factory=dict)

    # User-readable context flags for console output / summary.md.
    flags: list[str] = field(default_factory=list)

    # ── Computed downstream adjustments ──────────────────────────────

    @property
    def is_ffpe(self) -> bool:
        return self.preservation == "ffpe"

    @property
    def is_degraded(self) -> bool:
        return self.degradation_severity in ("moderate", "severe")

    def long_transcript_weight_factor(self) -> float:
        """Multiplier applied to long-transcript marker weights (#25).

        FFPE samples systematically under-represent long transcripts,
        which biases any decomposition marker with long-transcript-
        dominant reference signal.  Returns 1.0 when no degradation is
        detected and halves progressively as severity rises.
        """
        return {
            "severe":   0.3,
            "moderate": 0.5,
            "mild":     0.75,
            "none":     1.0,
        }.get(self.degradation_severity, 1.0)

    def purity_ci_widening_factor(self) -> float:
        """Multiplicative widening of purity confidence intervals (#26).

        A degraded sample yields noisier purity estimates because the
        upstream gene-expression measurements are noisier.  Returns a
        factor ≥ 1 that downstream CI construction multiplies into its
        half-width.
        """
        return {
            "severe":   1.6,
            "moderate": 1.3,
            "mild":     1.1,
            "none":     1.0,
        }.get(self.degradation_severity, 1.0)

    def summary_line(self) -> str:
        """One-line human summary for reports and stdout."""
        prep = {
            "poly_a":         "poly-A capture",
            "ribo_depleted":  "ribosomal-depleted",
            "total_rna":      "total RNA",
            "exome_capture":  "exome capture",
            "unknown":        "unknown library prep",
        }.get(self.library_prep, self.library_prep)
        pres = {
            "fresh_frozen": "fresh / frozen",
            "ffpe":         "FFPE",
            "degraded":     "partially degraded",
            "unknown":      "unknown preservation",
        }.get(self.preservation, self.preservation)
        deg = "" if self.degradation_severity == "none" else f", {self.degradation_severity} degradation"
        return f"{prep}, {pres}{deg}"


# ── Inference ─────────────────────────────────────────────────────────────


def _sum_tpm_for_symbols(tpm_by_symbol, symbol_set) -> float:
    total = 0.0
    for sym in symbol_set:
        v = tpm_by_symbol.get(sym)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0:
            total += float(v)
    return total


def _sum_tpm_for_prefixes(tpm_by_symbol, prefixes) -> float:
    total = 0.0
    for sym, v in tpm_by_symbol.items():
        if not isinstance(sym, str):
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)) or v <= 0:
            continue
        if sym.startswith(prefixes):
            total += float(v)
    return total


def _infer_library_prep(tpm_by_symbol, signals):
    """Infer library prep from histone + MT rRNA signals. Records
    evidence into ``signals`` and returns ``(library_prep, confidence)``.
    """
    total_tpm = sum(
        v for v in tpm_by_symbol.values()
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0
    )
    if total_tpm <= 0:
        return "unknown", 0.0

    histone_tpm = _sum_tpm_for_prefixes(tpm_by_symbol, _HISTONE_SYMBOL_PREFIXES)
    histone_frac = histone_tpm / total_tpm

    mt_rrna_tpm = _sum_tpm_for_symbols(tpm_by_symbol, _MT_RRNA_SYMBOLS)
    mt_mrna_tpm = _sum_tpm_for_symbols(tpm_by_symbol, _MT_MRNA_SYMBOLS)
    mt_total_tpm = mt_rrna_tpm + mt_mrna_tpm
    mt_fraction = mt_total_tpm / total_tpm
    mt_rrna_fraction_of_mt = (
        mt_rrna_tpm / mt_total_tpm if mt_total_tpm > 0 else None
    )

    signals["histone_fraction"] = round(histone_frac, 5)
    signals["mt_fraction"] = round(mt_fraction, 5)
    signals["mt_rrna_fraction_of_mt"] = (
        round(mt_rrna_fraction_of_mt, 4) if mt_rrna_fraction_of_mt is not None else None
    )

    # Exome / hybrid-capture RNA (Tempus xT, Twist RNA Exome, etc.) —
    # probes don't cover chrM, so BOTH MT-rRNA and MT-mRNA are
    # essentially zero. This is a much stronger signal than the
    # general "MT missing" floor, and — critically — it fires
    # independently of histone fraction. Hybrid-capture panels can
    # legitimately carry histone TPM in the "limbo" band (0.1-0.5%)
    # because some histones have probes and FFPE fragmentation adds
    # background. An earlier histone<0.2% gate wrongly routed these
    # samples to "unknown" (bug report 2026-04-14).
    if (
        mt_fraction < _THRESHOLDS["mt_fraction_exome_ceiling"]
        and (mt_rrna_tpm == 0.0)
    ):
        # Very high confidence when MT-rRNA is *exactly* zero — no
        # polyadenylation protocol and no total-RNA protocol can
        # produce that pattern by design.
        return "exome_capture", 0.9

    # Broader MT-missing signal (may be upstream chrM filter rather
    # than capture — treat as exome_capture at moderate confidence).
    if (
        mt_fraction < _THRESHOLDS["mt_fraction_suspicious_floor"]
        and histone_frac < _THRESHOLDS["histone_fraction_polyA_ceiling"]
    ):
        return "exome_capture", 0.6

    # Total RNA: histones high AND MT rRNAs dominate MT signal
    if (
        histone_frac >= _THRESHOLDS["histone_fraction_total_floor"]
        and mt_rrna_fraction_of_mt is not None
        and mt_rrna_fraction_of_mt >= _THRESHOLDS["mt_rrna_fraction_of_mt_total_floor"]
    ):
        return "total_rna", 0.9

    # Ribo-depleted: histones high AND MT rRNAs depleted
    if (
        histone_frac >= _THRESHOLDS["histone_fraction_total_floor"]
        and mt_rrna_fraction_of_mt is not None
        and mt_rrna_fraction_of_mt <= _THRESHOLDS["mt_rrna_fraction_of_mt_depleted_ceiling"]
    ):
        return "ribo_depleted", 0.85

    # Poly-A: histones absent AND MT rRNAs absent
    if (
        histone_frac <= _THRESHOLDS["histone_fraction_polyA_ceiling"]
        and (mt_rrna_fraction_of_mt is None or mt_rrna_fraction_of_mt <= _THRESHOLDS["mt_rrna_fraction_of_mt_depleted_ceiling"])
        and mt_fraction >= _THRESHOLDS["mt_fraction_suspicious_floor"]
    ):
        return "poly_a", 0.8

    # Ambiguous: histones low, MT signal mixed. Most clinical cohorts
    # are poly-A so default there at low confidence.
    if histone_frac <= _THRESHOLDS["histone_fraction_polyA_ceiling"]:
        return "poly_a", 0.5

    return "unknown", 0.2


def _infer_preservation_and_degradation(tpm_by_symbol, signals):
    """Infer preservation + degradation severity using the gene-length
    pair index from ``data/degradation-gene-pairs.csv``.

    Returns ``(preservation, preservation_confidence, severity, index)``.
    """
    from .gene_sets_cancer import degradation_gene_pairs

    ratios = []
    n_short_expressed = 0
    for short_gene, long_gene, expected in degradation_gene_pairs():
        short_tpm = tpm_by_symbol.get(short_gene)
        long_tpm = tpm_by_symbol.get(long_gene)
        if short_tpm is None or (isinstance(short_tpm, float) and math.isnan(short_tpm)) or short_tpm <= 1:
            continue
        if long_tpm is None or (isinstance(long_tpm, float) and math.isnan(long_tpm)) or expected <= 0:
            continue
        n_short_expressed += 1
        observed = float(long_tpm) / float(short_tpm)
        ratios.append(observed / float(expected))

    signals["n_degradation_pairs"] = n_short_expressed

    if not ratios:
        return "unknown", 0.0, "none", None

    import numpy as np
    index = float(np.median(ratios))
    signals["degradation_index"] = round(index, 3)

    if index < _THRESHOLDS["degradation_pair_severe"]:
        return "ffpe", 0.85, "severe", index
    if index < _THRESHOLDS["degradation_pair_moderate"]:
        return "ffpe", 0.7, "moderate", index
    if index < 0.55:
        return "degraded", 0.6, "mild", index
    # Upper bound (bug 2026-04-14): index >> 1.0 indicates systematic
    # over-representation of long transcripts — exon-capture enrichment,
    # NOT fresh RNA. The length-pair signal can't report preservation
    # in that regime because the reference calibration (fresh tissue,
    # uncapture-enriched) doesn't apply. Return ``unknown`` so
    # downstream consumers don't treat a capture-biased sample as
    # confidently fresh.
    if index > _THRESHOLDS["degradation_pair_biased_upper"]:
        return "unknown", 0.0, "none", index
    return "fresh_frozen", 0.7, "none", index


def _build_tpm_by_symbol(df_gene_expr):
    """Return {symbol: max_TPM}, working whether the input frame has
    a symbol column directly or only a gene_id column.

    Prefers a direct symbol column (``gene_symbol``/``symbol``/``Symbol``)
    so that synthetic test frames with just ``(gene_symbol, TPM)`` work
    without requiring a gene-ID column. Falls back to
    ``tumor_purity._build_sample_tpm_by_symbol`` (which maps gene IDs
    through the HPA/pan-cancer reference) for frames without a symbol
    column.
    """
    sym_col = next(
        (c for c in ("gene_symbol", "Symbol", "symbol", "GeneName") if c in df_gene_expr.columns),
        None,
    )
    tpm_col = "TPM" if "TPM" in df_gene_expr.columns else next(
        (c for c in df_gene_expr.columns if c.lower() == "tpm"), None
    )
    if sym_col is not None and tpm_col is not None:
        out = {}
        for _, row in df_gene_expr.iterrows():
            sym = row[sym_col]
            if not isinstance(sym, str) or not sym:
                continue
            try:
                tpm = float(row[tpm_col])
            except (TypeError, ValueError):
                continue
            if math.isnan(tpm):
                continue
            if sym not in out or tpm > out[sym]:
                out[sym] = tpm
        return out
    # Fall back to the gene-ID path.
    from .tumor_purity import _build_sample_tpm_by_symbol
    return _build_sample_tpm_by_symbol(df_gene_expr)


def _summarise_expression_distribution(tpm_by_symbol, signals):
    """Compute shape of the full expression distribution.

    The whole range of TPM values carries library-prep and depth
    information that is independent of the histone / MT / length-pair
    probes above: detection breadth (how many genes are seen at all),
    dynamic range, tail heaviness, and median expressed level.

    Adds fields to ``signals``:

    - ``genes_detected_above_0p5_tpm`` / ``above_1_tpm`` / ``above_10_tpm``
    - ``log2_tpm_median`` (over expressed genes only) and IQR
    - ``high_expression_tail_share`` — fraction of total TPM captured by
      the top 1% of genes (platforms with aggressive poly-A capture and
      shallow depth concentrate more mass in fewer genes)
    """
    import numpy as np

    values = np.array(
        [float(v) for v in tpm_by_symbol.values() if v > 0],
        dtype=float,
    )
    n_genes = int(values.size)
    signals["n_genes_with_any_expression"] = n_genes
    if n_genes == 0:
        return
    signals["genes_detected_above_0p5_tpm"] = int((values > 0.5).sum())
    signals["genes_detected_above_1_tpm"] = int((values > 1.0).sum())
    signals["genes_detected_above_10_tpm"] = int((values > 10.0).sum())

    expressed = values[values > 1.0]
    if expressed.size:
        log2_expr = np.log2(expressed + 1.0)
        signals["log2_tpm_median"] = round(float(np.median(log2_expr)), 3)
        signals["log2_tpm_iqr"] = round(
            float(np.subtract(*np.percentile(log2_expr, [75, 25]))), 3
        )
        signals["log2_tpm_p95"] = round(float(np.percentile(log2_expr, 95)), 3)

    total = float(values.sum())
    if total > 0:
        sorted_desc = np.sort(values)[::-1]
        top_n = max(1, int(round(0.01 * n_genes)))
        signals["top_1pct_share_of_total_tpm"] = round(
            float(sorted_desc[:top_n].sum()) / total, 4
        )
        # Panel-vs-whole-transcriptome heuristic (#68): whole-
        # transcriptome samples carry ~10–15% of total TPM in the top
        # 50 genes; targeted panels concentrate much more sharply.
        signals["top_50_share_of_total_tpm"] = round(
            float(sorted_desc[:50].sum()) / total, 4
        ) if n_genes >= 50 else None
        signals["top_2000_share_of_total_tpm"] = round(
            float(sorted_desc[:2000].sum()) / total, 4
        ) if n_genes >= 2000 else None
        # Likely a targeted panel when BOTH signals fire (few detected
        # genes AND most TPM concentrated in a narrow top). Using AND
        # avoids flagging normal-tissue nTPM references (which are
        # strongly concentrated at the top but still cover >10k genes)
        # while still catching sparse real panels (few genes, heavy
        # top-2000 share).
        #
        # When the sample has fewer than 2000 detected genes total,
        # top-2000 share is trivially 1.0 — treat it that way rather
        # than ``None`` so the heuristic fires correctly on very small
        # panels.
        n_det_1 = signals.get("genes_detected_above_1_tpm", 0) or 0
        top_2000 = signals.get("top_2000_share_of_total_tpm")
        effective_top_2000 = top_2000 if top_2000 is not None else 1.0
        signals["likely_targeted_panel"] = bool(
            n_det_1 < 4000 and effective_top_2000 >= 0.95
        )


def infer_sample_context(df_gene_expr) -> SampleContext:
    """Infer a ``SampleContext`` from a TPM expression table.

    This is the FIRST stage of the unified attribution flow — it runs
    before cancer-type inference, decomposition, and any downstream
    analysis that might be biased by FFPE / library-prep artifacts.

    Parameters
    ----------
    df_gene_expr : pd.DataFrame
        Expression frame with at minimum a ``gene_symbol`` (or
        ``Symbol``/``symbol``) column and a ``TPM`` column. Accepts
        the full pirlygenes expression format as well — any frame that
        works with the existing purity pipeline works here.

    Returns
    -------
    SampleContext
        ``library_prep``, ``preservation``, ``degradation_severity``,
        ``missing_mt``, diagnostic ``signals``, and human-readable
        ``flags``.
    """
    try:
        tpm_by_symbol = _build_tpm_by_symbol(df_gene_expr)
    except (KeyError, ValueError, TypeError):
        # Malformed / test-harness frames without gene-id or symbol
        # columns cannot yield a context. Return a conservative unknown
        # context so downstream code still has a valid object to read.
        return SampleContext(
            library_prep="unknown",
            preservation="unknown",
            degradation_severity="none",
            missing_mt=True,
            flags=["Sample context inference skipped — input frame has no gene column"],
        )
    # Drop NaN
    tpm_by_symbol = {
        s: v for s, v in tpm_by_symbol.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }

    signals = {}

    _summarise_expression_distribution(tpm_by_symbol, signals)
    library_prep, lp_conf = _infer_library_prep(tpm_by_symbol, signals)
    preservation, pr_conf, severity, index = _infer_preservation_and_degradation(
        tpm_by_symbol, signals
    )

    # Missing-MT detection (#61). We check the count of MT symbols that
    # map to any TPM > 0 in the sample. ``missing_mt`` is a distinct
    # signal from preservation — it's about the quant table, not the RNA.
    mt_found = sum(
        1 for s in (_MT_RRNA_SYMBOLS | _MT_MRNA_SYMBOLS)
        if s in tpm_by_symbol and tpm_by_symbol[s] > 0
    )
    signals["mt_genes_detected"] = mt_found
    signals["mt_genes_total"] = len(_MT_RRNA_SYMBOLS) + len(_MT_MRNA_SYMBOLS)
    missing_mt = mt_found <= 1

    flags = []
    if missing_mt:
        flags.append(
            "Mitochondrial genes missing from quant table — degradation "
            "signal from MT fraction is unreliable; FFPE detection falls "
            "back to gene-length pair index only"
        )
    if library_prep == "unknown":
        flags.append("Library prep inconclusive — histone + MT-rRNA signals did not match any profile")
    else:
        flags.append(
            f"Library prep: {library_prep.replace('_', ' ')} "
            f"(confidence {lp_conf:.0%})"
        )
    if preservation == "ffpe":
        flags.append(
            f"Preservation: FFPE / heavily degraded "
            f"(length-pair index {index:.2f})"
        )
    elif preservation == "degraded":
        flags.append(f"Preservation: partial degradation (length-pair index {index:.2f})")
    elif preservation == "fresh_frozen":
        flags.append(f"Preservation: fresh / frozen (length-pair index {index:.2f})")
    elif preservation == "unknown" and index is not None and index > 1.0:
        # Capture-enriched libraries (exome / hybrid) inflate the long/
        # short ratio because long genes have more probes. Flag that
        # the length-pair test can't report preservation in this regime.
        flags.append(
            f"Preservation unknown — length-pair index {index:.2f} is "
            f"above fresh-reference range (>{_THRESHOLDS['degradation_pair_biased_upper']:.1f}), "
            "consistent with exon-capture enrichment (long transcripts "
            "over-represented). Orthogonal signals (read-length, 3′ bias) "
            "needed to assess FFPE status."
        )

    return SampleContext(
        library_prep=library_prep,
        library_prep_confidence=lp_conf,
        preservation=preservation,
        preservation_confidence=pr_conf,
        degradation_severity=severity,
        degradation_index=index,
        missing_mt=missing_mt,
        signals=signals,
        flags=flags,
    )


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_sample_context(sample_context: SampleContext, save_to_filename: str,
                        save_dpi: int = 150) -> Optional[str]:
    """Standalone PNG summarising ``sample_context`` as the first panel
    a user sees in an analyze report.

    Two-row layout (single PNG — plot-crowding preference):

    - **Top**: text block with library prep, preservation, degradation
      severity, missing-MT flag, and confidence.
    - **Bottom**: three horizontal bars showing the diagnostic fractions
      (histone mRNA fraction of total; MT rRNA fraction of MT; overall
      MT fraction of total) with the thresholds used for the call.
    """
    import matplotlib.pyplot as plt

    fig, (ax_text, ax_bars) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 5.5),
        gridspec_kw={"height_ratios": [1, 1.5]},
    )
    ax_text.axis("off")

    prep_label = {
        "poly_a": "Poly-A capture",
        "ribo_depleted": "Ribosomal depletion",
        "total_rna": "Total RNA",
        "exome_capture": "Exome capture",
        "unknown": "Unknown",
    }.get(sample_context.library_prep, sample_context.library_prep)

    pres_label = {
        "fresh_frozen": "Fresh / frozen",
        "ffpe": "FFPE",
        "degraded": "Partial degradation",
        "unknown": "Unknown",
    }.get(sample_context.preservation, sample_context.preservation)

    severity_color = {
        "none": "#2e8b57",
        "mild": "#daa520",
        "moderate": "#d2691e",
        "severe": "#b22222",
    }.get(sample_context.degradation_severity, "#666666")

    header_lines = [
        f"Library prep:  {prep_label}   "
        f"(confidence {sample_context.library_prep_confidence:.0%})",
        f"Preservation:  {pres_label}",
        (
            f"Degradation:   {sample_context.degradation_severity}"
            + (f"  (length-pair index {sample_context.degradation_index:.2f})"
               if sample_context.degradation_index is not None else "")
        ),
    ]
    if sample_context.missing_mt:
        header_lines.append("⚠ MT genes missing from quant table")

    ax_text.text(
        0.02, 0.95, "Sample context",
        fontsize=14, fontweight="bold", va="top",
    )
    for i, line in enumerate(header_lines):
        ax_text.text(
            0.02, 0.70 - 0.22 * i, line,
            fontsize=11, va="top", family="monospace",
            color=severity_color if i == 2 else "black",
        )

    # Bottom panel: diagnostic bars
    signals = sample_context.signals
    hist_frac = signals.get("histone_fraction", 0.0) or 0.0
    mt_frac = signals.get("mt_fraction", 0.0) or 0.0
    mt_rrna_frac = signals.get("mt_rrna_fraction_of_mt") or 0.0

    bars = [
        ("Histone fraction\n(replication-dependent mRNAs)",
         hist_frac, _THRESHOLDS["histone_fraction_total_floor"],
         "Total/ribo-dep floor"),
        ("MT fraction\n(of total sample TPM)",
         mt_frac, _THRESHOLDS["mt_fraction_suspicious_floor"],
         "Suspicious-floor"),
        ("MT-rRNA / MT-total",
         mt_rrna_frac, _THRESHOLDS["mt_rrna_fraction_of_mt_total_floor"],
         "Total-RNA floor"),
    ]
    y = [i for i in range(len(bars))]
    values = [b[1] for b in bars]
    labels = [b[0] for b in bars]
    thresholds = [(b[2], b[3]) for b in bars]

    ax_bars.barh(y, values, color="#4682b4", alpha=0.8)
    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels(labels, fontsize=9)
    ax_bars.invert_yaxis()
    ax_bars.set_xlim(0, max(max(values) * 1.2, 1.0))
    ax_bars.set_xlabel("Fraction", fontsize=10)
    for yi, (thr, thr_name) in zip(y, thresholds):
        ax_bars.axvline(thr, color="#888888", linestyle="--", linewidth=0.8)
        ax_bars.text(
            thr, yi - 0.4, thr_name,
            fontsize=7, color="#666666", ha="left",
        )
    for yi, val in zip(y, values):
        ax_bars.text(val, yi, f"  {val:.3f}", va="center", fontsize=9)

    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.set_title("Library-prep diagnostic signals", fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename


def plot_degradation_index(
    df_gene_expr,
    sample_context: SampleContext,
    save_to_filename: str,
    save_dpi: int = 150,
) -> Optional[str]:
    """Scatter of expected vs observed long/short pair ratios (#27).

    One point per gene pair from ``data/degradation-gene-pairs.csv``:
    x = expected (fresh-tissue calibrated) ratio, y = observed in this
    sample. Diagonal = no degradation. Points below the diagonal flag
    preferential loss of long transcripts. The plot annotates the
    median observed/expected ratio and the severity call from the
    sample context, so users can see whether the call is driven by a
    systematic shift across pairs or a few outliers.

    Returns the filename on success, ``None`` when no pair has the
    short-gene expressed (``s_tpm > 1``) so the plot would be empty.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from .gene_sets_cancer import degradation_gene_pairs

    tpm_by_symbol = _build_tpm_by_symbol(df_gene_expr)

    expected_vals = []
    observed_vals = []
    labels = []
    for short_sym, long_sym, expected in degradation_gene_pairs():
        s = tpm_by_symbol.get(short_sym)
        long_tpm = tpm_by_symbol.get(long_sym)
        if s is None or not (s > 1):
            continue
        if long_tpm is None or expected <= 0:
            continue
        expected_vals.append(float(expected))
        observed_vals.append(float(long_tpm) / float(s))
        labels.append(f"{short_sym}/{long_sym}")

    if not expected_vals:
        return None

    expected_arr = np.array(expected_vals)
    observed_arr = np.array(observed_vals)
    deviation = observed_arr / np.maximum(expected_arr, 1e-6)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Diagonal guide.
    diag_lo = 0.0
    diag_hi = max(expected_arr.max(), observed_arr.max()) * 1.1
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi],
            linestyle="--", color="#888888", linewidth=0.8,
            label="expected = observed (no degradation)")

    sc = ax.scatter(
        expected_arr, observed_arr,
        c=np.log2(np.maximum(deviation, 1e-3)),
        cmap="RdYlGn", edgecolors="black", linewidth=0.3, s=60,
    )
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("log2(observed / expected)", fontsize=9)

    ax.set_xlabel("Expected long/short ratio (fresh-tissue calibration)", fontsize=10)
    ax.set_ylabel("Observed long/short ratio (this sample)", fontsize=10)
    ax.set_xlim(diag_lo, diag_hi)
    ax.set_ylim(diag_lo, diag_hi)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title_parts = [f"Degradation index — {sample_context.degradation_severity}"]
    if sample_context.degradation_index is not None:
        title_parts.append(f"median index {sample_context.degradation_index:.2f}")
    title_parts.append(f"{len(expected_vals)} pairs")
    ax.set_title(" · ".join(title_parts), fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_to_filename, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    return save_to_filename
