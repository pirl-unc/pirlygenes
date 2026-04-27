"""Data-backed rare cancer hypotheses from RNA expression.

These rules are intentionally conservative. They promote a report scope
for rare, registry-only cancers that lack a TCGA expression cohort, while
keeping the TCGA classifier result as expression context and requiring
orthogonal diagnostic confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _split_semicolon(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if result != result:
        return float(default)
    return result


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_bool(value: object, default: bool = False) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text or text == "nan":
        return bool(default)
    return text in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class RareCancerRnaInference:
    """One report-scope hypothesis promoted from RNA surrogate evidence."""

    cancer_type: str
    rule_id: str
    surrogate: str
    surrogate_tpm: float
    threshold_tpm: float
    top_reference_cancer_type: str
    confidence: str
    support_genes: tuple[str, ...]
    basis: str
    confirmatory_tests: str
    caveat: str
    source: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "cancer_type": self.cancer_type,
            "rule_id": self.rule_id,
            "surrogate": self.surrogate,
            "surrogate_tpm": round(self.surrogate_tpm, 3),
            "threshold_tpm": self.threshold_tpm,
            "top_reference_cancer_type": self.top_reference_cancer_type,
            "confidence": self.confidence,
            "support_genes": list(self.support_genes),
            "basis": self.basis,
            "confirmatory_tests": self.confirmatory_tests,
            "caveat": self.caveat,
            "source": self.source,
        }


def rare_cancer_rna_surrogate_rules_df():
    """Return the curated rare-cancer RNA-surrogate rule table."""
    from .load_dataset import get_data

    return get_data("rare-cancer-rna-surrogates")


def rare_cancer_fusion_rules_df():
    """Return the curated direct-fusion rare-cancer rule table."""
    from .load_dataset import get_data

    return get_data("rare-cancer-fusion-rules")


def _record_value(record, key: str, default=None):
    if hasattr(record, "get"):
        return record.get(key, default)
    return getattr(record, key, default)


def _rule_gene_set(value: object) -> set[str]:
    genes = set(_split_semicolon(value))
    return {gene.upper() for gene in genes if gene}


def _rule_gene_display(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "?"
    return "/".join(_split_semicolon(text)) if ";" in text else text


def _fusion_record_key(record) -> tuple[str, str, int | None]:
    gene_a = str(_record_value(record, "gene_a", "") or "").strip().upper()
    gene_b = str(_record_value(record, "gene_b", "") or "").strip().upper()
    pair_key = "::".join(sorted((gene_a, gene_b)))
    source = str(_record_value(record, "source_path", "") or "")
    row_index = _record_value(record, "row_index", None)
    return pair_key, source, row_index


def _fusion_rule_match_detail(rule, record) -> dict[str, str] | None:
    gene_a = str(_record_value(record, "gene_a", "") or "").strip().upper()
    gene_b = str(_record_value(record, "gene_b", "") or "").strip().upper()
    if not gene_a or not gene_b:
        return None
    rule_a = _rule_gene_set(rule.get("gene_a"))
    rule_b = _rule_gene_set(rule.get("gene_b"))
    matching = str(rule.get("matching") or "oriented_or_unoriented").strip().lower()

    def _in(gene: str, genes: set[str]) -> bool:
        return "*" in genes or gene in genes

    direct = _in(gene_a, rule_a) and _in(gene_b, rule_b)
    reverse = _in(gene_b, rule_a) and _in(gene_a, rule_b)
    strict = matching in {"strict", "strict_5to3", "ordered", "direct", "5to3"}
    record_orientation = str(_record_value(record, "orientation", "") or "").lower()
    if direct:
        return {
            "matched_orientation": "as_reported",
            "orientation_note": "reported pair matches expected 5-prime/3-prime rule orientation",
        }
    if reverse and not strict and record_orientation != "5prime_3prime":
        return {
            "matched_orientation": "reverse_of_expected",
            "orientation_note": (
                "reported pair is reverse of the curated 5-prime/3-prime rule; "
                "confirm caller orientation"
            ),
        }
    return None


def _fusion_rule_specificity(rule) -> int:
    score = 0
    for column in ("gene_a", "gene_b"):
        genes = _rule_gene_set(rule.get(column))
        if genes and "*" not in genes:
            score += 1
    return score


def match_rare_cancer_fusion_rules(fusion_records) -> list[dict[str, Any]]:
    """Return curated rare-cancer fusion findings for supplied calls.

    The rule table is oriented as ``gene_a`` = expected 5-prime partner and
    ``gene_b`` = expected 3-prime partner. Matching is deliberately loose by
    default because caller outputs and manually supplied lists often omit or
    invert orientation. Findings retain both the reported pair and the expected
    5-prime/3-prime rule pair.
    """
    records = list(fusion_records or [])
    if not records:
        return []
    rules = rare_cancer_fusion_rules_df().fillna("")
    hits: list[tuple[tuple[float, int, float], dict[str, Any]]] = []
    confidence_rank = {"high": 2, "moderate": 1, "low": 0}
    for _, rule in rules.iterrows():
        min_support = _safe_float(rule.get("min_total_support"), 0.0)
        for record in records:
            match_detail = _fusion_rule_match_detail(rule, record)
            if not match_detail:
                continue
            support_total = _record_value(record, "support_total", None)
            if support_total is not None and _safe_float(support_total, 0.0) < min_support:
                continue
            confidence = str(rule.get("confidence") or "high").strip()
            promote = _safe_bool(rule.get("promote_report_scope"), default=True)
            specificity = _fusion_rule_specificity(rule)
            finding = {
                "cancer_type": str(rule.get("cancer_code") or "").strip(),
                "label": str(rule.get("label") or "").strip(),
                "rule_id": str(rule.get("rule_id") or "").strip(),
                "promote_report_scope": promote,
                "specificity": specificity,
                "expected_pair": (
                    f"{_rule_gene_display(rule.get('gene_a'))}"
                    f"--{_rule_gene_display(rule.get('gene_b'))}"
                ),
                "matched_orientation": match_detail["matched_orientation"],
                "orientation_note": match_detail["orientation_note"],
                "fusion": {
                    "gene_a": str(_record_value(record, "gene_a", "") or ""),
                    "gene_b": str(_record_value(record, "gene_b", "") or ""),
                    "pair": f"{_record_value(record, 'gene_a', '')}--{_record_value(record, 'gene_b', '')}",
                    "support_total": support_total,
                    "effect": str(_record_value(record, "effect", "") or ""),
                    "frame": str(_record_value(record, "frame", "") or ""),
                    "caller": str(_record_value(record, "caller", "") or ""),
                    "confidence": str(_record_value(record, "confidence", "") or ""),
                    "reportable": str(_record_value(record, "reportable", "") or ""),
                    "orientation": str(_record_value(record, "orientation", "") or ""),
                },
                "confidence": confidence,
                "basis": str(rule.get("basis") or "").strip(),
                "confirmatory_tests": str(rule.get("confirmatory_tests") or "").strip(),
                "caveat": str(rule.get("caveat") or "").strip(),
                "source": str(rule.get("source") or "").strip(),
            }
            support_score = _safe_float(support_total, min_support or 1.0)
            hits.append(
                (
                    (
                        1.0 if promote else 0.0,
                        confidence_rank.get(confidence.lower(), 0),
                        specificity,
                        support_score,
                    ),
                    finding,
                )
            )
    if not hits:
        return []
    hits.sort(key=lambda item: item[0], reverse=True)
    findings = [item[1] for item in hits]

    max_specificity_by_record: dict[tuple[str, str, int | None], int] = {}
    for finding in findings:
        key = _fusion_record_key(finding["fusion"])
        max_specificity_by_record[key] = max(
            max_specificity_by_record.get(key, 0),
            _safe_int(finding.get("specificity"), 0),
        )
    filtered: list[dict[str, Any]] = []
    for finding in findings:
        key = _fusion_record_key(finding["fusion"])
        if _safe_int(finding.get("specificity"), 0) < max_specificity_by_record.get(key, 0):
            continue
        filtered.append(finding)
    return filtered


def infer_rare_cancer_report_scope_from_fusions(fusion_records, analysis=None):
    """Return a rare-cancer report-scope hypothesis from direct fusions."""
    findings = match_rare_cancer_fusion_rules(fusion_records)
    if not findings:
        return None
    candidate_trace = (analysis or {}).get("candidate_trace") or []
    top_reference = (
        str(candidate_trace[0].get("code") or "").strip() if candidate_trace else ""
    )
    for finding in findings:
        if not finding.get("promote_report_scope"):
            continue
        if not finding.get("cancer_type"):
            continue
        result = dict(finding)
        result["top_reference_cancer_type"] = top_reference
        return result
    return None


def infer_rare_cancer_report_scope_from_rna(df_expr, analysis):
    """Return a rare-cancer report-scope hypothesis, or ``None``.

    The classifier still supplies the reference expression context. This
    helper only promotes rare entities that have a curated, high-specificity
    RNA surrogate rule and pass broad-context gates.
    """
    try:
        from .common import build_sample_tpm_by_symbol

        sample_tpm = build_sample_tpm_by_symbol(df_expr)
    except Exception:
        return None

    candidate_trace = analysis.get("candidate_trace") or []
    top_reference = (
        str(candidate_trace[0].get("code") or "").strip() if candidate_trace else ""
    )
    top_codes = [
        str(row.get("code") or "").strip()
        for row in candidate_trace
        if str(row.get("code") or "").strip()
    ]

    rules = rare_cancer_rna_surrogate_rules_df().fillna("")
    hits: list[tuple[tuple[float, int, float], RareCancerRnaInference]] = []
    confidence_rank = {"high": 2, "moderate": 1, "low": 0}
    for _, rule in rules.iterrows():
        primary_gene = str(rule.get("primary_gene") or "").strip()
        if not primary_gene:
            continue
        primary_tpm = _safe_float(sample_tpm.get(primary_gene), 0.0)
        min_tpm = _safe_float(rule.get("min_tpm"), 0.0)
        if primary_tpm < min_tpm:
            continue

        context_codes = set(_split_semicolon(rule.get("context_codes")))
        context_top_k = _safe_int(rule.get("context_top_k"), 5)
        context_slice = set(top_codes[:context_top_k])
        if context_codes and not (context_codes & context_slice):
            continue

        excluded_context_codes = set(_split_semicolon(rule.get("excluded_context_codes")))
        if excluded_context_codes and (excluded_context_codes & context_slice):
            continue

        support_min = _safe_float(rule.get("support_min_tpm"), 0.0)
        support_genes = []
        for gene in _split_semicolon(rule.get("required_support_genes")):
            if _safe_float(sample_tpm.get(gene), 0.0) >= support_min:
                support_genes.append(gene)
        min_support = _safe_int(rule.get("min_support_genes"), 0)
        if len(support_genes) < min_support:
            continue

        inference = RareCancerRnaInference(
            cancer_type=str(rule.get("cancer_code") or "").strip(),
            rule_id=str(rule.get("rule_id") or "").strip(),
            surrogate=primary_gene,
            surrogate_tpm=primary_tpm,
            threshold_tpm=min_tpm,
            top_reference_cancer_type=top_reference,
            confidence=str(rule.get("confidence") or "moderate").strip(),
            support_genes=tuple(support_genes),
            basis=str(rule.get("basis") or "").strip(),
            confirmatory_tests=str(rule.get("confirmatory_tests") or "").strip(),
            caveat=str(rule.get("caveat") or "").strip(),
            source=str(rule.get("source") or "").strip(),
        )
        sort_key = (
            confidence_rank.get(inference.confidence.lower(), 0),
            len(support_genes),
            primary_tpm / min_tpm if min_tpm > 0 else primary_tpm,
        )
        hits.append((sort_key, inference))

    if not hits:
        return None
    hits.sort(key=lambda item: item[0], reverse=True)
    return hits[0][1].public_dict()
