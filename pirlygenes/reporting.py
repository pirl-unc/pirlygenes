"""Shared report-facing helpers for evidence and attribution reliability.

These helpers are intentionally narrow: they do not change the raw
attribution math, only how downstream markdown decides whether a row is
safe to summarize as credible tumor-linked signal. The goal is to keep
the headline markdown blocks aligned with the caveats already present in
the tables and TSVs.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re


def _truthy(value) -> bool:
    if value is None:
        return False
    try:
        if value != value:
            return False
    except Exception:
        pass
    try:
        return bool(value)
    except Exception:
        return False


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _safe_float(value, default=0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if result != result:
        return float(default)
    return result


def _safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


@lru_cache(maxsize=1)
def _cancer_registry_display_names():
    try:
        from .gene_sets_cancer import cancer_type_registry

        df = cancer_type_registry()
    except Exception:
        return {}

    mapping = {}
    for _, row in df.iterrows():
        code = _clean_text(row.get("code"))
        if not code:
            continue
        name = _clean_text(row.get("name"))
        subtype_key = _clean_text(row.get("subtype_key"))
        if name:
            mapping[code] = name.split("(")[0].strip()
        elif subtype_key:
            mapping[code] = subtype_key.replace("_", " ").strip()
    return mapping


def cancer_code_display_name(code, fallback=None):
    text = _clean_text(code)
    if not text:
        return _clean_text(fallback) or "this cancer type"
    reg_name = _cancer_registry_display_names().get(text)
    if reg_name:
        return reg_name
    fallback_text = _clean_text(fallback)
    if fallback_text:
        return fallback_text
    return text.replace("_", " ").strip()


def subtype_curation_scope_note(
    panel_code,
    *,
    panel_subtype=None,
    base_code=None,
    base_name=None,
    noun="therapy evidence",
):
    panel_name = cancer_code_display_name(panel_code).lower()
    base_label = cancer_code_display_name(base_code, fallback=base_name).lower()
    subtype_label = _clean_text(panel_subtype).replace("_", " ").lower()
    if subtype_label:
        focus = f"{subtype_label} {panel_name}".strip()
    else:
        focus = panel_name
    if not base_label or focus == base_label:
        return f"Using {focus}-specific {noun}."
    return f"Using {focus}-specific {noun} rather than the broader {base_label} list."


def tumor_attribution_context(row):
    """Summarize how securely a row stays tumor-attributed."""
    observed = _safe_float(row.get("observed_tpm"), 0.0)
    mid_tpm = _safe_float(row.get("attr_tumor_tpm"), 0.0)
    low_tpm = _safe_float(row.get("attr_tumor_tpm_low"), mid_tpm)
    high_tpm = _safe_float(row.get("attr_tumor_tpm_high"), mid_tpm)
    mid_frac = _safe_float(row.get("attr_tumor_fraction"), 0.0)
    low_frac = _safe_float(row.get("attr_tumor_fraction_low"), mid_frac)
    high_frac = _safe_float(row.get("attr_tumor_fraction_high"), mid_frac)
    support_fraction = _safe_float(
        row.get("attr_support_fraction"),
        1.0 if (mid_tpm >= 1.0 and mid_frac >= 0.30) else 0.0,
    )
    low_tpm, mid_tpm, high_tpm = sorted([max(0.0, low_tpm), max(0.0, mid_tpm), max(0.0, high_tpm)])
    low_frac, mid_frac, high_frac = sorted(
        [
            max(0.0, min(1.0, low_frac)),
            max(0.0, min(1.0, mid_frac)),
            max(0.0, min(1.0, high_frac)),
        ]
    )
    support_fraction = max(0.0, min(1.0, support_fraction))

    notes = []
    if _truthy(row.get("matched_normal_over_predicted")):
        notes.append("matched-normal reference overshoots the sample")
    if _truthy(row.get("low_purity_cap_applied")):
        notes.append("low-purity cap is active")
    if _truthy(row.get("tme_explainable")) and support_fraction < 1.0:
        notes.append("healthy-tissue-only explanations remain plausible")
    if _truthy(row.get("tme_dominant")):
        notes.append("most fitted signal remains non-tumor")

    if high_tpm < 1.0 or high_frac < 0.30 or _truthy(row.get("tme_dominant")):
        tier = "background_dominant"
        label = "background-dominant"
        summary = "non-tumor compartments remain the simpler explanation"
    elif (
        low_tpm >= 1.0
        and low_frac >= 0.30
        and support_fraction >= 0.67
        and not _truthy(row.get("matched_normal_over_predicted"))
    ):
        tier = "tumor_supported"
        label = "tumor-supported"
        summary = "tumor-attributed signal stays material across the range"
    else:
        tier = "mixed_source"
        label = "mixed-source"
        summary = "both tumor and benign/background sources remain plausible"

    if observed > 0:
        band = (
            f"{mid_tpm:.0f} TPM "
            f"(model interval {low_tpm:.0f}-{high_tpm:.0f}; "
            f"{mid_frac:.0%} tumor, {low_frac:.0%}-{high_frac:.0%} interval)"
        )
    else:
        band = f"{mid_tpm:.0f} TPM"

    return {
        "tier": tier,
        "label": label,
        "summary": summary,
        "notes": notes,
        "band": band,
        "observed_tpm": observed,
        "attr_tumor_tpm": mid_tpm,
        "attr_tumor_tpm_low": low_tpm,
        "attr_tumor_tpm_high": high_tpm,
        "attr_tumor_fraction": mid_frac,
        "attr_tumor_fraction_low": low_frac,
        "attr_tumor_fraction_high": high_frac,
        "attr_support_fraction": support_fraction,
    }


def tpm_semantics_note() -> str:
    """One reader-facing explanation of bulk vs modeled tumor-core TPM."""
    return (
        "**TPM semantics:** Bulk TPM is the measured RNA abundance in the mixed "
        "specimen. Tumor-core TPM is a model-derived estimate after subtracting "
        "matched-normal and TME-attributable signal. Use tumor-core TPM for "
        "tumor-cell target prioritization only when attribution is tumor-supported "
        "or mixed-source; for immune/stromal markers, agent-only rows, and "
        "expression-independent indications, bulk RNA is contextual and the "
        "clinical biomarker must come from the indicated assay."
    )


_IMMUNE_CHECKPOINT_AGENTS = re.compile(
    r"\b("
    r"pembrolizumab|nivolumab|ipilimumab|dostarlimab|relatlimab|"
    r"atezolizumab|avelumab|durvalumab"
    r")\b",
    re.IGNORECASE,
)
_TARGET_EXPRESSION_INDICATION = re.compile(
    r"\b(pd[- ]?l1|pd[- ]?1|cps|tps|ihc|overexpress|expression|expressing)\b",
    re.IGNORECASE,
)
_MUTATION_INDICATION = re.compile(
    r"\b("
    r"mutation|mutant|v600|exon\s*\d+|fusion|rearrangement|"
    r"amplification|amplified|amp\b|her2\+|brca[- ]?(mut|mutation)"
    r")\b",
    re.IGNORECASE,
)
_MSI_HIGH_INDICATION = re.compile(
    r"\b(msi[- ]?h|msi[- ]?high|dmmr|deficient\s+mmr|mismatch\s+repair\s+deficien)",
    re.IGNORECASE,
)
_MMR_PROFICIENT = re.compile(
    r"\b(pmmr|mmr[- ]?proficient|mismatch\s+repair\s+proficient|mss|msi[- ]?stable)\b",
    re.IGNORECASE,
)


def indication_biomarker(target_row) -> str:
    """Return the typed biomarker that gates a curated therapy row.

    The CSV may eventually carry an explicit ``indication_biomarker`` column.
    Until then this central inference prevents report renderers from treating
    histology/MSI/TMB/mutation-gated therapies as if target RNA expression were
    the approval criterion.
    """
    explicit = _clean_text(
        target_row.get("indication_biomarker")
        if hasattr(target_row, "get") else None
    ).lower()
    if explicit:
        return explicit

    text = " ".join(
        _clean_text(target_row.get(key))
        for key in ("indication", "rationale", "agent", "agent_class")
        if hasattr(target_row, "get")
    )
    low = text.lower()
    if _MSI_HIGH_INDICATION.search(low) and not _MMR_PROFICIENT.search(low):
        return "msi_high"
    if "tmb" in low or "tumor mutational burden" in low:
        return "tmb_high"
    if _MUTATION_INDICATION.search(text):
        return "mutation"

    agent = _clean_text(target_row.get("agent") if hasattr(target_row, "get") else "")
    if _IMMUNE_CHECKPOINT_AGENTS.search(agent) and not _TARGET_EXPRESSION_INDICATION.search(text):
        return "histology_only"

    return "target_expression"


def indication_biomarker_label(target_row) -> str:
    biomarker = indication_biomarker(target_row)
    return {
        "target_expression": "target expression",
        "msi_high": "MSI-H / dMMR",
        "tmb_high": "TMB-high",
        "mutation": "mutation / fusion / amplification",
        "histology_only": "histology indication",
    }.get(biomarker, biomarker.replace("_", " "))


def expression_independent_indication(target_row) -> bool:
    return indication_biomarker(target_row) != "target_expression"


def expression_independent_interpretation(target_row) -> str:
    label = indication_biomarker_label(target_row)
    if indication_biomarker(target_row) == "histology_only":
        return "expression-independent indication — confirm clinical eligibility"
    return f"expression-independent indication — confirm {label} status"


def tumor_attribution_band_text(row):
    return tumor_attribution_context(row)["band"]


def tumor_band_available(row):
    """Whether a row carries a usable tumor-attributed range."""
    for key in (
        "attr_tumor_tpm",
        "attr_tumor_tpm_low",
        "attr_tumor_tpm_high",
        "attr_tumor_fraction",
        "attr_tumor_fraction_high",
    ):
        value = row.get(key)
        if value is None:
            continue
        text = _clean_text(value)
        if text:
            return True
    return False


def tumor_band_cell(row):
    """Compact ``mid (low-high)`` cell for markdown tables."""
    if not tumor_band_available(row):
        return "—"
    ctx = tumor_attribution_context(row)
    return (
        f"{ctx['attr_tumor_tpm']:.0f} "
        f"({ctx['attr_tumor_tpm_low']:.0f}-{ctx['attr_tumor_tpm_high']:.0f})"
    )


def target_reliability_reasons(row, *, category=None):
    """Return ordered reader-facing caveats for a target/expression row."""
    source = tumor_attribution_context(row)
    reasons = []
    if source["tier"] == "background_dominant":
        reasons.append("background-dominant")
    elif source["tier"] == "mixed_source":
        reasons.append("mixed-source")
    if _truthy(row.get("matched_normal_over_predicted")):
        reasons.append("matched-normal over-predicted")
    if _truthy(row.get("smooth_muscle_stromal_leakage")):
        reasons.append("possible smooth-muscle stromal leakage")
    if _truthy(row.get("tme_explainable")) and source["tier"] != "tumor_supported":
        reasons.append("healthy-tissue-explainable")
    if _truthy(row.get("low_purity_cap_applied")):
        reasons.append("low-purity capped")
    return reasons


def target_reliability_status(row, *, category=None, target_row=None):
    """Classify a row as ``supported``, ``provisional``, or ``unsupported``."""
    if target_row is not None and expression_independent_indication(target_row):
        return "provisional"
    source = tumor_attribution_context(row)
    if source["tier"] == "background_dominant":
        return "unsupported"
    if _truthy(row.get("matched_normal_over_predicted")):
        return "provisional"
    if _truthy(row.get("smooth_muscle_stromal_leakage")):
        return "provisional"
    if source["tier"] == "mixed_source":
        return "provisional"
    return "supported"


_ESSENTIAL_TISSUE_COLS = {
    "brain": [
        "nTPM_cerebral_cortex", "nTPM_cerebellum", "nTPM_basal_ganglia",
        "nTPM_hippocampal_formation", "nTPM_amygdala", "nTPM_midbrain",
        "nTPM_hypothalamus", "nTPM_spinal_cord", "nTPM_choroid_plexus",
    ],
    "heart": ["nTPM_heart_muscle"],
    "liver": ["nTPM_liver"],
    "lung": ["nTPM_lung"],
    "kidney": ["nTPM_kidney"],
    "bone_marrow": ["nTPM_bone_marrow"],
    "spleen": ["nTPM_spleen"],
    "pancreas": ["nTPM_pancreas"],
    "colon": ["nTPM_colon", "nTPM_rectum", "nTPM_duodenum", "nTPM_small_intestine"],
    "stomach": ["nTPM_stomach"],
}


@lru_cache(maxsize=1)
def _pan_cancer_normal_expression_index():
    try:
        from .gene_sets_cancer import pan_cancer_expression

        df = pan_cancer_expression().copy()
        if "Ensembl_Gene_ID" not in df.columns:
            return None
        df["Ensembl_Gene_ID"] = df["Ensembl_Gene_ID"].astype(str).str.split(".", n=1).str[0]
        df = df.drop_duplicates(subset="Ensembl_Gene_ID").set_index("Ensembl_Gene_ID")
        return df
    except Exception:
        return None


def _healthy_expression_profile(gene_id):
    gene_id = _clean_text(gene_id).split(".", 1)[0]
    if not gene_id:
        return {}
    ref = _pan_cancer_normal_expression_index()
    if ref is None or gene_id not in ref.index:
        return {}
    row = ref.loc[gene_id]
    profile = {}
    for tissue, columns in _ESSENTIAL_TISSUE_COLS.items():
        values = []
        for column in columns:
            if column not in row.index:
                continue
            try:
                value = float(row[column])
            except Exception:
                continue
            if value == value:
                values.append(value)
        if values:
            profile[tissue] = max(values)
    return profile


def normal_expression_context(row):
    """Return a coarse reader-facing normal-expression context label."""
    category_label = str(row.get("category") or "").upper()
    if _truthy(row.get("is_cta")) or category_label == "CTA":
        return {
            "tier": "cta_restricted",
            "label": "CTA / immune-privileged",
            "summary": "CTA-style restricted normal expression",
            "details": [],
        }

    matched_tissue = _clean_text(row.get("matched_normal_tissue")).replace("_", " ")
    matched_normal_tpm = _safe_float(row.get("matched_normal_tpm"), 0.0)
    tumor_tpm = _safe_float(row.get("attr_tumor_tpm"), 0.0)
    attr_top = _clean_text(row.get("attr_top_compartment")).replace("_", " ")
    details = []
    same_lineage = False
    if matched_tissue:
        same_lineage = (
            attr_top == f"matched normal {matched_tissue}"
            or _truthy(row.get("matched_normal_over_predicted"))
            or matched_normal_tpm >= max(1.0, tumor_tpm * 0.5)
        )

    healthy_profile = _healthy_expression_profile(row.get("gene_id"))
    vital_detail = ""
    if healthy_profile:
        top_tissue, top_value = max(healthy_profile.items(), key=lambda item: item[1])
        if top_value >= 10.0:
            vital_detail = f"{top_tissue.replace('_', ' ')} expression is appreciable"

    if _truthy(row.get("broadly_expressed")) or _safe_int(row.get("n_healthy_tissues_expressed"), 0) >= 15:
        details.append("broader healthy-tissue signal is present outside the likely tissue of origin")
    if vital_detail:
        details.append(vital_detail)

    if same_lineage:
        return {
            "tier": "same_lineage_expected",
            "label": "same-lineage expected",
            "summary": f"benign {matched_tissue} lineage expression is expected",
            "details": details,
        }

    if vital_detail:
        return {
            "tier": "vital_tissue_concern",
            "label": "vital-tissue concern",
            "summary": vital_detail,
            "details": details[1:] if details else [],
        }

    if details:
        return {
            "tier": "broad_healthy_expression",
            "label": "broad healthy expression",
            "summary": details[0],
            "details": details[1:],
        }

    return {
            "tier": "restricted_outside_lineage",
            "label": "restricted outside likely tissue of origin",
            "summary": "limited healthy-tissue signal outside the likely tissue of origin",
        "details": [],
    }


def clinical_maturity_info(target_row, target_panel=None):
    """Summarize maturity from the current row and its curated siblings."""
    phase = _clean_text(target_row.get("phase"))
    phase_label = {
        "approved": "approved",
        "phase_3": "late clinical",
        "phase_2": "mid-clinical",
        "phase_1": "early clinical",
        "preclinical": "preclinical",
    }.get(phase, phase or "curated")
    agent_class = _clean_text(target_row.get("agent_class")).replace("_", " ")
    summary = phase_label
    if agent_class:
        summary += f" {agent_class}"

    if target_panel is None or len(target_panel) == 0:
        return {
            "tier": phase_label,
            "summary": summary,
            "n_agents": 0,
            "n_modalities": 0,
        }

    symbol = _clean_text(target_row.get("symbol"))
    if not symbol or "symbol" not in target_panel.columns:
        return {
            "tier": phase_label,
            "summary": summary,
            "n_agents": 0,
            "n_modalities": 0,
        }

    try:
        sub = target_panel[target_panel["symbol"].astype(str) == symbol]
    except Exception:
        sub = None

    if sub is None or len(sub) <= 1:
        return {
            "tier": phase_label,
            "summary": summary,
            "n_agents": 1 if sub is not None and len(sub) == 1 else 0,
            "n_modalities": 1 if sub is not None and len(sub) == 1 else 0,
        }

    n_agents = (
        sub["agent"].fillna("").astype(str).str.strip()
        .replace("nan", "").loc[lambda s: s.ne("")].nunique()
        if "agent" in sub.columns else 0
    )
    n_modalities = (
        sub["agent_class"].fillna("").astype(str).str.strip()
        .replace("nan", "").loc[lambda s: s.ne("")].nunique()
        if "agent_class" in sub.columns else 0
    )
    extras = []
    if n_agents > 1:
        extras.append(f"{n_agents} curated agents")
    if n_modalities > 1:
        extras.append(f"{n_modalities} modalities")
    if extras:
        summary += f" ({', '.join(extras)})"
    return {
        "tier": phase_label,
        "summary": summary,
        "n_agents": int(n_agents),
        "n_modalities": int(n_modalities),
    }


def clinical_maturity_summary(target_row, target_panel=None):
    return clinical_maturity_info(target_row, target_panel=target_panel)["summary"]


_STANDARD_PATH_TEXT = re.compile(
    r"\b("
    r"standard(?:\s+of\s+care)?|standard\s+backbone|backbone|"
    r"first[- ]line|frontline|newly\s+diagnosed|1l\b|"
    r"adjuvant|neoadjuvant|maintenance|combined\s+with\s+chemo"
    r")\b",
    re.IGNORECASE,
)
_LATER_LINE_TEXT = re.compile(
    r"\b("
    r"pretreated|previously\s+treated|relapsed[/ -]?refractory|"
    r"refractory|post[- ]|after\s+.+\btherapy|after\s+.+\btreatment|"
    r"second[- ]line|third[- ]line|subsequent\s+lines?|"
    r"2l\b|3l\b|r/r|resistance\s+setting|prior\s+(?:line|lines|therapy|"
    r"therapies|arpi|taxane)|at\s+least\s+one\s+prior|>=\s*2|≥\s*2"
    r")\b",
    re.IGNORECASE,
)
_AR_PATHWAY_AGENTS = re.compile(
    r"\b(enzalutamide|apalutamide|darolutamide|abiraterone)\b",
    re.IGNORECASE,
)
_ER_ENDOCRINE_AGENTS = re.compile(
    r"\b("
    r"elacestrant|fulvestrant|tamoxifen|letrozole|anastrozole|exemestane|"
    r"palbociclib|ribociclib|abemaciclib|aromatase\s+inhibitor|serd|"
    r"endocrine\s+therapy"
    r")\b",
    re.IGNORECASE,
)
_HER2_DIRECTED_AGENTS = re.compile(
    r"\b("
    r"trastuzumab|pertuzumab|tucatinib|lapatinib|neratinib|margetuximab|"
    r"trastuzumab\s+deruxtecan|t-dxd"
    r")\b",
    re.IGNORECASE,
)
_MAPK_RTK_AGENTS = re.compile(
    r"\b("
    r"osimertinib|erlotinib|gefitinib|afatinib|dacomitinib|amivantamab|"
    r"alectinib|brigatinib|ceritinib|crizotinib|lorlatinib|entrectinib|"
    r"larotrectinib|selpercatinib|pralsetinib|capmatinib|tepotinib|"
    r"sotorasib|adagrasib|dabrafenib|trametinib|encorafenib|binimetinib|"
    r"vemurafenib|cetuximab|panitumumab|erdafitinib"
    r")\b",
    re.IGNORECASE,
)
THERAPY_PATH_TIERS = frozenset(
    {
        "approved_standard",
        "approved_indication_matched",
        "approved_later_line",
        "late_clinical",
        "trial_follow_up",
        "preclinical",
        "off_label",
    }
)
_THERAPY_PATH_RANK = {
    "approved_standard": 0,
    "approved_indication_matched": 1,
    "approved_later_line": 2,
    "late_clinical": 3,
    "trial_follow_up": 4,
    "preclinical": 6,
    "off_label": 7,
}
_THERAPY_PATH_DEFAULT_NOTE = {
    "approved_standard": "confirm indication and line of therapy",
    "approved_indication_matched": "confirm clinical eligibility",
    "approved_later_line": "confirm prior therapies and indication-specific eligibility",
    "late_clinical": "not default standard",
    "trial_follow_up": "not default standard",
    "preclinical": "not a clinical recommendation",
    "off_label": "confirm rationale and alternatives",
}
_THERAPY_EXPOSURE_RULES = (
    {
        "axis": "AR_signaling",
        "state": "down",
        "symbols": {"AR"},
        "agent_re": _AR_PATHWAY_AGENTS,
        "class_re": re.compile(r"\bandrogen\b", re.IGNORECASE),
        "disease_re": re.compile(
            r"\b(ar[- ]?axis\s+suppressed|ar\s+signaling\s+suppressed|"
            r"adt\s+exposure|androgen\s+deprivation|arpi)\b",
            re.IGNORECASE,
        ),
        "axis_label": "AR-axis",
        "exposure_label": "ADT or ARPI",
    },
    {
        "axis": "ER_signaling",
        "state": "down",
        "symbols": {"ESR1", "PGR"},
        "agent_re": _ER_ENDOCRINE_AGENTS,
        "class_re": re.compile(r"\bhormone\b", re.IGNORECASE),
        "disease_re": re.compile(
            r"\b(er[- ]?axis\s+suppressed|endocrine[- ]?exposed|"
            r"endocrine\s+therapy|aromatase\s+inhibitor|serd)\b",
            re.IGNORECASE,
        ),
        "axis_label": "ER-axis",
        "exposure_label": "endocrine therapy",
    },
    {
        "axis": "HER2_signaling",
        "state": "down",
        "symbols": {"ERBB2", "HER2"},
        "agent_re": _HER2_DIRECTED_AGENTS,
        "class_re": re.compile(r"\bher2\b", re.IGNORECASE),
        "disease_re": re.compile(
            r"\b(her2[- ]?axis\s+suppressed|anti[- ]?her2|"
            r"her2[- ]?directed)\b",
            re.IGNORECASE,
        ),
        "axis_label": "HER2-axis",
        "exposure_label": "HER2-directed therapy",
    },
    {
        "axis": "MAPK_EGFR_signaling",
        "state": "down",
        "symbols": {
            "EGFR", "BRAF", "KRAS", "NRAS", "MAP2K1", "MAP2K2",
            "MEK1", "MEK2", "ALK", "ROS1", "RET", "MET", "NTRK1",
            "NTRK2", "NTRK3", "FGFR1", "FGFR2", "FGFR3",
        },
        "agent_re": _MAPK_RTK_AGENTS,
        "class_re": re.compile(r"\b(tki|kinase)\b", re.IGNORECASE),
        "disease_re": re.compile(
            r"\b(egfr[- ]?tki|mapk[-/ ]?pathway\s+suppressed|"
            r"targeted\s+kinase\s+inhibitor)\b",
            re.IGNORECASE,
        ),
        "axis_label": "MAPK/RTK pathway",
        "exposure_label": "targeted kinase inhibitor",
    },
)


def _target_symbol(target_row) -> str:
    return _clean_text(
        target_row.get("symbol") if hasattr(target_row, "get") else ""
    ).upper()


def _agent_text(target_row) -> str:
    return _clean_text(
        target_row.get("agent") if hasattr(target_row, "get") else ""
    )


def _agent_class_text(target_row) -> str:
    return _clean_text(
        target_row.get("agent_class") if hasattr(target_row, "get") else ""
    ).lower()


def _phase_text(target_row) -> str:
    return _clean_text(
        target_row.get("phase") if hasattr(target_row, "get") else ""
    ).lower()


def _therapy_row_text(target_row) -> str:
    if not hasattr(target_row, "get"):
        return ""
    return " ".join(
        _clean_text(target_row.get(key))
        for key in ("agent", "agent_class", "indication", "rationale")
    )


def _therapy_row_matches_exposure_rule(target_row, rule: dict) -> bool:
    symbol = _target_symbol(target_row)
    if symbol and symbol in rule.get("symbols", set()):
        return True
    text = _therapy_row_text(target_row)
    agent_re = rule.get("agent_re")
    if agent_re is not None and agent_re.search(text):
        return True
    class_re = rule.get("class_re")
    if class_re is not None and class_re.search(_agent_class_text(target_row)):
        return True
    return False


def _therapy_path_context_for_tier(target_row, tier: str, note: str = "") -> str:
    agent_class = _agent_class_text(target_row)
    if tier == "approved_standard":
        prefix = "guideline-standard approved pathway"
    elif tier == "approved_indication_matched":
        prefix = "approved biomarker/indication-matched pathway"
    elif tier == "approved_later_line":
        prefix = (
            "approved radioligand pathway"
            if "radioligand" in agent_class
            else "approved later-line pathway"
        )
    elif tier == "late_clinical":
        prefix = "late-clinical follow-up"
    elif tier == "trial_follow_up":
        prefix = "clinical-trial follow-up"
    elif tier == "preclinical":
        prefix = "preclinical follow-up"
    elif tier == "off_label":
        prefix = "off-label follow-up"
    else:
        return ""

    suffix = _clean_text(note) or _THERAPY_PATH_DEFAULT_NOTE.get(tier, "")
    if suffix:
        return f"{prefix}; {suffix}"
    return prefix


def _explicit_therapy_path_info(target_row) -> dict | None:
    tier = _clean_text(
        target_row.get("treatment_path_tier")
        if hasattr(target_row, "get") else ""
    ).lower()
    if not tier:
        return None
    if tier not in THERAPY_PATH_TIERS:
        return None
    note = _clean_text(
        target_row.get("eligibility_note") if hasattr(target_row, "get") else ""
    )
    return {
        "tier": tier,
        "rank": _THERAPY_PATH_RANK.get(tier, 99),
        "context": _therapy_path_context_for_tier(target_row, tier, note),
        "source": "curated",
    }


def _inferred_therapy_path_info(target_row) -> dict:
    phase = _phase_text(target_row)
    agent_class = _agent_class_text(target_row)
    text = _therapy_row_text(target_row)
    is_standard = _STANDARD_PATH_TEXT.search(text) is not None
    is_later_line = _LATER_LINE_TEXT.search(text) is not None

    if phase == "approved":
        if "radioligand" in agent_class:
            return {
                "tier": "approved_later_line",
                "rank": 2,
                "context": (
                    "approved radioligand pathway; confirm imaging/eligibility "
                    "and prior-line requirements"
                ),
                "source": "inferred",
            }
        if is_standard:
            return {
                "tier": "approved_standard",
                "rank": 0,
                "context": (
                    "guideline-standard approved pathway; confirm the indication "
                    "and line of therapy"
                ),
                "source": "inferred",
            }
        if is_later_line:
            return {
                "tier": "approved_later_line",
                "rank": 2,
                "context": (
                    "approved later-line pathway; confirm prior therapies and "
                    "indication-specific eligibility"
                ),
                "source": "inferred",
            }
        return {
            "tier": "approved_indication_matched",
            "rank": 1,
            "context": (
                "approved biomarker/indication-matched pathway; confirm clinical "
                "eligibility"
            ),
            "source": "inferred",
        }

    phase_context = {
        "phase_3": ("late_clinical", 3, "late-clinical follow-up, not default standard"),
        "phase_2": ("trial_follow_up", 4, "clinical-trial follow-up, not default standard"),
        "phase_1": ("trial_follow_up", 5, "clinical-trial follow-up, not default standard"),
        "preclinical": ("preclinical", 6, "preclinical follow-up, not a clinical recommendation"),
        "off_label": ("off_label", 7, "off-label follow-up; confirm rationale and alternatives"),
    }
    tier, rank, context = phase_context.get(phase, ("unknown", 99, ""))
    return {"tier": tier, "rank": rank, "context": context, "source": "inferred"}


def _therapy_path_info(target_row) -> dict:
    return _explicit_therapy_path_info(target_row) or _inferred_therapy_path_info(target_row)


def therapy_path_tier(target_row) -> str:
    """Data-driven treatment-path tier used by report sorting/wording."""
    return _therapy_path_info(target_row)["tier"]


def _exposure_rule_active(rule: dict, *, analysis=None, disease_state=None) -> bool:
    axis_down = (
        _therapy_axis_state(analysis, str(rule.get("axis") or ""))
        == str(rule.get("state") or "").lower()
    )
    disease_re = rule.get("disease_re")
    disease_match = (
        disease_re.search(_clean_text(disease_state)) is not None
        if disease_re is not None else False
    )
    return bool(axis_down or disease_match)


def _therapy_axis_state(analysis, axis: str) -> str:
    if not isinstance(analysis, dict):
        return ""
    scores = analysis.get("therapy_response_scores") or {}
    score = scores.get(axis) if hasattr(scores, "get") else None
    if score is None:
        return ""
    if hasattr(score, "get"):
        return _clean_text(score.get("state")).lower()
    return _clean_text(getattr(score, "state", "")).lower()


def therapy_state_caution(target_row, *, analysis=None, disease_state=None) -> str:
    """Warn when a candidate therapy matches an exposure pattern already seen.

    This is intentionally phrased as a medication-reconciliation prompt, not
    a clinical contraindication. Single-sample RNA can suggest exposure but
    cannot determine whether the drug is current, prior, tolerated, or failed.
    """
    for rule in _THERAPY_EXPOSURE_RULES:
        if not _therapy_row_matches_exposure_rule(target_row, rule):
            continue
        if not _exposure_rule_active(rule, analysis=analysis, disease_state=disease_state):
            continue
        return (
            f"{rule['axis_label']} RNA/signaling is already suppressed "
            f"(current/prior {rule['exposure_label']} signal); verify the "
            "medication list before treating this as new-start therapy"
        )
    return ""


def therapy_path_context(target_row, *, analysis=None, disease_state=None) -> str:
    """Brief reader-facing treatment-path context for curated therapy rows."""
    return _therapy_path_info(target_row)["context"]


def therapy_path_rank(target_row, *, analysis=None, disease_state=None) -> int:
    """Sort standard paths ahead of exploratory rows in concise reports."""
    return int(_therapy_path_info(target_row)["rank"])


def target_interpretation_summary(
    target_row,
    expression_row,
    target_panel=None,
    *,
    analysis=None,
    disease_state=None,
):
    """Return a compact integrated summary for a curated target row."""
    source = tumor_attribution_context(expression_row)
    normal = normal_expression_context(expression_row)
    maturity = (
        clinical_maturity_summary(target_row, target_panel=target_panel)
        if target_row is not None else ""
    )
    if target_row is not None and expression_independent_indication(target_row):
        parts = [expression_independent_interpretation(target_row), source["band"]]
    else:
        parts = [source["label"], source["band"]]
    parts.append(normal["label"])
    details = list(normal.get("details") or [])
    if details:
        parts.append(details[0])
    path_context = (
        therapy_path_context(target_row, analysis=analysis, disease_state=disease_state)
        if target_row is not None else ""
    )
    if path_context:
        parts.append(path_context)
    caution = (
        therapy_state_caution(target_row, analysis=analysis, disease_state=disease_state)
        if target_row is not None else ""
    )
    if caution:
        parts.append(f"current-therapy check: {caution}")
    if maturity:
        parts.append(maturity)
    return "; ".join(part for part in parts if part)


def partition_tumor_core_rows(ranges_df, min_tumor_tpm=1.0):
    """Split expression rows by report-facing tumor-core reliability."""
    if ranges_df is None or len(ranges_df) == 0 or "attr_tumor_tpm" not in ranges_df.columns:
        empty = ranges_df.iloc[0:0] if ranges_df is not None else None
        return empty, empty, empty

    eligible = ranges_df[ranges_df["attr_tumor_tpm"].astype(float) >= float(min_tumor_tpm)].copy()
    if eligible.empty:
        empty = eligible.iloc[0:0]
        return empty, empty, empty

    statuses = [
        target_reliability_status(row)
        for _, row in eligible.iterrows()
    ]
    eligible["_report_reliability"] = statuses
    supported = eligible[eligible["_report_reliability"] == "supported"].copy()
    provisional = eligible[eligible["_report_reliability"] == "provisional"].copy()
    unsupported = eligible[eligible["_report_reliability"] == "unsupported"].copy()
    return supported, provisional, unsupported


def summarize_reliability_reasons(rows, top_n=3):
    """Summarize the most common reliability caveats in a row set."""
    if rows is None or len(rows) == 0:
        return ""
    counter = Counter()
    for _, row in rows.iterrows():
        counter.update(target_reliability_reasons(row))
    if not counter:
        return ""
    return ", ".join(reason for reason, _ in counter.most_common(top_n))


def resolved_subtype_code_for_analysis(analysis, ranges_df=None):
    """Return the final subtype/cancer code implied by the analysis."""
    candidate_trace = analysis.get("candidate_trace") or []
    if not candidate_trace:
        return None
    winning_subtype = _clean_text(candidate_trace[0].get("winning_subtype"))
    if not winning_subtype:
        return None

    tumor_tpm_by_symbol = analysis.get("tumor_tpm_by_symbol")
    if not tumor_tpm_by_symbol and ranges_df is not None:
        try:
            tumor_tpm_by_symbol = {
                str(row["symbol"]): float(row.get("attr_tumor_tpm") or 0.0)
                for _, row in ranges_df.iterrows()
                if str(row.get("symbol") or "")
            }
        except Exception:
            tumor_tpm_by_symbol = None

    try:
        from .degenerate_subtype import resolve_degenerate_subtype

        decomposition = analysis.get("decomposition") or {}
        site_template = decomposition.get("best_template")
        resolution = resolve_degenerate_subtype(
            winning_subtype,
            site_template=site_template,
            tumor_tpm_by_symbol=tumor_tpm_by_symbol,
        )
        winning_subtype = _clean_text(
            resolution.get("final_subtype")
        ) or winning_subtype
    except Exception:
        pass
    return winning_subtype or None


def _match_curated_subtype(parent_code, *candidates):
    """Return the exact curated subtype string matching any candidate."""
    if not parent_code:
        return None
    try:
        from .gene_sets_cancer import cancer_key_genes_subtypes

        curated_subtypes = [
            _clean_text(item)
            for item in cancer_key_genes_subtypes(parent_code)
        ]
    except Exception:
        return None

    curated_subtypes = [item for item in curated_subtypes if item]
    if not curated_subtypes:
        return None

    exact_lookup = {item: item for item in curated_subtypes}
    lower_lookup = {item.lower(): item for item in curated_subtypes}

    for candidate in candidates:
        raw = _clean_text(candidate)
        if not raw:
            continue
        normalized = raw.replace("-", "_")
        variants = (
            raw,
            normalized,
            raw.lower(),
            normalized.lower(),
            raw.upper(),
            normalized.upper(),
        )
        for variant in variants:
            if variant in exact_lookup:
                return exact_lookup[variant]
        for variant in variants:
            match = lower_lookup.get(str(variant).lower())
            if match:
                return match
    return None


def cancer_key_genes_lookup_for_analysis(cancer_code, analysis, ranges_df=None):
    """Return ``(panel_code, panel_subtype)`` for curated report lookups.

    Most samples use their top-level cancer code directly. Some need a
    subtype-filtered panel (for example ``SARC`` + ``leiomyosarcoma``),
    while others resolve onto a different curated code entirely (for
    example a ``SARC`` umbrella call that degeneracy-resolution upgrades
    to ``OS``).
    """
    default_code = _clean_text(cancer_code)
    resolved_code = resolved_subtype_code_for_analysis(
        analysis,
        ranges_df=ranges_df,
    )

    try:
        from .gene_sets_cancer import (
            cancer_key_genes_cancer_types,
            cancer_type_registry,
        )

        curated_codes = {
            _clean_text(code)
            for code in cancer_key_genes_cancer_types()
        }
        if resolved_code and resolved_code in curated_codes:
            return resolved_code, None

        if resolved_code:
            reg = cancer_type_registry()
            match = reg[reg["code"] == resolved_code]
            if not match.empty:
                row = match.iloc[0]
                parent_code = _clean_text(row.get("parent_code"))
                subtype_key = _clean_text(row.get("subtype_key"))
                suffix = ""
                if parent_code and resolved_code.startswith(parent_code + "_"):
                    suffix = resolved_code[len(parent_code) + 1:]
                matched_subtype = _match_curated_subtype(
                    parent_code,
                    subtype_key,
                    suffix,
                )
                if parent_code and matched_subtype:
                    return parent_code, matched_subtype

        if default_code and default_code in curated_codes:
            return default_code, None
    except Exception:
        pass

    return default_code or resolved_code, None


def subtype_key_for_analysis(analysis, ranges_df=None):
    """Return the curated subtype key implied by the current analysis."""
    winning_subtype = resolved_subtype_code_for_analysis(
        analysis,
        ranges_df=ranges_df,
    )
    if not winning_subtype:
        return None

    try:
        from .gene_sets_cancer import cancer_type_registry

        reg = cancer_type_registry()
        match = reg[reg["code"] == winning_subtype]
        if match.empty:
            return None
        row = match.iloc[0]
        subtype_key = str(row.get("subtype_key") or "").strip()
        if subtype_key and subtype_key.lower() != "nan":
            return subtype_key
        parent_code = str(row.get("parent_code") or "").strip()
        code = str(row.get("code") or "").strip()
        if parent_code and code.startswith(parent_code + "_"):
            suffix = code[len(parent_code) + 1:].strip().lower()
            return suffix or None
    except Exception:
        return None
    return None
