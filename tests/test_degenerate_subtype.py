"""Tests for the degenerate-subtype registry + fusion-surrogate catalog (#198).

Pins the OS/DDLPS bone-vs-soft-tissue tiebreaker, the Ewing/DSRCT/ARMS
fusion-surrogate voting, the NE-axis site disambiguator, and the
data-integrity contract on both CSV files.
"""

from pirlygenes.degenerate_subtype import (
    degenerate_subtype_pairs,
    fusion_surrogate_expression,
    fusion_surrogate_genes_for,
    resolve_degenerate_subtype,
)
from pirlygenes.gene_sets_cancer import cancer_type_registry


def test_degenerate_pairs_csv_loads_and_parses():
    df = degenerate_subtype_pairs()
    assert len(df) >= 5
    for _, row in df.iterrows():
        assert isinstance(row["members"], list)
        assert len(row["members"]) >= 2
        assert isinstance(row["tiebreaker_mapping"], dict)
        assert row["tiebreaker_rule"] in {
            "site_template", "fusion_surrogate", "marker_combo",
        }


def test_degenerate_pair_members_are_real_registry_codes():
    """Every member listed must exist in the registry."""
    reg_codes = set(cancer_type_registry()["code"])
    df = degenerate_subtype_pairs()
    # Some advanced pair entries reference subtype codes we have not
    # yet added to the registry (SARC_CIC_DUX4, SARC_BCOR, EPN_RELA,
    # MBL_SHH/WNT/MYC, LAML_APL). Allow those as a transitional
    # allowlist; shrink when the registry grows to include them.
    allow_not_yet_in_registry = {
        "SARC_CIC_DUX4", "SARC_BCOR", "EPN_RELA",
        "MBL_SHH", "MBL_WNT", "MBL_MYC",
    }
    unknown = set()
    for _, row in df.iterrows():
        for member in row["members"]:
            if member not in reg_codes and member not in allow_not_yet_in_registry:
                unknown.add(member)
    assert not unknown, (
        f"Pair members not in registry and not allowlisted: {sorted(unknown)}"
    )


def test_fusion_surrogate_csv_loads():
    df = fusion_surrogate_expression()
    assert len(df) >= 30
    required_cols = {
        "fusion_class", "surrogate_gene", "surrogate_role",
        "cancer_code", "rationale",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"fusion-surrogate schema missing: {missing}"


def test_fusion_surrogate_roles_are_valid():
    df = fusion_surrogate_expression()
    valid = {"activated", "activated_ectopic", "myogenic_lineage"}
    bad = set(df["surrogate_role"]) - valid
    assert not bad, f"unknown surrogate_role values: {bad}"


# ── Resolver contract ──────────────────────────────────────────────


def test_resolver_returns_no_pair_for_unlisted_subtype():
    result = resolve_degenerate_subtype(
        winning_subtype="BRCA_LumA",
        site_template="solid_primary",
    )
    assert result["status"] == "no_pair"
    assert result["final_subtype"] == "BRCA_LumA"


def test_resolver_returns_no_pair_for_none_subtype():
    result = resolve_degenerate_subtype(winning_subtype=None)
    assert result["status"] == "no_pair"


def test_resolver_os_vs_ddlps_bone_site_swaps_to_os():
    """Canonical pfo004-shape input: classifier picked DDLPS, site is
    bone, so the resolver should swap DDLPS → OS."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_DDLPS",
        site_template="met_bone",
    )
    assert result["status"] == "corrected", result
    assert result["final_subtype"] == "OS"
    assert "site_template" in result["reason"]


def test_resolver_os_vs_ddlps_retroperitoneum_confirms_ddlps():
    """When classifier picks DDLPS and site is retroperitoneum, the
    tiebreaker agrees → confirmed."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_DDLPS",
        site_template="primary_retroperitoneum",
    )
    assert result["status"] == "confirmed"
    assert result["final_subtype"] == "SARC_DDLPS"


def test_resolver_os_vs_ddlps_unknown_site_leaves_degenerate():
    """When site doesn't appear in the mapping, resolver returns
    degenerate so the markdown layer can emit the ambiguity."""
    result = resolve_degenerate_subtype(
        winning_subtype="SARC_DDLPS",
        site_template="solid_primary",  # not in OS/DDLPS mapping
    )
    assert result["status"] == "degenerate"
    assert result["final_subtype"] == "SARC_DDLPS"  # left unchanged
    assert "degenerate between" in result["reason"]


def test_resolver_ewing_fate1_nr0b1_confirms_ewing():
    """FATE1 + NR0B1 high → EWS-FLI1 signature → confirms Ewing."""
    result = resolve_degenerate_subtype(
        winning_subtype="EWS",
        tumor_tpm_by_symbol={
            "FATE1": 80.0,
            "NR0B1": 50.0,
            "WT1": 0.0,
        },
    )
    assert result["status"] == "confirmed"
    assert result["final_subtype"] == "EWS"


def test_resolver_dsrct_wt1_corrects_from_ewing():
    """If classifier picked EWS but WT1 is high and FATE1/NR0B1 are
    silent, DSRCT is the real call."""
    result = resolve_degenerate_subtype(
        winning_subtype="EWS",
        tumor_tpm_by_symbol={
            "FATE1": 0.0,
            "NR0B1": 0.0,
            "WT1": 200.0,
        },
    )
    # DSRCT has a single surrogate gene (WT1) while EWS has multiple,
    # so the 1-vote margin for DSRCT shouldn't outrank EWS unless the
    # EWS genes are silent. Here EWS genes are 0 TPM → 0 votes, DSRCT
    # gets 1 vote → corrected.
    assert result["status"] == "corrected"
    assert result["final_subtype"] == "SARC_DSRCT"


def test_resolver_pannet_vs_midnet_pancreas_site():
    result = resolve_degenerate_subtype(
        winning_subtype="MID_NET",
        site_template="primary_pancreas",
    )
    assert result["status"] == "corrected"
    assert result["final_subtype"] == "PANNET"


def test_resolver_squamous_axis_lung_corrects_hnsc_to_lusc():
    result = resolve_degenerate_subtype(
        winning_subtype="HNSC",
        site_template="primary_lung",
    )
    assert result["status"] == "corrected"
    assert result["final_subtype"] == "LUSC"


def test_resolver_mcl_ccnd1_corrects_from_cll():
    """CCND1 high drives MCL even if classifier picked CLL."""
    result = resolve_degenerate_subtype(
        winning_subtype="CLL",
        tumor_tpm_by_symbol={"CCND1": 200.0, "SOX11": 80.0, "BCL6": 0.0},
    )
    assert result["status"] == "corrected"
    assert result["final_subtype"] == "MCL"


def test_resolver_no_fusion_panel_tpm_leaves_degenerate():
    """When no TPM dict is provided for a fusion-surrogate pair, the
    resolver returns degenerate rather than guessing."""
    result = resolve_degenerate_subtype(
        winning_subtype="EWS",
        tumor_tpm_by_symbol=None,
    )
    assert result["status"] == "degenerate"


# ── Fusion-surrogate genes by cancer ────────────────────────────────


def test_fusion_surrogate_genes_for_ewing():
    hits = fusion_surrogate_genes_for("EWS")
    genes = {h["gene"] for h in hits}
    for required in ("FATE1", "NR0B1", "PHOX2B"):
        assert required in genes, f"EWS surrogate missing: {required}"


def test_fusion_surrogate_genes_for_nutm():
    hits = fusion_surrogate_genes_for("NUTM")
    genes = {h["gene"] for h in hits}
    # NUTM1 itself + PRAME are the critical surrogates — without them
    # the NUT-carcinoma diagnosis from bulk RNA is not reachable.
    for required in ("NUTM1", "PRAME", "MAGEA1"):
        assert required in genes, f"NUTM surrogate missing: {required}"


def test_fusion_surrogate_pan_cancer_applies_to_any_code():
    """NTRK/ROS1 fusion surrogates are marked pan_cancer and should
    return for any cancer code that receives them."""
    hits = fusion_surrogate_genes_for("LUAD")
    genes = {h["gene"] for h in hits}
    for required in ("NTRK1", "NTRK3", "ROS1", "ALK"):
        assert required in genes, f"LUAD fusion surrogate missing: {required}"


def test_brief_renders_corrected_subtype():
    """End-to-end pin: when the analysis dict carries a liposarcoma
    winning_subtype but the decomposition top template is met_bone, the
    summary.md should render osteosarcoma-consistent + a subtype note
    explaining the swap. Reproduces the pfo004 failure mode."""
    from pirlygenes.brief import build_summary

    analysis = {
        "purity": {
            "overall_estimate": 0.74,
            "overall_lower": 0.42,
            "overall_upper": 1.0,
        },
        "purity_confidence": type("PT", (), {"tier": "low"})(),
        "sample_context": None,
        "cancer_name": "Sarcoma",
        "candidate_trace": [
            {"code": "SARC", "winning_subtype": "SARC_LPS_UNSPEC"},
        ],
        "decomposition": {
            "best_template": "met_bone",
            "best_cancer_type": "SARC",
            "hypotheses": [
                {"template": "met_bone", "cancer_type": "SARC", "score": 0.22},
            ],
        },
        "tumor_tpm_by_symbol": {"MDM2": 877, "CDK4": 73, "FRS2": 137},
    }
    summary = build_summary(
        analysis,
        ranges_df=None,
        cancer_code="SARC",
        disease_state="",
        sample_id="synthetic-bone-mdm2",
    )
    assert "osteosarcoma-consistent" in summary, summary
    assert "site_template tiebreaker swapped" in summary, summary
    assert "OS" in summary


def test_refs_populated_for_each_pair():
    """Every pair row should cite at least one reference (PMID or
    empty-cell for canonical/textbook cases). Enforces the curation
    bar from issue #198."""
    df = degenerate_subtype_pairs()
    # Allow at most one row with empty refs (the trivial COAD/READ
    # site case — canonical and doesn't need a citation).
    empty_refs = df[df["refs"].fillna("").astype(str).str.strip().eq("")]
    assert len(empty_refs) <= 1, (
        f"Too many degenerate-pair rows without a PMID citation: "
        f"{empty_refs['pair_id'].tolist()}"
    )
