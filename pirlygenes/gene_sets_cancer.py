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

import warnings

from .load_dataset import get_data


# ---------- Therapy target registry ----------

_THERAPY_REGISTRY = {
    "ADC": ["ADC-trials", "ADC-approved"],
    "ADC-trials": ["ADC-trials"],
    "ADC-approved": ["ADC-approved"],
    "TCR-T": ["TCR-T-trials", "TCR-T-approved"],
    "TCR-T-trials": ["TCR-T-trials"],
    "TCR-T-approved": ["TCR-T-approved"],
    "CAR-T": ["CAR-T-approved"],
    "CAR-T-approved": ["CAR-T-approved"],
    "bispecific-antibodies": ["bispecific-antibodies-approved"],
    "bispecific-antibodies-approved": ["bispecific-antibodies-approved"],
    "radioligand": ["radioligand-targets"],
    "multispecific-TCE": ["multispecific-tcell-engager-trials"],
}


# ---------- Low-level field extractors ----------


def get_field_from_gene_set(
    gene_set_name,
    candidate_columns,
    try_lower_case=True,
    try_upper_case=True,
    try_no_underscore=True,
    try_plural=True,
):
    if try_plural:
        candidate_columns = candidate_columns + [c + "s" for c in candidate_columns]
    if try_no_underscore:
        candidate_columns = candidate_columns + [
            c.replace("_", "") for c in candidate_columns
        ]
    if try_lower_case:
        candidate_columns = candidate_columns + [c.lower() for c in candidate_columns]
    if try_upper_case:
        candidate_columns = candidate_columns + [c.upper() for c in candidate_columns]

    df = get_data(gene_set_name)
    result = set()
    for column in candidate_columns:
        if column in df.columns:
            for x in df[column]:
                if type(x) is str:
                    result.update([xi.strip() for xi in x.split(";")])
    return result


def get_target_gene_name_set(
    name,
    candidate_symbol_columns=[
        "Tumor_Target_Symbol",
        "Symbol",
        "Gene_Name",
    ],
):
    return get_field_from_gene_set(name, candidate_symbol_columns)


def get_target_gene_id_set(
    name,
    candidate_id_columns=[
        "Tumor_Target_Ensembl_Gene_ID",
        "Tumor_Target_Ensembl_GeneID",
        "Tumor_Target_Gene_ID",
        "Tumor_Target_GeneID",
        "Target_Ensembl_Gene_ID",
        "Target_Ensembl_GeneID",
        "Target_Gene_ID",
        "Target_GeneID",
        "Ensembl_Gene_ID",
        "Ensembl_GeneID",
        "Gene_ID",
    ],
):
    return get_field_from_gene_set(name, candidate_id_columns)


def get_target_gene_id_to_name(name):
    """Extract paired {Ensembl_Gene_ID: Symbol} dict from a therapy dataset.

    Handles semicolon-separated multi-target entries by positional pairing.
    """
    df = get_data(name)

    sym_candidates = ["Tumor_Target_Symbol", "Tumor_Target_Symbols", "Symbol", "Symbols", "Gene_Name"]
    id_candidates = [
        "Tumor_Target_Ensembl_Gene_ID", "Tumor_Target_Ensembl_Gene_IDs",
        "Ensembl_Gene_ID", "Ensembl_Gene_IDs", "Ensembl_GeneIDs", "Gene_ID",
    ]

    sym_col = next((c for c in sym_candidates if c in df.columns), None)
    id_col = next((c for c in id_candidates if c in df.columns), None)

    if sym_col is None or id_col is None:
        # Fall back to separate extraction
        names = get_target_gene_name_set(name)
        ids = get_target_gene_id_set(name)
        return {gid: gid for gid in ids} if ids else {n: n for n in names}

    result = {}
    for _, row in df.iterrows():
        syms_raw = str(row[sym_col]).strip() if row[sym_col] is not None else ""
        ids_raw = str(row[id_col]).strip() if row[id_col] is not None else ""
        if not syms_raw or syms_raw.lower() in ("nan", "none"):
            continue
        syms = [s.strip() for s in syms_raw.split(";") if s.strip()]
        ids = [i.strip() for i in ids_raw.split(";") if i.strip() and i.strip().startswith("ENSG")]
        for i, sym in enumerate(syms):
            if i < len(ids):
                result[ids[i]] = sym
    return result


# ---------- Generic therapy target accessors ----------


def therapy_target_gene_names(therapy):
    """Gene symbols targeted by a therapy type.

    Parameters
    ----------
    therapy : str
        Key from _THERAPY_REGISTRY (e.g. "ADC", "TCR-T", "CAR-T",
        "bispecific-antibodies", "radioligand", "multispecific-TCE").
    """
    if therapy not in _THERAPY_REGISTRY:
        raise ValueError(
            f"Unknown therapy '{therapy}'. Available: {sorted(_THERAPY_REGISTRY.keys())}"
        )
    result = set()
    for csv_name in _THERAPY_REGISTRY[therapy]:
        result.update(get_target_gene_name_set(csv_name))
    return result


def therapy_target_gene_ids(therapy):
    """Ensembl gene IDs targeted by a therapy type."""
    if therapy not in _THERAPY_REGISTRY:
        raise ValueError(
            f"Unknown therapy '{therapy}'. Available: {sorted(_THERAPY_REGISTRY.keys())}"
        )
    result = set()
    for csv_name in _THERAPY_REGISTRY[therapy]:
        result.update(get_target_gene_id_set(csv_name))
    return result


def therapy_target_gene_id_to_name(therapy):
    """Therapy targets as {Ensembl_Gene_ID: Symbol} dict (fast path for plotting)."""
    if therapy not in _THERAPY_REGISTRY:
        raise ValueError(
            f"Unknown therapy '{therapy}'. Available: {sorted(_THERAPY_REGISTRY.keys())}"
        )
    result = {}
    for csv_name in _THERAPY_REGISTRY[therapy]:
        result.update(get_target_gene_id_to_name(csv_name))
    return result


# ---------- Mitochondrial genes ----------
def mitochondrial_genes_df(role=None):
    """DataFrame of mitochondrially-encoded genes (chrM).

    Parameters
    ----------
    role : str or None
        ``"protein_coding"`` — the 13 OXPHOS subunits (MT-CO*, MT-ND*,
        MT-CYB, MT-ATP6/8).
        ``"rRNA"`` — MT-RNR1, MT-RNR2.
        ``None`` (default) — all 15 rows.

    Returns
    -------
    pd.DataFrame
        Columns: Symbol, Ensembl_Gene_ID, Role.
    """
    df = get_data("mitochondrial-genes")
    if role is not None:
        df = df[df["Role"] == role]
    return df


def mitochondrial_gene_names(role=None):
    """Set of mitochondrially-encoded gene symbols. See `mitochondrial_genes_df`."""
    return set(mitochondrial_genes_df(role=role)["Symbol"])


def mitochondrial_gene_ids(role=None):
    """Set of mitochondrially-encoded Ensembl gene IDs. See `mitochondrial_genes_df`."""
    return set(mitochondrial_genes_df(role=role)["Ensembl_Gene_ID"])


# ---------- Culture-stress genes ----------
def culture_stress_genes_df(category=None):
    """DataFrame of culture-stress-UP genes used to flag cell-line samples.

    Genes upregulated in culture-adapted cells vs primary tissue (Yu et al.,
    Nat Commun 2019). Split across categories: HSP / glycolysis /
    proliferation / ER_stress / oxidative_stress / glutamine.
    """
    df = get_data("culture-stress-genes")
    if category is not None:
        df = df[df["Category"] == category]
    return df


def culture_stress_gene_names(category=None):
    return set(culture_stress_genes_df(category=category)["Symbol"])


def culture_stress_gene_ids(category=None):
    return set(culture_stress_genes_df(category=category)["Ensembl_Gene_ID"])


# ---------- Tumor microenvironment (TME) markers ----------
def tme_markers_df(cell_type=None):
    """DataFrame of TME marker genes split by cell type.

    Used to detect the presence of tumor microenvironment (absent in cell
    lines, present in tissue biopsies). `cell_type` is one of ``T_cell``,
    ``B_cell``, ``myeloid``, ``fibroblast``, ``endothelial``, or None for
    all markers.
    """
    df = get_data("tme-markers")
    if cell_type is not None:
        df = df[df["Cell_Type"] == cell_type]
    return df


def tme_marker_gene_names(cell_type=None):
    return set(tme_markers_df(cell_type=cell_type)["Symbol"])


def tme_marker_gene_ids(cell_type=None):
    return set(tme_markers_df(cell_type=cell_type)["Ensembl_Gene_ID"])


# ---------- Degradation gene pairs (transcript length FFPE index) ----------
def degradation_gene_pairs_df():
    """DataFrame of (short, long) gene pairs with expected long/short ratios.

    Used by the transcript-length degradation index: in fresh tissue the
    ratio is near 1; in FFPE / degraded RNA the long transcript is
    preferentially lost. Each row has short/long symbol + Ensembl ID +
    expected ratio calibrated from 83 fresh/frozen reference columns.
    """
    return get_data("degradation-gene-pairs")


def degradation_gene_pairs():
    """List of (short_symbol, long_symbol, expected_ratio) tuples.

    Convenience view for callers that don't need Ensembl IDs. Returns the
    pairs in the order they appear in the CSV.
    """
    df = degradation_gene_pairs_df()
    return [
        (row["Short_Gene_Symbol"], row["Long_Gene_Symbol"], float(row["Expected_Long_Over_Short_Ratio"]))
        for _, row in df.iterrows()
    ]


# ---------- Cancer lineage genes ----------
def lineage_genes_df(cancer_type=None):
    """DataFrame of per-TCGA-cancer lineage genes.

    Lineage genes are retained in metastases and specific enough to
    calibrate tumor purity. Each row is (Cancer_Type, Symbol,
    Ensembl_Gene_ID). Filter by `cancer_type` to get one type's genes.
    """
    df = get_data("lineage-genes")
    if cancer_type is not None:
        df = df[df["Cancer_Type"] == cancer_type]
    return df


def lineage_gene_symbols(cancer_type):
    """List of lineage gene symbols for a given TCGA cancer type code.

    Preserves CSV order (which encodes the curator's intent about which
    markers are most load-bearing for that cancer type).
    """
    df = lineage_genes_df(cancer_type=cancer_type)
    return df["Symbol"].tolist()


def lineage_gene_ids(cancer_type):
    """List of lineage Ensembl IDs for a given TCGA cancer type code."""
    df = lineage_genes_df(cancer_type=cancer_type)
    return df["Ensembl_Gene_ID"].tolist()


def lineage_genes_by_cancer_type():
    """Dict of {TCGA_code: [Symbol, ...]} for all cancer types.

    Primarily for consumers that historically used a pre-built dict. Built
    once per process via the `get_data` cache.
    """
    df = get_data("lineage-genes")
    return {
        code: group["Symbol"].tolist()
        for code, group in df.groupby("Cancer_Type", sort=False)
    }


# ---------- Cancer family panels ----------
def cancer_family_panels_df(family=None):
    """DataFrame of broad-family signature panels used for cancer-type scoring.

    Family labels include ``PROSTATE``, ``CRC``, ``GASTRIC``, ``ESCA_SQ``,
    ``SQUAMOUS``, ``MESENCHYMAL``, ``RENAL``, ``GLIAL``, ``MELANOCYTIC``.
    """
    df = get_data("cancer-family-panels")
    if family is not None:
        df = df[df["Family"] == family]
    return df


def cancer_family_panel(family):
    """List of Symbols for one cancer family. See `cancer_family_panels_df`."""
    return cancer_family_panels_df(family=family)["Symbol"].tolist()


def cancer_family_panels():
    """Dict of {family_label: [Symbol, ...]} for all families."""
    df = get_data("cancer-family-panels")
    return {
        family: group["Symbol"].tolist()
        for family, group in df.groupby("Family", sort=False)
    }


# ---------- Housekeeping genes ----------
def housekeeping_gene_names(core_only=False):
    df = get_data("housekeeping-genes")
    if core_only:
        df = df[df["Category"] == "Core"]
    return set(df["Symbol"])


def housekeeping_gene_ids(core_only=False):
    df = get_data("housekeeping-genes")
    if core_only:
        df = df[df["Category"] == "Core"]
    return set(df["Ensembl_Gene_ID"])


# ---------- FFPE-sensitive marker panel (issue #72 follow-up) ----------
#
# Curated panel of genes whose expression behaviour distinguishes FFPE
# from fresh / frozen RNA *independently of the matched gene-pair length
# index* (which becomes unreliable under exon-capture enrichment because
# probe density biases long transcripts).
#
# Two directions:
#
# - ``drops_in_ffpe``: genes that systematically under-detect in FFPE
#   for length-independent reasons (extreme transcript length, long
#   3′UTR with low mRNA stability, structural fragility).
# - ``stable_in_ffpe``: short, abundant reference genes that retain
#   detectable expression even in heavily degraded FFPE.
#
# The score is ``geomean(stable_TPM) / geomean(drops_TPM)`` after
# pseudocounting; values >> reference suggest FFPE.

def ffpe_sensitive_markers_df(direction=None):
    """Return the FFPE-sensitive marker panel.

    Parameters
    ----------
    direction : {"drops_in_ffpe", "stable_in_ffpe", None}
        If specified, return only one direction; otherwise return all.
    """
    df = get_data("ffpe-sensitive-markers")
    if direction is not None:
        df = df[df["direction"] == direction]
    return df


def ffpe_sensitive_gene_names(direction=None):
    return set(ffpe_sensitive_markers_df(direction)["symbol"])


def ffpe_sensitive_gene_ids(direction=None):
    return set(ffpe_sensitive_markers_df(direction)["ensembl_gene_id"])


# ---------- Extended housekeeping (issue #60) ----------
#
# A single authoritative panel of genes that cannot discriminate between
# cell types because they are ubiquitously expressed at high levels, or
# because their apparent abundance reflects artifact (MT copy number,
# FFPE short-transcript survival, clonal IG/TR rearrangement) rather
# than biology. Excluded from:
#
# - Marker selection (``engine._select_marker_rows``) — never picked as
#   a decomposition-component discriminator.
# - Tumor-attribution ranking (``estimate_tumor_expression_ranges``) —
#   flagged ``excluded_from_ranking`` so they don't show up as top
#   therapy targets when their observed TPM is high for housekeeping
#   reasons.
#
# Kept declarative (regex families + curated classics) so the panel can
# be audited at a glance and extended without a CSV churn cycle. The
# ``scope`` on each family lets B2M stay visible in the ranking output
# (it's a legitimate MHC-I context gene) while still being skipped from
# decomposition markers (the NNLS shouldn't treat it as a lineage
# discriminator).

import re as _re  # noqa: E402

_EXTENDED_HK_FAMILIES = [
    # (family_name, compiled_regex, scope)
    # scope ∈ {"markers", "ranking", "both"}
    ("mitochondrial_chrM",          _re.compile(r"MT-.*"),                              "both"),
    ("cytosolic_ribosomal",         _re.compile(r"(RPL|RPS)\d.*"),                      "both"),
    ("mito_ribosomal",              _re.compile(r"(MRPL|MRPS)\d.*"),                    "both"),
    ("translation_factors",         _re.compile(r"(EEF|EIF)\d.*"),                      "both"),
    ("hnRNPs",                      _re.compile(r"HNRNP[A-Z]\d?[A-Z]?\d?"),             "both"),
    ("splicing_SR_proteins",        _re.compile(r"SRSF\d+"),                            "both"),
    ("snRNPs",                      _re.compile(r"SNRP[A-Z]\d?"),                       "both"),
    ("proteasome_subunits",         _re.compile(r"PSM[ABCDEF]\d+"),                     "both"),
    ("cyclophilins",                _re.compile(r"PPI[A-Z]\d?"),                        "both"),
    ("tubulin",                     _re.compile(r"TUB(A|B|G)\d+[A-Z]?"),                "both"),
    ("rearranged_ig_segments",      _re.compile(r"(IGH|IGK|IGL)[VDJC]\d.*"),            "both"),
    ("rearranged_tcr_segments",     _re.compile(r"(TRA|TRB|TRG|TRD)[VDJC]\d.*"),        "both"),
]

# Classic housekeeping symbols that don't follow a family regex. Keep
# these as an enumerated allow-list because bare prefix matching would
# catch too much. Each row declares the exclusion scope so we can keep
# biologically-meaningful genes (B2M) in the ranking output.
_EXTENDED_HK_CLASSICS = {
    # symbol: scope
    "ACTB":   "both",
    "ACTG1":  "both",
    "GAPDH":  "both",
    "HPRT1":  "both",
    "TPT1":   "both",
    "RACK1":  "both",
    "B2M":    "markers",   # MHC-I context — keep visible in ranking.
    "PGK1":   "both",
    "YWHAZ":  "both",
    "UBC":    "both",
    "PPIA":   "both",
}


def _extended_hk_symbol_match(symbol):
    """Return ``(family_name, scope)`` if the symbol matches any family
    in the extended HK panel, else ``(None, None)``.
    """
    if not isinstance(symbol, str) or not symbol:
        return None, None
    for family, pattern, scope in _EXTENDED_HK_FAMILIES:
        if pattern.fullmatch(symbol):
            return family, scope
    classic_scope = _EXTENDED_HK_CLASSICS.get(symbol)
    if classic_scope is not None:
        return "classic_hk", classic_scope
    return None, None


def is_extended_housekeeping_symbol(symbol, scope="markers"):
    """Return True when the symbol should be excluded for the given
    scope (``"markers"``, ``"ranking"``, or ``"both"``).

    Issue #60. ``scope="markers"`` is the superset — every symbol
    excluded from ranking is also excluded from marker selection.
    ``scope="ranking"`` excludes only what's unambiguously uninteresting
    as a therapy target (keeps B2M, since MHC-I readers need it).
    """
    family, family_scope = _extended_hk_symbol_match(symbol)
    if family is None:
        return False
    if family_scope == "both":
        return True
    return family_scope == scope


def extended_housekeeping_symbols(scope="markers"):
    """Return a set of classic-symbol housekeeping members for the
    given scope. Regex families are not enumerated — use
    ``is_extended_housekeeping_symbol`` for runtime membership tests.
    """
    out = set()
    for symbol, sc in _EXTENDED_HK_CLASSICS.items():
        if sc == "both" or sc == scope:
            out.add(symbol)
    return out


# ---------- Surface proteins (surfaceome) ----------
def _surface_proteins_df(category=None):
    df = get_data("surface-proteins")
    if category is not None:
        df = df[df["Category"] == category]
    return df


def surface_protein_gene_names(validated_only=False):
    """Human cell surface protein gene symbols.

    Parameters
    ----------
    validated_only : bool
        If True, return only CSPA mass-spec validated proteins.
        If False (default), include ML-predicted surfaceome.
    """
    cat = "CSPA_validated" if validated_only else None
    return set(_surface_proteins_df(cat)["Symbol"])


def surface_protein_gene_ids(validated_only=False):
    """Human cell surface protein Ensembl gene IDs.

    Parameters
    ----------
    validated_only : bool
        If True, return only CSPA mass-spec validated proteins.
        If False (default), include ML-predicted surfaceome.
    """
    cat = "CSPA_validated" if validated_only else None
    df = _surface_proteins_df(cat)
    return set(df.loc[df["Ensembl_Gene_ID"].astype(str).str.startswith("ENSG"), "Ensembl_Gene_ID"])


def surface_protein_evidence():
    """Return the full surface protein DataFrame with categories and sources."""
    return get_data("surface-proteins")


# ---------- Cancer surfaceome (TCSA) ----------
def _cancer_surfaceome_df(min_cancer_types=None):
    df = get_data("cancer-surfaceome")
    if min_cancer_types is not None:
        df = df[df["num_cancer_types"] >= min_cancer_types]
    return df


def cancer_surfaceome_gene_names(min_cancer_types=None):
    """Tumor-specific surface protein gene symbols from TCSA (Lin et al. 2021).

    These are L3-tier (highest stringency) surface proteins with
    tumor-specific overexpression vs normal tissue, excluding
    immune-cell-expressed genes.

    Parameters
    ----------
    min_cancer_types : int or None
        If set, only return genes overexpressed in at least this many
        cancer types.
    """
    return set(_cancer_surfaceome_df(min_cancer_types)["Symbol"])


def cancer_surfaceome_gene_ids(min_cancer_types=None):
    """Tumor-specific surface protein Ensembl gene IDs from TCSA."""
    return set(_cancer_surfaceome_df(min_cancer_types)["Ensembl_Gene_ID"])


def cancer_surfaceome_gene_id_to_name(min_cancer_types=None):
    """Tumor-specific surface protein {Ensembl_Gene_ID: Symbol} dict from TCSA."""
    df = _cancer_surfaceome_df(min_cancer_types)
    return dict(zip(df["Ensembl_Gene_ID"], df["Symbol"]))


def cancer_surfaceome_evidence(min_cancer_types=None):
    """Full TCSA evidence DataFrame with cancer types and druggability."""
    return _cancer_surfaceome_df(min_cancer_types)


# ---------- Pan-cancer expression ----------
#
# The full pan-cancer-expression CSV is ~16k rows × 85 cols, and the
# housekeeping normalization rescales 83 value columns. Recomputing on every
# call is wasteful because ~every purity / ranking / plotting path asks for
# the same `normalize="housekeeping"` snapshot. We cache on a hashable form
# of the args and hand back `.copy()` so callers keep their mutation freedom.
_PAN_CANCER_CACHE = {}


def _pan_cancer_cache_key(genes, normalize, log_transform):
    genes_key = None if genes is None else frozenset(str(g).upper() for g in genes)
    return (genes_key, normalize, bool(log_transform))


def pan_cancer_expression(genes=None, normalize=None, log_transform=False):
    """Expression across 50 normal tissues (nTPM) and 33 TCGA cancer types.

    Normal tissues from HPA v23 consensus nTPM. Cancer types from HPA
    (21 types, median FPKM) and GDC/STAR reprocessing (12 additional types,
    median TPM). Column names are prefixed with ``nTPM_`` or ``FPKM_``.

    Parameters
    ----------
    genes : iterable of str, optional
        Gene symbols or Ensembl IDs to filter to. If None, returns all
        ~3,100 genes in pirlygenes gene sets.
    normalize : str or None
        ``"percentile"`` — within-column percentile ranks (0–100).
        ``"housekeeping"`` — fold over median housekeeping expression.
        ``None`` (default) — raw values.
    log_transform : bool
        If True, apply log2(x + 1) after normalization (or to raw values
        if normalize is None). Recommended for visualization.

    Returns
    -------
    pd.DataFrame
        A defensive copy of an internally-cached frame. Safe to mutate.
    """
    import numpy as np

    cache_key = _pan_cancer_cache_key(genes, normalize, log_transform)
    cached = _PAN_CANCER_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    df = get_data("pan-cancer-expression")
    if genes is not None:
        genes_upper = {str(g).upper() for g in genes}
        mask = (
            df["Ensembl_Gene_ID"].str.upper().isin(genes_upper)
            | df["Symbol"].str.upper().isin(genes_upper)
        )
        df = df[mask]

    value_cols = [c for c in df.columns if c.startswith("nTPM_") or c.startswith("FPKM_")]

    if normalize is not None:
        df = df.copy()
        if normalize == "percentile":
            for col in value_cols:
                vals = df[col].astype(float)
                df[col] = vals.rank(pct=True) * 100
        elif normalize == "housekeeping":
            hk_ids = housekeeping_gene_ids()
            hk_mask = df["Ensembl_Gene_ID"].isin(hk_ids)
            for col in value_cols:
                vals = df[col].astype(float)
                hk_median = vals[hk_mask].median()
                if not (np.isnan(hk_median) or hk_median <= 0):
                    df[col] = vals / hk_median
                else:
                    df[col] = np.nan
        else:
            raise ValueError(f"normalize must be 'percentile', 'housekeeping', or None, got {normalize!r}")

    if log_transform:
        df = df.copy() if normalize is None else df
        for col in value_cols:
            df[col] = np.log2(df[col].astype(float) + 1)

    _PAN_CANCER_CACHE[cache_key] = df
    return df.copy()


def cancer_types():
    """Return list of available TCGA cancer type codes."""
    df = get_data("pan-cancer-expression")
    return sorted(c.replace("FPKM_", "") for c in df.columns if c.startswith("FPKM_"))


def cancer_expression(cancer_type, genes=None):
    """Expression for a single cancer type as a simple gene-level DataFrame.

    Parameters
    ----------
    cancer_type : str
        TCGA code or alias (e.g. ``"PRAD"``, ``"prostate"``).
    genes : iterable of str, optional
        Gene symbols or Ensembl IDs to filter to.

    Returns
    -------
    pd.DataFrame
        Columns: Ensembl_Gene_ID, Symbol, expression (housekeeping-normalized).
    """
    from .plot import resolve_cancer_type
    code = resolve_cancer_type(cancer_type)
    df = pan_cancer_expression(genes=genes, normalize="housekeeping")
    col = f"FPKM_{code}"
    return df[["Ensembl_Gene_ID", "Symbol", col]].rename(columns={col: "expression"})


def top_enriched_per_cancer_type(n=10, min_fold=3.0, min_expression=0.01, disjoint=False):
    """Top N most cancer-type-specific genes vs pan-cancer median.

    Parameters
    ----------
    n : int
        Number of genes per cancer type.
    min_fold : float
        Minimum fold-change over median of other cancer types.
    min_expression : float
        Minimum housekeeping-normalized expression.
    disjoint : bool
        If True, each gene is assigned to only the cancer type where it
        has the highest fold-change. Genes shared across types are removed
        from all but the best-matching type.

    Returns
    -------
    dict[str, list[str]]
        {TCGA_code: [gene_symbol, ...]} sorted by fold-change descending.
    """
    df = pan_cancer_expression(normalize="housekeeping")
    fpkm_cols = [c for c in df.columns if c.startswith("FPKM_")]

    if disjoint:
        import numpy as _np
        # Vectorized: compute fold-change matrix (genes x cancer types)
        expr_matrix = df[fpkm_cols].astype(float)
        symbols = df["Symbol"].values

        # For each cancer column, fold = expr / median(others)
        all_folds = {}
        for col in fpkm_cols:
            code = col.replace("FPKM_", "")
            other_cols = [c for c in fpkm_cols if c != col]
            other_med = expr_matrix[other_cols].median(axis=1)
            fold = (expr_matrix[col] + 0.001) / (other_med + 0.001)
            all_folds[code] = fold.values

        codes = [col.replace("FPKM_", "") for col in fpkm_cols]
        fold_matrix = _np.column_stack([all_folds[c] for c in codes])  # (genes, cancers)
        expr_max = expr_matrix.values  # (genes, cancers)

        # For each gene, find the cancer type with the highest fold-change
        best_cancer_idx = fold_matrix.argmax(axis=1)
        best_fold = fold_matrix.max(axis=1)

        # Filter: must meet fold and expression thresholds in their best cancer
        cancer_genes = {}
        for i in range(len(symbols)):
            ci = best_cancer_idx[i]
            if best_fold[i] >= min_fold and expr_max[i, ci] >= min_expression:
                code = codes[ci]
                cancer_genes.setdefault(code, []).append((symbols[i], best_fold[i]))

        result = {}
        for code, genes in cancer_genes.items():
            genes.sort(key=lambda x: -x[1])
            result[code] = [sym for sym, _ in genes[:n]]
        return result
    else:
        result = {}
        for col in fpkm_cols:
            code = col.replace("FPKM_", "")
            other_cols = [c for c in fpkm_cols if c != col]
            expr = df[col].astype(float)
            other_med = df[other_cols].astype(float).median(axis=1)
            fold = (expr + 0.001) / (other_med + 0.001)
            mask = (expr >= min_expression) & (fold >= min_fold)
            top_idx = fold[mask].nlargest(n).index
            result[code] = df.loc[top_idx, "Symbol"].tolist()
        return result


def cancer_type_gene_sets(cancer_type):
    """Curated gene sets for a specific cancer type, grouped by role.

    Parameters
    ----------
    cancer_type : str
        TCGA code or alias (e.g. ``"PRAD"``, ``"prostate"``).

    Returns
    -------
    dict[str, dict[str, str]]
        {role: {ensembl_id: symbol}} for use as gene_sets in plotting.
        Returns empty dict if no curated genes exist for that cancer type.
    """
    from .plot import resolve_cancer_type
    code = resolve_cancer_type(cancer_type)
    try:
        df = get_data("cancer-type-genes")
    except Exception:
        return {}
    sub = df[df["Cancer_Type"] == code]
    if sub.empty:
        return {}
    result = {}
    for role, group in sub.groupby("Role"):
        result[role] = dict(zip(group["Ensembl_Gene_ID"], group["Symbol"]))
    return result


def cancer_enriched_genes(cancer_type, min_fold=3.0, min_expression=0.01):
    """Genes enriched in a specific cancer type vs the pan-cancer median.

    Parameters
    ----------
    cancer_type : str
        TCGA code or alias (e.g. ``"PRAD"``, ``"prostate"``).
    min_fold : float
        Minimum fold-change over median of all other cancer types.
    min_expression : float
        Minimum housekeeping-normalized expression in the target cancer type.

    Returns
    -------
    pd.DataFrame
        Columns: Ensembl_Gene_ID, Symbol, expression, other_median, fold_change.
        Sorted by fold_change descending.
    """
    from .plot import resolve_cancer_type
    code = resolve_cancer_type(cancer_type)
    df = pan_cancer_expression(normalize="housekeeping")
    fpkm_cols = [c for c in df.columns if c.startswith("FPKM_")]
    target_col = f"FPKM_{code}"
    other_cols = [c for c in fpkm_cols if c != target_col]

    result = df[["Ensembl_Gene_ID", "Symbol"]].copy()
    result["expression"] = df[target_col].astype(float)
    result["other_median"] = df[other_cols].astype(float).median(axis=1)
    result["fold_change"] = (result["expression"] + 0.001) / (result["other_median"] + 0.001)

    result = result[
        (result["expression"] >= min_expression) &
        (result["fold_change"] >= min_fold)
    ].sort_values("fold_change", ascending=False)
    return result.reset_index(drop=True)


# ---------- Multispecific T-cell Engagers (custom filtering) ----------

def _tce_filtered_df(pmhc=None):
    """Return TCE trial rows, optionally filtered by format.

    Parameters
    ----------
    pmhc : bool or None
        True  = only pMHC-targeting TCEs (TCR-based formats)
        False = only surface-antigen-targeting TCEs (antibody-based formats)
        None  = all TCEs (trials + approved pMHC TCEs)
    """
    import pandas as pd

    df = get_data("multispecific-tcell-engager-trials")
    if pmhc is not False:
        df_approved = get_data("bispecific-antibodies-approved")
        if "Format" in df_approved.columns:
            is_tcr_approved = df_approved["Format"].str.contains(
                "TCR|ImmTAC", case=False, na=False
            )
            df = pd.concat([df, df_approved[is_tcr_approved]], ignore_index=True)

    if pmhc is not None and "Format" in df.columns:
        is_tcr = df["Format"].str.contains("TCR|ImmTAC", case=False, na=False)
        df = df[is_tcr] if pmhc else df[~is_tcr]
    return df


def _extract_genes_from_df(df, candidate_columns):
    result = set()
    for column in candidate_columns:
        if column in df.columns:
            for x in df[column]:
                if type(x) is str:
                    result.update([xi.strip() for xi in x.split(";")])
    return result


_NAME_COLS = ["Tumor_Target_Symbols", "Tumor_Target_Symbol", "Symbol", "Gene_Name"]
_ID_COLS = [
    "Tumor_Target_Gene_IDs", "Tumor_Target_Ensembl_Gene_IDs",
    "Tumor_Target_Ensembl_Gene_ID", "Ensembl_Gene_ID", "Gene_ID",
]


def pMHC_TCE_target_gene_names():
    """pMHC-targeting TCE gene symbols (TCR-based: ImmTAC, TCR-scFv)."""
    return _extract_genes_from_df(_tce_filtered_df(pmhc=True), _NAME_COLS)


def pMHC_TCE_target_gene_ids():
    """pMHC-targeting TCE Ensembl gene IDs."""
    return _extract_genes_from_df(_tce_filtered_df(pmhc=True), _ID_COLS)


def surface_TCE_target_gene_names():
    """Surface-antigen-targeting TCE gene symbols (antibody-based bispecifics)."""
    return _extract_genes_from_df(_tce_filtered_df(pmhc=False), _NAME_COLS)


def surface_TCE_target_gene_ids():
    """Surface-antigen-targeting TCE Ensembl gene IDs."""
    return _extract_genes_from_df(_tce_filtered_df(pmhc=False), _ID_COLS)


def _tce_gene_id_to_name(pmhc):
    """Paired {Ensembl_Gene_ID: Symbol} dict from TCE data."""
    df = _tce_filtered_df(pmhc=pmhc)
    sym_col = next((c for c in _NAME_COLS if c in df.columns), None)
    id_col = next((c for c in _ID_COLS if c in df.columns), None)
    if sym_col is None or id_col is None:
        names = _extract_genes_from_df(df, _NAME_COLS)
        ids = _extract_genes_from_df(df, _ID_COLS)
        return {gid: gid for gid in ids} if ids else {n: n for n in names}
    result = {}
    for _, row in df.iterrows():
        syms_raw = str(row[sym_col]).strip() if row[sym_col] is not None else ""
        ids_raw = str(row[id_col]).strip() if row[id_col] is not None else ""
        if not syms_raw or syms_raw.lower() in ("nan", "none"):
            continue
        syms = [s.strip() for s in syms_raw.split(";") if s.strip()]
        ids = [i.strip() for i in ids_raw.split(";") if i.strip() and i.strip().startswith("ENSG")]
        for i, sym in enumerate(syms):
            if i < len(ids):
                result[ids[i]] = sym
    return result


def pMHC_TCE_target_gene_id_to_name():
    """pMHC-targeting TCE: {Ensembl_Gene_ID: Symbol}."""
    return _tce_gene_id_to_name(pmhc=True)


def surface_TCE_target_gene_id_to_name():
    """Surface-antigen-targeting TCE: {Ensembl_Gene_ID: Symbol}."""
    return _tce_gene_id_to_name(pmhc=False)


# ---------- Deprecated therapy wrappers ----------
# Use therapy_target_gene_names/ids/id_to_name() instead.

def _deprecate(old_name, therapy, ret):
    def fn():
        warnings.warn(
            f"{old_name}() is deprecated, use therapy_target_gene_{ret}('{therapy}')",
            DeprecationWarning, stacklevel=2,
        )
        return {"names": therapy_target_gene_names, "ids": therapy_target_gene_ids}[ret](therapy)
    fn.__name__ = old_name
    fn.__doc__ = f"Deprecated. Use therapy_target_gene_{ret}('{therapy}')."
    return fn


ADC_trial_target_gene_names = _deprecate("ADC_trial_target_gene_names", "ADC-trials", "names")
ADC_trial_target_gene_ids = _deprecate("ADC_trial_target_gene_ids", "ADC-trials", "ids")
ADC_approved_target_gene_names = _deprecate("ADC_approved_target_gene_names", "ADC-approved", "names")
ADC_approved_target_gene_ids = _deprecate("ADC_approved_target_gene_ids", "ADC-approved", "ids")
ADC_target_gene_names = _deprecate("ADC_target_gene_names", "ADC", "names")
ADC_target_gene_ids = _deprecate("ADC_target_gene_ids", "ADC", "ids")
TCR_T_trial_target_get_names = _deprecate("TCR_T_trial_target_get_names", "TCR-T-trials", "names")
TCR_T_trial_target_get_ids = _deprecate("TCR_T_trial_target_get_ids", "TCR-T-trials", "ids")
TCR_T_approved_target_gene_names = _deprecate("TCR_T_approved_target_gene_names", "TCR-T-approved", "names")
TCR_T_approved_target_gene_ids = _deprecate("TCR_T_approved_target_gene_ids", "TCR-T-approved", "ids")
TCR_T_target_gene_names = _deprecate("TCR_T_target_gene_names", "TCR-T", "names")
TCR_T_target_gene_ids = _deprecate("TCR_T_target_gene_ids", "TCR-T", "ids")
CAR_T_approved_target_gene_names = _deprecate("CAR_T_approved_target_gene_names", "CAR-T", "names")
CAR_T_approved_target_gene_ids = _deprecate("CAR_T_approved_target_gene_ids", "CAR-T", "ids")
CAR_T_target_gene_names = _deprecate("CAR_T_target_gene_names", "CAR-T", "names")
CAR_T_target_gene_ids = _deprecate("CAR_T_target_gene_ids", "CAR-T", "ids")
multispecific_tcell_engager_trial_target_gene_names = _deprecate("multispecific_tcell_engager_trial_target_gene_names", "multispecific-TCE", "names")
multispecific_tcell_engager_trial_target_gene_ids = _deprecate("multispecific_tcell_engager_trial_target_gene_ids", "multispecific-TCE", "ids")
multispecific_tcell_engager_target_gene_names = _deprecate("multispecific_tcell_engager_target_gene_names", "multispecific-TCE", "names")
multispecific_tcell_engager_target_gene_ids = _deprecate("multispecific_tcell_engager_target_gene_ids", "multispecific-TCE", "ids")
bispecific_antibody_approved_target_gene_names = _deprecate("bispecific_antibody_approved_target_gene_names", "bispecific-antibodies", "names")
bispecific_antibody_approved_target_gene_ids = _deprecate("bispecific_antibody_approved_target_gene_ids", "bispecific-antibodies", "ids")
bispecific_antibody_target_gene_names = _deprecate("bispecific_antibody_target_gene_names", "bispecific-antibodies", "names")
bispecific_antibody_targets_gene_ids = _deprecate("bispecific_antibody_targets_gene_ids", "bispecific-antibodies", "ids")
radio_target_gene_names = _deprecate("radio_target_gene_names", "radioligand", "names")
radio_target_gene_ids = _deprecate("radio_target_gene_ids", "radioligand", "ids")
radioligand_target_gene_names = _deprecate("radioligand_target_gene_names", "radioligand", "names")
radioligand_target_gene_ids = _deprecate("radioligand_target_gene_ids", "radioligand", "ids")


# ---------- Cancer-testis antigens (CTA) ----------
def _cta_df(filtered_only=False, exclude_never_expressed=False):
    try:
        from tsarina.evidence import CTA_evidence

        df = CTA_evidence()
    except ImportError:
        from .load_dataset import get_data

        df = get_data("cancer-testis-antigens")
    mask = True
    if filtered_only and "filtered" in df.columns:
        mask = df["filtered"].astype(str).str.lower() == "true"
    if exclude_never_expressed and "never_expressed" in df.columns:
        mask = mask & ~(df["never_expressed"].astype(str).str.lower() == "true")
    return df[mask] if not isinstance(mask, bool) else df


def _cta_by_column(column, filtered_only=False, exclude_never_expressed=False):
    subset = _cta_df(
        filtered_only=filtered_only,
        exclude_never_expressed=exclude_never_expressed,
    )
    result = set()
    if column in subset.columns:
        for x in subset[column]:
            if isinstance(x, str):
                result.update([xi.strip() for xi in x.split(";")])
    return result


def CTA_gene_names():
    """CTA gene symbols: filtered AND expressed (>= 2 nTPM somewhere).

    This is the recommended default for pMHC discovery. Excludes
    never-expressed CTAs (no protein data + max RNA < 2 nTPM).
    For the full filtered set including never-expressed, use
    ``CTA_filtered_gene_names()``.
    """
    return _cta_by_column("Symbol", filtered_only=True, exclude_never_expressed=True)


def CTA_gene_ids():
    """CTA Ensembl gene IDs: filtered AND expressed."""
    return _cta_by_column("Ensembl_Gene_ID", filtered_only=True, exclude_never_expressed=True)


def CTA_gene_id_to_name():
    """CTA {Ensembl_Gene_ID: Symbol} dict: filtered AND expressed."""
    subset = _cta_df(filtered_only=True, exclude_never_expressed=True)
    if "Ensembl_Gene_ID" not in subset.columns or "Symbol" not in subset.columns:
        return {}

    result = {}
    for _, row in subset[["Ensembl_Gene_ID", "Symbol"]].dropna().iterrows():
        gene_id = str(row["Ensembl_Gene_ID"]).strip()
        symbol = str(row["Symbol"]).strip()
        if not gene_id or gene_id.lower() in {"nan", "none"}:
            continue
        if not symbol or symbol.lower() in {"nan", "none"}:
            continue
        result[gene_id] = symbol
    return result


def CTA_filtered_gene_names():
    """All CTA gene symbols that pass the HPA filter (including never-expressed)."""
    return _cta_by_column("Symbol", filtered_only=True)


def CTA_filtered_gene_ids():
    """All CTA Ensembl gene IDs that pass the HPA filter (including never-expressed)."""
    return _cta_by_column("Ensembl_Gene_ID", filtered_only=True)


def CTA_never_expressed_gene_names():
    """CTA genes that pass filter but have no meaningful HPA expression.

    No protein data AND max RNA nTPM < 2. These are in source databases
    but lack positive evidence of tissue restriction from HPA.
    """
    return CTA_filtered_gene_names() - CTA_gene_names()


def CTA_never_expressed_gene_ids():
    """CTA Ensembl gene IDs that pass filter but have no meaningful HPA expression."""
    return CTA_filtered_gene_ids() - CTA_gene_ids()


def CTA_unfiltered_gene_names():
    """All CTA gene symbols from all source databases (unfiltered).

    This is the full CTA universe — use for excluding CTA genes from
    a non-CTA comparison set.  Any gene in this set was identified as
    a candidate CTA by at least one source database.
    """
    return get_target_gene_name_set("cancer-testis-antigens")


def CTA_unfiltered_gene_ids():
    """All CTA Ensembl gene IDs from all source databases (unfiltered)."""
    return get_target_gene_id_set("cancer-testis-antigens")


def CTA_excluded_gene_names():
    """CTA genes that FAIL the reproductive-tissue filter.

    These are candidate CTAs with evidence of somatic tissue expression.
    Use this set to exclude from a non-CTA comparison set: they should
    not be in the clean CTA set (they leak into healthy tissue) but also
    should not be in a non-CTA set (they are still CTA candidates).
    """
    return CTA_unfiltered_gene_names() - CTA_filtered_gene_names()


def CTA_excluded_gene_ids():
    """CTA Ensembl gene IDs that FAIL the reproductive-tissue filter."""
    return CTA_unfiltered_gene_ids() - CTA_filtered_gene_ids()


def CTA_evidence():
    """Return the full CTA evidence DataFrame with HPA tissue-restriction columns.

    Columns
    -------
    Symbol, Aliases, Full_Name, Function, Ensembl_Gene_ID,
    source_databases, biotype, Canonical_Transcript_ID
        Gene identity fields.
    protein_reproductive : bool or "no data"
        True if all IHC-detected tissues (excluding thymus) are in
        {testis, ovary, placenta}.
    protein_thymus : bool or "no data"
        True if protein detected in thymus.
    rna_reproductive : bool
        True if every tissue with >=1 nTPM (excluding thymus) is in
        {testis, ovary, placenta}.
    rna_thymus : bool
        True if thymus nTPM >= 1.
    protein_reliability : str
        Best HPA antibody reliability for this gene: "Enhanced",
        "Supported", "Approved", "Uncertain", or "no data".
    protein_strict_expression : str
        Semicolon-separated list of tissues where protein is detected
        (excluding thymus), or "no data" / "not detected".
    rna_reproductive_frac : float
        Fraction of total nTPM (excluding thymus) in core reproductive
        tissues, computed from raw nTPM values.
    rna_reproductive_and_thymus_frac : float
        Same but with thymus nTPM added to numerator and denominator.
    rna_deflated_reproductive_frac : float
        (1 + repro_deflated) / (1 + total_deflated) where each tissue
        is deflated via max(0, nTPM - 1).  The +1 pseudocount prevents
        0/0 for very-low-expression genes.
    rna_deflated_reproductive_and_thymus_frac : float
        Same but with thymus deflated nTPM added to the reproductive
        numerator.
    rna_80_pct_filter, rna_90_pct_filter, rna_95_pct_filter : bool
        Whether deflated reproductive fraction >= 80/90/95%.
    filtered : bool
        Final inclusion flag with tiered RNA thresholds based on protein
        antibody reliability.  True when protein is reproductive-only
        (or absent) and deflated RNA fraction meets the tier threshold:
        - Enhanced → RNA >=80%
        - Supported → RNA >=90%
        - Approved → RNA >=95%
        - Uncertain or no protein data → RNA >=99%
        Genes with protein in non-reproductive tissues always fail.
        Non-protein-coding genes (biotype != protein_coding) always fail.
    rna_max_ntpm : float
        Maximum nTPM across all tissues for this gene.
    never_expressed : bool
        True if no HPA protein data AND maximum RNA nTPM < 2.
    """
    try:
        from tsarina.evidence import CTA_evidence as _tsarina_evidence

        return _tsarina_evidence()
    except ImportError:
        from .load_dataset import get_data

        return get_data("cancer-testis-antigens")


from dataclasses import dataclass  # noqa: E402

import pandas as pd  # noqa: E402


@dataclass(frozen=True)
class CTAPartitionSets:
    """Three-way partition of protein-coding genes as sets.

    Attributes
    ----------
    cta : set[str]
        Expressed, reproductive-restricted CTAs. Source of CTA pMHCs.
    cta_never_expressed : set[str]
        CTAs from source databases with no meaningful HPA expression
        (no protein data + max RNA < 2 nTPM).
    non_cta : set[str]
        All other protein-coding genes, including CTAs that fail the
        reproductive-tissue filter (somatic expression detected).
    """

    cta: set
    cta_never_expressed: set
    non_cta: set


@dataclass(frozen=True)
class CTAPartitionDataFrames:
    """Three-way partition of protein-coding genes as DataFrames.

    Attributes
    ----------
    cta : pd.DataFrame
        Expressed, reproductive-restricted CTAs with full evidence columns.
    cta_never_expressed : pd.DataFrame
        Never-expressed CTAs with full evidence columns.
    non_cta : pd.DataFrame
        All other protein-coding genes (Symbol, Ensembl_Gene_ID).
    """

    cta: pd.DataFrame
    cta_never_expressed: pd.DataFrame
    non_cta: pd.DataFrame


def _build_partition(ensembl_release=112):
    """Shared logic for building the three-way partition."""
    from pyensembl import EnsemblRelease

    ensembl = EnsemblRelease(ensembl_release)
    evidence_df = CTA_evidence()

    all_pc_genes = {
        g.gene_id: g.gene_name
        for g in ensembl.genes()
        if g.biotype == "protein_coding"
    }
    all_pc_ids = set(all_pc_genes.keys())

    filtered_mask = evidence_df["filtered"].astype(str).str.lower() == "true"
    never_expr_mask = evidence_df["never_expressed"].astype(str).str.lower() == "true"

    cta_mask = filtered_mask & ~never_expr_mask
    never_expressed_mask = filtered_mask & never_expr_mask

    cta_ids = set(evidence_df.loc[cta_mask, "Ensembl_Gene_ID"])
    never_expressed_ids = set(evidence_df.loc[never_expressed_mask, "Ensembl_Gene_ID"])
    non_cta_ids = all_pc_ids - cta_ids - never_expressed_ids

    return all_pc_genes, evidence_df, cta_mask, never_expressed_mask, cta_ids, never_expressed_ids, non_cta_ids


def CTA_partition_gene_ids(ensembl_release=112) -> CTAPartitionSets:
    """Partition all protein-coding genes into CTA / never-expressed / non-CTA
    as sets of Ensembl gene IDs.

    CTAs that fail the reproductive-tissue filter (somatic expression)
    are included in ``non_cta``.

    Examples
    --------
    >>> p = CTA_partition_gene_ids()
    >>> "ENSG00000147381" in p.cta   # MAGEA4
    True
    >>> len(p.cta & p.non_cta)       # no overlap
    0
    """
    _, _, _, _, cta_ids, never_expressed_ids, non_cta_ids = _build_partition(ensembl_release)
    return CTAPartitionSets(
        cta=cta_ids,
        cta_never_expressed=never_expressed_ids,
        non_cta=non_cta_ids,
    )


def CTA_partition_gene_names(ensembl_release=112) -> CTAPartitionSets:
    """Partition all protein-coding genes into CTA / never-expressed / non-CTA
    as sets of gene symbols.

    CTAs that fail the reproductive-tissue filter (somatic expression)
    are included in ``non_cta``.

    Examples
    --------
    >>> p = CTA_partition_gene_names()
    >>> "MAGEA4" in p.cta
    True
    >>> "TP53" in p.non_cta
    True
    """
    all_pc_genes, evidence_df, cta_mask, never_expressed_mask, _, _, _ = _build_partition(ensembl_release)
    all_pc_names = set(all_pc_genes.values())

    cta_names = set(evidence_df.loc[cta_mask, "Symbol"])
    never_expressed_names = set(evidence_df.loc[never_expressed_mask, "Symbol"])
    non_cta_names = all_pc_names - cta_names - never_expressed_names

    return CTAPartitionSets(
        cta=cta_names,
        cta_never_expressed=never_expressed_names,
        non_cta=non_cta_names,
    )


def CTA_partition_dataframes(ensembl_release=112) -> CTAPartitionDataFrames:
    """Partition all protein-coding genes into CTA / never-expressed / non-CTA
    as DataFrames.

    The ``cta`` and ``cta_never_expressed`` DataFrames include all CTA
    evidence columns. The ``non_cta`` DataFrame has Symbol and
    Ensembl_Gene_ID columns.

    CTAs that fail the reproductive-tissue filter (somatic expression)
    are included in ``non_cta``.

    Examples
    --------
    >>> p = CTA_partition_dataframes()
    >>> "rna_deflated_reproductive_frac" in p.cta.columns
    True
    """
    all_pc_genes, evidence_df, cta_mask, never_expressed_mask, _, _, non_cta_ids = _build_partition(ensembl_release)

    non_cta_records = [
        {"Symbol": all_pc_genes[gid], "Ensembl_Gene_ID": gid}
        for gid in sorted(non_cta_ids)
        if gid in all_pc_genes
    ]

    return CTAPartitionDataFrames(
        cta=evidence_df.loc[cta_mask].copy().reset_index(drop=True),
        cta_never_expressed=evidence_df.loc[never_expressed_mask].copy().reset_index(drop=True),
        non_cta=pd.DataFrame(non_cta_records),
    )


def CTA_partition(return_type="gene_ids", ensembl_release=112):
    """Deprecated — use CTA_partition_gene_ids, CTA_partition_gene_names,
    or CTA_partition_dataframes instead."""
    if return_type == "gene_ids":
        return CTA_partition_gene_ids(ensembl_release)
    elif return_type == "gene_names":
        return CTA_partition_gene_names(ensembl_release)
    elif return_type == "dataframes":
        return CTA_partition_dataframes(ensembl_release)
    else:
        raise ValueError(f"return_type must be 'gene_ids', 'gene_names', or 'dataframes', got {return_type!r}")


# ---------- Cancer-type key genes (biomarkers + therapy targets) #110 ----------
def cancer_key_genes_df():
    """Full curated table of clinician-relevant biomarker and therapy-
    target genes per cancer type (#110).

    One row per (cancer_code, symbol, role, agent?) — so genes with
    multiple approved agents appear as multiple rows. Columns:
    ``cancer_code, subtype, symbol, role (biomarker|target), agent,
    agent_class, phase (approved|phase_3|phase_2|phase_1|preclinical),
    indication, rationale, source``.

    The curation bar is "genes a clinician would ask about because they
    have clear prognostic value or gate access to an active therapy."
    Some cancer types have no well-validated targets and are omitted
    from the CSV rather than padded — an empty lookup is a valid
    result.
    """
    from .load_dataset import get_data

    return get_data("cancer-key-genes")


def cancer_biomarker_genes(cancer_code, subtype=None):
    """Ordered list of biomarker gene symbols for ``cancer_code``.

    Empty list when the cancer type is not yet curated — callers should
    treat that as "no clinician-relevant biomarker panel available" and
    render an appropriate placeholder, not synthesize fallback genes.

    ``subtype`` (optional) filters to a specific subtype string (e.g.
    ``"synovial_sarcoma"`` under ``cancer_code="SARC"``, #126). When
    ``None``, returns **all** biomarker rows for the cancer code
    regardless of subtype — suitable for samples whose subtype isn't
    yet determined at report time.
    """
    df = cancer_key_genes_df()
    sub = df[(df["cancer_code"] == cancer_code) & (df["role"] == "biomarker")]
    if subtype is not None:
        sub = sub[sub["subtype"].fillna("").astype(str) == subtype]
    return list(sub["symbol"].dropna().astype(str).unique())


def cancer_therapy_targets(cancer_code, subtype=None):
    """Subset of :func:`cancer_key_genes_df` for therapy-target rows of
    ``cancer_code``. Returns a DataFrame so callers can render the
    Phase / Indication / Agent columns without re-joining.

    ``subtype`` (optional) filters to a specific subtype; see
    :func:`cancer_biomarker_genes` for semantics.
    """
    df = cancer_key_genes_df()
    sub = df[(df["cancer_code"] == cancer_code) & (df["role"] == "target")]
    if subtype is not None:
        sub = sub[sub["subtype"].fillna("").astype(str) == subtype]
    return sub.copy().reset_index(drop=True)


def cancer_key_genes_cancer_types():
    """Return the set of cancer codes currently curated in the CSV.
    Use to show a follow-up banner for non-covered cancer types.
    """
    df = cancer_key_genes_df()
    return sorted(df["cancer_code"].dropna().astype(str).unique())


def cancer_key_genes_subtypes(cancer_code):
    """Return the list of curated subtypes for ``cancer_code``.

    SARC is the primary user — subtype-stratified for leiomyosarcoma
    / WD-DDLPS / MRCLS / synovial_sarcoma / DSRCT / GIST /
    ewing_sarcoma (#126). Returns ``[]`` for cancer codes without
    subtype rows (curation is a single flat panel).
    """
    df = cancer_key_genes_df()
    sub = df[df["cancer_code"] == cancer_code]
    subtypes = (
        sub["subtype"].fillna("").astype(str).replace("nan", "").str.strip()
    )
    return sorted(s for s in set(subtypes) if s)
