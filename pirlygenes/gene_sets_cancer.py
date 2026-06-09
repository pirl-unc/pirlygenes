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

from __future__ import annotations

import threading
import warnings
from functools import lru_cache

from .load_dataset import get_data


# Hand-curated common-name aliases. Keyed by lowercase / underscored
# variant; values are canonical codes from cancer-type-registry.csv.
# The registry CSV is the source-of-truth for valid codes and their
# display names — see :data:`CANCER_TYPE_NAMES` below.
CANCER_TYPE_ALIASES = {
    "prostate": "PRAD", "breast": "BRCA", "lung_adeno": "LUAD",
    "lung_squamous": "LUSC", "melanoma": "SKCM", "skin": "SKCM",
    "colon": "COAD", "colorectal": "COAD", "rectal": "READ",
    "pancreatic": "PAAD", "pancreas": "PAAD", "liver": "LIHC",
    "kidney_clear": "KIRC", "kidney_papillary": "KIRP",
    "kidney_chromophobe": "KICH", "kidney": "KIRC", "ovarian": "OV",
    "ovary": "OV", "cervical": "CESC", "cervix": "CESC",
    "bladder": "BLCA", "stomach": "STAD", "gastric": "STAD",
    "glioblastoma": "GBM", "gbm": "GBM", "head_neck": "HNSC",
    "hnscc": "HNSC", "thyroid": "THCA", "endometrial": "UCEC",
    "uterine": "UCEC", "testicular": "TGCT", "testis": "TGCT",
    "sarcoma": "SARC", "adrenocortical": "ACC", "adrenal": "ACC",
    "cholangiocarcinoma": "CHOL", "bile_duct": "CHOL", "dlbcl": "DLBC",
    "lymphoma": "DLBC", "esophageal": "ESCA", "esophagus": "ESCA",
    "aml": "LAML", "leukemia": "LAML", "low_grade_glioma": "LGG",
    "lgg": "LGG", "glioma": "LGG", "mesothelioma": "MESO",
    "pheochromocytoma": "PCPG", "paraganglioma": "PCPG",
    "thymoma": "THYM", "uterine_carcinosarcoma": "UCS",
    "uveal_melanoma": "UVM",
}

# Backward-compat aliases for the Phase-C code renames (old code -> new code).
# resolve_cancer_type consults these so trufflepig / external callers keep
# working without a literal migration. Keep entries here permanently for every
# rename wave (the audit doc promises old codes never hard-break).
_RENAMED_CODE_ALIASES = {
    "OS": "SARC_OS", "EWS": "SARC_EWS", "CHON": "SARC_CHON", "CHOR": "SARC_CHOR",
    "GCTB": "SARC_GCTB", "ESS_LG": "SARC_ESS_LG", "ESS_HG": "SARC_ESS_HG",
    "RMS_ERMS": "SARC_RMS_ERMS", "RMS_ARMS": "SARC_RMS_ARMS",
    "RMS_PRMS": "SARC_RMS_PRMS", "RMS_SSRMS": "SARC_RMS_SSRMS",
    "HNSC_HPV_pos": "HNSC_HPVpos", "HNSC_HPV_neg": "HNSC_HPVneg",
    # #288 neuroendocrine wave: NET_ (well-diff) / NEC_ (poorly-diff) scheme.
    "MID_NET": "NET_MIDGUT", "REC_NET": "NET_RECTAL", "LUNG_NET_LC": "NET_LUNG",
    "MEC": "NEC_MERKEL",
    # #288 follow-up: full NET_<site> consistency + spell out the ambiguous "LC"
    # (lung-carcinoid vs large-cell). Chain the pre-5.16 codes to the final name.
    "PANNET": "NET_PANCREAS",
    "LUNG_NET_LCNEC": "NEC_LUNG_LARGECELL", "NEC_LUNG_LC": "NEC_LUNG_LARGECELL",
    # #288 one-separator normalization (subtype token has no internal underscore)
    "NBL_MYCN_amp": "NBL_MYCNamp", "NBL_MYCN_nonamp": "NBL_MYCNnonamp",
    "LAML_ELN_Fav": "LAML_ELNfav", "LAML_ELN_Int": "LAML_ELNint",
    "LAML_ELN_Adv": "LAML_ELNadv",
}
_RENAMED_CODE_ALIASES_UPPER = {k.upper(): v for k, v in _RENAMED_CODE_ALIASES.items()}


class _CancerTypeNamesView:
    """Read-only ``{code: display_name}`` view backed by the registry CSV.

    Trufflepig and downstream consumers treat ``CANCER_TYPE_NAMES`` as
    a dict (``.get(code)``, ``code in CANCER_TYPE_NAMES``, iteration).
    Loading from the CSV at first access keeps the dict in lock-step
    with the registry — adding a new subtype row to
    ``cancer-type-registry.csv`` automatically extends the dict
    without a code change here.

    The loaded mapping is cached on the instance after the first
    access; downstream callers in trufflepig hit this in tight loops
    (``resolve_cancer_type`` per sample × per candidate). A second
    cached map (``_name_to_code``) holds the lowercased reverse
    lookup used by display-name resolution. Both are protected by a
    ``threading.Lock`` so concurrent first-call paths don't both pay
    the build cost.

    Call ``clear_cache()`` (or the module-level
    :func:`_clear_caches`) to drop the caches — tests that monkey-
    patch ``get_data`` to swap fixture registries need this.
    """

    def __init__(self):
        self._cache: dict | None = None
        self._name_to_code_cache: dict | None = None
        self._lock = threading.Lock()

    def _load(self):
        if self._cache is None:
            with self._lock:
                if self._cache is None:
                    df = get_data("cancer-type-registry")
                    # DataFrame-level filter: drop NaN and
                    # empty/whitespace names before building the dict
                    # so missing values never reach ``str(NaN) == "nan"``.
                    df = df[df["name"].notna()]
                    df = df[df["name"].astype(str).str.strip().ne("")]
                    self._cache = dict(
                        zip(df["code"].astype(str), df["name"].astype(str))
                    )
        return self._cache

    def _name_to_code(self):
        """Lowercased ``display_name → code`` for case-insensitive
        display-name resolution. Cached alongside ``_cache``."""
        if self._name_to_code_cache is None:
            with self._lock:
                if self._name_to_code_cache is None:
                    self._name_to_code_cache = {
                        name.lower(): code for code, name in self._load().items()
                    }
        return self._name_to_code_cache

    def clear_cache(self):
        """Drop the cached dicts. Force a re-read on next access."""
        with self._lock:
            self._cache = None
            self._name_to_code_cache = None

    def __getitem__(self, key):
        return self._load()[key]

    def get(self, key, default=None):
        return self._load().get(key, default)

    def __contains__(self, key):
        return key in self._load()

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def keys(self):
        return self._load().keys()

    def values(self):
        return self._load().values()

    def items(self):
        return self._load().items()

    def __repr__(self):
        return f"_CancerTypeNamesView({len(self._load())} codes)"


# Registry-backed view of cancer code → display name. Reads
# ``cancer-type-registry.csv`` lazily on first access and caches the
# resolved mapping so adding a new subtype row automatically broadens
# ``resolve_cancer_type`` and trufflepig's label lookups without
# repeated CSV-parse / iterrows cost on the hot path.
CANCER_TYPE_NAMES = _CancerTypeNamesView()


def _clear_caches():
    """Reset every registry-backed cache in this module.

    Test hook for swapping the registry CSV via a monkey-patched
    ``get_data``; not part of the public surface.
    """
    CANCER_TYPE_NAMES.clear_cache()
    _cta_protein_group_index.cache_clear()


def resolve_cancer_type(cancer_type, *, strict=True):
    """Resolve a cancer type name or alias to a registry code.

    Accepts:
    - canonical registry codes (``"PRAD"``, ``"SARC_DDLPS"``,
      ``"LAML_APL"``);
    - hand-curated common-name aliases (``"prostate"``, ``"melanoma"``);
    - the registry display name (``"Prostate Adenocarcinoma"``),
      case-insensitive.

    Returns the registry code, or ``None`` if ``cancer_type`` is ``None``.
    For an unknown input: raises ``ValueError`` when ``strict=True`` (default),
    or returns ``None`` when ``strict=False`` (a non-raising lookup for callers
    that want to branch instead of catch).
    """
    if cancer_type is None:
        return None
    raw = str(cancer_type).strip()
    if not raw:
        if strict:
            raise ValueError("Empty cancer type")
        return None

    alias_key = raw.lower().replace(" ", "_").replace("-", "_")
    if alias_key in CANCER_TYPE_ALIASES:
        return CANCER_TYPE_ALIASES[alias_key]

    # Backward-compat: old codes renamed in a Phase-C wave resolve to the new
    # code (case-insensitive), so external callers don't hard-break.
    if raw in _RENAMED_CODE_ALIASES:
        return _RENAMED_CODE_ALIASES[raw]
    if raw.upper() in _RENAMED_CODE_ALIASES_UPPER:
        return _RENAMED_CODE_ALIASES_UPPER[raw.upper()]

    registry = CANCER_TYPE_NAMES  # registry-backed view
    if raw in registry:
        return raw
    upper = raw.upper()
    if upper in registry:
        return upper

    # Display-name lookup (e.g. "Prostate Adenocarcinoma" → "PRAD").
    # The reverse map is cached on the view so this is O(1).
    name_to_code = registry._name_to_code()
    if raw.lower() in name_to_code:
        return name_to_code[raw.lower()]

    if not strict:
        return None
    raise ValueError(
        f"Unknown cancer type {cancer_type!r}. "
        f"Valid registry codes: {sorted(registry.keys())}. "
        f"Common-name aliases: {sorted(CANCER_TYPE_ALIASES.keys())}."
    )


def canonical_cancer_code(code):
    """Map a possibly-renamed cancer code to its canonical current code.

    Pure, registry-free alias lookup over :data:`_RENAMED_CODE_ALIASES`
    (case-insensitive): a pre-rename code like ``"MID_NET"`` or
    ``"PANNET"`` returns its current name (``"NET_MIDGUT"`` /
    ``"NET_PANCREAS"``); any other value — including already-canonical
    codes and non-codes — is returned unchanged. Unlike
    :func:`resolve_cancer_type` this never validates against the registry
    or raises, so the shard-writer can normalize codes on every upsert
    without coupling the expression layer to the registry view.
    """
    if code is None or code != code:  # None or NaN (NaN != NaN)
        return code
    raw = str(code).strip()
    if raw in _RENAMED_CODE_ALIASES:
        return _RENAMED_CODE_ALIASES[raw]
    return _RENAMED_CODE_ALIASES_UPPER.get(raw.upper(), raw)


def format_cancer_code_label(code):
    """Plot-friendly display label for a cancer-type code.

    A trailing ``pos`` / ``neg`` molecular-status suffix becomes a superscript
    ``⁺`` / ``⁻`` (``HNSC_HPVpos`` → ``HNSC_HPV⁺``, ``HNSC_HPVneg`` →
    ``HNSC_HPV⁻``); every other code is returned unchanged. Uses Unicode
    superscript glyphs so it renders in any matplotlib text without mathtext
    escaping."""
    s = str(code)
    if s.endswith("pos"):
        return s[:-3] + "⁺"  # superscript plus
    if s.endswith("neg"):
        return s[:-3] + "⁻"  # superscript minus
    return s


def cancer_type_info(cancer_type):
    """Resolve any synonym/alias/display-name to a cancer type and return its
    **canonical info** as a dict — the one call to go from a messy input to
    everything the registry knows about that type.

    Routes the input through :func:`resolve_cancer_type` (so ``"prostate"``,
    ``"Prostate Adenocarcinoma"``, an old renamed code, or ``"PRAD"`` all
    work), then assembles the registry row plus the derived fields that live
    in their own tables: ``burden_category`` and ``tmb`` (parent-inherited).

    Returns ``None`` if ``cancer_type`` is ``None``; raises ``ValueError`` for
    an unknown input (same contract as :func:`resolve_cancer_type`).

    Keys: ``code``, ``name``, ``family``, ``primary_tissue``,
    ``primary_template``, ``parent_code``, ``subtype_key``, ``pediatric``,
    ``differentiation``, ``expression_source``, ``source_cohort``,
    ``source_pmid``, ``notes``, ``viral_etiology``, ``viral_agent``,
    ``fusion_driven``, ``fusion_driver``, ``burden_category``, ``tmb``.
    """
    import pandas as pd

    code = resolve_cancer_type(cancer_type)
    if code is None:
        return None
    reg = cancer_type_registry().set_index("code")
    row = reg.loc[code] if code in reg.index else None
    info = {"code": code, "name": CANCER_TYPE_NAMES.get(code) or code}
    for col in ("family", "primary_tissue", "primary_template", "parent_code",
                "subtype_key", "pediatric", "differentiation",
                "expression_source", "source_cohort", "source_pmid", "notes",
                "viral_etiology", "viral_agent", "fusion_driven",
                "fusion_driver"):
        val = None if row is None else row.get(col)
        if val is not None and (isinstance(val, str) or not pd.isna(val)):
            # Coerce numpy scalars (e.g. numpy.bool_ for pediatric) to native
            # Python types so the dict is JSON-serializable.
            info[col] = val.item() if hasattr(val, "item") else val
        else:
            info[col] = None
    info["burden_category"] = burden_category(code)
    tmb = cancer_tmb(code)
    info["tmb"] = float(tmb) if tmb is not None else None
    return info


def cancer_type_synonyms(cancer_type):
    """Reverse synonym lookup: every alias that resolves TO a cancer code.

    Returns a sorted list of the common-name aliases (``CANCER_TYPE_ALIASES``),
    registry display name, and pre-rename old codes (``_RENAMED_CODE_ALIASES``)
    that all resolve to the canonical code — the inverse of
    :func:`resolve_cancer_type`. ``[]`` for an unknown input rather than raising.
    """
    try:
        code = resolve_cancer_type(cancer_type)
    except ValueError:
        return []
    if code is None:
        return []
    syns = {a for a, c in CANCER_TYPE_ALIASES.items() if c == code}
    syns |= {o for o, n in _RENAMED_CODE_ALIASES.items() if n == code}
    name = CANCER_TYPE_NAMES.get(code)
    if name:
        syns.add(name)
    syns.discard(code)
    return sorted(syns)


def viral_status(cancer_type):
    """``{'etiology': ..., 'agent': ...}`` for a cancer type.

    ``etiology`` ∈ {``'defining'``, ``'subset'``, ``'none'``} — whether a virus
    defines the entity/subtype (HPV→cervical/HPV+ HNSC, EBV→nasopharyngeal,
    MCPyV→Merkel, HHV8→Kaposi), drives a meaningful subset (EBV→gastric/DLBC,
    HBV/HCV→HCC), or has no established role. ``agent`` names the virus (or
    ``''``). Synonym-resolved; raises ``ValueError`` on unknown input.
    """
    info = cancer_type_info(cancer_type)
    if info is None:
        return None
    return {
        "etiology": info.get("viral_etiology") or "none",
        "agent": info.get("viral_agent") or "",
    }


def fusion_status(cancer_type):
    """``{'status': ..., 'driver': ...}`` for a cancer type.

    ``status`` ∈ {``'defining'``, ``'subtype'``, ``'rare'``, ``'none'``} —
    whether a gene fusion defines the entity (EWSR1-FLI1→Ewing, SS18-SSX→
    synovial), defines a recurrent subtype within it (TMPRSS2-ERG→prostate),
    occurs rarely, or has no established role. ``driver`` lists the canonical
    fusion(s) (sourced from / cross-checked against ``cancer-fusions.csv``).
    Synonym-resolved; raises ``ValueError`` on unknown input.
    """
    info = cancer_type_info(cancer_type)
    if info is None:
        return None
    return {
        "status": info.get("fusion_driven") or "none",
        "driver": info.get("fusion_driver") or "",
    }


def tissue_of_origin(cancer_type):
    """The cancer type's tissue/cell of origin (registry ``primary_tissue``).
    Synonym-resolved; ``None`` for unknown tissue, raises on unknown input."""
    info = cancer_type_info(cancer_type)
    return None if info is None else info.get("primary_tissue")


# Human-readable display names for the registry's ``family`` slugs, so
# consumers (e.g. trufflepig) don't hardcode the labels. See #309.
_FAMILY_DISPLAY_NAMES = {
    "carcinoma-breast": "Breast carcinoma",
    "carcinoma-gi": "Gastrointestinal carcinoma",
    "carcinoma-gu": "Genitourinary carcinoma",
    "carcinoma-head-neck": "Head & neck carcinoma",
    "carcinoma-lung": "Lung carcinoma",
    "carcinoma-mesothelial": "Mesothelioma",
    "carcinoma-other": "Other carcinoma",
    "carcinoma-skin": "Non-melanoma skin carcinoma",
    "cns": "CNS tumor",
    "embryonal": "Embryonal tumor",
    "endocrine": "Endocrine tumor",
    "germ-cell": "Germ cell tumor",
    "heme-bcell": "B-cell neoplasm",
    "heme-myeloid": "Myeloid neoplasm",
    "heme-plasma": "Plasma cell neoplasm",
    "heme-tcell": "T-cell neoplasm",
    "melanoma": "Melanoma",
    "neuroendocrine": "Neuroendocrine neoplasm",
    "salivary": "Salivary gland carcinoma",
    "sarcoma": "Sarcoma",
    "thymic": "Thymic epithelial tumor",
}


def family_display_name(family):
    """Human-readable label for a registry ``family`` slug (e.g.
    ``"heme-bcell"`` -> ``"B-cell neoplasm"``). Falls back to a title-cased
    de-slugged form for any family without a curated label."""
    if family is None:
        return None
    key = str(family).strip()
    if key in _FAMILY_DISPLAY_NAMES:
        return _FAMILY_DISPLAY_NAMES[key]
    return key.replace("-", " ").replace("_", " ").strip().capitalize()


def cancer_type_families():
    """``{family_slug: display_name}`` for every family present in the registry,
    so callers can render a family picker without hardcoding labels (#309)."""
    fams = (
        cancer_type_registry()["family"].dropna().astype(str).unique().tolist()
    )
    return {f: family_display_name(f) for f in sorted(fams)}


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

    sym_candidates = [
        "Tumor_Target_Symbol",
        "Tumor_Target_Symbols",
        "Symbol",
        "Symbols",
        "Gene_Name",
    ]
    id_candidates = [
        "Tumor_Target_Ensembl_Gene_ID",
        "Tumor_Target_Ensembl_Gene_IDs",
        "Ensembl_Gene_ID",
        "Ensembl_Gene_IDs",
        "Ensembl_GeneIDs",
        "Gene_ID",
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
        ids = [
            i.strip()
            for i in ids_raw.split(";")
            if i.strip() and i.strip().startswith("ENSG")
        ]
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
        ``"tRNA"`` — the 22 mitochondrial tRNAs (MT-TA, MT-TC, …).
        ``None`` (default) — all 37 rows.

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
        (
            row["Short_Gene_Symbol"],
            row["Long_Gene_Symbol"],
            float(row["Expected_Long_Over_Short_Ratio"]),
        )
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


# ---------- Cancer lineage panels (parent → child discrimination) ----------
#
# Sibling to ``cancer_family_panels`` but at the child level. Once the
# parent family wins (e.g. SQUAMOUS), the lineage panel for each child
# (BRCA_BASAL vs ESCA vs LUSC vs CESC vs HNSC vs BLCA_BASAL) picks the
# right child cohort using tissue-of-origin markers rather than the
# basal-keratin set every squamous cancer shares.
#
# Schema: Family, Child_Code, Symbol, Ensembl_Gene_ID, Direction
#   Direction is "high" (gene elevated in the child) or "low" (gene
#   depressed — e.g. NKX2-1 low in LUSC discriminates from LUAD).
#
# Curated per GitHub issue #266 and regenerated via
# ``scripts/generate_cancer_lineage_panels.py``.


def cancer_lineage_panels_df(family=None, child_code=None):
    """DataFrame of child-level lineage-discriminating markers.

    ``family`` filters to one parent family (e.g. ``"SQUAMOUS"``);
    ``child_code`` filters to one child cohort (e.g. ``"BRCA_BASAL"``).
    Pass both for the panel of one specific child.
    """
    df = get_data("cancer-lineage-panels")
    if family is not None:
        df = df[df["Family"] == family]
    if child_code is not None:
        df = df[df["Child_Code"] == child_code]
    return df


def cancer_lineage_panel(child_code):
    """List of (Symbol, Direction) tuples for one child cohort.

    Returns ``[("KLK3", "high"), ("KLK2", "high"), …]``. Pair this
    with cohort expression to score a candidate child cohort during
    the second pass of parent-then-child cancer-type scoring.
    """
    df = cancer_lineage_panels_df(child_code=child_code)
    return list(zip(df["Symbol"], df["Direction"]))


def cancer_lineage_panels():
    """Nested dict: ``{family: {child_code: [(Symbol, Direction), …]}}``.

    Use for the second-pass child-resolution step after a parent
    family has won the first-pass cancer-type scoring.
    """
    df = get_data("cancer-lineage-panels")
    out: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for (family, child), group in df.groupby(
        ["Family", "Child_Code"], sort=False,
    ):
        out.setdefault(str(family), {})[str(child)] = list(
            zip(group["Symbol"], group["Direction"])
        )
    return out


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
    ("mitochondrial_chrM", _re.compile(r"MT-.*"), "both"),
    ("cytosolic_ribosomal", _re.compile(r"(RPL|RPS)\d.*"), "both"),
    ("mito_ribosomal", _re.compile(r"(MRPL|MRPS)\d.*"), "both"),
    ("translation_factors", _re.compile(r"(EEF|EIF)\d.*"), "both"),
    ("hnRNPs", _re.compile(r"HNRNP[A-Z]\d?[A-Z]?\d?"), "both"),
    ("splicing_SR_proteins", _re.compile(r"SRSF\d+"), "both"),
    ("snRNPs", _re.compile(r"SNRP[A-Z]\d?"), "both"),
    ("proteasome_subunits", _re.compile(r"PSM[ABCDEF]\d+"), "both"),
    ("cyclophilins", _re.compile(r"PPI[A-Z]\d?"), "both"),
    ("tubulin", _re.compile(r"TUB(A|B|G)\d+[A-Z]?"), "both"),
    ("rearranged_ig_segments", _re.compile(r"(IGH|IGK|IGL)[VDJC]\d.*"), "both"),
    ("rearranged_tcr_segments", _re.compile(r"(TRA|TRB|TRG|TRD)[VDJC]\d.*"), "both"),
]

# Classic housekeeping symbols that don't follow a family regex. Keep
# these as an enumerated allow-list because bare prefix matching would
# catch too much. Each row declares the exclusion scope so we can keep
# biologically-meaningful genes (B2M) in the ranking output.
_EXTENDED_HK_CLASSICS = {
    # symbol: scope
    "ACTB": "both",
    "ACTG1": "both",
    "GAPDH": "both",
    "HPRT1": "both",
    "TPT1": "both",
    "RACK1": "both",
    "B2M": "markers",  # MHC-I context — keep visible in ranking.
    "PGK1": "both",
    "YWHAZ": "both",
    "UBC": "both",
    "PPIA": "both",
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
    return set(
        df.loc[
            df["Ensembl_Gene_ID"].astype(str).str.startswith("ENSG"), "Ensembl_Gene_ID"
        ]
    )


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


# ---------- Cancer-type registry ----------


def cancer_type_registry():
    """Return the cancer-type registry: one row per code with family / tissue / template / parent / source.

    The registry is a richer superset of TCGA — it covers non-TCGA heme
    malignancies (CLL, MM, MCL, FL, HL, MPN, etc.), pediatric cancers
    (OS, RMS, Ewing, NBL, Wilms, RT, MBL, ATRT, RB, HEPB), the
    neuroendocrine axis (panNET, midgut carcinoid, lung NE, SCLC +
    ASCL1/NEUROD1/POU2F3/YAP1 subtypes, Merkel cell), and rare entities
    (NUT carcinoma, adenoid cystic, medullary thyroid, chordoma, NPC).
    Each row carries:

    - ``code`` — canonical short code (TCGA codes preserved, plus new)
    - ``family`` — broad grouping (carcinoma-gi, sarcoma, heme-myeloid …)
    - ``primary_tissue`` — tissue of origin (bone, lymph_node, pancreas …)
    - ``primary_template`` — preferred pirlygenes decomposition template
      (``solid_primary``, ``heme_marrow``, ``primary_bone``, …) — the
      latter bone / cartilage / adipose / muscle variants are planned
      additions (#follow-up) that will let osteosarcoma / chondrosarcoma /
      liposarcoma route to a tissue-appropriate template instead of the
      soft-tissue SARC default
    - ``parent_code`` — if this row is a subtype of another registry
      entry (e.g. ``LAML_APL`` has ``parent_code=LAML``)
    - ``expression_source`` — where to find median expression (TCGA,
      TARGET, BEATAML, CoMMpass, curated, none)
    - ``notes`` — one-line clinical / therapeutic context

    Returns a defensive copy so callers can mutate freely.
    """
    return get_data("cancer-type-registry").copy()


def cancer_types_in_family(family):
    """Return cancer-type codes belonging to a registry family.

    Families are lineage/organ-system only (age is a separate ``pediatric``
    flag): ``sarcoma`` spans soft-tissue + bone + rhabdomyosarcomas regardless
    of patient age; ``neuroendocrine`` spans NET + NEC across organs;
    ``heme-myeloid`` covers LAML + MDS + MPN + CML etc.
    """
    df = cancer_type_registry()
    return df[df["family"] == family]["code"].tolist()


def cancer_types_by_tissue(primary_tissue):
    """Return cancer-type codes whose primary tissue matches.

    Useful for site-aware hypothesis generation — passing ``bone``
    returns osteosarcoma, Ewing, chondrosarcoma, chordoma, etc.
    """
    df = cancer_type_registry()
    return df[df["primary_tissue"] == primary_tissue]["code"].tolist()


def cancer_type_subtypes_of(parent_code):
    """Return registry subtypes of a given parent cancer code.

    For example ``cancer_type_subtypes_of("LAML")`` returns
    ``["LAML_APL", "LAML_ELNfav", "LAML_ELNint", "LAML_ELNadv"]``.
    Lets the second-pass subtype classifier enumerate candidates.
    """
    df = cancer_type_registry()
    return df[df["parent_code"] == parent_code]["code"].tolist()


def mixture_cohort_codes():
    """Return parent codes flagged as mixture cohorts in the registry (#171).

    A mixture cohort is a parent code whose TCGA / reference median is a
    biological union of lineage-distinct subtypes. Running classification
    against the parent median drowns subtype-specific markers; the
    classifier instead evaluates each subtype's lineage panel
    independently and takes the max (see
    :func:`pirlygenes.tumor_purity.estimate_tumor_purity`).

    Example: ``SARC`` = leiomyosarcoma ∪ liposarcoma ∪ myxofibrosarcoma ∪
    undifferentiated pleomorphic ∪ synovial ∪ MPNST. MYH11 is a
    leiomyosarcoma marker but near-zero at TCGA-SARC median because LMS
    is only ~26% of the cohort.
    """
    df = cancer_type_registry()
    if "mixture_cohort" not in df.columns:
        return []
    flag = df["mixture_cohort"].fillna(False).astype(bool)
    return df.loc[flag, "code"].tolist()


def is_mixture_cohort(code):
    """True when ``code`` is a mixture cohort per the registry (#171)."""
    return code in set(mixture_cohort_codes())


def sarcoma_lineage_codes(*, with_expression_only=False):
    """Return every registry code that is a sarcoma by lineage.

    After the Phase-C ontology refactor every sarcoma is in the ``SARC_``
    namespace under one ``family == "sarcoma"`` bucket: soft-tissue
    (``SARC``/``SARC_LMS``/``SARC_UPS``/…), bone (``SARC_OS``, ``SARC_EWS``,
    ``SARC_CHON``, ``SARC_CHOR``, ``SARC_GCTB``), rhabdomyosarcomas
    (``SARC_RMS_*``) and endometrial stromal sarcomas (``SARC_ESS_LG/HG``) —
    bone and pediatric sarcomas were folded in from the retired
    ``pediatric-bone`` / ``pediatric-soft`` families (age is now a separate
    ``pediatric`` flag). This is the membership the ``SARC`` grand-union
    aggregate pools over.

    ``with_expression_only`` drops codes with no per-sample expression source
    (the ``LITERATURE_CURATED`` entries — e.g. ESS_LG/HG, GCTB, SARC_SFT),
    leaving only codes that actually contribute samples to a pooled aggregate.
    """
    df = cancer_type_registry()
    sub = df[df["family"].astype(str) == "sarcoma"]
    if with_expression_only:
        src = sub["expression_source"].astype(str).str.lower()
        sub = sub[~src.isin(["curated", "nan", ""])]
    return sub["code"].tolist()


# Computed cohort aggregates: "view" cohorts that pool the per-sample values of
# several atom cohorts by histology or source, rather than being a single frozen
# matrix. Backed by ``cancer-cohort-aggregates.csv`` ({aggregate_code:
# [member_code,...]}); the pan-sarcoma ``SARC`` grand union is computed from the
# registry family (so it tracks new atoms automatically) rather than enumerated.
def cohort_aggregates_df():
    """Return the curated ``cancer-cohort-aggregates.csv`` long table
    (``aggregate_code, member_code, basis``) — the explicit histology
    rollup cohorts (e.g. ``SARC_RMS`` ← the four rhabdomyosarcoma subtypes;
    ``SARC_LPS`` ← the liposarcoma subtypes)."""
    return get_data("cancer-cohort-aggregates")


# ---------- Cohort vocabulary (first-class registry, #296) ----------
#
# ``cohort-registry.csv`` is the single authority for "what cohorts exist, what
# kind are they, which cancer types draw from them." Unlike
# ``available_cancer_expression_references`` (a per-``(cancer_code,
# source_cohort)`` manifest of *shards*), it also lists the **computed
# aggregates** (``COMPUTED_PAN_SARCOMA``) and the **literature-curated**
# placeholder — so every ``source_cohort`` value anywhere validates against one
# list. Regenerate with ``scripts/generate_cohort_registry.py``.
#
# ``kind`` prefixes (the cross-source combining rule keys off this — see
# :func:`source_prefixed_references`):
#   treehouse  — Treehouse compendium (RSEM); incl. TCGA-reprocessed subsets
#   tcga/target/beataml/mmrf/cllmap/cgci/ucologne — named bulk-RNA-seq projects
#   geo        — GEO/SRA/recount3-sourced series (bulk RNA-seq or microarray)
#   computed   — computed aggregate (no frozen shard; pools member cohorts)
#   curated    — registry entry with no built expression matrix


def cohort_registry_df():
    """The first-class cohort vocabulary (#296): one row per ``cohort_id`` with
    ``prefix, kind, source_project, assay, n_samples, n_codes, is_computed,
    member_cohorts, provenance``. The authority to validate any ``source_cohort``
    against — includes the computed aggregates and literature-curated cohorts
    that are absent from :func:`available_cancer_expression_references`."""
    return get_data("cohort-registry")


def cohort_registry():
    """``{cohort_id: {column: value}}`` view of :func:`cohort_registry_df`."""
    df = cohort_registry_df()
    return {str(r["cohort_id"]): {k: r[k] for k in df.columns if k != "cohort_id"}
            for _, r in df.iterrows()}


def cohort_kind(cohort_id):
    """The ``kind`` (pipeline family) of a cohort_id (``treehouse``, ``geo``,
    ``computed``, …), or ``None`` if unknown. The cross-source combining rule
    keys off this: pool absolute clean-TPM only *within* one kind/cohort; combine
    *across* kinds in rank / z-space (TPM is not comparable across pipelines)."""
    df = cohort_registry_df()
    hit = df.loc[df["cohort_id"].astype(str) == str(cohort_id), "kind"]
    return str(hit.iloc[0]) if len(hit) else None


def known_cohort_ids():
    """Frozenset of every valid ``cohort_id`` (the validation authority)."""
    return frozenset(cohort_registry_df()["cohort_id"].astype(str))


def cohort_aggregates():
    """``{aggregate_code: [member_code, ...]}`` for every computed-aggregate
    cohort: the curated histology rollups (``SARC_RMS``, ``SARC_LPS``) plus the
    pan-sarcoma ``SARC`` grand union — every ``family == 'sarcoma'`` atom that is
    not itself an aggregate (``SARC`` is itself a registry code but resolves to
    the computed union; its TCGA-SARC samples are already folded into the
    histology atoms, so there is no separate frozen ``SARC`` shard)."""
    df = cohort_aggregates_df()
    out = {}
    for agg, grp in df.groupby("aggregate_code"):
        out[str(agg)] = list(dict.fromkeys(grp["member_code"].astype(str)))
    # pan-sarcoma grand union under the bare SARC code, computed from family;
    # exclude the aggregates AND SARC itself (no self-membership / circularity).
    aggs = set(out) | {"SARC"}
    out["SARC"] = [c for c in sarcoma_lineage_codes() if c not in aggs]
    return out


def cohort_aggregate_members(aggregate_code):
    """Member atom codes pooled by a computed-aggregate cohort, or ``None`` if
    ``aggregate_code`` is not an aggregate. ``SARC`` is the pan-sarcoma grand
    union; ``SARC_RMS`` / ``SARC_LPS`` are the curated histology rollups."""
    return cohort_aggregates().get(str(aggregate_code))


# ---------- Cancer-pathway panels (tumor-evidence Step-0 signals) ----------
#
# Each panel is a coordinated-program set — genes that move together in
# tumors vs normal tissue. Empirical median-fold-change numbers below
# are from a pan-cancer TCGA-vs-matched-HPA-normal sweep of 20
# epithelial cancer-tissue pairs. Ship as public API so consumers can
# reuse the same definitions for downstream analysis (scoring,
# signature calibration, cross-cohort comparison).


_PROLIFERATION_PANEL_GENES = (
    # Empirically-selected from pan-cancer sweep: each gene has
    # median fold ≥ 3 across 20 epithelial cancers vs matched normal,
    # and coordinated with the rest of the mitotic program.
    "MKI67",  # median fold 4.0x
    "TOP2A",  # 4.3
    "CCNB1",  # 3.1
    "CCNB2",  # 3.7
    "CDC20",  # 6.1
    "CDK1",  # 2.2
    "UBE2C",  # 5.9
    "TPX2",  # 4.8
    "CENPF",  # 14.3 — strongest single marker
    "FOXM1",  # 6.3
    "PLK1",  # 3.8
    "AURKA",  # 2.1
    "BIRC5",  # 3.8
)


_HYPOXIA_PANEL_GENES = (
    # Carbonic anhydrase 9 is the classic HIF1α-driven hypoxia marker;
    # SLC2A1 (GLUT1) is the Warburg/hypoxia crosslink. Median fold
    # across pan-cancer sweep — CA9 12.1x, SLC2A1 9.1x.
    "CA9",
    "SLC2A1",
    "LDHA",
    "ENO1",
    "PGK1",
)


_GLYCOLYSIS_PANEL_GENES = (
    # Warburg / glycolysis panel. Most individual fold-changes are
    # modest (1-2x over normal) because glycolytic enzymes are high
    # everywhere; the ABSOLUTE tumor TPM is what's striking (PKM 557,
    # ENO1 723, GAPDH 2575 across pan-cancer). SLC2A1 (GLUT1) and
    # ALDOA give the best fold-change contrast.
    "HK2",
    "LDHA",
    "PKM",
    "SLC2A1",
    "ENO1",
    "PGK1",
    "ALDOA",
    "PFKP",
)


_DDR_ACTIVATION_PANEL_GENES = (
    # DNA-damage-response / replication-stress markers. Modest fold-
    # change (1-2x) but co-upregulated under replication stress.
    "RAD51",
    "RAD51AP1",
    "CHEK1",
    "CHEK2",
    "ATR",
    "BRCA1",
)


_ONCOFETAL_STRICT_GENES = (
    # Genes near-zero in any adult somatic tissue, re-expressed in
    # tumors as an oncofetal program. Stricter than the CTA panel —
    # these must pass HPA-check of median nTPM < 1 outside
    # reproductive / trophoblastic cells.
    "AFP",
    "LIN28A",
    "LIN28B",
    "TPBG",  # 5T4
    "PLAC1",
    "CGB",
    "CGB1",
    "CGB2",
    "CGB3",
    "NANOG",
    "POU5F1",  # OCT4
)


def proliferation_panel_gene_names():
    """Coordinated cell-cycle / mitotic-program panel (13 genes).

    Empirically-selected from a pan-cancer sweep: each gene has
    median fold-change ≥ 3 across 20 epithelial cancer-tissue pairs.
    Used as a Step-0 tumor-evidence signal (healthy_vs_tumor) and
    available here for downstream scoring / calibration.
    """
    return list(_PROLIFERATION_PANEL_GENES)


def hypoxia_panel_gene_names():
    """Hypoxia-response panel — CA9, SLC2A1, LDHA, ENO1, PGK1.

    CA9 is the strongest single-gene marker (median fold 12x across
    pan-cancer). Used as a Step-0 tumor-evidence signal and for
    downstream hypoxia-score calculation.
    """
    return list(_HYPOXIA_PANEL_GENES)


def glycolysis_panel_gene_names():
    """Warburg / glycolysis panel — 8 canonical enzymes.

    Modest per-gene fold-change over normal (these are high
    everywhere) but coordinated upregulation is a cancer hallmark.
    Absolute tumor TPMs are dramatic (PKM 557, ENO1 723, GAPDH 2575
    averaged across pan-cancer medians).
    """
    return list(_GLYCOLYSIS_PANEL_GENES)


def ddr_activation_panel_gene_names():
    """DNA-damage-response / replication-stress panel (6 genes)."""
    return list(_DDR_ACTIVATION_PANEL_GENES)


def oncofetal_strict_gene_names():
    """Oncofetal / embryonic-stemness strict panel (11 genes).

    Near-zero in any adult somatic tissue, re-expressed in many
    cancers. Strictest specificity — SOX2 / KLF4 / IGF2 / HMGA2 /
    HLA-G were intentionally excluded from this tier because they
    are physiologically expressed in adult niches (neural
    progenitors, gut stem cells, smooth muscle, etc.) and would
    false-positive as tumor evidence.
    """
    return list(_ONCOFETAL_STRICT_GENES)


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
    "Tumor_Target_Gene_IDs",
    "Tumor_Target_Ensembl_Gene_IDs",
    "Tumor_Target_Ensembl_Gene_ID",
    "Ensembl_Gene_ID",
    "Gene_ID",
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
        ids = [
            i.strip()
            for i in ids_raw.split(";")
            if i.strip() and i.strip().startswith("ENSG")
        ]
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
            DeprecationWarning,
            stacklevel=2,
        )
        return {"names": therapy_target_gene_names, "ids": therapy_target_gene_ids}[
            ret
        ](therapy)

    fn.__name__ = old_name
    fn.__doc__ = f"Deprecated. Use therapy_target_gene_{ret}('{therapy}')."
    return fn


ADC_trial_target_gene_names = _deprecate(
    "ADC_trial_target_gene_names", "ADC-trials", "names"
)
ADC_trial_target_gene_ids = _deprecate("ADC_trial_target_gene_ids", "ADC-trials", "ids")
ADC_approved_target_gene_names = _deprecate(
    "ADC_approved_target_gene_names", "ADC-approved", "names"
)
ADC_approved_target_gene_ids = _deprecate(
    "ADC_approved_target_gene_ids", "ADC-approved", "ids"
)
ADC_target_gene_names = _deprecate("ADC_target_gene_names", "ADC", "names")
ADC_target_gene_ids = _deprecate("ADC_target_gene_ids", "ADC", "ids")
TCR_T_trial_target_get_names = _deprecate(
    "TCR_T_trial_target_get_names", "TCR-T-trials", "names"
)
TCR_T_trial_target_get_ids = _deprecate(
    "TCR_T_trial_target_get_ids", "TCR-T-trials", "ids"
)
TCR_T_approved_target_gene_names = _deprecate(
    "TCR_T_approved_target_gene_names", "TCR-T-approved", "names"
)
TCR_T_approved_target_gene_ids = _deprecate(
    "TCR_T_approved_target_gene_ids", "TCR-T-approved", "ids"
)
TCR_T_target_gene_names = _deprecate("TCR_T_target_gene_names", "TCR-T", "names")
TCR_T_target_gene_ids = _deprecate("TCR_T_target_gene_ids", "TCR-T", "ids")
CAR_T_approved_target_gene_names = _deprecate(
    "CAR_T_approved_target_gene_names", "CAR-T", "names"
)
CAR_T_approved_target_gene_ids = _deprecate(
    "CAR_T_approved_target_gene_ids", "CAR-T", "ids"
)
CAR_T_target_gene_names = _deprecate("CAR_T_target_gene_names", "CAR-T", "names")
CAR_T_target_gene_ids = _deprecate("CAR_T_target_gene_ids", "CAR-T", "ids")
multispecific_tcell_engager_trial_target_gene_names = _deprecate(
    "multispecific_tcell_engager_trial_target_gene_names", "multispecific-TCE", "names"
)
multispecific_tcell_engager_trial_target_gene_ids = _deprecate(
    "multispecific_tcell_engager_trial_target_gene_ids", "multispecific-TCE", "ids"
)
multispecific_tcell_engager_target_gene_names = _deprecate(
    "multispecific_tcell_engager_target_gene_names", "multispecific-TCE", "names"
)
multispecific_tcell_engager_target_gene_ids = _deprecate(
    "multispecific_tcell_engager_target_gene_ids", "multispecific-TCE", "ids"
)
bispecific_antibody_approved_target_gene_names = _deprecate(
    "bispecific_antibody_approved_target_gene_names", "bispecific-antibodies", "names"
)
bispecific_antibody_approved_target_gene_ids = _deprecate(
    "bispecific_antibody_approved_target_gene_ids", "bispecific-antibodies", "ids"
)
bispecific_antibody_target_gene_names = _deprecate(
    "bispecific_antibody_target_gene_names", "bispecific-antibodies", "names"
)
bispecific_antibody_target_gene_ids = _deprecate(
    "bispecific_antibody_target_gene_ids", "bispecific-antibodies", "ids"
)
radio_target_gene_names = _deprecate("radio_target_gene_names", "radioligand", "names")
radio_target_gene_ids = _deprecate("radio_target_gene_ids", "radioligand", "ids")
radioligand_target_gene_names = _deprecate(
    "radioligand_target_gene_names", "radioligand", "names"
)
radioligand_target_gene_ids = _deprecate(
    "radioligand_target_gene_ids", "radioligand", "ids"
)


# ---------- Cancer-testis antigens (CTA) ----------
# CTA curation is owned by tsarina — the single source of truth. pirlygenes
# re-exports tsarina's evidence table, gene-set accessors, and protein-coding
# partition so downstream consumers (trufflepig, analyses, normalize) keep one
# stable import path while the filter / never-expressed / partition logic lives
# in exactly one place. tsarina is a hard dependency (#289/#290/#291); the only
# pirlygenes-local addition is CTA_gene_id_to_name(), a convenience mapping with
# no tsarina equivalent.
from tsarina.evidence import CTA_evidence as _tsarina_CTA_evidence  # noqa: E402
from tsarina.gene_sets import (  # noqa: E402,F401
    CTA_by_axes,
    CTA_excluded_gene_ids,
    CTA_excluded_gene_names,
    CTA_filtered_gene_ids,
    CTA_filtered_gene_names,
    CTA_gene_ids,
    CTA_gene_names,
    CTA_never_expressed_gene_ids,
    CTA_never_expressed_gene_names,
    CTA_placental_restricted_gene_ids,
    CTA_placental_restricted_gene_names,
    CTA_testis_restricted_gene_ids,
    CTA_testis_restricted_gene_names,
    CTA_unfiltered_gene_ids,
    CTA_unfiltered_gene_names,
    cta_symbol_for_alias,
)
from tsarina.partition import (  # noqa: E402,F401
    CTAPartitionDataFrames,
    CTAPartitionSets,
    CTA_partition_dataframes,
    CTA_partition_gene_ids,
    CTA_partition_gene_names,
)


@lru_cache(maxsize=1)
def _cta_protein_group_index():
    """``({protein_group: [member_symbol, ...]}, {member_symbol: protein_group})``
    over ``cta-protein-groups`` — the curated identical / near-identical CTA
    paralog families (e.g. NY-ESO-1 = CTAG1A + CTAG1B). Members are sorted."""
    g = get_data("cta-protein-groups")
    by_group: dict[str, list[str]] = {}
    member_to_group: dict[str, str] = {}
    for grp, sub in g.groupby("protein_group"):
        members = sorted({str(m) for m in sub["member_symbol"]})
        by_group[str(grp)] = members
        for m in members:
            member_to_group[m] = str(grp)
    return by_group, member_to_group


def cta_paralog_symbols(name: str) -> list[str]:
    """Every CTA gene symbol encoding the same antigen as ``name`` (its protein
    group), including ``name`` itself.

    Where :func:`cta_symbol_for_alias` (tsarina) returns the single *canonical*
    symbol for an alias (``"NY-ESO-1" -> "CTAG1B"``), this returns the whole
    interchangeable set from ``cta-protein-groups``:
    ``"NY-ESO-1" -> ["CTAG1A", "CTAG1B"]``, ``"XAGE1" -> ["XAGE1A", "XAGE1B"]``,
    ``"MAGEA3" -> ["MAGEA3", "MAGEA6"]``. A CTA with no paralog group resolves to
    just its own official symbol; an input that is not a recognized CTA returns
    ``[]``. Accepts aliases, official symbols, or protein-group names.
    """
    if not name:
        return []
    by_group, member_to_group = _cta_protein_group_index()
    sym = cta_symbol_for_alias(name)
    key = sym or str(name).strip()
    if key in member_to_group:
        return list(by_group[member_to_group[key]])
    if key in by_group:  # a protein-group name with no member alias
        return list(by_group[key])
    return [sym] if sym else []


def CTA_evidence():
    """Full CTA evidence DataFrame with HPA tissue-restriction columns.

    Delegates to :func:`tsarina.evidence.CTA_evidence` — tsarina is the single
    source of truth for CTA curation. See tsarina for the column semantics
    (``passes_filters`` / ``filtered``, ``never_expressed``,
    ``rna_*_pct_filter``, ``rna_deflated_reproductive_frac``, ``rna_max_ntpm``,
    the per-tissue nTPM columns, …).
    """
    return _tsarina_CTA_evidence()


def CTA_gene_id_to_name():
    """CTA ``{Ensembl_Gene_ID: Symbol}`` for the filtered + expressed set.

    pirlygenes convenience over tsarina's :func:`CTA_gene_ids` /
    :func:`CTA_gene_names`: tsarina exposes the id and symbol sets but not the
    id→symbol mapping. Restricted to the same filtered-and-expressed CTAs as
    :func:`CTA_gene_ids`.
    """
    ids = CTA_gene_ids()
    ev = CTA_evidence()
    result = {}
    for gid, sym in zip(ev["Ensembl_Gene_ID"].astype(str), ev["Symbol"].astype(str)):
        gid = gid.strip()
        sym = sym.strip()
        if gid in ids and sym and sym.lower() not in {"nan", "none"}:
            result[gid] = sym
    return result


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
        raise ValueError(
            f"return_type must be 'gene_ids', 'gene_names', or 'dataframes', got {return_type!r}"
        )


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


THERAPY_BENEFIT_TIERS = (
    "curative",
    "durable_rfs",
    "major_survival",
    "high_response",
    "meaningful_pfs",
    "incremental",
    "modest",
    "unclear",
)

THERAPY_TOXICITY_TIERS = (
    "minimal",
    "low",
    "moderate",
    "high",
    "very_high",
    "unclear",
)

_THERAPY_EVIDENCE_REQUIRED_COLUMNS = (
    "agent",
    "agent_class",
    "target_symbol",
    "cancer_code",
    "subtype",
    "line_of_therapy",
    "setting",
    "endpoint_type",
    "endpoint_value",
    "benefit_tier",
    "toxicity_tier",
    "grade3_plus_ae_rate",
    "discontinuation_rate",
    "boxed_warning",
    "major_toxicities",
    "source_type",
    "source_id",
    "evidence_transfer",
    "evidence_notes",
)


def _nonempty_strings(series):
    return set(series.dropna().astype(str).str.strip()) - {""}


def _filter_text_value(df, column, value):
    if value is None:
        return df
    wanted = str(value).strip().casefold()
    values = df[column].fillna("").astype(str).str.strip().str.casefold()
    return df[values == wanted]


def _validate_therapy_benefit_toxicity_evidence(df):
    missing = set(_THERAPY_EVIDENCE_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            "therapy-benefit-toxicity-evidence.csv is missing columns: "
            f"{sorted(missing)}"
        )

    invalid_benefit = _nonempty_strings(df["benefit_tier"]) - set(
        THERAPY_BENEFIT_TIERS
    )
    if invalid_benefit:
        raise ValueError(f"Invalid benefit_tier values: {sorted(invalid_benefit)}")

    invalid_toxicity = _nonempty_strings(df["toxicity_tier"]) - set(
        THERAPY_TOXICITY_TIERS
    )
    if invalid_toxicity:
        raise ValueError(f"Invalid toxicity_tier values: {sorted(invalid_toxicity)}")

    postmarket = df["source_type"].fillna("").astype(str).str.strip().eq(
        "postmarket_signal"
    )
    postmarket_with_incidence = postmarket & df["grade3_plus_ae_rate"].notna()
    postmarket_with_incidence = postmarket_with_incidence & df[
        "grade3_plus_ae_rate"
    ].astype(str).str.strip().ne("")
    if postmarket_with_incidence.any():
        bad_agents = sorted(df.loc[postmarket_with_incidence, "agent"].astype(str))
        raise ValueError(
            "postmarket_signal rows must not provide incidence-like "
            f"grade3_plus_ae_rate values: {bad_agents}"
        )


def therapy_benefit_toxicity_evidence(
    *,
    agent=None,
    cancer_code=None,
    subtype=None,
    line_of_therapy=None,
    source_type=None,
    include_transferred=True,
):
    """Curated benefit/toxicity evidence rows for therapy ranking.

    The table is keyed by agent with optional disease, subtype, and
    line-of-therapy context. It is deliberately separate from expression
    values: target expression must never be used to infer survival
    benefit or toxicity on its own.

    Parameters filter exact text values case-insensitively. When both
    ``cancer_code`` and ``subtype`` are supplied, parent cancer-code rows
    with blank subtype are kept alongside exact subtype rows because
    those broader rows can still apply to the subtype. Subtype-only
    filtering returns only exact subtype rows. Set
    ``include_transferred=False`` to drop cross-indication rows that
    require an explicit eligibility caveat before use.
    """
    from .load_dataset import get_data

    df = get_data("therapy-benefit-toxicity-evidence")
    _validate_therapy_benefit_toxicity_evidence(df)

    df = _filter_text_value(df, "agent", agent)
    df = _filter_text_value(df, "cancer_code", cancer_code)
    df = _filter_text_value(df, "line_of_therapy", line_of_therapy)
    df = _filter_text_value(df, "source_type", source_type)

    if subtype is not None:
        wanted = str(subtype).strip().casefold()
        values = df["subtype"].fillna("").astype(str).str.strip().str.casefold()
        if cancer_code is None:
            df = df[values == wanted]
        else:
            df = df[(values == "") | (values == wanted)]

    if not include_transferred:
        transfer = (
            df["evidence_transfer"].fillna("").astype(str).str.strip().str.casefold()
        )
        df = df[transfer != "cross_indication"]

    return df.copy().reset_index(drop=True)


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
    subtypes = sub["subtype"].fillna("").astype(str).replace("nan", "").str.strip()
    return sorted(s for s in set(subtypes) if s)


# ── #202 narrative gene sets + #198 degenerate-pair / fusion-surrogate
# public exposure. These CSVs are already auto-loaded by
# ``load_dataset.get_data``; re-exporting the high-level accessors
# here so ``pirlygenes.gene_sets_cancer`` is the single discovery
# surface for every curated gene set bundled with the package.


def _narrative_gene_sets():
    """Return ``{set_name: tuple[str]}`` from ``narrative-gene-sets.csv``."""
    df = get_data("narrative-gene-sets")
    out: dict[str, tuple[str, ...]] = {}
    for _, row in df.iterrows():
        name = str(row.get("set_name") or "").strip()
        members = str(row.get("members") or "")
        parsed = tuple(m.strip() for m in members.split(";") if m.strip())
        if name and parsed:
            out[name] = parsed
    return out


def narrative_gene_sets_df():
    """Return the raw ``narrative-gene-sets.csv`` DataFrame.

    Columns: ``set_name``, ``members`` (``;``-delimited), ``notes``.
    Each row is a named gene set referenced by the disease-state rule
    engine (``AR_targets``, ``HER2_amplicon``, ``NE_markers``, ...).
    Use :func:`narrative_gene_set` to look up members by name.
    """
    return get_data("narrative-gene-sets")


def narrative_gene_set(set_name):
    """Return the tuple of gene symbols in a named narrative set, or
    ``()`` when unknown. Case-sensitive."""
    return _narrative_gene_sets().get(set_name, ())


def narrative_gene_set_names():
    """Return the list of known narrative gene-set names."""
    return sorted(_narrative_gene_sets().keys())


def degenerate_subtype_pairs_df():
    """Return the raw ``degenerate-subtype-pairs.csv`` DataFrame (#198).

    Returned in raw text form — semicolon-delimited members, pipe-delimited
    mapping strings. Analysis-side parsing lives in
    :func:`trufflepig.degenerate_subtype.degenerate_subtype_pairs`.
    """
    return get_data("degenerate-subtype-pairs")


def fusion_surrogate_expression_df():
    """Return the raw ``fusion-surrogate-expression.csv`` DataFrame
    (#198) — genes whose expression serves as a deterministic
    surrogate for a specific fusion/translocation class."""
    return get_data("fusion-surrogate-expression")


def rare_cancer_rna_surrogate_rules_df():
    """Return ``rare-cancer-rna-surrogates.csv``.

    Rows encode hypothesis-level report-scope rules for rare cancers that
    lack a bundled TCGA expression cohort but have a high-specificity RNA
    marker, such as NUTM1 for NUT carcinoma or TBXT for chordoma.
    """
    return get_data("rare-cancer-rna-surrogates")


def rare_cancer_fusion_rules_df():
    """Return ``rare-cancer-fusion-rules.csv`` direct-fusion rules."""
    return get_data("rare-cancer-fusion-rules")


def fusion_expression_effect_rules_df():
    """Return ``fusion-expression-effects.csv`` downstream-expression rules."""
    return get_data("fusion-expression-effects")


def mutation_expression_effect_rules_df():
    """Return ``mutation-expression-effects.csv`` expression-effect rules."""
    return get_data("mutation-expression-effects")


def fusion_surrogate_genes_for_cancer(cancer_code):
    """Return the list of ``{gene, fusion_class, role, rationale}``
    dicts applicable to a cancer code (includes ``pan_cancer``
    entries).

    The data lives in ``fusion-surrogate-expression.csv``. The
    parsing/joining logic moved to
    :func:`trufflepig.degenerate_subtype.fusion_surrogate_genes_for`;
    this thin wrapper is preserved for backwards compatibility.
    """
    df = get_data("fusion-surrogate-expression")
    out = []
    for _, row in df.iterrows():
        scope = str(row.get("cancer_code", "") or "")
        scope_codes = {code.strip() for code in scope.split(";") if code.strip()}
        if cancer_code in scope_codes or "pan_cancer" in scope_codes:
            out.append(
                {
                    "gene": row.get("surrogate_gene"),
                    "fusion_class": row.get("fusion_class"),
                    "role": row.get("surrogate_role"),
                    "rationale": row.get("rationale", ""),
                }
            )
    return out


def disease_state_rules_df():
    """Return the raw ``disease-state-rules.csv`` DataFrame —
    declarative per-cancer narrative rules consumed by trufflepig's
    disease-state narrative composer."""
    return get_data("disease-state-rules")


def cancer_tmb_df():
    """Return the curated ``cancer-tmb.csv`` reference: median tumor
    mutational burden (mut/Mb) per cancer-type code, with a per-row
    published source/PMID and a confidence flag.

    Cohorts with no defensible published per-Mb median are present with a
    blank ``median_tmb_mut_mb`` (and a ``confidence`` of ``none``) so the
    gap is explicit rather than silently absent. Values mix WES-anchored
    medians (Lawrence 2013) with panel-based medians (Chalmers 2017) and
    disease-specific studies; see the ``source``/``notes`` columns — panel
    and WES TMB are not strictly comparable in the low-TMB range."""
    return get_data("cancer-tmb")


def cancer_tmb(cancer_type=None, *, inherit=True):
    """Median TMB (mut/Mb) for one cancer type, or the whole
    ``{code: median_tmb}`` map (codes with no published value omitted).

    ``cancer_type`` is resolved through :func:`resolve_cancer_type`, so aliases
    and display names work. When ``inherit`` (default), a code with no curated
    value of its own inherits its nearest ancestor's TMB by walking the registry
    ``parent_code`` chain — so molecular / histology subtypes (``LUAD_EGFR`` ->
    ``LUAD``, ``SCLC_ASCL1`` -> ``SCLC``, rare ``SARC_*`` -> ``SARC``) resolve
    without a curated row each. Returns ``None`` if neither the code nor any
    ancestor has a value."""
    df = cancer_tmb_df()
    vals = df.dropna(subset=["median_tmb_mut_mb"])
    mapping = dict(zip(vals["cancer_code"].astype(str),
                       vals["median_tmb_mut_mb"].astype(float)))
    if cancer_type is None:
        return mapping
    code = resolve_cancer_type(cancer_type)
    if code in mapping or not inherit:
        return mapping.get(code)
    # walk the registry parent chain to inherit an ancestor's value
    reg = cancer_type_registry().set_index("code")
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in mapping:
            return mapping[cur]
        if cur not in reg.index:
            break
        cur = str(reg.loc[cur].get("parent_code", "") or "").strip() or None
    return None


def cancer_apd1_response_df():
    """Return the curated ``cancer-apd1-response.csv`` reference: representative
    objective response rate (ORR, %) to anti-PD-1 **monotherapy**
    (pembrolizumab / nivolumab) per cancer-type code, with the drug, pivotal
    trial, treatment setting, a published source PMID/DOI, and a confidence
    flag.

    Intended as a per-cancer-type plotting axis (e.g. TMB vs aPD1 ORR, CTA
    burden vs aPD1 ORR). Values are representative anchors, not exact
    reproducible constants — they shift with data cutoff, line of therapy, and
    biomarker selection (PD-L1 / MSI / MMR); the ``setting`` and ``notes``
    columns record that context. Several cancers are strongly biomarker-
    dependent (COAD/READ MSI-H ~45% vs MSS ~0%; UCEC dMMR ~50% vs pMMR ~6%) —
    the row carries the all-comer blend and the split is noted."""
    return get_data("cancer-apd1-response")


def cancer_apd1_response(cancer_type=None, *, inherit=True):
    """Anti-PD-1 monotherapy ORR (%) for one cancer type, or the whole
    ``{code: orr_pct}`` map. ``cancer_type`` is resolved through
    :func:`resolve_cancer_type`; with ``inherit`` (default) a code with no
    curated row of its own inherits its nearest ancestor's value via the
    registry ``parent_code`` chain (so ``SCLC_ASCL1`` -> ``SCLC``,
    ``LUAD_KRAS`` -> ``LUAD``). Returns ``None`` if neither the code nor any
    ancestor has a value. Mirrors :func:`cancer_tmb`."""
    df = cancer_apd1_response_df()
    vals = df.dropna(subset=["apd1_orr_pct"])
    mapping = dict(zip(vals["cancer_code"].astype(str),
                       vals["apd1_orr_pct"].astype(float)))
    if cancer_type is None:
        return mapping
    code = resolve_cancer_type(cancer_type)
    if code in mapping or not inherit:
        return mapping.get(code)
    reg = cancer_type_registry().set_index("code")
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in mapping:
            return mapping[cur]
        if cur not in reg.index:
            break
        cur = str(reg.loc[cur].get("parent_code", "") or "").strip() or None
    return None


def cancer_fusions_df():
    """Return the curated ``cancer-fusions.csv`` reference: characteristic gene
    fusions / oncogenic translocations per cancer-type code.

    One row per concrete pairing (so promiscuous partner sets are explicit),
    with the 5'/3' partner genes and their protein families annotated
    (``gene_5prime_family`` / ``gene_3prime_family`` — e.g. FET, ETS, BET,
    MiT/TFE, PAX, FOX, RTK, Ig locus). ``is_defining`` marks the characteristic
    lesion of the entity; ``pathognomonic`` is the stronger claim that the
    gene-pair maps to a single entity (diagnostic in context). Characteristically
    fusion-negative entities carry a single ``fusion_family="(none)"`` row naming
    the real driver. ``mechanism`` distinguishes chimeric ``fusion_transcript``
    from ``ig_enhancer_hijack`` / ``promoter_swap`` / ``enhancer_hijack`` (which
    have no chimeric transcript — see ``rnaseq_detectable``)."""
    return get_data("cancer-fusions")


def cancer_fusions(cancer_type=None, *, defining_only=False, pathognomonic_only=False):
    """Fusion rows for one cancer type (resolved via :func:`resolve_cancer_type`),
    or the whole table when ``cancer_type`` is None. ``defining_only`` /
    ``pathognomonic_only`` filter to the characteristic / diagnostic rows."""
    df = cancer_fusions_df()
    if cancer_type is not None:
        df = df[df["cancer_code"].astype(str) == resolve_cancer_type(cancer_type)]
    # is_defining / pathognomonic load as bool (get_data coerces true/false);
    # compare via lowercased string so the filter works whether bool or str.
    if defining_only:
        df = df[df["is_defining"].astype(str).str.lower() == "true"]
    if pathognomonic_only:
        df = df[df["pathognomonic"].astype(str).str.lower() == "true"]
    return df.reset_index(drop=True)


def cancer_viral_antigens_df():
    """Return the curated ``cancer-viral-antigens.csv`` reference: per-oncovirus
    targetable viral antigens for virally-driven cancers.

    Columns: ``virus`` (HPV, EBV, HBV, MCPyV, HHV8, HTLV-1, …),
    ``integration_mode`` (``integrated`` / ``episomal``),
    ``targetable_antigens`` (``;``-separated viral genes, e.g. ``E6;E7``),
    ``associated_cohorts`` (``;``-separated registry codes), ``notes``,
    ``source`` (PMID/DOI). Complements the registry ``viral_etiology`` /
    ``viral_agent`` columns with the antigen-level detail — viral oncoantigens
    are a distinct targetable class (foreign, constitutively expressed,
    sometimes clonally integrated)."""
    return get_data("cancer-viral-antigens")


def cancer_viral_antigens(virus=None):
    """Targetable viral antigens. With ``virus`` given (case-insensitive),
    returns that virus's list of antigens (``[]`` if unknown); otherwise a
    ``{virus: [antigen, ...]}`` map over the whole table."""
    df = cancer_viral_antigens_df()
    def _split(s):
        return [a.strip() for a in str(s).split(";") if a.strip()]
    if virus is not None:
        v = str(virus).strip().lower()
        hit = df[df["virus"].astype(str).str.lower() == v]
        if hit.empty:
            return []
        return _split(hit.iloc[0]["targetable_antigens"])
    return {str(r.virus): _split(r.targetable_antigens) for r in df.itertuples()}


def viral_antigens_for_cancer(cancer_type):
    """``[(virus, [antigen, ...]), ...]`` for a registry cancer code (resolved
    via :func:`resolve_cancer_type`) — the reverse lookup over
    ``associated_cohorts``. Empty when the cancer has no curated viral antigen
    (i.e. not a virally-driven entity in the table)."""
    code = resolve_cancer_type(cancer_type)
    df = cancer_viral_antigens_df()
    out = []
    for r in df.itertuples():
        cohorts = {c.strip() for c in str(r.associated_cohorts).split(";")
                   if c.strip() and c.strip().lower() != "nan"}
        if code in cohorts:
            ants = [a.strip() for a in str(r.targetable_antigens).split(";") if a.strip()]
            out.append((str(r.virus), ants))
    return out


def fusion_partners(gene, *, side=None):
    """Return the set of fusion partners of ``gene`` observed in the table.

    ``side=None`` returns all partners (gene on either end); ``side="5prime"``
    returns partners when ``gene`` is the 5' member (i.e. the observed 3'
    partners); ``side="3prime"`` returns the observed 5' partners. Useful for
    making sense of promiscuous sets — e.g. ``fusion_partners("EWSR1")`` spans
    FLI1/ERG/WT1/ATF1/NR4A3/POU5F1/…, ``fusion_partners("NR4A3", side="3prime")``
    returns EWSR1/TAF15/TCF12."""
    g = str(gene).strip().upper()
    df = cancer_fusions_df()
    out = set()
    if side in (None, "5prime"):
        out |= set(df.loc[df["gene_5prime"].astype(str).str.upper() == g, "gene_3prime"])
    if side in (None, "3prime"):
        out |= set(df.loc[df["gene_3prime"].astype(str).str.upper() == g, "gene_5prime"])
    return {p for p in out if isinstance(p, str) and p.strip()}


def cancer_types_with_fusion(
    fusion=None, *, partner=None, partner_family=None,
    defining_only=False, as_rows=False,
):
    """Reverse fusion lookup: cancer types matching a fusion, a partner gene, or
    a partner *family* — the inverse of :func:`fusion_status` / :func:`cancer_fusions`.

    Exactly one of:
    - ``fusion="EWSR1-FLI1"`` — a directional ``5'-3'`` fusion string (case-
      insensitive) -> types carrying that fusion (``SARC_EWS``);
    - ``partner="EWSR1"`` — a partner gene on either end -> every type with a
      fusion involving it (``SARC_EWS``, ``SARC_DSRCT``, ``SARC_CCS``, …);
    - ``partner_family="FET"`` (or ``"ETS"``) — a partner-family tag from
      ``gene_5prime_family`` / ``gene_3prime_family``.

    ``defining_only`` restricts to is_defining rows. Returns sorted canonical
    cancer codes, or the matching fusion-table rows when ``as_rows=True``.
    """
    given = [x for x in (fusion, partner, partner_family) if x is not None]
    if len(given) != 1:
        raise ValueError(
            "pass exactly one of fusion=, partner=, or partner_family="
        )
    df = cancer_fusions(defining_only=defining_only)
    g5 = df["gene_5prime"].astype(str).str.upper()
    g3 = df["gene_3prime"].astype(str).str.upper()
    if fusion is not None:
        parts = str(fusion).upper().replace("::", "-").split("-")
        if len(parts) != 2:
            raise ValueError(f"fusion must look like '5GENE-3GENE'; got {fusion!r}")
        a, b = (p.strip() for p in parts)
        mask = (g5 == a) & (g3 == b)
    elif partner is not None:
        p = str(partner).strip().upper()
        mask = (g5 == p) | (g3 == p)
    else:
        fam = str(partner_family).strip().upper()
        f5 = df["gene_5prime_family"].astype(str).str.upper()
        f3 = df["gene_3prime_family"].astype(str).str.upper()
        mask = (f5 == fam) | (f3 == fam)
    hits = df[mask]
    if as_rows:
        return hits.reset_index(drop=True)
    return sorted({str(c) for c in hits["cancer_code"] if str(c).strip()})


def protein_family(gene):
    """Protein/gene family of a fusion partner (e.g. EWSR1->FET, FLI1->ETS,
    BRD4->BET, PAX3->PAX, FOXO1->FOX, ALK->RTK), or ``None`` if the gene has no
    clear family annotation in ``cancer-fusions.csv``."""
    g = str(gene).strip().upper()
    df = cancer_fusions_df()
    for col, fam in (("gene_5prime", "gene_5prime_family"),
                     ("gene_3prime", "gene_3prime_family")):
        hit = df.loc[df[col].astype(str).str.upper() == g, fam]
        for v in hit:
            if isinstance(v, str) and v.strip():
                return v
    return None


_BURDEN_METRICS = ("us_incidence_pct", "us_mortality_pct",
                   "world_incidence_pct", "world_mortality_pct")


def cancer_burden_df():
    """Return the curated ``cancer-incidence-mortality.csv`` reference: each
    cancer **burden category**'s share (%) of annual cancer incidence and
    mortality, for the US and worldwide, cited per row (ACS Cancer Facts &
    Figures 2024 / GLOBOCAN 2022). Rows flagged in ``notes`` as subsets/rollups
    (e.g. small_cell_lung, leukemia subtypes, sarcoma/pediatric aggregates)
    overlap others — don't sum them blindly. Incidence vs mortality diverge
    sharply (pancreas/lung high mortality:incidence; prostate/thyroid low), as
    do US vs worldwide (stomach/liver/cervix far larger globally)."""
    return get_data("cancer-incidence-mortality")


def cancer_code_burden_map():
    """Return ``{cancer_code: burden_category}`` from
    ``cancer-code-burden-map.csv``. This is now only the small set of **overrides**
    the registry ontology can't express on its own (e.g. ``SARC_KS`` -> Kaposi
    rather than soft-tissue; ``LAML`` -> AML rather than other-leukemia;
    ``HL`` -> Hodgkin; ``CTCL`` -> non-Hodgkin). Everything else is resolved by
    :func:`burden_category` from the registry's family + primary_tissue."""
    df = get_data("cancer-code-burden-map")
    return dict(zip(df["cancer_code"].astype(str),
                    df["burden_category"].astype(str)))


def cancer_burden(category=None, *, metric="us_incidence_pct"):
    """Burden share (%) for one category and metric, or the whole
    ``{category: pct}`` map. ``metric`` is one of ``us_incidence_pct``,
    ``us_mortality_pct``, ``world_incidence_pct``, ``world_mortality_pct``."""
    if metric not in _BURDEN_METRICS:
        raise ValueError(f"metric must be one of {_BURDEN_METRICS}")
    df = cancer_burden_df()
    mapping = dict(zip(df["burden_category"].astype(str),
                       df[metric].astype(float)))
    if category is None:
        return mapping
    return mapping.get(category)


# Burden categories are anatomic-site shares (how ACS/GLOBOCAN tabulate), so a
# cohort is resolved to its category straight from the **cancer-type registry
# ontology** — one source of truth — rather than a parallel hand-map: the
# sarcoma family splits bone vs soft tissue on primary_tissue, plasma-cell and
# a handful of leukemia/lymphoma exceptions resolve by family, and primary
# tissue decides everything else. ``cancer-code-burden-map.csv`` now holds only
# the few true exceptions the ontology can't express.

# Sarcoma family -> bone_and_joint when its primary_tissue is skeletal, else
# soft_tissue_sarcoma (covers the 40+ SARC_* / RMS_* / OS / EWS / CHOR codes).
_BONE_SARCOMA_TISSUES = {"bone", "cartilage", "notochord"}

# Registry primary_tissue -> burden category. Covers every non-heme tissue in
# the registry; heme tissues are routed by :data:`_HEME_TISSUE_BURDEN` below.
_PRIMARY_TISSUE_BURDEN = {
    "lung": "lung", "breast": "breast", "prostate": "prostate",
    "colon": "colorectal", "rectum": "colorectal",
    "pancreas": "pancreas", "liver": "liver", "bile_duct": "gallbladder_biliary",
    "stomach": "stomach", "esophagus": "esophagus",
    "small_intestine": "small_intestine",
    "bladder": "bladder", "kidney": "kidney", "kidney_cns_soft": "kidney",
    "ovary": "ovary", "endometrium": "uterus_endometrium", "cervix": "cervix",
    "vulva": "vulva", "vagina": "vagina", "penis": "penis",
    "urethra": "bladder", "anal_canal": "anus",
    "fallopian_tube": "ovary", "peritoneum_serous": "ovary",  # HGSC pooled with OV
    "gallbladder": "gallbladder_biliary",
    "thyroid": "thyroid", "thyroid_c_cell": "thyroid",
    "testis": "testicular_germ_cell",
    "pleura": "mesothelioma", "peritoneum": "mesothelioma",
    "oral_cavity": "head_and_neck", "oropharynx": "head_and_neck",
    "pharynx": "head_and_neck", "nasopharynx": "head_and_neck",
    "larynx": "head_and_neck", "salivary_gland": "head_and_neck",
    "midline_structures": "head_and_neck", "thymus": "head_and_neck",
    "thorax": "head_and_neck",
    "cerebrum": "brain_cns", "cerebellum": "brain_cns",
    "eye": "eye_ocular", "retina": "eye_ocular", "skin": "melanoma",
    "epidermis": "non_melanoma_skin",  # BCC / cSCC keratinocyte carcinomas
    "ependyma": "brain_cns", "sellar_suprasellar": "brain_cns",
    "pons_midline": "brain_cns", "pituitary": "brain_cns",
    "adrenal_cortex": "adrenal", "adrenal_medulla": "adrenal",
    "sympathetic_ganglia": "adrenal",
    "bone": "bone_and_joint", "cartilage": "bone_and_joint",
    "notochord": "bone_and_joint",
    "soft_tissue": "soft_tissue_sarcoma", "smooth_muscle": "soft_tissue_sarcoma",
    "skeletal_muscle": "soft_tissue_sarcoma", "adipose": "soft_tissue_sarcoma",
    "nerve_sheath": "soft_tissue_sarcoma",
    "vascular_endothelium": "soft_tissue_sarcoma", "gi_wall": "soft_tissue_sarcoma",
}
# Heme (non-plasma): lymph node -> lymphoma; marrow/blood/spleen -> leukemia.
# AML and Hodgkin are exceptions carried in cancer-code-burden-map.csv.
_HEME_TISSUE_BURDEN = {
    "lymph_node": "non_hodgkin_lymphoma",
    "bone_marrow": "leukemia_all_other", "peripheral_blood": "leukemia_all_other",
    "spleen_marrow": "leukemia_all_other",
}
# Last-resort family fallback when primary_tissue is blank/unmapped.
_FAMILY_BURDEN = {
    "sarcoma": "soft_tissue_sarcoma", "melanoma": "melanoma", "cns": "brain_cns",
    "carcinoma-skin": "non_melanoma_skin",
}


def burden_category(cancer_type):
    """Robustly resolve a cancer type (code, alias, or display name) to an
    anatomic burden category, driven by the cancer-type registry ontology.
    Order: normalize via :func:`resolve_cancer_type`; the small explicit
    ``cancer-code-burden-map`` *override* (walking the ``parent_code`` chain);
    then registry-driven — sarcoma family splits bone vs soft tissue, plasma
    cell -> myeloma, other heme by tissue, then ``primary_tissue``, then
    ``family``. Returns ``None`` only when nothing matches — callers should
    **warn**, not silently skip (an unmapped cohort is a coverage gap)."""
    try:
        code = resolve_cancer_type(cancer_type)
    except ValueError:
        return None
    if code is None:
        return None
    override = cancer_code_burden_map()
    reg = cancer_type_registry().set_index("code")
    # 1. explicit override (true exceptions only), walking up the parent chain
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur in override:
            return override[cur]
        if cur not in reg.index:
            break
        cur = str(reg.loc[cur].get("parent_code", "") or "").strip() or None
    # 2. registry-driven, walking up the parent chain for blank tissue/family
    cur, seen = code, set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur not in reg.index:
            break
        row = reg.loc[cur]
        family = str(row.get("family", "") or "")
        tissue = str(row.get("primary_tissue", "") or "")
        if family == "sarcoma":
            return ("bone_and_joint" if tissue in _BONE_SARCOMA_TISSUES
                    else "soft_tissue_sarcoma")
        if family == "heme-plasma":
            return "multiple_myeloma"
        if family.startswith("heme") and tissue in _HEME_TISSUE_BURDEN:
            return _HEME_TISSUE_BURDEN[tissue]
        if tissue in _PRIMARY_TISSUE_BURDEN:
            return _PRIMARY_TISSUE_BURDEN[tissue]
        if family in _FAMILY_BURDEN:
            return _FAMILY_BURDEN[family]
        cur = str(row.get("parent_code", "") or "").strip() or None
    return None


# Per-cohort median tumor-cell purity from TCGA (Aran et al., Nat Commun 2015).
# Used by trufflepig's purity-confidence reasoning; published here as reference
# data so consumers don't have to depend on the analysis package just to look
# up the canonical cohort baseline.
TCGA_MEDIAN_PURITY = {
    "ACC": 0.79, "BLCA": 0.59, "BRCA": 0.73, "CESC": 0.49, "CHOL": 0.68,
    "COAD": 0.59, "DLBC": 0.94, "ESCA": 0.50, "GBM": 0.83, "HNSC": 0.60,
    "KICH": 0.84, "KIRC": 0.72, "KIRP": 0.78, "LAML": 0.95, "LGG": 0.87,
    "LIHC": 0.73, "LUAD": 0.56, "LUSC": 0.67, "MESO": 0.55, "OV": 0.72,
    "PAAD": 0.42, "PCPG": 0.69, "PRAD": 0.69, "READ": 0.60, "SARC": 0.66,
    "SKCM": 0.65, "STAD": 0.40, "TGCT": 0.75, "THCA": 0.72, "THYM": 0.78,
    "UCEC": 0.71, "UCS": 0.65, "UVM": 0.85,
}
