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

"""Active OncoRef CTA target-selection policy and manifest provenance (#655).

PirlyGenes adopts OncoRef's active :mod:`oncoref.target_selection` catalog and
manifest metadata directly rather than hardcoding historical classifier
constants or inferring policy versions from dated output directories.

Policy boundary
---------------
* **Active OncoRef CTA policy** (default ``CTA_*`` APIs here and in
  :mod:`pirlygenes.gene_sets_cancer`): reads live membership, categories
  (``CTA``, ``Cancer Antigen only``, ``Reproductive Antigen only``, ``Other``,
  ``Unsafe``), ``cta_tier`` values, 4-way sex-restriction partitions
  (``CTA-M``, ``CTA-F``, ``CTA-MF``, ``CTA-P``), clinical organ/epitope
  clearance metadata, and manifest provenance from ``oncoref.target_selection``.
* **Curated clinical / tumor-marker panels** (``cancer-key-genes.csv``,
  ``cancer-type-genes.csv``, ``cancer-surfaceome.csv``, ``ADC-*.csv``,
  ``TCR-T-*.csv``, ``CAR-T-approved.csv``, lineage/family/discriminator panels):
  remain owned and curated by PirlyGenes and are never altered by OncoRef's
  research CTA selection policy.
* **Explicit historical v1 HPA compatibility helpers** (``CTA_historical_*`` and
  the frozen HPA-axis views ``CTA_testis_restricted_*``,
  ``CTA_placental_restricted_*``): import ``oncoref.cta`` explicitly for
  historical reproducibility without altering active defaults.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
import json

import pandas as pd

_FOUR_WAY_CTA_SEX_SETS: tuple[str, ...] = ("CTA-M", "CTA-F", "CTA-MF", "CTA-P")
_FOUR_WAY_RA_SEX_SETS: tuple[str, ...] = ("RA-M", "RA-F", "RA-MF", "RA-P")


def _clean_gene_id(value: str) -> str:
    return str(value).strip().split(".", 1)[0]


def _clean_text(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _parse_json_tuple(value) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            return tuple(str(item) for item in parsed if item)
        return (stripped,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if item)
    return (str(value),)


def CTA_policy_id() -> str:
    """Return OncoRef's active target-selection policy identifier."""
    from oncoref import target_selection

    return str(target_selection.POLICY_ID)


def CTA_policy_manifest() -> dict:
    """Return a detached manifest describing OncoRef's active CTA policy.

    All provenance fields are read dynamically from OncoRef's active
    :mod:`oncoref.target_selection` contract and metadata receipt rather than
    hardcoded constants or dated directory names.
    """
    import oncoref
    from oncoref import antigen_sets, species_target_selection, target_selection

    meta = deepcopy(target_selection.metadata())
    data_sha256 = str(meta.get("data_sha256") or "")
    policy_id = str(meta.get("policy_id") or target_selection.POLICY_ID)
    ensembl_release = int(antigen_sets.ENSEMBL_RELEASE)
    return {
        "policy_id": policy_id,
        "schema": str(meta.get("schema") or "oncoref.target-selection"),
        "version": meta.get("version"),
        "oncoref_version": str(oncoref.__version__),
        "reference_id": f"homo_sapiens:GRCh38:Ensembl:{ensembl_release}",
        "ensembl_release": ensembl_release,
        "data_sha256": data_sha256,
        "source_snapshot_id": f"sha256:{data_sha256}" if data_sha256 else "",
        "source_surface": "oncoref.target_selection",
        "release_ready": meta.get("release_ready"),
        "open_reviews": list(meta.get("open_reviews") or ()),
        "categories": list(target_selection.CATEGORIES),
        "cta_tiers": list(target_selection.CTA_TIERS),
        "sex_restriction_classes": list(species_target_selection.SEX_RESTRICTION_CLASSES),
        "summary": deepcopy(meta.get("summary") or {}),
        "metadata": meta,
    }


def _provenance_attrs() -> dict:
    manifest = CTA_policy_manifest()
    return {
        "policy_id": manifest["policy_id"],
        "oncoref_version": manifest["oncoref_version"],
        "reference_id": manifest["reference_id"],
        "data_sha256": manifest["data_sha256"],
        "source_snapshot_id": manifest["source_snapshot_id"],
        "source_surface": manifest["source_surface"],
    }


@lru_cache(maxsize=1)
def _gene_identity_index() -> dict[str, tuple[str | None, tuple[str, ...]]]:
    """Map canonical Ensembl gene ID to ``(canonical_symbol, synonyms)``."""
    from oncoref import gene_identity_catalog

    catalog = gene_identity_catalog()
    index: dict[str, tuple[str | None, tuple[str, ...]]] = {}
    for row in catalog.itertuples(index=False):
        gid = _clean_gene_id(row.canonical_gene_id)
        symbol = _clean_text(getattr(row, "canonical_symbol", None))
        synonyms = _parse_json_tuple(getattr(row, "gene_name_synonyms", None))
        index[gid] = (symbol, synonyms)
    return index


def _resolve_member_symbol(gene_id: str, fallback_label: str) -> str:
    from oncoref import gene_ids as oc_gene_ids

    identity = _gene_identity_index().get(gene_id)
    if identity is not None and identity[0]:
        return identity[0]
    return oc_gene_ids.canonical_gene_symbol(gene_id) or fallback_label or gene_id


@lru_cache(maxsize=1)
def _active_selection_gene_table() -> pd.DataFrame:
    """Project OncoRef's active ``target_selection.results()`` onto member genes."""
    from oncoref import target_selection

    raw = target_selection.results()
    manifest = CTA_policy_manifest()
    identity_index = _gene_identity_index()

    rows: list[dict] = []
    for decision in raw.to_dict("records"):
        members = tuple(_clean_gene_id(g) for g in _parse_json_tuple(decision.get("member_gene_ids")))
        label = str(decision.get("protein_group_label") or "")
        category = str(decision.get("category") or "")
        classification = str(decision.get("classification") or category)
        is_cta = bool(decision.get("is_cta", False))
        is_ca = bool(decision.get("is_cancer_antigen", False))
        is_ra = bool(decision.get("is_reproductive_antigen", False))
        is_candidate = bool(decision.get("is_target_candidate", is_ca or is_ra))
        cta_tier = _clean_text(decision.get("cta_tier"))
        sex_class = _clean_text(decision.get("sex_restriction_class"))
        sex_code = _clean_text(decision.get("sex_restriction_code"))
        policy_id = _clean_text(decision.get("policy")) or manifest["policy_id"]
        cleared_organs = _parse_json_tuple(decision.get("clinical_cleared_organs"))
        cleared_epitopes = _parse_json_tuple(decision.get("clinical_cleared_epitopes"))
        cleared_alleles = _parse_json_tuple(decision.get("clinical_hla_alleles"))
        cleared_transcripts = tuple(
            _clean_gene_id(t) for t in _parse_json_tuple(decision.get("clinical_cleared_transcripts"))
        )
        unverified_isoforms = tuple(
            _clean_gene_id(t) for t in _parse_json_tuple(decision.get("clinical_unverified_isoforms"))
        )
        rep_transcripts = tuple(
            _clean_gene_id(t) for t in _parse_json_tuple(decision.get("representative_transcripts"))
        )

        for gid in members:
            identity = identity_index.get(gid)
            symbol = _resolve_member_symbol(gid, label)
            synonyms = identity[1] if identity is not None else ()
            rows.append(
                {
                    "Symbol": symbol,
                    "Ensembl_Gene_ID": gid,
                    "is_current_cta": is_cta,
                    "selection_synonyms": synonyms,
                    "selection_entity_id": str(decision.get("entity_id") or ""),
                    "selection_entity_kind": "protein_group",
                    "selection_label": label,
                    "selection_gene_ids": members,
                    "selection_category": category,
                    "selection_subtype": _clean_text(decision.get("subtype")),
                    "selection_classification": classification,
                    "selection_cta_tier": cta_tier,
                    "selection_is_cancer_antigen": is_ca,
                    "selection_is_reproductive_antigen": is_ra,
                    "selection_is_target_candidate": is_candidate,
                    "selection_assessment_status": _clean_text(decision.get("assessment_status")),
                    "selection_identity_status": (
                        "resolved"
                        if identity is not None
                        else "registry_member_outside_canonical_catalog"
                    ),
                    "selection_policy_id": policy_id,
                    "selection_reference_id": manifest["reference_id"],
                    "selection_source_snapshot_id": manifest["source_snapshot_id"],
                    "selection_sex_restriction_class": sex_class,
                    "selection_sex_restriction_code": sex_code,
                    "selection_rare_cell_status": _clean_text(decision.get("rare_cell_status")),
                    "selection_nuclear_rna_caution": _clean_text(decision.get("nuclear_rna_caution")),
                    "selection_clinical_clearance_scope": _clean_text(
                        decision.get("clinical_clearance_scope")
                    ),
                    "selection_clinical_cleared_organs": cleared_organs,
                    "selection_clinical_cleared_epitopes": cleared_epitopes,
                    "selection_clinical_hla_alleles": cleared_alleles,
                    "selection_clinical_cleared_transcripts": cleared_transcripts,
                    "selection_clinical_unverified_isoforms": unverified_isoforms,
                    "selection_representative_transcripts": rep_transcripts,
                }
            )
    frame = pd.DataFrame(rows)
    if frame["Ensembl_Gene_ID"].duplicated().any():
        dups = sorted(set(frame.loc[frame["Ensembl_Gene_ID"].duplicated(), "Ensembl_Gene_ID"]))
        raise ValueError(
            "OncoRef target_selection has duplicate member gene IDs: " + ", ".join(dups)
        )
    frame.attrs.update(_provenance_attrs())
    return frame


def _normalize_filter_values(values: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(str(v) for v in values)


def CTA_selection_df(
    *,
    category: str | Iterable[str] | None = None,
    cta_tier: str | Iterable[str] | None = None,
    sex_restriction: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a detached copy of OncoRef's active ``target_selection.results()``.

    Parameters
    ----------
    category
        Optional category filter (e.g. ``"CTA"``, ``"Cancer Antigen only"``,
        ``"Reproductive Antigen only"``, ``"Other"``, ``"Unsafe"``).
    cta_tier
        Optional tier filter (``"core"``, ``"core_with_extended_exposure"``,
        ``"extended_only"``).
    sex_restriction
        Optional 4-way sex-restriction filter accepting either OncoRef set codes
        (``"CTA-M"``, ``"CTA-F"``, ``"CTA-MF"``, ``"CTA-P"``, ``"RA-M"``,
        ``"RA-F"``, ``"RA-MF"``, ``"RA-P"``) or class labels
        (``"Male-restricted"``, ``"Female-restricted"``,
        ``"Shared male & female"``, ``"Pregnancy / peripartum-only"``).
    """
    from oncoref import species_target_selection, target_selection

    df = target_selection.results().copy()
    mask = pd.Series(True, index=df.index)

    if category is not None:
        wanted = set(_normalize_filter_values(category))
        unknown = wanted - set(target_selection.CATEGORIES)
        if unknown:
            raise ValueError(
                f"Unknown target_selection category {sorted(unknown)}; "
                f"choose from {target_selection.CATEGORIES}"
            )
        mask &= df["category"].isin(wanted)

    if cta_tier is not None:
        wanted_tiers = set(_normalize_filter_values(cta_tier))
        unknown_tiers = wanted_tiers - set(target_selection.CTA_TIERS)
        if unknown_tiers:
            raise ValueError(
                f"Unknown cta_tier {sorted(unknown_tiers)}; "
                f"choose from {target_selection.CTA_TIERS}"
            )
        mask &= df["cta_tier"].isin(wanted_tiers)

    if sex_restriction is not None:
        wanted_sex = _normalize_filter_values(sex_restriction)
        sex_spec = species_target_selection.SEX_SETS
        valid_classes = set(species_target_selection.SEX_RESTRICTION_CLASSES)
        sex_mask = pd.Series(False, index=df.index)
        for token in wanted_sex:
            if token in sex_spec:
                sex_class, requires_cta = sex_spec[token]
                base = df["is_cta"] if requires_cta else df["is_reproductive_antigen"]
                sex_mask |= base & df["sex_restriction_class"].eq(sex_class)
            elif token in valid_classes:
                sex_mask |= df["sex_restriction_class"].eq(token)
            else:
                raise ValueError(
                    f"Unknown sex_restriction {token!r}; choose a set in "
                    f"{tuple(sex_spec)} or class in {species_target_selection.SEX_RESTRICTION_CLASSES}"
                )
        mask &= sex_mask

    out = df.loc[mask].reset_index(drop=True)
    out.attrs.update(_provenance_attrs())
    return out


def _group_mask(table: pd.DataFrame, group: str) -> pd.Series:
    from oncoref import antigen_sets, species_target_selection

    if group not in antigen_sets.ALL_SETS:
        raise ValueError(
            f"Unknown antigen set {group!r}; choose from {antigen_sets.ALL_SETS}"
        )
    if group in antigen_sets.EXCLUSION_CATEGORIES:
        return table["selection_category"].isin(antigen_sets.EXCLUSION_CATEGORIES[group])
    if group in species_target_selection.SEX_SETS:
        sex_class, requires_cta = species_target_selection.SEX_SETS[group]
        base = (
            table["is_current_cta"]
            if requires_cta
            else table["selection_is_reproductive_antigen"]
        )
        return base & table["selection_sex_restriction_class"].eq(sex_class)
    flag_col = {
        "CTA": "is_current_cta",
        "CA": "selection_is_cancer_antigen",
        "RA": "selection_is_reproductive_antigen",
        "CA|RA": "selection_is_target_candidate",
    }[group]
    return table[flag_col].astype(bool)


def CTA_antigen_set_gene_ids(group: str = "CTA") -> set[str]:
    """Return unversioned Ensembl gene IDs for an active OncoRef antigen set.

    Supports ``"CTA"``, ``"CA"``, ``"RA"``, ``"CA|RA"``, ``"Unsafe"``,
    ``"Other"``, ``"Unsafe|Other"``, and OncoRef 1.52.0's 4-way sex-restricted
    sets (``"CTA-M"``, ``"CTA-F"``, ``"CTA-MF"``, ``"CTA-P"``, ``"RA-M"``,
    ``"RA-F"``, ``"RA-MF"``, ``"RA-P"``, plus legacy shorthand aliases).
    """
    table = _active_selection_gene_table()
    mask = _group_mask(table, group)
    return set(table.loc[mask, "Ensembl_Gene_ID"].astype(str))


def CTA_antigen_set_gene_names(group: str = "CTA") -> set[str]:
    """Return canonical gene symbols for an active OncoRef antigen set."""
    table = _active_selection_gene_table()
    mask = _group_mask(table, group)
    return {s for s in table.loc[mask, "Symbol"].dropna().astype(str) if s}


def CTA_gene_ids() -> set[str]:
    """Active OncoRef CTA member Ensembl gene IDs (``target_selection`` CTA)."""
    return CTA_antigen_set_gene_ids("CTA")


def CTA_gene_names() -> set[str]:
    """Active OncoRef CTA canonical gene symbols (``target_selection`` CTA)."""
    return CTA_antigen_set_gene_names("CTA")


def CTA_gene_id_to_name() -> dict[str, str]:
    """Active OncoRef CTA ``{Ensembl_Gene_ID: Symbol}`` mapping."""
    table = _active_selection_gene_table()
    sub = table.loc[table["is_current_cta"], ["Ensembl_Gene_ID", "Symbol"]]
    return dict(zip(sub["Ensembl_Gene_ID"].astype(str), sub["Symbol"].astype(str)))


def CTA_filtered_gene_ids() -> set[str]:
    """Active OncoRef CTA member Ensembl gene IDs (alias of :func:`CTA_gene_ids`)."""
    return CTA_gene_ids()


def CTA_filtered_gene_names() -> set[str]:
    """Active OncoRef CTA canonical gene symbols (alias of :func:`CTA_gene_names`)."""
    return CTA_gene_names()


def CTA_never_expressed_gene_ids() -> set[str]:
    """Empty set under the active OncoRef policy (expression is required for CTA)."""
    return set()


def CTA_never_expressed_gene_names() -> set[str]:
    """Empty set under the active OncoRef policy (expression is required for CTA)."""
    return set()


def CTA_unfiltered_gene_ids() -> set[str]:
    """All Ensembl gene IDs assessed in OncoRef's active ``target_selection`` catalog."""
    table = _active_selection_gene_table()
    return set(table["Ensembl_Gene_ID"].astype(str))


def CTA_unfiltered_gene_names() -> set[str]:
    """All canonical gene symbols assessed in OncoRef's active ``target_selection`` catalog."""
    table = _active_selection_gene_table()
    return {s for s in table["Symbol"].dropna().astype(str) if s}


def CTA_excluded_gene_ids() -> set[str]:
    """Ensembl gene IDs classified ``Unsafe`` under OncoRef's active policy.

    ``Cancer Antigen only``, ``Reproductive Antigen only``, and ``Other``
    remain distinct categories rather than being merged into ``Unsafe``.
    """
    return CTA_antigen_set_gene_ids("Unsafe")


def CTA_excluded_gene_names() -> set[str]:
    """Canonical gene symbols classified ``Unsafe`` under OncoRef's active policy."""
    return CTA_antigen_set_gene_names("Unsafe")


def CTA_cancer_antigen_gene_ids() -> set[str]:
    """Active OncoRef Cancer Antigen (``CA`` = CTA + Cancer Antigen only) Ensembl IDs."""
    return CTA_antigen_set_gene_ids("CA")


def CTA_cancer_antigen_gene_names() -> set[str]:
    """Active OncoRef Cancer Antigen (``CA`` = CTA + Cancer Antigen only) symbols."""
    return CTA_antigen_set_gene_names("CA")


def CTA_reproductive_antigen_gene_ids() -> set[str]:
    """Active OncoRef Reproductive Antigen (``RA`` = CTA + RA-only) Ensembl IDs."""
    return CTA_antigen_set_gene_ids("RA")


def CTA_reproductive_antigen_gene_names() -> set[str]:
    """Active OncoRef Reproductive Antigen (``RA`` = CTA + RA-only) symbols."""
    return CTA_antigen_set_gene_names("RA")


def CTA_target_candidate_gene_ids() -> set[str]:
    """Active OncoRef candidate antigen union (``CA|RA``) Ensembl gene IDs."""
    return CTA_antigen_set_gene_ids("CA|RA")


def CTA_target_candidate_gene_names() -> set[str]:
    """Active OncoRef candidate antigen union (``CA|RA``) gene symbols."""
    return CTA_antigen_set_gene_names("CA|RA")


def CTA_by_tier(
    tier: str | Iterable[str], *, column: str = "Symbol"
) -> set[str]:
    """Return active OncoRef CTA genes belonging to one or more ``cta_tier`` values.

    Valid tiers are OncoRef's ``target_selection.CTA_TIERS``
    (``"core"``, ``"core_with_extended_exposure"``, ``"extended_only"``).
    """
    from oncoref import target_selection

    if column not in {"Symbol", "Ensembl_Gene_ID"}:
        raise ValueError(
            f"Unsupported column {column!r}; expected 'Symbol' or 'Ensembl_Gene_ID'"
        )
    wanted = set(_normalize_filter_values(tier))
    unknown = wanted - set(target_selection.CTA_TIERS)
    if unknown:
        raise ValueError(
            f"Unknown cta_tier {sorted(unknown)}; choose from {target_selection.CTA_TIERS}"
        )
    table = _active_selection_gene_table()
    mask = table["is_current_cta"] & table["selection_cta_tier"].isin(wanted)
    return {s for s in table.loc[mask, column].dropna().astype(str) if s}


def CTA_sex_partition_gene_ids() -> dict[str, set[str]]:
    """Return OncoRef 1.52.0's disjoint 4-way CTA sex-restriction partition by ID.

    Keys are ``"CTA-M"``, ``"CTA-F"``, ``"CTA-MF"``, and ``"CTA-P"``.
    """
    return {group: CTA_antigen_set_gene_ids(group) for group in _FOUR_WAY_CTA_SEX_SETS}


def CTA_sex_partition_gene_names() -> dict[str, set[str]]:
    """Return OncoRef 1.52.0's disjoint 4-way CTA sex-restriction partition by symbol.

    Keys are ``"CTA-M"``, ``"CTA-F"``, ``"CTA-MF"``, and ``"CTA-P"``.
    """
    return {group: CTA_antigen_set_gene_names(group) for group in _FOUR_WAY_CTA_SEX_SETS}


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    """Case-insensitive symbol/synonym index over active CTA genes."""
    table = _active_selection_gene_table()
    sub = table.loc[table["is_current_cta"], ["Symbol", "selection_synonyms"]]
    direct: dict[str, str] = {}
    synonyms: dict[str, list[str]] = {}
    for row in sub.itertuples(index=False):
        sym = str(row.Symbol)
        direct[sym.upper()] = sym
        for syn in row.selection_synonyms or ():
            key = str(syn).strip().upper()
            if key:
                synonyms.setdefault(key, []).append(sym)
    resolved = dict(direct)
    for key, targets in synonyms.items():
        unique = sorted(set(targets))
        if key not in resolved and len(unique) == 1:
            resolved[key] = unique[0]
    return resolved


def cta_symbol_for_alias(name: str) -> str | None:
    """Resolve an active CTA symbol or alias, falling back to historical CTA aliases."""
    if not name:
        return None
    token = str(name).strip()
    if not token:
        return None
    hit = _alias_index().get(token.upper())
    if hit is not None:
        return hit
    from oncoref import cta as _historical_cta

    return _historical_cta.cta_symbol_for_alias(token)


# --- Explicit historical v1 HPA CTA helpers ---------------------------------


def CTA_historical_gene_ids() -> set[str]:
    """Frozen historical v1 HPA CTA Ensembl gene IDs (``oncoref.cta``)."""
    from oncoref import cta as _historical_cta

    return set(_historical_cta.cta_gene_ids())


def CTA_historical_gene_names() -> set[str]:
    """Frozen historical v1 HPA CTA gene symbols (``oncoref.cta``)."""
    from oncoref import cta as _historical_cta

    return set(_historical_cta.cta_gene_names())


def CTA_historical_evidence() -> pd.DataFrame:
    """Frozen historical v1 HPA CTA evidence table (``oncoref.cta``)."""
    from oncoref import cta as _historical_cta

    return _historical_cta.cta_evidence().copy()


def clear_cta_policy_cache() -> None:
    """Clear in-process caches derived from OncoRef's active CTA selection."""
    _gene_identity_index.cache_clear()
    _active_selection_gene_table.cache_clear()
    _alias_index.cache_clear()
