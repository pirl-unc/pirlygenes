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

"""Centralized cancer-type metadata — the single place to ask what the library
knows about a cancer type, so per-type facts aren't curated piecemeal across
data tables.

Everything is keyed by the canonical registry code and accepts any synonym
(common name, display name, or a pre-rename old code) on input. The facts live
in ``data/cancer-type-registry.csv`` (one row per type); this package is the
thin, cohesive accessor surface over it.

    from pirlygenes import cancer_types as ct

    ct.info("RMS_ARMS")            # full record (name/family/tissue/parent/…)
    ct.synonyms("OS")             # ['SARC_OS' aliases…]  (reverse of resolve)
    ct.resolve("prostate")        # -> "PRAD"
    ct.canonical("MID_NET")       # -> "NET_MIDGUT" (pure alias map, no raise)
    ct.tissue_of_origin("NPC")    # -> "nasopharynx"
    ct.viral_status("HNSC_HPVpos")# -> {"etiology":"defining","agent":"HPV"}
    ct.fusion_status("SARC_EWS")  # -> {"status":"defining","driver":"EWSR1-FLI1; …"}
    ct.subtypes_of("SARC")        # direct children
    ct.in_family("sarcoma")       # all codes in a family
    ct.by_tissue("skeletal_muscle")
    ct.burden_category("SARC_RMS_ARMS")
    ct.tmb("SKCM")

Curation provenance:
- viral / fusion status are curated controlled vocabularies on the registry
  (``viral_etiology``/``viral_agent``, ``fusion_driven``/``fusion_driver``).
  Fusion calls are sourced from / cross-checked against the cited
  ``cancer-fusions.csv``; non-obvious values carry a PMID in that table.
  Cells with no established public source are left ``none``/blank, never
  fabricated.
"""

from __future__ import annotations

from ..gene_sets_cancer import (
    cancer_type_info as info,
    cancer_type_synonyms as synonyms,
    resolve_cancer_type as resolve,
    canonical_cancer_code as canonical,
    tissue_of_origin,
    viral_status,
    fusion_status,
    cancer_type_subtypes_of as subtypes_of,
    cancer_types_in_family as in_family,
    cancer_types_by_tissue as by_tissue,
    burden_category,
    cancer_tmb as tmb,
    cancer_type_registry as registry,
)

__all__ = [
    "info",
    "synonyms",
    "resolve",
    "canonical",
    "tissue_of_origin",
    "viral_status",
    "fusion_status",
    "subtypes_of",
    "in_family",
    "by_tissue",
    "burden_category",
    "tmb",
    "registry",
]
