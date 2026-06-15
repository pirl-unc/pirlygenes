---
name: project_taxonomy_groupings_and_union_C
description: "#366 resolved as option C (bare code = all:-union default); cross-cutting subtype-groupings infra + CRC node + exclude_microarray_proxy shipped 5.22.31/.37"
metadata:
  type: project
---

Taxonomy/evidence decisions shipped June 2026 (v5.22.31–.37):

**#366 → option C (user's choice): a bare cancer code defaults to the `all:`
union** (the computed cross-source aggregate), and a source-specific cohort is
opt-in. Verified already-in-place: bare `SARC` resolves to the 30-atom all-union
(incl Ewing/OS), NOT the TCGA adult-STS cohort; the residual adult-STS miscall
is **trufflepig's** reference choice, not a pirlygenes resolution bug. `SARC_LPS`
intermediate node already exists.

**Cross-cutting subtype groupings (#385):** `data/cancer-subtype-groupings.csv`
(`group_code,axis,member_code,basis`) defines orthogonal cross-cuts DECOUPLED
from the `parent_code` tree — MSI/MSS/POLE/HPV_POS/HPV_NEG/MYCN_AMP/MYCN_NONAMP.
Accessor `cancer_subtype_group(group, under=)` (re-exported on
`pirlygenes.cancer_types`): `under=` intersects a group with a hierarchy node,
e.g. `subtype_group('MSI', under='CRC') == ['COAD_MSI','READ_MSI']`. A leaf groups
two ways: organ (parent_code) AND mechanism (grouping). Added `CRC` computed node
(parent of COAD/READ) via registry + cancer-cohort-aggregates + cohort-registry
COMPUTED_COLORECTAL.

**all:-union heterogeneity contract (#366 C, v5.22.37):** the union mixes member
cohorts of DIFFERENT assays (microarray/FPKM/RPKM/counts → clean TPM per-cohort
at build; see `processing_pipeline`) and DIFFERENT gene universes (~13k–61k). So:
microarray-proxy TPM is NOT cross-platform comparable; absent genes are
not_measurable NOT 0 (see [[feedback_missing_vs_zero]]); the reference returns
per-(gene,code,source) rows and NEVER pre-pools. `cancer_reference_expression(...,
exclude_microarray_proxy=True)` gives a pipeline-homogeneous poolable view.

**Remaining:** explicit `source:node` grammar selector (`tcga:SARC`); and
heterogeneity-safe POOLING into one summary (per-(gene,cohort) availability mask
+ n-weighting) = the [[project_cross_cohort_mixing_gene_availability]] roadmap.
