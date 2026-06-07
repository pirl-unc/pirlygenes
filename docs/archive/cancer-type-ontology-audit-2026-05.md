# Cancer-type registry ontology audit (Phase C)

> **Historical / largely implemented (Phase C).** This is the Phase-C planning
> audit; the restructure it proposes has since SHIPPED. Where the plan and the
> implementation diverge, the **code is authoritative**: `PANNET` was retired
> (`PANNET`→`NET_PANCREAS`; old code resolves via alias), and the AML-ELN
> codes shipped lowercase (`LAML_ELNadv/fav/int`, not `LAML_ELNAdv`). The
> `(130 codes)` figure is the May-2026 snapshot; the live registry has grown
> since — query `cancer_type_registry()` for current codes/counts. See #315.

Audit of `pirlygenes/data/cancer-type-registry.csv` (130 codes) against the
**WHO Classification of Tumours, 5th edition** series, and the resulting
restructure plan. Grounded in the WHO volumes (cited by volume; no fabricated
page-level citations):

- Soft Tissue and Bone Tumours (5th ed., 2020)
- Central Nervous System Tumours (5th ed., 2021)
- Thoracic Tumours (5th ed., 2021)
- Endocrine and Neuroendocrine Tumours (5th ed., 2022)
- Haematolymphoid Tumours (5th ed., 2022)
- Digestive System (2019), Urinary & Male Genital (2022), Female Genital (2020),
  Breast (2019), Head & Neck (2022), Skin (2023), **Paediatric Tumours (2023)**

## Core finding — `family` conflates two orthogonal axes

The single `family` column mixes **developmental age** (`pediatric-*`) with
**tissue/lineage**. WHO classifies by **lineage/organ-system, age-independent**;
the dedicated *Paediatric Tumours* volume cross-cuts the lineage volumes rather
than replacing them. The conflation actively fragments real lineages:

- **Sarcomas split across 3 families by age:** `sarcoma` (35) vs `pediatric-bone`
  (OS, EWS) vs `pediatric-soft` (RMS_ERMS/ARMS/SSRMS) — even though `RMS_PRMS`,
  `CHON`, `GCTB` already sit in `sarcoma`. WHO *Soft Tissue and Bone Tumours*
  places osteosarcoma, Ewing, all rhabdomyosarcomas, chondrosarcoma, GCTB in one
  lineage regardless of patient age. This is why the CTA-vs-TMB plot colored
  OS/EWS/RMS as "pediatric" instead of "sarcoma," fragmenting the cluster.
- **Neuroendocrine scattered:** `net` (11, and it mixes well-diff NET with
  poorly-diff NEC), while PCPG/MTC sit in `endocrine` and NBL in `pediatric-net`.
  WHO *Endocrine & Neuroendocrine Tumours* treats NENs as a cross-organ family
  and explicitly separates **NET (well-differentiated, G1–G3)** from **NEC
  (poorly-differentiated, small-/large-cell)**.

## Resolution — `family` = lineage only; add an orthogonal `pediatric` flag

### Family reassignments
| Codes | From | To | WHO basis |
|---|---|---|---|
| OS, EWS, RMS_ERMS/ARMS/SSRMS (→`SARC_*`) | pediatric-bone, pediatric-soft | **sarcoma** | Soft Tissue & Bone 2020 |
| ATRT, MBL(+WNT/SHH/G3/G4) | pediatric-cns | **cns** | CNS 2021 (embryonal CNS tumours) |
| NBL(+MYCN subtypes) | pediatric-net | **neuroendocrine** | Endocrine & NE 2022 (peripheral neuroblastic, neural-crest) |
| WILMS, HEPB, RB, RT | pediatric-embryonal/eye/liver | **embryonal** | Paediatric 2023 (nephroblastoma, hepatoblastoma, retinoblastoma, rhabdoid = embryonal/blastemal) |
| (rename) `net` family | net | **neuroendocrine** | Endocrine & NE 2022 |

Retire the `pediatric-bone`, `pediatric-soft`, `pediatric-cns`,
`pediatric-embryonal`, `pediatric-eye`, `pediatric-liver`, `pediatric-net`
families. Add a boolean **`pediatric`** column set on the codes that were under
them (OS/EWS/RMS, ATRT/MBL, WILMS/HEPB/RB/RT, NBL) — age as a flag, not a family.

Note (judgment call): `embryonal` is treated as a *lineage* descriptor
(blastemal/embryonal neoplasms), not an age bucket; ATRT (intracranial rhabdoid)
goes to `cns`, RT (extracranial rhabdoid) to `embryonal`. PCPG/MTC stay in
`endocrine` (neural-crest endocrine) and additionally carry a neuroendocrine tag.

### Code renames (one-separator + family conventions; every old code keeps a backward-compat alias in CANCER_TYPE_ALIASES so trufflepig/external callers don't break)

**Sarcoma — `SARC_` prefix everywhere:**
OS→SARC_OS, EWS→SARC_EWS, CHON→SARC_CHON, CHOR→SARC_CHOR, GCTB→SARC_GCTB,
RMS_ERMS→SARC_RMS_ERMS, RMS_ARMS→SARC_RMS_ARMS, RMS_PRMS→SARC_RMS_PRMS,
RMS_SSRMS→SARC_RMS_SSRMS, ESS_LG→SARC_ESS_LG, ESS_HG→SARC_ESS_HG,
SARC_LPS_UNSPEC→SARC_LPSunspec.

**Neuroendocrine — keep famous codes (`SCLC`, `PANNET`), `NET_`/`NEC_` for the rest:**
MID_NET→NET_MIDGUT, REC_NET→NET_RECTAL, LUNG_NET_LC→NET_LUNG,
LUNG_NET_LCNEC→NEC_LUNG_LC, MEC→NEC_MERKEL. (`NET_`/`NEC_` prefix encodes the
well-diff vs poorly-diff distinction; `SCLC` is NEC, `PANNET` is NET — recorded
via family + a `differentiation` note rather than renaming the famous codes.)

**One-separator normalization (collapse a split single concept token):**
HNSC_HPV_pos→HNSC_HPVpos, HNSC_HPV_neg→HNSC_HPVneg,
NBL_MYCN_amp→NBL_MYCNamp, NBL_MYCN_nonamp→NBL_MYCNnonamp,
LAML_ELN_Fav→LAML_ELNFav, LAML_ELN_Int→LAML_ELNInt, LAML_ELN_Adv→LAML_ELNAdv.
(Genuine hierarchy levels like `SARC_RMS_ERMS` keep their separators.)

### `rare` is another non-lineage bucket (dissolve it)
Like `pediatric-*` grouped by age, `family=rare` groups by *frequency*: it holds
`NUTM` (a carcinoma) and `CHOR` (a notochordal/axial bone tumour) — no shared
lineage. Resolution:
- `CHOR` → **sarcoma** (`SARC_CHOR`); WHO Soft Tissue & Bone classifies chordoma
  among notochordal tumours. (Already in `sarcoma_lineage_codes`.)
- `NUTM` → **carcinoma**; WHO files NUT carcinoma in Thoracic + Head & Neck as a
  midline poorly-differentiated carcinoma. No organ-specific carcinoma family
  fits, so place it in a new `carcinoma-other` (organ-less / undifferentiated
  carcinomas) lineage family. Retire `rare`.

Note: `UCS` (uterine carcinosarcoma) is a metaplastic **carcinoma** (stays
`carcinoma-gu`) — the only category it shares with NUT carcinoma is "carcinoma"
(both poorly-differentiated, lineage-obscured); they are not co-located because
they have no shared organ/lineage.

## Minor / accepted as-is
- `UCS` (carcinosarcoma) in `carcinoma-gu` — WHO views it as metaplastic
  carcinoma (epithelial); acceptable.
- `MESO` `carcinoma-mesothelial` — mesothelioma is its own WHO mesothelial
  category; label slightly imprecise but clear. (Could rename family `mesothelial`.)
- `carcinoma-gu` lumps urinary + male + female genital (WHO separates) —
  acceptable coarsening for now.

## Execution status / sequencing
1. This audit doc (the spec). ✅
2. Registry restructure: code renames + family reassignments + `pediatric` flag +
   backward-compat aliases + `differentiation` note for NET/NEC. Propagate renamed
   codes across the curated CSVs (cancer-type-genes, drivers, key-genes, lineage,
   surfaceome, therapy, tmb, fusions), `cohorts.py`, and tests.
3. Sarcoma data rebuild: TCGA-SARC histology atoms + computed `SARC`/`TCGA_SARC`/
   `SARC_<hist>` aggregates (build-time pooling of per-sample atoms).
4. Plot fix: lineage coloring uses the corrected `family` (pediatric sarcomas now
   color as sarcoma); regenerate CTA coverage bars + CTA-vs-TMB.
5. DATA_VERSION bump + tarball; deploy on merge. trufflepig = separate coordinated PR.

See also the curation issues filed upstream where tsarina owns the data
(tsarina #77 aliases, #78 expression threshold, #79 missing CT genes).
