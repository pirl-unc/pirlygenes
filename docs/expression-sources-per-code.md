# Expression-source manifest — per cancer code

One row per cancer code in `pirlygenes/data/cancer-type-registry.csv`
giving the best-identified public source of per-sample expression
data, the implementation status, and a pointer to the
fetcher/builder script.

This is the "find one or more sources for each one" deliverable for
the expression-data refresh project. See also:
- `docs/expression-data-audit.md` for the bucket-level summary.
- `docs/expression-data-refresh-plan.md` for the rolling milestone
  plan.

## Headline counts (after current session)

- **94 / 125 codes** with v5.3 extended stats populated (+93 since
  session start).
- **31 codes** still need work. All have identified sources
  (or are documented as no-robust-source rare entities); none are
  blocked by missing knowledge.

## Status A — v5.3 populated (94 codes)

Bundled via per-sample builders. Re-runnable via
`pirlygenes build <source-id>` or `pirlygenes build <cancer-code>`.

| group | codes | builder |
|---|---|---|
| Treehouse PolyA, pediatric/sarcoma | ATRT, EWS, HEPB, MBL, NUTM, OS, RMS_ARMS/ERMS/PRMS/SSRMS, SARC_LMS/LPS_UNSPEC/MYXFIB/SYN/UPS | `scripts/sweep_treehouse_polya_cohorts.py` |
| Treehouse PolyA, sarcoma rare | SARC_ANGIO, SARC_ASPS, SARC_DSRCT, SARC_EPITH, SARC_GIST, SARC_IMT, SARC_IFS, SARC_MPNST, SARC_EHE, SARC_LGFMS | `scripts/sweep_sarc_rare_subtypes.py` |
| Treehouse RiboD | CHOR, RB | `scripts/sweep_treehouse_ribod_cohorts.py` |
| Treehouse TCGA subset | ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LAML, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SARC_DDLPS, SARC_PLEOLPS, SARC_WDLPS, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, UVM | `scripts/sweep_treehouse_tcga_cohorts.py`, `scripts/sweep_treehouse_tcga_glioma_split.py`, `scripts/sweep_treehouse_sarc_subtypes.py`, `scripts/sweep_sarc_rare_subtypes.py` |
| Treehouse TCGA × cBioPortal subtype overlays | BRCA_Basal, BRCA_HER2, BRCA_LumA, BRCA_LumB, BRCA_Normal, HNSC_HPV_pos, HNSC_HPV_neg, LUAD_EGFR, LUAD_KRAS, LUAD_STK11 | `scripts/sweep_treehouse_tcga_brca_pam50.py`, `_hnsc_hpv.py`, `_luad_mutations.py` |
| GDC per-project | BL, MM, B_ALL, T_ALL, NBL_MYCN_amp, NBL_MYCN_nonamp, RT, WILMS | `scripts/build_bl_gdc_*`, `build_mmrf_*`, `build_target_all_*`, `build_target_subprojects.py` |
| BeatAML (GDC + ELN approx) | LAML_APL, LAML_ELN_Fav, LAML_ELN_Int, LAML_ELN_Adv | `scripts/build_beataml_reference_expression.py` |
| Other per-sample | CLL (Broad CLLMAP), CTCL (GSE171811 scRNA), CML/MDS/MCL/MPN (GEO heme builders), PANNET (GSE118014), CHON (GSE299759), SCLC (cBioPortal UCologne datahub) | various build_*.py |

## Status B — missing (31 codes)

Grouped by blocker type.

### Single external GEO accession needed (1 code)

| code | source | status |
|---|---|---|
| **SARC_MYXLPS** (Myxoid Liposarcoma) | TCGA-SARC has none; need external like **GSE128064** (Hofvander 2018) | not yet implemented; small builder mirroring `build_chon_reference_expression.py` would do it |

### Aggregate codes that don't need own rows (1 code)

| code | resolution |
|---|---|
| **NBL** (Neuroblastoma umbrella) | Aggregate of NBL_MYCN_amp + NBL_MYCN_nonamp (both populated). Per audit convention, umbrella codes inherit data via children. Could add a re-aggregated NBL row by combining cached per-sample matrices, but not strictly necessary. |

### Subtype overlays — need paper supplement (8 codes)

These require parsing supplementary tables from a specific paper to
get sample → subtype labels, then overlaying onto an existing parent
cohort. cBioPortal does not expose these classifications.

| codes | parent cohort (already bundled) | classifier source |
|---|---|---|
| MBL_G3, MBL_G4, MBL_SHH, MBL_WNT | MBL n=125 from Treehouse PolyA | Cavalli 2017 (PMID 28617753) supplements (DKFZ MBL study). Sample IDs in supplement use ICGC_MB### / SJMB### conventions that don't directly join Treehouse `th_dataset_id` (THR33_xxxx_S01 etc.). Need EGAZ → ICGC_MB mapping or SJMB cross-ref. |
| SCLC_ASCL1, SCLC_NEUROD1, SCLC_POU2F3, SCLC_YAP1 | SCLC n=81 from cBioPortal datahub | Gay 2021 (PMID 33442380) supplementary table. Their sample IDs may or may not match `sclc_ucologne_2015_S000xx`. Worth trying. |

### Literature-curated, per-accession custom builder (21 codes)

Each needs a per-accession script (download supplementary, parse
format-specific structure, harmonize gene IDs, run standard stats
pipeline). Some are microarray and need RMA normalization + probe→gene
mapping; flag with a tech note in `notes`.

| code | candidate source | tech | rough n | est effort |
|---|---|---|---|---|
| HL | **GSE12453** (Steidl 2010, microdissected HRS cells) | microarray | ~30 | 2 hr (probe→gene + RMA) |
| FL | **GSE65135** (Dave 2004) or **GSE16131** (Brodtkorb 2014) | microarray | ~50 | 2 hr |
| HCL | **GSE16729** (Tiacci 2011) | microarray | ~20 | 2 hr |
| NPC | **GSE12452** (Sengupta 2006) + **GSE34573** (Bose 2013) | microarray | ~30+30 | 2-3 hr |
| MEC | **GSE161517** (Verhaegen 2022) | RNA-seq (RAW.tar) | ~30 | 1-2 hr (extract per-sample featureCounts) |
| MTC | **GSE32662** (Pringle 2012) | microarray | ~15-20 | 2 hr |
| LUNG_NET_LC | **GSE118336** (Alcala 2019) | RNA-seq (RAW.tar) | ~35 | 1-2 hr |
| LUNG_NET_LCNEC | **GSE118336** (Alcala 2019, same accession; histology split) | RNA-seq | ~75 | (same as LC) |
| MID_NET | **GSE65286** (Andersson 2016) | RNA-seq | ~30 | 1-2 hr |
| GCTB | **GSE150404** (Lutfi 2022) | RNA-seq (RAW.tar) | ~10-15 | 1-2 hr |
| SARC_DFSP | **GSE51445** | microarray | ~10 | 2 hr |
| SARC_CCS | **GSE160741** (Bartenstein 2021) | RNA-seq | ~10 | 1-2 hr |
| SARC_PEC | scattered small studies; **GSE189557** | varies | ≤10 | 2-3 hr; flag low n |
| SARC_KS | **GSE16162** + scattered | microarray | ~20 | 2-3 hr |
| SARC_SFT | **GSE51425** (Vivero 2014) | microarray | ~10 | 2 hr |
| SARC_EMC | **GSE21050** (very rare) | microarray | ≤10 | 2 hr; flag low n |
| ACINIC | **GSE40631** (~15 salivary panel subset) | microarray | ~15 | 2 hr |
| ADCC | **GSE59702** (Stenman 2014) + small GEO sets | microarray | ~20 | 2 hr |
| ESS_HG | **GSE36110** + ICGC | rare | <20 | 2-3 hr |
| ESS_LG | same as ESS_HG | rare | <20 | (same) |
| PCN | closest is MM; small GEO sets | varies | <20 | 2 hr; flag overlap |

### Total effort estimate for the remaining 31

- SARC_MYXLPS: ~1 hr
- NBL umbrella: ~30 min
- 8 subtype overlays: ~2-4 hr each = 16-32 hr (depending on whether sample ID joins work)
- 21 lit-curated: ~2 hr each avg = 42 hr

**Total: 60-75 hours of focused work** to bring coverage from 94 → 125.

That's 1.5-2 weeks at 8 hrs/day, multi-session.

## Run any builder via the CLI

```
pirlygenes build list                      # enumerate all sources + scripts
pirlygenes build <source-id>               # e.g. pirlygenes build cgci-blgsp
pirlygenes build <CANCER_CODE>             # e.g. pirlygenes build BL
pirlygenes build target-nbl -- --only TARGET-NBL  # passthrough args
```

The dispatcher reads `pirlygenes/data/expression_sources.yaml` and
invokes the right script with conventional --summary-output /
--samples-output / --cache-dir defaults.

## Notes for new-builder authors

- Always write per-source shards via
  `pirlygenes.expression.stats.upsert_to_shard` — never the legacy
  single-file path.
- Microarray cohorts need a `notes` cell flagging "microarray; not
  directly TPM-comparable" so consumers can hold tech caveats.
- Subtype overlays use distinct `source_cohort` tags
  (`<PARENT_SOURCE>_<CLASSIFIER>`) so each subtype set ends up in
  its own shard and doesn't push the parent past the 100 MB GitHub
  limit.
