# Expression-source manifest — per cancer code

> **Historical snapshot (May 2026).** The per-code table below predates the
> Phase-C cancer-type renames and the later source expansion, so some rows use
> retired codes (`EWS`→`SARC_EWS`, `OS`→`SARC_OS`, `RMS_*`→`SARC_RMS_*`,
> `CHOR`→`SARC_CHOR`, `PANNET`→`NET_PANCREAS`, `MID_NET`→`NET_MIDGUT`,
> `REC_NET`→`NET_RECTAL`, `LUNG_NET_LC`→`NET_LUNG`,
> `LUNG_NET_LCNEC`→`NEC_LUNG_LARGECELL`, `NBL_MYCN_*`→`NBL_MYCN*`,
> `LAML_ELN_*`→`LAML_ELN*`) and stale counts. For the **current** inventory
> query the live accessors instead:
> ```python
> from pirlygenes.gene_sets_cancer import cancer_type_registry
> from pirlygenes.expression import available_cancer_expression_references
> cancer_type_registry()                     # registry codes (source of truth)
> available_cancer_expression_references()   # packaged per-cohort references
> ```
> Old codes still resolve via `resolve_cancer_type` aliases. See issue #315.

One row per cancer code in `pirlygenes/data/cancer-type-registry.csv`
giving the best-identified public source of per-sample expression
data, the implementation status, and a pointer to the
fetcher/builder script.

This is the "find one or more sources for each one" deliverable for
the expression-data refresh project. See also:
- `expression-data-audit-2026-05.md` for the bucket-level summary.
- `../expression-data-refresh-plan.md` for the rolling milestone
  plan.

## Headline counts (after current session)

- **112 / 125 codes** with v5.3 extended stats populated
  (+111 since session start, **90% coverage**).
- **13 codes** still need work. Each is documented below with the
  specific blocker (controlled-access source / FASTQ-only requiring
  alignment / microarray-only / no public per-sample data at all).

## Status A — v5.3 populated (112 codes)

Bundled via per-sample builders. Re-runnable via
`pirlygenes build <source-id>` or `pirlygenes build <cancer-code>`.

| group | codes | builder |
|---|---|---|
| Treehouse PolyA, pediatric/sarcoma | ATRT, EWS, HEPB, MBL, NUTM, OS, RMS_ARMS/ERMS/PRMS/SSRMS, SARC_LMS/LPS_UNSPEC/MYXFIB/SYN/UPS | `scripts/sweep_treehouse_polya_cohorts.py` |
| Treehouse PolyA, sarcoma rare | SARC_ANGIO, SARC_ASPS, SARC_DSRCT, SARC_EPITH, SARC_GIST, SARC_IMT, SARC_IFS, SARC_MPNST, SARC_EHE, SARC_LGFMS | `scripts/sweep_sarc_rare_subtypes.py` |
| Treehouse RiboD | CHOR, RB | `scripts/sweep_treehouse_ribod_cohorts.py` |
| Treehouse TCGA subset | ACC, BLCA, BRCA, CESC, CHOL, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LAML, LGG, LIHC, LUAD, LUSC, MESO, OV, PAAD, PCPG, PRAD, READ, SARC, SARC_DDLPS, SARC_PLEOLPS, SARC_WDLPS, SKCM, STAD, TGCT, THCA, THYM, UCEC, UCS, UVM | `scripts/sweep_treehouse_tcga_cohorts.py`, `scripts/sweep_treehouse_tcga_glioma_split.py`, `scripts/sweep_treehouse_sarc_subtypes.py`, `scripts/sweep_sarc_rare_subtypes.py` |
| Treehouse TCGA × cBioPortal subtype overlays | BRCA_Basal, BRCA_HER2, BRCA_LumA, BRCA_LumB, BRCA_Normal, HNSC_HPV_pos, HNSC_HPV_neg, LUAD_EGFR, LUAD_KRAS, LUAD_STK11 | `scripts/sweep_treehouse_tcga_brca_pam50.py`, `_hnsc_hpv.py`, `_luad_mutations.py` |
| Treehouse MBL marker-gene subgroups | MBL_WNT, MBL_SHH, MBL_G3, MBL_G4 | `scripts/build_mbl_subgroups_marker_classifier.py` (approximate) |
| SCLC TF-dominance subtypes | SCLC_ASCL1, SCLC_NEUROD1, SCLC_POU2F3, SCLC_YAP1 | `scripts/build_sclc_subtypes_tf_dominance.py` (approximate) |
| Treehouse pediatric/other | NPC (n=4), NBL umbrella | `scripts/sweep_sarc_rare_subtypes.py`, `build_target_subprojects.py` |
| Generic GEO matrix (Tier A) | HL (n=5), FL (n=6), SARC_CCS (n=5), SARC_PEC (n=69), SARC_KS (n=10), ADCC (n=99) | `scripts/build_geo_matrix.py` (YAML-driven generic builder); per-source config in expression_sources.yaml |
| IARC DRMetrics LNEN | LUNG_NET_LC (n=118), LUNG_NET_LCNEC (n=69) | `scripts/build_lnen_drmetrics_reference_expression.py` |
| GDC per-project | BL, MM, B_ALL, T_ALL, NBL_MYCN_amp, NBL_MYCN_nonamp, RT, WILMS | `scripts/build_bl_gdc_*`, `build_mmrf_*`, `build_target_all_*`, `build_target_subprojects.py` |
| BeatAML (GDC + ELN approx) | LAML_APL, LAML_ELN_Fav, LAML_ELN_Int, LAML_ELN_Adv | `scripts/build_beataml_reference_expression.py` |
| Other per-sample | CLL (Broad CLLMAP), CTCL (GSE171811 scRNA), CML/MDS/MCL/MPN (GEO heme builders), PANNET (GSE118014), CHON (GSE299759), SCLC (cBioPortal UCologne datahub) | various build_*.py |

## Status B — still missing (13 codes)

Each row below explains why this code can't be built from a clean
single-file public per-sample RNA-seq source. Almost all could be
filled if we add either (a) a FASTQ alignment pipeline, (b) a
microarray probe→gene mapper, or (c) controlled-access credentials.

| code | candidate source | blocker | rough effort to unblock |
|---|---|---|---|
| **MEC** (Merkel Cell Carcinoma) | SRA **PRJNA775071** (Sundqvist 2022, n=102) | FASTQ only on SRA — no processed matrix | Add a FASTQ→Salmon quant pipeline (~1 day; or use ARCHS4 if it has the runs) |
| **MTC** (Medullary Thyroid Carcinoma) | GSA-Human **HRA002698** (Shi 2022, n=101) | Controlled-access (Chinese GSA-Human application required) | Apply for GSA-Human access; or use the small 2026 E-MTAB-15198 (n=72) once its data files become available |
| ~~MID_NET~~ (Midgut Carcinoid) | **DONE** 2026-05-27 via `scripts/build_gse98894_midgut_net.py` (GSE98894 RAW.tar Entrez→ENSG via NCBI gene_info → length-norm TPM). 81 SI-NET liver mets tagged `tumor_origin=metastasis, metastasis_site=liver`. Also wrote PANNET and REC_NET shards from the same cohort. | resolved | n/a |
| **GCTB** (Giant Cell Tumor of Bone) | GEO **GSE149209** (Khazaei 2020, n=38) | Cell lines, not primary tumors | Use anyway with explicit cell-line caveat in notes, or wait for a primary-GCTB cohort to be deposited |
| **HCL** (Hairy Cell Leukemia) | GEO **GSE63790** (n=6) | Microarray (Affy U133+2), not RNA-seq | Add a microarray builder with RMA + probe→gene mapper (the Affy annotation table is available; ~3-4 hr per platform) |
| **SARC_MYXLPS** (Myxoid Liposarcoma) | GEO **GSE21050** (n=~20 MLPS subset of 310 STS) | Microarray (Affy HG-U133+2); the audit's GSE128064 was methylation (wrong); GSE28866 is 3SEQ peak counts (not gene expression) | Microarray builder needed; or wait for a public bulk RNA-seq MLPS cohort |
| **SARC_EMC** (Extraskeletal Myxoid Chondrosarcoma) | GEO **GSE4303** (Subramanian 2005, n=10) | Old-school microarray (mixed cDNA platforms GPL2937-3335) | Microarray builder + multi-platform annotation; rare so low priority |
| **SARC_DFSP** (Dermatofibrosarcoma Protuberans) | None public | Zhang 2022 cohort (n=14) never deposited; closest fallback is GSE3930 microarray | Aggregate small case-series; flag as low-n |
| **SARC_SFT** (Solitary Fibrous Tumor) | GSA-Human **HRA008581** (Zhao 2025, n=88 CNS-SFT) | Controlled-access (Chinese GSA-Human application required) | Apply for GSA-Human access |
| **ESS_HG** (High-grade Endometrial Stromal Sarcoma) | GEO **GSE85383** (Yoshida 2017, n=4) | Microarray (Agilent SurePrint G3) | Microarray builder needed |
| **ESS_LG** (Low-grade Endometrial Stromal Sarcoma) | Same GSE85383 (n=9) | Microarray | Same as ESS_HG |
| **PCN** (Solitary Plasmacytoma) | None | No public bulk RNA-seq cohort of solitary plasmacytoma exists (distinct from MM CoMMpass) | Aggregate small case-series; or accept MMRF as proxy with disease-distinction caveat |
| **ACINIC** (Acinic Cell Carcinoma) | SRA **PRJNA433667** (Lee 2020, n=21) | FASTQ only on SRA — agent's earlier GSE76354 candidate was cell-line, not patient tumors | Add FASTQ→Salmon pipeline |

### Truly blocked: 5
- **PCN**, **SARC_DFSP**: no public per-sample RNA-seq source identified

### Blocked on adding a missing infrastructure piece: 7
- FASTQ→TPM pipeline (Salmon/STAR-RSEM): would unlock **MEC**, **ACINIC**
- **GPL570 microarray builder** (built 2026-05-27 in `pirlygenes/builders/affy_gpl570.py` + `scripts/build_gpl570_microarray.py`; awaiting a vetted GEO source for each — initial candidates GSE63790 [HCL: drug timecourse] and GSE21050 [SARC_MYXLPS: cohort is UPS/LMS/DDLPS, zero MLPS] were dropped on audit). Would unlock **HCL**, **SARC_MYXLPS** when a real primary-tumor cohort is found.
- Other-platform microarray support (Affy GPL2937 / Agilent GPL22303): would unlock **SARC_EMC**, **ESS_HG**, **ESS_LG**
- Controlled-access credentials (Chinese GSA-Human): would unlock **MTC**, **SARC_SFT**

### Blocked on per-cohort custom work: 1
- **GCTB**: accept cell-line GSE149209 with caveat, or wait for primary-tumor cohort

### Resolved 2026-05-27
- **MID_NET**, **PANNET** (liver-met variant), **REC_NET**: built via `scripts/build_gse98894_midgut_net.py` from GSE98894 (Alvarez 2018). All tagged `tumor_origin=metastasis, metastasis_site=liver` since the cohort is liver biopsies. REC_NET registry code added with this build.
- **Schema v5.4**: added `tumor_origin` and `metastasis_site` columns to every shard. All 33 existing shards backfilled (primary | mixed) via `scripts/backfill_tumor_origin.py`.
- **NCBI Entrez↔HUGO helper** (`pirlygenes/builders/ncbi_gene_info.py`): caches `Homo_sapiens.gene_info.gz` once, exposes `harmonize_entrez_via_ncbi(matrix, …)` for any Entrez-keyed GEO matrix going forward.

Grouped by blocker type.

### Single external GEO accession needed (1 code)

| code | source | status |
|---|---|---|
| **SARC_MYXLPS** (Myxoid Liposarcoma) | TCGA-SARC has none; need external like **GSE128064** (Hofvander 2018) | not yet implemented; small builder mirroring `build_chon_reference_expression.py` would do it |

### Aggregate codes (resolved as filled subtypes)

NBL umbrella now populated (n=155 aggregate over MYCN-amp + nonamp).
MBL and SCLC subgroups built via approximate marker-gene / TF-dominance
classifiers; rigorous paper-supplement-based versions can replace these
in a future pass.

### Literature-curated, per-accession custom builder (20 codes)

Each needs a per-accession script (download supplementary, parse
format-specific structure, harmonize gene IDs, run standard stats
pipeline). Many of the audit-suggested accessions in this list still
need verification — quick recon during the session found several
"candidate" accessions are wrong tumor type. Below is the working list;
each row needs a real per-cohort accession hunt before implementation.

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

### Total effort estimate for the remaining 21

- SARC_MYXLPS: ~1-2 hr (GSE128064 is methylation, not RNA; the
  audit's GEO accession was wrong; needs a real RNA-seq MLPS search)
- 20 lit-curated: ~1-3 hr each avg = 30-60 hr
  - microarray accessions need probe→gene mapping per platform (GPL)
  - several audit-suggested accessions turned out to be the wrong
    tumor type on quick recon (e.g. GSE161517 is prostate ChIP-seq,
    not Merkel cell RNA-seq); a real recon pass is required before
    each builder

**Total: 30-60 hours of focused per-accession work** to bring
coverage from 104 → 125. Multi-session.

The truly rare entities (ACINIC, ESS_HG/LG, PCN, SARC_PEC,
SARC_EMC) may not have any clean public per-sample RNA-seq source
in 2026 — they're best documented as "no robust source identified;
small case-series available but each <20 samples". The session-end
pirlygenes coverage of 83% reflects this realistic ceiling for
many of these very-rare cancer types.

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
