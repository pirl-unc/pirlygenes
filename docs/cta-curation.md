# CTA provenance and curation

This report consumes the exact oncoref 1.8.206 implementation from
[oncoref PR #562](https://github.com/pirl-unc/oncoref/pull/562). It combines the
complete-paper source expansion, #559 gene-level citations and pirlygenes #629
count/plot correction. The committed report was regenerated against the published
PyPI wheel, with the installed and imported owner versions checked.

The active starting pool is the complete union of the ten-paper minimum cover
of the expanded retained panel. It contains **2,537 coding candidates** and
produces **624 retained genes** (all 297 previously retained plus 327 added by
source expansion). Ten is the exact minimum among eleven fully imported source
papers; it is not a minimum over all literature.

| Stage | Genes or source identities |
|---|---:|
| Full selected paper union | 3,895 |
| Canonically mapped | 3,654 |
| Protein coding | 2,537 |
| Family exclusions applied | 2,523 |
| HPA restriction | 880 |
| Default specificity and expression rules | 624 |

The 880 HPA passes are an intermediate gate. Outcome charts count the 624-gene
public default. TRIM64 remains excluded. All active candidates have primary-paper
DOIs and exact source locations; nine historical-only rows remain separately
archived. The per-gene report flags 29 legacy retained genes whose selected-paper links
provide normal-reproductive expression only. Their previous retention policy is
preserved, with that tumor-expression evidence limitation explicit.

Coding eligibility in this report uses oncoref's pinned canonical annotation.
The separate tsarina partition uses an Ensembl 112 coding background. Four
rejected candidates differ across those annotations: CFAP144P1
(ENSG00000164556; Ensembl 112 processed pseudogene), SMIM10L2B-AS1
(ENSG00000228372), RBAKDN (ENSG00000273313), and MSL3B (ENSG00000293137;
the latter three are Ensembl 112 lncRNAs). All four fail HPA filtering and
remain outside the default panel; the partition's retained set is exactly
oncoref's 624 genes.

Paper membership does not establish locus-specific protein, HLA
presentation, immune recognition, normal-tissue exclusivity or clinical safety.

[All 12 regenerated figures, vector PDF](audits/cta-unified-20260924/pirlygenes-cta-curation-figures.pdf)
· [Figure index and audits](audits/cta-unified-20260924/index.md)
· [Candidate provenance](audits/cta-unified-20260924/cta-candidate-provenance.csv)
· [Minimum-cover certificate](audits/cta-unified-20260924/cta-minimum-source-cover.json)

![Source Venn](audits/cta-unified-20260924/cta-source-venn.png)
![Additional source Venns](audits/cta-unified-20260924/cta-landscape-source-venn.png)
![Intake funnel](audits/cta-unified-20260924/cta-stage-funnel.png)
![Per-paper funnel](audits/cta-unified-20260924/cta-publication-funnel.png)
![Outcome](audits/cta-unified-20260924/cta-filter-outcome.png)

The remaining historical-resource and placental Venn diagrams are explicitly
labelled audits. The all-paper matrix and exact-intersection table cover every
selected source, including Bai's single EGFL6 target. All counts use unique
canonical loci; nested tables within a paper are not counted as separate papers.

Run in a dedicated environment matching the pinned owner version:

```sh
.venv-figures/bin/python analyses/cta_curation_figures.py
pirlygenes plot cta-curation --out cta_curation_out
```

The batch wrapper verifies the installed and imported owner versions before and
after rendering. Figures delegate to oncoref's owner implementation and ship
300-dpi PNG and vector PDF pairs. This CTA-only report does not represent a full
regeneration of tumor-cohort, survival, coverage or response analyses. Expression
artifact DATA_VERSION pins are unchanged.
