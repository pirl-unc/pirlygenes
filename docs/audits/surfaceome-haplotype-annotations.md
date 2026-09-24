# Surfaceome haplotype annotation audit

The [original surfaceome](https://wlab.ethz.ch/surfaceome/) (PMID:30373828)
explicitly lists KIR3DS1 / ENSG00000276534 and KIR2DL2 / ENSG00000275546.
These are alternate haplotypes, not primary-assembly replacements. Keep their
published identifiers in the surface-protein panel. [UniProt O43469](https://www.uniprot.org/uniprotkb/O43469/entry)
also cross-references the KIR3DS1 accession.

The complete Ensembl patch/haplotype annotations show:

| Source symbol | Published ID | Ensembl 111 name | Ensembl 112 name |
| --- | --- | --- | --- |
| KIR3DS1 | ENSG00000276534 | blank | blank |
| KIR2DL2 | ENSG00000275546 | KIR2DL2 | blank |

The integrity check previously classified an unnamed haplotype as a gene swap
when another haplotype had the symbol. It now recognizes only these two exact,
source-supported pairs in `surface-proteins`, and only when the current name is
blank. A named different gene, missing accession, different ID or different
source table still fails. No gene-panel membership or identifier is changed.

Validation ran the complete curated-panel guard against the actual
[Ensembl 111 GTF](https://ftp.ensembl.org/pub/release-111/gtf/homo_sapiens/Homo_sapiens.GRCh38.111.chr_patch_hapl_scaff.gtf.gz)
and [Ensembl 112 GTF](https://ftp.ensembl.org/pub/release-112/gtf/homo_sapiens/Homo_sapiens.GRCh38.112.chr_patch_hapl_scaff.gtf.gz).
Both passed after the correction. Exact GTF hashes and contigs are recorded in
[surfaceome-haplotype-annotations.json](surfaceome-haplotype-annotations.json).
The 15-test ontology module includes positive and corruption cases; lint passes.
