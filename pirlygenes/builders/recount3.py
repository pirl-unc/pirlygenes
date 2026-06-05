"""recount3 → clean-TPM ingestion (build-time).

recount3 (Wilks 2021, Genome Biology) uniformly re-quantified ~750k public
human + mouse RNA-seq runs from SRA/GTEx/TCGA with the Monorail pipeline
(STAR alignment → base-level coverage), summarized to genes against several
annotations. We consume the **Gencode v26** gene summaries (``G026``).

Why we want it: many GEO series ship a *processed* matrix whose gene
universe is incomplete — HTSeq counts keyed to a stale Entrez/GTF set,
author-chosen FPKM tables, etc. — leaving genuine coverage holes. The
canonical example is GSE98894 (SRP107025), whose HTSeq counts never
annotated the near-identical CTA paralogs ``XAGE1A``/``XAGE1B`` so they are
*absent* (``not_measurable``), not zero. recount3's Gencode-v26 summaries
include them, so swapping the source replaces spotty coverage with the full
~60k-gene Gencode universe on one consistent scale.

──────────────────────────────────────────────────────────────────────────
NORMALIZATION — recount3 "gene sums" → clean TPM
──────────────────────────────────────────────────────────────────────────
recount3 ``gene_sums`` are NOT read counts. Each value is the **sum of
per-base read coverage over the gene's disjoint exonic bases** (the area
under the coverage curve, AUC, restricted to the gene model). So a gene of
exonic length L expressed at depth d contributes ≈ d·L — coverage scales
with *both* expression and length.

Step 1 — length-normalize to a TPM (per sample column j):

    rate[g,j]  = gene_sums[g,j] / bp_length[g]          # depth, length removed
    TPM[g,j]   = 1e6 * rate[g,j] / Σ_h rate[h,j]         # renormalize to 1e6

  ``bp_length`` is the gene's **exonic** length (Σ disjoint exon widths)
  under Gencode v26 — NOT the gene span. We read it from recount3's own
  annotation GTF (the score column), so the denominator matches exactly the
  bases the coverage was summed over. Using a gene-span length here would
  inflate the denominator for intron-rich genes and bias their TPM down.

  Note what cancels: gene_sums also carry a per-sample read-length and
  library-size factor (coverage ≈ reads · read_len). Because TPM is a
  *within-sample relative* quantity, those per-sample constants divide out
  in the Σ_h renormalization — so we need NEITHER the read length NOR the
  recount3 AUC/library-size metadata to get TPM. (Those matter only for
  cross-sample *count* methods, which we don't use.)

Step 2 — harmonize identifiers: strip the ``.<version>`` (and ``_PAR_Y``)
  suffix from the Gencode IDs and sum any collisions, yielding one row per
  unversioned ENSG (our standard ID convention).

Step 3 — clean TPM (v3): pin the technical-RNA groups (mtDNA, rRNA-like,
  mt-like pseudogene, polyA-bias lncRNA) **and ribosomal-protein mRNA +
  pseudogenes** to fixed per-gene reference values (Treehouse-PolyA medians,
  cohort-independent) and rescale the remaining genes to fill the 1e6 budget,
  via the ONE shared helper used everywhere else
  (:func:`pirlygenes.expression.normalize.clean_tpm_matrix` +
  :func:`clean_tpm_removal_mask`). Ribosomal proteins are excluded by
  default (housekeeping, not tumor-specific, and the dominant multi-mapping
  destabilizer of the zero-sum TPM denominator across pipelines); curated
  cancer targets are never censored.

The result is an ENSG × sample clean-TPM matrix on the same scale and with
the same technical-RNA treatment as our GDC/Treehouse/GEO shards, ready for
the usual per-(gene, cancer_code) median/q1/q3/n summarization.

──────────────────────────────────────────────────────────────────────────
COVERAGE / CAVEATS
──────────────────────────────────────────────────────────────────────────
* recount3's human SRA release is a frozen ~2019 snapshot — studies
  deposited after it (most 2023+ GEO series) are absent and must still be
  built from their native matrices or a FASTQ rebuild.
* STAR splits multimapping reads across near-identical paralogs, so the
  *per-paralog* value for things like XAGE1A/XAGE1B is approximate — but it
  is now *measured* (a real row) rather than absent, which lets the QC layer
  treat it as real-but-uncertain instead of not_measurable.
* Gencode v26 (recount3) vs GENCODE v36 (GDC) vs RSEM (Treehouse): all are
  "RNA-seq TPM" and mix acceptably for cohort-level summaries, same as the
  sources we already combine.

Routing samples → cancer codes and the final stat summarization are
per-source concerns left to the calling builder (it reuses the existing
series-matrix / metadata routing); this module's job is the source-agnostic
gene_sums → clean-TPM transform plus the recount3 fetch + metadata helpers.
"""
from __future__ import annotations

import gzip
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd

from pirlygenes.builders.gene_mapping import strip_version
from pirlygenes.builders.geo_matrix import normalize_to_tpm
from pirlygenes.expression.normalize import clean_tpm_matrix

ANNOTATION = "G026"  # Gencode v26 gene summaries
_S3_BASE = "https://recount-opendata.s3.amazonaws.com/recount3/release/human"
_ANNOTATION_GTF = (
    f"{_S3_BASE}/annotations/gene_sums/human.gene_sums.{ANNOTATION}.gtf.gz"
)


def gene_sums_url(srp: str) -> str:
    """S3 URL of the recount3 SRA gene-sums matrix for one study (SRP)."""
    return (
        f"{_S3_BASE}/data_sources/sra/gene_sums/{srp[-2:]}/{srp}/"
        f"sra.gene_sums.{srp}.{ANNOTATION}.gz"
    )


def metadata_url(srp: str, kind: str = "sra") -> str:
    """S3 URL of a recount3 SRA metadata table (``sra`` attributes, etc.).

    ``kind`` ∈ {``sra``, ``recount_project``, ``recount_qc``,
    ``recount_seq_qc``}; ``sra`` carries the GEO sample attributes used for
    routing (external_id = GSM, characteristics, …).
    """
    return (
        f"{_S3_BASE}/data_sources/sra/metadata/{srp[-2:]}/{srp}/"
        f"sra.{kind}.{srp}.MD.gz"
    )


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as h:
        while chunk := r.read(1 << 20):
            h.write(chunk)
    tmp.replace(dest)
    return dest


@lru_cache(maxsize=2)
def fetch_gene_annotation(cache_dir: Path) -> pd.DataFrame:
    """Gencode-v26 gene annotation: unversioned ENSG → ``bp_length`` (exonic)
    + ``Symbol``.

    Read from recount3's gene-sums annotation GTF, where the score column
    (field 6) holds the disjoint-exon ``bp_length`` and the ``gene_name``
    attribute holds the HGNC symbol. Collisions after version/``_PAR_Y``
    stripping are summed (bp_length) / first-wins (symbol) — a handful of
    PAR genes on chrY; the matching gene-sums rows are summed too, so the
    coverage/length rate stays consistent. Cached so a multi-source
    ``--all`` run parses the ~60k-row GTF once.
    """
    path = _download(_ANNOTATION_GTF, cache_dir / f"human.gene_sums.{ANNOTATION}.gtf.gz")
    rows: list[tuple[str, int, str]] = []
    name_re = re.compile(r'gene_id "([^"]+)".*?gene_name "([^"]+)"')
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "gene":
                continue
            m = name_re.search(c[8])
            if not m:
                continue
            ensg = strip_version(m.group(1))
            rows.append((ensg, int(c[5]), m.group(2)))
    ann = pd.DataFrame(rows, columns=["Ensembl_Gene_ID", "bp_length", "Symbol"])
    # PAR_Y / version collisions: sum lengths, keep first symbol
    ann = ann.groupby("Ensembl_Gene_ID", as_index=False).agg(
        bp_length=("bp_length", "sum"), Symbol=("Symbol", "first")
    )
    return ann.set_index("Ensembl_Gene_ID")


def fetch_gene_sums(srp: str, cache_dir: Path) -> pd.DataFrame:
    """recount3 gene-sums matrix for ``srp``: rows = versioned Gencode IDs,
    columns = SRR run accessions, values = exonic coverage sums."""
    path = _download(gene_sums_url(srp), cache_dir / f"sra.gene_sums.{srp}.{ANNOTATION}.gz")
    return pd.read_csv(path, sep="\t", comment="#", index_col=0)


def gene_sums_to_tpm(gene_sums: pd.DataFrame, bp_length: pd.Series) -> pd.DataFrame:
    """Length-normalize coverage gene-sums to TPM (step 1 + 2 of the recipe).

    Returns an unversioned-ENSG × sample TPM matrix (each column sums to 1e6
    over genes with a known length). See the module docstring for the math.

    Coverage-over-exonic-bases divided by exonic length is mathematically the
    same shape as read-counts divided by length, so this delegates to the
    shared :func:`pirlygenes.builders.geo_matrix.normalize_to_tpm`
    ``raw_counts`` path — recount3 takes the *same* length-normalization
    route as every count-based GEO/GDC builder, not a private one.
    """
    gs = gene_sums.copy()
    gs.index = [strip_version(i) for i in gs.index]
    gs = gs.groupby(level=0).sum()                     # collapse version/_PAR_Y dups
    lengths_kb = (bp_length / 1000.0).rename("gene_length_kb")
    return normalize_to_tpm(gs, unit="raw_counts", gene_lengths_kb=lengths_kb)


def to_clean_tpm(tpm: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared clean-TPM transform (step 3): pin censored genes to
    per-gene reference values, rescale the rest to fill 1e6.

    ``annotation`` supplies Symbol per ENSG (from
    :func:`fetch_gene_annotation`); rows missing from it fall back to the
    ENSG as symbol so they're simply kept.
    """
    gene_table = pd.DataFrame({
        "Symbol": annotation["Symbol"].reindex(tpm.index).fillna(
            pd.Series(tpm.index, index=tpm.index)
        ).to_numpy(),
        "Ensembl_Gene_ID": tpm.index.to_numpy(),
    })
    gene_table.index = tpm.index
    return clean_tpm_matrix(tpm, gene_table=gene_table)


def recount3_clean_tpm(srp: str, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full path: fetch SRP gene-sums + annotation, return (clean-TPM matrix,
    annotation). The matrix is unversioned-ENSG × SRR; route + summarize it
    with the caller's per-source logic."""
    annotation = fetch_gene_annotation(cache_dir)
    gene_sums = fetch_gene_sums(srp, cache_dir)
    tpm = gene_sums_to_tpm(gene_sums, annotation["bp_length"])
    return to_clean_tpm(tpm, annotation), annotation


def parse_sample_attributes(packed: str) -> dict[str, str]:
    """Unpack recount3's ``key;;value|key;;value`` sample_attributes string
    into a dict (keys/values stripped; lower-case keys as recount3 stores)."""
    out: dict[str, str] = {}
    for kv in str(packed).split("|"):
        if ";;" in kv:
            k, v = kv.split(";;", 1)
            out[k.strip()] = v.strip()
    return out


def aggregate_runs_to_samples(
    gene_sums: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    keep_runs: set[str] | None = None,
    run_col: str = "external_id",
    sample_col: str = "sample_acc",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse per-run (SRR) coverage to per-biological-sample.

    recount3 columns are sequencing runs; a sample sequenced over several
    lanes appears as several SRR columns. Coverage is additive, so we **sum**
    a sample's runs *before* length-normalizing (TPM is a per-sample
    quantity). Optionally restrict to ``keep_runs`` first (per-source
    inclusion filter).

    Returns ``(sample_gene_sums, sample_meta)`` — gene_sums with one column
    per ``sample_col`` value, and a one-row-per-sample metadata frame
    (indexed by sample id, carrying the first run's attributes/title).
    """
    meta = metadata.copy()
    meta[run_col] = meta[run_col].astype(str)
    runs = [c for c in gene_sums.columns if c in set(meta[run_col])]
    if keep_runs is not None:
        runs = [r for r in runs if r in keep_runs]
    meta = meta[meta[run_col].isin(runs)]
    run_to_sample = dict(zip(meta[run_col], meta[sample_col].astype(str)))
    gs = gene_sums[runs].copy()
    gs.columns = [run_to_sample[c] for c in gs.columns]
    sample_gs = gs.T.groupby(level=0).sum().T          # sum runs per sample
    sample_meta = meta.drop_duplicates(sample_col).copy()
    sample_meta[sample_col] = sample_meta[sample_col].astype(str)
    sample_meta = sample_meta.set_index(sample_col).reindex(sample_gs.columns)
    return sample_gs, sample_meta


def fetch_sample_metadata(srp: str, cache_dir: Path) -> pd.DataFrame:
    """recount3 ``sra`` metadata for ``srp`` (one row per SRR run).

    Carries ``external_id`` (GSM), ``sample_title``, and the pipe-packed
    ``sample_attributes`` (GEO characteristics) needed to route runs to
    cancer codes. Indexed by SRR (``rail_id``/``external_id`` columns vary
    by release; the run accession is in ``run_acc``)."""
    path = _download(metadata_url(srp), cache_dir / f"sra.sra.{srp}.MD.gz")
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)
