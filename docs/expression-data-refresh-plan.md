# Expression-data refresh plan

Multi-session project. Owner: this is the rolling work plan; update it
as milestones land.

## Goal

Bring every cancer cohort in `pirlygenes/data/cancer-reference-expression.csv.gz`
into the same normalized TPM space, with a uniform extended-stat suite
(median / Q1 / Q3 / mean / std / min / max / p5 / p10 / p90 / p95 plus
n_samples / n_detected), generated from per-sample data wherever
possible. Add the 32 TCGA bulk-adult cohorts that are currently
missing from the long-format reference table.

Driven by a `pirlygenes downloads / build / plot` CLI backed by a
single YAML registry of data sources.

## Status

**Session 1, 2026-05-26 — landed:**

- Package-boundary clarification memorialized (pirlygenes CAN have a
  CLI for cohort-level ops; only `analyze` stays in trufflepig).
- Plan doc (this file).
- `pirlygenes/data/expression_sources.yaml` — registry of every
  source.
- `pirlygenes/cli.py` resurrected as an argparse CLI hosting
  `downloads`, `build`, `plot` subcommands. `analyze` and siblings
  keep returning the migration message.
- `pirlygenes downloads list` and `pirlygenes downloads cache-dir`
  implemented (read-only). Other subcommands scaffold and raise
  `NotImplementedError` pointing at the relevant milestone below.
- CLAUDE.md "Package boundary" section rewritten to reflect the
  post-v5.2 split.
- PyYAML added to dependencies.

**Session 2, 2026-05-26 — landed:**

- **Milestone 1 (schema extension).** Added 15 new columns to
  `cancer-reference-expression.csv.gz` (raw: std/min/max/p5/p10/p90/p95;
  clean: mean/std/min/max/p5/p10/p90/p95). Append-only — existing
  17-column legacy order preserved. New columns are NaN for every
  existing row; future builder re-runs will populate them.
- `pirlygenes/expression/stats.py` — shared `REFERENCE_COLUMNS`
  constant + `assign_stats(out, raw, clean)` /
  `compute_cohort_stats()` / `round_stat_columns()` helpers so every
  builder lands rows with the same shape.
- All 6 per-cohort builders (`build_bl_gdc`, `build_cllmap`,
  `build_ctcl_scrna`, `build_geo_heme`, `build_mmrf`,
  `build_target_all`) updated to use the shared helpers.
  `import_cancer_specific_expression.py` updated too — summary-only
  imports keep extended-stat columns NaN.
- `pirlygenes data list` + `pirlygenes/data_inventory.py` — read-only
  inventory of the bundled `cancer-reference-expression.csv.gz`,
  grouped per source cohort with per-cancer-code row counts and
  n_samples.

**Note on data state:** the schema is extended, but no builder has
been *re-run*. Every row's new-column values are NaN until the
relevant `build_*` script runs end-to-end against its data source.
That's milestones 3-4 (TCGA fresh) and 8 (existing-cohort sweep).

**Session 3, 2026-05-26 evening — landed:**

- **Milestone 6 (Treehouse compendium fetcher).** Downloaded the
  Treehouse 25.01 PolyA + RiboD compendia (6.19 GB + 1.1 GB), built
  per-cohort summarizer (`pirlygenes/builders/treehouse.py`), added
  thin sweep wrappers for PolyA / RiboD / TCGA-subset. 47 cohorts
  built via this path total: 15 pediatric PolyA + 2 RiboD (CHOR/RB)
  + 31 TCGA bulk-adult (ACC, BLCA, ..., UVM).
- **Milestone 8 (per-cohort re-runs) — partial.** GEO heme (CML /
  MDS / MCL / MPN), CLLMAP (CLL), CTCL re-ran with v5.3 stats. GDC
  builders BL / MMRF / TARGET-ALL queued in background.
- **Sharded data layout.** GitHub rejected pushing a 147 MB CSV (over
  the 100 MB hard limit). Split into per-source-cohort shards under
  `pirlygenes/data/cancer-reference-expression/<source_cohort>.csv.gz`;
  largest 90 MB. New shared `pirlygenes.expression.stats.upsert_to_shard`
  helper; every per-cohort builder + the Treehouse sweep use it.
  Loader updated to glob + concat shards transparently.
- **TCGA cancer codes registry sync.** 31 TCGA codes' `source_cohort`
  in `cancer-type-registry.csv` updated from `TCGA_XENA_TOIL` →
  `TREEHOUSE_POLYA_25_01_TCGA_SUBSET` to match actually-bundled data.

Queued for follow-up sessions, in priority order:

2. **`pirlygenes build <source-id>` dispatcher.** Read the YAML
   registry, dispatch to the matching builder (existing scripts/
   modules, hoisted into `pirlygenes/builders/`). `pirlygenes build all`
   iterates every entry.
3. **TCGA builder.** Parameterized GDC STAR-counts builder taking a
   project_id from the registry. Run for all 32 TCGA projects in the
   registry; produce extended-stat rows in
   `cancer-reference-expression.csv.gz`.
4. **BL re-run** (smallest existing GDC cohort) as proof that the
   extended-stat schema works end-to-end on a real cohort with
   per-sample data.
5. **`pirlygenes downloads fetch <source-id>` / `prune`** —
   implement the writeable cache management.
6. **Treehouse compendium fetcher.** Download the public
   POLYA / RiboD compendium, subset by cohort label, run the same
   per-sample summarizer to backfill ATRT / EWS / HEPB / NUTM / OS /
   RMS_* / SARC_LMS / SARC_LPS_UNSPEC / SARC_MYXFIB / SARC_SYN /
   SARC_UPS / CHOR / RB with extended stats.
7. **`pirlygenes plot` subcommand.** Cohort-level plots — the
   FATE1-across-cohorts plot in `local_reports/fate1_ews_cohort_summary.png`
   is the prototype.
8. **Per-cohort re-runs.** Sweep every remaining `build_*` cohort
   (MMRF, TARGET ALL / RT / NBL / WT, CGCI BL [if not done in #4],
   CLLMAP, BeatAML, SCLC UCologne, GSE accessions) and rebuild with
   extended stats.
9. **Trufflepig cross-package update.** Trufflepig consumers of
   `cancer-reference-expression.csv.gz` may rely on the current
   column set — audit, then update.

## Schema

Current long-format columns in `cancer-reference-expression.csv.gz`:

    Ensembl_Gene_ID, Symbol, cancer_code, source_cohort, source_project,
    source_version, TPM_median, TPM_q1, TPM_q3, TPM_mean,
    TPM_clean_median, TPM_clean_q1, TPM_clean_q3,
    n_samples, n_detected, processing_pipeline, notes

Target extension (milestone 1):

    + TPM_std, TPM_min, TPM_max, TPM_p5, TPM_p10, TPM_p90, TPM_p95
    + TPM_clean_std, TPM_clean_min, TPM_clean_max,
      TPM_clean_p5, TPM_clean_p10, TPM_clean_p90, TPM_clean_p95

For rows imported via `import_cancer_specific_expression.py` from
summary-only sources, new columns stay NaN until milestone 6
(Treehouse) lands.

## TPM-conversion math caveat

`fpkm_i / sum(fpkms) * 1e6` is the **per-sample** TPM identity.
Applied to a vector of per-cohort medians, it yields a TPM-proxy that
is *not* the median of per-sample TPMs. The current `_add_summary_clean_tpm`
in `scripts/import_cancer_specific_expression.py` accepts that
approximation by design (the "raw per-sample values are not bundled"
notes column flags it). Every cohort that the rebuild touches at the
per-sample level escapes this approximation; cohorts that stay on
summary-import retain it and the notes column should continue to say
so.

## Data sources

See `pirlygenes/data/expression_sources.yaml` for the authoritative
list. Each entry carries: `id`, `cancer_codes`, `category`,
`source_type`, `builder` (module path once builders are hoisted into
`pirlygenes/builders/`), source-specific identifiers (project_id /
accession / url), `unit`, `expected_size_gb`, `citation`, and
`special_handling` notes for any per-source quirks
(e.g. MMRF's "deterministic primary BM CD138+ sample per case"
inclusion rule).

## Cache convention

Default cache root: `~/.cache/pirlygenes/expression/<source_id>/`.
Override via `PIRLYGENES_CACHE` environment variable. Disk usage is
discoverable via `pirlygenes downloads list` (groups sources by
`category`, sorts by on-disk size descending).
