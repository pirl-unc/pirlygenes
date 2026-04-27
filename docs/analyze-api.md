# Analyze API Boundary

`pirlygenes analyze` is being prepared for extraction into a standalone
runner package, tentatively `trufflepig`. The current command still lives
in `pirlygenes.cli`, but new orchestration contracts live under
`pirlygenes.analyze` and are deliberately lightweight.

## Ownership Split

`pirlygenes` should remain the data/reference package:

- bundled CSVs and their schema tests
- cancer-type registry, aliases, panels, and expression references
- tumor/TME decomposition primitives
- plotting and report renderers while they still depend on local data

`trufflepig` should become the analysis runner:

- CLI argument parsing and run configuration
- input resolution and sample identity
- step order and information flow
- artifact manifest and output layout
- optional-step failure policy
- report packet assembly

The short-term rule is: if a decision is about what a cancer type means,
it belongs with `pirlygenes`; if it is about how a sample analysis moves
from one step to the next, it belongs under `pirlygenes.analyze` first.

## Current Contracts

`AnalyzeConfig`
: Immutable public options for one run. This replaces long positional
  plumbing between the CLI and the analysis body.

`InputResolution`
: The single resolved answer to "are we analyzing gene-level input,
  transcript-level input, or explicit paired inputs?" Downstream code
  consumes this instead of rechecking `genes`, `transcripts`, and
  `aggregate_gene_expression`.

`AnalyzePaths`
: Output directory, sample display ID, and filename prefix. Any future
  artifact-producing step should use this rather than reconstructing
  `prefix` by hand.

`AnalyzeRun`
: Mutable run state: config, resolved inputs, paths, step records, and
  artifact records. It is intentionally small and JSON-serializable.

`apply_sample_context_to_purity`
: The Step-0 context -> purity confidence flow. Sample preservation and
  degradation widen the purity interval before later steps consume it.

`should_adopt_decomposition_purity`
: The decomposition -> purity adoption guard. A decomposition result can
  replace classifier purity only when it agrees with the classifier,
  includes non-tumor components, and carries a populated purity result.

`*-manifest.json`
: Machine-readable run manifest discovered from emitted files plus the
  structured config/input/path/step trace. This is the first stable
  artifact for external automation.

## Target Analyze Flow

The runner should converge on these explicit step APIs:

1. `load_input(config, resolution) -> ExpressionTable`
2. `infer_context(expr) -> SampleContext`
3. `assess_healthy_vs_tumor(expr) -> Step0Result`
4. `call_cancer_type(expr, context, config) -> CancerCall`
5. `estimate_quality(expr, call, context) -> QualityResult`
6. `score_therapy_state(expr, call) -> TherapyState`
7. `decompose(expr, call, context, config) -> DecompositionResult`
8. `estimate_tumor_values(expr, call, purity, decomp) -> TumorValueTable`
9. `render_reports(state) -> ReportArtifacts`
10. `render_figures(state) -> FigureArtifacts`
11. `write_manifest(state) -> ManifestArtifact`

Each step should read named upstream objects and write a named result. It
should not mutate unrelated blocks in the shared `analysis` dictionary.
Until the legacy dictionary is retired, mutations should be isolated in
small helpers with tests.

## Cross-Cancer Consistency Rules

- Cancer-type aliases must resolve through the registry or the existing
  plot alias layer, never through report-specific hardcoding.
- Therapy panels should be selected through
  `cancer_key_genes_lookup_for_analysis` so parent/subtype behavior stays
  consistent.
- Report language should consume the same purity and cancer call objects
  used by plots and tables.
- Any cancer-specific exception needs either a data row in the registry or
  a small tested rule with a source comment.
- Missing expression references should degrade confidence or skip a view;
  they should not silently promote an unrelated cancer type.

## Extraction Checklist

- Keep `pirlygenes.analyze` importable without Matplotlib or large data
  loads.
- Move CLI-only code from `pirlygenes.cli` into small step functions that
  accept `AnalyzeRun` and return named result objects.
- Replace direct `print()` progress calls with a runner logger that can
  write console output and manifest events.
- Replace broad plot/report `except Exception` blocks with an optional
  step wrapper that records structured failures.
- Promote the manifest schema to a documented contract before moving the
  command to `trufflepig`.
