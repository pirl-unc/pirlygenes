"""Structured API boundary for the ``pirlygenes analyze`` pipeline.

The command-line implementation still lives in :mod:`pirlygenes.cli`, but
the data contracts in this package are intentionally free of plotting and
heavy reference imports. They define the handoff surface that can move to
the future ``trufflepig`` repository without dragging the whole gene-set
package with it.
"""

from .flow import (
    apply_sample_context_to_purity,
    build_analysis_parameters,
    build_analyze_paths,
    discover_output_artifacts,
    resolve_analyze_inputs,
    should_adopt_decomposition_purity,
    write_json,
)
from .comparison import (
    AnalyzeSummaryRecord,
    build_analyze_comparison_markdown,
    load_analyze_summary_record,
)
from .models import (
    AnalyzeArtifact,
    AnalyzeConfig,
    AnalyzePaths,
    AnalyzeRun,
    InputResolution,
    StepRecord,
)

__all__ = [
    "AnalyzeArtifact",
    "AnalyzeConfig",
    "AnalyzePaths",
    "AnalyzeRun",
    "AnalyzeSummaryRecord",
    "InputResolution",
    "StepRecord",
    "apply_sample_context_to_purity",
    "build_analysis_parameters",
    "build_analyze_comparison_markdown",
    "build_analyze_paths",
    "discover_output_artifacts",
    "load_analyze_summary_record",
    "resolve_analyze_inputs",
    "should_adopt_decomposition_purity",
    "write_json",
]
