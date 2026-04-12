# Licensed under the Apache License, Version 2.0

"""Broad-compartment sample decomposition with external purity anchoring.

Decomposes bulk RNA-seq into tumor + TME components (immune, stromal,
site-specific host cells) using weighted NNLS on component-enriched
marker genes.  Tumor purity is estimated separately, then the non-tumor
fraction is distributed across reference-supported compartments.
"""

from .engine import (
    decompose_sample,
    DecompositionResult,
    get_decomposition_parameters,
    infer_sample_mode,
)
from .plot import plot_decomposition_summary

__all__ = [
    "decompose_sample",
    "DecompositionResult",
    "get_decomposition_parameters",
    "infer_sample_mode",
    "plot_decomposition_summary",
]
