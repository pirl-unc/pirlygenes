# Licensed under the Apache License, Version 2.0

"""Expression decomposition into cell type components.

Uses constrained NNLS to decompose bulk RNA-seq into a mixture of
cell type reference profiles, with template-dependent constraints
(e.g. CTA genes zero in non-tumor components, IG genes zero in
non-B-cell components unless the tumor is B-cell origin).
"""

from .engine import decompose_sample, DecompositionResult

__all__ = ["decompose_sample", "DecompositionResult"]
