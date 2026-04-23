# Licensed under the Apache License, Version 2.0

"""Shared reference-line helpers for sample-TPM plots.

Every plot that renders per-gene sample TPM values benefits from a
consistent visual anchor for "where does this gene's expression sit
relative to the rest of the sample?". A single faint dashed line at
the sample's 90th-percentile TPM gives that anchor cheaply — readers
can immediately see which genes are in the top decile of the whole
transcriptome.

The percentile is computed from the expressed portion of the sample
(TPM > 0) rather than the full gene universe, so near-zero gene
noise doesn't pull the reference down toward zero.
"""

from __future__ import annotations

import numpy as np


_P90_LINE_COLOR = "#888888"
_P90_LINE_DASHES = (1, 4)
_P90_LINE_WIDTH = 0.8
_P90_LINE_ALPHA = 0.55
_P90_LABEL_FONTSIZE = 7
_P90_LABEL_COLOR = "#555555"


def compute_sample_p90_tpm(sample_tpm_by_symbol, *, min_tpm=0.0, percentile=90.0):
    """Return the nth-percentile TPM over expressed genes (TPM > ``min_tpm``).

    Parameters
    ----------
    sample_tpm_by_symbol : dict, pandas Series, or iterable of floats
    min_tpm : float
        Excludes genes at or below this TPM (default 0.0 = keep only
        positively-expressed genes).
    percentile : float
        Percentile to compute (default 90).

    Returns
    -------
    float or None
        The percentile value, or ``None`` when the sample has too few
        expressed genes (< 50) to compute a meaningful percentile.
    """
    try:
        import pandas as pd
        if isinstance(sample_tpm_by_symbol, dict):
            values = np.array(
                list(sample_tpm_by_symbol.values()), dtype=float,
            )
        elif isinstance(sample_tpm_by_symbol, pd.Series):
            values = sample_tpm_by_symbol.astype(float).to_numpy()
        else:
            values = np.asarray(list(sample_tpm_by_symbol), dtype=float)
    except Exception:
        return None

    expressed = values[(values > min_tpm) & np.isfinite(values)]
    if expressed.size < 50:
        return None
    return float(np.percentile(expressed, percentile))


def add_p90_reference_line(
    ax,
    sample_tpm_by_symbol,
    *,
    orientation="vertical",
    label_fmt=None,
    min_tpm=0.0,
    percentile=90.0,
):
    """Overlay a faint dashed line at the sample's 90th-percentile TPM.

    ``orientation="vertical"`` draws a vertical line at x = p90 — use
    for horizontal-bar plots where TPM is the x-axis.
    ``orientation="horizontal"`` draws a horizontal line at y = p90 —
    use for scatter / vertical-bar plots where TPM is the y-axis.

    The line is cheap visual chrome: no-op when the percentile can't
    be computed (too few expressed genes). A tiny label is placed
    near the line so the reader knows what the threshold represents.

    Returns the p90 TPM (float) or ``None`` when skipped.
    """
    p90 = compute_sample_p90_tpm(
        sample_tpm_by_symbol, min_tpm=min_tpm, percentile=percentile,
    )
    if p90 is None or p90 <= 0:
        return None

    if label_fmt is None:
        label_fmt = f"sample p{int(percentile)} ({{:.0f}} TPM)"

    if orientation == "vertical":
        ax.axvline(
            p90,
            color=_P90_LINE_COLOR,
            linestyle=(0, _P90_LINE_DASHES),
            linewidth=_P90_LINE_WIDTH,
            alpha=_P90_LINE_ALPHA,
            zorder=0.5,
        )
        ax.text(
            p90, ax.get_ylim()[1],
            " " + label_fmt.format(p90),
            ha="left", va="top",
            fontsize=_P90_LABEL_FONTSIZE,
            color=_P90_LABEL_COLOR,
            alpha=_P90_LINE_ALPHA + 0.2,
        )
    elif orientation == "horizontal":
        ax.axhline(
            p90,
            color=_P90_LINE_COLOR,
            linestyle=(0, _P90_LINE_DASHES),
            linewidth=_P90_LINE_WIDTH,
            alpha=_P90_LINE_ALPHA,
            zorder=0.5,
        )
        ax.text(
            ax.get_xlim()[1], p90,
            label_fmt.format(p90) + " ",
            ha="right", va="bottom",
            fontsize=_P90_LABEL_FONTSIZE,
            color=_P90_LABEL_COLOR,
            alpha=_P90_LINE_ALPHA + 0.2,
        )
    else:
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal'; got {orientation!r}"
        )
    return p90
