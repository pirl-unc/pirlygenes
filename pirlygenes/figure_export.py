"""Consistent publication exports for package and analysis figures."""
from pathlib import Path

import matplotlib


def save_figure(fig, path, *, dpi=300, **kwargs):
    """Save a lossless PNG (at least 300 dpi) and a matching vector PDF.

    Text and paths remain vector objects in PDF; raster artists such as imshow
    retain the requested resolution. Never build the PDF from the PNG.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    options = dict(bbox_inches="tight", facecolor="white", transparent=False)
    options.update(kwargs)
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(path, dpi=max(300, dpi), **options)
        if path.suffix.lower() == ".png":
            fig.savefig(path.with_suffix(".pdf"), dpi=max(300, dpi), **options)
