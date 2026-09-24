"""The compiled PDF must preserve searchable vector text at original size."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pytest

from analyses.regenerate_plots import _build_combined_pdf
from pirlygenes.figure_export import save_figure

PdfReader = pytest.importorskip("pypdf").PdfReader
pytest.importorskip("reportlab")


def test_vector_text_survives_compilation_without_downsampling(tmp_path):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([0, 1], [0, 1])
    ax.set_title("Searchable vector figure")
    png = tmp_path / "example" / "plot.png"
    save_figure(fig, png, dpi=100)  # Even a legacy caller gets the 300 dpi floor.
    plt.close(fig)
    with Image.open(png) as image:
        assert image.info["dpi"][0] >= 299
        assert image.width > 1000
    original = PdfReader(png.with_suffix(".pdf")).pages[0]
    compiled = PdfReader(_build_combined_pdf(tmp_path))
    page = compiled.pages[-1]
    assert "Searchable vector figure" in page.extract_text()
    assert float(page.mediabox.width) == float(original.mediabox.width)
    assert "example/plot.png" in page.extract_text()
    assert len(compiled.outline) == 1


def test_legacy_png_is_embedded_at_native_pixel_dimensions(tmp_path):
    png = tmp_path / "legacy.png"
    Image.new("RGB", (1700, 900), "white").save(png, dpi=(300, 300))
    compiled = PdfReader(_build_combined_pdf(tmp_path))
    images = list(compiled.pages[-1].images)
    assert len(images) == 1
    assert images[0].image.size == (1700, 900)
