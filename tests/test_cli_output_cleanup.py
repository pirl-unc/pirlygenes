from pathlib import Path

from pirlygenes.cli import _clean_prefix_outputs


def _touch(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_clean_prefix_removes_prefixed_files_only(tmp_path):
    # Prior-run files matching prefix base
    _touch(tmp_path / "sample-summary.md", "old")
    _touch(tmp_path / "sample-decomposition.png", "old")
    _touch(tmp_path / "figures" / "sample-targets.png", "old")
    # Files that must NOT be deleted
    _touch(tmp_path / "README.md", "keep")
    _touch(tmp_path / "other-user-file.txt", "keep")
    _touch(tmp_path / "figures" / "not-ours.png", "keep")

    prefix = str(tmp_path / "sample")
    removed = _clean_prefix_outputs(tmp_path, prefix)

    assert removed == 3
    assert not (tmp_path / "sample-summary.md").exists()
    assert not (tmp_path / "sample-decomposition.png").exists()
    assert not (tmp_path / "figures" / "sample-targets.png").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "other-user-file.txt").exists()
    assert (tmp_path / "figures" / "not-ours.png").exists()


def test_clean_prefix_removes_scatter_subdir(tmp_path):
    scatter = tmp_path / "sample-vs-cancer"
    _touch(scatter / "PRAD.png", "old")
    _touch(scatter / "LUAD.png", "old")
    _touch(tmp_path / "sample-summary.md", "old")

    removed = _clean_prefix_outputs(tmp_path, str(tmp_path / "sample"))

    assert removed == 3
    assert not scatter.exists()
    assert not (tmp_path / "sample-summary.md").exists()


def test_clean_prefix_custom_image_prefix(tmp_path):
    # When user supplies output_image_prefix="bg002", only bg002-* should be cleaned
    _touch(tmp_path / "bg002-summary.md", "old")
    _touch(tmp_path / "sample-summary.md", "from-different-sample")

    removed = _clean_prefix_outputs(tmp_path, str(tmp_path / "bg002"))
    assert removed == 1
    assert not (tmp_path / "bg002-summary.md").exists()
    assert (tmp_path / "sample-summary.md").exists()


def test_clean_prefix_empty_or_missing_dir(tmp_path):
    # Calling on an empty / nonexistent dir should be a no-op
    assert _clean_prefix_outputs(tmp_path, str(tmp_path / "sample")) == 0
    assert _clean_prefix_outputs(tmp_path / "doesnotexist", str(tmp_path / "doesnotexist" / "sample")) == 0
