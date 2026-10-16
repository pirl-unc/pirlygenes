"""Replay version collisions with local Git histories and mocked public indexes."""

import io
import json
import subprocess
from urllib.error import HTTPError

import pytest

from scripts import release_versions as versions


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=True,
    ).stdout.strip()


def commit(root, code="6.0.11", data="6.0.5"):
    path = root / "pirlygenes/version.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f'__version__ = "{code}"\nDATA_VERSION = "{data}"\n')
    git(root, "add", ".")
    git(root, "commit", "--allow-empty", "-m", f"Release {code} data {data}")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    root = tmp_path / "checkout"
    remote.mkdir()
    root.mkdir()
    git(remote, "init", "--bare", "--initial-branch=main")
    git(root, "init", "--initial-branch=main")
    git(root, "config", "user.name", "Release test")
    git(root, "config", "user.email", "release-test@example.invalid")
    git(root, "remote", "add", "origin", str(remote))
    commit(root)
    git(root, "push", "origin", "main")
    monkeypatch.setattr(versions, "published_versions", lambda: {"6.0.10"})
    return root


def test_pr_rejects_version_taken_while_ci_runs(repository):
    root = repository
    git(root, "switch", "-c", "data-upgrade")
    commit(root, "6.0.12", "6.0.12")
    git(root, "switch", "main")
    commit(root, "6.0.12", "6.0.5")
    with pytest.raises(versions.ReleaseVersionError, match="must be newer"):
        versions.check_pr_versions(base_ref="main", head_ref="data-upgrade", root=root)


def test_nonrelease_pr_keeps_its_original_version(repository):
    root = repository
    git(root, "switch", "-c", "bugfix")
    (root / "fix.txt").write_text("a code fix without a version bump")
    commit(root)
    git(root, "switch", "main")
    commit(root, "6.0.12", "6.0.12")
    assert not versions.check_pr_versions(base_ref="main", head_ref="bugfix", root=root)


@pytest.mark.parametrize("code,data", [("6.0.12", "6.0.5"), ("6.0.12", "6.0.12"), ("6.1.0", "6.0.5")])
def test_pr_accepts_new_code_or_data_release(repository, code, data):
    root = repository
    git(root, "switch", "-c", "release")
    commit(root, code, data)
    assert versions.check_pr_versions(base_ref="main", root=root)


def test_data_change_requires_new_package_version(repository):
    commit(repository, data="6.0.12")
    with pytest.raises(versions.ReleaseVersionError, match="package version"):
        versions.check_pr_versions(base_ref="HEAD~1", root=repository)


def test_pr_rejects_data_version_taken_on_main(repository):
    root = repository
    git(root, "switch", "-c", "release")
    commit(root, "6.0.13", "6.0.6")
    git(root, "switch", "main")
    commit(root, "6.0.12", "6.0.12")
    with pytest.raises(versions.ReleaseVersionError, match="data version"):
        versions.check_pr_versions(base_ref="main", head_ref="release", root=root)


@pytest.mark.parametrize("tag", ["v6.0.12", "v6.0.6"])
def test_pr_rejects_existing_code_or_data_tag(repository, tag):
    git(repository, "tag", tag)
    commit(repository, "6.0.12", "6.0.6")
    with pytest.raises(versions.ReleaseVersionError, match="already exists"):
        versions.check_pr_versions(base_ref="HEAD~1", root=repository)


def test_release_versions_are_numeric_and_not_executed(tmp_path):
    marker = tmp_path / "executed"
    source = f'__version__ = "6.0.10"\nDATA_VERSION = "6.0.9"\nopen({str(marker)!r}, "w").close()\n'
    assert versions.read_versions(source) == versions.ReleaseVersions("6.0.10", "6.0.9")
    assert not marker.exists()
    assert versions.release_tuple("6.0.10") > versions.release_tuple("6.0.9")
    with pytest.raises(versions.ReleaseVersionError, match="nonliteral"):
        versions.read_versions('__version__ = str(1)\nDATA_VERSION = "6.0.5"')


def test_publication_accepts_current_unpublished_main(repository):
    versions.check_publication(target_sha=git(repository, "rev-parse", "HEAD"), root=repository)


@pytest.mark.parametrize("published", [{"6.0.11"}, {"6.0.12"}])
def test_publication_rejects_published_or_superseded_version(repository, monkeypatch, published):
    monkeypatch.setattr(versions, "published_versions", lambda: published)
    with pytest.raises(versions.ReleaseVersionError, match="already published or superseded"):
        versions.check_publication(target_sha=git(repository, "rev-parse", "HEAD"), root=repository)


def advance_remote(root):
    old = git(root, "rev-parse", "HEAD")
    git(root, "switch", "-c", "other-release", "main")
    commit(root, "6.0.12", "6.0.12")
    git(root, "push", "origin", "HEAD:main")
    git(root, "switch", "--detach", old)
    return old


def test_publication_rechecks_main_after_long_tests(repository):
    old = advance_remote(repository)
    with pytest.raises(versions.ReleaseVersionError, match="not current origin/main"):
        versions.check_publication(target_sha=old, root=repository)


def test_allow_branch_does_not_allow_a_stale_release(repository):
    old = advance_remote(repository)
    with pytest.raises(versions.ReleaseVersionError, match="behind current origin/main"):
        versions.check_publication(target_sha=old, allow_branch=True, root=repository)


def test_allow_branch_accepts_a_new_release_based_on_current_main(repository):
    git(repository, "switch", "-c", "release")
    head = commit(repository, "6.0.12")
    versions.check_publication(target_sha=head, allow_branch=True, root=repository)


def test_publication_rejects_checkout_changed_during_tests(repository):
    old = git(repository, "rev-parse", "HEAD")
    commit(repository, "6.0.12")
    with pytest.raises(versions.ReleaseVersionError, match="checkout changed"):
        versions.check_publication(target_sha=old, root=repository)


def test_publication_rejects_tag_owned_by_another_commit(repository):
    git(repository, "tag", "v6.0.11")
    head = commit(repository)
    git(repository, "push", "origin", "main", "--tags")
    with pytest.raises(versions.ReleaseVersionError, match="belongs to another commit"):
        versions.check_publication(target_sha=head, root=repository)


@pytest.mark.parametrize("annotated", [False, True])
def test_publication_can_resume_its_own_data_release_tag(repository, annotated):
    if annotated:
        git(repository, "tag", "-a", "v6.0.11", "-m", "Data uploaded before wheel")
    else:
        git(repository, "tag", "v6.0.11")
    git(repository, "push", "origin", "--tags")
    versions.check_publication(target_sha=git(repository, "rev-parse", "HEAD"), root=repository)


@pytest.mark.parametrize("tag,accepted", [("v6.0.5", True), ("v6.0.12", False)])
def test_reused_data_tag_must_declare_that_data_version(repository, tag, accepted):
    git(repository, "tag", tag)
    head = commit(repository, "6.0.13", tag.removeprefix("v"))
    git(repository, "push", "origin", "main", "--tags")
    if accepted:
        versions.check_publication(target_sha=head, root=repository)
    else:
        with pytest.raises(versions.ReleaseVersionError, match="different data version"):
            versions.check_publication(target_sha=head, root=repository)


@pytest.mark.parametrize("status", [401, 404, 500])
def test_pypi_lookup_failures_are_not_treated_as_available(monkeypatch, status):
    def unavailable(*args, **kwargs):
        raise HTTPError("https://pypi.org", status, "unavailable", {}, None)
    monkeypatch.setattr(versions, "urlopen", unavailable)
    with pytest.raises(versions.ReleaseVersionError, match="cannot verify PyPI"):
        versions.published_versions()


@pytest.mark.parametrize("payload", [{}, {"releases": []}, {"releases": {}}])
def test_pypi_malformed_inventory_is_rejected(monkeypatch, payload):
    monkeypatch.setattr(versions, "urlopen", lambda *a, **k: io.StringIO(json.dumps(payload)))
    with pytest.raises(versions.ReleaseVersionError, match="cannot verify PyPI"):
        versions.published_versions()


def test_pypi_reads_the_full_release_history(monkeypatch):
    monkeypatch.setattr(versions, "urlopen", lambda *a, **k: io.StringIO('{"releases":{"6.0.11":[],"6.0.12":[]}}'))
    assert versions.published_versions() == {"6.0.11", "6.0.12"}


def test_ci_command_rejects_a_version_taken_after_the_pr_started(repository):
    import sys
    from pathlib import Path

    root = repository
    git(root, "switch", "-c", "release")
    commit(root, "6.0.12", "6.0.12")
    git(root, "switch", "main")
    commit(root, "6.0.12")
    script = root / "scripts/release_versions.py"
    script.parent.mkdir()
    script.write_text(Path(versions.__file__).read_text())
    result = subprocess.run(
        [sys.executable, str(script), "--base-ref", "main", "--head-ref", "release"],
        cwd=root, text=True, capture_output=True,
    )
    assert result.returncode == 1
    assert "must be newer than current main version 6.0.12" in result.stderr
