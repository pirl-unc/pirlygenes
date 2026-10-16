"""Check release versions without importing PirlyGenes or its data dependencies."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
VERSION_PATH = "pirlygenes/version.py"


class ReleaseVersionError(RuntimeError):
    """A release would reuse a version or publish from stale source."""


def release_tuple(value: str) -> tuple[int, int, int]:
    """Parse the project's three-part numeric release convention."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value):
        raise ReleaseVersionError(f"invalid release version: {value!r}")
    return tuple(int(part) for part in value.split("."))


@dataclass(frozen=True)
class ReleaseVersions:
    code: str
    data: str


def read_versions(source: str) -> ReleaseVersions:
    """Read literal version declarations, without executing source from a PR."""
    values = {}
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise ReleaseVersionError("cannot parse version declarations") from exc
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("__version__", "DATA_VERSION"):
                if target.id in values:
                    raise ReleaseVersionError(f"duplicate version declaration: {target.id}")
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise ReleaseVersionError(f"nonliteral version: {target.id}") from exc
                release_tuple(value)
                values[target.id] = value
    if set(values) != {"__version__", "DATA_VERSION"}:
        raise ReleaseVersionError("version module must declare __version__ and DATA_VERSION")
    return ReleaseVersions(values["__version__"], values["DATA_VERSION"])


def git_output(*args: str, root: Path = ROOT) -> str:
    """Run a checked Git query in the release checkout."""
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if result.returncode:
        raise ReleaseVersionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def versions_at(ref: str, *, root: Path = ROOT) -> ReleaseVersions:
    """Read the versions committed at a Git ref."""
    return read_versions(git_output("show", f"{ref}:{VERSION_PATH}", root=root))


def check_pr_versions(*, base_ref: str, head_ref: str = "HEAD", root: Path = ROOT) -> bool:
    """Reject changed versions that collide with the latest base or fetched tags.

    A PR that does not change either version may keep its original declarations:
    Git merges the newer base version normally. Data releases must also advance
    the package version, because a published wheel cannot acquire a new data pin.
    """
    fork = git_output("merge-base", base_ref, head_ref, root=root)
    original = versions_at(fork, root=root)
    candidate = versions_at(head_ref, root=root)
    if candidate == original:
        return False
    current = versions_at(base_ref, root=root)
    if release_tuple(candidate.code) <= release_tuple(current.code):
        raise ReleaseVersionError(
            f"PR package version {candidate.code} must be newer than "
            f"current {base_ref} version {current.code}; update the release versions"
        )
    changed = {candidate.code}
    if candidate.data != original.data:
        if release_tuple(candidate.data) <= release_tuple(current.data):
            raise ReleaseVersionError(
                f"PR data version {candidate.data} must be newer than "
                f"current {base_ref} data version {current.data}"
            )
        changed.add(candidate.data)
    tags = set(git_output("tag", "--list", root=root).splitlines())
    for version in sorted(changed):
        if f"v{version}" in tags:
            raise ReleaseVersionError(f"release tag v{version} already exists; choose a new version")
    return True


def remote_tag_commit(tag: str, *, root: Path = ROOT) -> str | None:
    """Resolve a remote lightweight or annotated tag to its source commit."""
    ref = f"refs/tags/{tag}"
    output = git_output("ls-remote", "origin", ref, f"{ref}^{{}}", root=root)
    refs = dict(line.split()[::-1] for line in output.splitlines())
    return refs.get(f"{ref}^{{}}", refs.get(ref))


def published_versions() -> set[str]:
    """Read PyPI's project index; connection and malformed-response errors abort."""
    request = Request(
        "https://pypi.org/pypi/pirlygenes/json",
        headers={"User-Agent": "pirlygenes-release/1", "Cache-Control": "no-cache"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        releases = payload["releases"]
        if not isinstance(releases, dict) or not releases:
            raise ValueError("missing release history")
    except (URLError, OSError, ValueError, KeyError, TypeError) as exc:
        raise ReleaseVersionError(f"cannot verify PyPI release history: {exc}") from exc
    return set(releases)


def check_publication(*, target_sha: str, allow_branch: bool = False, root: Path = ROOT) -> None:
    """Check current main, tag ownership and PyPI before any public writes."""
    git_output("fetch", "--no-tags", "origin", "main", root=root)
    current_sha = git_output("rev-parse", "FETCH_HEAD", root=root)
    actual_sha = git_output("rev-parse", "HEAD", root=root)
    if actual_sha != target_sha:
        raise ReleaseVersionError("checkout changed while preparing the release")
    if target_sha != current_sha:
        if not allow_branch:
            raise ReleaseVersionError("checkout is not current origin/main; fetch and update before releasing")
        if git_output("merge-base", current_sha, target_sha, root=root) != current_sha:
            raise ReleaseVersionError("release branch is behind current origin/main; update before releasing")
    candidate = versions_at(target_sha, root=root)
    if target_sha != current_sha:
        current = versions_at(current_sha, root=root)
        if release_tuple(candidate.code) <= release_tuple(current.code):
            raise ReleaseVersionError("release branch must advance the current main package version")
    published = published_versions()
    stable = [release_tuple(v) for v in published if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", v)]
    if candidate.code in published or (stable and release_tuple(candidate.code) <= max(stable)):
        raise ReleaseVersionError(f"package version {candidate.code} is already published or superseded on PyPI")
    code_tag = f"v{candidate.code}"
    owner = remote_tag_commit(code_tag, root=root)
    if owner is not None and owner != target_sha:
        raise ReleaseVersionError(f"release tag {code_tag} belongs to another commit; choose a new version")
    data_tag = f"v{candidate.data}"
    if data_tag == code_tag:
        return
    if remote_tag_commit(data_tag, root=root) is not None:
        git_output("fetch", "--no-tags", "origin", f"refs/tags/{data_tag}", root=root)
        if versions_at("FETCH_HEAD", root=root).data != candidate.data:
            raise ReleaseVersionError(f"tag {data_tag} was allocated to a different data version")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--head-ref", default="HEAD")
    args = parser.parse_args()
    try:
        changed = check_pr_versions(base_ref=args.base_ref, head_ref=args.head_ref)
    except ReleaseVersionError as exc:
        parser.exit(1, f"release version check failed: {exc}\n")
    print("Release versions are available" if changed else "No release version changes")


if __name__ == "__main__":
    main()
