"""Streamlined, idempotent release of pirlygenes — data bundle + PyPI + tag.

Replaces the manual `docs/phase-c-deploy-checklist.md` dance (and the bare
`deploy.sh`) with one ordered, re-runnable command that does the right thing
whether or not the data bundle changed.

The single hard constraint this encodes: a pip-installed user fetches
`pirlygenes-data-v<DATA_VERSION>.tar.gz` from the GitHub release tagged
`v<DATA_VERSION>` (see `pirlygenes.data_bundle.RELEASE_URL`). So when
`DATA_VERSION` changes, that tarball MUST be built and attached to its release
*before* the wheel goes to PyPI — otherwise installs 404 on the bundle. This
script orders the stages so that can't be gotten wrong, and skips the heavy data
step entirely on a code-only release (when the tarball already exists).

Stages (each idempotent — safe to re-run after a mid-way failure):
  1. preflight  — clean tree, expected branch, read __version__ / DATA_VERSION
  2. data       — if the v<DATA_VERSION> tarball asset is missing: build it from
                  pirlygenes/data, pre-extract into the version cache (so the
                  serial test.sh gate doesn't 404-hang), create or update the
                  v<DATA_VERSION> release, and upload the asset
  3. validate   — ./test.sh (lint + full suite) against the warm cache
  4. build      — python -m build (sdist + wheel)
  5. publish    — twine upload dist/*  (the one irreversible step)
  6. release    — ensure the v<__version__> GitHub release/tag exists, push tags

Usage:
    python scripts/release.py                 # DRY RUN: print the plan, touch nothing
    python scripts/release.py --execute       # run for real, prompting before
                                              # each irreversible/public action
    python scripts/release.py --execute --yes # run without per-step prompts
    python scripts/release.py --execute --skip-tests   # CI already green

Nothing happens without --execute. Even with it, the public/irreversible
actions (GitHub release create/upload, twine upload, tag push) prompt unless
--yes is given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pirlygenes import data_bundle  # noqa: E402
from pirlygenes.version import DATA_VERSION, __version__  # noqa: E402

DATA_DIR = REPO_ROOT / "pirlygenes" / "data"
COHORT_VIEWS_MANIFEST = (
    DATA_DIR / "cancer-reference-expression-views" / "_manifest.json"
)
TARBALL = REPO_ROOT / data_bundle.TARBALL_FILENAME           # pirlygenes-data-v<DV>.tar.gz
DATA_TAG = f"v{DATA_VERSION}"
CODE_TAG = f"v{__version__}"
CACHE_DIR = Path.home() / ".cache" / "pirlygenes" / "bundled_data" / DATA_TAG
EXPECTED_BRANCH = "main"


class Abort(SystemExit):
    pass


def _run(cmd: list[str], *, dry: bool, capture: bool = False, check: bool = True):
    """Run a command, or just print it in dry-run mode."""
    printable = " ".join(cmd)
    if dry:
        print(f"  [dry-run] {printable}")
        return ""
    print(f"  $ {printable}", flush=True)
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, text=True,
        capture_output=capture, check=check,
    )
    return (result.stdout or "") if capture else ""


def _confirm(prompt: str, *, dry: bool, assume_yes: bool) -> bool:
    if dry:
        print(f"  [dry-run] would prompt: {prompt}")
        return True
    if assume_yes:
        return True
    reply = input(f"  >>> {prompt} [y/N] ").strip().lower()
    return reply in ("y", "yes")


# ---------------------------------------------------------------------------
# stage 1: preflight
# ---------------------------------------------------------------------------

def preflight(*, dry: bool, allow_branch: bool) -> None:
    print(f"== preflight ==  code={CODE_TAG}  data={DATA_TAG}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        text=True, capture_output=True, check=True).stdout.strip()
    if dirty:
        raise Abort("working tree is dirty — commit or stash before releasing:\n"
                    + dirty)
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT,
        text=True, capture_output=True, check=True).stdout.strip()
    if branch != EXPECTED_BRANCH and not allow_branch:
        raise Abort(
            f"on branch {branch!r}, expected {EXPECTED_BRANCH!r}. Merge first, "
            f"or pass --allow-branch to release from here.")
    print(f"  clean tree on {branch}")


# ---------------------------------------------------------------------------
# stage 2: data bundle (conditional)
# ---------------------------------------------------------------------------

def _release_has_asset(tag: str, asset: str) -> bool:
    proc = subprocess.run(
        ["gh", "release", "view", tag, "--json", "assets"],
        cwd=REPO_ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        return False  # release doesn't exist yet
    assets = json.loads(proc.stdout or "{}").get("assets", [])
    return any(a.get("name") == asset for a in assets)


def _release_exists(tag: str) -> bool:
    return subprocess.run(
        ["gh", "release", "view", tag], cwd=REPO_ROOT,
        capture_output=True).returncode == 0


def validate_data_bundle_manifests() -> None:
    """Refuse to publish derived views stamped for another data release."""
    try:
        manifest = json.loads(COHORT_VIEWS_MANIFEST.read_text())
    except FileNotFoundError as exc:
        raise Abort(
            f"cohort-views manifest missing: {COHORT_VIEWS_MANIFEST}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise Abort(
            f"cohort-views manifest unreadable: {COHORT_VIEWS_MANIFEST}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise Abort(
            f"cohort-views manifest invalid: {COHORT_VIEWS_MANIFEST} must "
            "contain a JSON object"
        )
    if manifest.get("canonical_gene_ids") is not True:
        raise Abort(
            "cohort-views manifest must declare canonical_gene_ids=true: "
            f"{COHORT_VIEWS_MANIFEST}"
        )
    actual = str(manifest.get("data_version", ""))
    if actual != DATA_VERSION:
        raise Abort(
            "cohort-views manifest data_version mismatch: "
            f"{actual!r} != DATA_VERSION {DATA_VERSION!r}; bump DATA_VERSION "
            "before running scripts/generate_cohort_expression_views.py"
        )


def build_data_tarball(*, dry: bool) -> None:
    members = list(data_bundle.DOWNLOADABLE_PATHS)
    missing = [m for m in members if not (DATA_DIR / m).exists()]
    if missing:
        raise Abort(f"data artifacts missing from {DATA_DIR}: {missing}")
    print(f"  building {TARBALL.name} from {len(members)} artifacts")
    if dry:
        print(f"  [dry-run] tar -C {DATA_DIR} -czf {TARBALL.name} {' '.join(members)}")
        return
    with tarfile.open(TARBALL, "w:gz") as tar:
        for member in members:
            tar.add(DATA_DIR / member, arcname=member)
    size_mb = TARBALL.stat().st_size / 1e6
    print(f"  wrote {TARBALL.name} ({size_mb:.0f} MB)")


def pre_extract_cache(*, dry: bool) -> None:
    """Warm the version cache so the serial test.sh gate never 404-fetches the
    not-yet-published bundle (the DATA_VERSION release gotcha)."""
    print(f"  pre-extracting into {CACHE_DIR}")
    if dry:
        print(f"  [dry-run] mkdir -p {CACHE_DIR} && tar -xzf {TARBALL.name} -C {CACHE_DIR}")
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TARBALL, "r:gz") as tar:
        tar.extractall(CACHE_DIR)
    print("  cache warm")


def publish_data(*, dry: bool, assume_yes: bool, target_sha: str) -> None:
    print(f"== data bundle ==  tag={DATA_TAG}")
    validate_data_bundle_manifests()
    if _release_has_asset(DATA_TAG, TARBALL.name):
        print(f"  {TARBALL.name} already on release {DATA_TAG}; skipping data publish")
        return

    build_data_tarball(dry=dry)
    pre_extract_cache(dry=dry)

    if not _confirm(f"publish {TARBALL.name} to GitHub release {DATA_TAG}?",
                    dry=dry, assume_yes=assume_yes):
        raise Abort("data publish declined")

    if _release_exists(DATA_TAG):
        _run(["gh", "release", "upload", DATA_TAG, str(TARBALL), "--clobber"],
             dry=dry)
    else:
        _run(["gh", "release", "create", DATA_TAG, str(TARBALL),
              "--target", target_sha, "--title", DATA_TAG,
              "--notes", f"Data bundle {DATA_TAG}"], dry=dry)
    print(f"  data bundle live at {data_bundle.RELEASE_URL}")


# ---------------------------------------------------------------------------
# stages 3-6: validate, build, publish, tag
# ---------------------------------------------------------------------------

def validate(*, dry: bool, skip_tests: bool) -> None:
    print("== validate ==")
    if skip_tests:
        print("  --skip-tests: skipping ./test.sh")
        return
    _run(["./test.sh"], dry=dry)


def build_wheel(*, dry: bool) -> None:
    print("== build ==")
    if not dry:
        shutil.rmtree(REPO_ROOT / "dist", ignore_errors=True)
    pip_available = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).returncode == 0
    if not pip_available:
        _run([sys.executable, "-m", "ensurepip", "--upgrade"], dry=dry)
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "build", "twine"],
         dry=dry)
    _run([sys.executable, "-m", "build"], dry=dry)


def publish_pypi(*, dry: bool, assume_yes: bool) -> None:
    print("== publish to PyPI (IRREVERSIBLE) ==")
    if not _confirm(f"twine upload pirlygenes {CODE_TAG} to PyPI? "
                    "This cannot be undone.", dry=dry, assume_yes=assume_yes):
        raise Abort("PyPI publish declined")
    _run([sys.executable, "-m", "twine", "upload", "dist/*"], dry=dry)


def finalize_release(*, dry: bool, assume_yes: bool, target_sha: str) -> None:
    print(f"== release/tag ==  {CODE_TAG}")
    if _release_exists(CODE_TAG):
        print(f"  release {CODE_TAG} already exists (data release reused)")
    else:
        if not _confirm(f"create GitHub release/tag {CODE_TAG}?",
                        dry=dry, assume_yes=assume_yes):
            raise Abort("release creation declined")
        _run(["gh", "release", "create", CODE_TAG, "--target", target_sha,
              "--title", CODE_TAG, "--generate-notes"], dry=dry)
    _run(["git", "push", "--tags"], dry=dry)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="actually run (default is a dry run that touches nothing)")
    ap.add_argument("--yes", action="store_true",
                    help="skip the per-step confirmation prompts")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip ./test.sh (use only when CI is already green)")
    ap.add_argument("--allow-branch", action="store_true",
                    help=f"release from a branch other than {EXPECTED_BRANCH}")
    args = ap.parse_args()
    dry = not args.execute

    if dry:
        print("DRY RUN — nothing will be built, published, or pushed.\n"
              "Re-run with --execute to perform the release.\n")

    target_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        text=True, capture_output=True, check=True).stdout.strip()

    try:
        preflight(dry=dry, allow_branch=args.allow_branch)
        publish_data(dry=dry, assume_yes=args.yes, target_sha=target_sha)
        validate(dry=dry, skip_tests=args.skip_tests)
        build_wheel(dry=dry)
        publish_pypi(dry=dry, assume_yes=args.yes)
        finalize_release(dry=dry, assume_yes=args.yes, target_sha=target_sha)
    except Abort as exc:
        print(f"\nrelease aborted: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"\ndone: pirlygenes {CODE_TAG} (data {DATA_TAG})"
          if not dry else "\ndry run complete.")


if __name__ == "__main__":
    main()
