# Data bundle deploy checklist (DATA_VERSION bump + PyPI)

Expression data lives outside the wheel. In a git checkout, the in-repo
`pirlygenes/data/` artifacts are authoritative, but a pip-installed user fetches
`pirlygenes-data-v<DATA_VERSION>.tar.gz` from the matching GitHub Release. Any
change to a wheel-excluded data artifact needs a new tarball and a `DATA_VERSION`
bump in the same release.

## Streamlined path (preferred)

`python scripts/release.py` orchestrates everything below in one guarded
command — it builds + publishes the data tarball only when
`DATA_VERSION` changed, warms the version cache, runs `./test.sh`, builds, uploads
to PyPI, and creates the GitHub release/tag, in the one order that can't 404 a
pip install. Dry-run by default:

```bash
python scripts/release.py              # print the plan, touch nothing
python scripts/release.py --execute    # run for real (prompts before each public step)
```

`./deploy.sh` forwards to the same script with `--execute`; it accepts the
same options, including `--yes` and `--skip-tests`.

Before uploading data, and again after testing/building, the script fetches
current main and checks the package version against PyPI and the remote release
tags. A stale checkout, an already published/superseded package version, a tag
owned by another commit, or a failed GitHub/PyPI lookup stops publication. With
`--allow-branch`, the branch must descend from current main and declare a newer
package version. Data uploads never overwrite an existing asset.

CI checks changed package/data versions against current main and fetched tags
both before and after the long test suite. A data-version change also requires
a newer package version. Keep release PRs up to date before merging; requiring
up-to-date branches in branch protection closes the interval between the last
CI check and merge. These checks do not reserve versions on other publishers'
behalf.

A partial data upload can be reused when its tag belongs to the same commit.
Once the package has reached PyPI, rerunning publication is rejected. If only
GitHub release finalization was interrupted, verify the published artifacts and
finish that release/tag at the original release commit instead of uploading the
package again.

The manual steps below document the artifact layout; use the guarded script for
publication.

## Upgrading oncoref

Oncoref upgrades are deliberately maintainer-run because a new owner release
can change reference availability, sample QC, taxonomy, gene identity, or other
biological semantics. Pirlygenes also ships compatibility artifacts whose
metadata records the exact oncoref package and data versions used to build the
matrices.
A dependency-only bot PR would therefore be incomplete.

Use the deterministic helper from a clean, purpose-built branch:

```bash
python scripts/upgrade_oncoref.py check
python scripts/upgrade_oncoref.py prepare --version X.Y.Z
uv pip install -e '.[test]'
python scripts/upgrade_oncoref.py regenerate
```

`prepare` updates the exact owner pin and allocates the next pirlygenes
code/data patch version. After the new pin is installed, `regenerate` rebuilds
the cohort matrices, pan-cancer rollups, parity reports, and public-manifest
checksum. Before writing any artifact it verifies that every explicit rollup
shard still matches oncoref's selected summary cohort and fails closed on
source drift. Inspect the complete diff, run the focused upgrade tests and
normal serial test gate, then open a PR through the usual maintainer workflow.
The helper never creates branches or PRs and never merges or publishes a
release.

Files in the tarball (`pirlygenes.data_bundle.DOWNLOADABLE_PATHS`):
`cohort-expression-matrices/`, `pan-cancer-expression.csv`, and
`hpa-cell-type-expression.csv`. The source summary rows are served by oncoref;
pirlygenes no longer downloads its duplicate `cancer-reference-expression/`
runtime copy (#557).
(Per-cohort medoid representatives + per-gene percentile vectors moved to
oncoref in #208 — no longer in the tarball.)

## Steps (run by a maintainer with PyPI + GitHub-release credentials)

1. **Regenerate derived artifacts** when the migration touches expression data:
   ```bash
   python scripts/bake_canonical_reference_expression_artifacts.py
   # The matrices artifact is a cache of the summary shards, so regenerate it
   # whenever those shards change (it must run after the bake above):
   python scripts/generate_cohort_expression_matrices.py
   # (Per-gene percentile vectors + representative-sample medoids are owned by
   #  oncoref since #208 — regenerated in oncoref, not here.)
   ```

2. **Build the data tarball** from the in-repo data:
   ```bash
   V=6.0.0
   tar -C pirlygenes/data -czf pirlygenes-data-v$V.tar.gz \
       cohort-expression-matrices \
       pan-cancer-expression.csv \
       hpa-cell-type-expression.csv
   ```

3. **Pre-extract into the version cache** so the *serial* `test.sh` gate doesn't
   404-hang on `data_bundle.fetch()` (the DATA_VERSION release gotcha):
   ```bash
   mkdir -p ~/.cache/pirlygenes/bundled_data/v$V
   tar -C ~/.cache/pirlygenes/bundled_data/v$V -xzf pirlygenes-data-v$V.tar.gz
   ```

4. **Bump versions** in `pirlygenes/version.py`:
   - `__version__ = "$V"`
   - `DATA_VERSION = "$V"`

5. **Validate**: `COVERAGE=1 ./test.sh` (full suite, instrumented). The
   expression tests must pass against the pre-extracted cache for `$V`.

6. **Publish the data tarball** as the `v$V` GitHub release asset (the URL
   `data_bundle.RELEASE_URL` points at):
   ```bash
   gh release create v$V pirlygenes-data-v$V.tar.gz \
       --title "v$V" --notes "Data bundle refresh"
   ```
   (or attach to an existing release).

7. **Publish to PyPI**: `./deploy.sh` (version checks → data → lint/test → build →
   version recheck → PyPI → GitHub release). Requires PyPI credentials.
