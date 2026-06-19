# Data bundle deploy checklist (DATA_VERSION bump + PyPI)

Expression data lives outside the wheel. In a git checkout, the in-repo
`pirlygenes/data/` artifacts are authoritative, but a pip-installed user fetches
`pirlygenes-data-v<DATA_VERSION>.tar.gz` from the matching GitHub Release. Any
change to a wheel-excluded data artifact needs a new tarball and a `DATA_VERSION`
bump in the same release.

## Streamlined path (preferred)

`python scripts/release.py` orchestrates everything below in one idempotent,
re-runnable command — it builds + publishes the data tarball only when
`DATA_VERSION` changed, warms the version cache, runs `./test.sh`, builds, uploads
to PyPI, and creates/pushes the release tag, in the one order that can't 404 a
pip install. Dry-run by default:

```bash
python scripts/release.py              # print the plan, touch nothing
python scripts/release.py --execute    # run for real (prompts before each public step)
```

The manual steps below are kept as the reference the script automates.

Files in the tarball (`pirlygenes.data_bundle.DOWNLOADABLE_PATHS`):
`cancer-reference-expression/`, `cancer-reference-expression-views/`,
`cancer-reference-expression-representatives/`,
`cancer-reference-expression-percentiles/`, `pan-cancer-expression.csv`,
`hpa-cell-type-expression.csv`.

## Steps (run by a maintainer with PyPI + GitHub-release credentials)

1. **Regenerate derived artifacts** when the migration touches expression data:
   ```bash
   python scripts/bake_canonical_reference_expression_artifacts.py
   # The views artifact is a cache of the summary shards, so regenerate it
   # whenever those shards change (it must run after the bake above):
   python scripts/generate_cohort_expression_views.py
   # Percentiles + representatives only when per-sample source matrices change:
   python scripts/generate_cohort_gene_percentiles.py
   ```

2. **Build the data tarball** from the in-repo data:
   ```bash
   V=5.23.0
   tar -C pirlygenes/data -czf pirlygenes-data-v$V.tar.gz \
       cancer-reference-expression \
       cancer-reference-expression-views \
       cancer-reference-expression-representatives \
       cancer-reference-expression-percentiles \
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

7. **Publish to PyPI**: `./deploy.sh` (lint → test → build → twine upload → tag →
   push). Requires PyPI credentials.
