# Phase C deploy checklist (DATA_VERSION bump + PyPI)

The Phase C sarcoma rename changed `cancer_code` values inside the **wheel-excluded**
bundled expression shards (`cancer-reference-expression/*.csv.gz`). In a git
checkout the in-repo (renamed) shards are authoritative, so tests/dev/analyses
are already consistent. **A pip-installed user, however, fetches the
`pirlygenes-data-v<DATA_VERSION>.tar.gz` release asset** — which still carries the
old codes — so the rename is not live for installs until a new data tarball is
published and `DATA_VERSION` is bumped. These must happen together.

Files in the tarball (`pirlygenes.data_bundle.DOWNLOADABLE_PATHS`):
`cancer-reference-expression/`, `pan-cancer-expression.csv`,
`hpa-cell-type-expression.csv`.

## Steps (run by a maintainer with PyPI + GitHub-release credentials)

1. **Build the data tarball** from the renamed in-repo data:
   ```bash
   V=5.13.0
   tar -C pirlygenes/data -czf pirlygenes-data-v$V.tar.gz \
       cancer-reference-expression pan-cancer-expression.csv hpa-cell-type-expression.csv
   ```

2. **Pre-extract into the version cache** so the *serial* `test.sh` gate doesn't
   404-hang on `data_bundle.fetch()` (the DATA_VERSION release gotcha):
   ```bash
   mkdir -p ~/.cache/pirlygenes/bundled_data/v$V
   tar -C ~/.cache/pirlygenes/bundled_data/v$V -xzf pirlygenes-data-v$V.tar.gz
   ```

3. **Bump versions** in `pirlygenes/version.py`:
   - `__version__ = "5.13.0"`
   - `DATA_VERSION = "5.13.0"`

4. **Validate**: `COVERAGE=1 ./test.sh` (full suite, instrumented). The
   registry/expression/sarcoma/burden/tmb/aggregate tests must pass against the
   pre-extracted v5.13.0 cache.

5. **Publish the data tarball** as the `v5.13.0` GitHub release asset (the URL
   `data_bundle.RELEASE_URL` points at):
   ```bash
   gh release create v5.13.0 pirlygenes-data-v5.13.0.tar.gz \
       --title "v5.13.0" --notes "Phase C sarcoma rename + computed aggregates"
   ```
   (or attach to an existing release).

6. **Publish to PyPI**: `./deploy.sh` (lint → test → build → twine upload → tag →
   push). Requires PyPI credentials.

## trufflepig follow-up (separate repo/PR)

trufflepig references the old codes (`OS`, `EWS`, `CHOR`, `HNSC_HPV_pos`, …) and
the `{"OS": "SARC"}`-style parent maps. Audit + remap to the `SARC_` namespace
(`SARC_OS`, `SARC_EWS`, `HNSC_HPVpos`, …) after this release ships, since
pirlygenes resolvers will no longer recognise the bare old codes.
