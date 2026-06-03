#!/usr/bin/env python
"""Refresh the GitHub source-data mirror for small/medium build inputs.

Uploads the cached raw source files (GEO heme matrices + soft files, the
CLL-map TPM TSV) to a pinned GitHub release so ``pirlygenes build`` can fetch
them mirror-first instead of from flaky upstreams. See
:mod:`pirlygenes.builders.source_data_mirror`.

    python scripts/refresh_source_data_mirror.py --tag source-data-YYYYMMDD

After uploading, bump ``SOURCE_DATA_MIRROR_TAG`` in source_data_mirror.py.
Large GDC sources are intentionally NOT mirrored (too big for release assets).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

CACHE = Path.home() / ".cache" / "pirlygenes" / "expression"

# Files to mirror, by their (unique) basename → cached source path. The
# basename is what builders look up on the mirror (source_data_mirror.fetch).
MIRRORED_FILES = {
    "cllmap_rnaseq_tpms_full.tsv.gz": CACHE / "cllmap" / "cllmap_rnaseq_tpms_full.tsv.gz",
    "GSE100026_expressed_gene_RPKM.txt.gz": CACHE / "geo-heme" / "GSE100026" / "GSE100026_expressed_gene_RPKM.txt.gz",
    "GSE100026_family.soft.gz": CACHE / "geo-heme" / "GSE100026" / "GSE100026_family.soft.gz",
    "GSE271664_HTSeq_counts.csv.gz": CACHE / "geo-heme" / "GSE271664" / "GSE271664_HTSeq_counts.csv.gz",
    "GSE271664_family.soft.gz": CACHE / "geo-heme" / "GSE271664" / "GSE271664_family.soft.gz",
    "GSE283710_CD34_rawcounts.csv.gz": CACHE / "geo-heme" / "GSE283710" / "GSE283710_CD34_rawcounts.csv.gz",
    "GSE283710_family.soft.gz": CACHE / "geo-heme" / "GSE283710" / "GSE283710_family.soft.gz",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="release tag, e.g. source-data-20260603")
    parser.add_argument("--repo", default="pirl-unc/pirlygenes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    present, missing = [], []
    for name, path in MIRRORED_FILES.items():
        (present if path.exists() else missing).append(name)
    if missing:
        print(f"WARNING: not cached locally (skipped): {missing}")
    if not present:
        print("nothing to upload — fetch the sources into the cache first.")
        return 1
    print(f"uploading {len(present)} files to release {args.tag}: {present}")
    if args.dry_run:
        return 0

    # Create the release if it doesn't exist, then upload each asset.
    subprocess.run(
        ["gh", "release", "view", args.tag, "--repo", args.repo],
        capture_output=True,
    ).returncode == 0 or subprocess.run(
        ["gh", "release", "create", args.tag, "--repo", args.repo,
         "--title", f"Source-data mirror {args.tag}",
         "--notes", "Mirrored small/medium build-time source inputs "
                    "(GEO heme matrices, CLL-map TSV). See "
                    "pirlygenes.builders.source_data_mirror."],
        check=True,
    )
    for name in present:
        subprocess.run(
            ["gh", "release", "upload", args.tag, str(MIRRORED_FILES[name]),
             "--repo", args.repo, "--clobber"],
            check=True,
        )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
