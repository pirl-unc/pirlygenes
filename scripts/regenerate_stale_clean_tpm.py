"""Bring every cohort shard onto the shipped 16/9/75 clean-TPM contract.

Background. ``clean_tpm_matrix(censored_fill="fixed_fraction")`` splits the
censored block into three pinned compartments of the 1e6 budget — ribosomal
protein 16%, other technical RNA 9%, biology 75% (the shipped contract). The
*biology* compartment is identical to the older lumped-25% v4 (both pin biology
to 75%), so only the censored gene rows differ between the two.

Most shards were already regenerated on 16/9 (their values are correct; only the
self-describing text is stale). A handful of special-purpose shards (the MSI /
PAM50 / HPV / mutation splits, the glioma + SARC-LPS treehouse subsets, NUTM and
the SCLC TF-dominance split) were last built under lumped-25% v4 and carry
genuinely stale ``TPM_clean_*`` values.

This script recomputes ``TPM_clean_*`` for the stale shards from their cached
per-sample parquets and relabels the stale ``processing_pipeline`` / ``notes``
text on every shard so the data describes the normalization it actually uses.

Safety. A candidate parquet is only accepted when its recomputed *raw* stats
reproduce the shard's stored raw stats (same samples) — so a wrong-sample
parquet can never silently rewrite a shard. The recompute is asserted to leave
the biological rows' clean values unchanged (only censored rows move).
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pirlygenes.cohorts import ID_COLS
from pirlygenes.expression.stats import build_reference_rows
from pirlygenes.expression.normalize import clean_tpm_removal_mask

SHARD_DIR = Path("pirlygenes/data/cancer-reference-expression")
CACHE = Path.home() / ".cache" / "pirlygenes" / "expression"

# Shards last built under lumped-25% v4 (before the 16/9 split landed) — their
# censored-gene TPM_clean values are stale. (MBL is split by a subgroup
# classifier with no per-subgroup parquet; it is regenerated via its builder.)
STALE_SHARDS = [
    "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__GBM",
    "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__LGG",
    "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY__SARC_WDLPS",
    "TREEHOUSE_POLYA_25_01_TCGA_SARC_HISTOLOGY__SARC_DDLPS",
    "TREEHOUSE_POLYA_25_01_TCGA_SAMPLES__SARC_PLEOLPS",
    "TREEHOUSE_POLYA_25_01_TCGA_BRCA_PAM50",
    "TREEHOUSE_POLYA_25_01_TCGA_HNSC_HPV",
    "TREEHOUSE_POLYA_25_01_TCGA_LUAD_MUT",
    "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI__COAD_MSI",
    "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI__COAD_MSS",
    "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI__READ_MSI",
    "TREEHOUSE_POLYA_25_01_TCGA_COADREAD_MSI__READ_MSS",
    "UNC_NUTM1",
    "SCLC_UCOLOGNE_2015_TF_DOMINANCE",
]

# --- stale-text relabel -----------------------------------------------------
# (old substring -> 16/9/75 three-compartment equivalent)
_TEXT_SUBS = [
    ("clean_tpm_v4", "clean_tpm_16_9_75"),
    (
        "TPM_clean (v4) is two-compartment fixed-fraction: technical-RNA + "
        "ribosomal-protein genes are forced to 25% of the 1e6 budget and the "
        "remaining (biological) genes to 75%, each renormalized within its group",
        "TPM_clean is three-compartment fixed-fraction: ribosomal-protein genes "
        "are pinned to 16% of the 1e6 budget, other technical-RNA to 9%, and the "
        "remaining (biological) genes to 75%, each renormalized within its "
        "compartment",
    ),
    (
        "two-compartment fixed-fraction clean-TPM (technical 25% / biological "
        "75%, each renormalized within its group)",
        "three-compartment fixed-fraction clean-TPM (ribosomal-protein 16% / "
        "other-technical 9% / biological 75%, each renormalized within its "
        "compartment)",
    ),
]


def relabel_text(value: str) -> str:
    out = str(value)
    for old, new in _TEXT_SUBS:
        out = out.replace(old, new)
    return out


def relabel_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("processing_pipeline", "notes", "source_version"):
        if col in df.columns:
            df[col] = df[col].map(relabel_text)
    return df


# --- parquet index ----------------------------------------------------------
def _norm(name: str) -> str:
    """Normalize a cohort code or parquet stem for matching: lowercase, drop a
    leading ``tcga`` token and all underscores (``HNSC_HPVpos`` and
    ``tcga_hnsc_hpv_pos`` both collapse to ``hnschpvpos``)."""
    s = name.lower().replace("_per_sample_tpm", "").replace(".parquet", "")
    s = re.sub(r"^tcga", "", s)
    return s.replace("_", "")


def parquet_index() -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    for p in glob.glob(str(CACHE / "**" / "derived" / "*per_sample_tpm.parquet"),
                       recursive=True):
        idx.setdefault(_norm(os.path.basename(p)), []).append(Path(p))
    return idx


_RAW_CHECK_COLS = ["TPM_median", "TPM_q1", "TPM_q3", "TPM_mean", "TPM_p90"]


def _raw_matches(recomputed: pd.DataFrame, stored: pd.DataFrame) -> bool:
    """True iff the parquet's recomputed raw stats reproduce the shard's stored
    raw stats (same samples) — the safety gate for accepting a parquet."""
    a = recomputed.set_index("Ensembl_Gene_ID")
    b = stored.set_index("Ensembl_Gene_ID")
    common = a.index.intersection(b.index)
    if len(common) < 0.99 * len(b):
        return False
    diff = (a.loc[common, _RAW_CHECK_COLS] - b.loc[common, _RAW_CHECK_COLS]).abs()
    return float(diff.to_numpy().max()) < 1e-2


def recompute_code(code_rows: pd.DataFrame, candidates: list[Path]):
    """Recompute a single code's rows from the first parquet whose raw stats
    reproduce ``code_rows``. Returns (new_rows, parquet) or (None, None)."""
    m = code_rows.iloc[0]
    for pq_path in candidates:
        pq = pd.read_parquet(pq_path)
        if not set(ID_COLS).issubset(pq.columns):
            continue
        gene_table = pq[list(ID_COLS)].copy()
        sample_cols = [c for c in pq.columns if c not in ID_COLS]
        raw_values = pq[sample_cols]
        new = build_reference_rows(
            gene_table, raw_values,
            cancer_code=m["cancer_code"], source_cohort=m["source_cohort"],
            source_project=m["source_project"],
            source_version=relabel_text(m["source_version"]),
            processing_pipeline=relabel_text(m["processing_pipeline"]),
            notes=relabel_text(m["notes"]), tumor_origin=m["tumor_origin"],
        )
        if "metastasis_site" in code_rows.columns:
            new["metastasis_site"] = m.get("metastasis_site", np.nan)
        if _raw_matches(new, code_rows):
            return new[code_rows.columns], pq_path
    return None, None


def main() -> int:
    idx = parquet_index()
    # biological mask check needs a gene_table; use a representative shard's.
    failures: list[str] = []
    for stem in STALE_SHARDS:
        path = SHARD_DIR / f"{stem}.csv.gz"
        df = pd.read_csv(path)
        out_parts = []
        for code, code_rows in df.groupby("cancer_code", sort=False):
            cands = idx.get(_norm(code), [])
            new_rows, pq = recompute_code(code_rows, cands)
            if new_rows is None:
                failures.append(f"{stem}::{code}")
                out_parts.append(relabel_columns(code_rows.copy()))
                print(f"  ! {stem}::{code}: no matching parquet "
                      f"({len(cands)} candidates) — relabel only")
                continue
            # biology clean unchanged, censored clean changed (within this code's
            # rows, where Ensembl_Gene_ID is unique)
            removable = pd.Series(
                clean_tpm_removal_mask(
                    code_rows[["Ensembl_Gene_ID", "Symbol"]]).to_numpy(),
                index=code_rows["Ensembl_Gene_ID"].to_numpy())
            o = code_rows.set_index("Ensembl_Gene_ID")["TPM_clean_median"]
            n = new_rows.set_index("Ensembl_Gene_ID")["TPM_clean_median"]
            common = o.index.intersection(n.index)
            cen = removable.reindex(common).fillna(False).to_numpy()
            bio = ~cen
            bio_delta = (o.loc[common][bio] - n.loc[common][bio]).abs().max()
            cen_delta = (o.loc[common][cen] - n.loc[common][cen]).abs().max()
            print(f"  + {stem}::{code}: {pq.name} | biology max delta="
                  f"{bio_delta:.4g} (expect ~0), censored max delta={cen_delta:.4g}")
            assert bio_delta < 1e-3, f"biology changed for {stem}::{code}!"
            out_parts.append(new_rows)
        df_out = pd.concat(out_parts, ignore_index=True)
        df_out.to_csv(path, index=False)
    if failures:
        print(f"\nUNMATCHED (need builder rerun): {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
