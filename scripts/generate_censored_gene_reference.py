#!/usr/bin/env python
"""Generate censored-gene-reference-tpm.csv — the fixed per-gene surrogate
values used by the clean-TPM transform.

clean-TPM zeroes/replaces "censored" genes (technical RNA + ribosomal proteins)
because their quantification is unstable and dominates the zero-sum TPM
denominator differently across pipelines. To keep the *kept* genes from being
inflated AND to make the censored block identical in every cohort/pipeline,
each censored gene is held at a **constant per-gene reference value** rather
than dropped or flattened to one number.

That reference value is each censored gene's **median TPM across the entire
Treehouse 25.01 PolyA compendium** (the canonical reference cohort). Computed
once here and shipped, so a single GDC / GEO / patient sample gets the same
censored-gene values as Treehouse, decoupling the kept genes from each sample's
variable technical/ribosomal fraction.

    python scripts/generate_censored_gene_reference.py
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pirlygenes.expression.normalize import clean_tpm_removal_mask

CACHE = Path.home() / ".cache" / "pirlygenes" / "expression" / "treehouse-polya-25-01"
TPM_TSV = CACHE / "Tumor-25.01-Polya_hugo_log2tpm_58581genes_2025-02-27.tsv"
OUT = Path("pirlygenes/data/censored-gene-reference-tpm.csv")


def main() -> int:
    # 1. classify which compendium genes are censored (technical RNA + ribosomal
    #    protein). Symbol-keyed compendium; classification is symbol-driven.
    syms = pd.read_csv(TPM_TSV, sep="\t", usecols=[0]).iloc[:, 0].astype(str)
    gt = pd.DataFrame({"Symbol": syms, "Ensembl_Gene_ID": ""})
    mask = clean_tpm_removal_mask(gt)  # default: technical + ribosomal protein
    censored = set(syms[mask.to_numpy()])
    print(f"censored genes to reference: {len(censored)} of {len(syms)}")

    # 2. extract the censored rows across all samples (one awk pass).
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(sorted(censored)) + "\n")
        symfile = fh.name
    rows = Path(tempfile.mktemp(suffix=".tsv"))
    awk = ('BEGIN{while((getline l < "%s")>0) S[l]=1}'
           "NR==1||($1 in S)" % symfile)
    with rows.open("w") as out:
        subprocess.run(["awk", "-F\t", awk, str(TPM_TSV)], stdout=out, check=True)

    # 3. inverse log2(TPM+1) -> TPM, median per gene across all samples.
    raw = pd.read_csv(rows, sep="\t")
    raw = raw.rename(columns={raw.columns[0]: "Symbol"}).set_index("Symbol")
    tpm = np.clip(np.power(2.0, raw.to_numpy(dtype=float)) - 1.0, 0.0, None)
    median = pd.Series(np.median(tpm, axis=1), index=raw.index)

    out_df = (pd.DataFrame({"Symbol": median.index,
                            "reference_tpm": median.values.round(4)})
              .sort_values("Symbol"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out_df)} censored-gene reference TPMs "
          f"(sum={out_df.reference_tpm.sum():.0f}); "
          f"top: {out_df.nlargest(5, 'reference_tpm').to_dict('records')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
