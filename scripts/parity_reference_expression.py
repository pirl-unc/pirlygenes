"""CLI for the #557 reference-expression delegation parity harness.

Thin wrapper over :mod:`pirlygenes.expression.parity`. Writes a per-code CSV plus
a markdown report comparing pirlygenes' delegated source-union compatibility
rows against oncoref's canonical selected/QC-aware artifact view. See the module
docstring for what the deltas mean.

    python scripts/parity_reference_expression.py                 # every code
    python scripts/parity_reference_expression.py --codes PRAD MBL # a subset
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from pirlygenes.expression.parity import (
    DEFAULT_MIN_EXPR,
    format_markdown,
    parity_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codes",
        nargs="*",
        default=None,
        help="cancer_codes to compare (default: every code in the pirlygenes bundle)",
    )
    parser.add_argument("--min-expr", type=float, default=DEFAULT_MIN_EXPR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analyses/outputs/reference_expression_parity"),
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    df = parity_report(args.codes, min_expr=args.min_expr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "parity_by_code.csv"
    md_path = args.out_dir / "parity_report.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(format_markdown(df, args.min_expr))
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")

    ok = df[df["status"] == "ok"]
    if len(ok):
        print(
            f"\n{len(ok)} codes compared; n_samples match "
            f"{int(ok['n_samples_match'].sum())}/{len(ok)}; "
            f"median rel delta {ok['rel_median'].median():.4%}, "
            f"worst p95 {ok['rel_p95'].max():.4%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
