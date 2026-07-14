"""Parity checks: pirlygenes ``cancer_reference_expression`` vs oncoref's (#207).

pirlygenes ships pre-built per-(gene, cancer_code, source_cohort) summary rows in
``cancer-reference-expression.csv.gz``. oncoref computes the *same* summary rows
on demand from its source-matrix artifact (``oncoref.cancer_reference_expression``).
Both are meant to describe identical cohorts; this module quantifies where they
agree and where they diverge, so we can prove parity before pirlygenes retires
its own builders and becomes a pure consumer of ``oncoref.expression_builders``.

The comparison joins the two frames on ``(cancer_code, Ensembl_Gene_ID)`` for a
fixed normalization (``tpm_clean``) and reports, per cancer_code:

* ``n_samples`` agreement (the reference sample set behind each summary);
* the relative-delta distribution of the median ``expression`` for genes above a
  TPM floor (protein-coding genes agree to ~0.05%; the interesting signal is the
  ncRNA/lncRNA tail where gene-universe / canonicalization choices diverge);
* gene-universe deltas (genes present on only one side).

``scripts/parity_reference_expression.py`` is a thin CLI over this module; the
test suite imports :func:`parity_for_code` for a scoped regression check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# oncoref's read policy for a cohort must match the policy its artifact was built
# with; 'pass' is the common case, a handful of cohorts (e.g. MTC) were baked
# 'pass_or_warn'. We try the stricter policy first and fall back on the exact
# mismatch oncoref raises, so each cohort is read under its own baked policy.
_QC_FALLBACK_ORDER = ("pass", "pass_or_warn", "all")

# TPM floor below which a relative delta is dominated by float/rounding noise and
# a "100% delta" (one side rounds to 0) is not informative on its own.
DEFAULT_MIN_EXPR = 1.0

# Relative-delta threshold above which a scored gene is counted "divergent" — the
# gene-universe / canonicalization tail, an order of magnitude past float noise.
_DIVERGENT_REL = 0.5


def _legacy_clean_reference_frame() -> pd.DataFrame:
    """Load the frozen pirlygenes artifact as the clean-TPM parity schema."""
    from pirlygenes.expression.accessors import _load_cancer_reference_expression

    legacy = _load_cancer_reference_expression()
    if "normalization" in legacy.columns:
        return legacy
    columns = [
        "Ensembl_Gene_ID",
        "cancer_code",
        "source_cohort",
        "n_samples",
        "TPM_clean_median",
    ]
    out = legacy[columns].rename(columns={"TPM_clean_median": "expression"}).copy()
    out["normalization"] = "TPM_clean"
    return out


def _first_int(series: pd.Series, default: int = -1) -> int:
    """First value of ``series`` as an int, or ``default`` when empty/NaN."""
    if not len(series):
        return default
    val = series.iloc[0]
    return default if pd.isna(val) else int(val)


def _oncoref_reference(code: str, normalize: str):
    """Read oncoref's summary rows for one cancer_code under its baked QC policy.

    Returns ``(frame, sample_qc_used)`` or ``(None, reason)`` when oncoref cannot
    serve the code (unknown cohort, missing artifact, ...).
    """
    import oncoref

    last_err = None
    for qc in _QC_FALLBACK_ORDER:
        try:
            df = oncoref.cancer_reference_expression(
                cancer_types=code, normalize=normalize, sample_qc=qc
            )
            return df, qc
        except ValueError as err:
            # Only retry on the sample_qc-policy mismatch; other ValueErrors are
            # real (unknown code, bad normalize) and should surface.
            if "sample_qc" in str(err).lower() and "mismatch" in str(err).lower():
                last_err = err
                continue
            return None, f"{type(err).__name__}: {err}"
        except Exception as err:  # noqa: BLE001 - report, don't crash the sweep
            return None, f"{type(err).__name__}: {err}"
    return None, f"no QC policy matched ({last_err})"


def _pg_single_cohort(pg: pd.DataFrame, *, target_n: int) -> pd.DataFrame:
    """Reduce a per-code pirlygenes slice to one source_cohort.

    Picks the cohort whose ``n_samples`` equals oncoref's reference count
    (``target_n``) — that is the cohort oncoref computed from — and falls back to
    the richest cohort when no count matches. Single-cohort codes pass through
    unchanged. ``Ensembl_Gene_ID`` is de-duplicated defensively.

    If two cohorts tie on ``n_samples == target_n`` the first (groupby-sorted)
    wins; that is a benign ambiguity today (no code has same-size cohorts), and
    the value deltas would expose a wrong pick if one ever arose.
    """
    if pg["source_cohort"].nunique() > 1:
        n_by_cohort = pg.groupby("source_cohort")["n_samples"].first()
        exact = n_by_cohort[n_by_cohort == target_n]
        chosen = exact.index[0] if len(exact) else n_by_cohort.idxmax()
        pg = pg[pg["source_cohort"] == chosen]
    return pg.drop_duplicates("Ensembl_Gene_ID")


def parity_for_code(
    code: str,
    *,
    pg_frame: pd.DataFrame,
    normalize_pg: str = "TPM_clean",
    normalize_on: str = "tpm_clean",
    min_expr: float = DEFAULT_MIN_EXPR,
) -> dict:
    """Compare pirlygenes vs oncoref summary rows for a single cancer_code.

    ``pg_frame`` is a pre-loaded ``cancer_reference_expression()`` frame (loading
    the 4.7M-row CSV once and slicing per code is far cheaper than re-reading).
    Returns a flat dict of parity metrics; ``status`` is ``"ok"`` when both sides
    served the code, otherwise a short reason.
    """
    pg = pg_frame[
        (pg_frame["cancer_code"] == code)
        & (pg_frame["normalization"] == normalize_pg)
    ]
    if pg.empty:
        return {"cancer_code": code, "status": "pg-empty"}

    on, qc_used = _oncoref_reference(code, normalize_on)
    if on is None:
        return {
            "cancer_code": code,
            "status": "oncoref-missing",
            "detail": qc_used,
            "n_samples_pg": int(pg["n_samples"].max()),
            "n_genes_pg": int(pg["Ensembl_Gene_ID"].nunique()),
        }
    if on.empty:
        return {
            "cancer_code": code,
            "status": "oncoref-empty",
            "qc_used": qc_used,
            "n_samples_pg": int(pg["n_samples"].max()),
            "n_genes_pg": int(pg["Ensembl_Gene_ID"].nunique()),
        }

    # oncoref serves one canonical source_cohort per code today. A future
    # group/umbrella code (e.g. a pooled CRC) would instead expand into several
    # sub-cohorts under one label, repeating each Ensembl_Gene_ID per block —
    # that is not a single-cohort summary we can compare gene-for-gene, so flag it
    # rather than silently dedup down to whichever block happens to sort first.
    if on["Ensembl_Gene_ID"].duplicated().any():
        return {
            "cancer_code": code,
            "status": "oncoref-multi-cohort",
            "qc_used": qc_used,
            "n_samples_pg": int(pg["n_samples"].max()),
            "n_genes_pg": int(pg["Ensembl_Gene_ID"].nunique()),
        }

    # pirlygenes' frame can still carry several cohorts (e.g. SARC_DDLPS spans 3).
    # Comparing all pg rows against oncoref's one cohort is a many-to-many join
    # with arbitrary n_samples. Reduce the pg side to the cohort oncoref actually
    # used (matched by sample count; richest as a fallback) so the comparison is
    # apples-to-apples. (Cohort *labels* differ slightly across the two frames, so
    # we match on sample count, not name.)
    n_samp_on = _first_int(on.get("n_reference_samples", pd.Series(dtype=float)))
    pg = _pg_single_cohort(pg, target_n=n_samp_on)

    pg_g = set(pg["Ensembl_Gene_ID"])
    on_g = set(on["Ensembl_Gene_ID"])
    shared = pg_g & on_g

    merged = pg[["Ensembl_Gene_ID", "expression"]].merge(
        on[["Ensembl_Gene_ID", "expression"]],
        on="Ensembl_Gene_ID",
        suffixes=("_pg", "_on"),
    )
    denom = merged["expression_pg"].clip(lower=1e-9)
    merged["rel"] = (merged["expression_on"] - merged["expression_pg"]).abs() / denom

    scored = merged[merged["expression_pg"] >= min_expr]
    # A gene where pg has a real value but oncoref's is NaN yields rel=NaN, which
    # silently drops out of every rel_* metric and the divergent count. Surface it
    # separately so "oncoref can't value a gene pg can" isn't invisible.
    on_missing = scored["expression_on"].isna()
    n_on_missing = int(on_missing.sum())
    rel = scored.loc[~on_missing, "rel"]
    n_divergent = int((rel > _DIVERGENT_REL).sum())

    n_samp_pg = int(pg["n_samples"].iloc[0])

    return {
        "cancer_code": code,
        "status": "ok",
        "qc_used": qc_used,
        "n_samples_pg": n_samp_pg,
        "n_samples_on": n_samp_on,
        "n_samples_match": n_samp_pg == n_samp_on,
        "n_genes_pg": len(pg_g),
        "n_genes_on": len(on_g),
        "n_genes_shared": len(shared),
        "n_genes_pg_only": len(pg_g - on_g),
        "n_genes_on_only": len(on_g - pg_g),
        "n_scored": int(len(scored)),
        "rel_median": float(rel.median()) if len(rel) else np.nan,
        "rel_p95": float(rel.quantile(0.95)) if len(rel) else np.nan,
        "rel_max": float(rel.max()) if len(rel) else np.nan,
        "n_divergent": n_divergent,
        "n_on_missing_value": n_on_missing,
    }


def parity_report(
    codes: list[str] | None = None,
    *,
    min_expr: float = DEFAULT_MIN_EXPR,
) -> pd.DataFrame:
    """Run :func:`parity_for_code` across many cancer_codes; return a summary frame."""
    # The public accessor delegates to oncoref as of #557. Read the frozen
    # artifact through the explicit legacy adapter; calling the wrapper here
    # would compare oncoref with itself and hide migration deltas.
    pg_frame = _legacy_clean_reference_frame()
    if not codes:  # None or empty -> every code in the bundle
        codes = sorted(pg_frame["cancer_code"].unique())

    rows = [
        parity_for_code(code, pg_frame=pg_frame, min_expr=min_expr) for code in codes
    ]
    return pd.DataFrame(rows)


def format_markdown(df: pd.DataFrame, min_expr: float = DEFAULT_MIN_EXPR) -> str:
    """Render a :func:`parity_report` frame as a human-readable markdown report."""
    if "status" not in df.columns:
        return "# cancer_reference_expression parity: pirlygenes vs oncoref (#207)\n\n(no codes compared)\n"
    ok = df[df["status"] == "ok"]
    not_ok = df[df["status"] != "ok"]
    lines = [
        "# cancer_reference_expression parity: pirlygenes vs oncoref (#207)",
        "",
        "Compares pirlygenes' pre-built cohort summary rows against oncoref's "
        "on-demand computation from its source-matrix artifact, per "
        "`(cancer_code, Ensembl_Gene_ID)` at `tpm_clean`. Relative deltas are on "
        f"the median `expression`, over genes with pg TPM >= {min_expr}.",
        "",
        f"- cancer_codes compared: **{len(df)}**",
        f"- served by both sides: **{len(ok)}**",
    ]
    if len(ok):
        lines += [
            f"- n_samples agreement: **{int(ok['n_samples_match'].sum())}/{len(ok)}** codes match exactly",
            f"- median relative delta (across codes): **{ok['rel_median'].median():.4%}**",
            f"- worst-code p95 relative delta: **{ok['rel_p95'].max():.4%}**",
            "",
            "## Per-code detail",
            "",
            "| cancer_code | qc | n_samp pg/on | genes shared (pg-only/on-only) | rel median | rel p95 | divergent |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, r in ok.sort_values("rel_p95", ascending=False).iterrows():
            lines.append(
                f"| {r['cancer_code']} | {r['qc_used']} | "
                f"{r['n_samples_pg']}/{r['n_samples_on']}"
                f"{'' if r['n_samples_match'] else ' warn'} | "
                f"{r['n_genes_shared']} ({r['n_genes_pg_only']}/{r['n_genes_on_only']}) | "
                f"{r['rel_median']:.4%} | {r['rel_p95']:.4%} | {int(r['n_divergent'])} |"
            )
    if len(not_ok):
        lines += ["", "## Not comparable", ""]
        for _, r in not_ok.iterrows():
            detail = str(r.get("detail", "") or "").strip()
            suffix = f": {detail}" if detail else ""
            lines.append(
                f"- `{r['cancer_code']}` — {r['status']}{suffix}"
            )
    lines.append("")
    return "\n".join(lines)
