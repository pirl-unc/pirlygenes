"""Degenerate-subtype disambiguation (#198).

Several within-family subtypes share gene-expression signatures that
the classifier can't distinguish on expression alone — e.g. osteosarcoma
vs dedifferentiated liposarcoma both carry the 12q13-15 amplicon
(MDM2 + CDK4 + FRS2); Ewing / DSRCT / ARMS all carry a CD99+
small-blue-round-cell signature. The canonical tiebreaker is either
anatomic site (bone vs retroperitoneum) or a deterministic
fusion-surrogate expression pattern (FATE1/NR0B1 for EWS-FLI1;
NUTM1 + PRAME for NUT carcinoma; CCNB3 for BCOR-rearranged sarcoma).

This module loads two declarative catalogs:

- ``degenerate-subtype-pairs.csv`` — pairs/triples of subtypes that
  share a signature, plus the tiebreaker rule and mapping.
- ``fusion-surrogate-expression.csv`` — genes whose expression is a
  deterministic surrogate for a specific fusion/translocation class
  (ectopic expression from fusion-driven derepression, or promoter-
  swap hyperexpression).

``resolve_degenerate_subtype()`` consumes the catalogs and a minimal
context (current winning subtype, site template, a dict of observed
tumor TPMs) and returns the final subtype plus a provenance dict
explaining the call. When the catalog can't resolve the ambiguity,
the resolver returns ``status='degenerate'`` so the markdown layer
can render ``degenerate between {A, B} — insufficient context to
commit`` rather than forcing a false-confident pick.
"""

from functools import lru_cache

from .load_dataset import get_data


@lru_cache(maxsize=1)
def degenerate_subtype_pairs():
    """Load ``degenerate-subtype-pairs.csv``.

    Returns a DataFrame with ``pair_id``, ``members`` (list of subtype
    codes), ``shared_signature``, ``tiebreaker_rule``,
    ``tiebreaker_mapping`` (dict keyed by context value → subtype),
    and free-text ``notes`` / ``refs``.
    """
    df = get_data("degenerate-subtype-pairs")
    df["members"] = df["members"].fillna("").astype(str).apply(
        lambda s: [m.strip() for m in s.split(";") if m.strip()]
    )

    def _parse_mapping(value):
        if not isinstance(value, str) or not value.strip():
            return {}
        out = {}
        for part in value.split("|"):
            if ":" not in part:
                continue
            key, val = part.split(":", 1)
            out[key.strip()] = val.strip()
        return out

    df["tiebreaker_mapping"] = df["tiebreaker_mapping"].apply(_parse_mapping)
    return df


@lru_cache(maxsize=1)
def fusion_surrogate_expression():
    """Load ``fusion-surrogate-expression.csv``."""
    return get_data("fusion-surrogate-expression")


def _find_pair_for_subtype(subtype_code):
    """Return the degenerate-pair row whose ``members`` contain this
    subtype, or ``None``."""
    pairs = degenerate_subtype_pairs()
    for _, row in pairs.iterrows():
        if subtype_code in row["members"]:
            return row
    return None


def _resolve_site_template(pair_row, site_template):
    """Apply a ``site_template`` tiebreaker. Returns the mapped
    subtype or ``None`` when the site doesn't appear in the mapping."""
    if not site_template:
        return None
    mapping = pair_row["tiebreaker_mapping"] or {}
    return mapping.get(site_template)


def _resolve_fusion_surrogate(pair_row, tumor_tpm_by_symbol, min_tpm=1.0):
    """Apply a ``fusion_surrogate`` tiebreaker. Inspect the mapping
    keys (gene symbols) and count how many hits each target subtype
    gets in the observed TPM dict. Return the subtype with the most
    hits when the margin is ≥ 2, else ``None``."""
    mapping = pair_row["tiebreaker_mapping"] or {}
    if not mapping or not tumor_tpm_by_symbol:
        return None
    votes = {}
    for gene, subtype in mapping.items():
        tpm = tumor_tpm_by_symbol.get(gene)
        if tpm is None:
            continue
        if float(tpm) >= min_tpm:
            votes[subtype] = votes.get(subtype, 0) + 1
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    if len(ranked) == 1:
        return ranked[0][0]
    if ranked[0][1] - ranked[1][1] >= 2:
        return ranked[0][0]
    return None


def _resolve_marker_combo(pair_row, tumor_tpm_by_symbol, min_tpm=1.0):
    """Same shape as fusion_surrogate but used when a marker panel
    (not a fusion) disambiguates — e.g. CCND1 for MCL vs CLL."""
    return _resolve_fusion_surrogate(
        pair_row, tumor_tpm_by_symbol, min_tpm=min_tpm,
    )


def resolve_degenerate_subtype(
    winning_subtype,
    site_template=None,
    tumor_tpm_by_symbol=None,
):
    """Return the final subtype call + provenance.

    Parameters
    ----------
    winning_subtype : str or None
        The subtype the upstream classifier picked. ``None`` skips
        resolution (nothing to disambiguate).
    site_template : str or None
        Top decomposition template (e.g. ``met_bone``,
        ``primary_retroperitoneum``). Drives ``site_template``
        tiebreaker rules.
    tumor_tpm_by_symbol : dict[str, float] or None
        Tumor-attributed TPM per gene symbol. Drives ``fusion_surrogate``
        and ``marker_combo`` tiebreakers.

    Returns
    -------
    dict with keys:
        ``final_subtype`` — the resolved subtype, may equal
            ``winning_subtype`` when the catalog confirms or can't
            rule it out.
        ``status`` — ``'confirmed'`` (tiebreaker agrees with pick),
            ``'corrected'`` (pick swapped for another member),
            ``'degenerate'`` (tiebreaker inconclusive; caller should
            render ambiguity explicitly), or ``'no_pair'`` (subtype
            not in any degenerate pair).
        ``reason`` — short human-readable note for the markdown layer.
        ``pair_id`` — identifier of the matched pair (or ``None``).
        ``alternatives`` — list of the other members of the pair.
    """
    if not winning_subtype:
        return {
            "final_subtype": winning_subtype,
            "status": "no_pair",
            "reason": "",
            "pair_id": None,
            "alternatives": [],
        }

    pair_row = _find_pair_for_subtype(winning_subtype)
    if pair_row is None:
        return {
            "final_subtype": winning_subtype,
            "status": "no_pair",
            "reason": "",
            "pair_id": None,
            "alternatives": [],
        }

    members = pair_row["members"]
    alternatives = [m for m in members if m != winning_subtype]
    rule = pair_row["tiebreaker_rule"]
    pair_id = pair_row["pair_id"]

    resolved = None
    if rule == "site_template":
        resolved = _resolve_site_template(pair_row, site_template)
    elif rule == "fusion_surrogate":
        resolved = _resolve_fusion_surrogate(pair_row, tumor_tpm_by_symbol)
    elif rule == "marker_combo":
        resolved = _resolve_marker_combo(pair_row, tumor_tpm_by_symbol)

    if resolved is None:
        return {
            "final_subtype": winning_subtype,
            "status": "degenerate",
            "reason": (
                f"degenerate between {sorted(members)} — "
                f"{rule} tiebreaker inconclusive "
                f"(shared signature: {pair_row['shared_signature']})"
            ),
            "pair_id": pair_id,
            "alternatives": alternatives,
        }

    if resolved == winning_subtype:
        return {
            "final_subtype": winning_subtype,
            "status": "confirmed",
            "reason": (
                f"{rule} tiebreaker confirms {winning_subtype} "
                f"(shared signature: {pair_row['shared_signature']})"
            ),
            "pair_id": pair_id,
            "alternatives": alternatives,
        }

    return {
        "final_subtype": resolved,
        "status": "corrected",
        "reason": (
            f"{rule} tiebreaker swapped {winning_subtype} → {resolved} "
            f"(shared signature: {pair_row['shared_signature']})"
        ),
        "pair_id": pair_id,
        "alternatives": [m for m in members if m != resolved],
    }


def fusion_surrogate_genes_for(cancer_code):
    """Return the list of surrogate gene symbols associated with the
    given cancer code (either via the ``cancer_code`` column match or
    via a ``pan_cancer`` / semicolon-separated multi-code row)."""
    df = fusion_surrogate_expression()
    hits = []
    for _, row in df.iterrows():
        cc = str(row.get("cancer_code") or "")
        codes = {c.strip() for c in cc.split(";") if c.strip()}
        if cancer_code in codes or "pan_cancer" in codes:
            hits.append({
                "gene": row["surrogate_gene"],
                "fusion_class": row["fusion_class"],
                "role": row["surrogate_role"],
                "rationale": row.get("rationale", ""),
            })
    return hits
