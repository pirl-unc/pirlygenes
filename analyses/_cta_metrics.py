"""Shared CTA metric joins in the cDNA-identical expression space."""

from pirlygenes.expression.protein_groups import symbol_to_canonical


def fold_specific_9mer_weights(weights):
    """Use the largest member weight once per cDNA-identical proteoform.

    This uses the same collapse as the expression matrix, including curated
    overrides. The older 90%-identity protein-family table describes a different
    grouping and must not supply metric join keys.
    """
    members = symbol_to_canonical(kind="cdna")
    folded = {}
    for symbol, weight in weights.items():
        symbol = str(symbol).strip().upper()
        group = str(members.get(symbol, symbol)).upper()
        folded[group] = max(folded.get(group, 0.0), float(weight))
    return folded
