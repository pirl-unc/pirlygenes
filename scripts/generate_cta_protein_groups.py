#!/usr/bin/env python
"""Generate cta-protein-groups.csv: CTA genes whose canonical proteins are
>=90% amino-acid identical, collapsed into one "protein group" so that
near-identical paralogs (which one TCR / antibody / vaccine would address
together, and which RNA-seq often can't disambiguate) count once in
coverage/addressability analyses.

Grouping is by translated canonical-transcript protein (synonymous codon
differences are ignored — AA, not CDS), pairwise >=90% similarity
(difflib ratio) with single-linkage clustering. Plus curated overrides for
known same-protein duplicate loci whose *canonical* transcripts differ
(CTAG1A/CTAG1B -> NY-ESO-1; XAGE1A/XAGE1B -> XAGE1).

Re-run after the CTA set or Ensembl release changes:
    python scripts/generate_cta_protein_groups.py
"""
from __future__ import annotations

import difflib
from pathlib import Path

import pandas as pd
from pyensembl import EnsemblRelease

from pirlygenes import gene_sets_cancer as gsc

ENSEMBL_RELEASE = 112
SIMILARITY = 0.90
OUT = Path("pirlygenes/data/cta-protein-groups.csv")

# Curated same-protein groups missed by canonical-transcript AA (the loci pick
# different canonical isoforms) -> {group_name: members}.
_CURATED = {
    "NY-ESO-1": ["CTAG1A", "CTAG1B"],
    "XAGE1": ["XAGE1A", "XAGE1B"],
}


def _protein_seqs():
    rel = EnsemblRelease(ENSEMBL_RELEASE)
    ev = gsc.CTA_evidence()[["Symbol", "Canonical_Transcript_ID"]].dropna()
    prot = {}
    for r in ev.itertuples():
        tid = str(r.Canonical_Transcript_ID).split(".")[0]
        try:
            p = rel.transcript_by_id(tid).protein_sequence
        except Exception:
            p = None
        if p and len(p) > 10:
            prot[r.Symbol] = p.rstrip("*")
    return prot


def _cluster(prot):
    syms = sorted(prot)
    parent = {s: s for s in syms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(syms):
        pa, la = prot[a], len(prot[a])
        for b in syms[i + 1:]:
            pb = prot[b]
            if abs(len(pb) - la) > 0.15 * la:  # length prefilter
                continue
            if difflib.SequenceMatcher(None, pa, pb, autojunk=False).ratio() >= SIMILARITY:
                parent[find(a)] = find(b)
    groups = {}
    for s in syms:
        groups.setdefault(find(s), []).append(s)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def main():
    prot = _protein_seqs()
    auto = _cluster(prot)
    curated_members = {m for ms in _CURATED.values() for m in ms}
    rows = []
    for group_name, members in _CURATED.items():
        for m in members:
            rows.append((m, group_name, "curated_same_protein"))
    for g in auto:
        if set(g) & curated_members:  # don't double-handle curated members
            g = [m for m in g if m not in curated_members]
            if len(g) < 2:
                continue
        name = g[0]  # representative = alphabetically-first member
        for m in g:
            rows.append((m, name, f">={int(SIMILARITY*100)}%_aa_identity"))
    df = pd.DataFrame(rows, columns=["member_symbol", "protein_group", "basis"])
    df = df.sort_values(["protein_group", "member_symbol"]).drop_duplicates("member_symbol")
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {df.member_symbol.nunique()} genes -> "
          f"{df.protein_group.nunique()} protein groups")
    for name, g in df.groupby("protein_group"):
        print(f"  {name}: {list(g.member_symbol)}")


if __name__ == "__main__":
    main()
