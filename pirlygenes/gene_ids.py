# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Tuple, List

from tqdm import tqdm
import pyensembl
from pyensembl.shell import collect_all_installed_ensembl_releases

from .gene_names import get_alias_as_list, get_reverse_alias_as_list, short_gene_name

genomes = sorted(
    [
        g
        for g in collect_all_installed_ensembl_releases()
        if g.species.latin_name == "homo_sapiens"
    ],
    reverse=True,
    key=lambda g: g.release,
)

_installed_releases = " ".join([str(g.release) for g in genomes])

# ---------------------------------------------------------------------------
# Batch indexes — built once from the latest installed Ensembl release.
# ---------------------------------------------------------------------------

_gene_id_to_name: dict = {}
_transcript_id_to_gene_name: dict = {}
_indexes_built = False


def _build_indexes():
    global _gene_id_to_name, _transcript_id_to_gene_name, _indexes_built
    if _indexes_built:
        return
    if not genomes:
        _indexes_built = True
        return
    g = genomes[0]
    print(f"[index] Building gene/transcript index from Ensembl release {g.release}...")
    try:
        for gene in g.genes():
            gid = gene.id.split(".")[0]
            _gene_id_to_name[gid] = gene.name
    except Exception:
        pass
    try:
        for t in g.transcripts():
            tid = t.id.split(".")[0]
            _transcript_id_to_gene_name[tid] = t.gene_name
    except Exception:
        pass
    print(f"[index] {len(_gene_id_to_name)} genes, {len(_transcript_id_to_gene_name)} transcripts indexed")
    _indexes_built = True


def find_gene_name_from_ensembl_gene_id(
    gene_id: str, verbose: bool = False
) -> Optional[str]:
    _build_indexes()
    name = _gene_id_to_name.get(gene_id.split(".")[0])
    if name and verbose:
        print("Found %s -> %s" % (gene_id, name))
    return name


def find_gene_name_from_ensembl_transcript_id(
    t_id: str, verbose: bool = False
) -> Optional[str]:
    _build_indexes()
    name = _transcript_id_to_gene_name.get(t_id.split(".")[0])
    if name and verbose:
        print("Found %s -> %s" % (t_id, name))
    return name


def pick_best_gene(genes):
    if len(genes) == 0:
        raise ValueError("Expected at least one gene, got none")

    def sort_key(g):
        num_protein_coding = sum([t.is_protein_coding for t in g.transcripts])
        return (
            num_protein_coding,
            -g.name.count("."),
            len(g.name),
            g.name,
        )

    sorted_genes = sorted(genes, key=sort_key, reverse=True)
    if len(sorted_genes) == 0:
        raise ValueError("Lost genes after sorting: %s" % genes)
    return sorted_genes[0]


def find_gene_and_ensembl_release_by_name(
    name: str,
    verbose: bool = False,
) -> Optional[Tuple[pyensembl.Genome, pyensembl.Gene]]:

    for genome in genomes:
        candidates = set(
            [name, short_gene_name(name)]
            + get_alias_as_list(name)
            + get_reverse_alias_as_list(name)
        )
        for n in list(candidates):
            candidates.add(n.lower())
            candidates.add(n.upper())

        for n in candidates:
            if verbose:
                print("--> %s: %s" % (genome, n))
            try:
                genes = genome.genes_by_name(n)
            except Exception:
                genes = []
            if len(genes) >= 1:
                return genome, pick_best_gene(genes)


def find_gene_by_name_from_ensembl(
    name: str, verbose: bool = False
) -> Optional[pyensembl.Gene]:
    result = find_gene_and_ensembl_release_by_name(name, verbose=verbose)
    if result is not None:
        _, gene = result
        return gene


def find_gene_id_by_name_from_ensembl(name: str, verbose: bool = False) -> Optional[str]:
    gene = find_gene_by_name_from_ensembl(name, verbose=verbose)
    if gene is not None:
        return gene.id


def find_canonical_gene_id_and_name(gene_name: str) -> Tuple[Optional[str], Optional[str]]:
    gene = find_gene_by_name_from_ensembl(gene_name)
    if gene:
        return gene.id, gene.name
    else:
        return None, None


def find_canonical_gene_ids_and_names(
    gene_names: Sequence[str],
    verbose: bool = False,
) -> Tuple[List[Optional[str]], List[Optional[str]]]:

    gene_ids = []
    canonical_gene_names = []

    for gene_name in tqdm(gene_names, "Finding canonical gene IDs and names", disable=not verbose):
        gene_id, canonical_name = find_canonical_gene_id_and_name(gene_name)
        if verbose:
            print(
                "Found %s -> %s" % (gene_name, gene_id or "None")
                if gene_id
                else "Not found: %s" % gene_name
            )
        gene_ids.append(gene_id)
        canonical_gene_names.append(canonical_name)
    return gene_ids, canonical_gene_names
