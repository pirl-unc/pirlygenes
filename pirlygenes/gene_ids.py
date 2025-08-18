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

from typing import Sequence

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

print("Installed Ensembl genomes: %s" % " ".join([str(g.release) for g in genomes]))


def find_gene_name_from_ensembl_gene_id(
    gene_id: str, verbose: bool = True
) -> str | None:
    gene_name = None
    gene = None

    for genome in genomes:

        try:
            gene = genome.gene_by_id(gene_id)
        except:
            pass
        if gene:
            gene_name = gene.gene_name
        if gene_name:
            if verbose:
                print(
                    "Found %s -> %s in Ensembl v%d"
                    % (gene_id, gene_name, genome.release)
                )
            break
    return gene_name


def find_gene_name_from_ensembl_transcript_id(
    t_id: str, verbose: bool = True
) -> str | None:
    gene_name = None
    t = None

    for g in genomes:

        try:
            t = g.transcript_by_id(t_id)
        except:
            pass
        if t:
            gene_name = t.gene_name
        if gene_name:
            if verbose:
                print("Found %s -> %s in Ensembl v%d" % (t_id, gene_name, g.release))
            break
    return gene_name


def pick_best_gene(genes):
    if len(genes) == 0:
        raise ValueError("Expected at least one gene, got none")

    def sort_key(g):
        # prefer genes with:
        #   - more protein-coding transcripts
        #   - fewer dots in the name (e.g. "TP53" not "AC00003.1"
        #   - shorter name (eg PRAME not PRAMEL2949)
        #   - sort order (e.g. TP53-001 vs TP53-002)
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
) -> tuple[pyensembl.Genome, pyensembl.Gene] | None:

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
            except:
                genes = []
            if len(genes) >= 1:
                return genome, pick_best_gene(genes)


def find_gene_by_name_from_ensembl(name: str, verbose: bool = False) -> str | None:
    result = find_gene_and_ensembl_release_by_name(name, verbose=verbose)
    if result is not None:
        _, gene = result
        return gene


def find_gene_id_by_name_from_ensembl(name: str, verbose: bool = False) -> str | None:
    gene = find_gene_by_name_from_ensembl(name, verbose=verbose)
    if gene is not None:
        return gene.id


def find_canonical_gene_id_and_name(gene_name: str) -> tuple[str | None, str | None]:
    gene = find_gene_by_name_from_ensembl(gene_name)
    if gene:
        return gene.id, gene.name
    else:
        return None, None


def find_canonical_gene_ids_and_names(
    gene_names: Sequence[str],
) -> list[tuple[str | None, str | None]]:

    gene_ids = []
    canonical_gene_names = []

    for gene_name in tqdm(gene_names, "Finding canonical gene IDs and names"):
        gene_id, canonical_name = find_canonical_gene_id_and_name(gene_name)
        print(
            "Found %s -> %s" % (gene_name, gene_id or "None")
            if gene_id
            else "Not found: %s" % gene_name
        )
        gene_ids.append(gene_id)
        canonical_gene_names.append(canonical_name)
    return gene_ids, canonical_gene_names
