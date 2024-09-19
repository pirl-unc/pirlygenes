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

import pyensembl.shell
from pyensembl.shell import collect_all_installed_ensembl_releases
from .gene_aliases import get_alias_as_list, get_reverse_alias_as_list

genomes = [
    g
    for g in collect_all_installed_ensembl_releases()
    if g.species.latin_name == "homo_sapiens"
]


def find_name_from_ensembl(t_id: str, verbose: bool = True) -> str | None:
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


def find_gene_and_ensembl_release_by_name(
    name: str,
    verbose: bool = False,
) -> tuple[pyensembl.Genome, pyensembl.Gene] | None:

    for genome in genomes:
        candidates = set(
            [name] + get_alias_as_list(name) + get_reverse_alias_as_list(name)
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
            if len(genes) == 1:
                return (genome, genes[0])
            elif len(genes) > 1:
                coding_genes = [
                    gene for gene in genes if gene.biotype == "protein_coding"
                ]
                if len(coding_genes) >= 1:
                    return (genome, coding_genes[0])
                else:
                    return (genome, genes[0])


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

    for gene_name in gene_names:
        gene_id, canonical_name = find_canonical_gene_id_and_name(gene_name)
        gene_ids.append(gene_id)
        canonical_gene_names.append(canonical_name)
    return gene_ids, canonical_gene_names


def find_canonical_names(
    gene_names: Sequence[str],
) -> list[str]:
    return [x[1] for x in find_canonical_gene_ids_and_names(gene_names)]
