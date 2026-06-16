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

from collections import defaultdict

# -----------------------------------------------------------
# Let's rewrite the Ensembl gene names as the more
# commonly used names in the literature for the
# proteins they produce


aliases = {
    "MAGEA1": "MAGE-A1",
    "MAGEA2": "MAGE-A2",
    "MAGEA3": "MAGE-A3",
    "MAGEA4": "MAGE-A4",
    "MAGEA5": "MAGE-A5",
    "MAGEA6": "MAGE-A6",
    "MAGEA10": "MAGE-A10",
    "MAGEA12": "MAGE-A12",
    "PMEL": "gp100",
    "CD276": "B7-H3",
    "VTCN1": "B7-H4",
    "PDCD1LG2": "PD-L2",
    "CD274": "PD-L1",
    "MS4A1": "CD20",
    "FLT1": "VEGFR-1",
    "KDR": "VEGFR-2",
    "FLT4": "VEGFR-3",
    "TNFRSF8": "CD30",
    "TACSTD2": "TROP2",
    "FOLH1": "PSMA",
    "TNFRSF17": "BCMA",
    "PDCD1": "PD-1",
    "TNFRSF4": "OX40",
    "TNFRSF18": "GITR",
    "TNFSF4": "OX40L",
    "CYBB": "NOX2",
    "HAVCR2": "TIM-3",
    "VSIR": "VISTA",
    "POU1F1": "POUF1",
    "ATP5F1E": "ATP5E",
    # NOTE: do NOT add old->current entries here (e.g. PVRL4->NECTIN4). This dict
    # is current_symbol -> preferred *display label* and feeds reverse_aliases /
    # short_gene_name; a backwards entry makes the CURRENT symbol normalise to the
    # dead one (short_gene_name('NECTIN4') -> 'PVRL4'). Old->current synonyms are
    # resolved up front by the NCBI synonym layer (synonym_to_official('PVRL4') ->
    # 'NECTIN4'), so they don't belong here.
    "CTAG1B": "NY-ESO-1",
    "TP53": "p53",
    "FOLR1": "FRα",
    "FCRL5": "FCRH5",
}


# Display labels for *proteoform IDs* — the synthesized identifiers the
# cDNA/protein-identical collapse assigns to a folded paralog group (the merged
# member symbols, e.g. CTAG1A/B; see
# pirlygenes.expression.protein_groups.proteoform_id). These are display-only:
# unlike `aliases`, they do NOT feed the bidirectional mapping synonym pool
# (gene_mapping._curated_display_candidates), because a proteoform ID like
# CTAG1A/B is not a real HGNC locus and must not compete with the real gene
# (CTAG1B) for the reverse "NY-ESO-1 -> official symbol" resolution. A proteoform
# ID with no entry here displays as-is (e.g. XAGE1A/B), which is already legible.
proteoform_display_aliases = {
    "CTAG1A/B": "NY-ESO-1",                          # cDNA-identical NY-ESO pair
    "CT47A1/2/3/4/5/6/7/8/9/10/11/12": "CT47A",      # the 12-locus CT47A cluster
}


reverse_aliases = defaultdict(list)
for k, v in aliases.items():
    reverse_aliases[v].append(k)


def get_alias_as_list(name: str) -> list[str]:
    if name in aliases:
        return [aliases[name]]
    else:
        return []


def get_reverse_alias_as_list(name: str) -> list[str]:
    return reverse_aliases.get(name, [])


def display_name(name: str) -> str:
    if name in aliases:
        return aliases[name]
    return proteoform_display_aliases.get(name, name)


def short_gene_name(name: str) -> str:
    """
    Normalize gene names to their aliases.
    """
    normalized = str(name).strip()
    return sorted(reverse_aliases.get(normalized, [normalized.upper()]), key=len)[0]
