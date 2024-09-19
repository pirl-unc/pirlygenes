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
    "PVRL4": "NECTIN4",
    "CTAG1B": "NY-ESO-1",
    "TP53": "p53",
    "FOLR1": "FRα",
}

reverse_aliases = {}
for k, v in aliases.items():
    reverse_aliases[v] = k


def get_alias_as_list(name: str) -> list[str]:
    if name in aliases:
        return [aliases[name]]
    else:
        return []


def get_reverse_alias_as_list(name: str) -> list[str]:
    if name in reverse_aliases:
        return [reverse_aliases[name]]
    else:
        return []


def display_name(name: str) -> str:
    return aliases.get(name, name)
