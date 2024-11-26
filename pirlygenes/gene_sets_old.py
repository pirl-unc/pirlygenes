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

# TODO: move these all to CSVs

cytokine_receptors = [
    "IL2RA",
    "IL4R",
    "IL5RA",
    "IL6R",
    "IL17RA",
    "IFNAR1",
    "CD38",
]
fgf_receptors = ["FGFR1", "FGFR2", "FGFR3"]
vegf_receptors = ["FLT1", "KDR", "FLT4"]
bet_genes = ["BRD2", "BRD3", "BRD4", "BRDT"]
FGF_genes = ["FGF19", "FGF21", "FGF23"] + ["FGF%d" % i for i in range(1, 11)]
HDAC_genes = ["HDAC%d" % i for i in range(1, 11)]
Tcell_genes = [
    "CD3D",
    "CD4",
    "CD8A",
    "CD8B",
]
cytolytic_genes = ["GZMB", "PRF1"]

checkpoints = [
    "CD274",
    "PDCD1LG2",
    "CTLA4",
    "PDCD1",
    "TNFRSF4",
    "TNFRSF18",
    "TNFSF4",
    "ICOS",
    "ADORA2A",
    "ADORA2B",
    "BTLA",
    "IDO1",
    "LAG3",
    "HAVCR2",
    "CYBB",
    "VSIR",
    "SIGLEC7",
]

Bcell_genes = ["CD19", "MS4A1", "CD22", "SLAMF7", "CD79A", "CD79B"]
MHC1_genes = ["HLA-A", "HLA-B", "HLA-C", "B2M", "HLA-E"]
APM_genes = ["TAP1", "TAP2", "ERAP1", "ERAP2", "TAPBP", "CALR", "PDIA3", "CANX"]
cyclins = [
    "CCNA1",
    "CCNA2",
    "CCNB1",
    "CCNB2",
    "CCNB3",
    "CCND1",
    "CCND2",
    "CCND3",
    "CCNE1",
    "CCNE2",
]
TFs = ["FOXO1", "FOXO3", "FOXO4", "FOXM1", "ZBTB16", "SALL4"]
mitogens = ["VEGFA", "PDGFA", "PDGFB", "PDGFC", "PDGFD", "EGF", "TGFB2", "TGFB3"]
growth_receptor_genes = (
    ["EGFR", "ERBB2", "ERBB3", "ERBB4"]
    + vegf_receptors
    + fgf_receptors
    + ["PDGFRA", "PDGFRB", "TGFBR1", "TGFBR2"]
)
retinoid_receptor_genes = [
    "RORA",
    "RORB",
    "RORC",
    "RARA",
    "RARB",
    "RARG",
    "RXRA",
    "RXRB",
    "RXRG",
]

oncogenes = [
    "MYC",
    "YAP1",
    "WWTR1",
    "MYC",
    "MYCL",
    "MYCN",
    "BTK",
    "SYK",
    "SRC",
    "ALK",
    "KRAS",
    "HRAS",
    "NRAS",
    "MDM2",
    "CDK4",
    "CDK6",
    "CDK9",
    "RET",
]


TLR_signaling = ["IRAK1", "IRAK2", "IRAK4", "MYD88", "TRAF6", "IRAK3"]

epigenetic = ["NUTM1", "EZH2"] + bet_genes


surface_genes = [
    "TACSTD2",
    "CD276",
    "VTCN1",
    "FOLH1",
    "TNFRSF17",
    "TNFRSF8",
    "CD33",
    "CDH1",
]
