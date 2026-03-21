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

from .load_dataset import get_data


def get_field_from_gene_set(
    gene_set_name,
    candidate_columns,
    try_lower_case=True,
    try_upper_case=True,
    try_no_underscore=True,
    try_plural=True,
):
    if try_plural:
        candidate_columns = candidate_columns + [c + "s" for c in candidate_columns]
    if try_no_underscore:
        candidate_columns = candidate_columns + [
            c.replace("_", "") for c in candidate_columns
        ]
    if try_lower_case:
        candidate_columns = candidate_columns + [c.lower() for c in candidate_columns]
    if try_upper_case:
        candidate_columns = candidate_columns + [c.upper() for c in candidate_columns]

    df = get_data(gene_set_name)
    result = set()
    for column in candidate_columns:
        if column in df.columns:
            for x in df[column]:
                if type(x) is str:
                    result.update([xi.strip() for xi in x.split(";")])
    return result


def get_target_gene_name_set(
    name,
    candidate_symbol_columns=[
        "Tumor_Target_Symbol",
        "Symbol",
        "Gene_Name",
    ],
):
    return get_field_from_gene_set(name, candidate_symbol_columns)


def get_target_gene_id_set(
    name,
    candidate_id_columns=[
        "Tumor_Target_Ensembl_Gene_ID",
        "Tumor_Target_Ensembl_GeneID",
        "Tumor_Target_Gene_ID",
        "Tumor_Target_GeneID",
        "Target_Ensembl_Gene_ID",
        "Target_Ensembl_GeneID",
        "Target_Gene_ID",
        "Target_GeneID",
        "Ensembl_Gene_ID",
        "Ensembl_GeneID",
        "Gene_ID",
    ],
):
    return get_field_from_gene_set(name, candidate_id_columns)


# ---------- ADC ----------
def ADC_trial_target_gene_names():
    return get_target_gene_name_set("ADC-trials")


def ADC_trial_target_gene_ids():
    return get_target_gene_id_set("ADC-trials")


def ADC_approved_target_gene_names():
    return get_target_gene_name_set("ADC-approved")


def ADC_approved_target_gene_ids():
    return get_target_gene_id_set("ADC-approved")


def ADC_target_gene_names():
    return ADC_trial_target_gene_names().union(ADC_approved_target_gene_names())


def ADC_target_gene_ids():
    return ADC_trial_target_gene_ids().union(ADC_approved_target_gene_ids())


# ---------- TCR-T ----------
def TCR_T_trial_target_get_names():
    return get_target_gene_name_set("TCR-T-trials")


def TCR_T_trial_target_get_ids():
    return get_target_gene_id_set("TCR-T-trials")


def TCR_T_target_gene_names():
    return TCR_T_trial_target_get_names()


def TCR_T_target_gene_ids():
    return TCR_T_trial_target_get_ids()


# ---------- CAR-T ----------
def CAR_T_approved_target_gene_names():
    return get_target_gene_name_set("CAR-T-approved")


def CAR_T_approved_target_gene_ids():
    return get_target_gene_id_set("CAR-T-approved")


def CAR_T_target_gene_names():
    return CAR_T_approved_target_gene_names()


def CAR_T_target_gene_ids():
    return CAR_T_approved_target_gene_ids()


# ---------- Multispecific T-cell Engagers ----------
def multispecific_tcell_engager_trial_target_gene_names():
    return get_target_gene_name_set("multispecific-tcell-engager-trials")


def multispecific_tcell_engager_trial_target_gene_ids():
    return get_target_gene_id_set("multispecific-tcell-engager-trials")


def multispecific_tcell_engager_target_gene_names():
    return multispecific_tcell_engager_trial_target_gene_names()


def multispecific_tcell_engager_target_gene_ids():
    return multispecific_tcell_engager_trial_target_gene_ids()


# ---------- Bispecific antibodies ----------
def bispecific_antibody_approved_target_gene_names():
    return get_target_gene_name_set("bispecific-antibodies-approved")


def bispecific_antibody_approved_target_gene_ids():
    return get_target_gene_id_set("bispecific-antibodies-approved")


def bispecific_antibody_target_gene_names():
    return bispecific_antibody_approved_target_gene_names()


def bispecific_antibody_targets_gene_ids():
    return bispecific_antibody_approved_target_gene_ids()


# ---------- Radioligand therapies ----------
def radio_target_gene_names():
    return get_target_gene_name_set("radioligand-targets")


def radio_target_gene_ids():
    return get_target_gene_id_set("radioligand-targets")


def radioligand_target_gene_names():
    return radio_target_gene_names()


def radioligand_target_gene_ids():
    return radio_target_gene_ids()


# ---------- Cancer-testis antigens (CTA) ----------
def CTA_gene_names():
    """All CTA gene symbols (unfiltered)."""
    return get_target_gene_name_set("cancer-testis-antigens")


def CTA_gene_ids():
    """All CTA Ensembl gene IDs (unfiltered)."""
    return get_target_gene_id_set("cancer-testis-antigens")


def _cta_filtered(column):
    from .load_dataset import get_data

    df = get_data("cancer-testis-antigens")
    if "filtered" not in df.columns:
        return set()
    filtered_df = df[df["filtered"].astype(str).str.lower() == "true"]
    result = set()
    if column in df.columns:
        for x in filtered_df[column]:
            if isinstance(x, str):
                result.update([xi.strip() for xi in x.split(";")])
    return result


def CTA_filtered_gene_names():
    """CTA gene symbols that pass the reproductive-tissue filter."""
    return _cta_filtered("Symbol")


def CTA_filtered_gene_ids():
    """CTA Ensembl gene IDs that pass the reproductive-tissue filter."""
    return _cta_filtered("Ensembl_Gene_ID")


def CTA_evidence():
    """Return the full CTA evidence DataFrame with HPA tissue-restriction columns.

    Columns
    -------
    Symbol, Aliases, Full_Name, Function, Ensembl_Gene_ID
        Gene identity fields.
    protein_reproductive : bool or "no data"
        True if all IHC-detected tissues (excluding thymus) are in
        {testis, ovary, placenta}.
    protein_thymus : bool or "no data"
        True if protein detected in thymus.
    rna_reproductive : bool
        True if every tissue with >=1 nTPM (excluding thymus) is in
        {testis, ovary, placenta}.
    rna_thymus : bool
        True if thymus nTPM >= 1.
    protein_reliability : str
        Best HPA antibody reliability for this gene: "Enhanced",
        "Supported", "Approved", "Uncertain", or "no data".
    protein_strict_expression : str
        Semicolon-separated list of tissues where protein is detected
        (excluding thymus), or "no data" / "not detected".
    rna_reproductive_frac : float
        Fraction of total nTPM (excluding thymus) in core reproductive
        tissues, computed from raw nTPM values.
    rna_reproductive_and_thymus_frac : float
        Same but with thymus nTPM added to numerator and denominator.
    rna_deflated_reproductive_frac : float
        Like rna_reproductive_frac but using max(0, nTPM - 1) per tissue
        to suppress low-level basal noise.
    rna_deflated_reproductive_and_thymus_frac : float
        Deflated fraction including thymus in the reproductive numerator.
    rna_80_pct_filter, rna_90_pct_filter, rna_95_pct_filter : bool
        Whether deflated reproductive fraction >= 80/90/95%.
    filtered : bool
        Final inclusion flag with tiered RNA thresholds based on protein
        antibody reliability.  True when protein is reproductive-only
        (or absent) and deflated RNA fraction meets the tier threshold:
        - Enhanced → RNA >=80%
        - Supported → RNA >=90%
        - Approved → RNA >=95%
        - Uncertain or no protein data → RNA >=99%
        Genes with protein in non-reproductive tissues always fail.
    """
    from .load_dataset import get_data

    return get_data("cancer-testis-antigens")
