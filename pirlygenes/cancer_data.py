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


def get_target_gene_set(
    name, columns=["Tumor_Target_Symbol", "Tumor_Target_Symbols", "Symbols"]
):
    df = get_data(name)
    genes = set()
    for column in columns:
        if column in df.columns:
            for x in df[column]:
                if type(x) is str:
                    genes.update([xi.strip() for xi in x.split(";")])
    return genes


def get_ADC_trial_targets():
    return get_target_gene_set("ADC-trials")


def get_ADC_approved_targets():
    return get_target_gene_set("ADC-approved")


def get_ADC_targets():
    return get_ADC_trial_targets().union(get_ADC_approved_targets())


def get_TCR_T_trial_targets():
    return get_target_gene_set("TCR-T-trials")


def get_TCR_T_targets():
    return get_TCR_T_trial_targets()


def get_CAR_T_approved_targets():
    return get_target_gene_set("CAR-T-approved")


def get_CAR_T_targets():
    return get_CAR_T_approved_targets()


def get_multispecific_tcell_engager_trial_targets():
    return get_target_gene_set("multispecific-tcell-engager-trials")


def get_multispecific_tcell_engager_targets():
    return get_multispecific_tcell_engager_trial_targets()


def get_bispecific_antibody_approved_targets():
    return get_target_gene_set("bispecific-antibodies-approved")


def get_bispecific_antibody_targets():
    return get_bispecific_antibody_approved_targets()
