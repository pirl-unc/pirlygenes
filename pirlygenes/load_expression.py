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

import pandas as pd

from .gene_expression import aggregate_gene_expression as tx2gene

def load_expression_data(input_path, aggregate_gene_expression=False):
    
    if ".csv" in input_path:
        df = pd.read_csv(input_path)
    elif ".xlsx" in input_path:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unrecognized file format for {input_path}")
    
    df = df.rename(columns={"Gene Symbol": "gene", "Gene": "gene"})    

    columns = sorted(set(df.columns))


    if "gene" not in columns:
        raise ValueError(f"Gene column not found in {input_path}, available columns: {columns}")
    
    if aggregate_gene_expression:
        df = tx2gene(df)
    
    return df 