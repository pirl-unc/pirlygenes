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


from .version import print_name_and_version
from .load import load_all_dataframes

from argh import named, dispatch_commands

@named("data")
def print_dataset_sizes():
    for csv_file, df in load_all_dataframes():
        print("%s: %d rows" % (csv_file, len(df)))

@named("plot-expression")
def plot_expression(csv : str):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv)
    df.plot()
    plt.show()

def main():
    print_name_and_version()
    print("---")
    dispatch_commands([print_dataset_sizes, plot_expression])
 
if __name__ == "__main__":
    main()
