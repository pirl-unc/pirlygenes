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

__version__ = "5.20.1"

# Version of the downloadable data bundle (pirlygenes.data_bundle). Bump
# this ONLY when the reference data changes — it pins the bundle filename,
# GitHub-release tag, and on-disk cache. Decoupling it from __version__
# means a code-only release (bug/build-time fix) reuses the last uploaded
# bundle instead of forcing a redundant ~350 MB re-upload. Must always
# point at an existing `pirlygenes-data-v<DATA_VERSION>.tar.gz` release asset.
DATA_VERSION = "5.20.0"

version_string = f"v{__version__}"


def print_version():
    print(version_string)


def print_name_and_version():
    print(f"PIRLy Genes {version_string}")


if __name__ == "__main__":
    print_version()
