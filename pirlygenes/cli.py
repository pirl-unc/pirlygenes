"""Stub CLI for backwards compatibility.

The full analysis CLI moved to `trufflepig` in v5.0.0. This module
exists so any user who still has the ``pirlygenes`` console-script on
PATH (or who installs from an older spec) gets a clear migration
message rather than a silent ``ModuleNotFoundError``.

When the next major bump lands, this shim and its console-script entry
can be dropped.
"""

from __future__ import annotations

import sys


_MOVED_MESSAGE = """\
pirlygenes no longer ships an analysis CLI as of v5.0.0.

The `analyze`, `compare-analyze`, `plot-expression`, and
`plot-cancer-cohorts` commands all moved to `trufflepig`:

    pip install trufflepig
    trufflepig run --sample expr.tsv --workspace out --cancer-type BLCA
    trufflepig compare --workspace out/long --inputs out/A,out/B
    trufflepig data
    trufflepig cancers

See https://github.com/pirl-unc/trufflepig for the full migration.
The pirlygenes Python data API is unchanged — `from pirlygenes import
gene_sets_cancer, load_dataset, gene_ids, gene_names, expression_qc`
still works.
"""


def main():
    sys.stderr.write(_MOVED_MESSAGE)
    sys.exit(2)


if __name__ == "__main__":
    main()
