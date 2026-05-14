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

from .version import __version__


_MOVED_MESSAGE = """\
pirlygenes no longer ships an analysis CLI as of v5.0.0.

The `analyze`, `compare-analyze`, `plot-expression`, and
`plot-cancer-cohorts` commands all moved to `pirl-trufflepig`
(distributed under that name on PyPI because the bare `trufflepig`
name is owned by an unrelated package; the command + Python import
are both still `trufflepig`):

    pip install pirl-trufflepig
    trufflepig run --sample expr.tsv --workspace out --cancer-type BLCA
    trufflepig compare --workspace out/long --inputs out/A,out/B
    trufflepig data
    trufflepig cancers

See https://github.com/pirl-unc/trufflepig for the full migration.
The pirlygenes Python data API is unchanged — `from pirlygenes import
gene_sets_cancer, load_dataset, gene_ids, gene_names, gene_families`
still works. Expression matrices and QC normalization moved to
`trufflepig.reference` and `trufflepig.expression_qc` respectively.
"""


def main(argv: list[str] | None = None) -> int:
    """Migration-shim entry point.

    ``--help`` and ``--version`` exit ``0`` so CI/wrapper scripts that
    probe the console-script for a CLI don't see a hard failure. Any
    real subcommand invocation exits ``2`` with the migration message
    on stderr.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in {"-h", "--help"}:
        sys.stdout.write(_MOVED_MESSAGE)
        return 0
    if args[0] in {"-V", "--version"}:
        sys.stdout.write(f"pirlygenes {__version__}\n")
        return 0
    sys.stderr.write(_MOVED_MESSAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
