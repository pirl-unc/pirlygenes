#!/bin/bash
set -o errexit

# Coverage is OPT-IN: it's the dominant CPU cost of the suite (the C tracer
# hooks every line on a string-heavy ~1M-row workload). Local dev runs without
# it for speed; CI sets COVERAGE=1 to keep the coverage report. Tests still run
# serially (-n 0) to avoid the all-refs clean-TPM parallel-OOM.
./lint.sh
if [ -n "${COVERAGE:-}" ]; then
    python -m pytest -n 0 --cov=pirlygenes/ --cov-report=term-missing tests
else
    python -m pytest -n 0 tests
fi
