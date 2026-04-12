#!/bin/bash
set -o errexit

./lint.sh
# `-n auto` fans tests across worker processes via pytest-xdist. If the
# plugin isn't installed, the flag is a no-op pytest error, so we fall
# back to serial. Coverage is collected across workers by pytest-cov.
if python3 -c "import xdist" >/dev/null 2>&1; then
    pytest -n auto --cov=pirlygenes/ --cov-report=term-missing tests
else
    pytest --cov=pirlygenes/ --cov-report=term-missing tests
fi
