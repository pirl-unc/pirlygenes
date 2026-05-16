#!/bin/bash
set -o errexit

./lint.sh
python -m pytest -n 0 --cov=pirlygenes/ --cov-report=term-missing tests
