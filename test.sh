#!/bin/bash
set -o errexit

./lint.sh
pytest --cov=pirlygenes/ --cov-report=term-missing tests
