#!/bin/bash
set -o errexit

./lint.sh
# Coverage runs serial to avoid xdist + monkeypatch races on module globals.
pytest -o "addopts=" --cov=pirlygenes/ --cov-report=term-missing tests
