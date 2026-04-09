#!/bin/bash
set -o errexit

ruff check pirlygenes/ \
&& \
echo "Passes ruff check"
