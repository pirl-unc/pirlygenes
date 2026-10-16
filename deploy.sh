#!/usr/bin/env bash
set -euo pipefail

# Keep the historical entry point on the same guarded data + package release path.
exec "${PYTHON:-python3}" "$(dirname "$0")/scripts/release.py" --execute "$@"
