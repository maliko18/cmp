#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARCHIVE="${ROOT_DIR}/../projet-colab.zip"

cd "${ROOT_DIR}"
rm -f "${ARCHIVE}"
zip -r "${ARCHIVE}" common pso de scripts results article Makefile README.md
printf 'Archive Colab creee : %s\n' "${ARCHIVE}"
