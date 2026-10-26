#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

make de_gpu de_seq
./de_gpu --check-benchmarks
./de_seq --check-benchmarks

for fonction in 1 2 3 4; do
    for strategie in rand1 best1 currenttobest1; do
        echo "DE sequentiel, fonction ${fonction}, strategie ${strategie}"
        ./de_seq "${fonction}" 10 50 3 --strategie "${strategie}"

        echo "DE GPU, fonction ${fonction}, strategie ${strategie}"
        ./de_gpu "${fonction}" 10 50 3 --strategie "${strategie}"
    done
done
