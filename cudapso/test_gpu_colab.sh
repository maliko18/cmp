#!/usr/bin/env bash
set -euo pipefail

nvcc -O2 -arch=sm_75 -o pso_gpu main.cpp kernel.cu
g++ -O2 -std=c++17 -o pso_seq pso_seq.cpp

echo "Vérification des points optimaux x=o"
./pso_gpu --check-benchmarks
./pso_seq --check-benchmarks

for fonction in 1 2 3 4; do
    for variante in base ameliore; do
        echo "PSO séquentiel, fonction ${fonction}, variante ${variante}"
        ./pso_seq "${fonction}" 10 50 3 --variante "${variante}"

        echo "PSO GPU, fonction ${fonction}, variante ${variante}, meilleur global CPU"
        ./pso_gpu "${fonction}" 10 50 3 --gbest cpu --variante "${variante}"

        echo "PSO GPU, fonction ${fonction}, variante ${variante}, meilleur global GPU"
        ./pso_gpu "${fonction}" 10 50 3 --gbest gpu --variante "${variante}"
    done
done
