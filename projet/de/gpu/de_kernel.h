#pragma once

#include <cstdint>

#include "benchmarks.h"
#include "params.h"

extern "C" void cuda_de(const float* initialPositions,
                        const float* initialFitnesses,
                        float* bestPosition, float* bestFitness,
                        int function, int dimension, int population,
                        std::uint64_t seed,
                        benchmarks::de_parameters::Strategy strategy,
                        float mutationFactor, float crossoverRate,
                        double* elapsedSeconds);

extern "C" void cuda_de_check_benchmarks(float* results);
