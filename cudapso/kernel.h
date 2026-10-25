#pragma once

#include <cstdint>

#include "benchmarks.h"

enum class GBestMode : int {
    Cpu = 0,
    Gpu = 1
};

float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(const float* x, int dim, int function);

extern "C" void cuda_pso(float* positions, float* velocities,
                         float* pBests, float* pBestFitness, float* gBest,
                         float* bestFitness, int function, int dim, int pop,
                         std::uint64_t seed, GBestMode gBestMode,
                         benchmarks::Variant variant,
                         double* elapsedSeconds);

extern "C" void cuda_check_benchmarks(float* results);
