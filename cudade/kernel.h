#pragma once

#include <cstddef>

struct DeConfig {
    std::size_t dimensions;
    std::size_t populationSize;
    std::size_t maxEvaluations;
    int objectiveId;
    float lowerBound;
    float upperBound;
    float mutationFactor;
    float crossoverRate;
    unsigned int seed;
    float targetFitness;
};

extern "C" void cuda_de(float* population,
                         float* fitness,
                         const DeConfig& config);