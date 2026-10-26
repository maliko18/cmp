#include "kernel.h"

#include <cstdlib>

float host_fitness_function(const float* x, int dim, int function) {
    return benchmarks::fitness(x, dim, function);
}

float getRandom(float low, float high) {
    return low + (high - low) *
        (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
}

float getRandomClamped() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}
