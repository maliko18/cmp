#include "kernel.h"

#include <cstdlib>

/** Delegue l'evaluation CPU a la definition partagee des benchmarks. */
float host_fitness_function(const float* x, int dim, int function) {
    return benchmarks::fitness(x, dim, function);
}

/** Genere un flottant pseudo-aleatoire dans l'intervalle demande. */
float getRandom(float low, float high) {
    return low + (high - low) *
        (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
}

/** Genere un flottant pseudo-aleatoire borne entre zero et un. */
float getRandomClamped() {
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}
