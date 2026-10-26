#include "benchmarks.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/** Convertit un argument texte en entier strictement positif. */
int parsePositiveInt(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("Valeur invalide pour ") +
                                    name + ": " + text);
    }
    return static_cast<int>(value);
}

/** Verifie les valeurs optimales des quatre benchmarks en mode CPU. */
void checkBenchmarks() {
    constexpr int dim = 10;
    for (int function = 1; function <= 4; ++function) {
        std::vector<float> optimum(static_cast<std::size_t>(dim));
        for (int coordinate = 0; coordinate < dim; ++coordinate) {
            optimum[static_cast<std::size_t>(coordinate)] =
                benchmarks::offset(coordinate, function);
        }
        const float actual =
            benchmarks::fitness(optimum.data(), dim, function);
        const float expected = benchmarks::domain(function).bias;
        if (std::fabs(actual - expected) > 1.0e-4f) {
            throw std::runtime_error(
                "Echec du test x=o pour la fonction " +
                std::to_string(function));
        }
        std::cout << "OK fonction " << function << " : " << actual
                  << " (biais " << expected << ")\n";
    }
}

/**
 * Execute les iterations du PSO sequentiel.
 *
 * @return Temps d'execution des iterations, en secondes.
 *
 * La variante amelioree ajoute une inertie decroissante et une borne de
 * vitesse afin de limiter les divergences dans les grandes dimensions.
 */
float runPso(int function, int dim, int pop, std::uint64_t seed,
             benchmarks::Variant variant,
             std::vector<float>& positions, std::vector<float>& velocities,
             std::vector<float>& pBests, std::vector<float>& pBestFitness,
             std::vector<float>& gBest, float& bestFitness) {
    const std::size_t size =
        static_cast<std::size_t>(dim) * static_cast<std::size_t>(pop);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> randomWeight(0.0f, 1.0f);
    const int iterations = static_cast<int>((10000LL * dim) / pop);
    std::vector<float> newPositions(size);
    std::vector<float> newVelocities(size);
    const benchmarks::Domain bounds = benchmarks::domain(function);
    const float maximumVelocity = benchmarks::maximum_velocity(function);
    const auto start = std::chrono::steady_clock::now();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const float omega =
            benchmarks::inertia(variant, iteration, iterations);
        for (int particle = 0; particle < pop; ++particle) {
            const std::size_t startIndex =
                static_cast<std::size_t>(particle) * dim;
            for (int coordinate = 0; coordinate < dim; ++coordinate) {
                const std::size_t index =
                    startIndex + static_cast<std::size_t>(coordinate);
                const float r1 = randomWeight(generator);
                const float r2 = randomWeight(generator);
                float velocity =
                    omega * velocities[index] +
                    benchmarks::pso_parameters::cognitive * r1 *
                        (pBests[index] - positions[index]) +
                    benchmarks::pso_parameters::social * r2 *
                        (gBest[static_cast<std::size_t>(coordinate)] -
                         positions[index]);
                if (variant == benchmarks::Variant::Improved) {
                    velocity = std::fmax(
                        -maximumVelocity,
                        std::fmin(maximumVelocity, velocity));
                }
                float position = positions[index] + velocity;
                position = std::fmax(bounds.minimum,
                                     std::fmin(bounds.maximum, position));
                newVelocities[index] = velocity;
                newPositions[index] = position;
            }
        }
        velocities.swap(newVelocities);
        positions.swap(newPositions);

        for (int particle = 0; particle < pop; ++particle) {
            const std::size_t startIndex =
                static_cast<std::size_t>(particle) * dim;
            const float* position = positions.data() + startIndex;
            const float fitness =
                benchmarks::fitness(position, dim, function);
            if (fitness < pBestFitness[static_cast<std::size_t>(particle)]) {
                pBestFitness[static_cast<std::size_t>(particle)] = fitness;
                for (int coordinate = 0; coordinate < dim; ++coordinate) {
                    pBests[startIndex +
                           static_cast<std::size_t>(coordinate)] =
                        position[coordinate];
                }
            }
            if (pBestFitness[static_cast<std::size_t>(particle)] <
                bestFitness) {
                bestFitness =
                    pBestFitness[static_cast<std::size_t>(particle)];
                for (int coordinate = 0; coordinate < dim; ++coordinate) {
                    gBest[static_cast<std::size_t>(coordinate)] =
                        pBests[startIndex +
                               static_cast<std::size_t>(coordinate)];
                }
            }
        }
    }

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

}  // namespace

/**
 * Point d'entree du lanceur PSO sequentiel.
 *
 * Il prepare la population initiale, execute un ou plusieurs runs et ecrit
 * les resultats au format CSV.
 */
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-benchmarks") {
            checkBenchmarks();
            return 0;
        }
        if (argc != 5 && argc != 7) {
            throw std::invalid_argument(
                "Usage: ./pso_seq <fonction 1-4> <dim> <pop> <nb_runs> "
                "[--variante base|ameliore]");
        }

        const int function = parsePositiveInt(argv[1], "fonction");
        const int dim = parsePositiveInt(argv[2], "dim");
        const int pop = parsePositiveInt(argv[3], "pop");
        const int runs = parsePositiveInt(argv[4], "nb_runs");
        benchmarks::Variant variant = benchmarks::Variant::Base;
        if (argc == 7) {
            if (std::string(argv[5]) != "--variante" ||
                (std::string(argv[6]) != "base" &&
                 std::string(argv[6]) != "ameliore")) {
                throw std::invalid_argument(
                    "L'option doit etre --variante base ou --variante ameliore.");
            }
            if (std::string(argv[6]) == "ameliore") {
                variant = benchmarks::Variant::Improved;
            }
        }
        if (!benchmarks::is_valid_function(function)) {
            throw std::invalid_argument("La fonction doit etre comprise entre 1 et 4.");
        }
        if (pop > 10000LL * dim) {
            throw std::invalid_argument(
                "La population doit etre inferieure ou egale au budget d'evaluations.");
        }

        const std::uint64_t baseSeed =
            static_cast<std::uint64_t>(
                std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count());
        std::cout << "algo,version,fonction,dim,pop,run,seed,minimum,"
                     "erreur_vs_bias,temps_s\n";
        std::cout << std::setprecision(9);

        for (int run = 1; run <= runs; ++run) {
            const std::uint64_t seed = baseSeed +
                static_cast<std::uint64_t>(run - 1);
            const std::size_t size =
                static_cast<std::size_t>(dim) *
                static_cast<std::size_t>(pop);
            std::vector<float> positions(size);
            std::vector<float> velocities(size, 0.0f);
            std::vector<float> pBests(size);
            std::vector<float> pBestFitness(static_cast<std::size_t>(pop));
            std::vector<float> gBest(static_cast<std::size_t>(dim));
            float bestFitness = std::numeric_limits<float>::max();

            const benchmarks::Domain bounds = benchmarks::domain(function);
            std::mt19937_64 generator(seed);
            std::uniform_real_distribution<float> distribution(
                bounds.minimum, bounds.maximum);
            for (int particle = 0; particle < pop; ++particle) {
                const std::size_t startIndex =
                    static_cast<std::size_t>(particle) * dim;
                for (int coordinate = 0; coordinate < dim; ++coordinate) {
                    positions[startIndex +
                              static_cast<std::size_t>(coordinate)] =
                        distribution(generator);
                    pBests[startIndex +
                           static_cast<std::size_t>(coordinate)] =
                        positions[startIndex +
                                  static_cast<std::size_t>(coordinate)];
                }
                const float fitness = benchmarks::fitness(
                    positions.data() + startIndex, dim, function);
                pBestFitness[static_cast<std::size_t>(particle)] = fitness;
                if (fitness < bestFitness) {
                    bestFitness = fitness;
                    for (int coordinate = 0; coordinate < dim; ++coordinate) {
                        gBest[static_cast<std::size_t>(coordinate)] =
                            positions[startIndex +
                                      static_cast<std::size_t>(coordinate)];
                    }
                }
            }

            const double elapsed = runPso(
                function, dim, pop, seed, variant, positions, velocities, pBests,
                pBestFitness, gBest, bestFitness);
            const float error = bestFitness - bounds.bias;
            const char* algorithm =
                variant == benchmarks::Variant::Improved
                    ? "pso-ameliore" : "pso";
            std::cout << algorithm << ",seq," << function << ',' << dim << ',' << pop
                      << ',' << run << ',' << seed << ',' << bestFitness << ','
                      << error << ',' << elapsed << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Erreur: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
