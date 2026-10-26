#include "de_kernel.h"

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

using benchmarks::de_parameters::Strategy;

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

/** Convertit un argument texte en flottant fini compris dans un intervalle. */
float parseFloat(const char* text, const char* name, float minimum,
                 float maximum) {
    char* end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value) ||
        value < minimum || value > maximum) {
        throw std::invalid_argument(std::string("Valeur invalide pour ") +
                                    name + ": " + text);
    }
    return value;
}

/** Convertit le nom utilisateur en strategie DE interne. */
Strategy parseStrategy(const std::string& name) {
    if (name == "rand1") {
        return Strategy::Rand1;
    }
    if (name == "best1") {
        return Strategy::Best1;
    }
    if (name == "currenttobest1") {
        return Strategy::CurrentToBest1;
    }
    throw std::invalid_argument(
        "Strategie attendue : rand1, best1 ou currenttobest1.");
}

/** Retourne le nom de l'algorithme inscrit dans le CSV. */
const char* algorithmName(Strategy strategy) {
    switch (strategy) {
        case Strategy::Rand1: return "de-rand1";
        case Strategy::Best1: return "de-best1";
        case Strategy::CurrentToBest1: return "de-currenttobest1";
    }
    throw std::invalid_argument("Strategie DE inconnue.");
}

/** Verifie simultanement les benchmarks CPU et CUDA. */
void checkBenchmarks() {
    constexpr int dimension = 10;
    float gpuResults[4] = {};
    cuda_de_check_benchmarks(gpuResults);
    for (int function = 1; function <= 4; ++function) {
        std::vector<float> optimum(static_cast<std::size_t>(dimension));
        for (int coordinate = 0; coordinate < dimension; ++coordinate) {
            optimum[static_cast<std::size_t>(coordinate)] =
                benchmarks::offset(coordinate, function);
        }
        const float cpuValue = benchmarks::fitness(
            optimum.data(), dimension, function);
        const float expected = benchmarks::domain(function).bias;
        const float gpuValue = gpuResults[function - 1];
        if (std::fabs(cpuValue - expected) > 1.0e-4f ||
            std::fabs(gpuValue - expected) > 1.0e-4f) {
            throw std::runtime_error(
                "Echec du test x=o pour la fonction " +
                std::to_string(function));
        }
        std::cout << "OK fonction " << function << " : CPU=" << cpuValue
                  << ", GPU=" << gpuValue << " (biais " << expected
                  << ")\n";
    }
}

}  // namespace

/**
 * Point d'entree du lanceur DE CUDA.
 *
 * Il prepare une population aleatoire par run, calcule sa fitness initiale,
 * appelle cuda_de et ecrit une ligne CSV contenant le meilleur resultat.
 *
 * @return 0 en cas de succes, 1 si une exception est interceptee.
 */
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-benchmarks") {
            checkBenchmarks();
            return 0;
        }
        if (argc < 5 || argc > 11 || (argc - 5) % 2 != 0) {
            throw std::invalid_argument(
                "Usage: ./de_gpu <fonction 1-4> <dim> <pop> <nb_runs> "
                "[--strategie rand1|best1|currenttobest1] "
                "[--F 0.5] [--CR 0.9]");
        }

        const int function = parsePositiveInt(argv[1], "fonction");
        const int dimension = parsePositiveInt(argv[2], "dim");
        const int population = parsePositiveInt(argv[3], "pop");
        const int runs = parsePositiveInt(argv[4], "nb_runs");
        if (!benchmarks::is_valid_function(function)) {
            throw std::invalid_argument(
                "La fonction doit etre comprise entre 1 et 4.");
        }
        if (population < 4) {
            throw std::invalid_argument(
                "La population DE doit contenir au moins quatre individus.");
        }

        Strategy strategy = benchmarks::de_parameters::defaultStrategy;
        float mutationFactor =
            benchmarks::de_parameters::defaultMutationFactor;
        float crossoverRate =
            benchmarks::de_parameters::defaultCrossoverRate;
        bool strategySpecified = false;
        bool mutationSpecified = false;
        bool crossoverSpecified = false;
        for (int argument = 5; argument < argc; argument += 2) {
            const std::string option(argv[argument]);
            const std::string value(argv[argument + 1]);
            if (option == "--strategie" && !strategySpecified) {
                strategy = parseStrategy(value);
                strategySpecified = true;
            } else if (option == "--F" && !mutationSpecified) {
                mutationFactor = parseFloat(value.c_str(), "F", 0.0f, 2.0f);
                mutationSpecified = true;
            } else if (option == "--CR" && !crossoverSpecified) {
                crossoverRate = parseFloat(value.c_str(), "CR", 0.0f, 1.0f);
                crossoverSpecified = true;
            } else {
                throw std::invalid_argument(
                    "Options attendues : --strategie, --F et --CR "
                    "(sans repetition).");
            }
        }

        const long long budget = 10000LL * dimension;
        if (population > budget) {
            throw std::invalid_argument(
                "La population depasse le budget d'evaluations.");
        }
        if (budget / population > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Trop de generations DE.");
        }
        if (static_cast<std::size_t>(dimension) >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(population)) {
            throw std::invalid_argument("La taille de population est excessive.");
        }

        const std::size_t size =
            static_cast<std::size_t>(dimension) *
            static_cast<std::size_t>(population);
        const benchmarks::Domain bounds = benchmarks::domain(function);
        const std::uint64_t baseSeed =
            static_cast<std::uint64_t>(
                std::chrono::high_resolution_clock::now()
                    .time_since_epoch().count());

        std::cout << "algo,version,fonction,dim,pop,run,seed,minimum,"
                     "erreur_vs_bias,temps_s\n";
        std::cout << std::setprecision(9);
        for (int run = 1; run <= runs; ++run) {
            const std::uint64_t seed =
                baseSeed + static_cast<std::uint64_t>(run - 1);
            std::mt19937_64 generator(seed);
            std::uniform_real_distribution<float> distribution(
                bounds.minimum, bounds.maximum);
            std::vector<float> initialPositions(size);
            std::vector<float> initialFitnesses(
                static_cast<std::size_t>(population));
            std::vector<float> bestPosition(
                static_cast<std::size_t>(dimension));
            float bestFitness = std::numeric_limits<float>::max();

            for (int individual = 0; individual < population; ++individual) {
                const std::size_t offset =
                    static_cast<std::size_t>(individual) * dimension;
                for (int coordinate = 0; coordinate < dimension; ++coordinate) {
                    initialPositions[offset + coordinate] =
                        distribution(generator);
                }
                const float fitness = benchmarks::fitness(
                    initialPositions.data() + offset, dimension, function);
                initialFitnesses[static_cast<std::size_t>(individual)] =
                    fitness;
                if (fitness < bestFitness) {
                    bestFitness = fitness;
                    for (int coordinate = 0; coordinate < dimension;
                         ++coordinate) {
                        bestPosition[static_cast<std::size_t>(coordinate)] =
                            initialPositions[offset + coordinate];
                    }
                }
            }

            double elapsed = 0.0;
            cuda_de(initialPositions.data(), initialFitnesses.data(),
                    bestPosition.data(), &bestFitness, function, dimension,
                    population, seed, strategy, mutationFactor, crossoverRate,
                    &elapsed);
            const float error = bestFitness - bounds.bias;
            std::cout << algorithmName(strategy) << ",gpu," << function << ','
                      << dimension << ',' << population << ',' << run << ','
                      << seed << ',' << bestFitness << ',' << error << ','
                      << elapsed << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Erreur: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
