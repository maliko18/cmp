#include "benchmarks.h"
#include "params.h"

#include <algorithm>
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

int drawIndexExcluding(std::mt19937_64& generator, int population,
                       const int* excluded, int excludedCount) {
    int sorted[3] = {-1, -1, -1};
    for (int index = 0; index < excludedCount; ++index) {
        sorted[index] = excluded[index];
    }
    std::sort(sorted, sorted + excludedCount);

    std::uniform_int_distribution<int> distribution(
        0, population - excludedCount - 1);
    // Remappe le rang tire sans biais en sautant les indices exclus.
    int candidate = distribution(generator);
    for (int index = 0; index < excludedCount; ++index) {
        if (candidate >= sorted[index]) {
            ++candidate;
        }
    }
    return candidate;
}

void checkBenchmarks() {
    constexpr int dimension = 10;
    for (int function = 1; function <= 4; ++function) {
        std::vector<float> optimum(static_cast<std::size_t>(dimension));
        for (int coordinate = 0; coordinate < dimension; ++coordinate) {
            optimum[static_cast<std::size_t>(coordinate)] =
                benchmarks::offset(coordinate, function);
        }
        const float actual = benchmarks::fitness(
            optimum.data(), dimension, function);
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

const char* strategyName(Strategy strategy) {
    switch (strategy) {
        case Strategy::Rand1: return "rand1";
        case Strategy::Best1: return "best1";
        case Strategy::CurrentToBest1: return "currenttobest1";
    }
    throw std::invalid_argument("Strategie DE inconnue.");
}

const char* algorithmName(Strategy strategy) {
    switch (strategy) {
        case Strategy::Rand1: return "de-rand1";
        case Strategy::Best1: return "de-best1";
        case Strategy::CurrentToBest1: return "de-currenttobest1";
    }
    throw std::invalid_argument("Strategie DE inconnue.");
}

float runDifferentialEvolution(int function, int dimension, int population,
                               std::uint64_t seed, Strategy strategy,
                               float mutationFactor, float crossoverRate,
                               std::vector<float>& positions,
                               std::vector<float>& fitnesses,
                               std::vector<float>& trials,
                               std::vector<float>& trialFitnesses,
                               std::vector<float>& nextPositions,
                               std::vector<float>& nextFitnesses) {
    const int totalGenerations =
        static_cast<int>((10000LL * dimension) / population);
    const benchmarks::Domain bounds = benchmarks::domain(function);
    std::mt19937_64 generator(seed ^ 0x9E3779B97F4A7C15ULL);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_int_distribution<int> coordinate(0, dimension - 1);

    const auto start = std::chrono::steady_clock::now();
    for (int generation = 1; generation < totalGenerations; ++generation) {
        int bestIndex = 0;
        for (int individual = 1; individual < population; ++individual) {
            if (fitnesses[static_cast<std::size_t>(individual)] <
                fitnesses[static_cast<std::size_t>(bestIndex)]) {
                bestIndex = individual;
            }
        }

        for (int individual = 0; individual < population; ++individual) {
            const int excludedFirst[] = {individual};
            const int r1 = drawIndexExcluding(
                generator, population, excludedFirst, 1);
            const int excludedSecond[] = {individual, r1};
            const int r2 = drawIndexExcluding(
                generator, population, excludedSecond, 2);
            const int excludedThird[] = {individual, r1, r2};
            const int r3 = drawIndexExcluding(
                generator, population, excludedThird, 3);
            const int forcedCoordinate = coordinate(generator);
            const std::size_t individualOffset =
                static_cast<std::size_t>(individual) * dimension;
            const std::size_t r1Offset =
                static_cast<std::size_t>(r1) * dimension;
            const std::size_t r2Offset =
                static_cast<std::size_t>(r2) * dimension;
            const std::size_t r3Offset =
                static_cast<std::size_t>(r3) * dimension;
            const std::size_t bestOffset =
                static_cast<std::size_t>(bestIndex) * dimension;

            for (int coordinateIndex = 0;
                 coordinateIndex < dimension; ++coordinateIndex) {
                float mutant = 0.0f;
                switch (strategy) {
                    case Strategy::Rand1:
                        mutant =
                            positions[r1Offset + coordinateIndex] +
                            mutationFactor *
                                (positions[r2Offset + coordinateIndex] -
                                 positions[r3Offset + coordinateIndex]);
                        break;
                    case Strategy::Best1:
                        mutant =
                            positions[bestOffset + coordinateIndex] +
                            mutationFactor *
                                (positions[r1Offset + coordinateIndex] -
                                 positions[r2Offset + coordinateIndex]);
                        break;
                    case Strategy::CurrentToBest1:
                        mutant =
                            positions[individualOffset + coordinateIndex] +
                            mutationFactor *
                                (positions[bestOffset + coordinateIndex] -
                                 positions[individualOffset + coordinateIndex]) +
                            mutationFactor *
                                (positions[r1Offset + coordinateIndex] -
                                 positions[r2Offset + coordinateIndex]);
                        break;
                }
                const float parent =
                    positions[individualOffset + coordinateIndex];
                float trial = coordinateIndex == forcedCoordinate ||
                                      unit(generator) <= crossoverRate
                    ? mutant : parent;
                trial = std::fmax(bounds.minimum,
                                  std::fmin(bounds.maximum, trial));
                trials[individualOffset + coordinateIndex] = trial;
            }
            trialFitnesses[static_cast<std::size_t>(individual)] =
                benchmarks::fitness(
                    trials.data() + individualOffset, dimension, function);
        }

        for (int individual = 0; individual < population; ++individual) {
            const std::size_t individualOffset =
                static_cast<std::size_t>(individual) * dimension;
            const bool accept =
                trialFitnesses[static_cast<std::size_t>(individual)] <=
                fitnesses[static_cast<std::size_t>(individual)];
            const float* selected = accept
                ? trials.data() + individualOffset
                : positions.data() + individualOffset;
            std::copy(selected, selected + dimension,
                      nextPositions.data() + individualOffset);
            nextFitnesses[static_cast<std::size_t>(individual)] =
                accept
                    ? trialFitnesses[static_cast<std::size_t>(individual)]
                    : fitnesses[static_cast<std::size_t>(individual)];
        }

        positions.swap(nextPositions);
        fitnesses.swap(nextFitnesses);
    }

    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-benchmarks") {
            checkBenchmarks();
            return 0;
        }
        if (argc < 5 || argc > 11 || (argc - 5) % 2 != 0) {
            throw std::invalid_argument(
                "Usage: ./de_seq <fonction 1-4> <dim> <pop> <nb_runs> "
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
                if (value == "rand1") {
                    strategy = Strategy::Rand1;
                } else if (value == "best1") {
                    strategy = Strategy::Best1;
                } else if (value == "currenttobest1") {
                    strategy = Strategy::CurrentToBest1;
                } else {
                    throw std::invalid_argument(
                        "Strategie attendue : rand1, best1 ou currenttobest1.");
                }
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
            std::vector<float> positions(size);
            std::vector<float> fitnesses(
                static_cast<std::size_t>(population));
            std::vector<float> trials(size);
            std::vector<float> trialFitnesses(
                static_cast<std::size_t>(population));
            std::vector<float> nextPositions(size);
            std::vector<float> nextFitnesses(
                static_cast<std::size_t>(population));

            for (int individual = 0; individual < population; ++individual) {
                const std::size_t offset =
                    static_cast<std::size_t>(individual) * dimension;
                for (int coordinate = 0; coordinate < dimension; ++coordinate) {
                    positions[offset + coordinate] = distribution(generator);
                }
                fitnesses[static_cast<std::size_t>(individual)] =
                    benchmarks::fitness(
                        positions.data() + offset, dimension, function);
            }

            const double elapsed = runDifferentialEvolution(
                function, dimension, population, seed, strategy,
                mutationFactor, crossoverRate, positions, fitnesses, trials,
                trialFitnesses, nextPositions, nextFitnesses);
            const float minimum =
                *std::min_element(fitnesses.begin(), fitnesses.end());
            std::cout << algorithmName(strategy) << ",seq," << function << ','
                      << dimension << ',' << population << ',' << run << ','
                      << seed << ',' << minimum << ','
                      << minimum - bounds.bias << ',' << elapsed << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Erreur: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
