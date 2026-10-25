#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

const float kPi = 3.14159265358979323846f;
const float kMutationFactor = 0.5f;
const float kCrossoverRate = 0.9f;

struct Objective {
    int id;
    const char* name;
    float lowerBound;
    float upperBound;
};

Objective getObjective(int id) {
    switch (id) {
        case 0: return {0, "Levy", -5.12f, 5.12f};
        case 1: return {1, "Rastrigin", -5.0f, 5.0f};
        case 2: return {2, "Rosenbrock", -100.0f, 100.0f};
        case 3: return {3, "Griewank", -600.0f, 600.0f};
        case 4: return {4, "Sphere", -100.0f, 100.0f};
        default: throw std::invalid_argument("function_id doit etre compris entre 0 et 4");
    }
}

float evaluate(const std::vector<float>& position, int objectiveId) {
    float result = 0.0f;

    switch (objectiveId) {
        case 0: {  // Levy
            const float firstY = 1.0f + (position[0] - 1.0f) / 4.0f;
            result = std::pow(std::sin(kPi * firstY), 2.0f);
            for (std::size_t i = 0; i + 1 < position.size(); ++i) {
                const float y = 1.0f + (position[i] - 1.0f) / 4.0f;
                const float nextY = 1.0f + (position[i + 1] - 1.0f) / 4.0f;
                result += std::pow(y - 1.0f, 2.0f) *
                          (1.0f + 10.0f * std::pow(std::sin(kPi * nextY), 2.0f));
            }
            const float lastY = 1.0f + (position.back() - 1.0f) / 4.0f;
            return result + std::pow(lastY - 1.0f, 2.0f) *
                             (1.0f + std::pow(std::sin(2.0f * kPi * lastY), 2.0f));
        }
        case 1: {  // Rastrigin: min = 0 at origin
            for (float value : position) {
                result += value * value - 10.0f * std::cos(2.0f * kPi * value) + 10.0f;
            }
            return result;
        }
        case 2: {  // Rosenbrock: min = 0 at (1,1,...,1)
            for (std::size_t i = 0; i + 1 < position.size(); ++i) {
                const float value = position[i];
                const float nextValue = position[i + 1];
                result += 100.0f * std::pow(nextValue - value * value, 2.0f) +
                          std::pow(value - 1.0f, 2.0f);
            }
            return result;
        }
        case 3: {  // Griewank: min = 0 at origin
            float product = 1.0f;
            for (std::size_t i = 0; i < position.size(); ++i) {
                result += position[i] * position[i] / 4000.0f;
                product *= std::cos(position[i] / std::sqrt(static_cast<float>(i + 1)));
            }
            return result - product + 1.0f;
        }
        case 4: {  // Sphere: min = 0 at origin
            for (float value : position) {
                result += value * value;
            }
            return result;
        }
        default:
            throw std::invalid_argument("fonction objectif inconnue");
    }
}

struct Individual {
    std::vector<float> position;
    float fitness;
};

void runSequentialDe(std::size_t dimensions,
                     std::size_t populationSize,
                     std::size_t maxEvaluations,
                     const Objective& objective,
                     unsigned int seed,
                     float targetFitness = -std::numeric_limits<float>::infinity()) {
    if (dimensions == 0 || populationSize < 4) {
        throw std::invalid_argument("dimension doit etre positive et population doit etre au moins 4");
    }
    if (maxEvaluations < populationSize) {
        throw std::invalid_argument("max_evaluations doit etre au moins egal a la population");
    }
    if (objective.id == 2 && dimensions < 2) {
        throw std::invalid_argument("Rosenbrock necessite une dimension d'au moins 2");
    }

    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> positionDistribution(
        objective.lowerBound, objective.upperBound);
    std::uniform_real_distribution<float> unitDistribution(0.0f, 1.0f);
    std::uniform_int_distribution<std::size_t> indexDistribution(0, populationSize - 1);

    std::vector<Individual> population(populationSize);
    std::vector<float> trial(dimensions);
    std::vector<float> mutant(dimensions);
    std::size_t evaluations = 0;
    float globalBestFitness = std::numeric_limits<float>::infinity();

    const auto start = std::chrono::steady_clock::now();

    for (Individual& individual : population) {
        individual.position.resize(dimensions);
        for (float& value : individual.position) {
            value = positionDistribution(generator);
        }
        individual.fitness = evaluate(individual.position, objective.id);
        ++evaluations;
        globalBestFitness = std::min(globalBestFitness, individual.fitness);
    }

    while (evaluations < maxEvaluations && globalBestFitness > targetFitness) {
        for (std::size_t targetIndex = 0;
             targetIndex < populationSize && evaluations < maxEvaluations;
             ++targetIndex) {
            std::size_t firstIndex;
            std::size_t secondIndex;
            std::size_t thirdIndex;
            do { firstIndex = indexDistribution(generator); } while (firstIndex == targetIndex);
            do { secondIndex = indexDistribution(generator); }
            while (secondIndex == targetIndex || secondIndex == firstIndex);
            do { thirdIndex = indexDistribution(generator); }
            while (thirdIndex == targetIndex || thirdIndex == firstIndex || thirdIndex == secondIndex);

            const std::size_t forcedCoordinate =
                static_cast<std::size_t>(generator() % dimensions);
            for (std::size_t d = 0; d < dimensions; ++d) {
                mutant[d] = population[firstIndex].position[d] +
                    kMutationFactor *
                    (population[secondIndex].position[d] - population[thirdIndex].position[d]);

                if (unitDistribution(generator) <= kCrossoverRate || d == forcedCoordinate) {
                    trial[d] = mutant[d];
                } else {
                    trial[d] = population[targetIndex].position[d];
                }
                trial[d] = std::max(objective.lowerBound, std::min(objective.upperBound, trial[d]));
            }

            const float trialFitness = evaluate(trial, objective.id);
            ++evaluations;
            if (trialFitness <= population[targetIndex].fitness) {
                population[targetIndex].position = trial;
                population[targetIndex].fitness = trialFitness;
            }
            globalBestFitness = std::min(globalBestFitness, population[targetIndex].fitness);

            if (globalBestFitness <= targetFitness) break;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();

    std::cout << "Type\tFunction\tDim\tPopulation\tEvaluations\tTime(s)\tMinimum\tSeed\n";
    std::cout << "DE-sequential\t" << objective.name << '\t'
              << dimensions << '\t' << populationSize << '\t' << evaluations << '\t'
              << std::fixed << std::setprecision(6) << elapsedSeconds << '\t'
              << std::setprecision(9) << globalBestFitness << '\t' << seed << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 7) {
            throw std::invalid_argument(
                "usage: de_sequential [dimension] [population] [function_id] [max_evaluations] [seed] [target_fitness]");
        }

        const std::size_t dimensions = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 3;
        const std::size_t populationSize = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 512;
        const int objectiveId = argc > 3 ? std::stoi(argv[3]) : 0;
        const Objective objective = getObjective(objectiveId);
        const std::size_t maxEvaluations = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4]))
            : 30000ULL * populationSize;
        const unsigned int seed = argc > 5
            ? static_cast<unsigned int>(std::stoul(argv[5]))
            : static_cast<unsigned int>(
                  std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const float targetFitness = argc > 6 ? std::stof(argv[6]) : -std::numeric_limits<float>::infinity();

        runSequentialDe(dimensions, populationSize, maxEvaluations, objective, seed, targetFitness);
    } catch (const std::exception& error) {
        std::cerr << "Erreur: " << error.what() << '\n';
        return 1;
    }
    return 0;
}