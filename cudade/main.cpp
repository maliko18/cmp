#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "kernel.h"

namespace {

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

// Host-side fitness function for verification (matches device exactly)
float evaluate(const std::vector<float>& position, int objectiveId) {
    const float pi = 3.14159265358979323846f;
    float result = 0.0f;

    switch (objectiveId) {
        case 0: {  // Levy
            const float firstY = 1.0f + (position[0] - 1.0f) / 4.0f;
            result = std::pow(std::sin(pi * firstY), 2.0f);
            for (std::size_t i = 0; i + 1 < position.size(); ++i) {
                const float y = 1.0f + (position[i] - 1.0f) / 4.0f;
                const float nextY = 1.0f + (position[i + 1] - 1.0f) / 4.0f;
                result += std::pow(y - 1.0f, 2.0f) *
                          (1.0f + 10.0f * std::pow(std::sin(pi * nextY), 2.0f));
            }
            const float lastY = 1.0f + (position.back() - 1.0f) / 4.0f;
            return result + std::pow(lastY - 1.0f, 2.0f) *
                             (1.0f + std::pow(std::sin(2.0f * pi * lastY), 2.0f));
        }
        case 1: {  // Rastrigin
            for (float value : position) {
                result += value * value - 10.0f * std::cos(2.0f * pi * value) + 10.0f;
            }
            return result;
        }
        case 2: {  // Rosenbrock
            for (std::size_t i = 0; i + 1 < position.size(); ++i) {
                const float value = position[i];
                const float nextValue = position[i + 1];
                result += 100.0f * std::pow(nextValue - value * value, 2.0f) +
                          std::pow(value - 1.0f, 2.0f);
            }
            return result;
        }
        case 3: {  // Griewank
            float product = 1.0f;
            for (std::size_t i = 0; i < position.size(); ++i) {
                result += position[i] * position[i] / 4000.0f;
                product *= std::cos(position[i] / std::sqrt(static_cast<float>(i + 1)));
            }
            return result - product + 1.0f;
        }
        case 4: {  // Sphere
            for (float value : position) {
                result += value * value;
            }
            return result;
        }
        default:
            throw std::invalid_argument("fonction objectif inconnue");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 7) {
            throw std::invalid_argument(
                "usage: de_cuda [dimension] [population] [function_id] [max_evaluations] [seed] [target_fitness]");
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

        if (dimensions == 0 || populationSize < 4 || maxEvaluations < populationSize) {
            throw std::invalid_argument("dimension positive, population >= 4 et budget valide requis");
        }
        if (objectiveId == 2 && dimensions < 2) {
            throw std::invalid_argument("Rosenbrock necessite une dimension d'au moins 2");
        }

        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> positionDistribution(
            objective.lowerBound, objective.upperBound);

        std::vector<float> population(populationSize * dimensions);
        std::vector<float> fitness(populationSize);

        const auto start = std::chrono::steady_clock::now();
        for (float& value : population) {
            value = positionDistribution(generator);
        }
        for (std::size_t i = 0; i < populationSize; ++i) {
            std::vector<float> position(dimensions);
            std::copy_n(population.begin() + i * dimensions, dimensions, position.begin());
            fitness[i] = evaluate(position, objectiveId);
        }

        DeConfig config{dimensions, populationSize, maxEvaluations, objectiveId,
                        objective.lowerBound, objective.upperBound, 0.5f, 0.9f, seed, targetFitness};
        cuda_de(population.data(), fitness.data(), config);

        float bestFitness = std::numeric_limits<float>::infinity();
        for (float value : fitness) {
            bestFitness = std::min(bestFitness, value);
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsedSeconds = std::chrono::duration<double>(end - start).count();

        std::cout << "Type\tFunction\tDim\tPopulation\tEvaluations\tTime(s)\tMinimum\tSeed\n";
        std::cout << "DE-GPU\t" << objective.name << '\t' << dimensions << '\t'
                  << populationSize << '\t' << maxEvaluations << '\t'
                  << std::fixed << std::setprecision(6) << elapsedSeconds << '\t'
                  << std::setprecision(9) << bestFitness << '\t' << seed << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Erreur: " << error.what() << '\n';
        return 1;
    }

    return 0;
}