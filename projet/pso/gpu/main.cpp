#include "kernel.h"

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

void checkBenchmarks() {
    constexpr int dim = 10;
    float gpuResults[4] = {};
    cuda_check_benchmarks(gpuResults);
    for (int function = 1; function <= 4; ++function) {
        std::vector<float> optimum(static_cast<std::size_t>(dim));
        for (int coordinate = 0; coordinate < dim; ++coordinate) {
            optimum[static_cast<std::size_t>(coordinate)] =
                benchmarks::offset(coordinate, function);
        }
        const float actual =
            benchmarks::fitness(optimum.data(), dim, function);
        const float expected = benchmarks::domain(function).bias;
        const float gpuActual = gpuResults[function - 1];
        if (std::fabs(actual - expected) > 1.0e-4f ||
            std::fabs(gpuActual - expected) > 1.0e-4f) {
            throw std::runtime_error(
                "Echec du test x=o pour la fonction " +
                std::to_string(function));
        }
        std::cout << "OK fonction " << function << " : CPU=" << actual
                  << ", GPU=" << gpuActual << " (biais " << expected
                  << ")\n";
    }
}

void initializeSwarm(int function, int dim, int pop, std::uint64_t seed,
                     std::vector<float>& positions,
                     std::vector<float>& velocities,
                     std::vector<float>& pBests,
                     std::vector<float>& pBestFitness,
                     std::vector<float>& gBest, float& bestFitness) {
    const std::size_t size =
        static_cast<std::size_t>(dim) * static_cast<std::size_t>(pop);
    positions.resize(size);
    velocities.assign(size, 0.0f);
    pBests.resize(size);
    pBestFitness.resize(static_cast<std::size_t>(pop));
    gBest.resize(static_cast<std::size_t>(dim));

    const benchmarks::Domain bounds = benchmarks::domain(function);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> distribution(
        bounds.minimum, bounds.maximum);

    for (int particle = 0; particle < pop; ++particle) {
        float* position = positions.data() +
            static_cast<std::size_t>(particle) * dim;
        for (int coordinate = 0; coordinate < dim; ++coordinate) {
            position[coordinate] = distribution(generator);
            pBests[static_cast<std::size_t>(particle) * dim + coordinate] =
                position[coordinate];
        }
        pBestFitness[static_cast<std::size_t>(particle)] =
            benchmarks::fitness(position, dim, function);
        if (particle == 0 ||
            pBestFitness[static_cast<std::size_t>(particle)] < bestFitness) {
            bestFitness = pBestFitness[static_cast<std::size_t>(particle)];
            for (int coordinate = 0; coordinate < dim; ++coordinate) {
                gBest[static_cast<std::size_t>(coordinate)] =
                    position[coordinate];
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--check-benchmarks") {
            checkBenchmarks();
            return 0;
        }
        if (argc < 5 || argc > 9 || (argc - 5) % 2 != 0) {
            throw std::invalid_argument(
                "Usage: ./pso_gpu <fonction 1-4> <dim> <pop> <nb_runs> "
                "[--gbest cpu|gpu] [--variante base|ameliore]");
        }

        const int function = parsePositiveInt(argv[1], "fonction");
        const int dim = parsePositiveInt(argv[2], "dim");
        const int pop = parsePositiveInt(argv[3], "pop");
        const int runs = parsePositiveInt(argv[4], "nb_runs");
        if (!benchmarks::is_valid_function(function)) {
            throw std::invalid_argument("La fonction doit etre comprise entre 1 et 4.");
        }
        GBestMode mode = GBestMode::Cpu;
        benchmarks::Variant variant = benchmarks::Variant::Base;
        bool gBestSpecified = false;
        bool variantSpecified = false;
        for (int argument = 5; argument < argc; argument += 2) {
            const std::string option(argv[argument]);
            const std::string value(argv[argument + 1]);
            if (option == "--gbest" && !gBestSpecified &&
                (value == "cpu" || value == "gpu")) {
                mode = value == "gpu" ? GBestMode::Gpu : GBestMode::Cpu;
                gBestSpecified = true;
            } else if (option == "--variante" && !variantSpecified &&
                       (value == "base" || value == "ameliore")) {
                variant = value == "ameliore"
                    ? benchmarks::Variant::Improved
                    : benchmarks::Variant::Base;
                variantSpecified = true;
            } else {
                throw std::invalid_argument(
                    "Options attendues : --gbest cpu|gpu et "
                    "--variante base|ameliore.");
            }
        }

        const long long evaluations = 10000LL * dim;
        if (pop > evaluations) {
            throw std::invalid_argument(
                "La population doit etre inferieure ou egale au budget d'evaluations.");
        }
        if (evaluations / pop > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Le nombre d'iterations est trop grand.");
        }
        const std::string version =
            mode == GBestMode::Gpu ? "gpu-gbest-gpu" : "gpu-gbest-cpu";
        const char* algorithm =
            variant == benchmarks::Variant::Improved
                ? "pso-ameliore" : "pso";
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
            std::vector<float> positions;
            std::vector<float> velocities;
            std::vector<float> pBests;
            std::vector<float> pBestFitness;
            std::vector<float> gBest;
            float bestFitness = std::numeric_limits<float>::max();
            initializeSwarm(function, dim, pop, seed, positions, velocities,
                            pBests, pBestFitness, gBest, bestFitness);

            double elapsedSeconds = 0.0;
            cuda_pso(positions.data(), velocities.data(), pBests.data(),
                     pBestFitness.data(), gBest.data(), &bestFitness,
                     function, dim, pop, seed, mode, variant,
                     &elapsedSeconds);

            const float error =
                bestFitness - benchmarks::domain(function).bias;
            std::cout << algorithm << ',' << version << ',' << function << ',' << dim
                      << ',' << pop << ',' << run << ',' << seed << ','
                      << bestFitness << ',' << error << ',' << elapsedSeconds
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Erreur: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
