#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "de_kernel.h"

namespace {

constexpr int kThreads = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void checkKernelLaunch(const char* operation) {
    checkCuda(cudaGetLastError(), operation);
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : pointer_(nullptr) {
        if (count == 0 ||
            count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("Taille de memoire CUDA invalide.");
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                             count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    T* get() const { return pointer_; }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

private:
    T* pointer_;
};

class CudaEvent {
public:
    CudaEvent() : event_(nullptr) {
        checkCuda(cudaEventCreate(&event_), "cudaEventCreate");
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    cudaEvent_t get() const { return event_; }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

private:
    cudaEvent_t event_;
};

__global__ void initializeRandomStates(curandState* states, int population,
                                       std::uint64_t seed) {
    const int individual = blockIdx.x * blockDim.x + threadIdx.x;
    if (individual < population) {
        curand_init(seed, static_cast<unsigned long long>(individual), 0,
                    &states[individual]);
    }
}

__device__ int drawIndexExcluding(curandState* state, int population,
                                  int first, int second, int third,
                                  int excludedCount) {
    int excluded[3] = {first, second, third};
    for (int left = 0; left < excludedCount; ++left) {
        for (int right = left + 1; right < excludedCount; ++right) {
            if (excluded[right] < excluded[left]) {
                const int temporary = excluded[left];
                excluded[left] = excluded[right];
                excluded[right] = temporary;
            }
        }
    }
    const unsigned int choices =
        static_cast<unsigned int>(population - excludedCount);
    // Le rang tire est remappe en sautant les indices exclus.
    int candidate = static_cast<int>(curand(state) % choices);
    for (int index = 0; index < excludedCount; ++index) {
        if (candidate >= excluded[index]) {
            ++candidate;
        }
    }
    return candidate;
}

__global__ void reduceBest(const float* positions, const float* fitnesses,
                           float* bestPosition, float* bestFitness,
                           int dimension, int population) {
    __shared__ float partialFitness[kThreads];
    __shared__ int partialIndex[kThreads];

    const int thread = threadIdx.x;
    float minimum = FLT_MAX;
    int bestIndex = population;
    for (int individual = thread; individual < population;
         individual += blockDim.x) {
        const float candidate = fitnesses[individual];
        if (candidate < minimum ||
            (candidate == minimum && individual < bestIndex)) {
            minimum = candidate;
            bestIndex = individual;
        }
    }
    partialFitness[thread] = minimum;
    partialIndex[thread] = bestIndex;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            const float candidateFitness = partialFitness[thread + stride];
            const int candidateIndex = partialIndex[thread + stride];
            if (candidateFitness < partialFitness[thread] ||
                (candidateFitness == partialFitness[thread] &&
                 candidateIndex < partialIndex[thread])) {
                partialFitness[thread] = candidateFitness;
                partialIndex[thread] = candidateIndex;
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        const int winner = partialIndex[0];
        *bestFitness = partialFitness[0];
        for (int coordinate = 0; coordinate < dimension; ++coordinate) {
            bestPosition[coordinate] =
                positions[static_cast<std::size_t>(winner) * dimension +
                          coordinate];
        }
    }
}

__global__ void buildTrials(
    const float* currentPositions, const float* bestPosition,
    float* trialPositions, float* trialFitnesses, curandState* randomStates,
    int function, int dimension, int population, float lowerBound,
    float upperBound, float mutationFactor, float crossoverRate,
    benchmarks::de_parameters::Strategy strategy) {
    const int individual = blockIdx.x * blockDim.x + threadIdx.x;
    if (individual >= population) {
        return;
    }

    curandState state = randomStates[individual];
    const int r1 = drawIndexExcluding(&state, population, individual, -1, -1, 1);
    const int r2 =
        drawIndexExcluding(&state, population, individual, r1, -1, 2);
    const int r3 =
        drawIndexExcluding(&state, population, individual, r1, r2, 3);
    const int forcedCoordinate =
        static_cast<int>(curand(&state) %
                         static_cast<unsigned int>(dimension));
    const std::size_t individualOffset =
        static_cast<std::size_t>(individual) * dimension;
    const std::size_t r1Offset = static_cast<std::size_t>(r1) * dimension;
    const std::size_t r2Offset = static_cast<std::size_t>(r2) * dimension;
    const std::size_t r3Offset = static_cast<std::size_t>(r3) * dimension;

    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        float mutant = 0.0f;
        switch (strategy) {
            case benchmarks::de_parameters::Strategy::Rand1:
                mutant = currentPositions[r1Offset + coordinate] +
                    mutationFactor *
                        (currentPositions[r2Offset + coordinate] -
                         currentPositions[r3Offset + coordinate]);
                break;
            case benchmarks::de_parameters::Strategy::Best1:
                mutant = bestPosition[coordinate] +
                    mutationFactor *
                        (currentPositions[r1Offset + coordinate] -
                         currentPositions[r2Offset + coordinate]);
                break;
            case benchmarks::de_parameters::Strategy::CurrentToBest1:
                mutant =
                    currentPositions[individualOffset + coordinate] +
                    mutationFactor *
                        (bestPosition[coordinate] -
                         currentPositions[individualOffset + coordinate]) +
                    mutationFactor *
                        (currentPositions[r1Offset + coordinate] -
                         currentPositions[r2Offset + coordinate]);
                break;
        }

        const float parent = currentPositions[individualOffset + coordinate];
        float trial = coordinate == forcedCoordinate ||
                              curand_uniform(&state) <= crossoverRate
            ? mutant : parent;
        trial = fminf(upperBound, fmaxf(lowerBound, trial));
        trialPositions[individualOffset + coordinate] = trial;
    }
    trialFitnesses[individual] = benchmarks::fitness(
        trialPositions + individualOffset, dimension, function);
    randomStates[individual] = state;
}

__global__ void selectTrials(
    const float* currentPositions, const float* currentFitnesses,
    const float* trialPositions, const float* trialFitnesses,
    float* nextPositions, float* nextFitnesses,
    int dimension, int population) {
    const int individual = blockIdx.x * blockDim.x + threadIdx.x;
    if (individual >= population) {
        return;
    }

    const std::size_t offset =
        static_cast<std::size_t>(individual) * dimension;
    const bool accept =
        trialFitnesses[individual] <= currentFitnesses[individual];
    const float* selected = accept
        ? trialPositions + offset : currentPositions + offset;
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        nextPositions[offset + coordinate] = selected[coordinate];
    }
    nextFitnesses[individual] =
        accept ? trialFitnesses[individual] : currentFitnesses[individual];
}

__global__ void checkBenchmarkOptima(float* results) {
    const int function = threadIdx.x + 1;
    if (function > 4) {
        return;
    }
    float point[10];
    for (int coordinate = 0; coordinate < 10; ++coordinate) {
        point[coordinate] = benchmarks::offset(coordinate, function);
    }
    results[function - 1] = benchmarks::fitness(point, 10, function);
}

int blockCount(int count) {
    if (count <= 0 ||
        count > std::numeric_limits<int>::max() - (kThreads - 1)) {
        throw std::invalid_argument("Taille invalide pour un lancement CUDA.");
    }
    return (count + kThreads - 1) / kThreads;
}

}  // namespace

extern "C" void cuda_de_check_benchmarks(float* results) {
    if (results == nullptr) {
        throw std::invalid_argument("Le resultat du test CUDA est nul.");
    }
    DeviceBuffer<float> deviceResults(4);
    checkBenchmarkOptima<<<1, 4>>>(deviceResults.get());
    checkKernelLaunch("checkBenchmarkOptima");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize benchmarks");
    checkCuda(cudaMemcpy(results, deviceResults.get(), 4 * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy verification D2H");
}

extern "C" void cuda_de(
    const float* initialPositions, const float* initialFitnesses,
    float* bestPosition, float* bestFitness, int function, int dimension,
    int population, std::uint64_t seed,
    benchmarks::de_parameters::Strategy strategy, float mutationFactor,
    float crossoverRate, double* elapsedSeconds) {
    if (!benchmarks::is_valid_function(function) || dimension <= 0 ||
        population < 4 || initialPositions == nullptr ||
        initialFitnesses == nullptr || bestPosition == nullptr ||
        bestFitness == nullptr || elapsedSeconds == nullptr ||
        !std::isfinite(mutationFactor) || mutationFactor < 0.0f ||
        mutationFactor > 2.0f || !std::isfinite(crossoverRate) ||
        crossoverRate < 0.0f || crossoverRate > 1.0f) {
        throw std::invalid_argument("Parametres invalides pour cuda_de.");
    }
    if (strategy != benchmarks::de_parameters::Strategy::Rand1 &&
        strategy != benchmarks::de_parameters::Strategy::Best1 &&
        strategy != benchmarks::de_parameters::Strategy::CurrentToBest1) {
        throw std::invalid_argument("Strategie DE CUDA invalide.");
    }

    const long long budget = 10000LL * dimension;
    if (population > budget || budget / population > INT_MAX) {
        throw std::invalid_argument("Budget ou nombre de generations invalide.");
    }
    if (static_cast<std::size_t>(dimension) >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(population)) {
        throw std::invalid_argument("La taille de population est excessive.");
    }
    const std::size_t size =
        static_cast<std::size_t>(dimension) * population;
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::length_error("Taille de memoire CUDA excessive.");
    }

    const int blocks = blockCount(population);
    const int totalGenerations =
        static_cast<int>(budget / population);
    const benchmarks::Domain bounds = benchmarks::domain(function);

    // Les deux tampons courant/suivant et le tampon d'essais restent distincts.
    DeviceBuffer<float> currentPositionsA(size);
    DeviceBuffer<float> currentPositionsB(size);
    DeviceBuffer<float> trialPositions(size);
    DeviceBuffer<float> currentFitnessA(static_cast<std::size_t>(population));
    DeviceBuffer<float> currentFitnessB(static_cast<std::size_t>(population));
    DeviceBuffer<float> trialFitnesses(static_cast<std::size_t>(population));
    DeviceBuffer<float> deviceBestPosition(
        static_cast<std::size_t>(dimension));
    DeviceBuffer<float> deviceBestFitness(1);
    DeviceBuffer<curandState> randomStates(
        static_cast<std::size_t>(population));

    checkCuda(cudaMemcpy(currentPositionsA.get(), initialPositions,
                         size * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy population H2D");
    checkCuda(cudaMemcpy(currentFitnessA.get(), initialFitnesses,
                         static_cast<std::size_t>(population) * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy fitness H2D");
    initializeRandomStates<<<blocks, kThreads>>>(
        randomStates.get(), population, seed);
    checkKernelLaunch("initialisation cuRAND");
    reduceBest<<<1, kThreads>>>(
        currentPositionsA.get(), currentFitnessA.get(),
        deviceBestPosition.get(), deviceBestFitness.get(), dimension,
        population);
    checkKernelLaunch("reduction du meilleur initial");

    CudaEvent start;
    CudaEvent stop;
    checkCuda(cudaEventRecord(start.get()), "cudaEventRecord start");

    float* currentPositions = currentPositionsA.get();
    float* nextPositions = currentPositionsB.get();
    float* currentFitnesses = currentFitnessA.get();
    float* nextFitnesses = currentFitnessB.get();
    for (int generation = 1; generation < totalGenerations; ++generation) {
        buildTrials<<<blocks, kThreads>>>(
            currentPositions, deviceBestPosition.get(), trialPositions.get(),
            trialFitnesses.get(), randomStates.get(), function, dimension,
            population, bounds.minimum, bounds.maximum, mutationFactor,
            crossoverRate, strategy);
        checkKernelLaunch("construction et evaluation des essais");

        selectTrials<<<blocks, kThreads>>>(
            currentPositions, currentFitnesses, trialPositions.get(),
            trialFitnesses.get(), nextPositions, nextFitnesses, dimension,
            population);
        checkKernelLaunch("selection DE");

        float* previousPositions = currentPositions;
        currentPositions = nextPositions;
        nextPositions = previousPositions;
        float* previousFitnesses = currentFitnesses;
        currentFitnesses = nextFitnesses;
        nextFitnesses = previousFitnesses;

        reduceBest<<<1, kThreads>>>(
            currentPositions, currentFitnesses, deviceBestPosition.get(),
            deviceBestFitness.get(), dimension, population);
        checkKernelLaunch("reduction du meilleur courant");
    }

    checkCuda(cudaEventRecord(stop.get()), "cudaEventRecord stop");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize DE");
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
              "cudaEventElapsedTime");
    *elapsedSeconds = static_cast<double>(milliseconds) / 1000.0;

    checkCuda(cudaMemcpy(bestPosition, deviceBestPosition.get(),
                         static_cast<std::size_t>(dimension) * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy meilleur D2H");
    checkCuda(cudaMemcpy(bestFitness, deviceBestFitness.get(), sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy fitness du meilleur D2H");
}
