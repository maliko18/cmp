#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <limits>

#include "kernel.h"

namespace {

inline void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Philox RNG constants (4x32-bit)
__device__ __constant__ uint32_t philox_k[2] = {0x9E3779B9, 0xBB67AE85};

__device__ uint32_t mulhi(uint32_t a, uint32_t b) {
    return __umulhi(a, b);
}

__device__ void philox_round(uint32_t& x0, uint32_t& x1, uint32_t& k0, uint32_t& k1) {
    uint64_t t0 = (uint64_t)x0 * 0xD2511F53;
    uint64_t t1 = (uint64_t)x1 * 0xCD9E8D57;
    x0 = (uint32_t)t1 ^ k0;
    x1 = (uint32_t)t0 ^ k1;
    k0 += philox_k[0];
    k1 += philox_k[1];
}

__device__ float philox_rand(uint64_t seed, uint64_t sequence, uint32_t& state_idx) {
    // Philox 4x32: we use 2 rounds for speed
    uint32_t c0 = (uint32_t)(seed & 0xFFFFFFFF);
    uint32_t c1 = (uint32_t)(seed >> 32);
    uint32_t c2 = (uint32_t)(sequence & 0xFFFFFFFF);
    uint32_t c3 = (uint32_t)(sequence >> 32);

    uint32_t k0 = 0, k1 = 0;

    // Key schedule
    k0 += c0; k1 += c1;
    philox_round(c2, c3, k0, k1);
    philox_round(c0, c1, k0, k1);

    // Generate 4 outputs, return one based on state_idx
    uint32_t out = c0 ^ c2;
    state_idx = (state_idx + 1) & 3;
    return (float)out * (1.0f / 4294967296.0f);
}

// Simpler: XORSHIFT-based RNG per thread
__device__ uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ float device_rand(uint32_t& state) {
    return (float)xorshift32(state) * (1.0f / 4294967296.0f);
}

__device__ int device_rand_int(uint32_t& state, int max) {
    return (int)((float)xorshift32(state) * (1.0f / 4294967296.0f) * max);
}

__device__ float deviceFitness(const float* position, int dimensions, int objectiveId) {
    const float pi = 3.14159265358979323846f;
    float result = 0.0f;

    switch (objectiveId) {
        case 0: {  // Levy
            const float firstY = 1.0f + (position[0] - 1.0f) / 4.0f;
            result = powf(sinf(pi * firstY), 2.0f);
            for (int i = 0; i + 1 < dimensions; ++i) {
                const float y = 1.0f + (position[i] - 1.0f) / 4.0f;
                const float nextY = 1.0f + (position[i + 1] - 1.0f) / 4.0f;
                result += powf(y - 1.0f, 2.0f) *
                          (1.0f + 10.0f * powf(sinf(pi * nextY), 2.0f));
            }
            const float lastY = 1.0f + (position[dimensions - 1] - 1.0f) / 4.0f;
            return result + powf(lastY - 1.0f, 2.0f) *
                             (1.0f + powf(sinf(2.0f * pi * lastY), 2.0f));
        }
        case 1: {  // Rastrigin: min = 0 at origin
            for (int i = 0; i < dimensions; ++i) {
                result += position[i] * position[i] -
                          10.0f * cosf(2.0f * pi * position[i]) + 10.0f;
            }
            return result;
        }
        case 2: {  // Rosenbrock: min = 0 at (1,1,...,1)
            for (int i = 0; i + 1 < dimensions; ++i) {
                const float value = position[i];
                const float nextValue = position[i + 1];
                result += 100.0f * powf(nextValue - value * value, 2.0f) +
                          powf(value - 1.0f, 2.0f);
            }
            return result;
        }
        case 3: {  // Griewank: min = 0 at origin
            float product = 1.0f;
            for (int i = 0; i < dimensions; ++i) {
                result += position[i] * position[i] / 4000.0f;
                product *= cosf(position[i] / sqrtf(static_cast<float>(i + 1)));
            }
            return result - product + 1.0f;
        }
        case 4: {  // Sphere: min = 0 at origin
            for (int i = 0; i < dimensions; ++i) {
                result += position[i] * position[i];
            }
            return result;
        }
        default:
            return CUDART_INF_F;
    }
}

__global__ void deGeneration(float* population,
                             float* fitness,
                             int* firstIndices,
                             int* secondIndices,
                             int* thirdIndices,
                             int* forcedCoordinates,
                             int dimensions,
                             int populationSize,
                             int objectiveId,
                             float lowerBound,
                             float upperBound,
                             float mutationFactor,
                             float crossoverRate,
                             uint64_t rngSeed,
                             uint64_t rngSequence,
                             float* bestFitness) {
    const int target = blockIdx.x * blockDim.x + threadIdx.x;
    if (target >= populationSize) {
        return;
    }

    // Per-thread RNG state
    uint32_t rngState = (uint32_t)(rngSeed + target + rngSequence * populationSize);

    const int first = firstIndices[target];
    const int second = secondIndices[target];
    const int third = thirdIndices[target];
    const int forcedCoordinate = forcedCoordinates[target];
    const int targetOffset = target * dimensions;
    const int firstOffset = first * dimensions;
    const int secondOffset = second * dimensions;
    const int thirdOffset = third * dimensions;

    // Trial vector in registers/local memory (not shared)
    float trialFitness = 0.0f;

    for (int d = 0; d < dimensions; ++d) {
        const float mutant = population[firstOffset + d] +
            mutationFactor * (population[secondOffset + d] - population[thirdOffset + d]);

        const float crossoverRand = device_rand(rngState);
        const float value = (crossoverRand <= crossoverRate || d == forcedCoordinate)
            ? mutant
            : population[targetOffset + d];

        // Clamp to bounds
        const float clamped = fminf(upperBound, fmaxf(lowerBound, value));

        // Compute fitness incrementally (store trial in place, compute fitness at end)
        // We need the full trial vector for fitness, so store temporarily
        // Using a small local array in registers if dimensions <= 32, otherwise local memory
        // For simplicity, we'll recompute - but better: store trial in a register array
    }

    // Need to reconstruct trial for fitness - store it
    // Since dimensions can be large, use a small stack array approach
    // Actually, better: compute fitness by re-evaluating with same RNG decisions
    // But that's complex. Instead, let's use shared memory for trial only, sized to blockDim.x * maxDim
    // But we removed shared memory...
    // Compromise: use local array for trial (compiler will use registers if small, local mem if large)

    // Recompute trial and fitness
    // (We could store trial in shared memory per thread block if needed, but for now recompute)
    // Actually let's just recompute - it's not that expensive for the fitness function

    // Better approach: Store trial in local array
    float trialVal;
    for (int d = 0; d < dimensions; ++d) {
        const float mutant = population[firstOffset + d] +
            mutationFactor * (population[secondOffset + d] - population[thirdOffset + d]);
        const float crossoverRand = device_rand(rngState);
        trialVal = (crossoverRand <= crossoverRate || d == forcedCoordinate)
            ? mutant
            : population[targetOffset + d];
        // We need all dimensions for fitness...
        // This is the problem - fitness needs full vector
    }

    // Solution: Use shared memory but only for trial vectors (blockDim.x * dimensions)
    // But we want to avoid shared memory scaling with dimensions...
    // For now, use shared memory but limit block size to keep it small
}

__global__ void deGenerationV2(float* population,
                               float* fitness,
                               int* firstIndices,
                               int* secondIndices,
                               int* thirdIndices,
                               int* forcedCoordinates,
                               int dimensions,
                               int populationSize,
                               int objectiveId,
                               float lowerBound,
                               float upperBound,
                               float mutationFactor,
                               float crossoverRate,
                               uint64_t rngSeed,
                               uint64_t rngSequence,
                               float* bestFitness) {
    extern __shared__ float sharedTrial[];
    const int tid = threadIdx.x;
    const int target = blockIdx.x * blockDim.x + tid;
    if (target >= populationSize) {
        return;
    }

    uint32_t rngState = (uint32_t)(rngSeed + target + rngSequence * populationSize);

    const int first = firstIndices[target];
    const int second = secondIndices[target];
    const int third = thirdIndices[target];
    const int forcedCoordinate = forcedCoordinates[target];
    const int targetOffset = target * dimensions;
    const int firstOffset = first * dimensions;
    const int secondOffset = second * dimensions;
    const int thirdOffset = third * dimensions;

    float* myTrial = sharedTrial + tid * dimensions;

    for (int d = 0; d < dimensions; ++d) {
        const float mutant = population[firstOffset + d] +
            mutationFactor * (population[secondOffset + d] - population[thirdOffset + d]);
        const float crossoverRand = device_rand(rngState);
        const float value = (crossoverRand <= crossoverRate || d == forcedCoordinate)
            ? mutant
            : population[targetOffset + d];
        myTrial[d] = fminf(upperBound, fmaxf(lowerBound, value));
    }

    const float trialFitness = deviceFitness(myTrial, dimensions, objectiveId);

    if (trialFitness <= fitness[target]) {
        for (int d = 0; d < dimensions; ++d) {
            population[targetOffset + d] = myTrial[d];
        }
        fitness[target] = trialFitness;
    }

    // Update global best (atomic min)
    atomicMin((unsigned int*)bestFitness, __float_as_uint(trialFitness));
}

// Kernel to generate random indices on device
__global__ void generateIndices(int* firstIndices,
                                 int* secondIndices,
                                 int* thirdIndices,
                                 int* forcedCoordinates,
                                 int populationSize,
                                 int dimensions,
                                 uint64_t rngSeed,
                                 uint64_t rngSequence) {
    const int target = blockIdx.x * blockDim.x + threadIdx.x;
    if (target >= populationSize) return;

    uint32_t rngState = (uint32_t)(rngSeed + target + rngSequence * populationSize);

    int first, second, third;
    do { first = device_rand_int(rngState, populationSize); } while (first == target);
    do { second = device_rand_int(rngState, populationSize); } while (second == target || second == first);
    do { third = device_rand_int(rngState, populationSize); } while (third == target || third == first || third == second);

    firstIndices[target] = first;
    secondIndices[target] = second;
    thirdIndices[target] = third;
    forcedCoordinates[target] = device_rand_int(rngState, dimensions);
}

}  // namespace

extern "C" void cuda_de(float* population,
                         float* fitness,
                         const DeConfig& config) {
    const std::size_t populationValues = config.populationSize * config.dimensions;
    float* devicePopulation = nullptr;
    float* deviceFitness = nullptr;
    int* deviceFirst = nullptr;
    int* deviceSecond = nullptr;
    int* deviceThird = nullptr;
    int* deviceForced = nullptr;
    float* deviceBestFitness = nullptr;

    try {
        checkCuda(cudaMalloc(&devicePopulation, populationValues * sizeof(float)), "cudaMalloc population");
        checkCuda(cudaMalloc(&deviceFitness, config.populationSize * sizeof(float)), "cudaMalloc fitness");
        checkCuda(cudaMalloc(&deviceFirst, config.populationSize * sizeof(int)), "cudaMalloc first");
        checkCuda(cudaMalloc(&deviceSecond, config.populationSize * sizeof(int)), "cudaMalloc second");
        checkCuda(cudaMalloc(&deviceThird, config.populationSize * sizeof(int)), "cudaMalloc third");
        checkCuda(cudaMalloc(&deviceForced, config.populationSize * sizeof(int)), "cudaMalloc forced");
        checkCuda(cudaMalloc(&deviceBestFitness, sizeof(float)), "cudaMalloc bestFitness");

        checkCuda(cudaMemcpy(devicePopulation, population, populationValues * sizeof(float), cudaMemcpyHostToDevice), "copy population H2D");
        checkCuda(cudaMemcpy(deviceFitness, fitness, config.populationSize * sizeof(float), cudaMemcpyHostToDevice), "copy fitness H2D");

        // Initialize best fitness on device
        float hostBestFitness = std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < config.populationSize; ++i) {
            if (fitness[i] < hostBestFitness) hostBestFitness = fitness[i];
        }
        checkCuda(cudaMemcpy(deviceBestFitness, &hostBestFitness, sizeof(float), cudaMemcpyHostToDevice), "copy bestFitness H2D");

        // Dynamic block size to fit shared memory
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, 0), "getDeviceProperties");

        int threads = 256;
        std::size_t sharedBytes = static_cast<std::size_t>(threads) * config.dimensions * sizeof(float);

        // Reduce block size if shared memory exceeds limit
        while (sharedBytes > prop.sharedMemPerBlock && threads > 32) {
            threads /= 2;
            sharedBytes = static_cast<std::size_t>(threads) * config.dimensions * sizeof(float);
        }

        if (sharedBytes > prop.sharedMemPerBlock) {
            throw std::runtime_error("Shared memory required (" + std::to_string(sharedBytes) +
                ") exceeds device limit (" + std::to_string(prop.sharedMemPerBlock) +
                ") even with minimal block size. Dimensions too large for this implementation.");
        }

        const int blocks = static_cast<int>((config.populationSize + threads - 1) / threads);

        uint64_t rngSeed = config.seed;
        uint64_t rngSequence = 0;

        for (std::size_t evaluations = config.populationSize;
             evaluations < config.maxEvaluations && hostBestFitness > config.targetFitness;
             evaluations += config.populationSize, ++rngSequence) {

            // Generate indices on device
            generateIndices<<<blocks, threads>>>(deviceFirst, deviceSecond, deviceThird, deviceForced,
                                                  static_cast<int>(config.populationSize),
                                                  static_cast<int>(config.dimensions),
                                                  rngSeed, rngSequence);
            checkCuda(cudaGetLastError(), "generateIndices launch");

            // Run DE generation
            deGenerationV2<<<blocks, threads, sharedBytes>>>(
                devicePopulation, deviceFitness, deviceFirst, deviceSecond, deviceThird,
                deviceForced, static_cast<int>(config.dimensions),
                static_cast<int>(config.populationSize), config.objectiveId,
                config.lowerBound, config.upperBound, config.mutationFactor,
                config.crossoverRate, rngSeed, rngSequence, deviceBestFitness);
            checkCuda(cudaGetLastError(), "deGenerationV2 launch");

            // Copy back best fitness to check termination
            checkCuda(cudaMemcpy(&hostBestFitness, deviceBestFitness, sizeof(float), cudaMemcpyDeviceToHost), "copy bestFitness D2H");
        }

        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        checkCuda(cudaMemcpy(population, devicePopulation, populationValues * sizeof(float), cudaMemcpyDeviceToHost), "copy population D2H");
        checkCuda(cudaMemcpy(fitness, deviceFitness, config.populationSize * sizeof(float), cudaMemcpyDeviceToHost), "copy fitness D2H");

    } catch (...) {
        if (devicePopulation) cudaFree(devicePopulation);
        if (deviceFitness) cudaFree(deviceFitness);
        if (deviceFirst) cudaFree(deviceFirst);
        if (deviceSecond) cudaFree(deviceSecond);
        if (deviceThird) cudaFree(deviceThird);
        if (deviceForced) cudaFree(deviceForced);
        if (deviceBestFitness) cudaFree(deviceBestFitness);
        throw;
    }

    checkCuda(cudaFree(devicePopulation), "free population");
    checkCuda(cudaFree(deviceFitness), "free fitness");
    checkCuda(cudaFree(deviceFirst), "free first");
    checkCuda(cudaFree(deviceSecond), "free second");
    checkCuda(cudaFree(deviceThird), "free third");
    checkCuda(cudaFree(deviceForced), "free forced");
    checkCuda(cudaFree(deviceBestFitness), "free bestFitness");
}