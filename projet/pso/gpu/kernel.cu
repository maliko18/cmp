#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cfloat>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernel.h"

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
    checkCuda(cudaDeviceSynchronize(), operation);
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : pointer_(nullptr) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("Taille de mémoire CUDA trop grande.");
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
    CudaEvent() : event_() {
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

__global__ void initializeRandomStates(curandState* states,
                                       std::size_t size,
                                       std::uint64_t seed) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        curand_init(seed, static_cast<unsigned long long>(index), 0,
                    &states[index]);
    }
}

__global__ void updateParticles(float* positions, float* velocities,
                                const float* pBests, const float* gBest,
                                curandState* states, std::size_t size,
                                int dim, float lowerBound, float upperBound,
                                float omega, float maximumVelocity,
                                bool limitVelocity) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }

    curandState state = states[index];
    const float r1 = curand_uniform(&state);
    const float r2 = curand_uniform(&state);
    states[index] = state;

    const std::size_t coordinate = index % static_cast<std::size_t>(dim);
    float velocity =
        omega * velocities[index] +
        benchmarks::pso_parameters::cognitive * r1 *
            (pBests[index] - positions[index]) +
        benchmarks::pso_parameters::social * r2 *
            (gBest[coordinate] - positions[index]);
    if (limitVelocity) {
        velocity = fminf(maximumVelocity,
                         fmaxf(-maximumVelocity, velocity));
    }
    velocities[index] = velocity;

    float position = positions[index] + velocities[index];
    position = fminf(upperBound, fmaxf(lowerBound, position));
    positions[index] = position;
}

__global__ void updatePersonalBests(const float* positions, float* pBests,
                                    float* pBestFitness, int function,
                                    int dim, int pop) {
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= pop) {
        return;
    }

    const std::size_t start =
        static_cast<std::size_t>(particle) * static_cast<std::size_t>(dim);
    const float* position = positions + start;
    const float currentFitness = benchmarks::fitness(position, dim, function);
    if (currentFitness < pBestFitness[particle]) {
        pBestFitness[particle] = currentFitness;
        for (int coordinate = 0; coordinate < dim; ++coordinate) {
            pBests[start + static_cast<std::size_t>(coordinate)] =
                position[coordinate];
        }
    }
}

__global__ void reduceGlobalBest(const float* pBests,
                                 const float* pBestFitness, float* gBest,
                                 float* bestFitness, int dim, int pop) {
    __shared__ float minimums[kThreads];
    __shared__ int indices[kThreads];

    const int thread = threadIdx.x;
    float minimum = FLT_MAX;
    int bestIndex = pop;
    for (int particle = thread; particle < pop; particle += blockDim.x) {
        const float fitness = pBestFitness[particle];
        if (fitness < minimum ||
            (fitness == minimum && particle < bestIndex)) {
            minimum = fitness;
            bestIndex = particle;
        }
    }

    minimums[thread] = minimum;
    indices[thread] = bestIndex;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            const float candidateFitness = minimums[thread + stride];
            const int candidateIndex = indices[thread + stride];
            if (candidateFitness < minimums[thread] ||
                (candidateFitness == minimums[thread] &&
                 candidateIndex < indices[thread])) {
                minimums[thread] = candidateFitness;
                indices[thread] = candidateIndex;
            }
        }
        __syncthreads();
    }

    if (thread == 0) {
        const int winner = indices[0];
        *bestFitness = minimums[0];
        for (int coordinate = 0; coordinate < dim; ++coordinate) {
            gBest[coordinate] =
                pBests[static_cast<std::size_t>(winner) *
                       static_cast<std::size_t>(dim) + coordinate];
        }
    }
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

int blockCount(std::size_t size) {
    if (size == 0 ||
        size > std::numeric_limits<std::size_t>::max() - (kThreads - 1)) {
        throw std::invalid_argument("Taille invalide pour le lancement CUDA.");
    }
    const std::size_t blocks = (size + kThreads - 1) / kThreads;
    if (blocks > static_cast<std::size_t>(2147483647)) {
        throw std::invalid_argument("La population est trop grande pour CUDA.");
    }
    return static_cast<int>(blocks);
}

}  // namespace

extern "C" void cuda_check_benchmarks(float* results) {
    if (results == nullptr) {
        throw std::invalid_argument("Le resultat du test CUDA est nul.");
    }

    DeviceBuffer<float> deviceResults(4);
    checkBenchmarkOptima<<<1, 4>>>(deviceResults.get());
    checkKernelLaunch("checkBenchmarkOptima");
    checkCuda(cudaMemcpy(results, deviceResults.get(), 4 * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy verification D2H");
}

extern "C" void cuda_pso(float* positions, float* velocities,
                         float* pBests, float* pBestFitness, float* gBest,
                         float* bestFitness, int function, int dim, int pop,
                         std::uint64_t seed, GBestMode gBestMode,
                         benchmarks::Variant variant,
                         double* elapsedSeconds) {
    if (!benchmarks::is_valid_function(function) || dim <= 0 || pop <= 0 ||
        positions == nullptr || velocities == nullptr || pBests == nullptr ||
        pBestFitness == nullptr || gBest == nullptr || bestFitness == nullptr ||
        elapsedSeconds == nullptr) {
        throw std::invalid_argument("Parametres invalides pour cuda_pso.");
    }

    const std::size_t size =
        static_cast<std::size_t>(dim) * static_cast<std::size_t>(pop);
    const long long evaluations = 10000LL * dim;
    if (evaluations / pop > 2147483647LL) {
        throw std::invalid_argument("Le nombre d'iterations depasse la limite CUDA.");
    }
    if (gBestMode != GBestMode::Cpu && gBestMode != GBestMode::Gpu) {
        throw std::invalid_argument("Mode gBest CUDA invalide.");
    }
    if (variant != benchmarks::Variant::Base &&
        variant != benchmarks::Variant::Improved) {
        throw std::invalid_argument("Variante PSO CUDA invalide.");
    }
    const int blocks = blockCount(size);
    const int particleBlocks = static_cast<int>(
        (static_cast<long long>(pop) + kThreads - 1) / kThreads);
    const benchmarks::Domain bounds = benchmarks::domain(function);
    const float maximumVelocity = benchmarks::maximum_velocity(function);

    DeviceBuffer<float> devicePositions(size);
    DeviceBuffer<float> deviceVelocities(size);
    DeviceBuffer<float> devicePBests(size);
    DeviceBuffer<float> devicePBestFitness(static_cast<std::size_t>(pop));
    DeviceBuffer<float> deviceGBest(static_cast<std::size_t>(dim));
    DeviceBuffer<float> deviceBestFitness(1);
    DeviceBuffer<curandState> randomStates(size);
    std::vector<float> hostPBests;
    if (gBestMode == GBestMode::Cpu) {
        hostPBests.resize(size);
    }

    checkCuda(cudaMemcpy(devicePositions.get(), positions,
                         size * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy positions H2D");
    checkCuda(cudaMemcpy(deviceVelocities.get(), velocities,
                         size * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy velocities H2D");
    checkCuda(cudaMemcpy(devicePBests.get(), pBests, size * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy pBests H2D");
    checkCuda(cudaMemcpy(devicePBestFitness.get(), pBestFitness,
                         static_cast<std::size_t>(pop) * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy fitness H2D");
    checkCuda(cudaMemcpy(deviceGBest.get(), gBest,
                         static_cast<std::size_t>(dim) * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy gBest H2D");
    checkCuda(cudaMemcpy(deviceBestFitness.get(), bestFitness,
                         sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy gBest fitness H2D");

    CudaEvent start;
    CudaEvent stop;
    checkCuda(cudaEventRecord(start.get()), "cudaEventRecord start");

    initializeRandomStates<<<blocks, kThreads>>>(
        randomStates.get(), size, seed);
    checkKernelLaunch("initialisation cuRAND");

    const int iterations = static_cast<int>(evaluations / pop);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const float omega =
            benchmarks::inertia(variant, iteration, iterations);
        updateParticles<<<blocks, kThreads>>>(
            devicePositions.get(), deviceVelocities.get(),
            devicePBests.get(), deviceGBest.get(), randomStates.get(), size,
            dim, bounds.minimum, bounds.maximum, omega, maximumVelocity,
            variant == benchmarks::Variant::Improved);
        checkCuda(cudaGetLastError(), "lancement updateParticles");

        updatePersonalBests<<<particleBlocks, kThreads>>>(
            devicePositions.get(), devicePBests.get(),
            devicePBestFitness.get(), function, dim, pop);
        checkKernelLaunch("updatePersonalBests");

        if (gBestMode == GBestMode::Cpu) {
            checkCuda(cudaMemcpy(hostPBests.data(), devicePBests.get(),
                                 size * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy pBests D2H");
            for (int particle = 0; particle < pop; ++particle) {
                const float* candidate =
                    hostPBests.data() +
                    static_cast<std::size_t>(particle) * dim;
                const float fitness =
                    benchmarks::fitness(candidate, dim, function);
                if (fitness < *bestFitness) {
                    *bestFitness = fitness;
                    for (int coordinate = 0; coordinate < dim; ++coordinate) {
                        gBest[coordinate] = candidate[coordinate];
                    }
                }
            }
            checkCuda(cudaMemcpy(deviceGBest.get(), gBest,
                                 static_cast<std::size_t>(dim) * sizeof(float),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy gBest H2D");
        } else {
            reduceGlobalBest<<<1, kThreads>>>(
                devicePBests.get(), devicePBestFitness.get(),
                deviceGBest.get(), deviceBestFitness.get(), dim, pop);
            checkKernelLaunch("reduceGlobalBest");
        }
    }

    if (gBestMode == GBestMode::Gpu) {
        checkCuda(cudaMemcpy(gBest, deviceGBest.get(),
                             static_cast<std::size_t>(dim) * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy gBest D2H");
        checkCuda(cudaMemcpy(bestFitness, deviceBestFitness.get(),
                             sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy gBest fitness D2H");
    }

    checkCuda(cudaEventRecord(stop.get()), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize");
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
              "cudaEventElapsedTime");
    *elapsedSeconds = static_cast<double>(milliseconds) / 1000.0;
}
