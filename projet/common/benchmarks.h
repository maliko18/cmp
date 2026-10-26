#pragma once

#include <cmath>

#include "params.h"

#ifdef __CUDACC__
#define BENCHMARK_HD __host__ __device__
#else
#define BENCHMARK_HD
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace benchmarks {

struct Domain {
    float minimum;
    float maximum;
    float bias;
};

BENCHMARK_HD inline bool is_valid_function(int function) {
    return function >= 1 && function <= 4;
}

BENCHMARK_HD inline Domain domain(int function) {
    switch (function) {
        case 1: return {-100.0f, 100.0f, -450.0f};
        case 2: return {-5.0f, 5.0f, -330.0f};
        case 3: return {-100.0f, 100.0f, 390.0f};
        case 4: return {-600.0f, 600.0f, -180.0f};
        default: return {0.0f, 0.0f, 0.0f};
    }
}

BENCHMARK_HD inline float maximum_velocity(int function) {
    const Domain bounds = domain(function);
    return pso_parameters::maximumVelocityFactor *
           (bounds.maximum - bounds.minimum);
}

BENCHMARK_HD inline float offset(int coordinate, int function) {
    const Domain bounds = domain(function);
    const int sample = ((coordinate % 101) * 37 + 17) % 101;
    const float centered = static_cast<float>(sample) / 100.0f - 0.5f;
    return centered * (bounds.maximum - bounds.minimum) * 0.2f;
}

BENCHMARK_HD inline float fitness(const float* x, int dim, int function) {
    const Domain bounds = domain(function);
    float result = 0.0f;

    switch (function) {
        case 1:
            for (int i = 0; i < dim; ++i) {
                const float z = x[i] - offset(i, function);
                result += z * z;
            }
            return result + bounds.bias;

        case 2:
            for (int i = 0; i < dim; ++i) {
                const float z = x[i] - offset(i, function);
                result += z * z - 10.0f *
                    cosf(2.0f * static_cast<float>(M_PI) * z) + 10.0f;
            }
            return result + bounds.bias;

        case 3:
            for (int i = 0; i < dim - 1; ++i) {
                const float z = x[i] - offset(i, function) + 1.0f;
                const float next = x[i + 1] - offset(i + 1, function) + 1.0f;
                const float difference = z * z - next;
                const float distance = z - 1.0f;
                result += 100.0f * difference * difference +
                          distance * distance;
            }
            return result + bounds.bias;

        case 4: {
            float product = 1.0f;
            for (int i = 0; i < dim; ++i) {
                const float z = x[i] - offset(i, function);
                result += z * z / 4000.0f;
                product *= cosf(z / sqrtf(static_cast<float>(i + 1)));
            }
            return result - product + 1.0f + bounds.bias;
        }

        default:
            return NAN;
    }
}

}  // namespace benchmarks

#undef BENCHMARK_HD
