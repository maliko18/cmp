#pragma once

#ifdef __CUDACC__
#define PARAMS_HD __host__ __device__
#else
#define PARAMS_HD
#endif

namespace benchmarks {

enum class Variant {
    Base,
    Improved
};

namespace pso_parameters {
constexpr float baseInertia = 0.5f;
constexpr float cognitive = 1.5f;
constexpr float social = 1.5f;
constexpr float inertiaMaximum = 0.9f;
constexpr float inertiaMinimum = 0.4f;
constexpr float maximumVelocityFactor = 0.1f;
}  // namespace pso_parameters

PARAMS_HD inline float inertia(Variant variant, int iteration,
                               int iterations) {
    if (variant == Variant::Base || iterations <= 1) {
        return pso_parameters::baseInertia;
    }
    const float progress = static_cast<float>(iteration) /
                           static_cast<float>(iterations - 1);
    return pso_parameters::inertiaMaximum -
        (pso_parameters::inertiaMaximum - pso_parameters::inertiaMinimum) *
            progress;
}

namespace de_parameters {

enum class Strategy {
    Rand1,
    Best1,
    CurrentToBest1
};

constexpr Strategy defaultStrategy = Strategy::Rand1;
constexpr float defaultMutationFactor = 0.5f;
constexpr float defaultCrossoverRate = 0.9f;

}  // namespace de_parameters

}  // namespace benchmarks

#undef PARAMS_HD
