#pragma once

#ifdef __CUDACC__
#define PARAMS_HD __host__ __device__
#else
#define PARAMS_HD
#endif

namespace benchmarks {

enum class Variant {
    Base,      // Parametres historiques : inertie constante.
    Improved   // Inertie decroissante et vitesse bornee.
};

namespace pso_parameters {
constexpr float baseInertia = 0.5f;
constexpr float cognitive = 1.5f;
constexpr float social = 1.5f;
constexpr float inertiaMaximum = 0.9f;
constexpr float inertiaMinimum = 0.4f;
constexpr float maximumVelocityFactor = 0.1f;
}  // namespace pso_parameters

/**
 * Calcule le coefficient d'inertie utilise par PSO.
 *
 * La variante de base conserve une inertie fixe. La variante amelioree
 * diminue lineairement l'inertie pour favoriser l'exploration au debut puis
 * la convergence en fin de recherche.
 */
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
    Rand1,           // DE/rand/1.
    Best1,           // DE/best/1.
    CurrentToBest1   // DE/current-to-best/1.
};

constexpr Strategy defaultStrategy = Strategy::Rand1;
constexpr float defaultMutationFactor = 0.5f;
constexpr float defaultCrossoverRate = 0.9f;

}  // namespace de_parameters

}  // namespace benchmarks

#undef PARAMS_HD
