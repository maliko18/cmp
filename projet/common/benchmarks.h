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
    float minimum;  // Borne inferieure du domaine de recherche.
    float maximum;  // Borne superieure du domaine de recherche.
    float bias;     // Minimum theorique ajoute a la fonction.
};

/** Indique si l'identifiant correspond a l'un des quatre benchmarks. */
BENCHMARK_HD inline bool is_valid_function(int function) {
    return function >= 1 && function <= 4;
}

/**
 * Retourne le domaine de recherche et le biais de la fonction.
 *
 * Le biais est la valeur theorique minimale, utilisee pour calculer
 * l'erreur finale dans les fichiers CSV.
 */
BENCHMARK_HD inline Domain domain(int function) {
    switch (function) {
        case 1: return {-100.0f, 100.0f, -450.0f};
        case 2: return {-5.0f, 5.0f, -330.0f};
        case 3: return {-100.0f, 100.0f, 390.0f};
        case 4: return {-600.0f, 600.0f, -180.0f};
        default: return {0.0f, 0.0f, 0.0f};
    }
}

/** Retourne la vitesse maximale PSO proportionnelle a la largeur du domaine. */
BENCHMARK_HD inline float maximum_velocity(int function) {
    const Domain bounds = domain(function);
    return pso_parameters::maximumVelocityFactor *
           (bounds.maximum - bounds.minimum);
}

/**
 * Retourne le decalage deterministe de l'optimum pour une coordonnee.
 *
 * Le decalage rend les benchmarks non triviaux tout en conservant un optimum
 * connu, ce qui facilite la validation des algorithmes CPU et GPU.
 */
BENCHMARK_HD inline float offset(int coordinate, int function) {
    const Domain bounds = domain(function);
    const int sample = ((coordinate % 101) * 37 + 17) % 101;
    const float centered = static_cast<float>(sample) / 100.0f - 0.5f;
    return centered * (bounds.maximum - bounds.minimum) * 0.2f;
}

/**
 * Evalue une fonction benchmark shiftée.
 *
 * @param x Vecteur candidat de dimension dim.
 * @param dim Nombre de coordonnees a lire dans x.
 * @param function Identifiant du benchmark entre 1 et 4.
 * @return Fitness a minimiser, incluant le biais de la fonction.
 */
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
