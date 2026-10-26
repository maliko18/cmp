#pragma once

#include <cstdint>

#include "benchmarks.h"

enum class GBestMode : int {
    Cpu = 0,
    Gpu = 1
};

/** Retourne un nombre pseudo-aleatoire uniforme dans [low, high]. */
float getRandom(float low, float high);

/** Retourne un nombre pseudo-aleatoire uniforme dans [0, 1]. */
float getRandomClamped();

/** Evalue un benchmark depuis le code hote en reutilisant l'implementation commune. */
float host_fitness_function(const float* x, int dim, int function);

/**
 * Execute le PSO CUDA.
 *
 * @param positions Positions courantes aplaties des particules.
 * @param velocities Vitesses courantes aplaties.
 * @param pBests Meilleures positions individuelles connues.
 * @param pBestFitness Fitness associee a chaque meilleure position.
 * @param gBest Meilleure position globale.
 * @param bestFitness Fitness de la meilleure position globale.
 * @param function Numero du benchmark, entre 1 et 4.
 * @param dim Dimension du probleme.
 * @param pop Nombre de particules.
 * @param seed Graine des etats cuRAND.
 * @param gBestMode Mode de calcul du meilleur global, CPU ou GPU.
 * @param variant Variante PSO de base ou amelioree.
 * @param elapsedSeconds Duree mesuree des iterations GPU.
 */
extern "C" void cuda_pso(float* positions, float* velocities,
                         float* pBests, float* pBestFitness, float* gBest,
                         float* bestFitness, int function, int dim, int pop,
                         std::uint64_t seed, GBestMode gBestMode,
                         benchmarks::Variant variant,
                         double* elapsedSeconds);

/** Verifie sur GPU les valeurs des quatre benchmarks a leur optimum. */
extern "C" void cuda_check_benchmarks(float* results);
