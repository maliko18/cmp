#pragma once

#include <cstdint>

#include "benchmarks.h"
#include "params.h"

/**
 * Execute une campagne Differential Evolution sur le GPU.
 *
 * La population initiale et ses fitness sont fournies par le programme hote.
 * La fonction conserve les allocations et les generations sur le GPU, puis
 * recopie uniquement le meilleur individu et son score.
 *
 * @param initialPositions Population initiale aplatie, de taille dimension * population.
 * @param initialFitnesses Fitness correspondantes a la population initiale.
 * @param bestPosition Buffer hote recevant les coordonnees du meilleur individu.
 * @param bestFitness Buffer hote recevant la meilleure fitness finale.
 * @param function Numero de la fonction benchmark, entre 1 et 4.
 * @param dimension Nombre de coordonnees par individu.
 * @param population Nombre d'individus.
 * @param seed Graine de la generation aleatoire CUDA.
 * @param strategy Strategie de mutation DE utilisee.
 * @param mutationFactor Facteur F de mutation.
 * @param crossoverRate Taux CR du croisement binomial.
 * @param elapsedSeconds Pointeur recevant la duree des generations GPU.
 * @throws std::invalid_argument si les parametres ou pointeurs sont invalides.
 */
extern "C" void cuda_de(const float* initialPositions,
                        const float* initialFitnesses,
                        float* bestPosition, float* bestFitness,
                        int function, int dimension, int population,
                        std::uint64_t seed,
                        benchmarks::de_parameters::Strategy strategy,
                        float mutationFactor, float crossoverRate,
                        double* elapsedSeconds);

/**
 * Evalue les quatre benchmarks CUDA a leur optimum analytique.
 *
 * @param results Tableau de quatre floats rempli par la fonction.
 * @throws std::invalid_argument si results est nul.
 */
extern "C" void cuda_de_check_benchmarks(float* results);
