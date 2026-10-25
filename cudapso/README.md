# PSO CUDA et séquentiel

Les quatre fonctions shiftées, leurs domaines, leurs biais et leur décalage
déterministe sont définis dans `benchmarks.h`, partagé par les deux programmes.
Cette en-tête contient également les paramètres communs aux variantes.
Le budget est de `10 000 x dim` évaluations, soit
`(10 000 x dim) / pop` itérations entières.

## Compilation

Sur une machine avec CUDA 12 et une T4 :

```sh
nvcc -O2 -arch=sm_75 -o pso_gpu main.cpp kernel.cu
```

Pour la version séquentielle :

```sh
g++ -O2 -std=c++17 -o pso_seq pso_seq.cpp
```

Ou utilisez `make`, `make gpu` et `make seq`.

## Exécution

```sh
./pso_gpu <fonction 1-4> <dim> <pop> <nb_runs> [--gbest cpu|gpu] [--variante base|ameliore]
./pso_seq <fonction 1-4> <dim> <pop> <nb_runs> [--variante base|ameliore]
```

Les deux exécutables écrivent l'en-tête CSV puis une ligne par exécution.
La variante par défaut est `base` et conserve l'inertie fixe `0.5`, sans
limite de vitesse. `--variante ameliore` active une inertie décroissante de
`0.9` à `0.4` et limite chaque composante de la vitesse à
`0.1 x (borne_max - borne_min)`. Ce facteur a été réduit de `0.2` à `0.1`
après les essais Sphere en dimension 100 : le facteur initial n'a pas convergé
de façon fiable en trois exécutions. Dans le CSV, la colonne `algo` vaut
`pso` ou `pso-ameliore`.

La version GPU utilise `--gbest cpu` par défaut; `--gbest gpu` effectue la
réduction du meilleur global sur le GPU. Les deux options peuvent être
fournies dans n'importe quel ordre.

Le test de cohérence des quatre optima s'exécute avec :

```sh
./pso_seq --check-benchmarks
./pso_gpu --check-benchmarks
```

Sur Google Colab, lancez `bash test_gpu_colab.sh` pour compiler et exécuter
les vérifications CPU et GPU avec les deux modes de meilleur global.
