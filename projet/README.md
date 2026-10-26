# PSO et Differential Evolution (DE) sur CUDA

## Arborescence

```text
projet/
  common/       benchmarks.h et params.h communs CPU/GPU
  pso/gpu/      version CUDA du PSO
  pso/seq/      version séquentielle du PSO
  de/gpu/       version CUDA du DE
  de/seq/       version séquentielle du DE
  scripts/      tests Colab et création d'une archive
  results/      CSV de campagne (vide au départ)
  article/      article IEEE et bibliographie
  _archive/     anciens fichiers conservés hors du build
  Makefile
  README.md
```

Les quatre fonctions shiftées sont définies une seule fois dans
`common/benchmarks.h` : Sphere (biais -450, domaine [-100,100]), Rastrigin
(-330, [-5,5]), Rosenbrock (390, [-100,100]) et Griewank (-180, [-600,600]).
Leur décalage déterministe place l'optimum en `x=o`. Les paramètres PSO
historiques et les valeurs par défaut du DE sont dans `common/params.h`.

## Compilation

Avec `g++`, GNU Make et CUDA (T4 / architecture `sm_75`) :

```sh
make                 # pso_gpu, pso_seq, de_gpu, de_seq
make seq             # versions CPU seulement
make gpu             # versions CUDA
make CUDA_ARCH=sm_75
```

Les exécutables sont créés à la racine du projet. `make check-seq` compile et
vérifie les deux versions séquentielles sans nécessiter `nvcc`. Pour compiler
manuellement, les sources utilisent `-Icommon`; par exemple :

```sh
g++ -O2 -std=c++17 -Icommon -o pso_seq pso/seq/pso_seq.cpp
g++ -O2 -std=c++17 -Icommon -o de_seq de/seq/de_seq.cpp
nvcc -O2 -std=c++17 -arch=sm_75 -Icommon -Ipso/gpu \
  -o pso_gpu pso/gpu/main.cpp pso/gpu/kernel.cu pso/gpu/kernel.cpp
nvcc -O2 -std=c++17 -arch=sm_75 -Icommon -Ide/gpu \
  -o de_gpu de/gpu/de_main.cpp de/gpu/de_kernel.cu
```

## PSO

```sh
./pso_gpu <fonction 1-4> <dim> <pop> <nb_runs> [--gbest cpu|gpu] [--variante base|ameliore]
./pso_seq <fonction 1-4> <dim> <pop> <nb_runs> [--variante base|ameliore]
```

Le PSO conserve ses algorithmes et paramètres existants : variante `base` par
défaut avec `omega=0.5`, `c1=c2=1.5`; la variante `ameliore` utilise une
inertie décroissante de 0.9 à 0.4 et une vitesse maximale de 0.1 fois la
largeur du domaine. Le mode `--gbest cpu` est le défaut GPU; `--gbest gpu`
calcule le meilleur global par réduction CUDA. Son budget historique et son
comportement ne sont pas modifiés par cette réorganisation.
Conformément à son code existant, le PSO évalue d'abord la population initiale
puis effectue `floor((10 000 x dim) / pop)` itérations; ces évaluations
initiales s'ajoutent donc aux itérations. Le DE, lui, compte son initialisation
dans son budget strict et effectue une génération d'essais de moins.

## Differential Evolution

```sh
./de_gpu <fonction 1-4> <dim> <pop> <nb_runs> [--strategie rand1|best1|currenttobest1] [--F 0.5] [--CR 0.9]
./de_seq <fonction 1-4> <dim> <pop> <nb_runs> [--strategie rand1|best1|currenttobest1] [--F 0.5] [--CR 0.9]
```

La stratégie par défaut est `rand1` / `rand/1/bin`; `F=0.5` et `CR=0.9`
sont définis dans `common/params.h`. Les stratégies `best1` et
`currenttobest1` utilisent respectivement les mutations `best/1` et
`current-to-best/1`, avec croisement binomial. Les indices de mutation sont
distincts et exclus de l'individu cible. Les paramètres acceptés sont
`0 <= F <= 2` et `0 <= CR <= 1`; la population doit contenir au moins quatre
individus.

Le budget est `10 000 x dim` évaluations. L'évaluation initiale de la
population compte dans ce budget : le nombre de générations complètes est
`floor((10 000 x dim) / pop)`, première population incluse, donc le DE exécute
une génération initiale puis une génération d'essais de moins. Si le budget
n'est pas divisible par `pop`, le reliquat inférieur à une population n'est
pas utilisé. La règle est identique en séquentiel et sur GPU. L'initialisation
et les transferts sont hors du temps CSV; le temps DE mesuré couvre uniquement
les générations d'essais (chrono CPU ou événements CUDA).

Sur GPU, un thread traite un individu et boucle sur ses coordonnées. Les
essais sont d'abord construits et évalués depuis la population courante, puis
un kernel de sélection écrit la population suivante dans un tampon distinct;
les tampons courant/suivant sont échangés avant la réduction du meilleur.
Ainsi aucun kernel ne lit et n'écrit simultanément la même population. La
boucle a un nombre fixe de générations, sans transfert GPU-CPU ni
synchronisation intermédiaire.

Chaque exécutable accepte `--check-benchmarks`. Les programmes écrivent une
ligne d'en-tête puis le CSV `algo,version,fonction,dim,pop,run,seed,minimum,erreur_vs_bias,temps_s`.

## Vérifications sur Colab

Dans un environnement Colab avec une GPU T4, lancez depuis ce dossier :

```sh
bash scripts/test_gpu_colab.sh
bash scripts/test_de_colab.sh
```

`scripts/make_zip.sh` crée `../projet-colab.zip`, avec les sources et scripts
actifs (sans `_archive/`, exécutables ni CSV générés), prêt à être transféré
sur Colab. Les résultats de campagne doivent être sauvegardés dans `results/`.

## Campagne experimentale complete

Le protocole complet couvre `Dim={10,50,100}`, `Pop={50,100,500}`, les
4 fonctions et 10 runs par configuration. Chaque CSV est ecrit dans
`results/`, un fichier par algorithme/version, avec son en-tete unique.
Le lanceur ajoute seulement les numeros de run manquants et peut reprendre
apres interruption sans recalculer les lignes deja presentes. Ne supprimez
pas les CSV si vous souhaitez reprendre une campagne. Le budget de evaluations
est celui implemente dans les programmes; voir la distinction entre budget
historique PSO et budget DE ci-dessus.

### Sur PC (Windows)

Compiler les executables sequentiels depuis une invite de commandes
Developper Visual Studio, a la racine de `projet/` :

```powershell
cl /nologo /O2 /EHsc /std:c++17 /Icommon /Fe:pso_seq.exe pso/seq/pso_seq.cpp
cl /nologo /O2 /EHsc /std:c++17 /Icommon /Fe:de_seq.exe de/seq/de_seq.cpp
```

Python 3 et `pandas`, `scipy`, `matplotlib` sont necessaires. Lancer d'abord
la campagne rapide (3 runs, dimensions 10 et 50) :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\run_campaign_seq.ps1 --rapide
```

Puis, pour completer les configurations et passer a 10 runs :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\run_campaign_seq.ps1
```

Le meme lanceur peut etre appele avec Python directement :
`py -3 scripts/run_campaign.py --device seq --rapide`. Les options de mode
rapide sont `--rapide`; sans cette option la campagne complete est lancee.
Si la politique d'execution locale interdit les fichiers PS, utiliser
directement la commande Python precedente au lieu du wrapper PS.
Les scripts affichent une estimation du temps restant avant le premier calcul
et la progression de chaque configuration. Le temps total depend du CPU,
des versions de compilateur et des CSV deja presents.

Selon les temps de reference fournis, une campagne complete sans resultats
anterieurs est estimee a environ **36,5 min en sequentiel**. Le GPU est estime
a environ **23,3 min**, dont le DE GPU est une approximation conservative
faute de temps de reference CUDA DE. Ces estimations sont indicatives; la
campagne `--rapide` a pris environ 3,7 min sur la machine Windows de
developpement.

### Sur Colab / GPU

Copier ou extraire le projet dans Colab, activer une GPU T4, puis compiler et
lancer :

```sh
cd projet
make all
bash scripts/run_campaign_gpu.sh --rapide
bash scripts/run_campaign_gpu.sh
```

La campagne GPU inclut PSO ameliore et DE rand1 en priorite, les autres
variantes ensuite, et une comparaison PSO `--gbest cpu` limitee a
`Dim=100`, `Pop=100,500`. Le calcul des durees GPU dans l'estimation est
explicitement approximatif : PSO gBest GPU se base sur la mesure preliminaire
de 0,34 s/run, PSO gBest CPU sur 1,0 s/run, et faute de mesure DE GPU celle-ci
utilise l'equivalence CPU de 2,5 s/run. Ces valeurs ne sont pas des resultats
de campagne.

Pour rapatrier tous les CSV de Colab en une archive :

```sh
zip -r results_colab.zip results -i 'results/*.csv'
```

Dans une cellule Colab :

```python
from google.colab import files
files.download("results_colab.zip")
```

Extraire les CSV recus dans `projet/results/` sur le PC. L'analyseur signalera
les fichiers absents et calculera ce qui est disponible :

```powershell
py -3 scripts/analyse.py
```

Il ecrit les tableaux CSV et LaTeX dans `article/tables/`, les figures PDF
dans `article/figures/`, les resume descriptifs, speedups et tests non
parametriques. Les comparaisons utilisent le rank-sum de Mann--Whitney pour
deux groupes independants et Kruskal--Wallis entre les strategies DE. `=`
indique que les echantillons sont identiques et qu'un test n'est pas defini;
les p-values n'impliquent pas de correction pour comparaisons multiples.

Pour verifier l'analyse sans confondre valeurs fictives et experiences :

```powershell
py -3 scripts/generate_test_data.py
py -3 scripts/analyse.py --input-dir results_test `
  --table-dir results_test/analysis/tables `
  --figure-dir results_test/analysis/figures
```

Les CSV de `results_test/` sont artificiels, sont marques par
`FAKE_DATA_NOTICE.txt` et ne doivent jamais etre copies dans `results/` ni
cites dans l'article.
