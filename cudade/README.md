# DE CUDA parallel

This is a separate CUDA implementation of Differential Evolution. The original PSO folders are unchanged.

The GPU uses one thread per target individual. Each generation performs:

```text
V = Xr1 + F * (Xr2 - Xr3)
U[d] = V[d] with probability CR, otherwise X[d]
replace X by U when f(U) <= f(X)
```

The default strategy is `DE/rand/1/bin` with `F=0.5` and `CR=0.9`.

## Google Colab

Upload this folder as `cudade.zip`, select a T4 GPU runtime, then run:

```python
!unzip -oq /content/cudade.zip -d /content
%cd /content/cudade
!nvcc -O2 -arch=sm_75 -o de_cuda main.cpp kernel.cu
!./de_cuda
```

Arguments:

```text
de_cuda [dimension] [population] [function_id] [max_evaluations] [seed] [target_fitness]
```

Example required-project configuration:

```python
!./de_cuda 10 50 1 100000 42
# With early termination at fitness 1e-6:
!./de_cuda 10 50 1 100000 42 1e-6
```

Function IDs: `0=Levy`, `1=Rastrigin`, `2=Rosenbrock`, `3=Griewank`, `4=Sphere`.

The current CUDA implementation prepares random indices on the host and executes mutation, crossover, objective evaluation and selection on the GPU. This is a working parallel baseline. Before the final report, compare it with the exact strategy and parameter values required by `ImprovedDE.pdf`.
