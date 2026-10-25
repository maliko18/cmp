# DE sequential CPU

This folder is a separate CPU reference implementation of Differential Evolution. The original PSO folders are unchanged.

The implementation uses the classical `DE/rand/1/bin` strategy:

```text
mutant = Xr1 + F * (Xr2 - Xr3)
trial[d] = mutant[d] with probability CR, otherwise target[d]
replace target if trial is not worse
```

Defaults are dimension `3`, population `512`, Levy (`function_id=0`) and `30000 * population` evaluations. The default parameters are `F=0.5` and `CR=0.9`.

Build on Windows from a Developer PowerShell:

```powershell
cl /EHsc /O2 /std:c++14 main.cpp /Fe:de_sequential.exe
.\de_sequential.exe
```

Or run another configuration:

```powershell
.\de_sequential.exe 10 50 1 100000 42
# With early termination at fitness 1e-6:
.\de_sequential.exe 10 50 1 100000 42 1e-6
```

Function IDs: `0=Levy`, `1=Rastrigin`, `2=Rosenbrock`, `3=Griewank`, `4=Sphere`.

This is the CPU reference version. The CUDA DE version should reproduce the same mutation, crossover, selection, bounds and evaluation budget before comparing GPU and CPU results. If `ImprovedDE.pdf` specifies another DE strategy, update `F`, `CR` and the mutation rule to match that paper exactly.
