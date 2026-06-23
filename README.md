## Wave Equation Solver - 2D
**Author:** \
Camilo Andrés Martínez-Mejía (11022105) \
camiloandres1.martinez@mail.polimi.it

### Introduction
This project implements a parallel finite element solver, built on [deal.II](https://www.dealii.org/) and Trilinos, for the second-order wave equation `rho(x) * u_tt - div(c(x) * grad(u)) = f(x,t)` on a 2D domain, with support for (possibly time-dependent) Dirichlet boundary conditions. Time integration uses the theta-method (`theta=0` explicit, `theta=0.5` Crank-Nicolson, `theta=1` implicit Euler) on top of a Pk Lagrange finite element space. The core solver lives in [`src/WaveEquation.hpp`](src/WaveEquation.hpp)/[`src/WaveEquation.cpp`](src/WaveEquation.cpp) and is driven by four executables: a demo (`exercise-wave`) plus three numerical studies (`convergence-wave`, `energy-wave`, `dispersion-wave`) covering accuracy, energy stability, and phase error respectively, each documented in its own section below.

### Compiling
To build the executables, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
This produces four executables in `build`:
- `exercise-wave`: Gaussian-pulse demo on the unit square, homogeneous Dirichlet BCs with reflecting boundaries. VTU output is enabled by default, so it also doubles as a visualization demo (load the `.pvtu` series in ParaView for example).
- `convergence-wave`: manufactured-solution convergence study, spatial and temporal sweeps, P1 with theta=0.5.
- `energy-wave`: discrete energy conservation/instability study for the theta-method time-stepping scheme.
- `dispersion-wave`: numerical phase (dispersion) error study for a prescribed plane wave.

Each can be run from `build` with
```bash
$ ./executable-name
```

The study runner scripts below (`run_convergence_study.sh`, `run_energy_study.sh`, `run_dispersion_study.sh`), on the other hand, are meant to be run from the repository root, not from `build`: each one takes `build_dir` (default `build`) and `out_dir` as arguments relative to the current directory, builds the corresponding executable into `build_dir` if it isn't there yet, and creates `out_dir` (e.g. `results/convergence`) if it doesn't exist, writing the CSVs there. So a typical first run from a clean checkout is just:
```bash
$ mkdir build && cd build && cmake .. && make && cd ..
$ ./run_convergence_study.sh build results/convergence
$ ./run_energy_study.sh build results/energy
$ ./run_dispersion_study.sh build results/dispersion
```
which leaves `build/` (executables) and `results/` (CSVs and, after plotting, PNGs) side by side at the repository root.

### Convergence study (`convergence-wave`)
`run_convergence_study.sh` runs `convergence-wave` through both the spatial (h) and temporal (dt) sweeps, for two manufactured-solution cases: homogeneous (`omega = pi*sqrt(2)`, `f == 0`) and forced (`omega = 2*pi`, `f != 0`).
```bash
$ ./run_convergence_study.sh [build_dir] [out_dir]
```
For example:
```bash
$  ./run_convergence_study.sh build results/convergence
```

This writes `convergence_spatial_omega<value>.csv` and `convergence_temporal_omega<value>.csv` to `out_dir`. The plotting script, `plot_convergence.py`, lives in `scripts/` and reads from `results/convergence/` by default:
```bash
$ python3 scripts/plot_convergence.py                                       # reads and writes results/convergence/
$ python3 scripts/plot_convergence.py --data-dir results/convergence --out-dir scripts/out  # explicit dirs
```

`convergence-wave` also has an `example` mode: a single fixed-resolution run (`N=64`, `dt=1e-3`) of the manufactured solution with VTU output enabled, meant for visualization rather than error measurement:
```bash
$ ./convergence-wave example              # omega = 2*pi (forced) by default
$ ./convergence-wave example 4.442882938  # omega = pi*sqrt(2) (homogeneous, f=0)
```
This writes a `.pvtu`/`.vtu` series to `./results-example-omega<value>/`, which can be loaded in a visualization tool, for example ParaView.

### Energy study (`energy-wave`)
`energy-wave` logs the discrete energy `E^n = 0.5*(V^n)^T*M*V^n + 0.5*(U^n)^T*K*U^n` at every timestep for the homogeneous case (mesh `h=1/64`, `dt=0.02`, `T_final=35` unless noted), writing CSVs under `results/energy/` relative to the working directory:
```bash
$ ./energy-wave                                  # theta=0.5 (Crank-Nicolson) only -> energy_homogeneous.csv
$ ./energy-wave --with-theta1                    # also runs theta=1.0 (Implicit Euler) -> energy_theta1.csv
$ ./energy-wave --with-theta0                    # also runs theta=0 (explicit) instability check at dt=0.005/0.05
$ ./energy-wave --with-theta1 --with-theta0      # both flags can be combined
```
theta=0 is only conditionally stable, so those runs use a blow-up guard that stops early once `E^n` exceeds 100x its initial value.

`run_energy_study.sh` wraps `energy-wave` the same way `run_convergence_study.sh` wraps `convergence-wave`: it builds the binary if needed, then runs it once with both flags, writing all 4 CSVs straight into `out_dir`.
```bash
$ ./run_energy_study.sh [build_dir] [out_dir]
```
For example:
```bash
$ ./run_energy_study.sh build results/energy
```

The plotting scripts `plot_energy.py` and `plot_energy_theta0_instability.py` live in `scripts/` and read from `results/energy/` by default:
```bash
$ python3 scripts/plot_energy.py
$ python3 scripts/plot_energy_theta0_instability.py
$ python3 scripts/plot_energy.py --data-dir results/energy --out-dir scripts/out  # explicit dirs
```

### Dispersion study (`dispersion-wave`)
`dispersion-wave` measures numerical phase (dispersion) error for a prescribed plane wave. It performs three experiments:

- spatial sweep (Experiment 1)
- temporal sweep (Experiment 2)
- cancellation sweep (Experiment 3)

Each experiment may be executed for the axis-aligned and diagonal propagation directions where applicable. The study writes phase-error CSVs into the chosen output directory.

Run the helper script:
```bash
$ ./run_dispersion_study.sh [build_dir] [out_dir] [experiment] [np]
```
Examples:
```bash
$ ./run_dispersion_study.sh build results/dispersion                # all experiments, single rank
$ ./run_dispersion_study.sh build results/dispersion spatial 4      # spatial experiment, 4 MPI ranks
```

The plotting script `plot_dispersion.py` lives in `scripts/` and reads the five CSVs (`dispersion_spatial_axis.csv`, `dispersion_spatial_diagonal.csv`, `dispersion_temporal.csv`, `dispersion_cancellation_axis.csv`, `dispersion_cancellation_diagonal.csv`) from `results/dispersion/` by default, writing `dispersion_spatial.png`, `dispersion_temporal.png`, and `dispersion_cancellation.png` (plus least-squares log-log slopes printed to the console):
```bash
$ python3 scripts/plot_dispersion.py
$ python3 scripts/plot_dispersion.py --data-dir results/dispersion --out-dir scripts/out  # explicit dirs
```
