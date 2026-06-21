### Organizing the source code
Please place all your sources into the `src` folder.

Binary files must not be uploaded to the repository (including executables).

Mesh files should not be uploaded to the repository. If applicable, upload `gmsh` scripts with suitable instructions to generate the meshes (and ideally a Makefile that runs those instructions). If not applicable, consider uploading the meshes to a different file sharing service, and providing a download link as part of the building and running instructions.

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
This produces three executables in `build`:
- `exercise-wave`: Gaussian-pulse demo on the unit square, homogeneous Dirichlet BCs with reflecting boundaries.
- `convergence-wave`: manufactured-solution convergence study, spatial and temporal sweeps, P1 with theta=0.5.
- `energy-wave`: discrete energy conservation/instability study for the theta-method time-stepping scheme.

Each can be run from `build` with
```bash
$ ./executable-name
```

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
`dispersion-wave` measures numerical phase (dispersion) error for a
prescribed plane wave. It performs three experiments:

- spatial sweep (Experiment 1)
- temporal sweep (Experiment 2)
- cancellation sweep (Experiment 3)

Each experiment may be executed for the axis-aligned and diagonal
propagation directions where applicable. The study writes phase-error CSVs
into the chosen output directory.

Run the helper script:
```bash
$ ./run_dispersion_study.sh [build_dir] [out_dir] [experiment] [np]
```
Examples:
```bash
$ ./run_dispersion_study.sh build results/dispersion                # all experiments, single rank
$ ./run_dispersion_study.sh build results/dispersion spatial 4      # spatial experiment, 4 MPI ranks
```