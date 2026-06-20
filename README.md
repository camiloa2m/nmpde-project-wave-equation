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
Three executables are produced in `build`:
- `exercise-wave` — Gaussian-pulse demo on the unit square (homogeneous Dirichlet BCs, reflecting boundaries).
- `convergence-wave` — manufactured-solution spatial/temporal convergence study (Q1, theta=0.5).
- `energy-wave` — discrete energy conservation/instability study for the theta-method time-stepping scheme.

Each can be run from `build` with
```bash
$ ./executable-name
```

### Convergence study (`convergence-wave`)
`run_convergence_study.sh` runs `convergence-wave` across both spatial (h) and temporal (dt) sweeps, for the homogeneous (`omega = pi*sqrt(2)`, `f == 0`) and forced (`omega = 2*pi`, `f != 0`) manufactured-solution cases:
```bash
$ ./run_convergence_study.sh [build_dir] [out_dir]
```
Produces `convergence_spatial_omega<value>.csv` / `convergence_temporal_omega<value>.csv` in `out_dir`. Results and the `plot_convergence.py` plotting script live in `results/convergence/`.

### Energy study (`energy-wave`)
`energy-wave` logs the discrete energy `E^n = 0.5*(V^n)^T*M*V^n + 0.5*(U^n)^T*K*U^n` at every timestep for the homogeneous case (mesh `h=1/64`, `dt=0.02`, `T_final=35` unless noted), writing CSVs under `results/energy/`:
```bash
$ ./energy-wave                 # theta=0.5 (Crank-Nicolson) only -> energy_homogeneous.csv
$ ./energy-wave --with-theta1   # also runs theta=1.0 (Implicit Euler) -> energy_theta1.csv
$ ./energy-wave --with-theta0   # also runs theta=0 (explicit) instability check at dt=0.005/0.05
```
The theta=0 runs use a blow-up guard that stops early once `E^n` exceeds 100x its initial value (theta=0 is only conditionally stable). Plotting scripts (`plot_energy.py`, `plot_energy_theta0_instability.py`) and resulting PNGs live alongside the CSVs in `results/energy/`.