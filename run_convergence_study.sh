#!/usr/bin/env bash
# Runs the manufactured-solution convergence study (src/convergence-wave.cpp).
# Produces CSV files depending on which sweep(s) and omega case(s) ran:
#   convergence_spatial_omega<value>.csv
#   convergence_temporal_omega<value>.csv
#
# Notes on omega cases:
#  - omega = pi*sqrt(2)  -> f == 0 identically (homogeneous case)
#  - omega = 2*pi        -> f != 0             (forced case)
#
# Usage:
#   ./run_convergence_study.sh [build_dir] [out_dir] [sweep] [omega_case] [np]
#
#   sweep:      "all" (default), "spatial", or "temporal".
#   omega_case: "all" (default), "homogeneous", or "forced".
#   np:         number of MPI ranks (default 1). When >1, runs via
#               `mpirun -n np`.
#
# Note: convergence-wave runs with enable_output=false, so this script only
# writes the final-time error CSVs (no VTU/PVTU files).

set -euo pipefail

BUILD_DIR="${1:-build}"
OUT_DIR="${2:-convergence_results}"
SWEEP_ARG="${3:-all}"
OMEGA_CASE="${4:-all}"
NP="${5:-1}"
EXE="$BUILD_DIR/convergence-wave"

# If the binary hasn't been built yet, configure + build it now.
if [[ ! -x "$EXE" ]]; then
  echo "Building convergence-wave in $BUILD_DIR ..."
  cmake -S . -B "$BUILD_DIR" >/dev/null
  cmake --build "$BUILD_DIR" --target convergence-wave -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
fi

mkdir -p "$OUT_DIR"

# Resolve to an absolute path before run_case cd's into $OUT_DIR, otherwise
# a relative path like "build/convergence-wave" breaks once we're not in
# the repo root anymore.
EXE_ABS="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"

OMEGA_HOMOGENEOUS=$(python3 -c "import math; print(math.pi*math.sqrt(2))")  # f == 0
OMEGA_FORCED=$(python3 -c "import math; print(2*math.pi)")                  # f != 0

case "$SWEEP_ARG" in
  all)      sweeps=(spatial temporal) ;;
  spatial)  sweeps=(spatial) ;;
  temporal) sweeps=(temporal) ;;
  *) echo "Unknown sweep '$SWEEP_ARG'. Use 'all', 'spatial', or 'temporal'." >&2; exit 1 ;;
esac

case "$OMEGA_CASE" in
  all)         omegas=("$OMEGA_HOMOGENEOUS:homogeneous, f==0" "$OMEGA_FORCED:forced, f!=0") ;;
  homogeneous) omegas=("$OMEGA_HOMOGENEOUS:homogeneous, f==0") ;;
  forced)      omegas=("$OMEGA_FORCED:forced, f!=0") ;;
  *) echo "Unknown omega_case '$OMEGA_CASE'. Use 'all', 'homogeneous', or 'forced'." >&2; exit 1 ;;
esac

# cd into the output dir so the CSV the binary writes lands there.
run_case () {
  local sweep="$1"
  local omega="$2"
  local label="$3"

  echo "=== Running $sweep sweep, omega=$omega ($label), np=$NP ==="
  if [[ "$NP" -gt 1 ]]; then
    ( cd "$OUT_DIR" && mpirun -n "$NP" "$EXE_ABS" "$sweep" "$omega" )
  else
    ( cd "$OUT_DIR" && "$EXE_ABS" "$sweep" "$omega" )
  fi
}

for sweep in "${sweeps[@]}"; do
  for entry in "${omegas[@]}"; do
    omega="${entry%%:*}"
    label="${entry#*:}"
    run_case "$sweep" "$omega" "$label"
  done
done

echo
echo "Done. CSV files written to $OUT_DIR:"
ls "$OUT_DIR"/*.csv
