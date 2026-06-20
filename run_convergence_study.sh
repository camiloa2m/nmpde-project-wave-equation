#!/usr/bin/env bash
# Runs the manufactured-solution convergence study (src/convergence-wave.cpp)
# across both the spatial (h) and temporal (dt) sweeps, for both omega cases:
#   - omega = pi*sqrt(2)  -> f == 0 identically (homogeneous case)
#   - omega = 2*pi        -> f != 0             (forced case)
#
# Produces 4 CSV files in $OUT_DIR:
#   convergence_spatial_omega<value>.csv
#   convergence_temporal_omega<value>.csv
#
# Usage:
#   ./run_convergence_study.sh [build_dir] [out_dir]
#
# NOTE: convergence-wave.cpp runs WaveEquation with enable_output=false, so no
# VTU/PVTU snapshots are written during these sweeps -- only the final-time
# L2/H1 errors are computed and recorded in the CSVs.

set -euo pipefail

BUILD_DIR="${1:-build}"
OUT_DIR="${2:-convergence_results}"
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

# cd into the output dir so the CSV the binary writes lands there.
run_case () {
  local sweep="$1"
  local omega="$2"
  local label="$3"

  echo "=== Running $sweep sweep, omega=$omega ($label) ==="
  ( cd "$OUT_DIR" && "$EXE_ABS" "$sweep" "$omega" )
}

for sweep in spatial temporal; do
  run_case "$sweep" "$OMEGA_HOMOGENEOUS" "homogeneous, f==0"
  run_case "$sweep" "$OMEGA_FORCED"      "forced, f!=0"
done

echo
echo "Done. CSV files written to $OUT_DIR:"
ls "$OUT_DIR"/*.csv
