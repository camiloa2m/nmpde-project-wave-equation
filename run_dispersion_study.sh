#!/usr/bin/env bash
# Runs the plane-wave numerical dispersion study (src/dispersion-wave.cpp).
# Produces CSV files depending on which experiment(s) ran:
#   dispersion_spatial_axis.csv   / dispersion_spatial_diagonal.csv
#   dispersion_temporal.csv
#   dispersion_cancellation_axis.csv / dispersion_cancellation_diagonal.csv
#
# Usage:
#   ./run_dispersion_study.sh [build_dir] [out_dir] [experiment] [np]
#
#   experiment: "all" (default), "spatial", "temporal", or "cancellation"
#               (forwarded as argv[1] to dispersion-wave).
#   np:         number of MPI ranks (default 1). When >1, runs via
#               `mpirun -n np`.
#
# Note: dispersion-wave runs with enable_output=false, so this script only
# writes the phase-error CSVs (no VTU/PVTU files).

set -euo pipefail

BUILD_DIR="${1:-build}"
OUT_DIR="${2:-results/dispersion}"
EXPERIMENT="${3:-all}"
NP="${4:-1}"
EXE="$BUILD_DIR/dispersion-wave"

# If the binary hasn't been built yet, configure + build it now.
if [[ ! -x "$EXE" ]]; then
  echo "Building dispersion-wave in $BUILD_DIR ..."
  cmake -S . -B "$BUILD_DIR" >/dev/null
  cmake --build "$BUILD_DIR" --target dispersion-wave -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
fi

mkdir -p "$OUT_DIR"

# Resolve to an absolute path before cd'ing into $OUT_DIR, otherwise a
# relative path like "build/dispersion-wave" breaks once we're not in the
# repo root anymore.
EXE_ABS="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"

if [[ "$NP" -gt 1 ]]; then
  RUN_CMD=(mpirun -n "$NP" "$EXE_ABS" "$EXPERIMENT")
else
  RUN_CMD=("$EXE_ABS" "$EXPERIMENT")
fi

echo "=== Running dispersion-wave, experiment=$EXPERIMENT, np=$NP ==="

# cd into the output dir so the CSVs the binary writes land there.
( cd "$OUT_DIR" && "${RUN_CMD[@]}" )

echo
echo "Done. CSV files written to $OUT_DIR:"
ls "$OUT_DIR"/*.csv
