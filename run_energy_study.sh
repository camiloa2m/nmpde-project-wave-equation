#!/usr/bin/env bash
# Runs the discrete energy conservation/instability study (src/energy-wave.cpp)
# for all three theta cases. Produces 4 CSV files:
#   energy_homogeneous.csv   (theta=0.5, Crank-Nicolson)
#   energy_theta1.csv        (theta=1.0, implicit Euler)
#   energy_theta0_dt005.csv / energy_theta0_dt05.csv    (theta=0, explicit)
#
# Usage:
#   ./run_energy_study.sh [build_dir] [out_dir]
#
# enable_output=false, so this only writes the (t,energy) CSV logs, no VTU/PVTU.

set -euo pipefail

BUILD_DIR="${1:-build}"
OUT_DIR="${2:-energy_results}"
EXE="$BUILD_DIR/energy-wave"

# Build if missing.
if [[ ! -x "$EXE" ]]; then
  echo "Building energy-wave in $BUILD_DIR ..."
  cmake -S . -B "$BUILD_DIR" >/dev/null
  cmake --build "$BUILD_DIR" --target energy-wave -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
fi

mkdir -p "$OUT_DIR"

# Need absolute paths before we cd elsewhere, or "build/energy-wave" breaks.
EXE_ABS="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"
OUT_DIR_ABS="$(cd "$OUT_DIR" && pwd)"

# energy-wave hardcodes output to "./results/energy" and create_directories()
# Run from a real scratch dir, move CSVs after.
RUN_DIR="$(mktemp -d)"

echo "=== Running energy-wave --with-theta1 --with-theta0 ==="
( cd "$RUN_DIR" && "$EXE_ABS" --with-theta1 --with-theta0 )
mv "$RUN_DIR"/results/energy/*.csv "$OUT_DIR_ABS"/
rm -rf "$RUN_DIR"

echo
echo "Done. CSV files written to $OUT_DIR:"
ls "$OUT_DIR"/*.csv
