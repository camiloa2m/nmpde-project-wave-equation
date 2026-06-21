#!/usr/bin/env bash
# Runs the discrete energy conservation/instability study (src/energy-wave.cpp).
# Produces CSV files depending on which theta case(s) ran:
#   energy_homogeneous.csv   (theta=0.5, Crank-Nicolson -- always runs)
#   energy_theta1.csv        (theta=1.0, implicit Euler)
#   energy_theta0_dt005.csv / energy_theta0_dt05.csv    (theta=0, explicit)
#
# Usage:
#   ./run_energy_study.sh [build_dir] [out_dir] [theta_case] [np]
#
#   theta_case: "all" (default), "cn" (theta=0.5 only, no flags),
#               "theta1" (adds --with-theta1), "theta0" (adds --with-theta0).
#   np:         number of MPI ranks (default 1). When >1, runs via
#               `mpirun -n np`.
#
# enable_output=false, so this only writes the (t,energy) CSV logs, no VTU/PVTU.

set -euo pipefail

BUILD_DIR="${1:-build}"
OUT_DIR="${2:-energy_results}"
THETA_CASE="${3:-all}"
NP="${4:-1}"
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

case "$THETA_CASE" in
  all)    FLAGS=(--with-theta1 --with-theta0) ;;
  cn)     FLAGS=() ;;
  theta1) FLAGS=(--with-theta1) ;;
  theta0) FLAGS=(--with-theta0) ;;
  *) echo "Unknown theta_case '$THETA_CASE'. Use 'all', 'cn', 'theta1', or 'theta0'." >&2; exit 1 ;;
esac

if [[ "$NP" -gt 1 ]]; then
  RUN_CMD=(mpirun -n "$NP" "$EXE_ABS" "${FLAGS[@]}")
else
  RUN_CMD=("$EXE_ABS" "${FLAGS[@]}")
fi

# energy-wave hardcodes output to "./results/energy" and create_directories()
# Run from a real scratch dir, move CSVs after.
RUN_DIR="$(mktemp -d)"

echo "=== Running energy-wave ${FLAGS[*]:-(theta=0.5 only)}, np=$NP ==="
( cd "$RUN_DIR" && "${RUN_CMD[@]}" )
mv "$RUN_DIR"/results/energy/*.csv "$OUT_DIR_ABS"/
rm -rf "$RUN_DIR"

echo
echo "Done. CSV files written to $OUT_DIR:"
ls "$OUT_DIR"/*.csv
