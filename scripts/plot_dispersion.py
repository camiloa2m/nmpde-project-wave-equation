#!/usr/bin/env python3
"""Plot numerical dispersion studies for the wave equation solver
(P1, theta=0.5 Crank-Nicolson).

Reads the five CSVs produced by dispersion-wave from --data-dir and writes
dispersion_spatial.png, dispersion_temporal.png, dispersion_cancellation.png
to --out-dir. Also prints least-squares log-log slopes to the console.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "results", "dispersion")

EXCLUDED_DT_TEMPORAL = 0.00625


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def loglog_slope(x, y):
    """Least-squares slope of log(y) vs log(x)."""
    logx = np.log(x)
    logy = np.log(np.abs(y))
    slope, intercept = np.polyfit(logx, logy, 1)
    return slope, intercept


def reference_line(x_anchor, y_anchor, slope, x_range):
    """Reference slope line through (x_anchor, y_anchor) over x_range."""
    x = np.array(x_range)
    y = y_anchor * (x / x_anchor) ** slope
    return x, y


def plot_spatial(axis_csv, diagonal_csv, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))

    series = {
        "Axis-aligned (angle=0)": (axis_csv, "tab:blue", "o"),
        "Diagonal (angle=pi/4)": (diagonal_csv, "tab:red", "s"),
    }

    slopes = {}
    all_h = []
    axis_rows_sorted = None

    for label, (path, color, marker) in series.items():
        rows = sorted(read_csv(path), key=lambda r: r["h"])
        h = np.array([r["h"] for r in rows])
        err = np.abs(np.array([r["phase_error_radians"] for r in rows]))
        all_h.extend(h.tolist())

        ax.loglog(h, err, color=color, marker=marker, label=label)

        slope, _ = loglog_slope(h, err)
        slopes[label] = slope

        if label == "Axis-aligned (angle=0)":
            axis_rows_sorted = rows

    # O(h^2) reference line anchored at the axis direction's coarsest point
    # (axis_rows_sorted is ascending by h, so coarsest = last entry).
    h_min, h_max = min(all_h), max(all_h)
    anchor = axis_rows_sorted[-1]
    x_ref, y_ref = reference_line(
        anchor["h"], abs(anchor["phase_error_radians"]), 2, (h_min, h_max)
    )
    ax.loglog(x_ref, y_ref, color="0.7", linestyle="--", marker="", label=r"O($h^2$)")

    ax.set_xlabel("Mesh size h")
    ax.set_ylabel("Phase error magnitude (rad)")
    ax.set_title("Spatial numerical dispersion vs mesh size")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    fig.tight_layout()
    out_path = os.path.join(out_dir, "dispersion_spatial.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {out_path}")
    print("Figure A (spatial) least-squares log-log slopes:")
    for label, slope in slopes.items():
        print(f"  {label:25s} slope = {slope:.4f}")

    return slopes


def plot_temporal(temporal_csv, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))

    rows = sorted(read_csv(temporal_csv), key=lambda r: r["dt"])
    kept = [r for r in rows if r["dt"] != EXCLUDED_DT_TEMPORAL]
    dropped = [r for r in rows if r["dt"] == EXCLUDED_DT_TEMPORAL]

    dt_kept = np.array([r["dt"] for r in kept])
    err_kept = np.abs(np.array([r["phase_error_radians"] for r in kept]))

    ax.loglog(dt_kept, err_kept, color="tab:blue", marker="o", label="Phase error")

    if dropped:
        dt_drop = np.array([r["dt"] for r in dropped])
        err_drop = np.abs(np.array([r["phase_error_radians"] for r in dropped]))
        ax.loglog(
            dt_drop,
            err_drop,
            color="tab:blue",
            marker="o",
            markerfacecolor="none",
            linestyle="none",
            label="Near noise floor (excluded from fit)",
        )
        ax.annotate(
            "sign flips here\n(near noise floor)",
            xy=(dt_drop[0], err_drop[0]),
            xytext=(dt_drop[0] * 1.3, err_drop[0] * 3),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
        )

    slope, _ = loglog_slope(dt_kept, err_kept)

    # O(dt^2) reference line anchored at the largest dt point.
    anchor = max(kept, key=lambda r: r["dt"])
    all_dt = [r["dt"] for r in rows]
    dt_min, dt_max = min(all_dt), max(all_dt)
    x_ref, y_ref = reference_line(
        anchor["dt"], abs(anchor["phase_error_radians"]), 2, (dt_min, dt_max)
    )
    ax.loglog(x_ref, y_ref, color="0.7", linestyle="--", marker="", label=r"O($dt^2$)")

    ax.set_xlabel("Time step dt")
    ax.set_ylabel("Phase error magnitude (rad)")
    ax.set_title("Temporal numerical dispersion vs time step")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "dispersion_temporal.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {out_path}")
    print(
        f"Figure B (temporal) least-squares log-log slope "
        f"(dt in {[r['dt'] for r in kept]}): slope = {slope:.4f}"
    )
    for r in dropped:
        print(
            f"  excluded: dt={r['dt']} -> phase_error={r['phase_error_radians']:.5e} "
            "(near noise floor, sign flip)"
        )

    return slope


def plot_cancellation(axis_csv, diagonal_csv, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    panels = [
        (axes[0], axis_csv, "axis direction", 1.0, "tab:blue"),
        (axes[1], diagonal_csv, "diagonal direction", 1.0 / np.sqrt(2), "tab:red"),
    ]

    for ax, path, panel_label, theory_x, color in panels:
        rows = sorted(read_csv(path), key=lambda r: r["dt_over_h"])
        x = np.array([r["dt_over_h"] for r in rows])
        y = np.array([r["phase_error_radians"] for r in rows])

        ax.plot(x, y, color=color, marker="o", linestyle="-")
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
        ax.axvline(
            theory_x,
            color="0.3",
            linestyle="--",
            linewidth=1,
            label=f"theoretical cancellation\n(dt/h = {theory_x:.3f})",
        )

        ax.set_xlabel("dt / h")
        ax.set_ylabel("Phase error (rad, signed)")
        ax.set_title(f"Cancellation effect: {panel_label}")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "dispersion_cancellation.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing the dispersion CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write the PNGs to (default: same as --data-dir)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)

    plot_spatial(
        os.path.join(data_dir, "dispersion_spatial_axis.csv"),
        os.path.join(data_dir, "dispersion_spatial_diagonal.csv"),
        out_dir,
    )
    plot_temporal(os.path.join(data_dir, "dispersion_temporal.csv"), out_dir)
    plot_cancellation(
        os.path.join(data_dir, "dispersion_cancellation_axis.csv"),
        os.path.join(data_dir, "dispersion_cancellation_diagonal.csv"),
        out_dir,
    )


if __name__ == "__main__":
    main()
