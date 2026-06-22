#!/usr/bin/env python3
"""Plot spatial and temporal convergence studies for the wave
equation MMS solver (P1, theta=0.5 Crank-Nicolson).

Reads the four CSVs produced by convergence-wave (case,refinement,N,h,dt,
L2_error,H1_error) from --data-dir and writes convergence_spatial.png /
convergence_temporal.png to --out-dir. Also prints least-squares log-log
slopes to the console.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "results", "convergence")

OMEGA_HOMOGENEOUS = "omega4.44288"  # omega = pi*sqrt(2); homogeneous (no forcing)
OMEGA_FORCED = "omega6.28319"  # omega = 2*pi; forced problem (nonzero source)

EXCLUDED_DT = 0.0025

# Fixed mesh size h used for the temporal convergence study. Chosen to
# expose time-discretization error while avoiding excessive spatial noise.
TEMPORAL_STUDY_H = 0.0078125

OMEGA_LABEL = {
    OMEGA_HOMOGENEOUS: r"homogeneous ($\omega=\pi\sqrt{2}$)",
    OMEGA_FORCED: r"forced ($\omega=2\pi$)",
}
OMEGA_LINESTYLE = {
    OMEGA_HOMOGENEOUS: "-",
    OMEGA_FORCED: "--",
}
OMEGA_MARKER = {
    OMEGA_HOMOGENEOUS: "o",
    OMEGA_FORCED: "s",
}
ERROR_COLOR = {
    "L2_error": "tab:blue",
    "H1_error": "tab:red",
}
ERROR_LABEL = {
    "L2_error": "L2",
    "H1_error": "H1",
}


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "case": row["case"],
                    "refinement": int(row["refinement"]),
                    "N": int(row["N"]),
                    "h": float(row["h"]),
                    "dt": float(row["dt"]),
                    "L2_error": float(row["L2_error"]),
                    "H1_error": float(row["H1_error"]),
                }
            )
    return rows


def loglog_slope(x, y):
    """Least-squares slope of log(y) vs log(x)."""
    logx = np.log(x)
    logy = np.log(y)
    slope, intercept = np.polyfit(logx, logy, 1)
    return slope, intercept


def reference_line(x_anchor, y_anchor, slope, x_range):
    """Reference slope line through (x_anchor, y_anchor) over x_range."""
    x = np.array(x_range)
    y = y_anchor * (x / x_anchor) ** slope
    return x, y


def spatial_floor_at(spatial_csvs, h):
    """Return L2 error at mesh size `h` for each omega case.

    The returned values serve as the spatial error floor: any temporal
    study point with L2 below this floor can no longer be attributed
    solely to time discretization and should be treated with caution.
    """
    floors = {}
    for omega_key, path in spatial_csvs.items():
        rows = read_csv(path)
        row = next(r for r in rows if abs(r["h"] - h) < 1e-12)
        floors[omega_key] = row["L2_error"]
    return floors


def plot_spatial(spatial_csvs, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))

    all_h = []
    slopes = {}

    for omega_key, path in spatial_csvs.items():
        rows = sorted(read_csv(path), key=lambda r: r["h"])
        h = np.array([r["h"] for r in rows])
        all_h.extend(h.tolist())

        for err_key in ("L2_error", "H1_error"):
            err = np.array([r[err_key] for r in rows])
            label = f"{ERROR_LABEL[err_key]}, {OMEGA_LABEL[omega_key]}"
            ax.loglog(
                h,
                err,
                color=ERROR_COLOR[err_key],
                linestyle=OMEGA_LINESTYLE[omega_key],
                marker=OMEGA_MARKER[omega_key],
                label=label,
            )

            slope, _ = loglog_slope(h, err)
            slopes[(omega_key, err_key)] = slope

    # Draw reference O(h^2) and O(h) lines anchored at the coarsest
    # homogeneous data point to compare theoretical rates with measured data.
    h_min, h_max = min(all_h), max(all_h)

    rows_ref = sorted(read_csv(spatial_csvs[OMEGA_HOMOGENEOUS]), key=lambda r: r["h"])
    h0 = rows_ref[0]["h"]
    l2_0 = rows_ref[0]["L2_error"]
    h1_0 = rows_ref[0]["H1_error"]

    x_ref, y_ref2 = reference_line(h0, l2_0, 2, (h_min, h_max))
    ax.loglog(x_ref, y_ref2, color="0.7", linestyle="--", marker="", label=r"O($h^2$)")

    x_ref, y_ref1 = reference_line(h0, h1_0, 1, (h_min, h_max))
    ax.loglog(x_ref, y_ref1, color="0.7", linestyle=":", marker="", label=r"O($h$)")

    ax.set_xlabel("Mesh size h")
    ax.set_ylabel("Error")
    ax.set_title(r"Spatial convergence (P1, $\theta$=0.5)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "convergence_spatial.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {out_path}")
    print("Figure 1 (spatial) least-squares log-log slopes:")
    for omega_key in spatial_csvs:
        for err_key in ("L2_error", "H1_error"):
            print(
                f"  {OMEGA_LABEL[omega_key]:35s} {ERROR_LABEL[err_key]}: "
                f"slope = {slopes[(omega_key, err_key)]:.4f}"
            )


def plot_temporal(temporal_csvs, out_dir, spatial_floors):
    fig, ax = plt.subplots(figsize=(7, 6))

    slopes = {}
    excluded = []
    floor_compromised = []
    floor_trusted = []

    all_dt = []
    for omega_key, path in temporal_csvs.items():
        rows = sorted(read_csv(path), key=lambda r: r["dt"])

        kept = [r for r in rows if r["dt"] != EXCLUDED_DT]
        dropped = [r for r in rows if r["dt"] == EXCLUDED_DT]
        excluded.extend((omega_key, r) for r in dropped)

        # Flag kept points whose L2 falls below the independently-measured
        # spatial error floor at this h. Those points no longer isolate the
        # temporal error and are unreliable indicators of dt^2 convergence,
        # although we still plot and include them in the fit.
        floor = spatial_floors[omega_key]
        below_floor = [r for r in kept if r["dt"] == 0.005 and r["L2_error"] < floor]
        above_floor = [r for r in kept if r["dt"] == 0.005 and r["L2_error"] >= floor]
        floor_compromised.extend((omega_key, r, floor) for r in below_floor)
        floor_trusted.extend((omega_key, r, floor) for r in above_floor)

        dt = np.array([r["dt"] for r in kept])
        l2 = np.array([r["L2_error"] for r in kept])
        all_dt.extend(dt.tolist())

        ax.loglog(
            dt,
            l2,
            color=ERROR_COLOR["L2_error"],
            linestyle=OMEGA_LINESTYLE[omega_key],
            marker=OMEGA_MARKER[omega_key],
            label=f"L2, {OMEGA_LABEL[omega_key]}",
        )

        if below_floor:
            ax.loglog(
                [r["dt"] for r in below_floor],
                [r["L2_error"] for r in below_floor],
                linestyle="none",
                marker=OMEGA_MARKER[omega_key],
                markersize=11,
                markerfacecolor="white",
                markeredgecolor=ERROR_COLOR["L2_error"],
                markeredgewidth=1.8,
                zorder=5,
            )

        slope, _ = loglog_slope(dt, l2)
        slopes[omega_key] = slope

    # Add an O(dt^2) reference line anchored at homogeneous dt=0.02 for
    # visual comparison with the measured temporal convergence.
    rows_ref = sorted(read_csv(temporal_csvs[OMEGA_HOMOGENEOUS]), key=lambda r: r["dt"])
    anchor = next(r for r in rows_ref if r["dt"] == 0.02)
    dt_min, dt_max = min(all_dt), max(all_dt)
    x_ref, y_ref = reference_line(anchor["dt"], anchor["L2_error"], 2, (dt_min, dt_max))
    ax.loglog(x_ref, y_ref, color="0.6", linestyle="--", marker="", label=r"O($dt^2$)")

    ax.set_xlabel("Time step dt")
    ax.set_ylabel("L2 error")
    ax.set_title(r"Temporal convergence (P1, $\theta$=0.5, h=0.0078125)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    note_lines = ["dt = 0.0025 excluded (spatial error floor at h=0.0078125):"]
    for omega_key, r in excluded:
        note_lines.append(
            f"  {omega_key}: dt=0.0025 -> L2_error={r['L2_error']:.5e} (excluded)"
        )
    if floor_compromised:
        note_lines.append("dt = 0.005 below independently-measured spatial floor (open marker):")
        for omega_key, r, floor in floor_compromised:
            note_lines.append(
                f"  {omega_key}: L2_error={r['L2_error']:.5e} < floor={floor:.5e} (compromised)"
            )
        for omega_key, r, floor in floor_trusted:
            note_lines.append(
                f"  {omega_key}: L2_error={r['L2_error']:.5e} >= floor={floor:.5e} "
                f"-> dt=0.005 trustworthy"
            )
    note = "\n".join(note_lines)
    ax.text(
        0.98,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"),
    )

    fig.tight_layout()
    out_path = os.path.join(out_dir, "convergence_temporal.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {out_path}")
    print("Excluded points (spatial error floor, not plotted/fitted):")
    for omega_key, r in excluded:
        print(f"  {omega_key}: dt=0.0025 -> L2_error={r['L2_error']:.5e} (excluded)")
    print("dt=0.005 points compromised by the spatial error floor (plotted with open marker, still fitted):")
    for omega_key, r, floor in floor_compromised:
        print(f"  {omega_key}: L2_error={r['L2_error']:.5e} < floor={floor:.5e} (compromised)")
    for omega_key, r, floor in floor_trusted:
        print(f"  {omega_key}: L2_error={r['L2_error']:.5e} >= floor={floor:.5e} (trustworthy)")
    print("Figure 2 (temporal) least-squares log-log slopes (dt in {0.02, 0.01, 0.005}):")
    for omega_key in temporal_csvs:
        print(f"  {OMEGA_LABEL[omega_key]:35s} L2: slope = {slopes[omega_key]:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing the convergence CSVs (default: {DEFAULT_DATA_DIR})",
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

    spatial_csvs = {
        OMEGA_HOMOGENEOUS: os.path.join(data_dir, "convergence_spatial_omega4.44288.csv"),
        OMEGA_FORCED: os.path.join(data_dir, "convergence_spatial_omega6.28319.csv"),
    }
    temporal_csvs = {
        OMEGA_HOMOGENEOUS: os.path.join(data_dir, "convergence_temporal_omega4.44288.csv"),
        OMEGA_FORCED: os.path.join(data_dir, "convergence_temporal_omega6.28319.csv"),
    }

    plot_spatial(spatial_csvs, out_dir)
    spatial_floors = spatial_floor_at(spatial_csvs, TEMPORAL_STUDY_H)
    plot_temporal(temporal_csvs, out_dir, spatial_floors)


if __name__ == "__main__":
    main()
