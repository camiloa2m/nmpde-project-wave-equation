#!/usr/bin/env python3
"""Visualize the theta=0 (fully explicit) numerical instability.

This script compares two runs (dt=0.005 and dt=0.05) for the homogeneous
manufactured solution (omega = pi*sqrt(2)). It reads the CSVs
`energy_theta0_dt005.csv` and `energy_theta0_dt05.csv` (columns: ``t``,
``energy``). Each CSV was truncated once the discrete energy exceeded
100*E(0) to avoid plotting the runaway tail. The script writes
`energy_theta0_instability.png` and prints fitted growth-rate diagnostics.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "results", "energy")

# Values <= PLATEAU_FACTOR * E(0) define the early near-flat plateau and
# are excluded from the exponential-growth fit window used to estimate k.
PLATEAU_FACTOR = 1.5


def read_csv(path):
    t, energy = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            energy.append(float(row["energy"]))
    return t, energy


def growth_rate(t, energy, label):
        """Estimate the exponential blow-up rate k two ways and print details.

        Methods:
        - endpoint: k = ln(E_final/E_0) / t_final computed over the whole series.
        - fitted: least-squares slope of log(E) vs t restricted to the points
            after the initial plateau (E > PLATEAU_FACTOR * E(0)).

        Returns a tuple (k_endpoint, k_fit). Detailed window info is printed for
        diagnostic purposes.
        """
    t = np.asarray(t)
    energy = np.asarray(energy)
    e0 = energy[0]

    k_endpoint = np.log(energy[-1] / e0) / t[-1]

    mask = energy > PLATEAU_FACTOR * e0
    # Include the last plateau point as the fit window's anchor so the
    # regression captures the onset of exponential growth rather than
    # starting one sample late.
    first_unstable = np.argmax(mask)
    start = max(first_unstable - 1, 0)
    t_fit = t[start:]
    e_fit = energy[start:]

    k_fit = np.polyfit(t_fit, np.log(e_fit), 1)[0]

    print(f"  [{label}] endpoint k = ln(E_final/E_0)/t_final = {k_endpoint:.6g}"
          f"  (window: t=[{t[0]:.6g}, {t[-1]:.6g}], all {len(t)} points)")
    print(f"  [{label}] fitted   k = slope of log(E) vs t (least squares) = {k_fit:.6g}"
          f"  (window: t=[{t_fit[0]:.6g}, {t_fit[-1]:.6g}], "
          f"{len(t_fit)} points: t = {[f'{x:.6g}' for x in t_fit]})")

    return k_endpoint, k_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing the energy CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write the PNG to (default: same as --data-dir)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)

    csv_dt005 = os.path.join(data_dir, "energy_theta0_dt005.csv")
    csv_dt05 = os.path.join(data_dir, "energy_theta0_dt05.csv")

    t005, e005 = read_csv(csv_dt005)
    t05, e05 = read_csv(csv_dt05)

    e0 = e005[0]
    n_steps_005 = len(t005) - 1
    n_steps_05 = len(t05) - 1
    t_blowup_005 = t005[-1]
    t_blowup_05 = t05[-1]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t005, e005, "o-", color="tab:blue", label="theta=0, dt=0.005")
    ax.plot(t05, e05, "o-", color="tab:red", label="theta=0, dt=0.05")
    ax.axhline(e0, color="gray", linestyle="--", linewidth=1.0, label="E(0)")

    ax.set_yscale("log")
    xmax = 1.3 * max(t_blowup_005, t_blowup_05)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("t")
    ax.set_ylabel("Discrete energy E(t) (log scale)")
    ax.set_title("Numerical instability of theta=0 (fully explicit scheme)")
    ax.legend()

    ax.text(
        0.02,
        0.98,
        f"Both runs blew up (E > 100*E(0)) within a fraction of a time unit:\n"
        f"  dt=0.005: {n_steps_005} steps (t={t_blowup_005:.3g})\n"
        f"  dt=0.05:  {n_steps_05} steps (t={t_blowup_05:.3g})",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    fig.tight_layout()
    out_path = os.path.join(out_dir, "energy_theta0_instability.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("=== theta = 0 (fully explicit) instability ===")
    print(f"  E(0)                       = {e0:.10g}")
    print(f"  dt=0.005: blew up after {n_steps_005} steps, at t = {t_blowup_005:.6g}, "
          f"E = {e005[-1]:.6g}")
    print(f"  dt=0.05:  blew up after {n_steps_05} steps, at t = {t_blowup_05:.6g}, "
          f"E = {e05[-1]:.6g}")
    print()
    print("Growth rate k (E(t) ~ E(0) * exp(k*t)):")
    growth_rate(t005, e005, "dt=0.005")
    growth_rate(t05, e05, "dt=0.05 ")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
