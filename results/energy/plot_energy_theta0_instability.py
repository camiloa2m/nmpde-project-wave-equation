#!/usr/bin/env python3
"""Plot the theta=0 (fully explicit) numerical instability for the wave
equation solver, homogeneous manufactured solution (omega = pi*sqrt(2)).

Reads energy_theta0_dt005.csv and energy_theta0_dt05.csv (columns t,energy),
both truncated early once E^n exceeded 100*E(0), and writes
energy_theta0_instability.png next to this script.
"""

import csv
import os

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_DT005 = os.path.join(SCRIPT_DIR, "energy_theta0_dt005.csv")
CSV_DT05 = os.path.join(SCRIPT_DIR, "energy_theta0_dt05.csv")


def read_csv(path):
    t, energy = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            energy.append(float(row["energy"]))
    return t, energy


def main():
    t005, e005 = read_csv(CSV_DT005)
    t05, e05 = read_csv(CSV_DT05)

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
    out_path = os.path.join(SCRIPT_DIR, "energy_theta0_instability.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("=== theta = 0 (fully explicit) instability ===")
    print(f"  E(0)                       = {e0:.10g}")
    print(f"  dt=0.005: blew up after {n_steps_005} steps, at t = {t_blowup_005:.6g}, "
          f"E = {e005[-1]:.6g}")
    print(f"  dt=0.05:  blew up after {n_steps_05} steps, at t = {t_blowup_05:.6g}, "
          f"E = {e05[-1]:.6g}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
