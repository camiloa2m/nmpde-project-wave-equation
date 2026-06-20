#!/usr/bin/env python3
"""Plot the discrete energy conservation study for the wave equation solver
(Q1, theta-method time-stepping), homogeneous manufactured solution
(omega = pi*sqrt(2)).

Reads energy_homogeneous.csv (theta=0.5) and energy_theta1.csv (theta=1.0),
both with columns t,energy, and writes energy_theta05.png /
energy_comparison_log.png next to this script.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_THETA05 = os.path.join(SCRIPT_DIR, "energy_homogeneous.csv")
CSV_THETA1 = os.path.join(SCRIPT_DIR, "energy_theta1.csv")


def read_csv(path):
    t, energy = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            energy.append(float(row["energy"]))
    return np.array(t), np.array(energy)


def plot_theta05(t, e):
    e0 = e[0]
    emax, emin = e.max(), e.min()
    rel_variation = (emax - emin) / e0

    pad = 0.1 * (emax - emin) if emax > emin else 0.1 * abs(e0)
    ylim = (emin - pad, emax + pad)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, e, color="tab:blue", linewidth=0.8, label="E(t)")
    ax.axhline(e0, color="black", linestyle="--", linewidth=1.0, label="E(0)")

    ax.set_ylim(ylim)
    ax.set_xlabel("t")
    ax.set_ylabel("Discrete energy E(t)")
    ax.set_title("Discrete energy vs time, theta=0.5 (Crank-Nicolson)")
    ax.legend()

    ax.text(
        0.02,
        0.02,
        f"Relative variation: {rel_variation:.1e}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    fig.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "energy_theta05.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return e0, emax, emin, rel_variation


def plot_comparison(t05, e05, t1, e1):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t05, e05, color="tab:blue", linestyle="-",
            linewidth=1.2, label="theta = 0.5 (Crank-Nicolson)")
    ax.plot(t1, e1, color="tab:orange", linestyle="--",
            linewidth=1.2, label="theta = 1.0 (Implicit Euler)")

    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Discrete energy E(t) (log scale)")
    ax.set_title("Discrete energy decay: theta=0.5 vs theta=1.0")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "energy_comparison_log.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    t05, e05 = read_csv(CSV_THETA05)
    t1, e1 = read_csv(CSV_THETA1)

    e0_05, emax_05, emin_05, rel_var_05 = plot_theta05(t05, e05)
    plot_comparison(t05, e05, t1, e1)

    e0_1 = e1[0]
    eT_1 = e1[-1]
    ratio_1 = eT_1 / e0_1
    T_final = t1[-1]
    # Fit E(t) ~ E(0) * exp(-k*t) to a single decay rate over the whole run.
    k = -np.log(ratio_1) / T_final

    print("=== theta = 0.5 (Crank-Nicolson) ===")
    print(f"  E(0)               = {e0_05:.10g}")
    print(f"  max(E)             = {emax_05:.10g}")
    print(f"  min(E)             = {emin_05:.10g}")
    print(f"  (max-min)/E(0)     = {rel_var_05:.3e}")
    print()
    print("=== theta = 1.0 (Implicit Euler) ===")
    print(f"  E(0)               = {e0_1:.10g}")
    print(f"  E(T_final={T_final:g}) = {eT_1:.10g}")
    print(f"  E(T_final)/E(0)    = {ratio_1:.6g}")
    print(f"  estimated decay rate k = -ln(E(T)/E(0))/T_final = {k:.6g}")
    print()
    print(f"Saved: {os.path.join(SCRIPT_DIR, 'energy_theta05.png')}")
    print(f"Saved: {os.path.join(SCRIPT_DIR, 'energy_comparison_log.png')}")


if __name__ == "__main__":
    main()
