"""DSMC studies: 0D relaxation to equilibrium and planar Couette flow across Knudsen numbers."""

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsmc import continuum_slip_shear, couette, free_molecular_shear, relaxation

U = 0.4
KNS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def relaxation_figure():
    v, hist, sigma = relaxation()
    t, H, kurt, coll = hist.T
    tau_c = 1 / (sigma * 4 / np.sqrt(np.pi))
    late = t > 8
    rate = np.polyfit(t[late], coll[late], 1)[0]
    theory = 0.5 * len(v) * sigma * 4 / np.sqrt(np.pi)
    print(f"collision rate {rate:.1f} /time, theory 1/2 N n sigma <g> = {theory:.1f}  (ratio {rate / theory:.4f})")
    print(f"final kurtosis {kurt[-1]:.3f} (Maxwellian: 3), mean collision time {tau_c:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    c = np.linspace(-4.5, 4.5, 300)
    axes[0].hist(v[:, 0], bins=80, density=True, alpha=0.6, label="DSMC, t = 15 (≈10 collision times)")
    axes[0].plot(c, np.exp(-c**2 / 2) / np.sqrt(2 * np.pi), "k-", label="Maxwell–Boltzmann, T = 1")
    axes[0].plot(c, np.where(np.abs(c) < np.sqrt(3), 1 / (2 * np.sqrt(3)), 0), "r--", lw=1, label="initial: |v| = √3 shell")
    axes[0].set(xlabel="v_x", ylabel="f(v_x)", title="Velocity distribution")
    axes[0].legend(fontsize=7)
    axes[1].plot(t / tau_c, H, "C0-")
    axes[1].axhline(-0.5 * np.log(2 * np.pi) - 0.5, color="k", ls="--", label="Maxwellian value")
    axes[1].set(xlabel="t / τ_c", ylabel="H = ∫ f ln f dv_x", title="Boltzmann H-theorem")
    axes[1].legend(fontsize=8)
    axes[2].plot(t / tau_c, kurt, "C2-")
    axes[2].axhline(3, color="k", ls="--", label="Maxwellian (3)")
    axes[2].set(xlabel="t / τ_c", ylabel="⟨v_x⁴⟩ / ⟨v_x²⟩²", title="Kurtosis")
    axes[2].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/relaxation.png", dpi=120)
    plt.close(fig)


def couette_study():
    rows, profiles = [], {}
    tau_fm = free_molecular_shear(U)
    for kn in KNS:
        t0 = time.perf_counter()
        y, u, T, tau, d, pxy = couette(kn, U=U)
        profiles[kn] = (y, u, T)
        bulk = -pxy[(y > 0.25) & (y < 0.75)].mean()
        rows.append((kn, tau / tau_fm, bulk / tau_fm, continuum_slip_shear(kn, U) / tau_fm, u[0] + U / 2))
        print(f"Kn = {kn:5.2f}: tau/tau_fm = {tau / tau_fm:.4f} (bulk {bulk / tau_fm:.4f}), "
              f"NS+slip {continuum_slip_shear(kn, U) / tau_fm:.4f}, wall slip {u[0] + U / 2:.4f}  ({time.perf_counter() - t0:.0f} s)")
    rows = np.array(rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    kn_line = np.logspace(-2, 1.2, 200)
    a1.semilogx(kn_line, [continuum_slip_shear(k, U) / tau_fm for k in kn_line], "k--", label="Navier–Stokes + first-order slip")
    a1.semilogx(kn_line, [U / 1.0 * 5 / 16 * np.sqrt(1 / np.pi) * (np.sqrt(2) * np.pi * k) / tau_fm for k in kn_line],
                "k:", label="Navier–Stokes, no slip")
    a1.axhline(1.0, color="grey", lw=1, label="free-molecular limit")
    a1.semilogx(rows[:, 0], rows[:, 1], "o", ms=7, color="C3", label="DSMC (wall momentum flux)")
    a1.set(xlabel="Kn = λ / H", ylabel="τ / τ_fm", ylim=(0, 1.1), title="Wall shear stress across flow regimes")
    a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=8)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(KNS)))
    for c, kn in zip(colors, KNS):
        y, u, _ = profiles[kn]
        a2.plot(u / U, y, "-", color=c, label=f"Kn = {kn}")
    a2.plot([-0.5, 0.5], [0, 1], "k--", lw=1, label="no-slip continuum")
    a2.set(xlabel="u / U", ylabel="y / H", title="Velocity profiles: slip grows with Kn")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/couette.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4))
    for c, kn in zip(colors, KNS):
        y, _, T = profiles[kn]
        ax.plot(T, y, color=c, label=f"Kn = {kn}")
    ax.set(xlabel="T / T_wall", ylabel="y / H", title="Viscous heating and temperature jump")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/temperature.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    relaxation_figure()
    if "--relaxation-only" not in sys.argv:
        couette_study()
