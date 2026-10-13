"""A compact Direct Simulation Monte Carlo (DSMC) code for a hard-sphere gas.

Units: molecular mass m = 1, Boltzmann constant k = 1, reference temperature T = 1,
reference number density n = 1. The most probable speed is c_m = sqrt(2kT/m) = sqrt(2).

Algorithm (Bird 1994):
  1. move particles ballistically, apply wall boundary conditions
  2. sort particles into cells
  3. collide with the No-Time-Counter (NTC) scheme:
       N_cand = 1/2 N_c (N_c - 1) F_N (sigma g)_max dt / V_c
     accept a pair with probability g / g_max; hard-sphere scattering is isotropic
     in the centre-of-mass frame, so post-collision relative velocities are uniform on a sphere
  4. sample macroscopic moments
"""

from __future__ import annotations

import numpy as np


def random_unit_vectors(rng, n):
    cos_t = 2 * rng.random(n) - 1
    sin_t = np.sqrt(1 - cos_t**2)
    phi = 2 * np.pi * rng.random(n)
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def hard_sphere_viscosity(d, T=1.0):
    """Chapman-Enskog first-approximation viscosity of hard spheres (m = k = 1)."""
    return 5 / (16 * d**2) * np.sqrt(T / np.pi)


def mean_free_path(d, n=1.0):
    return 1 / (np.sqrt(2) * np.pi * d**2 * n)


def collide_cell(rng, v, idx, fn_sigma_dt_over_vol, gmax):
    """NTC collisions among particles `idx` of one cell. Returns (#collisions, new gmax)."""
    nc = len(idx)
    if nc < 2:
        return 0, gmax
    expected = 0.5 * nc * (nc - 1) * fn_sigma_dt_over_vol * gmax
    n_cand = int(expected + rng.random())
    if n_cand == 0:
        return 0, gmax
    i = rng.integers(0, nc, n_cand)
    j = (i + rng.integers(1, nc, n_cand)) % nc  # j != i
    A, Bp = idx[i], idx[j]
    n_coll = 0
    # Candidate pairs are processed in rounds of pairs with distinct particles, which is
    # equivalent to Bird's sequential loop (a particle may collide more than once per step).
    while len(A):
        # a pair goes in this round if it is the first pair (in order) touching both of its
        # particles; the first remaining pair always qualifies, so the loop terminates
        k = np.arange(len(A))
        first = np.full(v.shape[0], len(A))
        np.minimum.at(first, A, k)
        np.minimum.at(first, Bp, k)
        now = (first[A] == k) & (first[Bp] == k)
        a, b = A[now], Bp[now]
        A, Bp = A[~now], Bp[~now]
        g = np.linalg.norm(v[a] - v[b], axis=1)
        gmax = max(gmax, g.max())
        acc = rng.random(len(a)) < g / gmax
        a, b, g = a[acc], b[acc], g[acc]
        vcm = 0.5 * (v[a] + v[b])
        gnew = g[:, None] * random_unit_vectors(rng, len(a))
        v[a] = vcm + 0.5 * gnew
        v[b] = vcm - 0.5 * gnew
        n_coll += len(a)
    return n_coll, gmax


# ---------------------------------------------------------------- 0D relaxation


def relaxation(n_particles=50_000, d=0.3, n_density=1.0, t_end=15.0, dt=0.01, seed=0):
    """Spatially homogeneous hard-sphere gas starting from a 'two-shell' velocity distribution
    (all speeds equal, random directions). Tracks H(t), <v_x^4> and the collision count."""
    rng = np.random.default_rng(seed)
    v = np.sqrt(3.0) * random_unit_vectors(rng, n_particles)  # |v| = sqrt(3) -> T = 1
    vol = n_particles / n_density  # one cell holding everything; F_N = 1
    sigma = np.pi * d**2
    gmax = 2 * np.sqrt(3.0)
    t, hist, total_coll = 0.0, [], 0
    bins = np.linspace(-5, 5, 81)
    while t < t_end - 1e-12:
        c, gmax = collide_cell(rng, v, np.arange(n_particles), sigma * dt / vol, gmax)
        total_coll += c
        t += dt
        hx, _ = np.histogram(v[:, 0], bins=bins, density=True)
        H = np.sum(hx[hx > 0] * np.log(hx[hx > 0])) * (bins[1] - bins[0])
        kurt = np.mean(v[:, 0] ** 4) / np.mean(v[:, 0] ** 2) ** 2
        hist.append((t, H, kurt, total_coll))
    return v, np.array(hist), sigma


# ---------------------------------------------------------------- 1D Couette flow


def couette(kn, U=0.4, H=1.0, n_cells=None, ppc=150, steps=None, sample_from=None, seed=0, n_sample=4000):
    """Planar Couette flow between diffuse walls at y = 0 (u = -U/2) and y = H (u = +U/2), T_w = 1.

    kn = lambda / H with the hard-sphere mean free path at n = 1. Returns cell centres,
    mean velocity u(y), temperature T(y), and the wall shear stress (momentum flux)."""
    rng = np.random.default_rng(seed)
    lam = kn * H
    d = np.sqrt(1 / (np.sqrt(2) * np.pi * lam))  # hard-sphere diameter giving this mean free path
    if n_cells is None:  # cells must be a fraction of the mean free path
        n_cells = max(50, int(np.ceil(5 * H / lam)))
    sigma = np.pi * d**2
    N = n_cells * ppc
    Fn = H / N  # real molecules per simulated particle (n = 1 per unit length x unit area)
    dy = H / n_cells
    dt = 0.25 * min(dy, lam) / np.sqrt(2)
    if sample_from is None:
        # transient: several viscous relaxation times H^2 / (pi^2 nu), and at least ~20 wall-to-wall transits
        t_relax = max(6 * H**2 / (np.pi**2 * hard_sphere_viscosity(d)), 20 * H / np.sqrt(2))
        sample_from = int(np.ceil(t_relax / dt))
    if steps is None:
        steps = sample_from + n_sample
    y = H * rng.random(N)
    v = rng.normal(0, 1, (N, 3))  # Maxwellian at T = 1 (variance kT/m = 1)
    # start from the slip-corrected linear profile to shorten the transient (steady state does not depend on it)
    slope0 = continuum_slip_shear(kn, U, H) / hard_sphere_viscosity(d) if kn < 1 else U / (H + 2 * lam)
    v[:, 0] += slope0 * (y - 0.5 * H)
    gmax = np.full(n_cells, 4.0)
    sum_u = np.zeros(n_cells)
    sum_T = np.zeros(n_cells)
    sum_v = np.zeros(n_cells)
    sum_uv = np.zeros(n_cells)
    count = np.zeros(n_cells)
    mom_flux = 0.0  # x-momentum given to the top wall
    n_samples = 0
    for step in range(steps):
        y += v[:, 1] * dt
        for wall_y, wall_u, sign in ((0.0, -U / 2, 1.0), (H, U / 2, -1.0)):
            hit = (y < 0) if wall_y == 0 else (y > H)
            if not hit.any():
                continue
            nh = hit.sum()
            if wall_y == H and step >= sample_from:
                mom_flux += v[hit, 0].sum()
            # diffuse reflection at T_w = 1: normal component from the flux-weighted (Rayleigh) distribution
            vn = np.sqrt(-2 * np.log(1 - rng.random(nh)))
            v[hit, 1] = sign * vn
            v[hit, 0] = rng.normal(wall_u, 1, nh)
            v[hit, 2] = rng.normal(0, 1, nh)
            if wall_y == H and step >= sample_from:
                mom_flux -= v[hit, 0].sum()
            # time of flight after reflection is ignored below the cell scale: place at the wall
            y[hit] = wall_y + sign * 1e-12
        cell = np.minimum((y / dy).astype(int), n_cells - 1)
        order = np.argsort(cell, kind="stable")
        starts = np.searchsorted(cell[order], np.arange(n_cells + 1))
        for c in range(n_cells):
            idx = order[starts[c] : starts[c + 1]]
            _, gmax[c] = collide_cell(rng, v, idx, Fn * sigma * dt / dy, gmax[c])
        if step >= sample_from:
            n_samples += 1
            cnt = np.bincount(cell, minlength=n_cells)
            su = np.bincount(cell, weights=v[:, 0], minlength=n_cells)
            count += cnt
            sum_u += su
            sum_T += np.bincount(cell, weights=(v**2).sum(axis=1), minlength=n_cells)
            sum_v += np.bincount(cell, weights=v[:, 1], minlength=n_cells)
            sum_uv += np.bincount(cell, weights=v[:, 0] * v[:, 1], minlength=n_cells)
    u = sum_u / count
    vbar = sum_v / count
    T = (sum_T / count - u**2 - vbar**2) / 3
    n_cell = count / n_samples * Fn / dy  # number density in each cell
    pxy = n_cell * (sum_uv / count - u * vbar)  # kinetic shear stress n <v_x' v_y'>
    tau = mom_flux * Fn / (n_samples * dt)  # force per unit area on the top wall
    yc = (np.arange(n_cells) + 0.5) * dy
    return yc, u, T, abs(tau), d, pxy


def free_molecular_shear(U, n=1.0, T=1.0):
    """Shear stress between diffuse plates in the free-molecular limit: tau = n U sqrt(kT / (2 pi m))."""
    return n * U * np.sqrt(T / (2 * np.pi))


def continuum_slip_shear(kn, U, H=1.0):
    """Navier-Stokes with first-order Maxwell slip (coefficient 1.016 for hard spheres, Cercignani):
    tau = mu U / (H + 2 * 1.016 * sqrt(pi)/2 * lambda)  using  lambda_mu = 16 mu / (5 n sqrt(2 pi m k T))."""
    lam = kn * H
    d = np.sqrt(1 / (np.sqrt(2) * np.pi * lam))
    mu = hard_sphere_viscosity(d)
    lam_mu = 16 * mu / (5 * np.sqrt(2 * np.pi))  # viscosity-based mean free path (n = m = k = T = 1)
    slip = 1.016 * lam_mu
    return mu * U / (H + 2 * slip)
