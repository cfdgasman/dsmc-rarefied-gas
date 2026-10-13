import numpy as np
import pytest

from dsmc import continuum_slip_shear, couette, free_molecular_shear, hard_sphere_viscosity, mean_free_path, relaxation
from dsmc.core import collide_cell


def test_collisions_conserve_momentum_and_energy():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, (2000, 3))
    p0, e0 = v.sum(axis=0), (v**2).sum()
    n, _ = collide_cell(rng, v, np.arange(2000), 0.05, 6.0)
    assert n > 0
    assert np.allclose(v.sum(axis=0), p0, atol=1e-10)
    assert (v**2).sum() == pytest.approx(e0, rel=1e-12)


def test_collision_rate_matches_hard_sphere_theory():
    v, hist, sigma = relaxation(n_particles=20_000, t_end=12.0, seed=3)
    t, _, kurt, coll = hist.T
    late = t > 7
    rate = np.polyfit(t[late], coll[late], 1)[0]
    assert rate == pytest.approx(0.5 * len(v) * sigma * 4 / np.sqrt(np.pi), rel=0.02)
    assert kurt[-1] == pytest.approx(3.0, abs=0.1)  # relaxed to a Maxwellian


def test_mean_free_path_and_viscosity_are_consistent():
    d = 0.3
    # hard spheres: mu = (5 pi / 32) n m cbar lambda, cbar = sqrt(8 k T / (pi m))
    cbar = np.sqrt(8 / np.pi)
    assert hard_sphere_viscosity(d) == pytest.approx(5 * np.pi / 32 * cbar * mean_free_path(d), rel=1e-12)


def test_near_free_molecular_couette():
    _, _, _, tau, _, _ = couette(20.0, n_cells=20, ppc=100, steps=1500, sample_from=500)
    assert tau / free_molecular_shear(0.4) == pytest.approx(0.97, abs=0.05)


def test_slip_theory_limits():
    assert continuum_slip_shear(1e-4, 0.4) < 0.01 * free_molecular_shear(0.4)
