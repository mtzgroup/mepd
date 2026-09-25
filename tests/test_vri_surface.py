from __future__ import annotations

import numpy as np

from mepd.discovery.vri_surface import Axes, interpolate


def test_axes_wilson_matrix_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3)) * 2.0
    axes = Axes(X.copy(), rng.normal(size=18), bonds_a=[(0, 1), (2, 3)], bonds_b=[(1, 4)])
    q0, B = axes(X)
    assert q0[0] == 0.0  # at the reference geometry the progress coordinate is zero
    h = 1e-6
    fd = np.zeros_like(B)
    for k in range(18):
        e = np.zeros(18); e[k] = h
        fd[:, k] = (axes(X.reshape(-1) + e)[0] - axes(X.reshape(-1) - e)[0]) / (2 * h)
    np.testing.assert_allclose(B, fd, atol=1e-7)


def test_shepard_interpolation_reproduces_data_points():
    data = [{"q": [0.0, 0.0], "e": 1.0, "g": [0.0, 0.0], "K": None},
            {"q": [1.0, 0.0], "e": 3.0, "g": [0.0, 0.0], "K": [[2.0, 0.0], [0.0, 2.0]]}]
    E = interpolate(data, np.array([0.0, 1.0]), np.array([0.0]))
    np.testing.assert_allclose(E[0], [1.0, 3.0], atol=1e-6)
