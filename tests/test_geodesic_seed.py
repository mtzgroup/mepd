import numpy as np

from mepd.geodesic_interpolation2 import interpolation
from mepd.geodesic_interpolation2.interpolation import _MidpointFinder


def test_midpoint_nudge_uses_seeded_rng(monkeypatch):
    atoms = ["H", "H"]
    geom1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    geom2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])

    monkeypatch.setattr(_MidpointFinder, "_INITIAL_GUESS_COEFFS", [0.5])
    monkeypatch.setattr(
        interpolation,
        "get_bond_list",
        lambda *args, **kwargs: ([], np.array([], dtype=float)),
    )
    monkeypatch.setattr(
        _MidpointFinder,
        "_least_squares_minimize",
        lambda self, initial_guess_flat: initial_guess_flat,
    )
    monkeypatch.setattr(
        _MidpointFinder,
        "_evaluate_midpoint_candidate_locally",
        lambda self, trial_midpoint: (trial_midpoint, 0.0),
    )

    first = interpolation.mid_point(
        atoms,
        geom1,
        geom2,
        tol=1e-2,
        nudge=0.2,
        rng=np.random.default_rng(7),
    )
    second = interpolation.mid_point(
        atoms,
        geom1,
        geom2,
        tol=1e-2,
        nudge=0.2,
        rng=np.random.default_rng(7),
    )
    different_seed = interpolation.mid_point(
        atoms,
        geom1,
        geom2,
        tol=1e-2,
        nudge=0.2,
        rng=np.random.default_rng(8),
    )

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, different_seed)
