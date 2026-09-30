"""The geodesic speed-ups must not change results: the vendored trf/lsmr solver
matches scipy.optimize.least_squares bit for bit, and the vectorized Jacobian
matches the block-by-block assembly."""
import numpy as np
import pytest
import scipy.sparse as sps
from scipy.optimize import least_squares as scipy_least_squares

from mepd.geodesic_interpolation2 import _fast_lsq
from mepd.geodesic_interpolation2.morsegeodesic import MorseGeodesic, run_geodesic_get_smoother

pytestmark = pytest.mark.skipif(not _fast_lsq.ENABLED, reason="numba unavailable: scipy fallback in use")


def _problem(seed=0, m=120, n=30):
    rng = np.random.default_rng(seed)
    A = sps.random(m, n, density=0.15, random_state=seed, format="csr") + sps.eye(m, n, format="csr")
    b = rng.standard_normal(m)

    def fun(x):
        return A @ x + 0.1 * np.sin(x).sum() - b + 0.05 * (A @ x) ** 3

    def jac(x):
        Ax = A @ x
        return sps.csc_matrix(sps.diags(1 + 0.15 * Ax ** 2) @ A + 0.1 * np.cos(x)[None, :])

    return fun, jac, rng.standard_normal(n)


@pytest.mark.parametrize("loss,x_scale,max_nfev", [("linear", None, None), ("soft_l1", "jac", 15),
                                                   ("soft_l1", "jac", None)])
def test_least_squares_matches_scipy_bitwise(loss, x_scale, max_nfev):
    fun, jac, x0 = _problem()
    ref = scipy_least_squares(fun, x0, jac=jac, method="trf", tr_solver="lsmr", loss=loss,
                              x_scale=x_scale if x_scale is not None else 1.0, max_nfev=max_nfev,
                              ftol=None if max_nfev else 1e-8, xtol=None if max_nfev else 1e-8,
                              gtol=2e-3)
    got = _fast_lsq.least_squares(fun, x0, jac=jac, method="trf", tr_solver="lsmr", loss=loss,
                                  x_scale=x_scale, max_nfev=max_nfev,
                                  ftol=None if max_nfev else 1e-8, xtol=None if max_nfev else 1e-8,
                                  gtol=2e-3)
    assert np.array_equal(ref.x, got.x)
    assert ref.nfev == got.nfev and ref.status == got.status


def _path():
    rng = np.random.default_rng(3)
    A = rng.uniform(-2.0, 2.0, size=(9, 3))
    B = A + rng.normal(scale=0.6, size=A.shape)
    atoms = ["C", "C", "O", "H", "H", "H", "H", "N", "H"]
    return atoms, np.array([A, B])


@pytest.mark.parametrize("ignore_atoms", [[], [0, 4]])
def test_vectorized_jacobian_matches_block_assembly(ignore_atoms):
    atoms, X = _path()
    smoother = run_geodesic_get_smoother((atoms, X), nimages=6, friction=1e-3, ignore_atoms=ignore_atoms)
    mg = MorseGeodesic(atoms, smoother.path, 1.7, friction=1e-3, ignore_atoms=ignore_atoms)
    mg._compute_disps(1e-3, np.zeros(4 * mg.num_cart_coords))
    fast = mg._compute_disp_grad(1, 5, 1e-3, True)
    mg._pairs_sorted = False  # force the original COO block-by-block path
    slow = mg._compute_disp_grad(1, 5, 1e-3, True)
    fast, slow = sps.csr_array(fast), sps.csr_array(sps.csc_matrix(slow))
    assert np.array_equal(fast.indptr, slow.indptr)
    assert np.array_equal(fast.indices, slow.indices)
    assert np.array_equal(fast.data, slow.data)
