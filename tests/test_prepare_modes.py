from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mepd.elementarystep import ModeSet, _expected_trans_rot_dof, prepare_modes
from mepd.nodes.node import StructureNode
from qcdata import Structure

# A real Cartesian Hessian (Hartree/bohr^2) and geometry (bohr) for water,
# computed with g-xTB -- hardcoded here (rather than calling the real
# binary) so this test suite stays fast and doesn't depend on gxtb being
# installed, while still exercising prepare_modes against real chemistry
# instead of an arbitrary matrix.
_WATER_GEOMETRY_BOHR = np.array([
    [0.014256194335015994, 0.7516261533573632, 0.0],
    [-1.4496148393633863, -0.34845258120846456, 0.0],
    [1.4353586450283702, -0.4031735721488972, 0.0],
])

_WATER_HESSIAN = np.array([
    [0.5652378355, -0.0028967667, -1.968e-06, -0.2900153526, -0.2193260618, 1.3951e-06, -0.2752224828, 0.2222228285, 5.729e-07],
    [-0.0028967667, 0.412519451, -7.099e-07, -0.1674765399, -0.1988641315, 9.317e-07, 0.1703733067, -0.2136553195, -2.218e-07],
    [-1.968e-06, -7.099e-07, 0.0, 1.3046e-06, 1.1127e-06, -0.0, 6.634e-07, -4.028e-07, -0.0],
    [-0.2900153526, -0.1674765399, 1.3046e-06, 0.3239712214, 0.1924382509, -1.4667e-06, -0.0339558687, -0.024961711, 1.621e-07],
    [-0.2193260618, -0.1988641315, 1.1127e-06, 0.1924382509, 0.1820991238, -9.896e-07, 0.0268878109, 0.0167650077, -1.231e-07],
    [1.3951e-06, 9.317e-07, -0.0, -1.4667e-06, -9.896e-07, 0.0, 7.16e-08, 5.79e-08, -0.0],
    [-0.2752224828, 0.1703733067, 6.634e-07, -0.0339558687, 0.0268878109, 7.16e-08, 0.3091783516, -0.1972611175, -7.35e-07],
    [0.2222228285, -0.2136553195, -4.028e-07, -0.024961711, 0.0167650077, 5.79e-08, -0.1972611175, 0.1968903118, 3.449e-07],
    [5.729e-07, -2.218e-07, -0.0, 1.621e-07, -1.231e-07, -0.0, -7.35e-07, 3.449e-07, 0.0],
])


def _water_node() -> StructureNode:
    return StructureNode(structure=Structure(
        symbols=["O", "H", "H"], geometry=_WATER_GEOMETRY_BOHR, charge=0, multiplicity=1,
    ))


def _raw_eigendecomposition_result():
    """A hessian_result exposing only a raw Cartesian Hessian -- forces
    `prepare_modes` through its bottom-of-the-line eigendecomposition
    fallback (source="hessian_eigendecomposition"), which returns all 3N
    eigenvectors including translations/rotations."""
    return SimpleNamespace(results=SimpleNamespace(hessian=_WATER_HESSIAN))


def test_expected_trans_rot_dof():
    assert _expected_trans_rot_dof(_WATER_GEOMETRY_BOHR) == 6  # non-linear triatomic
    assert _expected_trans_rot_dof(np.array([[0.0, 0.0, 0.0]])) == 3  # single atom
    linear = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert _expected_trans_rot_dof(linear) == 5  # colinear atoms
    assert _expected_trans_rot_dof(np.array([0.0, 0.0])) == 0  # not a real 3-D geometry


def test_prepare_modes_via_raw_eigendecomposition_fallback_drops_exactly_six():
    """Acceptance test from the hessian-sample report: on a water Hessian,
    the raw-eigenvector fallback path returns exactly 3 (9 - 6) modes, with
    n_dropped_trans_rot == 6."""
    node = _water_node()
    with pytest.warns(UserWarning, match="translation/rotation"):
        mode_set = prepare_modes(_raw_eigendecomposition_result(), node)

    assert mode_set.source == "hessian_eigendecomposition"
    assert len(mode_set) == 3
    assert mode_set.n_dropped_trans_rot == 6
    # ascending order
    assert mode_set.freqs_wavenumber == sorted(mode_set.freqs_wavenumber)


def test_prepare_modes_force_constants_match_hessian_eigenvalues_exactly():
    """k_i = u_i^T H u_i must reproduce the exact Hessian eigenvalue for a
    genuine (orthonormal) eigenvector -- the direct correctness check for
    the force-constant formula, independent of any frequency-unit question."""
    eigvals, eigvecs = np.linalg.eigh(_WATER_HESSIAN)
    node = _water_node()

    with pytest.warns(UserWarning, match="translation/rotation"):
        mode_set = prepare_modes(_raw_eigendecomposition_result(), node)

    # The 3 kept modes are the 3 largest-magnitude eigenvalues (indices 6,7,8
    # after ascending eigh); prepare_modes re-sorts by (broken-scale, but
    # monotonic in eigenvalue) frequency, so just check each returned force
    # constant is present among the genuine vibrational eigenvalues.
    vibrational_eigvals = sorted(eigvals[6:])
    assert sorted(round(k, 6) for k in mode_set.force_constants) == [
        round(v, 6) for v in vibrational_eigvals
    ]


def test_prepare_modes_via_normal_modes_cartesian_path_does_not_drop_anything():
    """When the QC program's own result already excludes translations/
    rotations (the common case), prepare_modes must not drop anything
    (n_dropped_trans_rot == 0) and must not warn."""
    eigvals, eigvecs = np.linalg.eigh(_WATER_HESSIAN)
    node = _water_node()
    # The 3 genuine vibrational eigenvectors, deliberately given out of
    # order to also exercise the ascending-sort behavior.
    vib_indices = [8, 6, 7]
    modes = [eigvecs[:, i].reshape(3, 3) for i in vib_indices]
    real_freqs_wavenumber = [3587.5, 1639.2, 3548.6]  # matches vib_indices' order

    result = SimpleNamespace(
        results=SimpleNamespace(
            normal_modes_cartesian=modes,
            freqs_wavenumber=real_freqs_wavenumber,
            hessian=_WATER_HESSIAN,
        )
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mode_set = prepare_modes(result, node)

    assert mode_set.source == "normal_modes_cartesian"
    assert mode_set.n_dropped_trans_rot == 0
    assert len(mode_set) == 3
    assert mode_set.freqs_wavenumber == [1639.2, 3548.6, 3587.5]  # sorted ascending
    # bend (lowest frequency) must be the softest mode
    assert mode_set.force_constants[0] < mode_set.force_constants[1]
    assert mode_set.force_constants[0] < mode_set.force_constants[2]
    assert all(k > 0 for k in mode_set.force_constants)


def test_prepare_modes_vectors_are_unit_normalized():
    node = _water_node()
    with pytest.warns(UserWarning, match="translation/rotation"):
        mode_set = prepare_modes(_raw_eigendecomposition_result(), node)
    for vec in mode_set.vectors:
        assert np.linalg.norm(vec) == pytest.approx(1.0)


def test_prepare_modes_on_ts_seed_sorts_imaginary_mode_first():
    """A TS seed's imaginary mode (negative frequency) must land at index 0
    after sorting -- it's the reaction coordinate."""
    eigvals, eigvecs = np.linalg.eigh(_WATER_HESSIAN)
    node = _water_node()
    vib_indices = [6, 7, 8]
    modes = [eigvecs[:, i].reshape(3, 3) for i in vib_indices]
    # Pretend the lowest vibrational mode is actually imaginary (a TS).
    freqs = [-1639.2, 3548.6, 3587.5]

    result = SimpleNamespace(
        results=SimpleNamespace(
            normal_modes_cartesian=modes, freqs_wavenumber=freqs, hessian=_WATER_HESSIAN,
        )
    )
    mode_set = prepare_modes(result, node)

    assert mode_set.freqs_wavenumber[0] < 0
    assert mode_set.imaginary_indices == [0]


def test_prepare_modes_with_no_hessian_data_returns_empty_modeset():
    node = _water_node()
    mode_set = prepare_modes(SimpleNamespace(results=SimpleNamespace()), node)
    assert mode_set == ModeSet(
        vectors=[], freqs_wavenumber=[], force_constants=[],
        imaginary_indices=[], source="none", n_dropped_trans_rot=0,
    )
