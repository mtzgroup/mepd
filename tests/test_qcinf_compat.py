"""snap_rmsd for structures with unbonded atoms (bare ions), which
qcinf <= 0.4.1 cannot compare at all."""

from __future__ import annotations

import numpy as np
import pytest
from qcdata import Structure

from mepd._qcinf_compat import snap_rmsd

ANG = 1.8897259886

# water + Na+ + Na+ + Cl-: three unbonded atoms, two of the same element.
SYMBOLS = ["O", "H", "H", "Na", "Na", "Cl"]
GEOM_A = np.array([[0.0, 0.0, 0.0], [0.76, 0.0, 0.5], [-0.76, 0.0, 0.5],
                   [3.0, 0.0, 0.0], [-3.0, 0.5, 0.0], [0.0, 3.5, 0.0]]) * ANG


def _s(geom, symbols=SYMBOLS):
    return Structure(symbols=symbols, geometry=geom, charge=1, multiplicity=1)


def test_identical_and_rigidly_moved_copies_are_zero():
    a = _s(GEOM_A)
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert snap_rmsd(a, a) == pytest.approx(0.0, abs=1e-8)
    assert snap_rmsd(a, _s(GEOM_A @ rot + 2.0)) == pytest.approx(0.0, abs=1e-6)


def test_same_element_ions_are_matched_not_taken_in_order():
    swapped = GEOM_A.copy()
    swapped[[3, 4]] = swapped[[4, 3]]          # the two Na+ trade places
    assert snap_rmsd(_s(GEOM_A), _s(swapped)) == pytest.approx(0.0, abs=1e-6)


def test_moved_ion_gives_its_share_of_the_rmsd():
    moved = GEOM_A.copy()
    moved[5, 1] += 1.0 * ANG                    # Cl- 1 Å further out
    assert snap_rmsd(_s(GEOM_A), _s(moved), units="angstrom") == pytest.approx(np.sqrt(1.0 / 6), rel=0.05)


def test_different_unbonded_atoms_are_not_isomorphic():
    other = ["O", "H", "H", "Na", "Na", "Br"]
    with pytest.raises(ValueError):
        snap_rmsd(_s(GEOM_A), _s(GEOM_A, other))


def test_fully_bonded_structures_take_the_plain_qcinf_path():
    import qcinf

    water = _s(GEOM_A[:3], ["O", "H", "H"])
    bent = _s(GEOM_A[:3] * np.array([1.0, 1.0, 1.1]), ["O", "H", "H"])
    assert snap_rmsd(water, bent) == pytest.approx(qcinf.snap_rmsd(water, bent))
