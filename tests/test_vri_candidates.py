from __future__ import annotations

import numpy as np
from qcdata import Structure

from mepd.discovery import vri_candidates as vc
from mepd.engines.engine import Engine
from mepd.nodes.node import StructureNode


def _water_dimer(offset: float = 0.0) -> StructureNode:
    """Two waters related by a C2 rotation about z; `offset` (bohr)
    stretches one O-H of the second water so the pair is only
    approximately symmetric."""
    w = np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 1.1], [-1.43, 0.0, 1.1]]) + [3.0, 0.0, 0.0]
    c2 = w * np.array([-1.0, -1.0, 1.0])
    c2[1] += offset * (c2[1] - c2[0]) / np.linalg.norm(c2[1] - c2[0])
    return StructureNode(structure=Structure(
        symbols=["O", "H", "H", "O", "H", "H"], geometry=np.vstack([w, c2]), charge=0, multiplicity=1,
    ))


def test_symmetrized_geometry_is_exactly_symmetric_under_the_swap():
    node = _water_dimer(offset=0.4)
    G = vc._partial_bond_graph(node)
    assert sorted(G.nodes) == [0, 3] and G.number_of_edges() == 0
    Z = vc._symmetrized_geometry(node, {0: 3, 3: 0})
    perm = [3, 4, 5, 0, 1, 2]
    # The swapped copy of the symmetrized geometry aligns onto itself (H pairs
    # may come out in either order, so compare the heavy atoms and the H set).
    assert vc.aligned_rmsd(Z[[3, 0]], Z[[0, 3]]) < 1e-8
    before = vc.aligned_rmsd(node.coords[perm], node.coords)
    after = min(vc.aligned_rmsd(Z[p], Z) for p in ([3, 4, 5, 0, 1, 2], [3, 5, 4, 0, 2, 1]))
    assert before > 0.05 and after < 1e-6


class _OODistanceEngine(Engine):
    """Energy as a function of the O...O distance only: `profile(t)` with
    t = 0 at `d0` and t = 1 at `d1` (bohr)."""

    def __init__(self, d0, d1, profile):
        self.d0, self.d1, self.profile = d0, d1, profile

    def _t(self, node):
        X = np.asarray(node.coords)
        return (np.linalg.norm(X[0] - X[3]) - self.d0) / (self.d1 - self.d0)

    def compute_energies(self, nodes):
        return [float(self.profile(self._t(n))) for n in nodes]

    def compute_gradients(self, nodes):
        return [np.zeros_like(np.asarray(n.coords)) for n in nodes]


def test_ridge_connected_accepts_downhill_path_and_rejects_a_dip_below_ts2():
    a = _water_dimer(0.0)
    X = a.coords.copy()
    X[3:] += np.array([1.0, 0.0, 0.0]) * np.sign(X[3, 0] - X[0, 0])  # pull the second water away
    b = a.update_coords(X)
    d0, d1 = np.linalg.norm(a.coords[0] - a.coords[3]), np.linalg.norm(X[0] - X[3])

    downhill = _OODistanceEngine(d0, d1, lambda t: -0.01 * t)
    res = vc.ridge_connected(a, b, downhill, n_images=7)
    assert res["connected"] is True
    assert res["ts2_rel_ts1_kcal"] < -5

    valley = _OODistanceEngine(d0, d1, lambda t: -0.001 * t - 0.02 * np.sin(np.pi * t))
    res = vc.ridge_connected(a, b, valley, n_images=7)
    assert res["connected"] is False
    assert res["min_interior_kcal"] < res["ts2_rel_ts1_kcal"] - 1.0
