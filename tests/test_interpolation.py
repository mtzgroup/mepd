"""Initial guess paths: geodesic (default), linear, LST and IDPP."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mepd.interpolation import idpp_path, interpolation_method, linear_path, lst_path

ANG = 1.8897259886
WATER = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]) * ANG
FLIPPED = WATER @ np.diag([-1.0, -1.0, 1.0])      # rotated 180 degrees about z: same distances


def _oh(x):
    return np.linalg.norm(x[1] - x[0]) / ANG, np.linalg.norm(x[2] - x[0]) / ANG


def test_linear_path_is_straight():
    p = linear_path(WATER, FLIPPED, 5)
    assert p.shape == (5, 3, 3)
    assert np.allclose(p[0], WATER) and np.allclose(p[-1], FLIPPED)
    assert np.allclose(p[2], (WATER + FLIPPED) / 2)


@pytest.mark.parametrize("fn", [lst_path, idpp_path])
def test_distance_based_paths_keep_bonds_where_linear_crushes_them(fn):
    mid = len(linear_path(WATER, FLIPPED, 9)) // 2
    lin = min(_oh(linear_path(WATER, FLIPPED, 9)[mid]))
    path = fn(WATER, FLIPPED, 9)
    assert np.allclose(path[0], WATER) and np.allclose(path[-1], FLIPPED)
    assert lin < 0.7                                         # linear: O-H squashed well below 0.96 A
    for x in path[1:-1]:
        assert min(_oh(x)) == pytest.approx(0.957, abs=0.05)  # LST / IDPP: bonds kept


def test_idpp_spaces_images_evenly():
    path = idpp_path(WATER, FLIPPED, 9)
    seg = np.linalg.norm(np.diff(path, axis=0).reshape(8, -1), axis=1)
    assert seg.std() / seg.mean() < 0.1


def test_frozen_atoms_follow_the_linear_path():
    frozen = np.array([0])
    for fn in (lst_path, idpp_path):
        path = fn(WATER, FLIPPED, 7, frozen)
        assert np.allclose(path[:, 0], linear_path(WATER, FLIPPED, 7)[:, 0])


def test_method_choice():
    assert interpolation_method(SimpleNamespace(interpolation="IDPP")) == "idpp"
    assert interpolation_method(SimpleNamespace()) == "geodesic"
    with pytest.raises(ValueError):
        interpolation_method(SimpleNamespace(interpolation="spline"))


@pytest.mark.parametrize("method", ["linear", "lst", "idpp"])
def test_initial_chain_builds_a_chain_of_fresh_nodes(method):
    from qcdata import Structure

    from mepd.chain import Chain
    from mepd.inputs import ChainInputs, GIInputs
    from mepd.interpolation import initial_chain
    from mepd.nodes.node import StructureNode

    a = StructureNode(structure=Structure(symbols=["O", "H", "H"], geometry=WATER))
    b = StructureNode(structure=Structure(symbols=["O", "H", "H"], geometry=FLIPPED + 0.3))
    ci = ChainInputs(interpolation=method)
    seed = Chain.model_validate({"nodes": [a, b], "parameters": ci})
    a._cached_energy = -1.0
    chain = initial_chain(seed, ci, GIInputs(nimages=8), align=False)
    assert len(chain) == 8
    assert np.allclose(chain[0].coords, a.coords) and np.allclose(chain[-1].coords, b.coords)
    assert chain[0] is not a and chain[0]._cached_energy is None     # fresh nodes, like a geodesic path
    aligned = initial_chain(seed, ci, GIInputs(nimages=8))           # default: end aligned onto start
    assert np.allclose(aligned[0].coords, a.coords) and not np.allclose(aligned[-1].coords, b.coords)


def test_the_removed_switch_in_older_profiles(tmp_path):
    from mepd.inputs import RunInputs

    base = 'engine_name = "gxtb"\npath_min_method = "NEB"\n[chain_inputs]\nk = 0.1\n'
    ok = tmp_path / "old.toml"
    ok.write_text(base + "use_geodesic_interpolation = true\n")      # every older profile has this
    assert RunInputs.open(ok).chain_inputs.interpolation == "geodesic"
    bad = tmp_path / "linear.toml"
    bad.write_text(base + "use_geodesic_interpolation = false\n")
    with pytest.raises(ValueError, match='interpolation = "linear"'):
        RunInputs.open(bad)
