"""The QM/MM free-energy surface (mepd.engines.mean_force) and the cage /
mean-force environments of a profile."""
from __future__ import annotations

import numpy as np
import pytest
from qcdata import Structure

from mepd.engines.mean_force import free_energy_profile, onto_solute
from mepd.nodes.node import StructureNode
from mepd.qmmm import QMMMRegion

# Methanol (QM, atoms 0-5) between three waters.
SYSTEM = """15

C   0.000  0.000  0.000
O   1.410  0.000  0.000
H  -0.360  1.030  0.000
H  -0.360 -0.510  0.890
H  -0.360 -0.510 -0.890
H   1.730  0.910  0.000
O   3.900  1.300  0.000
H   4.300  2.150  0.000
H   4.600  0.650  0.000
O  -2.900  0.300  2.300
H  -3.300  1.150  2.300
H  -3.600 -0.350  2.300
O   0.500 -3.200  0.400
H   0.900 -3.700 -0.300
H  -0.400 -3.500  0.500
"""


def _xtb():
    from mepd.programs import xtb_executable

    if not xtb_executable(download=False):
        pytest.skip("needs the xtb program")


def _reset_globals():
    from mepd.interpolation import set_global_frozen_atoms

    StructureNode.set_global_graph_atoms(None)
    set_global_frozen_atoms([])


def test_a_frame_is_turned_back_onto_the_solute():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(9, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    moved = x @ q + [1.0, -2.0, 0.5]
    moved[:3] += 0.01 * rng.normal(size=(3, 3))       # the restrained solute wobbles a little
    back = onto_solute(moved, x[:3], [0, 1, 2])
    assert np.allclose(back[:3], x[:3])                # the solute exactly where it was
    assert np.abs(back[3:] - x[3:]).max() < 0.05       # the environment came along


def test_free_energy_profile_integrates_the_mean_force():
    # A(s) = s^4 - s^2 along atom 0's x: the profile is A(s_i) - A(s_0).
    s = np.linspace(-1.0, 1.0, 41)
    nodes = []
    for v in s:
        geom = np.array([[v, 0.0, 0.0], [0.0, 3.0, 0.0]])
        n = StructureNode(structure=Structure(symbols=["H", "H"], geometry=geom), has_molecular_graph=False)
        grad = np.zeros_like(geom)
        grad[0, 0] = 4 * v ** 3 - 2 * v
        n._cached_energy, n._cached_gradient = 0.0, grad
        nodes.append(n)
    profile = free_energy_profile(nodes, solute=[0]) / 627.509474
    exact = (s ** 4 - s ** 2) - (s[0] ** 4 - s[0] ** 2)
    assert np.abs(profile - exact).max() < 5e-3      # trapezoid rule, 0.05 spacing


def test_mean_force_on_the_solute_only(tmp_path):
    _xtb()
    from mepd.engines.gfnff import XTBEngine
    from mepd.engines.mean_force import MeanForceEngine
    from mepd.engines.qmmm import QMMMEngine

    s = Structure.from_xyz(SYSTEM)
    region = QMMMRegion.build(s, "0-5", active_radius=None)
    base = QMMMEngine(base=XTBEngine(method="gfn2", n_parallel=1), region=region, n_parallel=2)
    mf = MeanForceEngine(base=base, equilibrate_ps=0.02, sample_ps=0.05, frames=3, n_parallel=1)
    assert mf.solute == [0, 1, 2, 3, 4, 5]
    node = StructureNode(structure=s, has_molecular_graph=False)
    out = mf.sample(node)
    g = out["gradient"]
    assert np.all(g[mf.environment] == 0.0) and np.abs(g[mf.solute]).max() > 1e-4
    assert out["frames"] >= 2 and np.isfinite(out["energy"]) and out["interaction_kcal"] < 0   # H-bonded waters
    # The engine caches it like any other, and refuses what it cannot do on this surface.
    assert np.allclose(mf.compute_gradients([node])[0], mf.compute_gradients([node])[0])
    with pytest.raises(NotImplementedError):
        mf.compute_hessian(node)


@pytest.mark.parametrize("environment", ["cage", "mean_force"])
def test_profile_environment_freezes_everything_but_the_qm_region(tmp_path, environment):
    _xtb()
    from mepd.engines.frozen import FrozenAtomsEngine
    from mepd.engines.mean_force import MeanForceEngine
    from mepd.inputs import RunInputs

    s = Structure.from_xyz(SYSTEM)
    QMMMRegion.build(s, "0-5", active_radius=6.0).save(tmp_path / "region.json")
    (tmp_path / "p.toml").write_text(f'engine_name = "xtb"\nqmmm_environment = "{environment}"\n'
                                     '[mean_force]\nsample_ps = 0.1\n[qmmm]\nfile = "region.json"\n')
    try:
        ri = RunInputs.open(tmp_path / "p.toml")
        assert isinstance(ri.engine, FrozenAtomsEngine)
        assert ri.engine.frozen == list(range(6, 15)) == list(ri.chain_inputs.frozen_atom_indices)
        assert isinstance(ri.engine.base, MeanForceEngine) == (environment == "mean_force")
        if environment == "mean_force":
            assert ri.engine.base.sample_ps == 0.1
    finally:
        _reset_globals()


def test_unknown_environment_is_refused(tmp_path):
    _xtb()
    from mepd.inputs import RunInputs

    QMMMRegion.build(Structure.from_xyz(SYSTEM), "0-5").save(tmp_path / "region.json")
    (tmp_path / "p.toml").write_text('engine_name = "xtb"\nqmmm_environment = "frozen"\n[qmmm]\nfile = "region.json"\n')
    try:
        with pytest.raises(Exception, match="relaxed, cage or mean_force"):
            RunInputs.open(tmp_path / "p.toml")
    finally:
        _reset_globals()
