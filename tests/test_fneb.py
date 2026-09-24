import numpy as np
import pytest
from qcdata import Structure

pytest.importorskip("ase")
from ase.calculators.emt import EMT  # noqa: E402

from mepd.chain import Chain  # noqa: E402
from mepd.engines.ase import ASEEngine  # noqa: E402
from mepd.inputs import RunInputs  # noqa: E402
from mepd.msmep import MSMEP  # noqa: E402
from mepd.nodes.node import StructureNode  # noqa: E402

ANGSTROM_TO_BOHR = 1.8897259886


def _node(xyz):
    return StructureNode(structure=Structure(
        symbols=["H", "C", "N"], geometry=np.array(xyz) * ANGSTROM_TO_BOHR,
        charge=0, multiplicity=1,
    ))


# HCN -> HNC on ASE's EMT potential: not meaningful chemistry, but cheap,
# deterministic, and it has no interior energy maximum between the endpoints.
START = _node([[0.35, 0.0, -1.02], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
END = _node([[0.35, 0.0, 2.12], [0.0, 0.0, 0.0], [0.0, 0.0, 1.17]])


def _msmep(**path_min_inputs):
    run_inputs = RunInputs(path_min_method="FNEB", path_min_inputs=path_min_inputs)
    run_inputs.engine = ASEEngine(calculator=EMT())
    run_inputs.gi_inputs.nimages = 7
    return MSMEP(run_inputs)


def _chain(msmep):
    return Chain.model_validate({
        "nodes": [START.copy(), END.copy()], "parameters": msmep.inputs.chain_inputs,
    })


def test_fneb_grows_and_minimizes_with_the_configured_engine():
    msmep = _msmep(max_grow_iter=4, max_min_iter=10)
    fneb, _ = msmep.run_minimize_chain(_chain(msmep))
    assert len(fneb.chain_trajectory[-1]) > 2
    assert fneb.grad_calls_made > 2


def test_fneb_linear_tangent_is_used_for_every_node_pair_step(monkeypatch):
    msmep = _msmep(max_grow_iter=2, max_min_iter=5, tangent="linear")
    fneb = msmep._construct_path_minimizer(initial_chain=_chain(msmep))
    monkeypatch.setattr(
        fneb, "_geodesic_tangent",
        lambda *a, **k: pytest.fail("linear tangents must not compute geodesic tangents"),
    )
    fneb.optimize_chain()
    assert len(fneb.chain_trajectory) > 2


def test_fneb_max_energy_growth_without_a_barrier_is_elementary():
    """Max-energy growth that finds no interior maximum leaves just the two
    endpoints; that used to crash the elementary-step check."""
    msmep = _msmep(todd_way=False, max_grow_iter=3, barrier_thre=0.01, use_xtb_grow=False)
    fneb, results = msmep.run_minimize_chain(_chain(msmep))
    assert results.is_elem_step
    assert len(fneb.chain_trajectory[-1]) == 2


def test_fneb_uses_configured_engine_for_max_energy_growth_by_default():
    msmep = _msmep()
    assert msmep.inputs.path_min_inputs.use_xtb_grow is False
