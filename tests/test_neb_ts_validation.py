from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

pytest.importorskip("ase")
from ase.calculators.emt import EMT  # noqa: E402

import mepd.irc as irc_module  # noqa: E402
from mepd.chain import Chain  # noqa: E402
from mepd.cli_common import _ts_guess_tasks_from_tree  # noqa: E402
from mepd.engines.ase import ASEEngine  # noqa: E402
from mepd.engines.gxtb import GXTBCalculator  # noqa: E402
from mepd.inputs import RunInputs  # noqa: E402
from mepd.msmep import MSMEP  # noqa: E402
from mepd.neb import NEB  # noqa: E402
from mepd.nodes.node import StructureNode  # noqa: E402
from mepd.TreeNode import TreeNode  # noqa: E402

ANGSTROM_TO_BOHR = 1.8897259886


def _node(xyz):
    return StructureNode(structure=Structure(
        symbols=["H", "C", "N"], geometry=np.array(xyz) * ANGSTROM_TO_BOHR,
        charge=0, multiplicity=1,
    ))


START = _node([[0.35, 0.0, -1.02], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
END = _node([[0.35, 0.0, 2.12], [0.0, 0.0, 0.0], [0.0, 0.0, 1.17]])


class _TSCapableEMT(ASEEngine):
    """EMT with a stand-in TS optimizer, so NEB's validation hook is live."""

    def compute_transition_state(self, node, keywords=None):
        return node


def _msmep(**path_min_inputs):
    run_inputs = RunInputs(path_min_method="NEB", path_min_inputs={"v": False, **path_min_inputs})
    run_inputs.engine = _TSCapableEMT(calculator=EMT())
    run_inputs.gi_inputs.nimages = 7
    return MSMEP(run_inputs)


def _chain(msmep):
    return Chain.model_validate({
        "nodes": [START.copy(), END.copy()], "parameters": msmep.inputs.chain_inputs,
    })


def test_gxtb_parallel_map_keeps_order():
    engine = GXTBCalculator(executable="unused", n_parallel=4)
    assert engine._map(lambda x: x * x, list(range(20))) == [x * x for x in range(20)]
    assert GXTBCalculator(executable="unused", n_parallel=1)._map(str, [1, 2]) == ["1", "2"]


def test_ts_validation_schedule_doubles_from_first_step():
    neb = SimpleNamespace(
        parameters=SimpleNamespace(stop_on_validated_ts=True, ts_validation_first_step=5),
        engine=_TSCapableEMT(calculator=EMT()),
    )
    due = [n for n in range(1, 90) if NEB._ts_validation_due(neb, n)]
    assert due == [5, 10, 20, 40, 80]
    neb.parameters.stop_on_validated_ts = False
    assert not any(NEB._ts_validation_due(neb, n) for n in range(1, 90))


def test_neb_stops_at_first_validated_ts_guess(monkeypatch):
    calls = []

    def fake_optimize(engine, guess):
        calls.append(guess)
        return guess, Chain.model_validate({"nodes": [START.copy(), END.copy()]})

    monkeypatch.setattr(irc_module, "optimize_ts_and_irc", fake_optimize)
    msmep = _msmep(max_steps=60, do_elem_step_checks=False)
    neb, results = msmep.run_minimize_chain(_chain(msmep))

    assert results.is_elem_step
    assert len(calls) == 1
    assert len(neb.chain_trajectory) == 6  # initial chain + 5 steps
    assert neb.validated_ts is not None


def test_neb_keeps_optimizing_while_the_irc_misses_the_endpoints(monkeypatch):
    elsewhere = _node([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2], [0.0, 0.0, 3.5]])
    monkeypatch.setattr(
        irc_module, "optimize_ts_and_irc",
        lambda engine, guess: (guess, Chain.model_validate({"nodes": [elsewhere, elsewhere.copy()]})),
    )
    msmep = _msmep(max_steps=25, do_elem_step_checks=False)
    neb = msmep._construct_path_minimizer(msmep._create_interpolation(_chain(msmep)))
    try:
        neb.optimize_chain()
    except Exception:
        pass
    assert getattr(neb, "validated_ts", None) is None
    assert len(neb.chain_trajectory) > 6


def test_validated_ts_survives_the_tree_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        irc_module, "optimize_ts_and_irc",
        lambda engine, guess: (guess, Chain.model_validate({"nodes": [START.copy(), END.copy()]})),
    )
    msmep = _msmep(max_steps=60, do_elem_step_checks=False)
    tree = msmep.run_recursive_minimize(_chain(msmep))
    tree.write_to_disk(tmp_path / "tree")

    back = TreeNode.read_from_disk(tmp_path / "tree")
    assert [n.index for n in back.depth_first_ordered_nodes] == [0]
    (label, guess), = _ts_guess_tasks_from_tree(back, "ts_")
    ts_node, irc_chain = guess.validated_ts_irc
    assert len(irc_chain) == 2
    assert np.allclose(ts_node.coords, tree.data.validated_ts.coords, atol=1e-6)
