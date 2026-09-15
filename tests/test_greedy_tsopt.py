from __future__ import annotations

import numpy as np
import pytest
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs, NEBInputs
from mepd.neb import NEB
from mepd.nodes.node import StructureNode
from mepd.optimizers.vpo import VelocityProjectedOptimizer
from mepd.TreeNode import TreeNode


def _water(x_offset: float = 0.0) -> Structure:
    return Structure(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.43355001758932 + x_offset, 0.0, 0.95295864902809],
                [-1.43355001758932, 0.0, 0.95295864902809],
            ],
            dtype=float,
        ),
        charge=0,
        multiplicity=1,
    )


def _water_node(x_offset: float = 0.0, energy: float | None = None) -> StructureNode:
    node = StructureNode(structure=_water(x_offset))
    if energy is not None:
        node._cached_energy = energy
        node._cached_gradient = np.zeros((3, 3))
    return node


def _make_chain(energies, x_offset_step: float = 0.05) -> Chain:
    nodes = [_water_node(x_offset_step * i, e) for i, e in enumerate(energies)]
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def _make_neb(trajectory_energies: list[list[float]], x_offset_step: float = 0.05) -> NEB:
    trajectory = [_make_chain(es, x_offset_step) for es in trajectory_energies]
    return NEB(
        initial_chain=trajectory[0],
        optimizer=VelocityProjectedOptimizer(),
        parameters=NEBInputs(),
        engine=None,
        optimized=trajectory[-1],
        chain_trajectory=trajectory,
    )


class _FakeEngine:
    def __init__(self, irc_chain: Chain | None = None):
        self.ts_calls = []
        self.irc_chain = irc_chain

    def compute_transition_state(self, node, keywords=None):
        self.ts_calls.append(node)
        return node

    def compute_irc_chain(self, ts_node, keywords=None):
        if self.irc_chain is not None:
            return self.irc_chain
        return Chain.model_validate({
            "nodes": [ts_node, ts_node.copy()],
            "parameters": ChainInputs(),
        })


def test_greedy_tsopt_attempts_every_history_entry():
    root_neb = _make_neb([[-1.0, -0.5, -1.1], [-1.0, -0.6, -1.2]], x_offset_step=0.05)
    child_neb = _make_neb([[-2.0, -1.5, -2.1]], x_offset_step=0.3)
    tree = TreeNode(data=root_neb, children=[TreeNode(data=child_neb, children=[], index=1)], index=0)

    engine = _FakeEngine()
    candidates = tree.greedy_tsopt(engine, dedup=False)

    assert len(candidates) == 2
    assert all(c.result.ok for c in candidates)
    assert candidates[0].source_index == 0
    assert candidates[1].source_index == 1
    assert len(engine.ts_calls) == 2


def test_greedy_tsopt_dedups_geometrically_identical_guesses():
    energies = [-1.0, -0.5, -1.1]
    root_neb = _make_neb([energies])
    # Same trajectory content -> identical TS guess node -> should be skipped
    # by dedup even though it comes from a different NEB/tree node.
    child_neb = _make_neb([energies])
    tree = TreeNode(data=root_neb, children=[TreeNode(data=child_neb, children=[], index=1)], index=0)

    engine = _FakeEngine()
    candidates = tree.greedy_tsopt(engine, dedup=True)

    assert len(candidates) == 1
    assert len(engine.ts_calls) == 1


def test_greedy_tsopt_skips_entries_with_no_chain_trajectory():
    root_neb = _make_neb([[-1.0, -0.5, -1.1]])
    empty_child = NEB(
        initial_chain=root_neb.chain_trajectory[0],
        optimizer=VelocityProjectedOptimizer(),
        parameters=NEBInputs(),
        engine=None,
    )
    tree = TreeNode(
        data=root_neb,
        children=[TreeNode(data=empty_child, children=[], index=1)],
        index=0,
    )

    engine = _FakeEngine()
    candidates = tree.greedy_tsopt(engine, dedup=False)

    assert len(candidates) == 1
    assert candidates[0].source_index == 0


def test_greedy_tsopt_with_irc_populates_irc_chain():
    root_neb = _make_neb([[-1.0, -0.5, -1.1]])
    tree = TreeNode(data=root_neb, children=[], index=0)

    engine = _FakeEngine()
    candidates = tree.greedy_tsopt(engine, run_irc=True, dedup=False)

    assert len(candidates) == 1
    result = candidates[0].result
    assert result.ok
    assert result.irc_chain is not None


def test_greedy_tsopt_records_failure_without_stopping_other_candidates():
    root_neb = _make_neb([[-1.0, -0.5, -1.1]], x_offset_step=0.05)
    child_neb = _make_neb([[-2.0, -1.5, -2.1]], x_offset_step=0.3)
    tree = TreeNode(data=root_neb, children=[TreeNode(data=child_neb, children=[], index=1)], index=0)

    class _FlakyEngine(_FakeEngine):
        def compute_transition_state(self, node, keywords=None):
            self.ts_calls.append(node)
            if len(self.ts_calls) == 1:
                raise RuntimeError("boom")
            return node

    engine = _FlakyEngine()
    candidates = tree.greedy_tsopt(engine, dedup=False)

    assert len(candidates) == 2
    assert not candidates[0].result.ok
    assert "boom" in candidates[0].result.error
    assert candidates[1].result.ok
