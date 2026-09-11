from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qcdata import Structure

from mepd.chain import Chain
from mepd.elementarystep import ElemStepResults
from mepd.engines.engine import Engine
from mepd.inputs import ChainInputs, RunInputs
from mepd.msmep import DEFAULT_CONSECUTIVE_SAME_PAIR_SPLIT_LIMIT, MSMEP
from mepd.nodes.node import StructureNode


class _FakeMsmepEngine(Engine):
    """A trivial engine just so run_recursive_minimize's preliminary
    gradient/energy calls (unrelated to the same-pair-split-limit logic
    under test) do not require a real quantum-chemistry backend."""

    def compute_energies(self, chain):
        return [0.0 for _ in chain]

    def compute_gradients(self, chain):
        return [np.zeros_like(n.coords) for n in chain]


def _fake_elem_step_result(new_structures) -> ElemStepResults:
    return ElemStepResults(
        is_elem_step=False,
        is_concave=True,
        splitting_criterion="maxima",
        minimization_results=[],
        number_grad_calls=0,
        new_structures=new_structures,
    )


def _topology_structure(kind: str, offset: float = 0.0) -> Structure:
    # "a": three atoms bonded in a row. "b": the second bond stretched far
    # enough to read as broken -- a real connectivity change, not a conformer
    # change, so graph-based species matching has real work to do.
    if kind == "a":
        geometry = np.array(
            [[0.0, 0.0, 0.0], [1.20 + offset, 0.0, 0.0], [2.16 + (2 * offset), 0.0, 0.0]]
        )
    elif kind == "b":
        geometry = np.array(
            [[0.0, 0.0, 0.0], [1.20 + offset, 0.0, 0.0], [5.20 + offset, 0.0, 0.0]]
        )
    else:
        raise ValueError(kind)
    return Structure(geometry=geometry, symbols=["C", "O", "H"], charge=0, multiplicity=1)


def _chain_from(structure_start: Structure, structure_end: Structure) -> Chain:
    return Chain.model_validate({
        "nodes": [StructureNode(structure=structure_start), StructureNode(structure=structure_end)],
        "parameters": ChainInputs(),
    })


def _msmep() -> MSMEP:
    return MSMEP(inputs=RunInputs(engine_name="gxtb", gxtb_engine_kwds={"executable": "gxtb"}))


def test_same_pair_split_limit_only_counts_the_branch_that_actually_repeats():
    """The core correctness property behind 'skip just that branch, not the
    whole tree': _set_child_same_pair_split_lineage() blindly stamps the same
    lineage dict onto every child of a split, but _consecutive_same_pair_split_count()
    re-validates each child's *own* endpoints against that lineage before
    counting it -- so a sibling with genuinely different endpoints reads 0
    regardless of what a same-pair sibling reads.
    """
    msmep = _msmep()
    parent = _chain_from(_topology_structure("a"), _topology_structure("a", offset=6.0))

    same_pair_child = _chain_from(_topology_structure("a"), _topology_structure("a", offset=6.0))
    different_pair_child = _chain_from(_topology_structure("a"), _topology_structure("b"))

    msmep._set_child_same_pair_split_lineage(
        [same_pair_child, different_pair_child], parent, count=3
    )

    assert msmep._consecutive_same_pair_split_count(same_pair_child) == 3
    assert msmep._consecutive_same_pair_split_count(different_pair_child) == 0


def test_same_pair_split_limit_trips_only_after_repeated_generations():
    """Simulates a branch that keeps re-splitting into the exact same
    endpoint pair, generation after generation, and confirms the limit is
    reached only once the count actually accumulates that high -- not on the
    first repeat, and not for a chain that was never part of that lineage."""
    msmep = _msmep()
    limit = msmep._resolve_same_pair_split_limit()
    assert limit == DEFAULT_CONSECUTIVE_SAME_PAIR_SPLIT_LIMIT

    branch = _chain_from(_topology_structure("a"), _topology_structure("a", offset=6.0))
    assert msmep._consecutive_same_pair_split_count(branch) == 0  # fresh chain, no lineage yet

    for generation in range(1, limit + 1):
        next_branch = _chain_from(_topology_structure("a"), _topology_structure("a", offset=6.0))
        msmep._set_child_same_pair_split_lineage([next_branch], branch, count=generation)
        branch = next_branch
        count = msmep._consecutive_same_pair_split_count(branch)
        assert count == generation
        if generation < limit:
            assert count < limit
    assert msmep._consecutive_same_pair_split_count(branch) >= limit

    # An entirely unrelated chain (never produced by that lineage) must read
    # 0 regardless of how deep the repeating branch's count got -- this is
    # the actual "does not affect the rest of the tree" guarantee.
    unrelated = _chain_from(_topology_structure("b"), _topology_structure("a"))
    assert msmep._consecutive_same_pair_split_count(unrelated) == 0


def test_same_pair_split_limit_does_not_discard_a_genuine_new_structure_discovery(monkeypatch):
    """The actual reported bug: previously, once a branch's inherited
    same-pair count reached the limit, ANY further split was discarded --
    even one that genuinely discovered new chemistry this generation --
    silently dropping a real find. A split that reports new_structures must
    now bypass the stop."""
    msmep = _msmep()
    msmep.inputs.engine = _FakeMsmepEngine()
    limit = msmep._resolve_same_pair_split_limit()

    seed_chain = _chain_from(_topology_structure("a"), _topology_structure("b"))
    at_limit_chain = _chain_from(_topology_structure("a"), _topology_structure("b"))
    msmep._set_child_same_pair_split_lineage([at_limit_chain], seed_chain, count=limit)
    assert msmep._consecutive_same_pair_split_count(at_limit_chain) == limit

    fake_neb = SimpleNamespace(chain_trajectory=[at_limit_chain])
    new_structure_node = StructureNode(structure=_topology_structure("b", offset=1.0))
    child_a = _chain_from(_topology_structure("a"), new_structure_node.structure)
    child_b = _chain_from(new_structure_node.structure, _topology_structure("b"))

    monkeypatch.setattr(
        msmep, "run_minimize_chain",
        lambda input_chain: (fake_neb, _fake_elem_step_result([new_structure_node])),
    )
    monkeypatch.setattr(
        msmep, "make_sequence_of_chains",
        lambda chain, split_method, minimization_results: [child_a, child_b],
    )

    history = msmep.run_recursive_minimize(at_limit_chain, max_depth=1)

    assert getattr(history, "leaf_status", None) != "same_pair_split_limit_reached"
    assert len(history.children) == 2  # the split proceeded -- not discarded


def test_same_pair_split_limit_still_stops_when_nothing_new_is_found(monkeypatch):
    """Contrast case: with no new_structures this generation (genuinely
    unproductive repetition), the limit still fires exactly as before."""
    msmep = _msmep()
    msmep.inputs.engine = _FakeMsmepEngine()
    limit = msmep._resolve_same_pair_split_limit()

    seed_chain = _chain_from(_topology_structure("a"), _topology_structure("b"))
    at_limit_chain = _chain_from(_topology_structure("a"), _topology_structure("b"))
    msmep._set_child_same_pair_split_lineage([at_limit_chain], seed_chain, count=limit)

    fake_neb = SimpleNamespace(chain_trajectory=[at_limit_chain])
    monkeypatch.setattr(
        msmep, "run_minimize_chain",
        lambda input_chain: (fake_neb, _fake_elem_step_result([])),
    )

    history = msmep.run_recursive_minimize(at_limit_chain, max_depth=1)

    assert getattr(history, "leaf_status", None) == "same_pair_split_limit_reached"
    assert history.children == []
