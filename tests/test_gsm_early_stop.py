from __future__ import annotations

import numpy as np
from qcdata import Structure

from mepd.chain import Chain
from mepd.nodes.node import StructureNode
from mepd.pathminimizers.gsm import GSM

NNODES_TARGET = 7
WINDOW = 3
RTOL = 0.02
MIN_DEPTH_KCAL = 1.0


def _chain(energies_hartree: list[float]) -> Chain:
    """A toy chain of the given length with hand-set energies; geometry is
    irrelevant to _track_persistent_minima, just needs to be valid/distinct
    per node."""
    nodes = []
    for i, e in enumerate(energies_hartree):
        node = StructureNode(
            structure=Structure(
                symbols=["H", "C", "N", "O", "H"],
                geometry=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0 + i * 0.1],
                        [0.0, 0.0, 2.0 + i * 0.1],
                        [0.0, 0.0, 3.0 + i * 0.1],
                        [0.0, 0.0, 4.0 + i * 0.1],
                    ],
                    dtype=float,
                ),
                charge=0,
                multiplicity=1,
            )
        )
        node._cached_energy = e
        nodes.append(node)
    return Chain.model_validate({"nodes": nodes, "parameters": {}})


def _fresh_state() -> dict:
    return {"index": None, "energy_kcal": None, "count": 0}


def _track(chain: Chain, state: dict) -> bool:
    return GSM._track_persistent_minima(
        chain, NNODES_TARGET, WINDOW, RTOL, MIN_DEPTH_KCAL, state
    )


def test_never_triggers_while_still_growing():
    """A too-short chain (growth not yet finished) never counts as signal,
    however stable its own profile looks."""
    state = _fresh_state()
    partial = _chain([0.0, -0.01, -0.005])
    assert _track(partial, state) is False
    assert state["count"] == 0


def test_stable_deep_minimum_triggers_after_window():
    """A genuine, stable local minimum triggers exactly on the `window`-th
    consecutive matching poll, not before. The very first poll of any
    sequence always reads as "not yet stable" too -- there's no prior
    length recorded to compare against yet -- so this actually takes
    `window + 1` polls in total, not `window`."""
    state = _fresh_state()
    w_profile = [0.0, 0.02, 0.005, 0.025, 0.03, 0.02, -0.002]  # dip at index 2
    results = [_track(_chain(w_profile), state) for _ in range(WINDOW + 2)]
    assert results == [False] * WINDOW + [True] * 2


def test_shallow_noise_level_minimum_never_triggers():
    """A "minimum" a fraction of a kcal/mol deep -- numerical noise, not a
    real intermediate -- never accumulates persistence, however long it
    looks stable for. Regression test: an earlier version without this
    depth gate triggered on exactly this kind of profile on the real
    oxy-Cope rearrangement, on a dip ~0.01 kcal/mol deep near the reactant."""
    state = _fresh_state()
    shallow = [0.0, 0.02, 0.0199, 0.021, 0.03, 0.02, -0.002]
    for _ in range(5):
        assert _track(_chain(shallow), state) is False
    assert state["count"] == 0


def test_single_barrier_no_interior_minimum_never_triggers():
    state = _fresh_state()
    single_barrier = [0.0, 0.02, 0.04, 0.06, 0.04, 0.02, -0.002]
    for _ in range(4):
        assert _track(_chain(single_barrier), state) is False


def test_drifting_minimum_never_stabilizes():
    """A minimum whose depth keeps changing by more than rtol never
    persists, even though a (shallower or deeper) minimum exists at every
    single poll."""
    state = _fresh_state()
    drift_profiles = [
        [0.0, 0.02, 0.005, 0.025, 0.03, 0.02, -0.002],
        [0.0, 0.02, 0.015, 0.025, 0.03, 0.02, -0.002],
        [0.0, 0.02, 0.010, 0.025, 0.03, 0.02, -0.002],
        [0.0, 0.02, 0.006, 0.025, 0.03, 0.02, -0.002],
    ]
    for prof in drift_profiles:
        assert _track(_chain(prof), state) is False
    assert state["count"] == 1


def test_shifting_index_same_depth_still_triggers():
    """A minimum that GSM's own reparametrization moves to a different node
    index between polls still accumulates persistence, as long as its depth
    stays stable -- tracking is by energy value, not node index. (The first
    poll always reads as "not yet stable" regardless, same as above.)"""
    state = _fresh_state()
    profiles = [
        [0.0, 0.02, 0.005, 0.025, 0.03, 0.02, -0.002],  # min at index 2
        [0.0, 0.021, 0.024, 0.0051, 0.025, 0.02, -0.002],  # shifted to index 3
        [0.0, 0.021, 0.024, 0.028, 0.0049, 0.02, -0.002],  # shifted to index 4
        [0.0, 0.021, 0.024, 0.028, 0.0050, 0.02, -0.002],  # settles at index 4
    ]
    results = [_track(_chain(prof), state) for prof in profiles]
    assert results == [False, False, False, True]


def test_growth_length_change_resets_persistence():
    """A change in chain length (growth resuming) resets accumulated
    persistence, even if a stable-looking minimum had already been seen at
    the previous length."""
    state = _fresh_state()
    stable = [0.0, 0.02, 0.005, 0.025, 0.03, 0.02, -0.002]
    assert _track(_chain(stable), state) is False
    assert _track(_chain(stable), state) is False
    # Growth adds a node -- length changes, persistence must reset even
    # though the caller might not otherwise expect a regression.
    grown = [0.0, 0.02, 0.01, 0.005, 0.025, 0.03, 0.02, -0.002]
    assert _track(_chain(grown), state) is False
    assert state["count"] == 0
