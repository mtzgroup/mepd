from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qcconst.constants import HARTREE_TO_KCAL_PER_MOL

from mepd.engines.engine import Engine
from mepd.discovery.hessian_sample import (
    _boltzmann_acceptance_probability,
    _candidate_acceptance_draw,
    run_hessian_global_optimization,
)
from mepd.inputs import ChainInputs
from mepd.nodes.node import XYNode

_HARTREE_TO_KCAL = float(HARTREE_TO_KCAL_PER_MOL)


def _kcal_to_hartree(kcal: float) -> float:
    return kcal / _HARTREE_TO_KCAL


class _FakeMultiWellEngine(Engine):
    """A deterministic multi-well toy engine for exercising the basin-hopping
    round loop without a real quantum-chemistry backend. A single Hessian
    mode ([1, 0]) drives two displaced candidates (+/-) per source; instead
    of real geometry mechanics, `compute_geometry_optimization` "snaps" a
    displaced candidate to one of a few fixed wells based on which well
    neighborhood the source sat in and which direction it was displaced --
    giving full, deterministic control over which wells are reachable from
    which other wells, across rounds.

    Wells (kcal/mol relative to the start well A, at x=0):
      A (x=0, seed): 0.0 kcal -- the starting structure
      B (x=10): -1.0 kcal -- downhill, reachable from A
      C (x=-10): +100.0 kcal -- a huge uphill dead end, reachable from A
      D (x=20): -2.0 kcal -- further downhill, reachable only from B
    """

    WELLS = {
        "A": (0.0, 0.0),
        "B": (10.0, _kcal_to_hartree(-1.0)),
        "C": (-10.0, _kcal_to_hartree(100.0)),
        "D": (20.0, _kcal_to_hartree(-2.0)),
    }

    def compute_energies(self, nodes):
        return [n._cached_energy if n._cached_energy is not None else 0.0 for n in nodes]

    def compute_gradients(self, nodes):
        raise NotImplementedError

    def _compute_hessian_result(self, node, **kwargs):
        modes = [np.array([1.0, 0.0])]
        freqs = [100.0]
        return SimpleNamespace(
            results=SimpleNamespace(normal_modes_cartesian=modes, freqs_wavenumber=freqs)
        )

    def compute_geometry_optimization(self, node, keywords=None):
        x = float(node.coords[0])
        if abs(x) < 5:
            target = "B" if x > 0 else "C"
        elif abs(x - 10) < 5:
            target = "D" if x > 10 else "A"
        else:
            target = "D"  # D is a local well: both displacement directions fall back into it
        well_x, energy = self.WELLS[target]
        return [XYNode(structure=np.array([well_x, 0.0]), _cached_energy=energy)]


def _seed():
    return XYNode(structure=np.array([0.0, 0.0]), _cached_energy=0.0)


def test_boltzmann_acceptance_probability_matches_metropolis_criterion():
    # non-uphill is always accepted
    assert _boltzmann_acceptance_probability(0.0, 298.15) == 1.0
    assert _boltzmann_acceptance_probability(-5.0, 298.15) == 1.0
    # uphill probability decreases with delta, increases with temperature
    p_small = _boltzmann_acceptance_probability(1.0, 298.15)
    p_large = _boltzmann_acceptance_probability(10.0, 298.15)
    assert 0.0 < p_large < p_small < 1.0
    p_hot = _boltzmann_acceptance_probability(1.0, 1000.0)
    assert p_hot > p_small


def test_candidate_acceptance_draw_is_deterministic_given_a_seed():
    draw_1 = _candidate_acceptance_draw(42, 0)
    draw_2 = _candidate_acceptance_draw(42, 0)
    assert draw_1 == draw_2
    assert 0.0 <= draw_1 < 1.0
    # different generation index (or seed) should (almost always) differ
    assert _candidate_acceptance_draw(42, 1) != draw_1
    assert _candidate_acceptance_draw(7, 0) != draw_1


def test_run_hessian_global_optimization_expands_downhill_and_rejects_huge_uphill():
    """From A: B (downhill) is accepted, C (+100 kcal/mol uphill) is
    rejected regardless of random seed (its Boltzmann probability is
    effectively zero at any reasonable temperature). From B: D (further
    downhill) is accepted; rediscovering A (isoenergetic with the seed) is
    rejected as a duplicate of the seed itself, not re-queued forever."""
    engine = _FakeMultiWellEngine()

    result = run_hessian_global_optimization(
        _seed(), engine, dr=0.5, max_candidates=2, temperature=298.15,
        energy_tolerance_kcal=1.0e-4, max_rounds=5, random_seed=123,
        chain_inputs=ChainInputs(node_rms_thre=1.0, node_ene_thre=1.0),
    )

    assert result.stopped_reason == "queue_exhausted"
    assert result.rounds_run == 3  # A->B (round 0), B->D (round 1), D self-loop rejected (round 2)
    accepted_x = sorted(float(n.coords[0]) for n in result.accepted_minima)
    assert accepted_x == [10.0, 20.0]  # B and D; never C, never a re-added A
    accepted_energies_kcal = sorted(
        (float(n.energy) - result.start_energy) * _HARTREE_TO_KCAL for n in result.accepted_minima
    )
    assert accepted_energies_kcal == pytest.approx([-2.0, -1.0], abs=1e-6)


def test_run_hessian_global_optimization_stops_at_max_rounds_if_reached_first():
    engine = _FakeMultiWellEngine()

    result = run_hessian_global_optimization(
        _seed(), engine, dr=0.5, max_candidates=2, temperature=298.15,
        max_rounds=1, random_seed=123,
        chain_inputs=ChainInputs(node_rms_thre=1.0, node_ene_thre=1.0),
    )

    assert result.stopped_reason == "max_rounds"
    assert result.rounds_run == 1
    assert len(result.round_summaries) == 1
    # only B should be found in a single round (D requires a second round from B)
    assert [round(float(n.coords[0]), 1) for n in result.accepted_minima] == [10.0]


def test_run_hessian_global_optimization_emits_minimum_accepted_events_live():
    """`on_event("minimum_accepted", ...)` must fire the instant each minimum
    is accepted -- not just be derivable from the final result -- so a caller
    can stream/write results out without waiting for the whole search."""
    engine = _FakeMultiWellEngine()
    events = []

    result = run_hessian_global_optimization(
        _seed(), engine, dr=0.5, max_candidates=2, temperature=298.15,
        energy_tolerance_kcal=1.0e-4, max_rounds=5, random_seed=123,
        chain_inputs=ChainInputs(node_rms_thre=1.0, node_ene_thre=1.0),
        on_event=lambda event, payload: events.append((event, payload)),
    )

    accepted_events = [payload for event, payload in events if event == "minimum_accepted"]
    assert len(accepted_events) == len(result.accepted_minima) == 2

    # Fired in acceptance order, 1-indexed, each carrying the actual node and
    # its energy relative to the seed (not the clipped/tolerance-adjusted value).
    assert [e["index"] for e in accepted_events] == [1, 2]
    assert [round(float(e["node"].coords[0]), 1) for e in accepted_events] == [10.0, 20.0]
    assert [round(e["rel_energy_kcal"], 6) for e in accepted_events] == pytest.approx(
        [-1.0, -2.0], abs=1e-6,
    )
    assert accepted_events[0]["round"] == 0
    assert accepted_events[1]["round"] == 1


def test_run_hessian_global_optimization_forwards_dr_values_to_each_round(monkeypatch):
    """--full-dr-scan should reach `run_hessian_sample` every round, not just
    the first -- and `dr` should be ignored while it does."""
    import mepd.discovery.hessian_sample as hessian_sample_module

    seen_dr_values = []
    original = hessian_sample_module.run_hessian_sample

    def spy(*args, **kwargs):
        seen_dr_values.append(kwargs.get("dr_values"))
        return original(*args, **kwargs)

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", spy)

    engine = _FakeMultiWellEngine()
    result = run_hessian_global_optimization(
        _seed(), engine, dr=999.0, dr_values=[0.5], max_candidates=2, temperature=298.15,
        max_rounds=5, random_seed=123,
        chain_inputs=ChainInputs(node_rms_thre=1.0, node_ene_thre=1.0),
    )

    assert result.rounds_run >= 1
    assert seen_dr_values  # at least one round ran
    assert all(dr_values == [0.5] for dr_values in seen_dr_values)


def test_run_hessian_global_optimization_rejects_nonpositive_temperature():
    engine = _FakeMultiWellEngine()
    with pytest.raises(ValueError):
        run_hessian_global_optimization(_seed(), engine, temperature=0.0)


def test_run_hessian_global_optimization_rejects_nonpositive_max_rounds():
    engine = _FakeMultiWellEngine()
    with pytest.raises(ValueError):
        run_hessian_global_optimization(_seed(), engine, max_rounds=0)
