from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mepd.engines.engine import Engine
from mepd.hessian_sample import run_hessian_sample
from mepd.inputs import ChainInputs
from mepd.nodes.node import XYNode


class _FakeHessianSampleEngine(Engine):
    """Deterministic toy engine for exercising `run_hessian_sample` without a
    real quantum-chemistry backend. Three normal modes (each sampled +/-)
    drive six displaced candidates; `_outcomes` maps them, in generation
    order, to canned optimization results.
    """

    def __init__(self, outcomes, received_keywords=None):
        self._outcomes = list(outcomes)
        self._call_index = 0
        self._received_keywords = received_keywords

    def compute_energies(self, nodes):
        return [n._cached_energy if n._cached_energy is not None else 0.0 for n in nodes]

    def compute_gradients(self, nodes):
        raise NotImplementedError

    def _compute_hessian_result(self, node, **kwargs):
        modes = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        freqs = [100.0, 150.0, 200.0]
        return SimpleNamespace(
            results=SimpleNamespace(normal_modes_cartesian=modes, freqs_wavenumber=freqs)
        )

    def compute_geometry_optimization(self, node, keywords=None):
        if self._received_keywords is not None:
            self._received_keywords.append(keywords)
        outcome = self._outcomes[self._call_index]
        self._call_index += 1
        if outcome == "fail":
            raise RuntimeError("mock optimization failure")
        coords, energy = outcome
        return [XYNode(structure=np.array(coords, dtype=float), _cached_energy=energy)]


class _FakeBatchEngine(_FakeHessianSampleEngine):
    """Same as above, but exposes a batch `compute_geometry_optimizations`."""

    def __init__(self, outcomes):
        super().__init__(outcomes)
        self.batch_calls = 0

    def compute_geometry_optimizations(self, nodes, keywords=None):
        self.batch_calls += 1
        trajectories = []
        for _ in nodes:
            outcome = self._outcomes[self._call_index]
            self._call_index += 1
            if outcome == "fail":
                trajectories.append([])
            else:
                coords, energy = outcome
                trajectories.append(
                    [XYNode(structure=np.array(coords, dtype=float), _cached_energy=energy)]
                )
        return trajectories


def _seed():
    return XYNode(structure=np.array([0.0, 0.0]), _cached_energy=0.0)


# generation order for the 6-candidate outcome lists below:
# +mode_a, -mode_a, +mode_b, -mode_b, +mode_c, -mode_c
_MIXED_OUTCOMES = [
    ((10.0, 10.0), -1.0),        # new minimum A
    ((0.0, 0.0), -0.0001),       # re-finds the seed geometry -- still "optimized", not filtered
    ((10.0001, 10.0001), -1.0),  # duplicates A
    "fail",                      # optimization raises
    ((-10.0, -10.0), -2.0),      # new minimum B
    ((20.0, 20.0), 5.0),         # higher energy than seed -- kept, not filtered by energy
]


def test_run_hessian_sample_reports_every_optimized_candidate_deduped():
    engine = _FakeHessianSampleEngine(_MIXED_OUTCOMES)

    result = run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=100)

    assert result.seed_energy == 0.0
    assert len(result.displaced_nodes) == 6
    assert result.candidates_clipped is False
    assert result.optimization_submission_mode == "serial"
    assert len(result.failed_candidates) == 1
    # 5 succeeded, one pair (A and its near-duplicate) collapses -> 4 unique
    assert len(result.optimized_nodes) == 5
    assert len(result.unique_minima) == 4
    energies = sorted(round(m.energy, 4) for m in result.unique_minima)
    assert energies == [-2.0, -1.0, -0.0001, 5.0]


def test_run_hessian_sample_passes_maxiter_through_keywords():
    received = []
    engine = _FakeHessianSampleEngine(["fail"] * 6, received_keywords=received)

    run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=6, maxiter=42)

    assert all(kw == {"coordsys": "cart", "maxiter": 42} for kw in received)


def test_run_hessian_sample_uses_batch_optimizer_when_available():
    engine = _FakeBatchEngine(_MIXED_OUTCOMES)

    result = run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=100)

    assert engine.batch_calls == 1
    assert result.optimization_submission_mode == "batch"
    assert len(result.optimized_nodes) == 5
    assert len(result.unique_minima) == 4


def test_run_hessian_sample_dedup_uses_given_chain_inputs_thresholds():
    # Two candidates 2.0 bohr apart: duplicates under a loose 5.0 threshold
    # (the ChainInputs default), distinct under a tight 0.1 threshold.
    outcomes = [
        ((0.0, 0.0), 0.0),
        ((2.0, 0.0), 0.0),
    ] + ["fail"] * 4

    loose = run_hessian_sample(
        _seed(), _FakeHessianSampleEngine(outcomes), dr=1.0, max_candidates=2,
        chain_inputs=ChainInputs(node_rms_thre=5.0, node_ene_thre=5.0),
    )
    assert len(loose.unique_minima) == 1

    tight = run_hessian_sample(
        _seed(), _FakeHessianSampleEngine(outcomes), dr=1.0, max_candidates=2,
        chain_inputs=ChainInputs(node_rms_thre=0.1, node_ene_thre=5.0),
    )
    assert len(tight.unique_minima) == 2


def test_run_hessian_sample_caps_at_max_candidates():
    engine = _FakeHessianSampleEngine(["fail"] * 6)

    result = run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=2)

    assert len(result.displaced_nodes) == 2
    assert result.candidates_clipped is True
    assert len(result.failed_candidates) == 2
    assert result.unique_minima == []


def test_run_hessian_sample_rejects_nonpositive_dr():
    engine = _FakeHessianSampleEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr=0.0, max_candidates=10)


def test_run_hessian_sample_rejects_nonpositive_max_candidates():
    engine = _FakeHessianSampleEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=0)


def test_run_hessian_sample_rejects_nonpositive_maxiter():
    engine = _FakeHessianSampleEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=10, maxiter=0)


def test_run_hessian_sample_requires_normal_modes():
    class _NoModesEngine(_FakeHessianSampleEngine):
        def _compute_hessian_result(self, node, **kwargs):
            return SimpleNamespace(
                results=SimpleNamespace(normal_modes_cartesian=[], freqs_wavenumber=[])
            )

    engine = _NoModesEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=10)
