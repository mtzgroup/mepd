from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mepd.engines.engine import Engine
from mepd.hessian_sample import run_hessian_sample
from mepd.nodes.node import XYNode


class _FakeHessianSampleEngine(Engine):
    """Deterministic toy engine for exercising `run_hessian_sample` without a
    real quantum-chemistry backend. Two normal modes (each sampled +/-) drive
    six displaced candidates; `_outcomes` maps them, in generation order, to
    canned optimization results covering every branch of the accept/reject
    logic.
    """

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self._call_index = 0

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
        outcome = self._outcomes[self._call_index]
        self._call_index += 1
        if outcome == "fail":
            raise RuntimeError("mock optimization failure")
        coords, energy = outcome
        return [XYNode(structure=np.array(coords, dtype=float), _cached_energy=energy)]


def _seed():
    return XYNode(structure=np.array([0.0, 0.0]), _cached_energy=0.0)


def test_run_hessian_sample_classifies_every_candidate():
    # generation order: +mode_a, -mode_a, +mode_b, -mode_b, +mode_c, -mode_c
    outcomes = [
        ((10.0, 10.0), -1.0),   # new minimum A: lower energy, distinct -> accepted
        ((0.0, 0.0), -0.0001),  # optimizes back to the seed (still lower-energy) -> duplicate of seed
        ((10.0001, 10.0001), -1.0),  # duplicates minimum A -> duplicate of accepted minimum
        "fail",                 # optimization raises -> failed
        ((-10.0, -10.0), -2.0),  # new minimum B: lower energy, distinct -> accepted
        ((20.0, 20.0), 5.0),    # higher energy than seed -> rejected
    ]
    engine = _FakeHessianSampleEngine(outcomes)

    result = run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=100)

    assert result.seed_energy == 0.0
    assert result.candidates_generated == 6
    assert result.candidates_clipped is False
    assert result.skipped_failed_optimization == 1
    assert result.skipped_not_lower_energy == 1
    assert result.skipped_duplicate == 2
    assert len(result.minima) == 2
    energies = sorted(m.energy for m in result.minima)
    assert energies == [-2.0, -1.0]


def test_run_hessian_sample_caps_at_max_candidates():
    outcomes = ["fail"] * 6
    engine = _FakeHessianSampleEngine(outcomes)

    result = run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=2)

    assert result.candidates_generated == 2
    assert result.candidates_clipped is True
    assert result.skipped_failed_optimization == 2
    assert result.minima == []


def test_run_hessian_sample_rejects_nonpositive_dr():
    engine = _FakeHessianSampleEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr=0.0, max_candidates=10)


def test_run_hessian_sample_rejects_nonpositive_max_candidates():
    engine = _FakeHessianSampleEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=0)


def test_run_hessian_sample_requires_normal_modes():
    class _NoModesEngine(_FakeHessianSampleEngine):
        def _compute_hessian_result(self, node, **kwargs):
            return SimpleNamespace(
                results=SimpleNamespace(normal_modes_cartesian=[], freqs_wavenumber=[])
            )

    engine = _NoModesEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=10)
