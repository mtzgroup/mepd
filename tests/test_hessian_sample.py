from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from qcconst.constants import BOHR_TO_ANGSTROM
from qcdata import Structure

from mepd.engines.engine import Engine
from mepd.discovery.hessian_sample import _effective_dr
from mepd.discovery.hessian_sample import run_hessian_sample
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode, XYNode


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


class _FakeHessianSampleEngineWithHessian(_FakeHessianSampleEngine):
    """Same toy engine, but its Hessian result also carries a real (diagonal)
    Cartesian Hessian matrix -- so force constants (and thus the harmonic
    displacement-energy estimate) are actually computable, and one mode is
    given a negative (imaginary) frequency to exercise reaction-coordinate
    tagging."""

    def _compute_hessian_result(self, node, **kwargs):
        modes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        freqs = [100.0, -150.0]  # second mode: imaginary, k_b < 0
        hessian = np.array([[4.0, 0.0], [0.0, -9.0]])
        return SimpleNamespace(
            results=SimpleNamespace(
                normal_modes_cartesian=modes, freqs_wavenumber=freqs, hessian=hessian,
            )
        )


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


class _FakeBatchEngineThatRaisesMidBatch(_FakeHessianSampleEngine):
    """A "batch" optimizer that is really a sequential loop under the hood
    (like GXTBCalculator/QCComputeEngine's non-ChemCloud fallback) and lets
    one candidate's failure raise and abort the whole call, instead of
    isolating it like a true batch backend (e.g. ChemCloud) does. Exercises
    `run_hessian_sample`'s fallback to per-candidate serial optimization."""

    def compute_geometry_optimizations(self, nodes, keywords=None):
        raise RuntimeError("simulated non-convergence blew up the whole batch call")


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


def test_run_hessian_sample_tags_outcome_and_mode_provenance_on_metadata():
    engine = _FakeHessianSampleEngine(_MIXED_OUTCOMES)

    result = run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=100)

    # generation order: +mode_a (new A), -mode_a (re-finds seed),
    # +mode_b (duplicates A), -mode_b (fails), +mode_c (new B), -mode_c (new)
    assert [m.outcome for m in result.optimized_metadata] == [
        "new_minimum", "returned_to_seed_basin", "known_minimum",
        "new_minimum", "new_minimum",
    ]
    assert result.failed_candidates[0]["meta"].outcome == "opt_failed"

    # final_energy_kcal_rel_seed matches (candidate energy - seed energy) in kcal/mol
    hartree_to_kcal = 627.5094740630558
    for node, meta in zip(result.optimized_nodes, result.optimized_metadata):
        assert meta.final_energy_kcal_rel_seed == pytest.approx(
            node.energy * hartree_to_kcal, rel=1e-4
        )

    # The fake engine's Hessian result carries no Hessian matrix, so force
    # constants (and thus the harmonic displacement-energy estimate) aren't
    # available -- must be None, not a bogus number.
    assert all(m.displacement_energy_kcal is None for m in result.displaced_metadata)
    # All three fake modes are real (positive-frequency) vibrations.
    assert all(not m.is_reaction_coordinate for m in result.displaced_metadata)
    assert all(m.mode_source == "normal_modes_cartesian" for m in result.displaced_metadata)


def test_run_hessian_sample_computes_displacement_energy_and_reaction_coordinate_flag():
    """With a real Hessian matrix available, displacement_energy_kcal must
    be the exact harmonic estimate (1/2 k a^2, converted to kcal/mol), and
    the negative-frequency mode must be tagged as the reaction coordinate."""
    engine = _FakeHessianSampleEngineWithHessian(["fail"] * 4)

    result = run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=100)

    effective_dr = 1.0 * (2 ** 0.5)  # dr * sqrt(natoms), natoms=2 for this toy node
    hartree_to_kcal = 627.5094740630558
    # prepare_modes sorts ascending by frequency, so the imaginary mode
    # (freq=-150, k=-9.0) sorts to index 0 and the real one (freq=100,
    # k=+4.0) to index 1 -- the reverse of the engine's own mode order.
    expected_energy_kcal = {
        0: 0.5 * -9.0 * effective_dr**2 * hartree_to_kcal,  # imaginary mode, k=-9.0
        1: 0.5 * 4.0 * effective_dr**2 * hartree_to_kcal,   # real mode, k=+4.0
    }
    for meta in result.displaced_metadata:
        assert meta.displacement_energy_kcal == pytest.approx(
            expected_energy_kcal[meta.mode_index], rel=1e-6
        )
        assert meta.is_reaction_coordinate == (meta.mode_index == 0)


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


def _fake_structure_node(n_atoms: int) -> StructureNode:
    return StructureNode(structure=Structure(
        symbols=["H"] * n_atoms,
        geometry=np.zeros((n_atoms, 3)),
        charge=0, multiplicity=1,
    ))


def test_effective_dr_gives_size_invariant_per_atom_rms_displacement():
    """`displace_by_dr` normalizes the mode to unit norm across the full 3N
    Cartesian vector, so per-atom RMS displacement is `effective_dr /
    sqrt(N)`. `_effective_dr` must return `dr * sqrt(N)` (not `dr * N`) for
    that per-atom RMS to actually equal `dr`, independent of molecule size --
    otherwise larger molecules get systematically harder kicks, biasing
    novelty yield and failure rate by molecular size for anyone comparing
    across a size range."""
    dr = 0.1
    rms_per_atom_angstrom = {
        n: (_effective_dr(_fake_structure_node(n), dr) / (n**0.5)) * BOHR_TO_ANGSTROM
        for n in (5, 20, 60)
    }
    assert rms_per_atom_angstrom[5] == pytest.approx(rms_per_atom_angstrom[20])
    assert rms_per_atom_angstrom[20] == pytest.approx(rms_per_atom_angstrom[60])
    assert rms_per_atom_angstrom[20] == pytest.approx(dr * BOHR_TO_ANGSTROM)


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


def test_run_hessian_sample_falls_back_to_serial_when_batch_call_raises():
    engine = _FakeBatchEngineThatRaisesMidBatch(_MIXED_OUTCOMES)

    result = run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=100)

    assert result.optimization_submission_mode == "batch_fallback_serial"
    # Same outcomes as the plain-serial/plain-batch paths (test_hessian_sample.py
    # above): one candidate genuinely fails, the rest are isolated and still
    # produce results -- a batch call raising outright must not lose everyone
    # else in the batch.
    assert len(result.failed_candidates) == 1
    assert len(result.optimized_nodes) == 5
    assert len(result.unique_minima) == 4


def test_run_hessian_sample_dr_values_scans_every_value_independently():
    # 3 modes x 2 directions x 2 dr values = 12 candidates.
    engine = _FakeHessianSampleEngine(["fail"] * 12)

    result = run_hessian_sample(
        _seed(), engine, dr_values=[0.5, 1.0], max_candidates=100,
    )

    assert len(result.displaced_nodes) == 12
    assert result.candidates_clipped is False
    scan_indices = sorted(m.dr_scan_index for m in result.displaced_metadata)
    assert scan_indices == [0] * 6 + [1] * 6
    drs = {round(m.dr, 6) for m in result.displaced_metadata}
    assert drs == {0.5, 1.0}


def test_run_hessian_sample_dr_values_caps_max_candidates_per_scan_value():
    # max_candidates=2 should clip each of the 2 dr values independently,
    # for 4 total displaced candidates -- not a single shared cap of 2.
    engine = _FakeHessianSampleEngine(["fail"] * 4)

    result = run_hessian_sample(
        _seed(), engine, dr_values=[0.5, 1.0], max_candidates=2,
    )

    assert len(result.displaced_nodes) == 4
    assert result.candidates_clipped is True


def test_run_hessian_sample_dr_values_ignores_dr():
    engine_scan = _FakeHessianSampleEngine(["fail"] * 6)
    engine_single = _FakeHessianSampleEngine(["fail"] * 6)

    scanned = run_hessian_sample(_seed(), engine_scan, dr=999.0, dr_values=[0.5], max_candidates=100)
    single = run_hessian_sample(_seed(), engine_single, dr=0.5, max_candidates=100)

    assert [round(m.effective_dr, 6) for m in scanned.displaced_metadata] == [
        round(m.effective_dr, 6) for m in single.displaced_metadata
    ]


def test_run_hessian_sample_rejects_empty_or_nonpositive_dr_values():
    engine = _FakeHessianSampleEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr_values=[], max_candidates=10)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr_values=[0.5, -1.0], max_candidates=10)


def test_run_hessian_sample_rejects_invalid_amplitude_policy():
    engine = _FakeHessianSampleEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, amplitude_policy="nonsense", max_candidates=10)


def test_run_hessian_sample_rejects_dr_values_with_energy_policy():
    engine = _FakeHessianSampleEngineWithHessian(["fail"] * 4)
    with pytest.raises(ValueError):
        run_hessian_sample(
            _seed(), engine, amplitude_policy="energy", dr_values=[0.1, 0.2], max_candidates=10,
        )


def test_run_hessian_sample_rejects_target_energy_kcal_values_with_fixed_cartesian_policy():
    engine = _FakeHessianSampleEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(
            _seed(), engine, amplitude_policy="fixed_cartesian",
            target_energy_kcal_values=[5.0, 10.0], max_candidates=10,
        )


def test_run_hessian_sample_rejects_nonpositive_target_energy_kcal():
    engine = _FakeHessianSampleEngineWithHessian(["fail"] * 4)
    with pytest.raises(ValueError):
        run_hessian_sample(
            _seed(), engine, amplitude_policy="energy", target_energy_kcal=0.0, max_candidates=10,
        )


def test_run_hessian_sample_energy_policy_calibrates_amplitude_per_mode():
    """a_i = sqrt(2 * E_target / k_i) exactly, for a mode with a real,
    known force constant; the imaginary mode instead gets the fixed
    imaginary_mode_amplitude, since harmonic calibration is undefined for
    negative curvature."""
    engine = _FakeHessianSampleEngineWithHessian(["fail"] * 4)
    target_energy_kcal = 20.0
    imaginary_mode_amplitude = 0.42

    result = run_hessian_sample(
        _seed(), engine, amplitude_policy="energy", target_energy_kcal=target_energy_kcal,
        imaginary_mode_amplitude=imaginary_mode_amplitude, max_candidates=10,
    )

    hartree_to_kcal = 627.5094740630558
    target_hartree = target_energy_kcal / hartree_to_kcal
    # prepare_modes sorts the imaginary mode (freq=-150, k=-9.0) to index 0
    # and the real mode (freq=100, k=+4.0) to index 1.
    expected_effective_dr = {
        0: imaginary_mode_amplitude,
        1: (2.0 * target_hartree / 4.0) ** 0.5,
    }
    assert len(result.displaced_metadata) == 4  # 2 modes x 2 directions
    for meta in result.displaced_metadata:
        assert meta.effective_dr == pytest.approx(expected_effective_dr[meta.mode_index], rel=1e-9)
        assert meta.amplitude_policy == "energy"
        assert meta.target_energy_kcal == target_energy_kcal
        assert meta.dr is None


def test_run_hessian_sample_energy_policy_skips_near_zero_force_constant_modes():
    """A mode whose harmonic force constant is (numerically) zero would
    otherwise need an infinite amplitude to reach any nonzero target energy
    -- it must be skipped, not silently produce a divergent/garbage
    displacement."""

    class _FakeZeroForceConstantEngine(_FakeHessianSampleEngine):
        def _compute_hessian_result(self, node, **kwargs):
            from types import SimpleNamespace
            modes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
            freqs = [100.0, 150.0]
            hessian = np.array([[4.0, 0.0], [0.0, 0.0]])  # second mode: k=0
            return SimpleNamespace(
                results=SimpleNamespace(
                    normal_modes_cartesian=modes, freqs_wavenumber=freqs, hessian=hessian,
                )
            )

    engine = _FakeZeroForceConstantEngine(["fail"] * 2)
    result = run_hessian_sample(
        _seed(), engine, amplitude_policy="energy", target_energy_kcal=10.0, max_candidates=10,
    )

    # Only the k=4.0 mode produces candidates (2, one per direction); the
    # k=0 mode is skipped entirely.
    assert len(result.displaced_metadata) == 2
    assert all(m.mode_index == 0 for m in result.displaced_metadata)


def test_run_hessian_sample_energy_policy_forwards_target_energy_kcal_values_to_each_scan(monkeypatch):
    engine = _FakeHessianSampleEngineWithHessian(["fail"] * 8)

    result = run_hessian_sample(
        _seed(), engine, amplitude_policy="energy",
        target_energy_kcal_values=[5.0, 15.0], max_candidates=100,
    )

    # 2 modes x 2 directions x 2 scan values
    assert len(result.displaced_metadata) == 8
    scan_indices = sorted({m.dr_scan_index for m in result.displaced_metadata})
    assert scan_indices == [0, 1]
    targets = sorted({m.target_energy_kcal for m in result.displaced_metadata})
    assert targets == [5.0, 15.0]


def test_run_hessian_sample_requires_normal_modes():
    class _NoModesEngine(_FakeHessianSampleEngine):
        def _compute_hessian_result(self, node, **kwargs):
            return SimpleNamespace(
                results=SimpleNamespace(normal_modes_cartesian=[], freqs_wavenumber=[])
            )

    engine = _NoModesEngine(["fail"] * 6)
    with pytest.raises(ValueError):
        run_hessian_sample(_seed(), engine, dr=1.0, max_candidates=10)



def test_validate_unique_minima_rejects_and_rededuplicates(monkeypatch):
    """Hessian validation of hessian-sample minima: a minimum that stays a
    saddle is moved to rejected_minima; a rescued one replaces the original
    and is re-deduplicated against the others."""
    import mepd.elementarystep as es
    from mepd.discovery import hessian_sample as hs

    def _simple_node(x):
        return StructureNode(structure=Structure(
            symbols=["H", "H"], geometry=np.array([[0.0, 0.0, 0.0], [1.4 + x, 0.0, 0.0]]),
            charge=0, multiplicity=1,
        ))

    a, b, c = (_simple_node(x) for x in (0.0, 0.5, 1.0))
    result = hs.HessianSampleResult(seed_energy=0.0, hessian_result=None, frequencies_wavenumber=[])
    result.unique_minima = [a, b, c]

    def fake_validate(node, engine, **kw):
        if node is b:   # saddle that cannot be rescued
            return node, {"is_minimum": False, "min_frequency": -120.0, "rescued": False, "validation": "saddle"}
        if node is c:   # rescued onto a's geometry -> duplicate of a
            return a.copy(), {"is_minimum": True, "min_frequency": 80.0, "rescued": True, "validation": "ok"}
        return node, {"is_minimum": True, "min_frequency": 90.0, "rescued": False, "validation": "ok"}

    monkeypatch.setattr(es, "validate_minimum_with_rescue", fake_validate)
    monkeypatch.setattr(hs, "_dedupe_minima_nodes",
                        lambda nodes, ci: [n for i, n in enumerate(nodes)
                                           if not any(np.allclose(n.coords, m.coords) for m in nodes[:i])])
    hs._validate_unique_minima(result, engine=None, chain_inputs=None, frequency_cutoff=0.0, rescue_displacement=0.3)
    assert len(result.unique_minima) == 1 and np.allclose(result.unique_minima[0].coords, a.coords)
    assert result.minima_validation == [{"is_minimum": True, "min_frequency": 90.0, "rescued": False, "validation": "ok"}]
    assert result.rejected_minima == [b] and result.rejected_validation[0]["min_frequency"] == -120.0



def test_hessian_rescue_escalates_push_until_a_minimum(monkeypatch):
    """A rescue that fails at the configured push retries at 0.3 and 0.5
    bohr (both directions), stopping at the first success."""
    import mepd.elementarystep as es

    node = _fake_structure_node(2)
    tried = []
    monkeypatch.setattr(es, "_lowest_frequency_mode", lambda hess, n: (np.ones((2, 3)), -300.0))
    monkeypatch.setattr(es, "displace_by_dr", lambda node, displacement, dr: tried.append(round(dr, 3)) or node)
    monkeypatch.setattr(es, "_run_geom_opt", lambda n, engine: [n])

    def validate(n, engine, frequency_cutoff):
        ok = abs(tried[-1]) >= 0.3
        return es.HessianMinimaValidation(is_minimum=ok, min_frequency=50.0 if ok else -300.0,
                                          reason="ok" if ok else "saddle", hessian_result=object())

    monkeypatch.setattr(es, "_validate_hessian_minimum", validate)
    start = es.HessianMinimaValidation(is_minimum=False, min_frequency=-300.0, hessian_result=object())
    rescued, check, _ = es._hessian_rescue_failed_candidate(
        node, engine=None, validation=start, frequency_cutoff=0.0, rescue_displacement=0.1,
        verbose=False, label="t")
    assert tried == [0.1, -0.1, 0.3]                     # escalated once, stopped at the first success
    assert rescued is node and check.is_minimum and "+0.300 bohr push" in check.reason

    tried.clear()
    monkeypatch.setattr(es, "_validate_hessian_minimum",
                        lambda n, engine, frequency_cutoff: es.HessianMinimaValidation(
                            is_minimum=False, min_frequency=-10.0, reason="saddle", hessian_result=object()))
    rescued, check, _ = es._hessian_rescue_failed_candidate(
        node, engine=None, validation=start, frequency_cutoff=0.0, rescue_displacement=0.1,
        verbose=False, label="t")
    assert tried == [0.1, -0.1, 0.3, -0.3, 0.5, -0.5] and rescued is None and not check.is_minimum
    assert es._rescue_schedule(0.3) == [0.3, 0.5] and es._rescue_schedule(0.7) == [0.7]
