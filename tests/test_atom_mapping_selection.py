from __future__ import annotations

import numpy as np
import pytest
from qcdata.models.structure import Structure

pytest.importorskip("slapmapper")

from mepd.atom_mapping import suggest_atom_mapping_candidates  # noqa: E402
from mepd.atom_mapping_selection import (  # noqa: E402
    METRICS,
    build_candidates,
    maybe_realign_pair,
    score_candidate,
    score_candidate_all_metrics,
    select_best_candidate,
)
from mepd.inputs import RunInputs  # noqa: E402


def _propene() -> Structure:
    """CH2=CH-CH3: C0 (=CH2, symmetric Hs 3,4), C1 (=CH-, H5), C2 (-CH3, Hs 6,7,8)."""
    symbols = ["C", "C", "C", "H", "H", "H", "H", "H", "H"]
    geometry = np.array(
        [
            [0.000, 1.303, 0.000],
            [0.000, 0.000, 0.000],
            [1.501, -0.400, 0.000],
            [-0.920, 1.860, 0.000],
            [0.920, 1.860, 0.000],
            [-0.950, -0.550, 0.000],
            [1.550, -1.030, 0.870],
            [1.550, -1.030, -0.870],
            [2.300, 0.320, 0.000],
        ]
    )
    return Structure(symbols=symbols, geometry=geometry, charge=0, multiplicity=1)


_PROPENE_SCRAMBLED_ORDER = [2, 1, 0, 6, 7, 8, 5, 3, 4]


def _scrambled_propene() -> Structure:
    start = _propene()
    order = _PROPENE_SCRAMBLED_ORDER
    return Structure(
        symbols=np.asarray(start.symbols)[order],
        geometry=np.asarray(start.geometry)[order],
        charge=0,
        multiplicity=1,
    )


class _FakeEnergyEngine:
    """A minimal engine stand-in: `compute_energies` assigns a cheap,
    deterministic energy (sum of squared coordinates) per node instead of
    running real QM -- enough to exercise `score_candidate`'s "gi-energy"
    path without an electronic-structure engine."""

    def compute_energies(self, chain):
        for node in chain.nodes:
            node._cached_energy = 0.01 * float(np.sum(np.asarray(node.coords) ** 2))


def _run_inputs_for_test() -> RunInputs:
    run_inputs = RunInputs(
        path_min_method="NEB",
        gi_inputs={"nimages": 4},
    )
    run_inputs.engine = _FakeEnergyEngine()
    return run_inputs


def test_build_candidates_includes_identity_and_each_mapping():
    start = _propene()
    end = _scrambled_propene()
    atom_maps = suggest_atom_mapping_candidates(start, end)

    candidates = build_candidates(start, end, atom_maps)

    labels = [c.label for c in candidates]
    assert labels[0] == "identity"
    assert candidates[0].end_structure is end
    assert "mapping_0" in labels
    mapping_candidate = next(c for c in candidates if c.label == "mapping_0")
    assert np.allclose(np.asarray(mapping_candidate.end_structure.geometry), np.asarray(start.geometry))


def test_build_candidates_deduplicates_identical_orders():
    start = _propene()
    end = _scrambled_propene()
    atom_maps = suggest_atom_mapping_candidates(start, end)
    # Duplicate the same candidate mapping several times -- must not produce
    # multiple "mapping_i" entries for the same permutation.
    candidates = build_candidates(start, end, atom_maps + atom_maps)
    labels = [c.label for c in candidates]
    assert len(labels) == len(set(labels))


@pytest.mark.parametrize("metric", METRICS)
def test_score_candidate_returns_a_finite_scalar_for_every_metric(metric):
    start = _propene()
    end = _scrambled_propene()
    atom_maps = suggest_atom_mapping_candidates(start, end)
    candidates = build_candidates(start, end, atom_maps)
    run_inputs = _run_inputs_for_test()

    for candidate in candidates:
        score, chain = score_candidate(candidate, metric, start, run_inputs)
        assert isinstance(score, float)
        assert np.isfinite(score)
        assert len(chain) >= 2


def test_score_candidate_unknown_metric_raises():
    start = _propene()
    end = _scrambled_propene()
    candidates = build_candidates(start, end, [])
    run_inputs = _run_inputs_for_test()
    with pytest.raises(ValueError, match="Unknown atom-mapping selection metric"):
        score_candidate(candidates[0], "not-a-metric", start, run_inputs)


def test_score_candidate_all_metrics_matches_individual_scores():
    start = _propene()
    end = _scrambled_propene()
    atom_maps = suggest_atom_mapping_candidates(start, end)
    candidates = build_candidates(start, end, atom_maps)
    run_inputs = _run_inputs_for_test()

    candidate = candidates[0]
    all_scores, _ = score_candidate_all_metrics(candidate, start, run_inputs)
    assert set(all_scores) == set(METRICS)
    for metric in METRICS:
        assert np.isfinite(all_scores[metric])


def test_select_best_candidate_picks_lowest_scoring_option(monkeypatch):
    import mepd.atom_mapping_selection as selection_module

    start = _propene()
    end = _scrambled_propene()
    atom_maps = suggest_atom_mapping_candidates(start, end)
    candidates = build_candidates(start, end, atom_maps)
    run_inputs = _run_inputs_for_test()

    def fake_score(candidate, metric, start_structure, run_inputs):
        return (0.0 if candidate.label == "identity" else 10.0), None

    monkeypatch.setattr(selection_module, "score_candidate", fake_score)

    result = select_best_candidate(candidates, "gi-energy", start, run_inputs)
    assert result.winner.label == "identity"
    assert result.scores["identity"] == 0.0


def test_select_best_candidate_veto_margin_prefers_identity_on_small_improvement(monkeypatch):
    import mepd.atom_mapping_selection as selection_module

    start = _propene()
    end = _scrambled_propene()
    atom_maps = suggest_atom_mapping_candidates(start, end)
    candidates = build_candidates(start, end, atom_maps)
    run_inputs = _run_inputs_for_test()

    def fake_score(candidate, metric, start_structure, run_inputs):
        return (10.0 if candidate.label == "identity" else 9.0), None

    monkeypatch.setattr(selection_module, "score_candidate", fake_score)

    # Mapping is better by only 1.0 -- below a veto_margin of 5.0, identity wins.
    result = select_best_candidate(candidates, "gi-energy", start, run_inputs, veto_margin=5.0)
    assert result.winner.label == "identity"

    # With no margin, the marginally-better mapping wins.
    result = select_best_candidate(candidates, "gi-energy", start, run_inputs, veto_margin=0.0)
    assert result.winner.label != "identity"


def test_maybe_realign_pair_no_op_for_already_matching_structures():
    start = _propene()
    end = _propene()
    result, changed = maybe_realign_pair(start, end, _run_inputs_for_test())
    assert changed is False
    assert result is end


def test_maybe_realign_pair_no_op_for_mismatched_atom_counts():
    start = _propene()
    end = Structure(
        symbols=start.symbols[:-1], geometry=np.asarray(start.geometry)[:-1],
        charge=0, multiplicity=1,
    )
    result, changed = maybe_realign_pair(start, end, _run_inputs_for_test())
    assert changed is False
    assert result is end


def test_maybe_realign_pair_realigns_when_a_mapping_wins(monkeypatch):
    import mepd.atom_mapping_selection as selection_module

    def fake_score(candidate, metric, start_structure, run_inputs):
        return (10.0 if candidate.label == "identity" else 0.0), None

    monkeypatch.setattr(selection_module, "score_candidate", fake_score)

    start = _propene()
    end = _scrambled_propene()
    result, changed = maybe_realign_pair(start, end, _run_inputs_for_test())

    assert changed is True
    assert np.allclose(np.asarray(result.geometry), np.asarray(start.geometry))
