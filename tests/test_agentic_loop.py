from __future__ import annotations

from pathlib import Path

import pytest

from mepd.agentic import loop
from mepd.agentic.evaluate import EvalResult, ReactionResult
from mepd.agentic.llm_client import ParamProposal, ScriptedBackend
from mepd.inputs import RunInputs


def _run_inputs() -> RunInputs:
    return RunInputs(engine_name="gxtb", gxtb_engine_kwds={"executable": "/bin/true"})


def _eval(fitness: float, success: bool = True) -> EvalResult:
    per_reaction = [ReactionResult(name="x", success=success, wall_clock_s=1.0)]
    return EvalResult(per_reaction=per_reaction, fitness=fitness, success_rate=1.0 if success else 0.0)


def _scripted_run_benchmark(fitness_sequence):
    """A fake `evaluate.run_benchmark` that returns canned `EvalResult`s in
    order, one call per invocation (including the seed evaluation)."""
    it = iter(fitness_sequence)

    def fake_run_benchmark(run_inputs, *, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return _eval(next(it))

    return fake_run_benchmark


def test_seed_is_evaluated_as_iteration_zero_and_becomes_initial_best(monkeypatch, tmp_path):
    monkeypatch.setattr(loop.evaluate, "run_benchmark", _scripted_run_benchmark([100.0]))
    backend = ScriptedBackend([])  # loop should stop before asking for any proposal

    state = loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=backend, output_dir=tmp_path,
        max_iterations=0,
    )

    assert len(state.history) == 1
    assert state.history[0].is_seed is True
    assert state.best_fitness == 100.0
    assert state.best_iteration == 0


def test_accepts_only_improving_candidates(monkeypatch, tmp_path):
    # seed=100, iter1=50 (worse, rejected), iter2=150 (better, accepted)
    monkeypatch.setattr(loop.evaluate, "run_benchmark", _scripted_run_benchmark([100.0, 50.0, 150.0]))
    backend = ScriptedBackend([
        ParamProposal(values={"gi_inputs": {"nimages": 14}}),
        ParamProposal(values={"gi_inputs": {"nimages": 16}}),
    ])

    state = loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=backend, output_dir=tmp_path,
        max_iterations=2,
    )

    assert len(state.history) == 3
    assert state.history[1].accepted is False
    assert state.history[2].accepted is True
    assert state.best_fitness == 150.0
    assert state.best_iteration == 2
    assert state.best_params["gi_inputs"]["nimages"] == 16


def test_invalid_or_empty_proposal_is_logged_as_no_op_and_consumes_iteration(monkeypatch, tmp_path):
    monkeypatch.setattr(loop.evaluate, "run_benchmark", _scripted_run_benchmark([100.0]))
    backend = ScriptedBackend([
        ParamProposal(values={}, valid=True),  # explicit no-op
        ParamProposal(values={"gi_inputs": {"nimages": 20}}, valid=False, error="boom"),
    ])

    state = loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=backend, output_dir=tmp_path,
        max_iterations=2,
    )

    assert len(state.history) == 3  # seed + 2 no-op iterations
    assert state.history[1].eval_result is None
    assert state.history[1].accepted is False
    assert state.history[2].eval_result is None
    assert state.best_fitness == 100.0  # unchanged from seed


def test_unknown_params_are_dropped_before_evaluation(monkeypatch, tmp_path):
    seen_candidates = []

    def fake_run_benchmark(run_inputs, *, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        seen_candidates.append(run_inputs)
        return _eval(100.0 if len(seen_candidates) == 1 else 200.0)

    monkeypatch.setattr(loop.evaluate, "run_benchmark", fake_run_benchmark)
    backend = ScriptedBackend([
        ParamProposal(values={"engine_name": "not-allowed", "gi_inputs": {"nimages": 14}}),
    ])

    state = loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=backend, output_dir=tmp_path,
        max_iterations=1,
    )

    assert state.best_params["gi_inputs"]["nimages"] == 14
    assert state.best_params["engine_name"] == "gxtb"  # untouched, not "not-allowed"


def test_patience_stops_early(monkeypatch, tmp_path):
    monkeypatch.setattr(
        loop.evaluate, "run_benchmark",
        _scripted_run_benchmark([100.0, 50.0, 40.0, 30.0, 20.0]),
    )
    backend = ScriptedBackend([
        ParamProposal(values={"gi_inputs": {"nimages": 14}}),
        ParamProposal(values={"gi_inputs": {"nimages": 15}}),
        ParamProposal(values={"gi_inputs": {"nimages": 16}}),
        ParamProposal(values={"gi_inputs": {"nimages": 17}}),
    ])

    state = loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=backend, output_dir=tmp_path,
        max_iterations=10, patience=2,
    )

    # seed + 2 non-improving iterations, then stop (patience=2).
    assert len(state.history) == 3
    assert all(not rec.accepted for rec in state.history[1:])


def test_resume_continues_from_persisted_state(monkeypatch, tmp_path):
    monkeypatch.setattr(loop.evaluate, "run_benchmark", _scripted_run_benchmark([100.0, 150.0]))
    backend = ScriptedBackend([ParamProposal(values={"gi_inputs": {"nimages": 14}})])

    state1 = loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=backend, output_dir=tmp_path,
        max_iterations=1,
    )
    assert len(state1.history) == 2
    assert (tmp_path / "state.json").exists()

    # A fresh backend + a fresh run_benchmark stub -- resuming shouldn't
    # re-evaluate the seed or re-ask for iteration 1's proposal.
    monkeypatch.setattr(loop.evaluate, "run_benchmark", _scripted_run_benchmark([200.0]))
    backend2 = ScriptedBackend([ParamProposal(values={"gi_inputs": {"nimages": 18}})])

    state2 = loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=backend2, output_dir=tmp_path,
        max_iterations=2, resume=True,
    )

    assert len(state2.history) == 3
    assert state2.history[0].is_seed is True
    assert state2.history[1].proposed_params == {"gi_inputs.nimages": 14}
    assert state2.history[2].proposed_params == {"gi_inputs.nimages": 18}


def test_stopping_event_fires_when_patience_and_max_iterations_coincide(monkeypatch, tmp_path):
    """Regression test: when `since_improvement` reaches `patience` on
    exactly the same iteration that also exhausts `max_iterations`, the
    while loop exits via its normal condition (not the `break`), so the
    post-loop notification must be driven by an explicit "did we break
    early" flag -- re-deriving it from `since_improvement`/`patience`
    after the fact (the original, buggy approach) misses this case and
    emits no stopping event at all."""
    monkeypatch.setattr(
        loop.evaluate, "run_benchmark", _scripted_run_benchmark([100.0, 50.0, 40.0]),
    )
    backend = ScriptedBackend([
        ParamProposal(values={"gi_inputs": {"nimages": 14}}),
        ParamProposal(values={"gi_inputs": {"nimages": 15}}),
    ])
    events = []

    state = loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=backend, output_dir=tmp_path,
        max_iterations=2, patience=2,
        on_event=lambda event, payload: events.append(event),
    )

    assert len(state.history) == 3  # seed + 2 rejected iterations
    stopping_events = [e for e in events if e.startswith("stopped_")]
    assert stopping_events == ["stopped_max_iterations"]


def test_no_resume_starts_fresh_even_if_state_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(loop.evaluate, "run_benchmark", _scripted_run_benchmark([100.0, 100.0]))
    backend = ScriptedBackend([])

    loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=backend, output_dir=tmp_path,
        max_iterations=0,
    )
    state = loop.run_tuning_loop(
        seed_run_inputs=_run_inputs(), backend=ScriptedBackend([]), output_dir=tmp_path,
        max_iterations=0, resume=False,
    )
    assert len(state.history) == 1  # re-evaluated the seed, not resumed
