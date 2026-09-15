"""The single-agent hill-climbing tuning loop: propose a candidate parameter
set (via an `LLMBackend`), evaluate it against the benchmark, keep it only
if it improves on the current best, and persist every iteration so a
crashed or interrupted run can resume exactly where it left off.

Deliberately typer-free (the `on_event` callback convention, matching
`mepd/discovery/hessian_sample.py`, is how progress reaches the CLI layer)
so this module is unit-testable with a `ScriptedBackend` and an injected
fake `evaluate.run_benchmark`, with no real LLM/gxtb/subprocess involved.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from mepd.agentic import evaluate, schema
from mepd.agentic.llm_client import LLMBackend, TuningContext
from mepd.inputs import RunInputs

OnEvent = Optional[Callable[[str, dict], None]]

_HISTORY_WINDOW = 5


def _emit(on_event: OnEvent, event: str, payload: dict) -> None:
    if on_event is not None:
        on_event(event, payload)


@dataclass
class IterationRecord:
    iteration: int
    proposed_params: dict
    llm_raw_response: str
    llm_rationale: Optional[str]
    eval_result: Optional[dict]
    fitness: Optional[float]
    accepted: bool
    is_seed: bool
    wall_clock_s: float
    timestamp: str


@dataclass
class TuningState:
    seed_params: dict
    path_min_method: str
    best_params: dict
    best_fitness: Optional[float] = None
    best_iteration: int = 0
    history: list[IterationRecord] = field(default_factory=list)
    started_at: str = ""
    updated_at: str = ""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _state_path(output_dir: Path) -> Path:
    return output_dir / "state.json"


def _trajectory_path(output_dir: Path) -> Path:
    return output_dir / "trajectory.jsonl"


def _write_state(output_dir: Path, state: TuningState) -> None:
    tmp_path = output_dir / "state.json.tmp"
    tmp_path.write_text(json.dumps(asdict(state), indent=2))
    os.replace(tmp_path, _state_path(output_dir))


def _append_trajectory(output_dir: Path, record: IterationRecord) -> None:
    with open(_trajectory_path(output_dir), "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def _load_state(output_dir: Path) -> Optional[TuningState]:
    state_path = _state_path(output_dir)
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text())
    history = [IterationRecord(**rec) for rec in payload.pop("history", [])]
    return TuningState(history=history, **payload)


def _condensed_history(history: list[IterationRecord]) -> list[dict]:
    return [
        {
            "iteration": rec.iteration,
            "proposed_params": rec.proposed_params,
            "fitness": rec.fitness,
            "accepted": rec.accepted,
        }
        for rec in history[-_HISTORY_WINDOW:]
    ]


def _trailing_non_improving(history: list[IterationRecord]) -> int:
    """Number of consecutive non-accepted, non-seed iterations at the end
    of `history` -- used both when starting fresh and when resuming, so
    `patience` keeps working across a resume."""
    count = 0
    for rec in history:
        if rec.is_seed:
            continue
        count = 0 if rec.accepted else count + 1
    return count


def run_tuning_loop(
    *,
    seed_run_inputs: RunInputs,
    backend: LLMBackend,
    output_dir: Path,
    max_iterations: int = 10,
    patience: Optional[int] = None,
    timeout_per_reaction: float = 1200.0,
    parallel_reactions: bool = True,
    parallel_msmep: bool = False,
    resume: bool = True,
    on_event: OnEvent = None,
) -> TuningState:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_dict = seed_run_inputs.to_dict()
    param_schema = schema.get_schema(seed_run_inputs.path_min_method)
    schema_text = schema.render_schema_for_prompt(param_schema)
    json_schema = schema.to_json_schema(param_schema)
    seed_values = schema.extract_current_values(seed_dict, param_schema)

    state = _load_state(output_dir) if resume else None
    if state is None:
        state = TuningState(
            seed_params=seed_dict,
            path_min_method=seed_run_inputs.path_min_method,
            best_params=seed_dict,
            started_at=_now(),
            updated_at=_now(),
        )
        _emit(on_event, "seed_evaluating", {})
        seed_start = time.monotonic()
        seed_eval = evaluate.run_benchmark(
            seed_run_inputs,
            output_dir=output_dir / "iter_0000_seed",
            timeout_s=timeout_per_reaction,
            parallel_reactions=parallel_reactions,
            parallel_msmep=parallel_msmep,
        )
        seed_record = IterationRecord(
            iteration=0,
            proposed_params={},
            llm_raw_response="",
            llm_rationale="seed baseline",
            eval_result=asdict(seed_eval),
            fitness=seed_eval.fitness,
            accepted=True,
            is_seed=True,
            wall_clock_s=time.monotonic() - seed_start,
            timestamp=_now(),
        )
        state.best_fitness = seed_eval.fitness
        state.best_iteration = 0
        state.history.append(seed_record)
        _append_trajectory(output_dir, seed_record)
        _write_state(output_dir, state)
        _emit(on_event, "iteration_done", {"record": asdict(seed_record), "state": "seed"})
    else:
        _emit(on_event, "resumed", {"completed_iterations": len(state.history)})

    since_improvement = _trailing_non_improving(state.history)
    iteration = len(state.history)
    stopped_early = False

    while iteration <= max_iterations:
        if patience is not None and since_improvement >= patience:
            stopped_early = True
            _emit(on_event, "stopped_patience", {"since_improvement": since_improvement})
            break

        context = TuningContext(
            schema_text=schema_text,
            json_schema=json_schema,
            seed_params=seed_values,
            current_best_params=schema.extract_current_values(state.best_params, param_schema),
            current_best_fitness=state.best_fitness,
            recent_history=_condensed_history(state.history),
            iteration_index=iteration - 1,
            max_iterations=max_iterations,
        )

        _emit(on_event, "iteration_start", {"iteration": iteration})
        iter_start = time.monotonic()
        # LLMConnectionError deliberately propagates here -- tuning can't
        # proceed without the LLM, so this is a hard stop, not a
        # skip-and-continue case.
        proposal = backend.propose_params(context)

        if not proposal.valid or not proposal.values:
            record = IterationRecord(
                iteration=iteration,
                proposed_params=proposal.values,
                llm_raw_response=proposal.raw_response,
                llm_rationale=proposal.rationale,
                eval_result=None,
                fitness=None,
                accepted=False,
                is_seed=False,
                wall_clock_s=time.monotonic() - iter_start,
                timestamp=_now(),
            )
            state.history.append(record)
            state.updated_at = _now()
            _append_trajectory(output_dir, record)
            _write_state(output_dir, state)
            _emit(on_event, "iteration_done", {
                "record": asdict(record),
                "state": "no_op" if proposal.valid else "invalid",
                "warnings": [],
            })
            since_improvement += 1
            iteration += 1
            continue

        clamped_values, warnings = schema.clamp_and_validate(proposal.values, param_schema)
        candidate_dict = schema.apply_values(
            copy.deepcopy(state.best_params), param_schema, clamped_values
        )
        candidate_run_inputs = RunInputs(**copy.deepcopy(candidate_dict))

        eval_result = evaluate.run_benchmark(
            candidate_run_inputs,
            output_dir=output_dir / f"iter_{iteration:04d}",
            timeout_s=timeout_per_reaction,
            parallel_reactions=parallel_reactions,
            parallel_msmep=parallel_msmep,
        )
        accepted = state.best_fitness is None or eval_result.fitness > state.best_fitness
        if accepted:
            state.best_params = candidate_dict
            state.best_fitness = eval_result.fitness
            state.best_iteration = iteration
            since_improvement = 0
        else:
            since_improvement += 1

        record = IterationRecord(
            iteration=iteration,
            proposed_params=clamped_values,
            llm_raw_response=proposal.raw_response,
            llm_rationale=proposal.rationale,
            eval_result=asdict(eval_result),
            fitness=eval_result.fitness,
            accepted=accepted,
            is_seed=False,
            wall_clock_s=time.monotonic() - iter_start,
            timestamp=_now(),
        )
        state.history.append(record)
        state.updated_at = _now()
        _append_trajectory(output_dir, record)
        _write_state(output_dir, state)
        _emit(on_event, "iteration_done", {
            "record": asdict(record),
            "state": "accepted" if accepted else "rejected",
            "warnings": warnings,
        })

        iteration += 1

    if not stopped_early:
        _emit(on_event, "stopped_max_iterations", {"max_iterations": max_iterations})

    return state
