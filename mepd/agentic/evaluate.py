"""Evaluation harness for `mepd agentic tune`: runs the fixed 3-reaction
benchmark (`mepd.agentic.benchmark`) under a candidate `RunInputs` and
reduces the outcome to a single fitness score.

Split into a pure, engine-agnostic core (`score_reaction`/`compute_fitness`
-- fast-testable with `mepd.engines.flower.FlowerPotential`, exactly as
`tests/test_msmep.py` already does) and a subprocess-orchestration shell
(`run_benchmark`): each reaction runs in its own `python -m
mepd.agentic.worker` subprocess, one per reaction, for crash/hang isolation.
`timeout_timer.timeout()` (used elsewhere in this repo) is documented as
SIGALRM-based and unsafe off the main thread, so process-level timeout
(`subprocess.run(..., timeout=...)`) is used instead of that helper.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import mepd.chainhelpers as ch
from mepd.agentic.benchmark import BENCHMARK_REACTIONS, BenchmarkReaction
from mepd.chain import Chain
from mepd.errors import format_exception_message
from mepd.inputs import RunInputs
from mepd.msmep import MSMEP
from mepd.nodes.node import StructureNode
from qcdata import Structure

_SUCCESS_WEIGHT = 1_000_000.0
_GRAD_CALL_PENALTY = 0.01


@dataclass
class ReactionResult:
    name: str
    success: bool
    wall_clock_s: float
    total_opt_steps: int = 0
    total_grad_calls: int = 0
    n_leaves: int = 0
    n_splits: int = 0
    barrier_kcalmol: Optional[float] = None
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    timed_out: bool = False


@dataclass
class EvalResult:
    per_reaction: list[ReactionResult] = field(default_factory=list)
    fitness: float = 0.0
    success_rate: float = 0.0
    wall_clock_s_total: float = 0.0


def score_reaction(
    reaction: BenchmarkReaction,
    run_inputs: RunInputs,
    *,
    parallel_msmep: bool = False,
    parallel_workers: Optional[int] = None,
) -> ReactionResult:
    """Runs recursive MSMEP autosplitting for one benchmark reaction under
    `run_inputs` and reduces the resulting `TreeNode` to a `ReactionResult`.
    Never raises: any failure (electronic-structure error, non-convergence,
    a bad candidate parameter set) is captured as `success=False` with an
    `error_type`/`error_msg`, since a bad LLM proposal failing a reaction is
    an expected/scored outcome, not a bug."""
    start_time = time.monotonic()
    try:
        start_node = StructureNode(structure=Structure.open(str(reaction.start)))
        end_node = StructureNode(structure=Structure.open(str(reaction.end)))

        seed_chain = Chain.model_validate({
            "nodes": [start_node, end_node],
            "parameters": copy.deepcopy(run_inputs.chain_inputs),
        })
        initial_chain = ch.run_geodesic(
            chain=seed_chain,
            chain_inputs=copy.deepcopy(run_inputs.chain_inputs),
            nimages=run_inputs.gi_inputs.nimages,
            friction=run_inputs.gi_inputs.friction,
            nudge=run_inputs.gi_inputs.nudge,
            random_seed=run_inputs.gi_inputs.random_seed,
            align=run_inputs.gi_inputs.align,
            **(run_inputs.gi_inputs.extra_kwds or {}),
        )

        msmep = MSMEP(inputs=run_inputs)
        if parallel_msmep:
            history = msmep.run_parallel_recursive_minimize(
                initial_chain, max_workers=parallel_workers
            )
        else:
            history = msmep.run_recursive_minimize(initial_chain)
    except Exception as exc:
        return ReactionResult(
            name=reaction.name,
            success=False,
            wall_clock_s=time.monotonic() - start_time,
            error_type=type(exc).__name__,
            error_msg=format_exception_message(exc),
        )

    wall_clock_s = time.monotonic() - start_time
    ordered_leaves = history.ordered_leaves
    recoverable_indices = {leaf.index for leaf in history.recoverable_ordered_leaves}
    success = bool(ordered_leaves) and all(
        leaf.index in recoverable_indices for leaf in ordered_leaves
    )

    barrier_kcalmol = None
    error_type = None
    error_msg = None
    if success:
        try:
            barrier_kcalmol = float(history.output_chain.get_eA_chain())
        except Exception as exc:
            success = False
            error_type = type(exc).__name__
            error_msg = f"Could not assemble/score output chain: {format_exception_message(exc)}"
    if not success and error_type is None:
        error_type = "ElementaryStepCheckFailed"
        error_msg = "One or more leaves failed elementary-step/recoverability checks."

    return ReactionResult(
        name=reaction.name,
        success=success,
        wall_clock_s=wall_clock_s,
        total_opt_steps=history.get_num_opt_steps(),
        total_grad_calls=history.get_num_grad_calls(),
        n_leaves=len(ordered_leaves),
        n_splits=max(0, history.total_nodes - len(ordered_leaves)),
        barrier_kcalmol=barrier_kcalmol,
        error_type=error_type,
        error_msg=error_msg,
    )


def compute_fitness(per_reaction: list[ReactionResult]) -> float:
    """`success_rate` dominates by construction (deltas of ~1/3 of
    `_SUCCESS_WEIGHT`, far larger than any realistic wall-clock/grad-call
    penalty) so the loop always prefers more successes; among equally
    successful candidates, lower wall-clock/grad-call cost wins. Chemical
    -quality signals (barrier height, mechanism shape) are deliberately
    excluded -- they're chemistry-dependent, not something a parameter
    change can principled-ly improve."""
    if not per_reaction:
        return 0.0
    successes = [r for r in per_reaction if r.success]
    success_rate = len(successes) / len(per_reaction)
    cost = sum(r.wall_clock_s for r in successes) + _GRAD_CALL_PENALTY * sum(
        r.total_grad_calls for r in successes
    )
    return success_rate * _SUCCESS_WEIGHT - cost


RunnerFn = Callable[[list[str], float], Optional["subprocess.CompletedProcess"]]


def _default_runner(cmd: list[str], timeout_s: float) -> Optional["subprocess.CompletedProcess"]:
    try:
        return subprocess.run(cmd, timeout=timeout_s, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return None


def run_benchmark(
    run_inputs: RunInputs,
    *,
    output_dir: Path,
    timeout_s: float = 1200.0,
    parallel_reactions: bool = True,
    max_workers: Optional[int] = None,
    parallel_msmep: bool = False,
    parallel_msmep_workers: Optional[int] = None,
    reactions: Optional[list[BenchmarkReaction]] = None,
    _runner: Optional[RunnerFn] = None,
) -> EvalResult:
    """Runs each benchmark reaction in its own `mepd.agentic.worker`
    subprocess (one per reaction), enforcing `timeout_s` per reaction, and
    reduces the results to an `EvalResult`. `_runner` is an injectable seam
    (default: a real subprocess spawner) so tests can substitute a fake
    without touching subprocess machinery."""
    reactions = reactions if reactions is not None else BENCHMARK_REACTIONS
    runner = _runner or _default_runner
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_toml = output_dir / "candidate_inputs.toml"
    run_inputs.save(inputs_toml)

    def _run_one(reaction: BenchmarkReaction) -> ReactionResult:
        result_json = output_dir / f"{reaction.name}_result.json"
        cmd = [
            sys.executable, "-m", "mepd.agentic.worker",
            "--reaction", reaction.name,
            "--inputs", str(inputs_toml),
            "--result-json", str(result_json),
        ]
        if parallel_msmep:
            cmd.append("--parallel-msmep")
            if parallel_msmep_workers is not None:
                cmd += ["--parallel-workers", str(parallel_msmep_workers)]

        start = time.monotonic()
        completed = runner(cmd, timeout_s)
        wall_clock_s = time.monotonic() - start

        if completed is None:
            return ReactionResult(
                name=reaction.name, success=False, wall_clock_s=wall_clock_s,
                timed_out=True, error_type="TimeoutExpired",
                error_msg=f"Reaction '{reaction.name}' exceeded "
                          f"--timeout-per-reaction={timeout_s}s.",
            )
        if not result_json.exists():
            stderr_tail = (getattr(completed, "stderr", "") or "")[-2000:]
            return ReactionResult(
                name=reaction.name, success=False, wall_clock_s=wall_clock_s,
                error_type="WorkerCrashed",
                error_msg=(
                    f"Worker exited {completed.returncode} with no result "
                    f"file. stderr: {stderr_tail}"
                ),
            )
        payload = json.loads(result_json.read_text())
        return ReactionResult(**payload)

    if parallel_reactions and len(reactions) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers or len(reactions)
        ) as pool:
            per_reaction = list(pool.map(_run_one, reactions))
    else:
        per_reaction = [_run_one(r) for r in reactions]

    fitness = compute_fitness(per_reaction)
    success_rate = (
        sum(1 for r in per_reaction if r.success) / len(per_reaction)
        if per_reaction else 0.0
    )
    return EvalResult(
        per_reaction=per_reaction,
        fitness=fitness,
        success_rate=success_rate,
        wall_clock_s_total=sum(r.wall_clock_s for r in per_reaction),
    )
