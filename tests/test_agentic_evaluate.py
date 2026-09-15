from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

from mepd.agentic import evaluate
from mepd.agentic.benchmark import BenchmarkReaction
from mepd.agentic.evaluate import (
    EvalResult,
    ReactionResult,
    compute_fitness,
    run_benchmark,
    score_reaction,
)
from mepd.chain import Chain
from mepd.inputs import ChainInputs, RunInputs
from mepd.nodes.node import StructureNode
from mepd.TreeNode import TreeNode


def _result(success: bool, wall_clock_s: float = 1.0, total_grad_calls: int = 10) -> ReactionResult:
    return ReactionResult(
        name="x", success=success, wall_clock_s=wall_clock_s,
        total_grad_calls=total_grad_calls,
    )


def test_compute_fitness_success_rate_dominates_cost():
    all_success_expensive = [_result(True, wall_clock_s=1000.0), _result(True, wall_clock_s=1000.0)]
    partial_success_cheap = [_result(True, wall_clock_s=0.01), _result(False)]
    assert compute_fitness(all_success_expensive) > compute_fitness(partial_success_cheap)


def test_compute_fitness_prefers_cheaper_among_equal_success():
    cheap = [_result(True, wall_clock_s=1.0, total_grad_calls=5)]
    expensive = [_result(True, wall_clock_s=100.0, total_grad_calls=500)]
    assert compute_fitness(cheap) > compute_fitness(expensive)


def test_compute_fitness_empty_is_zero():
    assert compute_fitness([]) == 0.0


def test_compute_fitness_all_failures_is_non_positive():
    assert compute_fitness([_result(False), _result(False)]) <= 0.0


# --- score_reaction: exercises the real TreeNode/Chain reduction logic
# (get_num_opt_steps, get_num_grad_calls, ordered_leaves,
# recoverable_ordered_leaves, output_chain.get_eA_chain) against a
# hand-built-but-real TreeNode/Chain fixture, with MSMEP itself monkeypatched
# out -- this is what makes it fast (milliseconds, no engine/optimizer
# actually runs) while still exercising score_reaction's own logic for real.


def _water(x_offset: float = 0.0) -> Structure:
    return Structure(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.43355001758932 + x_offset, 0.0, 0.95295864902809],
                [-1.43355001758932, 0.0, 0.95295864902809],
            ],
            dtype=float,
        ),
        charge=0, multiplicity=1,
    )


def _write_xyz(fp: Path, structure: Structure) -> None:
    fp.write_text(structure.to_xyz())


def _reaction(tmp_path: Path) -> BenchmarkReaction:
    start_fp, end_fp = tmp_path / "start.xyz", tmp_path / "end.xyz"
    _write_xyz(start_fp, _water())
    _write_xyz(end_fp, _water(x_offset=0.2))
    return BenchmarkReaction(name="water", start=start_fp, end=end_fp)


def _run_inputs() -> RunInputs:
    return RunInputs(engine_name="gxtb", gxtb_engine_kwds={"executable": "/bin/true"})


def _energy_labeled_chain(energies_hartree: list[float]) -> Chain:
    nodes = []
    for i, energy in enumerate(energies_hartree):
        node = StructureNode(structure=_water(x_offset=0.05 * i))
        node._cached_energy = energy
        nodes.append(node)
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def _single_leaf_tree(chain: Chain, *, n_steps: int = 4, grad_calls: int = 8) -> TreeNode:
    fake_neb = SimpleNamespace(chain_trajectory=[chain] * n_steps, grad_calls_made=grad_calls)
    return TreeNode(data=fake_neb, children=[], index=0)


def test_score_reaction_success_reduces_tree_correctly(monkeypatch, tmp_path):
    reaction = _reaction(tmp_path)
    chain = _energy_labeled_chain([-76.0, -75.99, -75.995])  # barrier at node 1

    class FakeMSMEP:
        def __init__(self, inputs):
            pass

        def run_recursive_minimize(self, initial_chain):
            return _single_leaf_tree(chain, n_steps=6, grad_calls=12)

    monkeypatch.setattr(evaluate, "MSMEP", FakeMSMEP)

    result = score_reaction(reaction, _run_inputs())

    assert result.success is True
    assert result.name == "water"
    assert result.n_leaves == 1
    assert result.n_splits == 0
    assert result.total_opt_steps == 6
    assert result.total_grad_calls == 12
    assert result.barrier_kcalmol == pytest.approx((-75.99 - -76.0) * 627.5)
    assert result.error_type is None


def test_score_reaction_no_leaves_is_a_failure(monkeypatch, tmp_path):
    reaction = _reaction(tmp_path)

    class FakeMSMEP:
        def __init__(self, inputs):
            pass

        def run_recursive_minimize(self, initial_chain):
            # A root with no `.data` and no children -- ordered_leaves is empty.
            return TreeNode(data=None, children=[], index=0)

    monkeypatch.setattr(evaluate, "MSMEP", FakeMSMEP)

    result = score_reaction(reaction, _run_inputs())

    assert result.success is False
    assert result.error_type == "ElementaryStepCheckFailed"


def test_score_reaction_engine_exception_is_captured_not_raised(monkeypatch, tmp_path):
    reaction = _reaction(tmp_path)

    class ExplodingMSMEP:
        def __init__(self, inputs):
            pass

        def run_recursive_minimize(self, initial_chain):
            raise RuntimeError("boom")

    monkeypatch.setattr(evaluate, "MSMEP", ExplodingMSMEP)

    result = score_reaction(reaction, _run_inputs())

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert "boom" in result.error_msg


def test_run_benchmark_uses_injected_runner_and_aggregates(tmp_path):
    reactions = [
        BenchmarkReaction(name="a", start=tmp_path / "a_s.xyz", end=tmp_path / "a_e.xyz"),
        BenchmarkReaction(name="b", start=tmp_path / "b_s.xyz", end=tmp_path / "b_e.xyz"),
    ]
    run_inputs = _run_inputs()

    def fake_runner(cmd, timeout_s):
        result_json = Path(cmd[cmd.index("--result-json") + 1])
        reaction_name = cmd[cmd.index("--reaction") + 1]
        result_json.write_text(json.dumps({
            "name": reaction_name, "success": True, "wall_clock_s": 1.0,
            "total_opt_steps": 5, "total_grad_calls": 5, "n_leaves": 1,
            "n_splits": 0, "barrier_kcalmol": 10.0, "error_type": None,
            "error_msg": None, "timed_out": False,
        }))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    result = run_benchmark(
        run_inputs, output_dir=tmp_path / "out", reactions=reactions,
        parallel_reactions=False, _runner=fake_runner,
    )

    assert isinstance(result, EvalResult)
    assert result.success_rate == 1.0
    assert len(result.per_reaction) == 2
    assert {r.name for r in result.per_reaction} == {"a", "b"}


def test_run_benchmark_marks_timeout(tmp_path):
    reactions = [BenchmarkReaction(name="a", start=tmp_path / "s.xyz", end=tmp_path / "e.xyz")]

    def timeout_runner(cmd, timeout_s):
        return None  # simulates subprocess.run's TimeoutExpired path

    result = run_benchmark(
        _run_inputs(), output_dir=tmp_path / "out", reactions=reactions,
        parallel_reactions=False, _runner=timeout_runner,
    )
    assert result.per_reaction[0].timed_out is True
    assert result.per_reaction[0].success is False
    assert result.success_rate == 0.0


def test_run_benchmark_marks_worker_crash_when_no_result_file(tmp_path):
    reactions = [BenchmarkReaction(name="a", start=tmp_path / "s.xyz", end=tmp_path / "e.xyz")]

    def crash_runner(cmd, timeout_s):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    result = run_benchmark(
        _run_inputs(), output_dir=tmp_path / "out", reactions=reactions,
        parallel_reactions=False, _runner=crash_runner,
    )
    assert result.per_reaction[0].success is False
    assert result.per_reaction[0].error_type == "WorkerCrashed"
