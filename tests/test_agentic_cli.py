from __future__ import annotations

from pathlib import Path

import pytest
import typer

from mepd.agentic import benchmark
from mepd.agentic.cli import check_benchmark, show_schema, tune
from mepd.agentic.loop import IterationRecord, TuningState


_SEED_TOML = """\
engine_name = "gxtb"
path_min_method = "neb"

[gxtb_engine_kwds]
executable = "/bin/true"
n_threads = 1

[path_min_inputs]
max_steps = 500

[chain_inputs]
k = 0.1
delta_k = 0.09

[gi_inputs]
nimages = 12
"""


def _write_seed(tmp_path: Path) -> Path:
    fp = tmp_path / "seed.toml"
    fp.write_text(_SEED_TOML)
    return fp


def test_show_schema_prints_known_method(capsys):
    show_schema(path_min_method="NEB")
    out = capsys.readouterr().out
    assert "gi_inputs.nimages" in out


def test_show_schema_unknown_method_exits_1(capsys):
    with pytest.raises(typer.Exit):
        show_schema(path_min_method="FNEB")
    assert "No agentic-tuning knob schema" in capsys.readouterr().err


def test_check_benchmark_reports_ok(capsys):
    # Uses the real benchmark data on this checkout -- validate_benchmark_data
    # touches only local xyz files, no network/gxtb, so this is safe in CI.
    check_benchmark()
    out = capsys.readouterr().out
    assert "All benchmark reaction endpoints found." in out


def test_check_benchmark_reports_missing_files(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "validate_benchmark_data", lambda: ["oxycope: start endpoint not found at /nope"])
    with pytest.raises(typer.Exit):
        check_benchmark()
    assert "not found" in capsys.readouterr().out


def test_tune_writes_best_inputs_toml(monkeypatch, tmp_path):
    seed_fp = _write_seed(tmp_path)
    output_dir = tmp_path / "out"

    def fake_run_tuning_loop(*, seed_run_inputs, backend, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        best_params = seed_run_inputs.to_dict()
        best_params["gi_inputs"]["nimages"] = 16
        return TuningState(
            seed_params=seed_run_inputs.to_dict(),
            path_min_method="NEB",
            best_params=best_params,
            best_fitness=42.0,
            best_iteration=3,
            history=[
                IterationRecord(
                    iteration=0, proposed_params={}, llm_raw_response="",
                    llm_rationale="seed", eval_result=None, fitness=42.0,
                    accepted=True, is_seed=True, wall_clock_s=0.1,
                    timestamp="2026-01-01T00:00:00Z",
                )
            ],
        )

    monkeypatch.setattr("mepd.agentic.cli.loop.run_tuning_loop", fake_run_tuning_loop)

    tune(
        seed=seed_fp, output=output_dir, iterations=1, patience=None,
        backend_name="null", model="unused", ollama_base_url=None,
        temperature=0.4, timeout_per_reaction=60.0, parallel_reactions=True,
        parallel_msmep=False, resume=True,
    )

    best_toml = output_dir / "best_inputs.toml"
    assert best_toml.exists()
    assert "nimages = 16" in best_toml.read_text()


def test_tune_rejects_unknown_backend(tmp_path, capsys):
    seed_fp = _write_seed(tmp_path)
    with pytest.raises(typer.Exit):
        tune(
            seed=seed_fp, output=tmp_path / "out", iterations=1, patience=None,
            backend_name="not-a-backend", model="unused", ollama_base_url=None,
            temperature=0.4, timeout_per_reaction=60.0, parallel_reactions=True,
            parallel_msmep=False, resume=True,
        )
    assert "Unknown agentic-tuning backend" in capsys.readouterr().err
