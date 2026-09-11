from __future__ import annotations

import json

import pytest
import typer

import mepd.hessian_sample as hessian_sample_module
from mepd.cli import hessian_global as cli_hessian_global
from mepd.hessian_sample import HessianGlobalOptResult
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode
from qcdata import Structure
import numpy as np


def _water(x_offset: float = 0.0, energy: float | None = None) -> StructureNode:
    node = StructureNode(structure=Structure(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.43355001758932 + x_offset, 0.0, 0.95295864902809],
                [-1.43355001758932, 0.0, 0.95295864902809],
            ],
            dtype=float,
        ),
        charge=0,
        multiplicity=1,
    ))
    if energy is not None:
        node._cached_energy = energy
        node._cached_gradient = np.zeros((3, 3))
    return node


def _run_inputs_for_test() -> RunInputs:
    return RunInputs(engine_name="gxtb", gxtb_engine_kwds={"executable": "gxtb"})


def _call_hessian_global(**overrides):
    """Same rationale as the other `_call_*` helpers in this test suite --
    calling a Typer-decorated function directly does not resolve
    `typer.Option(...)` defaults, so every parameter must be explicit."""
    kwargs = dict(
        structure="O",
        inputs=None,
        charge=None,
        multiplicity=None,
        dr=0.1,
        max_candidates=100,
        maxiter=500,
        temperature=298.15,
        energy_tolerance_kcal=1.0e-4,
        max_rounds=100,
        random_seed=None,
        output=None,
    )
    kwargs.update(overrides)
    return cli_hessian_global(**kwargs)


def test_hessian_global_command_rejects_nonpositive_temperature(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(temperature=0.0, output=tmp_path / "out")


def test_hessian_global_command_rejects_nonpositive_max_rounds(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(max_rounds=0, output=tmp_path / "out")


def test_hessian_global_command_writes_accepted_minima_and_summary(tmp_path, monkeypatch):
    minimum_a = _water(x_offset=0.1, energy=-1.0)
    minimum_b = _water(x_offset=-0.1, energy=-2.0)

    def fake_run_hessian_global_optimization(seed_node, engine, **kwargs):
        return HessianGlobalOptResult(
            start_energy=0.0,
            rounds_run=2,
            stopped_reason="queue_exhausted",
            accepted_minima=[minimum_a, minimum_b],
            round_summaries=[
                {"round": 0, "sources": 1, "candidates_optimized": 4, "accepted": 1},
                {"round": 1, "sources": 1, "candidates_optimized": 4, "accepted": 1},
            ],
        )

    monkeypatch.setattr(
        hessian_sample_module, "run_hessian_global_optimization",
        fake_run_hessian_global_optimization,
    )

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "global_out"
    _call_hessian_global(structure="O", inputs=inputs_fp, output=output_dir, random_seed=7)

    assert (output_dir / "accepted_minima.xyz").exists()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["rounds_run"] == 2
    assert summary["stopped_reason"] == "queue_exhausted"
    assert summary["accepted_minima"] == 2
    assert summary["random_seed"] == 7
    assert len(summary["round_summaries"]) == 2
    assert len(summary["accepted_minima_rel_energies_kcal_mol"]) == 2


def test_hessian_global_command_reports_no_minima_found(tmp_path, monkeypatch, capsys):
    def fake_run_hessian_global_optimization(seed_node, engine, **kwargs):
        return HessianGlobalOptResult(
            start_energy=0.0, rounds_run=1, stopped_reason="queue_exhausted",
            accepted_minima=[], round_summaries=[{"round": 0, "sources": 1, "candidates_optimized": 4, "accepted": 0}],
        )

    monkeypatch.setattr(
        hessian_sample_module, "run_hessian_global_optimization",
        fake_run_hessian_global_optimization,
    )

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "global_out_empty"
    _call_hessian_global(structure="O", inputs=inputs_fp, output=output_dir)

    assert not (output_dir / "accepted_minima.xyz").exists()
    out = capsys.readouterr().out
    assert "No new minima were accepted." in out
