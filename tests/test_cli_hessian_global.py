from __future__ import annotations

import json

import pytest
import typer

import mepd.discovery.hessian_sample as hessian_sample_module
from mepd.discovery.cli import hessian_global as cli_hessian_global
from mepd.discovery.hessian_sample import HessianGlobalOptResult
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
        full_dr_scan=False,
        dr_scan_values="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
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


def test_hessian_global_command_streams_minima_as_found(tmp_path, monkeypatch, capsys):
    """The CLI must write each accepted minimum out (and print a discovery
    line) as `on_event("minimum_accepted", ...)` fires -- not only after the
    whole search returns -- so results are usable (e.g. for `mepd run`)
    without waiting on the full run."""
    minimum_a = _water(x_offset=0.1, energy=-1.0)
    minimum_b = _water(x_offset=-0.1, energy=-2.0)

    def fake_run_hessian_global_optimization(seed_node, engine, **kwargs):
        on_event = kwargs.get("on_event")
        if on_event is not None:
            on_event(
                "minimum_accepted",
                {"round": 0, "index": 1, "node": minimum_a, "rel_energy_kcal": -1.0},
            )
            on_event(
                "minimum_accepted",
                {"round": 0, "index": 2, "node": minimum_b, "rel_energy_kcal": -2.0},
            )
        return HessianGlobalOptResult(
            start_energy=0.0, rounds_run=1, stopped_reason="queue_exhausted",
            accepted_minima=[minimum_a, minimum_b],
            round_summaries=[{"round": 0, "sources": 1, "candidates_optimized": 4, "accepted": 2}],
        )

    monkeypatch.setattr(
        hessian_sample_module, "run_hessian_global_optimization",
        fake_run_hessian_global_optimization,
    )

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "global_out_stream"
    _call_hessian_global(structure="O", inputs=inputs_fp, output=output_dir)

    out = capsys.readouterr().out
    assert "New minimum #1" in out
    assert "New minimum #2" in out
    assert "ΔE=-1.00 kcal/mol" in out
    assert "ΔE=-2.00 kcal/mol" in out
    assert (output_dir / "accepted_minima.xyz").exists()


def test_hessian_global_command_rejects_invalid_dr_scan_values(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(
            full_dr_scan=True, dr_scan_values="0.1,not-a-number", output=tmp_path / "out",
        )
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(full_dr_scan=True, dr_scan_values="0.1,-0.2", output=tmp_path / "out")
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(full_dr_scan=True, dr_scan_values="", output=tmp_path / "out")


def test_hessian_global_command_forwards_dr_scan_values_when_enabled(tmp_path, monkeypatch):
    captured_kwargs = {}

    def fake_run_hessian_global_optimization(seed_node, engine, **kwargs):
        captured_kwargs.update(kwargs)
        return HessianGlobalOptResult(
            start_energy=0.0, rounds_run=1, stopped_reason="queue_exhausted",
            accepted_minima=[], round_summaries=[],
        )

    monkeypatch.setattr(
        hessian_sample_module, "run_hessian_global_optimization",
        fake_run_hessian_global_optimization,
    )

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "global_out_scan"
    _call_hessian_global(
        structure="O", inputs=inputs_fp, output=output_dir,
        full_dr_scan=True, dr_scan_values="0.2,0.4",
    )

    assert captured_kwargs["dr_values"] == [0.2, 0.4]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["full_dr_scan"] is True
    assert summary["dr_scan_values"] == [0.2, 0.4]


def test_hessian_global_command_dr_values_omitted_when_scan_disabled(tmp_path, monkeypatch):
    captured_kwargs = {}

    def fake_run_hessian_global_optimization(seed_node, engine, **kwargs):
        captured_kwargs.update(kwargs)
        return HessianGlobalOptResult(
            start_energy=0.0, rounds_run=1, stopped_reason="queue_exhausted",
            accepted_minima=[], round_summaries=[],
        )

    monkeypatch.setattr(
        hessian_sample_module, "run_hessian_global_optimization",
        fake_run_hessian_global_optimization,
    )

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "global_out_noscan"
    _call_hessian_global(structure="O", inputs=inputs_fp, output=output_dir)

    assert captured_kwargs["dr_values"] is None
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["full_dr_scan"] is False
    assert summary["dr_scan_values"] == []


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
