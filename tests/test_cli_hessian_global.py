from __future__ import annotations

import json
import subprocess

import pytest
import typer

import mepd.discovery.hessian_sample as hessian_sample_module
from mepd.discovery.cli import hessian_global as cli_hessian_global
from mepd.discovery.hessian_sample import HessianGlobalOptResult
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode
from qcdata import Structure
import numpy as np


def _install_fake_gxtb(monkeypatch, calls=None):
    """Same fake gxtb executable as test_cli_run.py's/test_cli_hessian_sample.py's
    helper of the same name -- lets `compute_geometry_optimization` run for
    real (against a scripted subprocess) without a real gxtb binary."""

    def fake_run(cmd, cwd, env, text, capture_output, check):
        if calls is not None:
            calls.append(cmd)
        if "--opt" in cmd:
            xyz_path = cwd / cmd[1]
            (cwd / "xtbopt.xyz").write_text(
                xyz_path.read_text().replace("Frame 0", "energy: -76.0")
            )
        else:
            (cwd / "energy").write_text(
                "$energy\n     1   -76.0   -76.0   -76.0\n$end\n"
            )
            (cwd / "gradient").write_text(
                "   1.0E-03   0.0E+00   2.0E-03\n"
                "  -1.0E-03   0.0E+00  -1.0E-03\n"
                "   0.0E+00   0.0E+00  -1.0E-03\n"
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="normal termination", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def _install_fake_gxtb_unconverged_opt(monkeypatch, calls=None):
    """Fake gxtb whose --opt run reports FAILED TO CONVERGE, mirroring a real
    optimization that exhausted its iteration budget without converging."""

    def fake_run(cmd, cwd, env, text, capture_output, check):
        if calls is not None:
            calls.append(cmd)
        if "--opt" in cmd:
            xyz_path = cwd / cmd[1]
            (cwd / "xtbopt.xyz").write_text(
                xyz_path.read_text().replace("Frame 0", "energy: -76.0")
            )
            stdout = "   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 1 ITERATIONS ***\n"
        else:
            (cwd / "energy").write_text(
                "$energy\n     1   -76.0   -76.0   -76.0\n$end\n"
            )
            (cwd / "gradient").write_text(
                "   1.0E-03   0.0E+00   2.0E-03\n"
                "  -1.0E-03   0.0E+00  -1.0E-03\n"
                "   0.0E+00   0.0E+00  -1.0E-03\n"
            )
            stdout = "normal termination"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


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
        minimize_seed=False,
        dr=0.1,
        amplitude_policy="fixed-cartesian",
        target_energy_kcal=25.0,
        imaginary_mode_amplitude=0.3,
        max_candidates=100,
        maxiter=500,
        temperature=298.15,
        energy_tolerance_kcal=1.0e-4,
        max_rounds=100,
        random_seed=None,
        acceptance_baseline="connected",
        full_dr_scan=False,
        dr_scan_values="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        full_energy_scan=False,
        energy_scan_values_kcal="25,50,100,200,400",
        validate_minima_with_hessian=False,
        hessian_minimum_frequency_cutoff=0.0,
        hessian_minima_rescue_displacement=0.1,
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


def test_hessian_global_command_rejects_invalid_acceptance_baseline(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(acceptance_baseline="nonsense", output=tmp_path / "out")


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


def test_hessian_global_command_minimizes_seed_when_requested(tmp_path, monkeypatch, capsys):
    calls = []
    _install_fake_gxtb(monkeypatch, calls=calls)

    seed_nodes_seen = []
    minimum = _water(x_offset=0.1, energy=-1.0)

    def fake_run_hessian_global_optimization(seed_node, engine, **kwargs):
        seed_nodes_seen.append(seed_node)
        return HessianGlobalOptResult(
            start_energy=0.0, rounds_run=1, stopped_reason="queue_exhausted",
            accepted_minima=[minimum],
            round_summaries=[{"round": 0, "sources": 1, "candidates_optimized": 4, "accepted": 1}],
        )

    monkeypatch.setattr(
        hessian_sample_module, "run_hessian_global_optimization",
        fake_run_hessian_global_optimization,
    )

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "global_out_min_seed"
    _call_hessian_global(structure="O", inputs=inputs_fp, minimize_seed=True, output=output_dir)

    assert any("--opt" in call for call in calls)
    assert seed_nodes_seen, "expected run_hessian_global_optimization to be called with the minimized seed"

    out = capsys.readouterr().out
    assert "Minimizing seed structure" in out

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["minimize_seed"] is True


def test_hessian_global_command_minimize_seed_hard_stops_on_nonconvergence(tmp_path, monkeypatch, capsys):
    _install_fake_gxtb_unconverged_opt(monkeypatch)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "global_out_min_seed_fail"
    with pytest.raises(typer.Exit) as exc_info:
        _call_hessian_global(structure="O", inputs=inputs_fp, minimize_seed=True, output=output_dir)

    assert exc_info.value.exit_code == 1
    out = capsys.readouterr().out
    assert "did not converge" in out
    assert "--minimize-seed" in out
    assert not output_dir.exists() or not (output_dir / "summary.json").exists()


def test_hessian_global_command_forwards_hessian_validation_flags(tmp_path, monkeypatch):
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

    output_dir = tmp_path / "global_out_validate"
    _call_hessian_global(
        structure="O", inputs=inputs_fp, output=output_dir,
        validate_minima_with_hessian=True,
        hessian_minimum_frequency_cutoff=25.0,
        hessian_minima_rescue_displacement=0.2,
    )

    assert captured_kwargs["validate_minima_with_hessian"] is True
    assert captured_kwargs["hessian_minimum_frequency_cutoff"] == 25.0
    assert captured_kwargs["hessian_minima_rescue_displacement"] == 0.2

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["validate_minima_with_hessian"] is True
    assert summary["hessian_minimum_frequency_cutoff"] == 25.0
    assert summary["hessian_minima_rescue_displacement"] == 0.2


def test_hessian_global_command_rejects_invalid_amplitude_policy(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(amplitude_policy="nonsense", output=tmp_path / "out")


def test_hessian_global_command_rejects_nonpositive_target_energy_kcal(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(target_energy_kcal=0.0, output=tmp_path / "out")


def test_hessian_global_command_full_dr_scan_overrides_conflicting_energy_policy(tmp_path, monkeypatch, capsys):
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

    output_dir = tmp_path / "out"
    _call_hessian_global(
        structure="O", inputs=inputs_fp, output=output_dir,
        amplitude_policy="energy", full_dr_scan=True, dr_scan_values="0.2,0.4",
    )

    assert captured_kwargs["amplitude_policy"] == "fixed_cartesian"
    out = capsys.readouterr().out
    assert "--full-dr-scan implies --amplitude-policy=fixed-cartesian" in out


def test_hessian_global_command_forwards_energy_policy_params(tmp_path, monkeypatch):
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

    output_dir = tmp_path / "global_out_energy_policy"
    _call_hessian_global(
        structure="O", inputs=inputs_fp, output=output_dir,
        amplitude_policy="energy", target_energy_kcal=15.0, imaginary_mode_amplitude=0.4,
    )

    assert captured_kwargs["amplitude_policy"] == "energy"
    assert captured_kwargs["target_energy_kcal"] == 15.0
    assert captured_kwargs["imaginary_mode_amplitude"] == 0.4

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["amplitude_policy"] == "energy"
    assert summary["target_energy_kcal"] == 15.0
    assert summary["imaginary_mode_amplitude"] == 0.4


def test_hessian_global_command_rejects_invalid_energy_scan_values(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(
            amplitude_policy="energy", full_energy_scan=True,
            energy_scan_values_kcal="10,not-a-number", output=tmp_path / "out",
        )
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(
            amplitude_policy="energy", full_energy_scan=True,
            energy_scan_values_kcal="10,-20", output=tmp_path / "out",
        )
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(
            amplitude_policy="energy", full_energy_scan=True,
            energy_scan_values_kcal="", output=tmp_path / "out",
        )


def test_hessian_global_command_full_energy_scan_infers_energy_policy(tmp_path, monkeypatch, capsys):
    """--full-energy-scan (or -E) alone, without also spelling out
    --amplitude-policy=energy, should just work -- this is the fix for the
    reported friction of `-E -ev ...` being rejected for "not" being under
    the energy policy when -E unambiguously means it should be."""
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

    output_dir = tmp_path / "out"
    _call_hessian_global(
        structure="O", inputs=inputs_fp, output=output_dir,
        full_energy_scan=True, energy_scan_values_kcal="700,1000,5000",
    )

    assert captured_kwargs["amplitude_policy"] == "energy"
    assert captured_kwargs["target_energy_kcal_values"] == [700.0, 1000.0, 5000.0]
    out = capsys.readouterr().out
    assert "--full-energy-scan implies --amplitude-policy=energy" in out


def test_hessian_global_command_rejects_both_scans_at_once(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_global(
            full_dr_scan=True, full_energy_scan=True, output=tmp_path / "out",
        )


def test_hessian_global_command_forwards_energy_scan_values_when_enabled(tmp_path, monkeypatch):
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

    output_dir = tmp_path / "global_out_energy_scan"
    _call_hessian_global(
        structure="O", inputs=inputs_fp, output=output_dir,
        amplitude_policy="energy", full_energy_scan=True, energy_scan_values_kcal="100,300",
    )

    assert captured_kwargs["amplitude_policy"] == "energy"
    assert captured_kwargs["target_energy_kcal_values"] == [100.0, 300.0]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["full_energy_scan"] is True
    assert summary["energy_scan_values_kcal"] == [100.0, 300.0]


def test_hessian_global_command_energy_scan_values_omitted_when_disabled(tmp_path, monkeypatch):
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

    output_dir = tmp_path / "global_out_no_energy_scan"
    _call_hessian_global(structure="O", inputs=inputs_fp, output=output_dir)

    assert captured_kwargs["target_energy_kcal_values"] is None
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["full_energy_scan"] is False
    assert summary["energy_scan_values_kcal"] == []
