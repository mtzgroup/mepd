from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import typer
from qcdata import Structure

import mepd.discovery.hessian_sample as hessian_sample_module
from mepd.cli import _load_structure_from_smiles_or_xyz
from mepd.discovery.cli import hessian_sample as cli_hessian_sample
from mepd.discovery.hessian_sample import HessianSampleCandidate, HessianSampleResult
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode


def _install_fake_gxtb(monkeypatch, calls=None):
    """Same fake gxtb executable as test_cli_run.py's helper of the same
    name -- lets `compute_geometry_optimization` run for real (against a
    scripted subprocess) without a real gxtb binary."""

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
        charge=0,
        multiplicity=1,
    )


def _run_inputs_for_test() -> RunInputs:
    return RunInputs(
        engine_name="gxtb",
        gxtb_engine_kwds={"executable": "gxtb"},
    )


def _call_hessian_sample(**overrides):
    """Same rationale as `_call_run`/`_call_ts` in test_cli_run.py -- calling a
    Typer-decorated function directly does not resolve `typer.Option(...)`
    defaults, so every parameter must be given an explicit value."""
    kwargs = dict(
        structure="O",
        inputs=None,
        charge=None,
        multiplicity=None,
        minimize_seed=False,
        dr=0.1,
        amplitude_policy="fixed-cartesian",
        target_energy_kcal=25.0,
        full_dr_scan=False,
        dr_scan_values="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        full_energy_scan=False,
        energy_scan_values_kcal="25,50,100,200,400",
        imaginary_mode_amplitude=0.3,
        max_candidates=100,
        maxiter=500,
        validate_minima_with_hessian=False,
        hessian_minimum_frequency_cutoff=0.0,
        hessian_minima_rescue_displacement=0.1,
        output=None,
    )
    kwargs.update(overrides)
    return cli_hessian_sample(**kwargs)


def test_load_structure_from_smiles_or_xyz_uses_existing_file(tmp_path):
    xyz_fp = tmp_path / "water.xyz"
    xyz_fp.write_text(_water().to_xyz())

    structure = _load_structure_from_smiles_or_xyz(str(xyz_fp), None, None)

    assert list(structure.symbols) == ["O", "H", "H"]


def test_load_structure_from_smiles_or_xyz_embeds_smiles():
    structure = _load_structure_from_smiles_or_xyz("O", None, None)

    assert sorted(structure.symbols) == ["H", "H", "O"]
    assert structure.geometry.shape == (3, 3)


def test_load_structure_from_smiles_or_xyz_applies_charge_and_multiplicity():
    structure = _load_structure_from_smiles_or_xyz("[OH-]", None, None)
    assert structure.charge == -1


def test_load_structure_from_smiles_or_xyz_rejects_invalid_input():
    with pytest.raises(typer.BadParameter):
        _load_structure_from_smiles_or_xyz("not_a_valid_smiles(((", None, None)


def test_load_structure_from_smiles_or_xyz_falls_back_to_openbabel_for_multi_fragment_smiles():
    # RDKit (the default backend) refuses multi-fragment SMILES outright;
    # openbabel embeds them fine -- exactly the kind of noncovalent complex
    # (a solute plus explicit waters) discovery commands want to explore.
    structure = _load_structure_from_smiles_or_xyz("C=C.O.O.O", None, None)

    assert sorted(structure.symbols).count("O") == 3
    assert sorted(structure.symbols).count("C") == 2
    assert structure.geometry.shape[0] == len(structure.symbols)


def test_hessian_sample_command_rejects_nonpositive_dr(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(dr=0.0, output=tmp_path / "out")


def test_hessian_sample_command_rejects_nonpositive_max_candidates(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(max_candidates=0, output=tmp_path / "out")


def test_hessian_sample_command_rejects_nonpositive_maxiter(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(maxiter=0, output=tmp_path / "out")


def test_hessian_sample_command_rejects_invalid_amplitude_policy(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(amplitude_policy="nonsense", output=tmp_path / "out")


def test_hessian_sample_command_rejects_nonpositive_target_energy_kcal(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(target_energy_kcal=0.0, output=tmp_path / "out")


def _meta(mode_index=0, direction="+", freq=100.0):
    return HessianSampleCandidate(
        mode_index=mode_index, direction=direction, frequency_wavenumber=freq,
        dr=0.1, effective_dr=0.3,
    )


def test_hessian_sample_command_writes_full_output_set(tmp_path, monkeypatch, capsys):
    displaced = [StructureNode(structure=_water(x_offset=0.05), _cached_energy=None) for _ in range(2)]
    minimum_a = StructureNode(structure=_water(x_offset=0.1), _cached_energy=-1.0)
    minimum_b = StructureNode(structure=_water(x_offset=-0.1), _cached_energy=-2.0)

    def fake_run_hessian_sample(seed_node, engine, *, dr, amplitude_policy, target_energy_kcal, imaginary_mode_amplitude, max_candidates, maxiter, chain_inputs, on_event=None, **kwargs):
        return HessianSampleResult(
            seed_energy=0.0,
            hessian_result=None,
            frequencies_wavenumber=[100.0, 150.0],
            displaced_nodes=displaced,
            displaced_metadata=[_meta(0, "+"), _meta(0, "-")],
            candidates_clipped=False,
            optimization_submission_mode="serial",
            optimized_nodes=[minimum_a, minimum_b],
            optimized_metadata=[_meta(0, "+"), _meta(1, "+")],
            failed_candidates=[{"meta": _meta(1, "-"), "error": "boom"}],
            unique_minima=[minimum_a, minimum_b],
        )

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_out"
    _call_hessian_sample(structure="O", inputs=inputs_fp, output=output_dir)

    assert (output_dir / "displaced.xyz").exists()
    assert (output_dir / "optimized.xyz").exists()
    assert (output_dir / "unique.xyz").exists()
    assert (output_dir / "summary.json").exists()

    import json
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["optimized_candidates"] == 2
    assert summary["failed_candidates"] == 1
    assert summary["unique_minima"] == 2
    assert summary["optimization_submission_mode"] == "serial"
    assert len(summary["unique_minima_rel_energies_kcal_mol"]) == 2
    assert len(summary["failed_candidate_details"]) == 1
    assert summary["failed_candidate_details"][0]["error"] == "boom"

    out = capsys.readouterr().out
    assert "Hessian Sample Complete" in out


def test_hessian_sample_command_exits_nonzero_when_all_optimizations_fail(tmp_path, monkeypatch, capsys):
    def fake_run_hessian_sample(seed_node, engine, *, dr, amplitude_policy, target_energy_kcal, imaginary_mode_amplitude, max_candidates, maxiter, chain_inputs, on_event=None, **kwargs):
        return HessianSampleResult(
            seed_energy=0.0,
            hessian_result=None,
            frequencies_wavenumber=[100.0],
            displaced_nodes=[StructureNode(structure=_water(), _cached_energy=None)],
            displaced_metadata=[_meta()],
            candidates_clipped=True,
            optimization_submission_mode="serial",
            optimized_nodes=[],
            optimized_metadata=[],
            failed_candidates=[{"meta": _meta(), "error": "boom"}],
            unique_minima=[],
        )

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_out_empty"
    with pytest.raises(typer.Exit):
        _call_hessian_sample(structure="O", inputs=inputs_fp, output=output_dir)

    out = capsys.readouterr().out
    assert "Reached --max-candidates" in out
    assert "All displaced-candidate optimizations failed." in out
    assert (output_dir / "summary.json").exists()


def test_hessian_sample_command_minimizes_seed_when_requested(tmp_path, monkeypatch, capsys):
    calls = []
    _install_fake_gxtb(monkeypatch, calls=calls)

    seed_nodes_seen = []
    minimum = StructureNode(structure=_water(x_offset=0.1), _cached_energy=-1.0)

    def fake_run_hessian_sample(seed_node, engine, *, dr, amplitude_policy, target_energy_kcal, imaginary_mode_amplitude, max_candidates, maxiter, chain_inputs, on_event=None, **kwargs):
        seed_nodes_seen.append(seed_node)
        return HessianSampleResult(
            seed_energy=0.0,
            hessian_result=None,
            frequencies_wavenumber=[100.0],
            displaced_nodes=[minimum],
            displaced_metadata=[_meta()],
            optimized_nodes=[minimum],
            optimized_metadata=[_meta()],
            unique_minima=[minimum],
        )

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_min_seed"
    _call_hessian_sample(structure="O", inputs=inputs_fp, minimize_seed=True, output=output_dir)

    assert any("--opt" in call for call in calls)
    assert seed_nodes_seen, "expected run_hessian_sample to be called with the minimized seed"

    out = capsys.readouterr().out
    assert "Minimizing seed structure" in out

    import json
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["minimize_seed"] is True


def test_hessian_sample_command_minimize_seed_hard_stops_on_nonconvergence(tmp_path, monkeypatch, capsys):
    _install_fake_gxtb_unconverged_opt(monkeypatch)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_min_seed_fail"
    with pytest.raises(typer.Exit) as exc_info:
        _call_hessian_sample(structure="O", inputs=inputs_fp, minimize_seed=True, output=output_dir)

    assert exc_info.value.exit_code == 1
    out = capsys.readouterr().out
    assert "did not converge" in out
    assert "--minimize-seed" in out
    assert not output_dir.exists() or not (output_dir / "summary.json").exists()


def test_hessian_sample_command_forwards_hessian_validation_flags_and_writes_rejected(tmp_path, monkeypatch):
    accepted = StructureNode(structure=_water(x_offset=0.1), _cached_energy=-1.0)
    rejected = StructureNode(structure=_water(x_offset=-0.1), _cached_energy=-2.0)
    captured_kwargs = {}

    def fake_run_hessian_sample(seed_node, engine, *, dr, amplitude_policy, target_energy_kcal, imaginary_mode_amplitude, max_candidates, maxiter, chain_inputs, on_event=None, **kwargs):
        captured_kwargs.update(kwargs)
        return HessianSampleResult(
            seed_energy=0.0,
            hessian_result=None,
            frequencies_wavenumber=[100.0],
            optimized_nodes=[accepted, rejected],
            optimized_metadata=[_meta(0, "+"), _meta(1, "+")],
            unique_minima=[accepted],
            hessian_validation_enabled=True,
            rejected_minima=[rejected],
            hessian_validation_rescue_grad_calls=3,
        )

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_validate"
    _call_hessian_sample(
        structure="O", inputs=inputs_fp, output=output_dir,
        validate_minima_with_hessian=True,
        hessian_minimum_frequency_cutoff=25.0,
        hessian_minima_rescue_displacement=0.2,
    )

    assert captured_kwargs["validate_minima_with_hessian"] is True
    assert captured_kwargs["hessian_minimum_frequency_cutoff"] == 25.0
    assert captured_kwargs["hessian_minima_rescue_displacement"] == 0.2
    assert (output_dir / "rejected.xyz").exists()

    import json
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["validate_minima_with_hessian"] is True
    assert summary["hessian_minimum_frequency_cutoff"] == 25.0
    assert summary["hessian_minima_rescue_displacement"] == 0.2
    assert summary["unique_minima"] == 1
    assert summary["rejected_minima"] == 1
    assert summary["hessian_validation_rescue_grad_calls"] == 3
    assert summary["output_files"]["rejected"] is not None


def test_hessian_sample_command_no_rejected_file_when_validation_disabled(tmp_path, monkeypatch):
    minimum = StructureNode(structure=_water(x_offset=0.1), _cached_energy=-1.0)

    def fake_run_hessian_sample(seed_node, engine, *, dr, amplitude_policy, target_energy_kcal, imaginary_mode_amplitude, max_candidates, maxiter, chain_inputs, on_event=None, **kwargs):
        return HessianSampleResult(
            seed_energy=0.0,
            hessian_result=None,
            frequencies_wavenumber=[100.0],
            optimized_nodes=[minimum],
            optimized_metadata=[_meta()],
            unique_minima=[minimum],
        )

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_no_validate"
    _call_hessian_sample(structure="O", inputs=inputs_fp, output=output_dir)

    assert not (output_dir / "rejected.xyz").exists()
    import json
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["rejected_minima"] == 0
    assert summary["output_files"]["rejected"] is None


def test_hessian_sample_command_rejects_invalid_dr_scan_values(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(
            full_dr_scan=True, dr_scan_values="0.1,not-a-number", output=tmp_path / "out",
        )
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(full_dr_scan=True, dr_scan_values="0.1,-0.2", output=tmp_path / "out")
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(full_dr_scan=True, dr_scan_values="", output=tmp_path / "out")


def test_hessian_sample_command_full_dr_scan_overrides_conflicting_energy_policy(tmp_path, monkeypatch, capsys):
    """--full-dr-scan unambiguously implies fixed-cartesian -- an explicit,
    conflicting --amplitude-policy=energy is overridden (with a warning)
    rather than rejected, so the flag alone is enough without also spelling
    out --amplitude-policy."""
    captured_kwargs = {}

    def fake_run_hessian_sample(seed_node, engine, *, dr, amplitude_policy, target_energy_kcal, imaginary_mode_amplitude, max_candidates, maxiter, chain_inputs, on_event=None, **kwargs):
        captured_kwargs["amplitude_policy"] = amplitude_policy
        captured_kwargs.update(kwargs)
        return HessianSampleResult(seed_energy=0.0, hessian_result=None, frequencies_wavenumber=[100.0])

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    with pytest.raises(typer.Exit):
        _call_hessian_sample(
            amplitude_policy="energy", full_dr_scan=True, dr_scan_values="0.2,0.4",
            inputs=inputs_fp, output=output_dir,
        )

    assert captured_kwargs["amplitude_policy"] == "fixed_cartesian"
    out = capsys.readouterr().out
    assert "--full-dr-scan implies --amplitude-policy=fixed-cartesian" in out


def test_hessian_sample_command_rejects_invalid_energy_scan_values(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(
            amplitude_policy="energy", full_energy_scan=True,
            energy_scan_values_kcal="10,not-a-number", output=tmp_path / "out",
        )
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(
            amplitude_policy="energy", full_energy_scan=True,
            energy_scan_values_kcal="10,-20", output=tmp_path / "out",
        )
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(
            amplitude_policy="energy", full_energy_scan=True,
            energy_scan_values_kcal="", output=tmp_path / "out",
        )


def test_hessian_sample_command_full_energy_scan_infers_energy_policy(tmp_path, monkeypatch, capsys):
    """--full-energy-scan unambiguously implies the energy policy -- using it
    without also passing --amplitude-policy=energy should just work, with a
    warning, rather than requiring the redundant flag."""
    captured_kwargs = {}

    def fake_run_hessian_sample(seed_node, engine, *, dr, amplitude_policy, target_energy_kcal, imaginary_mode_amplitude, max_candidates, maxiter, chain_inputs, on_event=None, **kwargs):
        captured_kwargs["amplitude_policy"] = amplitude_policy
        captured_kwargs.update(kwargs)
        return HessianSampleResult(seed_energy=0.0, hessian_result=None, frequencies_wavenumber=[100.0])

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    with pytest.raises(typer.Exit):
        _call_hessian_sample(
            full_energy_scan=True, energy_scan_values_kcal="100,300",
            inputs=inputs_fp, output=output_dir,
        )

    assert captured_kwargs["amplitude_policy"] == "energy"
    out = capsys.readouterr().out
    assert "--full-energy-scan implies --amplitude-policy=energy" in out


def test_hessian_sample_command_rejects_both_scans_at_once(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(
            full_dr_scan=True, full_energy_scan=True, output=tmp_path / "out",
        )


def test_hessian_sample_command_forwards_dr_scan_values_when_enabled(tmp_path, monkeypatch):
    captured_kwargs = {}

    def fake_run_hessian_sample(seed_node, engine, *, dr, amplitude_policy, target_energy_kcal, imaginary_mode_amplitude, max_candidates, maxiter, chain_inputs, on_event=None, **kwargs):
        captured_kwargs.update(kwargs)
        captured_kwargs["dr"] = dr
        return HessianSampleResult(
            seed_energy=0.0, hessian_result=None, frequencies_wavenumber=[100.0],
        )

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_dr_scan"
    with pytest.raises(typer.Exit):
        # No optimized candidates in the canned result -- exits nonzero, but
        # only after the summary (and the kwargs we're checking) are written.
        _call_hessian_sample(
            structure="O", inputs=inputs_fp, output=output_dir,
            full_dr_scan=True, dr_scan_values="0.2,0.4",
        )

    assert captured_kwargs["dr_values"] == [0.2, 0.4]
    import json
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["full_dr_scan"] is True
    assert summary["dr_scan_values"] == [0.2, 0.4]


def test_hessian_sample_command_forwards_energy_scan_values_when_enabled(tmp_path, monkeypatch):
    captured_kwargs = {}

    def fake_run_hessian_sample(seed_node, engine, *, dr, amplitude_policy, target_energy_kcal, imaginary_mode_amplitude, max_candidates, maxiter, chain_inputs, on_event=None, **kwargs):
        captured_kwargs.update(kwargs)
        captured_kwargs["amplitude_policy"] = amplitude_policy
        return HessianSampleResult(
            seed_energy=0.0, hessian_result=None, frequencies_wavenumber=[100.0],
        )

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_energy_scan"
    with pytest.raises(typer.Exit):
        _call_hessian_sample(
            structure="O", inputs=inputs_fp, output=output_dir,
            amplitude_policy="energy", full_energy_scan=True, energy_scan_values_kcal="100,300",
        )

    assert captured_kwargs["amplitude_policy"] == "energy"
    assert captured_kwargs["target_energy_kcal_values"] == [100.0, 300.0]
    import json
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["full_energy_scan"] is True
    assert summary["energy_scan_values_kcal"] == [100.0, 300.0]
