from __future__ import annotations

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
        dr=0.1,
        max_candidates=100,
        maxiter=500,
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


def _meta(mode_index=0, direction="+", freq=100.0):
    return HessianSampleCandidate(
        mode_index=mode_index, direction=direction, frequency_wavenumber=freq,
        dr=0.1, effective_dr=0.3,
    )


def test_hessian_sample_command_writes_full_output_set(tmp_path, monkeypatch, capsys):
    displaced = [StructureNode(structure=_water(x_offset=0.05), _cached_energy=None) for _ in range(2)]
    minimum_a = StructureNode(structure=_water(x_offset=0.1), _cached_energy=-1.0)
    minimum_b = StructureNode(structure=_water(x_offset=-0.1), _cached_energy=-2.0)

    def fake_run_hessian_sample(seed_node, engine, *, dr, max_candidates, maxiter, chain_inputs, on_event=None):
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
    def fake_run_hessian_sample(seed_node, engine, *, dr, max_candidates, maxiter, chain_inputs, on_event=None):
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
