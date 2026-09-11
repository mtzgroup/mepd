from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import typer
from qcdata import Structure

import mepd.hessian_sample as hessian_sample_module
from mepd.cli import _load_structure_from_smiles_or_xyz, hessian_sample as cli_hessian_sample
from mepd.hessian_sample import HessianSampleResult
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


def test_hessian_sample_command_rejects_nonpositive_dr(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(dr=0.0, output=tmp_path / "out")


def test_hessian_sample_command_rejects_nonpositive_max_candidates(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_hessian_sample(max_candidates=0, output=tmp_path / "out")


def test_hessian_sample_command_writes_discovered_minima(tmp_path, monkeypatch, capsys):
    minimum_a = StructureNode(structure=_water(x_offset=0.1), _cached_energy=-1.0)
    minimum_b = StructureNode(structure=_water(x_offset=-0.1), _cached_energy=-2.0)

    def fake_run_hessian_sample(seed_node, engine, *, dr, max_candidates):
        return HessianSampleResult(
            seed_energy=0.0,
            candidates_generated=4,
            candidates_clipped=False,
            minima=[minimum_a, minimum_b],
            skipped_not_lower_energy=1,
            skipped_duplicate=1,
            skipped_failed_optimization=0,
        )

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_out"
    _call_hessian_sample(structure="O", inputs=inputs_fp, output=output_dir)

    assert (output_dir / "minimum_001.xyz").exists()
    assert (output_dir / "minimum_002.xyz").exists()
    out = capsys.readouterr().out
    assert "Generated 4 candidate(s)" in out
    assert "Wrote" in out


def test_hessian_sample_command_reports_no_minima_found(tmp_path, monkeypatch, capsys):
    def fake_run_hessian_sample(seed_node, engine, *, dr, max_candidates):
        return HessianSampleResult(
            seed_energy=0.0,
            candidates_generated=4,
            candidates_clipped=True,
            minima=[],
            skipped_not_lower_energy=4,
            skipped_duplicate=0,
            skipped_failed_optimization=0,
        )

    monkeypatch.setattr(hessian_sample_module, "run_hessian_sample", fake_run_hessian_sample)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "hessian_sample_out_empty"
    _call_hessian_sample(structure="O", inputs=inputs_fp, output=output_dir)

    out = capsys.readouterr().out
    assert "Reached --max-candidates" in out
    assert "No new lower-energy minima were found." in out
    assert not output_dir.exists()
