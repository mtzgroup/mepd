from __future__ import annotations

import subprocess

import numpy as np
import pytest
from qcdata import Structure

from mepd.cli import _build_path_minimizer, run as cli_run
from mepd.chain import Chain
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


def _install_fake_gxtb(monkeypatch, calls=None):
    def fake_run(cmd, cwd, env, text, capture_output, check):
        if calls is not None:
            calls.append(cmd)
        if "--opt" in cmd:
            xyz_path = cwd / cmd[1]
            natoms_line = xyz_path.read_text().splitlines()[0]
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


def _run_inputs_for_test() -> RunInputs:
    return RunInputs(
        engine_name="gxtb",
        path_min_method="NEB",
        gxtb_engine_kwds={"executable": "gxtb"},
        gi_inputs={"nimages": 4},
        path_min_inputs={"max_steps": 2, "v": False, "do_elem_step_checks": False},
    )


def test_build_path_minimizer_dispatches_neb(monkeypatch):
    _install_fake_gxtb(monkeypatch)
    run_inputs = _run_inputs_for_test()
    node = StructureNode(structure=_water())
    chain = Chain.model_validate(
        {"nodes": [node, node.copy()], "parameters": run_inputs.chain_inputs}
    )
    minimizer = _build_path_minimizer(chain, run_inputs)
    from mepd.neb import NEB

    assert isinstance(minimizer, NEB)


def test_build_path_minimizer_rejects_unsupported_method(monkeypatch):
    _install_fake_gxtb(monkeypatch)
    run_inputs = _run_inputs_for_test()
    run_inputs.path_min_method = "MLPGI"
    node = StructureNode(structure=_water())
    chain = Chain.model_validate(
        {"nodes": [node, node.copy()], "parameters": run_inputs.chain_inputs}
    )
    with pytest.raises(Exception):
        _build_path_minimizer(chain, run_inputs)


def test_cli_run_writes_trajectory(tmp_path, monkeypatch):
    calls = []
    _install_fake_gxtb(monkeypatch, calls=calls)

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(0.3).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    cli_run(
        start=start_fp,
        end=end_fp,
        inputs=inputs_fp,
        charge=None,
        multiplicity=None,
        output=output_dir,
    )

    assert (output_dir / "mep_output.xyz").exists()
    assert len(calls) > 0


def test_cli_run_applies_charge_and_multiplicity(tmp_path, monkeypatch):
    seen_structures = []
    original_init = StructureNode.__init__

    def spying_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        seen_structures.append(self.structure)

    monkeypatch.setattr(StructureNode, "__init__", spying_init)
    _install_fake_gxtb(monkeypatch)

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(0.3).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    cli_run(
        start=start_fp,
        end=end_fp,
        inputs=inputs_fp,
        charge=1,
        multiplicity=2,
        output=output_dir,
    )

    assert seen_structures, "expected StructureNode to be constructed"
    assert seen_structures[0].charge == 1
    assert seen_structures[0].multiplicity == 2


def test_cli_run_minimize_ends(tmp_path, monkeypatch, capsys):
    calls = []
    _install_fake_gxtb(monkeypatch, calls=calls)

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(0.3).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    cli_run(
        start=start_fp,
        end=end_fp,
        inputs=inputs_fp,
        charge=None,
        multiplicity=None,
        minimize_ends=True,
        output=output_dir,
    )

    assert (output_dir / "mep_output.xyz").exists()
    assert any("--opt" in call for call in calls)
    out = capsys.readouterr().out
    assert "Minimizing input endpoints" in out
