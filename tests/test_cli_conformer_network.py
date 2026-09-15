from __future__ import annotations

import subprocess

import numpy as np
import pytest
import typer

from mepd.cli import conformer_network as cli_conformer_network
from mepd.inputs import RunInputs
from mepd.pot import Pot


def _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch):
    """Same rationale as the identically-named helper in test_cli_run.py /
    test_cli_network_splits.py: tying energy to geometry keeps genuinely
    different structures distinguishable instead of all reading as
    "identical" to MSMEP's endpoint check."""

    def fake_run(cmd, cwd, env, text, capture_output, check):
        xyz_path = cwd / cmd[1]
        lines = xyz_path.read_text().splitlines()
        coords = np.array([
            [float(x) for x in line.split()[1:4]] for line in lines[2:2 + int(lines[0])]
        ])
        energy = -76.0 + 0.01 * float(np.sum(coords**2))
        (cwd / "energy").write_text(f"$energy\n     1   {energy:.8f}   {energy:.8f}   {energy:.8f}\n$end\n")
        grad_lines = "\n".join(
            f"   {1.0E-03:.4E}   {0.0:.4E}   {2.0E-03:.4E}" for _ in coords
        )
        (cwd / "gradient").write_text(grad_lines + "\n")
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


def _call_conformer_network(**overrides):
    """Calling a Typer-decorated function directly does not resolve
    `typer.Option(...)` defaults, so every parameter must be given an
    explicit value (same rationale as the other CLI tests)."""
    kwargs = dict(
        start=None,
        end=None,
        inputs=None,
        charge=0,
        multiplicity=1,
        realign_atoms=False,
        backend="rdkit",
        n_conformers=10,
        n_embed=30,
        rmsd_cutoff=0.5,
        random_seed=0,
        minimize_ends=False,
        max_pairs=100,
        parallel=False,
        parallel_workers=None,
        validate_minima_with_hessian=False,
        hessian_minimum_frequency_cutoff=0.0,
        hessian_minima_rescue_displacement=0.1,
        output=None,
    )
    kwargs.update(overrides)
    return cli_conformer_network(**kwargs)


def test_conformer_network_rejects_nonpositive_n_conformers(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_conformer_network(start="CCCC", end="CC(C)C", n_conformers=0, output=tmp_path / "out")


def test_conformer_network_rejects_nonpositive_n_embed(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_conformer_network(start="CCCC", end="CC(C)C", n_embed=0, output=tmp_path / "out")


def test_conformer_network_rejects_nonpositive_max_pairs(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_conformer_network(start="CCCC", end="CC(C)C", max_pairs=0, output=tmp_path / "out")


def test_conformer_network_runs_all_pairs_and_builds_network(tmp_path, monkeypatch):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    # Butane / isobutane: same formula (so geodesic interpolation between any
    # conformer pair is well-posed), genuinely different connectivity (so no
    # pair ever degenerates to an identical reactant==product structure), and
    # both have real conformational freedom. Only the plumbing (generation ->
    # pairing -> MSMEP -> network) is under test here, not real chemistry.
    _call_conformer_network(
        start="CCCC", end="CC(C)C", inputs=inputs_fp,
        n_conformers=2, n_embed=30, output=output_dir,
    )

    pairs_dir = output_dir / "pairs"
    completed = [p for p in pairs_dir.iterdir() if (p / "tree" / "adj_matrix.txt").exists()]
    assert len(completed) >= 1

    network_path = output_dir / "network.json"
    assert network_path.exists()
    pot = Pot.read_from_disk(network_path)
    assert pot.number_of_nodes >= 1


def test_conformer_network_caps_at_max_pairs(tmp_path, monkeypatch, capsys):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_conformer_network(
        start="CCCC", end="CC(C)C", inputs=inputs_fp,
        n_conformers=2, n_embed=30, max_pairs=1, output=output_dir,
    )

    pairs_dir = output_dir / "pairs"
    completed = [p for p in pairs_dir.iterdir() if (p / "tree" / "adj_matrix.txt").exists()]
    assert len(completed) == 1
    out = capsys.readouterr().out
    assert "capping at --max-pairs=1" in out


def test_conformer_network_is_resumable_skips_completed_pairs(tmp_path, monkeypatch, capsys):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_conformer_network(
        start="CCCC", end="CC(C)C", inputs=inputs_fp,
        n_conformers=1, n_embed=10, output=output_dir,
    )
    _call_conformer_network(
        start="CCCC", end="CC(C)C", inputs=inputs_fp,
        n_conformers=1, n_embed=10, output=output_dir,
    )
    out = capsys.readouterr().out
    assert "Skipping pair (0, 1): already completed." in out
