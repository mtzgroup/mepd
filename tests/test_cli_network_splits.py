from __future__ import annotations

import subprocess

import numpy as np
import pytest
import typer
from qcdata import Structure

from mepd.cli import network_splits as cli_network_splits
from mepd.inputs import RunInputs
from mepd.pot import Pot


def _topology_structure(kind: str) -> Structure:
    """Three genuinely different C-O-H connectivities (not just conformers of
    one topology) -- see test_cli_network.py's identically-named helper for
    why this matters: two geometries that share a molecular graph can read as
    "identical" to MSMEP's endpoint check regardless of how far apart their
    raw coordinates are, short-circuiting before any real splitting runs."""
    if kind == "chain":  # C-O and O-H both bonded
        geometry = np.array([[0.0, 0.0, 0.0], [1.20, 0.0, 0.0], [2.40, 0.0, 0.0]])
    elif kind == "pair_plus_single":  # C-O bonded, O-H broken
        geometry = np.array([[0.0, 0.0, 0.0], [1.20, 0.0, 0.0], [6.40, 0.0, 0.0]])
    elif kind == "single_plus_pair":  # C-O broken, O-H bonded
        geometry = np.array([[0.0, 0.0, 0.0], [5.20, 0.0, 0.0], [6.40, 0.0, 0.0]])
    else:
        raise ValueError(kind)
    return Structure(geometry=geometry, symbols=["C", "O", "H"], charge=0, multiplicity=1)


def _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch):
    """Same rationale as the identically-named helper in test_cli_run.py: a
    fixed-energy fake gxtb makes every geometry look identical to MSMEP's
    endpoint-identity check, short-circuiting before any real splitting/
    network logic runs. Tying energy to geometry keeps genuinely different
    minima distinguishable."""

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


def _call_network_splits(**overrides):
    """Same rationale as `_call_run`/`_call_ts` in test_cli_run.py -- calling
    a Typer-decorated function directly does not resolve `typer.Option(...)`
    defaults, so every parameter must be given an explicit value."""
    kwargs = dict(
        minima=None,
        inputs=None,
        charge=None,
        multiplicity=None,
        mode="all-to-all",
        max_pairs=100,
        parallel=False,
        parallel_workers=None,
        validate_minima_with_hessian=False,
        hessian_minimum_frequency_cutoff=0.0,
        hessian_minima_rescue_displacement=0.1,
        same_pair_split_limit=5,
        output=None,
    )
    kwargs.update(overrides)
    return cli_network_splits(**kwargs)


def _write_minima(tmp_path, kinds) -> list:
    paths = []
    for i, kind in enumerate(kinds):
        fp = tmp_path / f"minimum_{i}.xyz"
        fp.write_text(_topology_structure(kind).to_xyz())
        paths.append(fp)
    return paths


def test_network_splits_rejects_fewer_than_two_minima(tmp_path):
    minima = _write_minima(tmp_path, ["chain"])
    with pytest.raises(typer.BadParameter):
        _call_network_splits(minima=minima, output=tmp_path / "out")


def test_network_splits_rejects_unsupported_mode(tmp_path):
    minima = _write_minima(tmp_path, ["chain", "pair_plus_single"])
    with pytest.raises(typer.BadParameter):
        _call_network_splits(minima=minima, mode="linear", output=tmp_path / "out")


def test_network_splits_rejects_nonpositive_max_pairs(tmp_path):
    minima = _write_minima(tmp_path, ["chain", "pair_plus_single"])
    with pytest.raises(typer.BadParameter):
        _call_network_splits(minima=minima, max_pairs=0, output=tmp_path / "out")


def test_network_splits_rejects_nonpositive_same_pair_split_limit(tmp_path):
    minima = _write_minima(tmp_path, ["chain", "pair_plus_single"])
    with pytest.raises(typer.BadParameter):
        _call_network_splits(minima=minima, same_pair_split_limit=0, output=tmp_path / "out")


def test_network_splits_runs_all_to_all_pairs_and_builds_network(tmp_path, monkeypatch):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    minima = _write_minima(tmp_path, ["chain", "pair_plus_single", "single_plus_pair"])
    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_network_splits(minima=minima, inputs=inputs_fp, output=output_dir)

    pairs_dir = output_dir / "pairs"
    # all-to-all over 3 minima: (0,1), (0,2), (1,2)
    for pair in ("pair_0_1", "pair_0_2", "pair_1_2"):
        assert (pairs_dir / pair / "tree" / "adj_matrix.txt").exists()

    network_path = output_dir / "network.json"
    assert network_path.exists()
    pot = Pot.read_from_disk(network_path)
    assert pot.number_of_nodes >= 3
    assert pot.graph.number_of_edges() >= 3


def test_network_splits_caps_at_max_pairs(tmp_path, monkeypatch, capsys):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    minima = _write_minima(tmp_path, ["chain", "pair_plus_single", "single_plus_pair"])
    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_network_splits(minima=minima, inputs=inputs_fp, max_pairs=1, output=output_dir)

    pairs_dir = output_dir / "pairs"
    completed = [p for p in pairs_dir.iterdir() if (p / "tree" / "adj_matrix.txt").exists()]
    assert len(completed) == 1
    out = capsys.readouterr().out
    assert "capping at --max-pairs=1" in out


def test_network_splits_is_resumable_skips_completed_pairs(tmp_path, monkeypatch, capsys):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    minima = _write_minima(tmp_path, ["chain", "pair_plus_single"])
    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_network_splits(minima=minima, inputs=inputs_fp, output=output_dir)

    # Re-running against the same --output should skip the already-completed pair.
    _call_network_splits(minima=minima, inputs=inputs_fp, output=output_dir)
    out = capsys.readouterr().out
    assert "Skipping pair (0, 1): already completed." in out
