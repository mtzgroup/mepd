from __future__ import annotations

import subprocess

import numpy as np
import pytest
import typer
from qcdata import Structure

from mepd.cli import _build_path_minimizer, _echo_run_inputs_summary, run as cli_run, ts as cli_ts
from mepd.chain import Chain
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode
from mepd.engines.gxtb import GXTBCalculator


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


def _run_inputs_for_test() -> RunInputs:
    return RunInputs(
        engine_name="gxtb",
        path_min_method="NEB",
        gxtb_engine_kwds={"executable": "gxtb"},
        gi_inputs={"nimages": 4},
        path_min_inputs={"max_steps": 2, "v": False, "do_elem_step_checks": False},
    )


def _call_run(**overrides):
    """Invoke the `run` CLI command as a plain function with every parameter
    given an explicit value.

    This matters: calling a Typer-decorated function directly (bypassing the
    Click/Typer runtime) does NOT resolve `typer.Option(...)` defaults to
    their wrapped values -- any parameter not passed explicitly keeps the raw
    `OptionInfo` object, which is truthy regardless of its wrapped default
    (`bool(typer.Option(False, ...))` is `True`). Omitting a boolean flag here
    would silently turn it "on". Always route calls through this helper.
    """
    kwargs = dict(
        start=None,
        end=None,
        inputs=None,
        charge=None,
        multiplicity=None,
        minimize_ends=False,
        recursive=False,
        parallel=False,
        parallel_workers=None,
        network_completion=False,
        network_completion_mode="linear",
        network_max_followups=25,
        validate_minima_with_hessian=True,
        hessian_minimum_frequency_cutoff=0.0,
        hessian_minima_rescue_displacement=0.1,
        same_pair_split_limit=5,
        use_tsopt=False,
        irc=False,
        output=None,
    )
    kwargs.update(overrides)
    return cli_run(**kwargs)


def _call_ts(**overrides):
    """Same rationale as `_call_run` -- see its docstring."""
    kwargs = dict(
        guess=None,
        inputs=None,
        charge=None,
        multiplicity=None,
        irc=False,
        output=None,
    )
    kwargs.update(overrides)
    return cli_ts(**kwargs)


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
    _call_run(start=start_fp, end=end_fp, inputs=inputs_fp, output=output_dir)

    assert (output_dir / "mep_output.xyz").exists()
    assert len(calls) > 0
    assert not any("--opt" in call for call in calls), "should not minimize ends by default"


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
    _call_run(
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
    _call_run(
        start=start_fp,
        end=end_fp,
        inputs=inputs_fp,
        minimize_ends=True,
        output=output_dir,
    )

    assert (output_dir / "mep_output.xyz").exists()
    assert any("--opt" in call for call in calls)
    out = capsys.readouterr().out
    assert "Minimizing input endpoints" in out


def test_cli_run_minimize_ends_hard_stops_on_nonconvergence(tmp_path, monkeypatch, capsys):
    _install_fake_gxtb_unconverged_opt(monkeypatch)

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(0.3).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    with pytest.raises(typer.Exit) as exc_info:
        _call_run(
            start=start_fp,
            end=end_fp,
            inputs=inputs_fp,
            minimize_ends=True,
            output=output_dir,
        )

    assert exc_info.value.exit_code == 1
    out = capsys.readouterr().out
    assert "did not converge" in out
    assert "Provide an already-minimized structure" in out
    assert "geometry_optimizer_kwds" in out
    assert not (output_dir / "mep_output.xyz").exists()


def test_cli_run_rejects_recursive_and_parallel_together(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_run(
            start=tmp_path / "start.xyz",
            end=tmp_path / "end.xyz",
            output=tmp_path / "out",
            recursive=True,
            parallel=True,
        )


def test_cli_run_rejects_invalid_network_completion_mode(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_run(
            start=tmp_path / "start.xyz",
            end=tmp_path / "end.xyz",
            output=tmp_path / "out",
            network_completion=True,
            network_completion_mode="bogus",
        )


def _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch):
    """A fake gxtb whose energy actually varies with geometry.

    `_install_fake_gxtb` above always returns the same fixed energy/gradient
    regardless of input coordinates, which is fine for tests that only check
    the compute plumbing fires -- but MSMEP's endpoint-identity check
    (`is_identical`, comparing energies within `node_ene_thre`) then sees zero
    energy difference between ANY two endpoints and legitimately concludes
    they're the same species, short-circuiting before any real splitting
    logic runs. This variant ties energy to geometry so endpoints that are
    actually far apart are correctly treated as distinct.
    """

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


def test_cli_run_recursive_writes_tree_and_summary(tmp_path, monkeypatch):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(6.0).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_run(
        start=start_fp,
        end=end_fp,
        inputs=inputs_fp,
        recursive=True,
        output=output_dir,
    )

    assert (output_dir / "tree").exists()
    assert (output_dir / "tree" / "adj_matrix.txt").exists()
    assert (output_dir / "mep_output.xyz").exists()


def test_cli_run_use_tsopt_writes_ts_per_leaf_when_recursive(tmp_path, monkeypatch):
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)
    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state",
        lambda self, node, keywords=None: node,
    )

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(6.0).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_run(
        start=start_fp, end=end_fp, inputs=inputs_fp, recursive=True,
        output=output_dir, use_tsopt=True,
    )

    assert (output_dir / "mep_output.xyz").exists()
    ts_leaf_files = list(output_dir.glob("ts_leaf_*.xyz"))
    assert len(ts_leaf_files) >= 1


def test_cli_run_rejects_irc_without_use_tsopt(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_run(
            start=tmp_path / "start.xyz",
            end=tmp_path / "end.xyz",
            output=tmp_path / "out",
            irc=True,
        )


def test_cli_run_rejects_nonpositive_same_pair_split_limit(tmp_path):
    with pytest.raises(typer.BadParameter):
        _call_run(
            start=tmp_path / "start.xyz",
            end=tmp_path / "end.xyz",
            output=tmp_path / "out",
            same_pair_split_limit=0,
        )


def test_cli_run_wires_same_pair_split_limit_into_path_min_inputs(tmp_path, monkeypatch):
    _install_fake_gxtb(monkeypatch)

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(6.0).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    seen = {}
    import mepd.cli as cli_module
    real_summary = cli_module._echo_run_inputs_summary

    def spy(run_inputs):
        seen["limit"] = run_inputs.path_min_inputs.recursive_same_pair_split_limit
        return real_summary(run_inputs)

    monkeypatch.setattr(cli_module, "_echo_run_inputs_summary", spy)

    output_dir = tmp_path / "out"
    _call_run(
        start=start_fp, end=end_fp, inputs=inputs_fp, output=output_dir,
        same_pair_split_limit=42,
    )

    assert seen["limit"] == 42


def test_cli_run_use_tsopt_writes_ts_after_single_neb(tmp_path, monkeypatch):
    """Regression coverage: mepd run used to have no way to automatically
    launch a TS optimization on its own result, unlike the upstream
    neb-dynamics `run --use-tsopt` this was ported from.

    Deliberately does not use capsys here: mepd/progress.py holds a
    module-level rich Console() constructed once at import time, and a
    capsys-captured stdout immediately followed by a rich-progress-heavy
    recursive run elsewhere in this file corrupts it ("I/O operation on
    closed file") -- pre-existing pytest/capsys/rich interaction, unrelated
    to this feature. File existence is sufficient to verify the behavior.
    """
    _install_fake_gxtb(monkeypatch)
    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state",
        lambda self, node, keywords=None: node,
    )

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(6.0).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_run(
        start=start_fp, end=end_fp, inputs=inputs_fp, output=output_dir, use_tsopt=True,
    )

    assert (output_dir / "mep_output.xyz").exists()
    assert (output_dir / "ts.xyz").exists()


def test_cli_run_use_tsopt_with_irc_after_single_neb(tmp_path, monkeypatch):
    _install_fake_gxtb(monkeypatch)
    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state",
        lambda self, node, keywords=None: node,
    )
    monkeypatch.setattr(
        GXTBCalculator, "compute_irc_chain",
        lambda self, ts_node, keywords=None: Chain.model_validate({
            "nodes": [ts_node, ts_node.copy()],
            "parameters": _run_inputs_for_test().chain_inputs,
        }),
        raising=False,
    )

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(6.0).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_run(
        start=start_fp, end=end_fp, inputs=inputs_fp, output=output_dir,
        use_tsopt=True, irc=True,
    )

    assert (output_dir / "ts.xyz").exists()
    assert (output_dir / "irc.xyz").exists()


def test_cli_run_network_completion_still_writes_mep_output(tmp_path, monkeypatch):
    """Regression test: --network-completion used to `return` right after
    _run_network_completion, before ever reaching the code that assembles and
    writes mep_output.xyz -- so the final path was silently never written."""
    _install_fake_gxtb_with_coordinate_dependent_energy(monkeypatch)

    start_fp = tmp_path / "start.xyz"
    end_fp = tmp_path / "end.xyz"
    start_fp.write_text(_water().to_xyz())
    end_fp.write_text(_water(6.0).to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "out"
    _call_run(
        start=start_fp,
        end=end_fp,
        inputs=inputs_fp,
        recursive=True,
        network_completion=True,
        output=output_dir,
    )

    assert (output_dir / "tree" / "adj_matrix.txt").exists()
    assert (output_dir / "network.json").exists()
    assert (output_dir / "mep_output.xyz").exists()


def test_ts_command_writes_optimized_structure(tmp_path, monkeypatch, capsys):
    guess_fp = tmp_path / "guess.xyz"
    guess_fp.write_text(_water().to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    def fake_compute_transition_state(self, node, keywords=None):
        return node

    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state", fake_compute_transition_state
    )

    output_dir = tmp_path / "ts_out"
    _call_ts(guess=guess_fp, inputs=inputs_fp, output=output_dir)

    ts_path = output_dir / "ts.xyz"
    assert ts_path.exists()
    out = capsys.readouterr().out
    assert "Wrote optimized TS structure" in out


def test_ts_command_reports_failure_without_crashing(tmp_path, monkeypatch, capsys):
    guess_fp = tmp_path / "guess.xyz"
    guess_fp.write_text(_water().to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    def failing_compute_transition_state(self, node, keywords=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state", failing_compute_transition_state
    )

    output_dir = tmp_path / "ts_out_fail"
    with pytest.raises(typer.Exit):
        _call_ts(guess=guess_fp, inputs=inputs_fp, output=output_dir)

    out = capsys.readouterr().out
    assert "Transition-state optimization failed" in out
    assert not (output_dir / "ts.xyz").exists()


def test_ts_command_with_irc(tmp_path, monkeypatch, capsys):
    guess_fp = tmp_path / "guess.xyz"
    guess_fp.write_text(_water().to_xyz())

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    def fake_compute_transition_state(self, node, keywords=None):
        return node

    def fake_compute_irc_chain(self, ts_node, keywords=None):
        return Chain.model_validate({
            "nodes": [ts_node, ts_node.copy()],
            "parameters": _run_inputs_for_test().chain_inputs,
        })

    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state", fake_compute_transition_state
    )
    monkeypatch.setattr(
        GXTBCalculator, "compute_irc_chain", fake_compute_irc_chain, raising=False
    )

    output_dir = tmp_path / "ts_irc_out"
    _call_ts(guess=guess_fp, inputs=inputs_fp, irc=True, output=output_dir)

    assert (output_dir / "ts.xyz").exists()
    assert (output_dir / "irc.xyz").exists()
    out = capsys.readouterr().out
    assert "Wrote IRC path" in out


def test_echo_run_inputs_summary_shows_full_resolved_engine_and_optimizer(capsys):
    """The printed summary must show every active engine/optimizer field --
    including ones left at their default and never mentioned in a TOML --
    not just the raw (possibly-partial) *_kwds dict that happened to be set."""
    run_inputs = RunInputs(
        engine_name="gxtb",
        gxtb_engine_kwds={"executable": "gxtb"},
        optimizer_kwds={"name": "cg"},
    )

    _echo_run_inputs_summary(run_inputs)

    out = capsys.readouterr().out
    # Optimizer fields never set in optimizer_kwds (only "name" was given):
    for field in ("adaptive_dt", "corr_increase_thre", "negative_steps_thre", "positive_steps_thre"):
        assert field in out
    # Engine fields never set in gxtb_engine_kwds (only "executable" was given):
    for field in ("extra_args", "keep_workdirs", "add_gxtb_flag", "n_threads"):
        assert field in out
    # The raw, partial *_kwds dicts should no longer be shown as their own section.
    assert "optimizer_kwds" not in out
    assert "gxtb_engine_kwds" not in out
    assert "optimizer (ConjugateGradient)" in out
    assert "engine (GXTBCalculator)" in out
