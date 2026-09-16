from __future__ import annotations

import numpy as np
import pytest
import typer
from qcdata import Structure

from mepd.cli import ts as cli_ts
from mepd.chain import Chain
from mepd.engines.gxtb import GXTBCalculator
from mepd.inputs import ChainInputs, NEBInputs, RunInputs
from mepd.molecule import Molecule
from mepd.neb import NEB
from mepd.nodes.node import StructureNode
from mepd.optimizers.vpo import VelocityProjectedOptimizer
from mepd.pot import Pot
from mepd.TreeNode import TreeNode


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


def _water_node(x_offset: float = 0.0, energy: float | None = None) -> StructureNode:
    node = StructureNode(structure=_water(x_offset))
    if energy is not None:
        node._cached_energy = energy
        node._cached_gradient = np.zeros((3, 3))
    return node


def _make_chain(energies) -> Chain:
    nodes = [_water_node(0.05 * i, e) for i, e in enumerate(energies)]
    return Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})


def _make_neb(trajectory_energies: list[list[float]]) -> NEB:
    trajectory = [_make_chain(es) for es in trajectory_energies]
    return NEB(
        initial_chain=trajectory[0],
        optimizer=VelocityProjectedOptimizer(),
        parameters=NEBInputs(),
        engine=None,
        optimized=trajectory[-1],
        chain_trajectory=trajectory,
    )


def _run_inputs_for_test() -> RunInputs:
    return RunInputs(
        engine_name="gxtb",
        path_min_method="NEB",
        gxtb_engine_kwds={"executable": "gxtb"},
        gi_inputs={"nimages": 4},
        path_min_inputs={"max_steps": 2, "v": False, "do_elem_step_checks": False},
    )


def _call_ts(**overrides):
    """Same rationale as the other `_call_*` helpers across this test suite
    -- calling a Typer-decorated function directly does not resolve
    `typer.Option(...)` defaults, so every parameter must be explicit."""
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


def test_ts_scans_a_split_tree_directory(tmp_path, monkeypatch):
    calls = []

    def fake_compute_transition_state(self, node, keywords=None):
        calls.append(node)
        return node

    monkeypatch.setattr(GXTBCalculator, "compute_transition_state", fake_compute_transition_state)

    neb = _make_neb([[-76.0, -75.6, -76.05], [-76.0, -75.4, -76.1]])
    tree = TreeNode(data=neb, children=[], index=0)
    tree_dir = tmp_path / "tree"
    tree.write_to_disk(tree_dir)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "ts_out"
    _call_ts(guess=tree_dir, inputs=inputs_fp, output=output_dir)

    assert (output_dir / "ts_leaf_0.xyz").exists()
    assert len(calls) == 1


def test_ts_only_optimizes_leaves_not_internal_split_nodes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state", lambda self, node, keywords=None: node
    )

    root_neb = _make_neb([[-76.0, -75.6, -76.05], [-76.0, -75.4, -76.1]])
    child_neb = _make_neb([[-76.1, -75.9, -76.2]])
    tree = TreeNode(data=root_neb, children=[TreeNode(data=child_neb, children=[], index=1)], index=0)
    tree_dir = tmp_path / "tree"
    tree.write_to_disk(tree_dir)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "ts_out"
    _call_ts(guess=tree_dir, inputs=inputs_fp, output=output_dir)

    # Only the leaf (index 1) represents a completed elementary step -- the
    # root (index 0) was split further and has no TS guess of its own.
    assert (output_dir / "ts_leaf_1.xyz").exists()
    assert not (output_dir / "ts_leaf_0.xyz").exists()


def test_ts_is_resumable_and_skips_already_optimized_guesses(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_compute_transition_state(self, node, keywords=None):
        calls.append(node)
        return node

    monkeypatch.setattr(GXTBCalculator, "compute_transition_state", fake_compute_transition_state)

    neb = _make_neb([[-76.0, -75.6, -76.05], [-76.0, -75.4, -76.1]])
    tree = TreeNode(data=neb, children=[], index=0)
    tree_dir = tmp_path / "tree"
    tree.write_to_disk(tree_dir)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "ts_out"
    _call_ts(guess=tree_dir, inputs=inputs_fp, output=output_dir)
    assert len(calls) == 1

    _call_ts(guess=tree_dir, inputs=inputs_fp, output=output_dir)
    assert len(calls) == 1, "already-optimized leaf must not be re-optimized"
    out = capsys.readouterr().out
    assert "Skipping ts_leaf_0: already optimized" in out


def test_ts_scans_a_mepd_conformers_output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state", lambda self, node, keywords=None: node
    )

    neb = _make_neb([[-76.0, -75.6, -76.05], [-76.0, -75.4, -76.1]])
    tree = TreeNode(data=neb, children=[], index=0)
    conformers_output = tmp_path / "conf_out"
    pair_tree_dir = conformers_output / "pairs" / "pair_0_1" / "tree"
    pair_tree_dir.parent.mkdir(parents=True)
    tree.write_to_disk(pair_tree_dir)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "ts_out"
    _call_ts(guess=conformers_output, inputs=inputs_fp, output=output_dir)

    assert (output_dir / "ts_pair_0_1_leaf_0.xyz").exists()


def test_ts_scans_a_network_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state", lambda self, node, keywords=None: node
    )

    pot = Pot(root=Molecule())
    pot.graph.add_node(0, td=_water_node(0.0, -76.0))
    pot.graph.add_node(1, td=_water_node(0.2, -75.9))
    pot.graph.add_edge(0, 1, list_of_nebs=[_make_chain([-76.0, -75.6, -75.9])])
    network_fp = tmp_path / "network.json"
    pot.write_to_disk(network_fp)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "ts_out"
    _call_ts(guess=network_fp, inputs=inputs_fp, output=output_dir)

    assert (output_dir / "ts_edge_0_1.xyz").exists()


def test_ts_scan_with_irc_writes_irc_per_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(
        GXTBCalculator, "compute_transition_state", lambda self, node, keywords=None: node
    )

    def fake_compute_irc_chain(self, ts_node, keywords=None):
        return Chain.model_validate({
            "nodes": [ts_node, ts_node.copy()],
            "parameters": ChainInputs(),
        })

    monkeypatch.setattr(GXTBCalculator, "compute_irc_chain", fake_compute_irc_chain, raising=False)

    neb = _make_neb([[-76.0, -75.6, -76.05], [-76.0, -75.4, -76.1]])
    tree = TreeNode(data=neb, children=[], index=0)
    tree_dir = tmp_path / "tree"
    tree.write_to_disk(tree_dir)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "ts_out"
    _call_ts(guess=tree_dir, inputs=inputs_fp, output=output_dir, irc=True)

    assert (output_dir / "ts_leaf_0.xyz").exists()
    assert (output_dir / "ts_leaf_0_irc.xyz").exists()


def test_ts_writes_an_energy_sidecar_for_the_optimized_ts_structure(tmp_path, monkeypatch):
    """`mepd visualize` reads a TS structure's energy from a `<label>.energies`
    sidecar next to `<label>.xyz` (see `Chain.from_xyz`) -- so the optimized
    TS node's cached energy must actually be written to disk, not dropped."""

    def fake_compute_transition_state(self, node, keywords=None):
        node._cached_energy = -76.05
        return node

    monkeypatch.setattr(GXTBCalculator, "compute_transition_state", fake_compute_transition_state)

    neb = _make_neb([[-76.0, -75.6, -76.05], [-76.0, -75.4, -76.1]])
    tree = TreeNode(data=neb, children=[], index=0)
    tree_dir = tmp_path / "tree"
    tree.write_to_disk(tree_dir)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    output_dir = tmp_path / "ts_out"
    _call_ts(guess=tree_dir, inputs=inputs_fp, output=output_dir)

    energies_fp = output_dir / "ts_leaf_0.energies"
    assert energies_fp.exists()
    assert np.loadtxt(energies_fp) == pytest.approx(-76.05)


def test_ts_rejects_directory_with_no_recognizable_shape(tmp_path):
    empty_dir = tmp_path / "not_a_result"
    empty_dir.mkdir()

    with pytest.raises(typer.BadParameter):
        _call_ts(guess=empty_dir)


def test_ts_reports_when_no_guesses_found(tmp_path, capsys):
    pot = Pot(root=Molecule())
    network_fp = tmp_path / "network.json"
    pot.write_to_disk(network_fp)

    inputs_fp = tmp_path / "inputs.toml"
    _run_inputs_for_test().save(inputs_fp)

    with pytest.raises(typer.Exit):
        _call_ts(guess=network_fp, inputs=inputs_fp, output=tmp_path / "ts_out")

    out = capsys.readouterr().out
    assert "No TS guesses found" in out
