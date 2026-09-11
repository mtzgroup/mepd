from __future__ import annotations

import numpy as np
import pytest
import typer
from qcdata import Structure

from mepd.cli import network_build as cli_network_build
from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.pot import Pot


def _call_network_build(**overrides):
    """Same rationale as the other `_call_*` helpers in this test suite --
    calling a Typer-decorated function directly does not resolve
    `typer.Option(...)` defaults, so every parameter must be explicit."""
    kwargs = dict(
        paths=None,
        pattern="*.xyz",
        recursive=False,
        charge=0,
        multiplicity=1,
        output=None,
    )
    kwargs.update(overrides)
    return cli_network_build(**kwargs)


def _topology_structure(kind: str, offset: float = 0.0) -> Structure:
    # "chain": three atoms bonded in a row. "pair_plus_single": the first bond
    # stays, the second is stretched far enough to read as broken -- a real
    # connectivity (topology) change, not just a conformer change.
    if kind == "chain":
        geometry = np.array(
            [[0.0, 0.0, 0.0], [1.20 + offset, 0.0, 0.0], [2.16 + (2 * offset), 0.0, 0.0]]
        )
    elif kind == "pair_plus_single":
        geometry = np.array(
            [[0.0, 0.0, 0.0], [1.20 + offset, 0.0, 0.0], [5.20 + offset, 0.0, 0.0]]
        )
    else:
        raise ValueError(kind)
    return Structure(geometry=geometry, symbols=["C", "O", "H"], charge=0, multiplicity=1)


def _write_single_leaf_tree(tree_dir):
    """Hand-build a one-leaf MSMEP tree directory (adj_matrix.txt + node_0.xyz
    with real cached energies/gradients), using the real Chain.write_to_disk
    writer so the on-disk format matches exactly what `mepd run --recursive`
    produces -- reactant/product have genuinely different connectivity so
    NetworkBuilder has real work to do (not a degenerate single-node case)."""
    reactant = StructureNode(structure=_topology_structure("chain"))
    reactant._cached_energy = 0.0
    reactant._cached_gradient = np.zeros_like(reactant.coords)
    ts_guess = StructureNode(structure=_topology_structure("chain", offset=0.15))
    ts_guess._cached_energy = 0.05
    ts_guess._cached_gradient = np.zeros_like(ts_guess.coords)
    product = StructureNode(structure=_topology_structure("pair_plus_single"))
    product._cached_energy = 0.01
    product._cached_gradient = np.zeros_like(product.coords)

    chain = Chain.model_validate(
        {"nodes": [reactant, ts_guess, product], "parameters": ChainInputs()}
    )
    tree_dir.mkdir(parents=True, exist_ok=True)
    (tree_dir / "adj_matrix.txt").write_text("1.0\n")
    chain.write_to_disk(tree_dir / "node_0.xyz")


def _write_irc_scan_dir(directory):
    xyz = directory / "edge_0_1.irc.xyz"
    frames = [
        [("C", 0.0), ("C", 1.20), ("H", 2.16)],
        [("C", 0.0), ("C", 1.20), ("H", 3.5)],
        [("C", 0.0), ("C", 1.20), ("H", 5.20)],
    ]
    lines = []
    for i, atoms in enumerate(frames):
        lines.extend(
            [str(len(atoms)), f"Frame {i}"]
            + [f"{symbol} {x:.4f} 0.0 0.0" for symbol, x in atoms]
        )
    xyz.write_text("\n".join(lines) + "\n")
    np.savetxt(xyz.with_suffix(".energies"), [-1.0, -0.9, -1.1])


def test_network_build_detects_a_tree_directory(tmp_path):
    tree_dir = tmp_path / "tree"
    _write_single_leaf_tree(tree_dir)

    output = tmp_path / "network.json"
    _call_network_build(paths=[tree_dir], output=output)

    assert output.exists()
    pot = Pot.read_from_disk(output)
    assert pot.number_of_nodes == 2
    assert pot.graph.number_of_edges() == 2  # forward + reverse


def test_network_build_combines_multiple_tree_directories(tmp_path):
    tree_a = tmp_path / "tree_a"
    tree_b = tmp_path / "tree_b"
    _write_single_leaf_tree(tree_a)
    _write_single_leaf_tree(tree_b)

    output = tmp_path / "network.json"
    _call_network_build(paths=[tree_a, tree_b], output=output)

    assert output.exists()
    pot = Pot.read_from_disk(output)
    assert pot.number_of_nodes == 2  # deduped -- same species in both trees


def test_network_build_reports_tree_failure_without_crashing(tmp_path):
    empty_dir = tmp_path / "empty_tree"
    empty_dir.mkdir()
    (empty_dir / "adj_matrix.txt").write_text("1.0\n")
    with pytest.raises(typer.Exit):
        _call_network_build(paths=[empty_dir], output=tmp_path / "out.json")


def test_network_build_detects_an_irc_scan_directory(tmp_path):
    _write_irc_scan_dir(tmp_path)

    output = tmp_path / "network.json"
    _call_network_build(paths=[tmp_path], output=output)

    assert output.exists()
    pot = Pot.read_from_disk(output)
    assert pot.number_of_nodes == 2


def test_network_build_reports_irc_failure_without_crashing(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(typer.Exit):
        _call_network_build(paths=[empty_dir], output=tmp_path / "out.json")


def test_network_build_rejects_mixing_tree_and_irc_directories(tmp_path):
    tree_dir = tmp_path / "tree"
    irc_dir = tmp_path / "irc"
    irc_dir.mkdir()
    _write_single_leaf_tree(tree_dir)
    _write_irc_scan_dir(irc_dir)

    with pytest.raises(typer.BadParameter):
        _call_network_build(paths=[tree_dir, irc_dir], output=tmp_path / "out.json")


def test_network_build_rejects_multiple_irc_directories(tmp_path):
    irc_a = tmp_path / "irc_a"
    irc_b = tmp_path / "irc_b"
    irc_a.mkdir()
    irc_b.mkdir()
    _write_irc_scan_dir(irc_a)
    _write_irc_scan_dir(irc_b)

    with pytest.raises(typer.BadParameter):
        _call_network_build(paths=[irc_a, irc_b], output=tmp_path / "out.json")
