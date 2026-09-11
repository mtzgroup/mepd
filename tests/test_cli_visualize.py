from __future__ import annotations

import json
import re

import numpy as np
import pytest
import typer
from qcdata import Structure

from mepd.cli import visualize as cli_visualize
from mepd.chain import Chain
from mepd.inputs import ChainInputs, NEBInputs
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


def _write_chain_xyz(fp, energies):
    _make_chain(energies).write_to_disk(fp)


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


def _call_visualize(**overrides):
    """Same rationale as the other `_call_*` helpers in this test suite --
    calling a Typer-decorated function directly does not resolve
    `typer.Option(...)` defaults, so every parameter must be explicit."""
    kwargs = dict(
        result_path=None,
        output=None,
        charge=0,
        multiplicity=1,
        no_open=True,
        show_atom_indices=False,
    )
    kwargs.update(overrides)
    return cli_visualize(**kwargs)


def _extract_nodes_payload(html: str) -> list[dict]:
    match = re.search(r"const nodes = (.*?);\nconst nodesByIndex", html, re.DOTALL)
    assert match, "nodes payload not found in rendered HTML"
    return json.loads(match.group(1))


def test_visualize_writes_html_with_interactive_scrubber(tmp_path):
    xyz_fp = tmp_path / "chain.xyz"
    _write_chain_xyz(xyz_fp, [-76.0, -75.9, -76.1])

    _call_visualize(result_path=xyz_fp)

    out_fp = tmp_path / "chain_visualize.html"
    assert out_fp.exists()
    html = out_fp.read_text()
    assert "3Dmol-min.js" in html
    assert 'id="frameSlider"' in html
    assert 'id="viewerContainer"' in html
    assert "function renderFrame(i)" in html
    assert "function showStructureXyz" in html

    nodes = _extract_nodes_payload(html)
    assert len(nodes) == 1
    frames = nodes[0]["trajectory"][0]["frames"]
    assert len(frames) == 3
    for frame in frames:
        assert "O" in frame["xyz"]  # real xyz text, not a pre-rendered doc


def test_visualize_labels_ts_guess_and_energies(tmp_path):
    xyz_fp = tmp_path / "chain.xyz"
    _write_chain_xyz(xyz_fp, [-76.0, -75.5, -76.1])  # node 1 is the energy maximum

    _call_visualize(result_path=xyz_fp)

    html = (tmp_path / "chain_visualize.html").read_text()
    nodes = _extract_nodes_payload(html)
    chain_payload = nodes[0]["trajectory"][0]
    assert chain_payload["ts_index"] == 1
    energies = [f["energy_kcal"] for f in chain_payload["frames"]]
    assert energies[0] == pytest.approx(0.0)
    assert energies[1] > energies[0]
    assert "TS guess" in html


def test_visualize_show_atom_indices_flag_is_forwarded(tmp_path, monkeypatch):
    xyz_fp = tmp_path / "chain.xyz"
    _write_chain_xyz(xyz_fp, [-76.0, -75.9])

    seen = {}
    import mepd.viz as viz_module
    real_render = viz_module.render_visualization_html

    def spy(obj, title="mepd visualization", show_atom_indices=False):
        seen["show_atom_indices"] = show_atom_indices
        return real_render(obj, title=title, show_atom_indices=show_atom_indices)

    monkeypatch.setattr(viz_module, "render_visualization_html", spy)

    _call_visualize(result_path=xyz_fp, show_atom_indices=True)

    assert seen["show_atom_indices"] is True


def test_visualize_handles_single_node_chain(tmp_path):
    """Regression coverage for the Chain.from_xyz single-node fix -- a
    one-minimum unique.xyz from hessian-sample must still visualize."""
    xyz_fp = tmp_path / "single.xyz"
    _write_chain_xyz(xyz_fp, [-76.0])

    _call_visualize(result_path=xyz_fp)

    html = (tmp_path / "single_visualize.html").read_text()
    assert "3Dmol-min.js" in html
    nodes = _extract_nodes_payload(html)
    assert len(nodes[0]["trajectory"][0]["frames"]) == 1


def test_visualize_respects_custom_output_path(tmp_path):
    xyz_fp = tmp_path / "chain.xyz"
    _write_chain_xyz(xyz_fp, [-76.0, -75.9])
    custom_out = tmp_path / "custom" / "report.html"

    _call_visualize(result_path=xyz_fp, output=custom_out)

    assert custom_out.exists()


def test_visualize_loads_a_split_tree_directory(tmp_path):
    root_neb = _make_neb([[-76.0, -75.6, -76.05], [-76.0, -75.4, -76.1]])
    child_neb = _make_neb([[-76.1, -75.9, -76.2]])
    tree = TreeNode(data=root_neb, children=[TreeNode(data=child_neb, children=[], index=1)], index=0)
    tree_dir = tmp_path / "tree"
    tree.write_to_disk(tree_dir)

    _call_visualize(result_path=tree_dir)

    html = (tmp_path / "tree_visualize.html").read_text()
    assert 'id="tree-node-0"' in html
    assert 'id="tree-node-1"' in html
    nodes = _extract_nodes_payload(html)
    assert len(nodes) == 2
    # root NEB kept 2 optimization steps -- the step slider has something to scrub
    root_payload = next(n for n in nodes if n["index"] == 0)
    assert len(root_payload["trajectory"]) == 2


def test_visualize_loads_a_bare_neb_history_directory(tmp_path):
    neb = _make_neb([[-76.0, -75.6, -76.05], [-76.0, -75.4, -76.1], [-76.0, -75.3, -76.15]])
    out_fp = tmp_path / "run.xyz"
    neb.write_to_disk(out_fp, write_history=True)
    history_dir = tmp_path / "run_history"
    assert history_dir.exists()

    _call_visualize(result_path=history_dir)

    html = (tmp_path / "run_history_visualize.html").read_text()
    nodes = _extract_nodes_payload(html)
    assert len(nodes) == 1
    assert len(nodes[0]["trajectory"]) == 3
    assert 'id="treeContainer"><' not in html or "svg" not in html.split('id="treeContainer">')[1][:20]


def test_visualize_loads_network_json(tmp_path):
    pot = Pot(root=Molecule())
    pot.graph.add_node(0, td=_water_node(0.0, -76.0))
    pot.graph.add_node(1, td=_water_node(0.2, -75.9))
    pot.graph.add_edge(0, 1, list_of_nebs=[_make_chain([-76.0, -75.6, -75.9])])
    network_fp = tmp_path / "network.json"
    pot.write_to_disk(network_fp)

    _call_visualize(result_path=network_fp)

    html = (tmp_path / "network_visualize.html").read_text()
    assert 'id="tree-node-0"' in html
    nodes = _extract_nodes_payload(html)
    assert len(nodes) == 1
    assert nodes[0]["source"] == 0
    assert nodes[0]["target"] == 1


def test_visualize_rejects_unrecognized_directory(tmp_path):
    empty_dir = tmp_path / "not_a_result"
    empty_dir.mkdir()

    with pytest.raises(typer.Exit):
        _call_visualize(result_path=empty_dir)
