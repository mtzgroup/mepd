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

    # x-axis is normalized cumulative path length (matches plot_chain/
    # plot_opt_history/generate_neb_plot elsewhere in this module), not
    # plain frame index -- monotonically increasing, first frame at 0,
    # last frame at 1.
    path_lengths = [f["path_length"] for f in frames]
    assert path_lengths[0] == pytest.approx(0.0)
    assert path_lengths[-1] == pytest.approx(1.0)
    assert path_lengths == sorted(path_lengths)
    assert "normalized path length" in html


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
    frames = nodes[0]["trajectory"][0]["frames"]
    assert len(frames) == 1
    # A single-frame chain must not divide-by-zero into a NaN path_length --
    # NaN is not valid JSON (Python's json module is lenient about it, but a
    # real browser's JSON.parse would reject the page outright).
    assert frames[0]["path_length"] == 0.0
    assert "NaN" not in html


def test_visualize_ts_output_shows_energies_relative_to_lowest(tmp_path):
    """A `mepd ts` output directory's TS structures should show their known
    energies, relative to the lowest-energy TS found -- so conformers can be
    compared at a glance, the same way `test_visualize_labels_ts_guess_and_
    energies` does for a chain's own frames."""
    ts_out = tmp_path / "ts_out"
    ts_out.mkdir()
    Chain.model_validate(
        {"nodes": [_water_node(0.0, -76.0)], "parameters": ChainInputs()}
    ).write_to_disk(ts_out / "ts_leaf_0.xyz")
    Chain.model_validate(
        {"nodes": [_water_node(0.2, -75.9)], "parameters": ChainInputs()}
    ).write_to_disk(ts_out / "ts_leaf_1.xyz")

    _call_visualize(result_path=ts_out)

    html = (tmp_path / "ts_out_visualize.html").read_text()
    nodes = _extract_nodes_payload(html)
    assert {n["group"] for n in nodes} == {"TS structures"}
    by_label = {n["label"]: n for n in nodes}

    e0 = by_label["ts_leaf_0"]["trajectory"][0]["frames"][0]["energy_kcal"]
    e1 = by_label["ts_leaf_1"]["trajectory"][0]["frames"][0]["energy_kcal"]
    assert e0 == pytest.approx(0.0)
    assert e1 == pytest.approx((-75.9 - (-76.0)) * 627.5)


def _write_species_xyz(path, atoms: list[tuple[str, float]]) -> None:
    """A minimal single-frame xyz with atoms spaced along x -- mirrors the
    fixture convention in test_irc_network.py: atoms 1.4 apart bond into one
    molecule (openbabel bond perception), atoms several Angstroms apart stay
    as separate, unbonded species."""
    path.write_text(
        "\n".join(
            [
                str(len(atoms)),
                "frame",
                *(f"{symbol} {x:.6f} 0.0 0.0" for symbol, x in atoms),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_irc_xyz(path, frames: list[list[tuple[str, float]]]) -> None:
    lines = []
    for index, atoms in enumerate(frames):
        lines.extend(
            [
                str(len(atoms)),
                f"frame {index}",
                *(f"{symbol} {x:.6f} 0.0 0.0" for symbol, x in atoms),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_visualize_groups_irc_paths_by_endpoint_match(tmp_path):
    """IRCs whose own start/end frames are connectivity-identical (a
    degenerate/failed IRC that relaxed back to the same species on both
    sides) must land in a different group than IRCs connecting two
    genuinely different species -- so a `mepd ts --irc` output with several
    TS conformers makes it obvious which IRCs found a real elementary step."""
    bonded_chain = [("C", 0.0), ("C", 1.4), ("C", 2.8)]
    spread_out = [("C", 0.0), ("C", 4.0), ("C", 8.0)]

    ts_out = tmp_path / "ts_out"
    ts_out.mkdir()
    _write_species_xyz(ts_out / "ts_leaf_0.xyz", bonded_chain)
    _write_species_xyz(ts_out / "ts_leaf_1.xyz", bonded_chain)
    _write_irc_xyz(ts_out / "ts_leaf_0_irc.xyz", [bonded_chain, bonded_chain])
    _write_irc_xyz(ts_out / "ts_leaf_1_irc.xyz", [bonded_chain, spread_out])

    _call_visualize(result_path=ts_out)

    html = (tmp_path / "ts_out_visualize.html").read_text()
    nodes = _extract_nodes_payload(html)
    groups = {n["label"]: n["group"] for n in nodes}
    assert groups["ts_leaf_0 IRC"] == "IRC paths (matching endpoints)"
    assert groups["ts_leaf_1 IRC"] == "IRC paths (different endpoints)"


def test_visualize_groups_ts_dir_by_channels_classification_when_present(tmp_path):
    """When `ts_out` is a `mepd channels` output's `ts/` directory (i.e. it
    has sibling `channels/`/`alternate-routes/` folders with `members.txt`
    from `_write_classified_group`), grouping must reflect that real,
    IRC-verified-against---start/--end classification instead of the
    generic per-IRC self-consistency check -- and a leaf that isn't in any
    classification folder (e.g. a failed/degenerate IRC) must say so
    explicitly rather than silently falling back to the generic label."""
    bonded_chain = [("C", 0.0), ("C", 1.4), ("C", 2.8)]
    spread_out = [("C", 0.0), ("C", 4.0), ("C", 8.0)]

    output = tmp_path / "channels_out"
    ts_out = output / "ts"
    ts_out.mkdir(parents=True)
    _write_species_xyz(ts_out / "ts_pair_0_1_leaf_0.xyz", bonded_chain)
    _write_irc_xyz(ts_out / "ts_pair_0_1_leaf_0_irc.xyz", [bonded_chain, spread_out])
    _write_species_xyz(ts_out / "ts_pair_2_3_leaf_0.xyz", bonded_chain)
    _write_irc_xyz(ts_out / "ts_pair_2_3_leaf_0_irc.xyz", [bonded_chain, spread_out])
    _write_species_xyz(ts_out / "ts_pair_4_5_leaf_0.xyz", bonded_chain)
    _write_irc_xyz(ts_out / "ts_pair_4_5_leaf_0_irc.xyz", [bonded_chain, bonded_chain])

    channel_dir = output / "channels" / "channel_0"
    channel_dir.mkdir(parents=True)
    (channel_dir / "members.txt").write_text("ts_pair_0_1_leaf_0\n")

    route_dir = output / "alternate-routes" / "route_0"
    route_dir.mkdir(parents=True)
    (route_dir / "members.txt").write_text("connects: A <-> B\nts_pair_2_3_leaf_0\n")

    _call_visualize(result_path=ts_out)

    html = (tmp_path / "channels_out" / "ts_visualize.html").read_text()
    nodes = _extract_nodes_payload(html)
    groups = {n["label"]: n["group"] for n in nodes}

    assert groups["ts_pair_0_1_leaf_0"] == "TS structures (Channel 0)"
    assert groups["ts_pair_0_1_leaf_0 IRC"] == "IRC paths (Channel 0)"
    assert groups["ts_pair_2_3_leaf_0"] == "TS structures (Alternate route 0)"
    assert groups["ts_pair_2_3_leaf_0 IRC"] == "IRC paths (Alternate route 0)"
    assert groups["ts_pair_4_5_leaf_0"] == "TS structures (unclassified)"
    assert groups["ts_pair_4_5_leaf_0 IRC"] == "IRC paths (unclassified -- matching endpoints)"


def _last_atom_x(node_entry) -> float:
    xyz_text = node_entry["trajectory"][0]["frames"][0]["xyz"]
    last_line = xyz_text.strip().splitlines()[-1]
    return float(last_line.split()[1])


def test_visualize_reorients_channel_irc_so_reactant_is_always_first(tmp_path):
    """Two different channels' IRCs can come out of GSM in opposite
    orientations (reactant-to-product vs product-to-reactant) -- since
    displayed energies are relative to chain[0], both must be shown
    reactant-first so their (chain[0]-relative) barrier heights are
    directly comparable, not one forward barrier and one reverse barrier."""
    bonded_chain = [("C", 0.0), ("C", 1.4), ("C", 2.8)]  # "reactant" molecule
    spread_out = [("C", 0.0), ("C", 4.0), ("C", 8.0)]  # "product" (unbonded)

    output = tmp_path / "channels_out"
    ts_out = output / "ts"
    ts_out.mkdir(parents=True)

    conformers_dir = output / "conformers"
    conformers_dir.mkdir(parents=True)
    _write_species_xyz(conformers_dir / "start.xyz", bonded_chain)

    # channel_0's IRC is already stored reactant-first.
    _write_species_xyz(ts_out / "ts_pair_0_1_leaf_0.xyz", bonded_chain)
    _write_irc_xyz(ts_out / "ts_pair_0_1_leaf_0_irc.xyz", [bonded_chain, spread_out])

    # channel_1's IRC is stored product-first (reversed) -- must be flipped.
    _write_species_xyz(ts_out / "ts_pair_2_3_leaf_0.xyz", bonded_chain)
    _write_irc_xyz(ts_out / "ts_pair_2_3_leaf_0_irc.xyz", [spread_out, bonded_chain])

    (output / "channels" / "channel_0").mkdir(parents=True)
    (output / "channels" / "channel_0" / "members.txt").write_text("ts_pair_0_1_leaf_0\n")
    (output / "channels" / "channel_1").mkdir(parents=True)
    (output / "channels" / "channel_1" / "members.txt").write_text("ts_pair_2_3_leaf_0\n")

    _call_visualize(result_path=ts_out)

    html = (tmp_path / "channels_out" / "ts_visualize.html").read_text()
    nodes = _extract_nodes_payload(html)
    by_label = {n["label"]: n for n in nodes}

    for label in ("ts_pair_0_1_leaf_0 IRC", "ts_pair_2_3_leaf_0 IRC"):
        assert _last_atom_x(by_label[label]) == pytest.approx(2.8, abs=0.05), (
            f"{label} should be shown reactant-first regardless of how its "
            "IRC file was originally oriented"
        )


def test_render_visualization_html_accepts_a_bare_list_of_nodes():
    """`viz.render_visualization_html` must also accept a plain list of Node
    objects directly (e.g. `run_hessian_sample(...).optimized_nodes`, or the
    old neb-dynamics `visualize_chain([...])` convention), not just a Chain
    -- treating it as the frames of one chain."""
    from mepd import viz

    nodes = [_water_node(0.05 * i, -76.0 + 0.01 * i) for i in range(3)]

    html = viz.render_visualization_html(nodes)

    parsed_nodes = _extract_nodes_payload(html)
    frames = parsed_nodes[0]["trajectory"][0]["frames"]
    assert len(frames) == 3


def test_render_visualization_html_rejects_empty_list():
    from mepd import viz

    with pytest.raises(ValueError):
        viz.render_visualization_html([])


def test_render_visualization_html_rejects_list_of_wrong_type():
    from mepd import viz

    with pytest.raises(TypeError):
        viz.render_visualization_html([1, 2, 3])


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
