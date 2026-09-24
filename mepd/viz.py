"""Optional visualization helpers for mepd.

Everything in this module is deliberately kept out of mepd's core
(chain.py, chainhelpers.py, neb.py, pathminimizers/pathminimizer.py) so
that the core package stays lightweight and installable without
matplotlib, IPython, or qcdata's structure-viewer extras. Install the
`viz` extra (`pip install mepd[viz]`) to use anything here.

Every function takes the object to visualize as its first argument,
mirroring the methods these were extracted from -- e.g. `viz.plot_chain(chain)`
replaces the old `chain.plot_chain()`; `viz.plot_opt_history(minimizer.chain_trajectory)`
replaces both the old `NEB`/`PathMinimizer` `.plot_opt_history()` methods
(they were identical, so this consolidates the duplicate into one function).
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from mepd.chain import Chain
from mepd.nodes.node import Node


def plot_chain(chain: Chain, norm_path: bool = True, dist_func: str = "mw_rmsd") -> None:
    import matplotlib.pyplot as plt

    s = 8
    fs = 18
    avail_dists = ["mw_rmsd", "geodesic"]

    if dist_func == "mw_rmsd":
        path_len = chain.path_length
    elif dist_func == "geodesic":
        path_len = chain.geodesic_path_length
    else:
        raise ValueError(f"Invalid dist_func: {dist_func}. Use one of {avail_dists}")

    if norm_path:
        path_len = path_len / sum(path_len)

    plt.subplots(figsize=(1.16 * s, s))
    plt.plot(path_len, (chain.energies - chain.energies[0]) * 627.5, "o--", label="neb")
    plt.ylabel("Energy (kcal/mol)", fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.show()


def animate_chain_trajectory(
    chain_traj, min_y=-100, max_y=100, max_x=1.1, min_x=-0.1, norm_path_len=True
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    figsize = 5
    fig, ax = plt.subplots(figsize=(1.618 * figsize, figsize))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    (line,) = ax.plot([], [], "o--", lw=3)

    def animate(chain):
        x = chain.integrated_path_length if norm_path_len else chain.path_length
        y = chain.energies_kcalmol
        line.set_data(x, y)
        line.set_color("skyblue")

    ani = FuncAnimation(fig, animate, frames=chain_traj)
    return HTML(ani.to_jshtml())


def generate_neb_plot(
    chain: List[Node],
    ind_node,
    figsize=(6.4, 4.8),
    grid=True,
    markersize=20,
    title="Energies across chain",
) -> str:
    """Renders a chain's energy profile to a base64-encoded PNG string."""
    import matplotlib.pyplot as plt
    import mepd.chainhelpers as ch

    try:
        energies = ch._energies_kcalmol(chain)
    except Exception:
        print("Cannot plot energies.")
        return ""

    fig, ax1 = plt.subplots(figsize=figsize)
    color = "tab:blue"
    ax1.set_xlabel("Path length")
    ax1.set_ylabel("Relative energies (kcal/mol)", color=color)
    markercolors = ["green"] * len(chain)
    markersizes = [markersize] * len(chain)
    markercolors[ind_node] = "gold"
    markersizes[ind_node] = markersize + 50

    path_len = ch.path_length(chain=chain)
    ax1.plot(path_len, energies, color="green")
    ax1.scatter(
        path_len, energies, label="Energy", marker="o", color=markercolors, s=markersizes
    )

    ax1.tick_params(axis="y", labelcolor=color)
    plt.title(title, pad=20)
    ax1.legend(loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return image_base64


_3DMOL_CDN_SCRIPT = '<script src="https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"></script>'


def _chain_path_lengths(chain: Chain) -> list[float]:
    """Normalized cumulative path length per frame (0 for the first frame, 1
    for the last) -- the same `integrated_path_length` metric the rest of
    this module plots against (plot_chain, plot_opt_history,
    generate_neb_plot), so the interactive plot's x-axis matches theirs
    instead of assuming images are evenly spaced along the path."""
    n = len(chain)
    if n <= 1:
        return [0.0] * n
    try:
        path_lengths = [float(x) for x in chain.integrated_path_length]
        if not all(np.isfinite(x) for x in path_lengths):
            raise ValueError("non-finite path length")
        return path_lengths
    except Exception:
        # Degenerate geometry (e.g. a distance function that chokes on some
        # frame pair, or -- for two very dissimilar frames, such as an IRC's
        # endpoints when they're genuinely different species -- an alignment
        # that divides by a near-zero norm and comes back NaN) -- fall back
        # to even spacing rather than failing the whole visualization, or
        # emitting a NaN that breaks the page's embedded JSON, over an
        # x-axis metric.
        return [i / (n - 1) for i in range(n)]


def _chain_payload(chain: Chain) -> dict:
    """One chain's worth of interactive-viewer data: every frame's xyz text
    (already Angstrom-scaled, straight from qcdata -- no extra conversion
    needed), its normalized position along the path, plus, when available,
    each frame's relative energy and which frame is the apparent TS (energy
    maximum)."""
    n = len(chain)
    has_energies = chain._energies_already_computed and n > 1
    energies_kcal = list(chain.energies_kcalmol) if has_energies else None
    ts_index = int(np.argmax(chain.energies)) if has_energies else None
    path_lengths = _chain_path_lengths(chain)
    frames = [
        {
            "xyz": node.structure.to_xyz(),
            "energy_kcal": energies_kcal[i] if energies_kcal else None,
            "path_length": path_lengths[i],
        }
        for i, node in enumerate(chain.nodes)
    ]
    return {"frames": frames, "ts_index": ts_index}


def _trajectory_payload(minimizer) -> list[dict]:
    """One NEB/PathMinimizer's optimization trajectory: one chain payload per
    optimization step, in order. Falls back to whatever single chain is
    available (`optimized` or `initial_chain`) if the full step-by-step
    trajectory was not kept."""
    trajectory = list(getattr(minimizer, "chain_trajectory", None) or [])
    if not trajectory:
        fallback = getattr(minimizer, "optimized", None) or getattr(minimizer, "initial_chain", None)
        trajectory = [fallback] if fallback is not None else []
    return [_chain_payload(c) for c in trajectory if c is not None and len(c) > 0]


def _tree_nodes_payload(tree) -> list[dict]:
    """Flatten a TreeNode split-tree into a list of selectable nodes, each
    carrying its own optimization trajectory. Nodes with no recoverable NEB
    data (failed splits) are skipped, but their children are still walked
    (re-parented to the nearest visualizable ancestor) so the tree stays
    connected."""
    nodes: list[dict] = []

    def walk(node, depth: int, parent_index) -> None:
        next_parent = parent_index
        data = getattr(node, "data", None)
        if data is not None:
            trajectory = _trajectory_payload(data)
            if trajectory:
                label = f"Node {node.index}" + (" (leaf)" if node.is_leaf else "")
                nodes.append({
                    "index": int(node.index),
                    "depth": depth,
                    "parent": parent_index,
                    "label": label,
                    "trajectory": trajectory,
                })
                next_parent = int(node.index)
        for child in node.children:
            walk(child, depth + 1, next_parent)

    walk(tree, 0, None)
    return nodes


def _network_edges_payload(pot) -> list[dict]:
    """Flatten a reaction-network `Pot` graph into a list of selectable
    edges, each carrying every candidate chain found for that edge
    (`list_of_nebs`) as its "trajectory" -- reusing the same step-slider
    concept as an optimization trajectory, since both are just "which chain
    among several for this edge/step do I want to look at right now"."""
    nodes: list[dict] = []
    for i, j, edge_data in pot.graph.edges(data=True):
        chains = edge_data.get("list_of_nebs") or []
        trajectory = [_chain_payload(c) for c in chains if c is not None and len(c) > 0]
        if not trajectory:
            continue
        nodes.append({
            "index": len(nodes),
            "source": int(i),
            "target": int(j),
            "label": f"Edge {i}→{j}",
            "trajectory": trajectory,
        })
    return nodes


@dataclass
class ChannelsResult:
    """The on-disk output of `mepd channels` (conformer pools + one
    completed MSMEP tree per reactant/product conformer pair, optionally an
    aggregated network.json), bundled for `render_visualization_html` to
    show as one clickable page: reactant conformers, product conformers,
    and completed MEP outputs, each its own group."""

    reactant_conformers: list = field(default_factory=list)
    product_conformers: list = field(default_factory=list)
    pairs: list = field(default_factory=list)  # list[tuple[str, Chain]]
    network: Optional[object] = None  # Pot | None


def _single_structure_payload(node, energy_baseline: Optional[float] = None) -> dict:
    """One bare structure (e.g. a conformer or TS structure, not an
    optimization frame) as a single-frame chain payload -- drops the "TS
    guess" framing, which doesn't mean anything for a single structure.

    `_chain_payload`'s own energy handling never fires here (it requires
    more than one frame, since a lone frame has no profile to plot), so this
    computes the displayed energy itself: relative to `energy_baseline`
    (e.g. the lowest-energy structure among several being compared) when
    given, otherwise relative to the node's own energy (always 0.0, just
    confirming an energy is known) if none is given."""
    from mepd.inputs import ChainInputs

    chain = Chain.model_validate({"nodes": [node], "parameters": ChainInputs()})
    payload = _chain_payload(chain)
    payload["ts_index"] = None
    energy = node._cached_energy
    if energy is not None:
        baseline = energy if energy_baseline is None else energy_baseline
        payload["frames"][0]["energy_kcal"] = (energy - baseline) * 627.5
    return payload


def _channels_nodes_payload(result: ChannelsResult) -> list[dict]:
    nodes: list[dict] = []

    def add(group: str, label: str, trajectory: list[dict]) -> None:
        if not trajectory:
            return
        nodes.append({
            "index": len(nodes),
            "depth": 0,
            "parent": None,
            "group": group,
            "label": label,
            "trajectory": trajectory,
        })

    for i, node in enumerate(result.reactant_conformers):
        add("Reactant conformers", f"Reactant conformer {i}", [_single_structure_payload(node)])
    for i, node in enumerate(result.product_conformers):
        add("Product conformers", f"Product conformer {i}", [_single_structure_payload(node)])
    for label, chain in result.pairs:
        if chain is not None and len(chain) > 0:
            add("Completed MEP outputs", label, [_chain_payload(chain)])
    if result.network is not None:
        for edge in _network_edges_payload(result.network):
            add("Aggregated network edges", edge["label"], edge["trajectory"])

    return nodes


@dataclass
class TsOutputResult:
    """The on-disk output of `mepd ts` (or `mepd run --use-tsopt`): one or
    more optimized transition-state structures (<label>.xyz), each optionally
    paired with an IRC path (<label>_irc.xyz, or irc.xyz for the bare "ts"
    label), bundled for `render_visualization_html` to show as one clickable
    page: TS structures and IRC paths, each its own group.

    `group_labels` (label -> "Channel <k>"/"Alternate channel <k> step
    <n>"/"Off-target exit channel <k>") is populated when this `ts/`
    directory sits inside a `mepd channels` output alongside its
    `channels/`/`alternate-channels/`/`offtarget-exit-channels/`
    classification folders -- when present, it overrides the generic
    `_irc_endpoint_match_label` self-consistency grouping below with the
    real classification `mepd channels` already computed (IRC-verified
    against the actual --start/--end pair, not just "are this IRC's own
    two ends different from each other")."""

    structures: list = field(default_factory=list)  # list[tuple[str, Node]]
    irc_paths: list = field(default_factory=list)  # list[tuple[str, Chain]]
    group_labels: dict = field(default_factory=dict)  # label -> "Channel <k>" | "Alternate channel <k> step <n>" | "Off-target exit channel <k>"


def _irc_endpoint_match_label(chain: Chain) -> str:
    """Whether this IRC's own two endpoints (its first and last frame) are
    the same molecular species, by connectivity (bond-isomorphism) -- the
    same check `mepd.irc_network.build_irc_network` uses to skip degenerate
    "connectivity-identical endpoints" IRCs when building a reaction
    network. A match usually means a failed/degenerate IRC (relaxed back to
    the same well on both sides, e.g. a bond-rotation TS) rather than a
    genuine two-minima elementary step, so this is worth surfacing as its
    own group rather than mixing it in with real steps."""
    from mepd.irc_network import _same_connectivity

    if len(chain) < 2:
        return "endpoints not comparable"
    start, end = chain.nodes[0], chain.nodes[-1]
    if getattr(start, "graph", None) is None or getattr(end, "graph", None) is None:
        return "endpoints not comparable"
    return "matching endpoints" if _same_connectivity(start, end) else "different endpoints"


def _ts_output_nodes_payload(result: TsOutputResult) -> list[dict]:
    nodes: list[dict] = []

    def add(group: str, label: str, trajectory: list[dict]) -> None:
        if not trajectory:
            return
        nodes.append({
            "index": len(nodes),
            "depth": 0,
            "parent": None,
            "group": group,
            "label": label,
            "trajectory": trajectory,
        })

    known_energies = [
        node._cached_energy for _, node in result.structures if node._cached_energy is not None
    ]
    baseline = min(known_energies) if known_energies else None
    for label, node in result.structures:
        classification = result.group_labels.get(label)
        if classification is None and result.group_labels:
            classification = "unclassified"
        group = f"TS structures ({classification})" if classification else "TS structures"
        add(group, label, [_single_structure_payload(node, baseline)])

    for label, chain in result.irc_paths:
        if chain is not None and len(chain) > 0:
            classification = result.group_labels.get(label)
            if classification is None and result.group_labels:
                # This `ts/` directory has real `mepd channels` classification
                # data, but this particular label isn't in it -- it was
                # dropped (failed TS-opt/IRC, or an IRC that never reached a
                # genuine second minimum). Keep the self-consistency label as
                # a diagnostic instead of hiding why.
                classification = f"unclassified -- {_irc_endpoint_match_label(chain)}"
            elif classification is None:
                classification = _irc_endpoint_match_label(chain)
            group = f"IRC paths ({classification})"
            add(group, f"{label} IRC", [_chain_payload(chain)])

    return nodes


def _grouped_nodes_html(nodes: list[dict]) -> str:
    """Grouped clickable list (not a tree/graph diagram): one section per
    group -- e.g. reactant conformers, product conformers, completed MEP
    outputs, aggregated network edges, TS structures, IRC paths -- each a
    row of buttons picking that node as the current selection via the
    shared `selectNode(index)`."""
    groups: dict[str, list[dict]] = {}
    for node in nodes:
        groups.setdefault(node["group"], []).append(node)

    sections = []
    for group_name, group_nodes in groups.items():
        buttons = "".join(
            f'<button class="node-button" id="list-node-{n["index"]}" '
            f'onclick="selectNode({n["index"]})" title="{html.escape(n["label"])}">'
            f'{html.escape(n["label"])}</button>'
            for n in group_nodes
        )
        sections.append(
            f'<div class="node-group"><h4>{html.escape(group_name)} ({len(group_nodes)})</h4>'
            f'<div class="node-group-buttons">{buttons}</div></div>'
        )
    return "".join(sections)


def render_visualization_html(
    obj,
    title: str = "mepd visualization",
    show_atom_indices: bool = False,
) -> str:
    """Render a standalone, self-contained HTML page for interactively
    exploring a `Chain`, a `NEB`/`PathMinimizer` optimization run, a full
    `TreeNode` MSMEP split-tree, a reaction-network `Pot` graph, or a bare
    list of `Node` objects (e.g. `run_hessian_sample(...).optimized_nodes`,
    or the old neb-dynamics `visualize_chain([...])` convention -- treated
    as the frames of one chain) -- whichever was given.

    One consistent widget handles all four, with controls that appear only
    when there is something to navigate:
      - Tree or network diagram (only for a TreeNode/Pot with more than one
        visualizable node/edge): click a tree node or a network edge to load
        its optimization run.
      - Trajectory-step slider (only if the selected run kept more than one
        optimization step, or the selected network edge has more than one
        candidate chain): scrub through them.
      - Frame slider (always): steps through the beads of whichever chain is
        currently selected, updating the 3D structure viewer and
        highlighting the matching point ("bead") on the energy-profile plot
        in lockstep. Opens on the TS guess (energy maximum) by default.

    Needs only a browser (3Dmol.js loaded from a CDN) -- no matplotlib,
    IPython, or qcdata's view extra, unlike the rest of this module.
    """
    import json

    from mepd.inputs import ChainInputs
    from mepd.pathminimizers.pathminimizer import PathMinimizer
    from mepd.pot import Pot
    from mepd.TreeNode import TreeNode

    if isinstance(obj, (list, tuple)):
        # A bare list of structures/nodes (e.g. `run_hessian_sample(...).
        # optimized_nodes`, or the old neb-dynamics `visualize_chain([...])`
        # convention) -- treat it as the frames of one chain.
        if not obj:
            raise ValueError("Cannot visualize an empty list of structures.")
        try:
            obj = Chain.model_validate({"nodes": list(obj), "parameters": ChainInputs()})
        except Exception as exc:
            raise TypeError(
                f"Cannot visualize a list of {type(obj[0]).__name__}; "
                "expected a list of Node objects (e.g. StructureNode)."
            ) from exc

    diagram_kind = None  # "tree", "network", or None (single chain/run, no diagram)
    diagram_obj = None

    if isinstance(obj, TreeNode):
        nodes = _tree_nodes_payload(obj)
        if not nodes:
            raise ValueError("Tree has no recoverable optimization data to visualize.")
        diagram_kind, diagram_obj = "tree", nodes
    elif isinstance(obj, Pot):
        nodes = _network_edges_payload(obj)
        if not nodes:
            raise ValueError("Network has no edges with recoverable chains to visualize.")
        diagram_kind, diagram_obj = "network", (obj, nodes)
    elif isinstance(obj, ChannelsResult):
        nodes = _channels_nodes_payload(obj)
        if not nodes:
            raise ValueError(
                "Nothing recoverable to visualize: no conformers and no "
                "completed MEP outputs found."
            )
        diagram_kind, diagram_obj = "grouped", nodes
    elif isinstance(obj, TsOutputResult):
        nodes = _ts_output_nodes_payload(obj)
        if not nodes:
            raise ValueError(
                "Nothing recoverable to visualize: no TS structures or IRC "
                "paths found."
            )
        diagram_kind, diagram_obj = "grouped", nodes
    elif isinstance(obj, PathMinimizer):
        trajectory = _trajectory_payload(obj)
        if not trajectory:
            raise ValueError("This object has no chain trajectory to visualize.")
        nodes = [{"index": 0, "depth": 0, "parent": None, "label": title, "trajectory": trajectory}]
    elif isinstance(obj, Chain):
        nodes = [{"index": 0, "depth": 0, "parent": None, "label": title, "trajectory": [_chain_payload(obj)]}]
    else:
        raise TypeError(
            f"Cannot visualize object of type {type(obj).__name__}; "
            "expected a Chain, a NEB/PathMinimizer, a TreeNode, a Pot, a "
            "ChannelsResult, a TsOutputResult, or a list of Node objects."
        )

    nodes_json = json.dumps(nodes)

    tree_html = ""
    if diagram_kind == "tree" and len(diagram_obj) > 1:
        tree_html = _tree_svg(diagram_obj)
    elif diagram_kind == "network":
        pot, edge_nodes = diagram_obj
        tree_html = _network_svg(pot, edge_nodes)
    elif diagram_kind == "grouped":
        tree_html = _grouped_nodes_html(diagram_obj)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
{_3DMOL_CDN_SCRIPT}
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
  #viewerContainer {{ width: 100%; height: 480px; position: relative; }}
  input[type=range] {{ width: min(720px, 90vw); }}
  .node-group {{ margin-bottom: 0.75rem; }}
  .node-group h4 {{ margin: 0 0 0.3rem 0; }}
  .node-group-buttons {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
  .node-button {{ padding: 0.3rem 0.6rem; border: 1px solid #ccc; border-radius: 0.3rem;
                  background: white; cursor: pointer; font-size: 0.85rem; }}
  .node-button.selected {{ background: #dbeafe; border-color: #2563eb; }}
</style>
</head>
<body>
<h2>{title}</h2>
<div id="treeContainer">{tree_html}</div>
<div id="stepControls" style="display:none;">
  <label for="stepSlider">Optimization step: <span id="stepLabel">0</span> / <span id="stepMax">0</span></label><br/>
  <input id="stepSlider" type="range" min="0" max="0" value="0" step="1" />
</div>
<label for="frameSlider">Frame: <span id="frameLabel">0</span> <span id="frameEnergy"></span></label><br/>
<input id="frameSlider" type="range" min="0" max="0" value="0" step="1" />
<div id="plotContainer"></div>
<div id="viewerContainer"></div>
<script>
const nodes = {nodes_json};
const nodesByIndex = Object.fromEntries(nodes.map((n) => [n.index, n]));
const showAtomIndices = {"true" if show_atom_indices else "false"};
let currentNode = nodes[0];
let currentStep = 0;
let currentFrame = 0;
let glviewer = null;

function showStructureXyz(xyzText) {{
  if (!glviewer) {{
    glviewer = $3Dmol.createViewer(document.getElementById("viewerContainer"), {{backgroundColor: "white"}});
  }}
  glviewer.clear();
  glviewer.removeAllLabels();
  const model = glviewer.addModel(xyzText, "xyz");
  glviewer.setStyle({{}}, {{stick: {{}}, sphere: {{scale: 0.3}}}});
  if (showAtomIndices) {{
    model.selectedAtoms({{}}).forEach((atom, idx) => {{
      glviewer.addLabel(String(idx), {{
        position: {{x: atom.x, y: atom.y, z: atom.z}},
        fontSize: 10, showBackground: false, fontColor: "black",
      }});
    }});
  }}
  glviewer.zoomTo();
  glviewer.render();
}}

function renderEnergyPlot(frames, width, height) {{
  width = width || 640;
  height = height || 220;
  const container = document.getElementById("plotContainer");
  const energies = frames.map((f) => f.energy_kcal);
  if (energies.some((e) => e === null || e === undefined)) {{
    container.innerHTML = "";
    return;
  }}
  const pathLengths = frames.map((f) => f.path_length);
  const margin = 36;
  const plotW = width - 2 * margin;
  const plotH = height - 2 * margin;
  let yMin = Math.min(...energies);
  let yMax = Math.max(...energies);
  if (yMin === yMax) yMax = yMin + 1.0;
  const sx = (t) => margin + t * plotW;
  const sy = (e) => margin + (1 - (e - yMin) / (yMax - yMin)) * plotH;
  const points = energies.map((e, i) => [sx(pathLengths[i]), sy(e)]);
  const polyline = points.map(([x, y]) => `${{x.toFixed(1)}},${{y.toFixed(1)}}`).join(" ");
  const circles = points.map(([x, y], i) => (
    `<circle id="point-${{i}}" cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="4" fill="#18834a" `
    + `style="cursor:pointer" onclick="renderFrame(${{i}})" />`
  )).join("");
  container.innerHTML = (
    `<svg id="energySvg" viewBox="0 0 ${{width}} ${{height}}" width="100%">`
    + `<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="white" />`
    + `<text x="${{width / 2}}" y="16" text-anchor="middle" font-size="13" fill="#222">`
    + `Energy profile (kcal/mol vs. normalized path length) -- click a point to jump to that frame</text>`
    + `<polyline fill="none" stroke="#18834a" stroke-width="2" points="${{polyline}}" />`
    + circles + `</svg>`
  );
}}

function renderFrame(i) {{
  const chain = currentNode.trajectory[currentStep];
  currentFrame = Math.max(0, Math.min(i, chain.frames.length - 1));
  const frame = chain.frames[currentFrame];
  document.getElementById("frameSlider").max = String(chain.frames.length - 1);
  document.getElementById("frameSlider").value = String(currentFrame);
  document.getElementById("frameLabel").textContent = String(currentFrame);
  let note = frame.energy_kcal === null || frame.energy_kcal === undefined
    ? "" : frame.energy_kcal.toFixed(2) + " kcal/mol";
  if (chain.ts_index !== null && currentFrame === chain.ts_index) note += " (TS guess)";
  document.getElementById("frameEnergy").textContent = note;
  showStructureXyz(frame.xyz);
  document.querySelectorAll("#energySvg circle").forEach((c) => {{
    c.setAttribute("r", "4");
    c.setAttribute("fill", "#18834a");
  }});
  const point = document.getElementById("point-" + currentFrame);
  if (point) {{
    point.setAttribute("r", "8");
    point.setAttribute("fill", "#f59e0b");
  }}
}}

function selectStep(step) {{
  const trajectory = currentNode.trajectory;
  currentStep = Math.max(0, Math.min(step, trajectory.length - 1));
  const chain = trajectory[currentStep];
  document.getElementById("stepControls").style.display = trajectory.length > 1 ? "block" : "none";
  document.getElementById("stepSlider").max = String(trajectory.length - 1);
  document.getElementById("stepSlider").value = String(currentStep);
  document.getElementById("stepLabel").textContent = String(currentStep);
  document.getElementById("stepMax").textContent = String(trajectory.length - 1);
  renderEnergyPlot(chain.frames);
  renderFrame(chain.ts_index !== null ? chain.ts_index : 0);
}}

function selectNode(index) {{
  currentNode = nodesByIndex[index];
  document.querySelectorAll("#treeContainer [data-default-width]").forEach((el) => {{
    el.setAttribute("stroke-width", el.getAttribute("data-default-width"));
  }});
  const selected = document.getElementById("tree-node-" + index);
  if (selected) {{
    selected.setAttribute("stroke-width", String(2 * parseFloat(selected.getAttribute("data-default-width"))));
  }}
  document.querySelectorAll("#treeContainer .node-button").forEach((el) => el.classList.remove("selected"));
  const selectedButton = document.getElementById("list-node-" + index);
  if (selectedButton) {{
    selectedButton.classList.add("selected");
  }}
  selectStep(currentNode.trajectory.length - 1);
}}

document.getElementById("stepSlider").addEventListener("input", (e) => selectStep(parseInt(e.target.value, 10)));
document.getElementById("frameSlider").addEventListener("input", (e) => renderFrame(parseInt(e.target.value, 10)));
selectNode(nodes[0].index);
</script>
</body>
</html>"""


def _tree_svg(nodes: list[dict], width: int = 900, height: int = 220) -> str:
    """A minimal split-tree diagram: one clickable circle per selectable
    node, laid out by depth (y) and sibling order at that depth (x), with
    lines to each node's parent. Clicking a node loads its optimization run
    into the frame scrubber below."""
    by_depth: dict[int, list[dict]] = {}
    for node in nodes:
        by_depth.setdefault(node["depth"], []).append(node)

    max_depth = max(by_depth)
    top_pad, bottom_pad, side_pad = 30, 30, 40
    positions: dict[int, tuple[float, float]] = {}
    for depth, siblings in by_depth.items():
        y = top_pad if max_depth == 0 else top_pad + depth * (height - top_pad - bottom_pad) / max_depth
        for i, node in enumerate(siblings):
            x = width / 2 if len(siblings) == 1 else side_pad + i * (width - 2 * side_pad) / (len(siblings) - 1)
            positions[node["index"]] = (x, y)

    lines = []
    for node in nodes:
        if node["parent"] is not None and node["parent"] in positions:
            px, py = positions[node["parent"]]
            x, y = positions[node["index"]]
            lines.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#b7b7b7" stroke-width="1.5" />')

    circles = []
    for node in nodes:
        x, y = positions[node["index"]]
        circles.append(
            f'<circle id="tree-node-{node["index"]}" cx="{x:.1f}" cy="{y:.1f}" r="12" '
            f'fill="#1f77b4" stroke="#0f4872" stroke-width="1.8" data-default-width="1.8" '
            f'style="cursor:pointer" onclick="selectNode({node["index"]})" />'
            f'<text x="{x:.1f}" y="{y + 26:.1f}" text-anchor="middle" font-size="11" fill="#333">{node["label"]}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%">'
        + "".join(lines)
        + "".join(circles)
        + "</svg>"
    )


def _network_svg(pot, edge_nodes: list[dict], width: int = 900, height: int = 320) -> str:
    """A minimal reaction-network diagram: species as small labeled circles
    (laid out with networkx's circular_layout -- already a base dependency,
    deterministic, and adequate for the modest graph sizes this package
    targets), with one clickable line per edge that has a recoverable chain.
    Clicking an edge loads its candidate chain(s) into the frame scrubber
    below, exactly like clicking a tree node."""
    import networkx as nx

    graph_nodes = list(pot.graph.nodes)
    if not graph_nodes:
        return ""
    layout = nx.circular_layout(graph_nodes)
    margin = 50

    def scale(pos) -> tuple[float, float]:
        x, y = pos
        return (
            margin + (x + 1) / 2 * (width - 2 * margin),
            margin + (y + 1) / 2 * (height - 2 * margin),
        )

    positions = {n: scale(p) for n, p in layout.items()}

    lines = []
    for edge in edge_nodes:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        lines.append(
            f'<line id="tree-node-{edge["index"]}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#7e7e7e" stroke-width="3" data-default-width="3" style="cursor:pointer" '
            f'onclick="selectNode({edge["index"]})" />'
        )

    circles = []
    for node_id, (x, y) in positions.items():
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="#1f77b4" stroke="#0f4872" stroke-width="1.5" />'
            f'<text x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle" font-size="11" fill="#333">{node_id}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%">'
        + "".join(lines)
        + "".join(circles)
        + "</svg>"
    )


def plot_opt_history(chain_trajectory: List[Chain], do_3d: bool = False) -> None:
    """Plot a NEB/MSMEP optimization's energy-profile history.

    For a `NEB`/`PathMinimizer` result, call as
    `plot_opt_history(minimizer.chain_trajectory)`.
    """
    import matplotlib.pyplot as plt

    s = 8
    fs = 18

    if do_3d:
        all_chains = chain_trajectory
        ens = np.array([c.energies - c.energies[0] for c in all_chains])
        all_integrated_path_lengths = np.array(
            [c.integrated_path_length for c in all_chains]
        )
        opt_step = np.array(list(range(len(all_chains))))
        s = 7
        ax = plt.figure(figsize=(1.16 * s, s)).add_subplot(projection="3d")

        x = opt_step
        ys = all_integrated_path_lengths
        zs = ens
        for i, (xind, y) in enumerate(zip(x, ys)):
            if i < len(ys) - 1:
                ax.plot(
                    [xind] * len(y), y, "o-", zs=zs[i],
                    color="gray", markersize=3, alpha=0.1,
                )
            else:
                ax.plot([xind] * len(y), y, "o-", zs=zs[i], color="blue", markersize=3)
        ax.grid(False)
        ax.set_xlabel("optimization step", fontsize=fs)
        ax.set_ylabel("integrated path length", fontsize=fs)
        ax.set_zlabel("energy (hartrees)", fontsize=fs)
        ax.view_init(elev=20.0, azim=-45)
        plt.tight_layout()
        plt.show()
    else:
        plt.subplots(figsize=(1.16 * s, s))
        for i, chain in enumerate(chain_trajectory):
            if i == len(chain_trajectory) - 1:
                plt.plot(chain.integrated_path_length, chain.energies, "o-", alpha=1)
            else:
                plt.plot(
                    chain.integrated_path_length, chain.energies,
                    "o-", alpha=0.1, color="gray",
                )
        plt.xlabel("Integrated path length", fontsize=fs)
        plt.ylabel("Energy (kcal/mol)", fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.show()


