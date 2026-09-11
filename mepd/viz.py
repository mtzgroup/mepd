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
import io
from typing import List

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


def animate_structure_list(structure_list):
    """Renders a list of qcdata Structure objects as an interactive 3D viewer."""
    from IPython.display import display, HTML
    from qcdata.view import generate_structure_viewer_html

    structure_html = generate_structure_viewer_html(structure_list)
    return display(HTML(structure_html))


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


def _energy_profile_svg(energies_kcal: List[float], width: int = 640, height: int = 220) -> str:
    """A minimal inline SVG line plot (frame index vs. relative energy) whose
    points are individually addressable by id (`point-<i>`), so JS can
    restyle one of them into a highlighted "bead" without redrawing the plot.
    """
    n = len(energies_kcal)
    margin = 36
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    y_min, y_max = min(energies_kcal), max(energies_kcal)
    if y_min == y_max:
        y_max = y_min + 1.0

    def sx(i: int) -> float:
        return margin + (i / max(n - 1, 1)) * plot_w

    def sy(e: float) -> float:
        return margin + (1 - (e - y_min) / (y_max - y_min)) * plot_h

    points = [(sx(i), sy(e)) for i, e in enumerate(energies_kcal)]
    polyline_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "".join(
        f'<circle id="point-{i}" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#18834a" '
        f'style="cursor:pointer" onclick="renderFrame({i})" />'
        for i, (x, y) in enumerate(points)
    )
    return f"""<svg id="energySvg" viewBox="0 0 {width} {height}" width="100%">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{width / 2}" y="16" text-anchor="middle" font-size="13" fill="#222">Energy profile (kcal/mol vs. frame 0) -- click a point to jump to that frame</text>
<polyline fill="none" stroke="#18834a" stroke-width="2" points="{polyline_pts}" />
{circles}
</svg>"""


def render_chain_html(
    chain: Chain,
    title: str = "mepd chain visualization",
    show_atom_indices: bool = False,
) -> str:
    """Render a standalone HTML page with an interactive frame scrubber: a
    slider steps through every frame in `chain`, updating the 3D structure
    viewer and highlighting the corresponding point ("bead") on the
    energy-profile plot in lockstep -- so a specific frame (e.g. the apparent
    TS) can be identified by index for follow-up (`mepd ts --guess`) while
    seeing exactly where it sits on the energy profile. Used by
    `mepd visualize`.
    """
    import json

    from qcdata.view import generate_structure_viewer_html

    structures = [node.structure for node in chain.nodes]
    n = len(structures)
    has_energies = chain._energies_already_computed and n > 1
    ind_ts = int(np.argmax(chain.energies)) if has_energies else None

    if n == 1:
        viewer_html = generate_structure_viewer_html(structures[0], titles=["Frame 0"])
        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
{_3DMOL_CDN_SCRIPT}
</head>
<body>
<h2>{title}</h2>
{viewer_html}
</body>
</html>"""

    # Pre-render one standalone structure-viewer document per frame; the
    # slider swaps an <iframe>'s srcdoc between them (base64-encoded so the
    # per-frame HTML -- which itself contains '<', '"', backticks -- can sit
    # safely inside a JS string literal without escaping).
    frame_docs_b64 = []
    for i, structure in enumerate(structures):
        frame_viewer = generate_structure_viewer_html(
            structure, titles=[f"Frame {i}"], show_indices=show_atom_indices
        )
        frame_doc = (
            f"<!doctype html><html><head><meta charset='utf-8'>"
            f"{_3DMOL_CDN_SCRIPT}</head><body>{frame_viewer}</body></html>"
        )
        frame_docs_b64.append(base64.b64encode(frame_doc.encode("utf-8")).decode("ascii"))

    energy_labels = []
    plot_html = ""
    if has_energies:
        energies_kcal = list(chain.energies_kcalmol)
        energy_labels = [f"{e:+.2f} kcal/mol" for e in energies_kcal]
        plot_html = _energy_profile_svg(energies_kcal)

    frames_json = json.dumps(frame_docs_b64)
    energy_labels_json = json.dumps(energy_labels)
    ts_note = f'if (i === {ind_ts}) note += " (TS guess)";' if ind_ts is not None else ""

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
</head>
<body>
<h2>{title}</h2>
<label for="frameSlider">Frame: <span id="frameLabel">0</span> <span id="frameEnergy"></span></label><br/>
<input id="frameSlider" type="range" min="0" max="{n - 1}" value="0" step="1" style="width: min(720px, 90vw);" />
<div id="plotContainer">{plot_html}</div>
<iframe id="structureFrame" style="width: 100%; height: 520px; border: 0;" title="Structure viewer"></iframe>
<script>
const frameDocs = {frames_json};
const energyLabels = {energy_labels_json};
function renderFrame(i) {{
  document.getElementById("frameSlider").value = String(i);
  document.getElementById("frameLabel").textContent = String(i);
  let note = energyLabels[i] || "";
  {ts_note}
  document.getElementById("frameEnergy").textContent = note;
  document.getElementById("structureFrame").srcdoc = atob(frameDocs[i]);
  document.querySelectorAll("#energySvg circle").forEach((c) => {{
    c.setAttribute("r", "4");
    c.setAttribute("fill", "#18834a");
  }});
  const point = document.getElementById("point-" + i);
  if (point) {{
    point.setAttribute("r", "8");
    point.setAttribute("fill", "#f59e0b");
  }}
}}
document.getElementById("frameSlider").addEventListener("input", (e) => renderFrame(parseInt(e.target.value, 10)));
renderFrame({ind_ts if ind_ts is not None else 0});
</script>
</body>
</html>"""


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


def animate_trajectory(
    traj, c_irc, xmin=-0.1, xmax=1.1, ymin=-1, ymax=200,
    return_anim=False, flip_chains=False,
):
    import matplotlib.pyplot as plt
    import matplotlib.animation
    from IPython.display import HTML
    import mepd.chainhelpers as ch

    fig, ax = plt.subplots()
    fs = 18

    (l,) = ax.plot([], [], "o-", label="fsm")
    rxn_coord = None
    if c_irc:
        rxn_coord = ch.get_rxn_coordinate(c_irc)
        disps = np.array(ch.get_projections(c_irc, rxn_coord))
        ax.plot(disps, c_irc.energies_kcalmol, "-", color="black", label="irc")
        ax.scatter(
            [disps[c_irc.energies.argmax()]], [max(c_irc.energies_kcalmol)],
            marker="x", color="black", s=50, label="TS",
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    frontconst = -1 if flip_chains else 1

    def animate(i):
        if c_irc:
            disps = np.array(
                ch.get_projections(traj[i], rxn_coord, ts_geom=c_irc.get_ts_node())
            )
            l.set_data(frontconst * disps, traj[i].energies_kcalmol)
        else:
            l.set_data(traj[i].integrated_path_length, traj[i].energies_kcalmol)

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(traj))
    plt.ylabel("Energies (kcal/mol)", fontsize=fs)
    plt.xlabel("Reaction coordinate", fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(fontsize=fs)
    plt.tight_layout()
    if return_anim:
        return ani
    return HTML(ani.to_jshtml())


def plot_chain_distances(neb) -> None:
    import matplotlib.pyplot as plt
    import mepd.chainhelpers as ch

    distances = ch._calculate_chain_distances(neb.chain_trajectory)
    fs = 18
    s = 8
    plt.subplots(figsize=(1.16 * s, s))
    plt.plot(distances, "o-")
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.ylabel("Distance to previous chain", fontsize=fs)
    plt.xlabel("Chain id", fontsize=fs)
    plt.show()


def plot_projector_history(neb, var="gradients") -> None:
    import matplotlib.pyplot as plt

    s = 8
    fs = 18
    plt.subplots(figsize=(1.16 * s, s))
    projs = []
    for i, chain in enumerate(neb.chain_trajectory):
        if i == 0:
            continue
        prev_chain = neb.chain_trajectory[i - 1]
        if var == "gradients":
            projs.append(prev_chain._gradient_correlation(chain))
        elif var == "tangents":
            projs.append(prev_chain._tangent_correlations(chain))
        else:
            raise ValueError(f"Unrecognized var: {var}")
    plt.plot(projs)
    plt.ylabel(f"NEB {var} correlation", fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("Optimization step", fontsize=fs)
    plt.show()


def plot_convergence_metrics(neb, do_indiv=False) -> None:
    import matplotlib.pyplot as plt
    import mepd.chainhelpers as ch

    ct = neb.chain_trajectory
    avg_rms_gperp = []
    max_rms_gperp = []
    avg_rms_g = []
    barr_height = []
    ts_gperp = []
    grad_infnorm = []

    for ind in range(1, len(ct)):
        avg_rms_g.append(sum(ct[ind].rms_gradients[1:-1]) / (len(ct[ind]) - 2))
        avg_rms_gperp.append(sum(ct[ind].rms_gperps[1:-1]) / (len(ct[ind]) - 2))
        max_rms_gperp.append(max(ct[ind].rms_gperps))
        barr_height.append(abs(ct[ind].get_eA_chain() - ct[ind - 1].get_eA_chain()))
        ts_node_ind = ct[ind].energies.argmax()
        ts_gperp.append(np.max(ch.get_g_perps(ct[ind])[ts_node_ind]))
        grad_infnorm.append(np.amax(abs(ch.compute_NEB_gradient(ct[ind]))))

    if do_indiv:
        def plot_with_hline(data, label, y_hline, hline_label, hline_color, ylabel):
            f, ax = plt.subplots()
            plt.plot(data, label=label)
            plt.ylabel(ylabel)
            xmin, xmax = ax.get_xlim()
            ax.hlines(
                y=y_hline, xmin=xmin, xmax=xmax,
                label=hline_label, linestyle="--", color=hline_color,
            )
            f.legend()
            plt.show()

        plot_with_hline(
            avg_rms_gperp, label="RMS Grad$_{\\perp}$",
            y_hline=neb.parameters.rms_grad_thre,
            hline_label="rms_grad_thre", hline_color="blue", ylabel="Gradient data",
        )
        plot_with_hline(
            max_rms_gperp, label="Max RMS Grad$_{\\perp}$",
            y_hline=neb.parameters.max_rms_grad_thre,
            hline_label="max_rms_grad_thre", hline_color="orange", ylabel="Gradient data",
        )
        plot_with_hline(
            ts_gperp, label="TS gperp",
            y_hline=neb.parameters.ts_grad_thre,
            hline_label="ts_grad_thre", hline_color="green", ylabel="Gradient data",
        )
        plot_with_hline(
            barr_height, label="barr_height_delta",
            y_hline=neb.parameters.barrier_thre,
            hline_label="barrier_thre", hline_color="purple", ylabel="Barrier height data",
        )
    else:
        data_list = [
            (avg_rms_gperp, "RMS Grad$_{\\perp}$", neb.parameters.rms_grad_thre, "rms_grad_thre", "blue"),
            (max_rms_gperp, "Max RMS Grad$_{\\perp}$", neb.parameters.max_rms_grad_thre, "max_rms_grad_thre", "orange"),
            (ts_gperp, "TS gperp", neb.parameters.ts_grad_thre, "ts_grad_thre", "green"),
            (grad_infnorm, "Grad infnorm", neb.parameters.ts_grad_thre, "grad_infnorm", "gray"),
        ]

        f, ax = plt.subplots()
        xmin = xmax = 0
        for data, label, hline, hline_label, color in data_list:
            ax.plot(data, label=label)
            xmin, xmax = ax.get_xlim()
            ax.hlines(y=hline, xmin=xmin, xmax=xmax, label=hline_label, linestyle="--", color=color)
        ax.set_ylabel("Gradient data")

        ax2 = ax.twinx()
        ax2.plot(barr_height, "o--", label="barr_height_delta", color="purple")
        ax2.set_ylabel("Barrier height data")
        ax2.hlines(
            y=neb.parameters.barrier_thre, xmin=xmin, xmax=xmax,
            label="barrier_thre", linestyle="--", color="purple",
        )

        f.legend(loc="upper left")
        plt.show()
