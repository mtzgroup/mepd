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


def plot_grad_delta_mag_history(neb) -> None:
    """Known broken, pre-existing upstream (not a port regression): calls
    `Chain._gradient_delta_mags`, which doesn't exist anywhere in this
    package or the upstream neb-dynamics source. Left as-is rather than
    guessing at the intended implementation.
    """
    import matplotlib.pyplot as plt

    s = 8
    fs = 18
    plt.subplots(figsize=(1.16 * s, s))
    projs = []
    for i, chain in enumerate(neb.chain_trajectory):
        if i == 0:
            continue
        prev_chain = neb.chain_trajectory[i - 1]
        projs.append(prev_chain._gradient_delta_mags(chain))
    plt.plot(projs)
    plt.ylabel("NEB |∆gradient|", fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.xlabel("Optimization step", fontsize=fs)
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
