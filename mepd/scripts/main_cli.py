from __future__ import annotations
import base64
import contextlib
import io
import json
import logging
from dataclasses import dataclass
import networkx as nx
import numpy as np
from typing import Any, List, Literal
from itertools import product
from mepd.pot import plot_results_from_pot_obj
from mepd.pot import Pot
from mepd.helper_functions import (
    compute_irc_chain,
    parse_nma_freq_data,
)
from mepd.inputs import NetworkInputs, ChainInputs
from mepd.NetworkBuilder import NetworkBuilder
from mepd.qcio_structure_helpers import read_multiple_structure_from_file
from mepd.nodes.nodehelpers import displace_by_dr
from mepd.msmep import MSMEP
from mepd.chain import Chain
from mepd.molecule import Molecule
from mepd.engines.engine import build_hessian_result_from_matrix
from mepd.TreeNode import TreeNode
from mepd.neb import NEB
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import (
    _is_connectivity_identical,
    is_identical,
)
from mepd.elementarystep import check_if_elem_step, _write_nodes_xyz
from mepd.inputs import RunInputs
from mepd.irc_network import build_irc_network
from mepd.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from mepd.geodesic_interpolation2.fileio import write_xyz

import typer
from typing_extensions import Annotated

import os
import tempfile
from openbabel import openbabel
from qcio import Structure, ProgramOutput
from qcio.view import generate_structure_viewer_html
from qcop.exceptions import ExternalProgramError
import sys
from pathlib import Path
import time
import traceback
from datetime import datetime
import webbrowser
import shutil

from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.status import Status
from rich.table import Table
from rich.box import Box
from rich.text import Text
from rich.syntax import Syntax
from rich import box
from mepd.chainhelpers import generate_neb_plot
from mepd.scripts._cli_results import (
    _cost_report_path,
    _create_recursive_request_record,
    _load_status_snapshot,
    _recursive_split_manifest_path,
    _request_record_summary,
    _resolve_status_artifact,
    _run_status_path,
    _summarize_network_file,
    _upsert_request_record,
    _write_chain_history_with_nan_fallback,
    _write_chain_with_nan_fallback,
    _write_json_atomic,
    _write_neb_results_with_history as _write_neb_results_with_history_impl,
    _write_cost_report,
    _write_recursive_split_manifest,
    _write_run_status,
)
from mepd.scripts._cli_runtime import (
    BANNER,
    _configure_cli_logging,
    console,
    create_progress,
    print_banner,
)
from mepd.scripts import _cli_visualize
from mepd.scripts.progress import stop_status

# Custom theme for Claude Code-like styling
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "header": "bold magenta",
    "dim": "dim",
})


class _SuppressWarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING


logging.getLogger().addFilter(_SuppressWarningFilter())


openbabel.obErrorLog.SetOutputLevel(0)


@contextlib.contextmanager
def _suppress_rdkit_valence_warnings():
    try:
        from rdkit import RDLogger  # type: ignore
    except Exception:
        yield
        return
    with contextlib.suppress(Exception):
        RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            RDLogger.EnableLog("rdApp.*")


def _parse_kmc_initial_condition_overrides(
    overrides: list[str] | None,
) -> dict[int, float] | None:
    if not overrides:
        return None

    parsed: dict[int, float] = {}
    for override in overrides:
        if "=" not in override:
            raise typer.BadParameter(
                f"Invalid --initial-condition '{override}'. Use NODE=VALUE, for example 0=1.0."
            )
        node_text, value_text = override.split("=", 1)
        try:
            node_index = int(node_text.strip())
            value = float(value_text.strip())
        except ValueError as exc:
            raise typer.BadParameter(
                f"Invalid --initial-condition '{override}'. Use NODE=VALUE, for example 0=1.0."
            ) from exc
        parsed[node_index] = value
    return parsed


def _parse_xyz_text_to_structures(
    xyz_text: str,
    *,
    charge: int | None = 0,
    multiplicity: int | None = 1,
) -> list[Structure]:
    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as handle:
        handle.write(xyz_text)
        temp_fp = Path(handle.name)
    try:
        return read_multiple_structure_from_file(
            str(temp_fp),
            charge=charge,
            spinmult=multiplicity,
        )
    finally:
        with contextlib.suppress(OSError):
            temp_fp.unlink()


app = typer.Typer(
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)

NetworkCompletionMode = Literal["linear", "all-to-all"]


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """Show banner before running any command."""
    _configure_cli_logging()
    if ctx.invoked_subcommand is None:
        print_banner()


_VisualizationData = _cli_visualize.VisualizationData
_truncate_label = _cli_visualize._truncate_label
_build_ascii_energy_profile = _cli_visualize._build_ascii_energy_profile


def _visualization_deps() -> _cli_visualize.VisualizationDeps:
    return _cli_visualize.VisualizationDeps(
        Chain=Chain,
        ChainInputs=ChainInputs,
        NEB=NEB,
        Path=Path,
        Pot=Pot,
        Structure=Structure,
        StructureNode=StructureNode,
        TreeNode=TreeNode,
        is_connectivity_identical=_is_connectivity_identical,
        nx=nx,
        np=np,
        read_multiple_structure_from_file=read_multiple_structure_from_file,
        reverse_chain=_reverse_chain,
        concat_chains=_concat_chains,
        collect_tree_layers_for_visualization=_collect_tree_layers_for_visualization,
        match_network_endpoint_indices_by_connectivity=_match_network_endpoint_indices_by_connectivity,
        find_pot_root_node_index=_find_pot_root_node_index,
        find_pot_target_node_index=_find_pot_target_node_index,
        best_path_by_apparent_barrier=_best_path_by_apparent_barrier,
        path_chain_from_pot=_path_chain_from_pot,
        best_chain_for_directed_edge=_best_chain_for_directed_edge,
        load_network_endpoint_hints=_load_network_endpoint_hints,
        load_network_endpoint_structures=_load_network_endpoint_structures,
    )


def _ascii_profile_for_chain(chain: Chain):
    try:
        energies = chain.energies_kcalmol
    except Exception as exc:
        message = _cli_visualize.format_exception_message(exc)
        console.print(f"[yellow]⚠ Could not compute energy profile: {message}[/yellow]")
        details = _cli_visualize.extract_electronic_structure_error_details(
            getattr(exc, "obj", None)
        )
        for detail in details:
            console.print(f"[yellow]  • {detail}[/yellow]")
        return

    labels = [str(i) for i, _ in enumerate(chain.nodes)]
    plot = _build_ascii_energy_profile(energies, labels)
    console.print("\nASCII Reaction Profile (Energy vs Node)")
    console.print(plot, markup=False)


def _write_neb_results_with_history(
    neb_result, fp: Path, write_qcio: bool = False
) -> bool:
    return _write_neb_results_with_history_impl(
        neb_result, fp, console=console, write_qcio=write_qcio
    )


def _maybe_hydrate_nodes_from_xyz_sidecars(
    *,
    geometries: str | Path,
    nodes: list[StructureNode],
    chain_inputs: ChainInputs,
    charge: int,
    multiplicity: int,
) -> bool:
    geom_fp = Path(geometries).expanduser().resolve()
    energies_fp = geom_fp.parent / f"{geom_fp.stem}.energies"
    gradients_fp = geom_fp.parent / f"{geom_fp.stem}.gradients"
    grad_shape_fp = geom_fp.parent / f"{geom_fp.stem}_grad_shapes.txt"
    legacy_grad_shape_fp = geom_fp.parent / "grad_shapes.txt"

    if not energies_fp.exists() or not gradients_fp.exists():
        return False
    if not grad_shape_fp.exists() and legacy_grad_shape_fp.exists():
        grad_shape_fp = legacy_grad_shape_fp
    if not grad_shape_fp.exists():
        console.print(
            f"[yellow]⚠ Found {energies_fp.name} and {gradients_fp.name} but no grad shape file. "
            "Skipping cache hydration and computing energies/gradients normally.[/yellow]"
        )
        return False

    try:
        try:
            cache_chain = Chain.from_xyz(
                fp=geom_fp,
                parameters=chain_inputs,
                charge=charge,
                spinmult=multiplicity,
            )
        except ValueError:
            cache_chain = Chain.from_xyz(
                fp=geom_fp,
                parameters=chain_inputs,
                charge=None,
                spinmult=None,
            )
    except Exception as exc:
        console.print(
            f"[yellow]⚠ Failed to load cache sidecars for {geom_fp.name}: {exc}. "
            "Continuing without cached values.[/yellow]"
        )
        return False

    if len(cache_chain.nodes) != len(nodes):
        console.print(
            f"[yellow]⚠ Cache sidecar node count mismatch for {geom_fp.name} "
            f"(cache={len(cache_chain.nodes)}, run={len(nodes)}). "
            "Continuing without cached values.[/yellow]"
        )
        return False

    has_full_cache = all(
        getattr(node, "_cached_energy", None) is not None
        and getattr(node, "_cached_gradient", None) is not None
        for node in cache_chain.nodes
    )
    if not has_full_cache:
        console.print(
            f"[yellow]⚠ Cache sidecars for {geom_fp.name} were incomplete. "
            "Continuing without cached values.[/yellow]"
        )
        return False

    for dst, src in zip(nodes, cache_chain.nodes):
        dst._cached_result = getattr(src, "_cached_result", None)
        dst._cached_energy = getattr(src, "_cached_energy", None)
        dst._cached_gradient = getattr(src, "_cached_gradient", None)

    console.print(
        f"[dim]Loaded cached energies/gradients from {energies_fp.name} and {gradients_fp.name}.[/dim]"
    )
    return True


def _collect_tree_layers_for_visualization(tree: TreeNode) -> list[dict]:
    return _cli_visualize._collect_tree_layers_for_visualization(tree)


def _load_visualization_data(
    result_path: Path,
    charge: int = 0,
    multiplicity: int = 1,
) -> _VisualizationData:
    return _cli_visualize._load_visualization_data(
        result_path=result_path,
        deps=_visualization_deps(),
        charge=charge,
        multiplicity=multiplicity,
    )


def _load_chain_for_visualization(
    result_path: Path,
    charge: int = 0,
    multiplicity: int = 1,
) -> Chain:
    return _cli_visualize._load_chain_for_visualization(
        result_path=result_path,
        deps=_visualization_deps(),
        charge=charge,
        multiplicity=multiplicity,
    )


def _generate_opt_history_plot_b64(
    chain_trajectory: list[Chain],
    selected_index: int,
) -> str:
    return ""


def _best_chain_for_directed_edge(pot: Pot, source: int, target: int) -> Chain:
    return _cli_visualize._best_chain_for_directed_edge(
        pot,
        source,
        target,
        _visualization_deps(),
    )


def _best_path_by_apparent_barrier(
    pot: Pot,
    root_idx: int,
    target_idx: int,
) -> tuple[list[int], float] | tuple[None, None]:
    return _cli_visualize._best_path_by_apparent_barrier(
        pot,
        root_idx,
        target_idx,
        _visualization_deps(),
    )


def _find_pot_root_node_index(pot: Pot) -> int | None:
    return _cli_visualize._find_pot_root_node_index(pot)


def _load_network_endpoint_hints(network_json_fp: Path) -> dict | None:
    return _cli_visualize._load_network_endpoint_hints(network_json_fp)


def _load_network_endpoint_structures(
    network_json_fp: Path,
) -> tuple[StructureNode | None, StructureNode | None]:
    return _cli_visualize._load_network_endpoint_structures(
        network_json_fp,
        _visualization_deps(),
    )


def _match_network_endpoint_indices_by_connectivity(
    pot: Pot,
    start_node: StructureNode | None,
    end_node: StructureNode | None,
) -> dict | None:
    return _cli_visualize._match_network_endpoint_indices_by_connectivity(
        pot,
        start_node,
        end_node,
        _visualization_deps(),
    )


def _find_pot_target_node_index(
    pot: Pot,
    target_idx_hint: int | None = None,
) -> int | None:
    return _cli_visualize._find_pot_target_node_index(
        pot,
        _visualization_deps(),
        target_idx_hint,
    )


def _path_chain_from_pot(pot: Pot, path: list[int]) -> Chain | None:
    return _cli_visualize._path_chain_from_pot(pot, path, _visualization_deps())


def _build_network_visualization_payload(
    pot: Pot,
    atom_indices: list[int] | None = None,
    endpoint_hints: dict | None = None,
) -> dict:
    return _cli_visualize._build_network_visualization_payload(
        pot,
        _visualization_deps(),
        atom_indices=atom_indices,
        endpoint_hints=endpoint_hints,
    )


def _build_chain_visualizer_html(
    chain: Chain,
    chain_trajectory: list[Chain] | None = None,
    tree_layers: list[dict] | None = None,
    network_payload: dict | None = None,
) -> str:
    return _cli_visualize._build_chain_visualizer_html(
        chain,
        chain_trajectory=chain_trajectory,
        tree_layers=tree_layers,
        network_payload=network_payload,
    )


def _parse_visualize_atom_indices(
    atom_indices: str | None = None,
) -> list[int] | None:
    return _cli_visualize._parse_visualize_atom_indices(
        atom_indices=atom_indices,
    )


def _subset_chain_for_visualization(
    chain: Chain,
    atom_indices: list[int],
) -> Chain:
    return _cli_visualize._subset_chain_for_visualization(
        chain,
        atom_indices,
        _visualization_deps(),
    )


def _subset_chain_trajectory_for_visualization(
    chain_trajectory: list[Chain],
    atom_indices: list[int],
) -> list[Chain]:
    return _cli_visualize._subset_chain_trajectory_for_visualization(
        chain_trajectory,
        atom_indices,
        _visualization_deps(),
    )


def _subset_tree_layers_for_visualization(
    tree_layers: list[dict],
    atom_indices: list[int],
) -> list[dict]:
    return _cli_visualize._subset_tree_layers_for_visualization(
        tree_layers,
        atom_indices,
        _visualization_deps(),
    )


def _compute_ts_node(engine, ts_guess: StructureNode, bigchem: bool = False):
    """Run TS optimization through the engine and normalize to (StructureNode|None, ProgramOutput|None)."""
    try:
        if bigchem and hasattr(engine, "_compute_ts_result"):
            raw_out = engine._compute_ts_result(
                node=ts_guess, use_bigchem=True)
            if getattr(raw_out, "success", False):
                return StructureNode(structure=raw_out.return_result), raw_out
            return None, raw_out

        if hasattr(engine, "compute_transition_state"):
            raw_out = engine.compute_transition_state(node=ts_guess)
            if isinstance(raw_out, StructureNode):
                return raw_out, None
            if isinstance(raw_out, ProgramOutput):
                if getattr(raw_out, "success", False):
                    return StructureNode(structure=raw_out.return_result), raw_out
                return None, raw_out
            if getattr(raw_out, "success", False) and getattr(raw_out, "return_result", None) is not None:
                return StructureNode(structure=raw_out.return_result), raw_out
            return None, raw_out

        if hasattr(engine, "_compute_ts_result"):
            raw_out = engine._compute_ts_result(
                node=ts_guess, use_bigchem=bigchem)
            if getattr(raw_out, "success", False):
                return StructureNode(structure=raw_out.return_result), raw_out
            return None, raw_out

        raise AttributeError(
            "Engine does not implement transition-state optimization.")
    except Exception as exc:
        program_output = getattr(exc, "program_output", None)
        if program_output is not None:
            return None, program_output
        raise


def _extract_normal_modes_from_hessian_result(
    hessres,
) -> tuple[list[np.ndarray], list[float]]:
    def _is_geometry_linear(geometry: np.ndarray, tol: float = 1e-7) -> bool:
        coords = np.asarray(geometry, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            return False
        natoms = int(coords.shape[0])
        if natoms <= 1:
            return False
        if natoms == 2:
            return True
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        return int(np.linalg.matrix_rank(centered, tol=tol)) <= 1

    def _expected_vibrational_mode_count(natoms: int, is_linear: bool) -> int:
        if natoms <= 1:
            return 0
        if natoms == 2:
            return 1
        return max(0, (3 * natoms) - (5 if is_linear else 6))

    def _infer_natoms_from_mode(mode: np.ndarray) -> int | None:
        arr = np.asarray(mode, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return int(arr.shape[0])
        if arr.ndim == 1 and arr.size % 3 == 0:
            return int(arr.size // 3)
        return None

    def _trim_rigid_body_modes_if_present(
        normal_modes: list[np.ndarray],
        frequencies: list[float],
    ) -> tuple[list[np.ndarray], list[float]]:
        if len(normal_modes) == 0 or len(normal_modes) != len(frequencies):
            return normal_modes, frequencies

        natoms: int | None = None
        is_linear = False
        structure = getattr(getattr(hessres, "input_data", None), "structure", None)
        geometry = getattr(structure, "geometry", None)
        if geometry is not None:
            geom = np.asarray(geometry, dtype=float)
            if geom.ndim == 2 and geom.shape[1] == 3:
                natoms = int(geom.shape[0])
                is_linear = _is_geometry_linear(geom)
        if natoms is None:
            natoms = _infer_natoms_from_mode(normal_modes[0])
            if natoms is None:
                return normal_modes, frequencies
            if natoms == 2:
                is_linear = True

        ncart = 3 * natoms
        if len(normal_modes) != ncart:
            return normal_modes, frequencies

        n_vib = _expected_vibrational_mode_count(natoms=natoms, is_linear=is_linear)
        if n_vib >= len(normal_modes):
            return normal_modes, frequencies
        n_drop = len(normal_modes) - n_vib
        if n_drop <= 0:
            return normal_modes, frequencies

        order = np.argsort(np.abs(np.asarray(frequencies, dtype=float)))
        drop_inds = {int(idx) for idx in order[:n_drop].tolist()}
        keep_inds = [idx for idx in range(len(normal_modes)) if idx not in drop_inds]
        return [normal_modes[idx] for idx in keep_inds], [frequencies[idx] for idx in keep_inds]

    results = getattr(hessres, "results", None)
    if results is None:
        raise ValueError("Hessian result is missing `results`.")

    modes = getattr(results, "normal_modes_cartesian", None)
    freqs = getattr(results, "freqs_wavenumber", None)
    if modes is not None and len(modes) > 0:
        normal_modes = [np.array(mode) for mode in modes]
        frequencies = [float(freq) for freq in (freqs or [])]
        return _trim_rigid_body_modes_if_present(normal_modes, frequencies)

    normal_modes, frequencies = parse_nma_freq_data(hessres)
    if len(normal_modes) == 0:
        raise ValueError("No normal modes found in Hessian result.")
    parsed_modes = [np.array(mode) for mode in normal_modes]
    parsed_freqs = [float(freq) for freq in frequencies]
    return _trim_rigid_body_modes_if_present(parsed_modes, parsed_freqs)


def _compute_hessian_result_for_sampling(engine, node: StructureNode):
    if hasattr(engine, "_compute_hessian_result"):
        return engine._compute_hessian_result(node)
    hessian = np.asarray(engine.compute_hessian(node), dtype=float)
    return build_hessian_result_from_matrix(node=node, hessian=hessian)


def _resolve_command_base_path(geometry: str, name: str | None) -> Path:
    if name is None:
        return Path.cwd() / Path(geometry).stem
    raw = Path(name)
    if raw.suffix:
        raw = raw.with_suffix("")
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    return raw


def _extract_minima_nodes(history: TreeNode) -> list[StructureNode]:
    """Extract minima candidates from recursive cheap-history leaves."""
    minima: list[StructureNode] = []
    for leaf in history.ordered_leaves:
        if not leaf.data or not leaf.data.chain_trajectory:
            continue
        final_chain = leaf.data.chain_trajectory[-1]
        if len(final_chain.nodes) == 0:
            continue
        minima.append(final_chain[0].copy())
        minima.append(final_chain[-1].copy())
    return minima


def _load_precomputed_refine_source(
    source: Any,
    *,
    charge: int,
    multiplicity: int,
) -> TreeNode | NEB | Chain | Pot:
    """Load a refine source from an object or filesystem path."""
    if isinstance(source, (TreeNode, NEB, Chain, Pot)):
        return source

    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            f"Precomputed source path does not exist: {source_path}"
        )

    def _resolve_network_source_chain_or_pot(network_fp: Path) -> Chain | Pot:
        with contextlib.suppress(Exception):
            return Pot.read_from_disk(network_fp)
        chain = _load_best_path_chain_from_network_json(network_fp)
        if chain is None:
            raise ValueError(
                f"Could not extract a best-path chain from network source: {network_fp}"
            )
        return chain

    if source_path.is_dir():
        adj_matrix_fp = source_path / "adj_matrix.txt"
        if adj_matrix_fp.exists():
            return TreeNode.read_from_disk(
                folder_name=source_path,
                charge=charge,
                multiplicity=multiplicity,
            )
        network_fps = sorted(source_path.glob("*_network.json"))
        if len(network_fps) == 1:
            return _resolve_network_source_chain_or_pot(network_fps[0])
        if len(network_fps) > 1:
            raise ValueError(
                "Network-completion directory contains multiple *_network.json files. "
                "Pass the desired network JSON file path explicitly."
            )
        raise ValueError(
            "Directory source must be a TreeNode folder containing adj_matrix.txt "
            "or a *_network_completion directory containing exactly one *_network.json."
        )

    if source_path.suffix.lower() == ".json":
        return _resolve_network_source_chain_or_pot(source_path)

    history_folder = source_path.parent / f"{source_path.stem}_history"
    if history_folder.exists():
        return NEB.read_from_disk(
            fp=source_path,
            history_folder=history_folder,
            charge=charge,
            multiplicity=multiplicity,
        )
    if source_path.suffix.lower() == ".xyz":
        try:
            chain = Chain.from_xyz(
                source_path,
                parameters=ChainInputs(),
                charge=charge,
                spinmult=multiplicity,
            )
        except ValueError:
            structures = read_multiple_structure_from_file(
                str(source_path), charge=None, spinmult=None
            )
            if len(structures) >= 2:
                return Chain.model_validate(
                    {
                        "nodes": [StructureNode(structure=s) for s in structures],
                        "parameters": ChainInputs(),
                    }
                )
        if len(chain.nodes) >= 2:
            return chain
    raise ValueError(
        "File source must be one of: network .json, "
        "NEB .xyz with sibling '<stem>_history/' folder, or multi-structure chain .xyz."
    )


def _extract_refine_source_chain_and_minima(
    source_obj: TreeNode | NEB | Chain | Pot,
) -> tuple[Chain, list[StructureNode], ChainInputs, str]:
    if isinstance(source_obj, TreeNode):
        cheap_output_chain = source_obj.output_chain
        cheap_minima = _extract_minima_nodes(source_obj)
        source_kind = "TreeNode"
    elif isinstance(source_obj, NEB):
        cheap_output_chain = (
            source_obj.chain_trajectory[-1]
            if source_obj.chain_trajectory
            else source_obj.optimized
        )
        if cheap_output_chain is None:
            raise ValueError("Loaded NEB source has no optimized chain.")
        cheap_minima = _extract_minima_nodes_from_chain(cheap_output_chain)
        source_kind = "NEB"
    elif isinstance(source_obj, Chain):
        cheap_output_chain = source_obj.copy()
        cheap_minima = [node.copy() for node in cheap_output_chain.nodes]
        source_kind = "NetworkBestPath"
    elif isinstance(source_obj, Pot):
        cheap_output_chain = _load_best_path_chain_from_pot(source_obj)
        if cheap_output_chain is None:
            raise ValueError("Loaded network source has no extractable path chain.")
        cheap_minima = [node.copy() for node in cheap_output_chain.nodes]
        source_kind = "Network"
    else:
        raise TypeError(
            "source_obj must be a TreeNode, NEB, Chain, or Pot instance."
        )

    if cheap_output_chain is None:
        raise ValueError("Precomputed source produced no output chain.")
    if len(cheap_output_chain.nodes) < 2:
        raise ValueError("Precomputed source output chain must contain at least 2 nodes.")
    return (
        cheap_output_chain,
        cheap_minima,
        cheap_output_chain.parameters,
        source_kind,
    )


def _extract_minima_nodes_from_chain(chain: Chain) -> list[StructureNode]:
    """Extract endpoints + strict local minima from a single optimized chain."""
    if len(chain.nodes) == 0:
        return []
    if len(chain.nodes) <= 2:
        return [node.copy() for node in chain.nodes]
    energies = chain.energies
    minima_inds = {0, len(chain.nodes) - 1}
    for i in range(1, len(chain.nodes) - 1):
        if energies[i] < energies[i - 1] and energies[i] < energies[i + 1]:
            minima_inds.add(i)
    return [chain.nodes[i].copy() for i in sorted(minima_inds)]


def _extract_local_maxima_nodes_from_chain(chain: Chain) -> list[StructureNode]:
    """Extract strict interior local maxima from a chain energy profile."""
    if len(chain.nodes) < 3:
        return []
    with contextlib.suppress(Exception):
        energies = np.asarray(chain.energies, dtype=float)
        if energies.shape[0] != len(chain.nodes):
            return []
        maxima: list[StructureNode] = []
        for i in range(1, len(chain.nodes) - 1):
            if (
                np.isfinite(energies[i - 1])
                and np.isfinite(energies[i])
                and np.isfinite(energies[i + 1])
                and energies[i] > energies[i - 1]
                and energies[i] > energies[i + 1]
            ):
                maxima.append(chain.nodes[i].copy())
        return maxima
    return []


def _dedupe_minima_nodes(nodes: list[StructureNode], chain_inputs: ChainInputs) -> list[StructureNode]:
    """Drop duplicate minima by geometry/graph identity."""
    unique_nodes: list[StructureNode] = []
    for node in nodes:
        duplicate = False
        for existing in unique_nodes:
            if is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False,
            ):
                duplicate = True
                break
        if not duplicate:
            unique_nodes.append(node.copy())
    return unique_nodes


def _load_best_path_chain_from_pot(
    pot: Pot,
    *,
    root_idx: int | None = None,
    target_idx: int | None = None,
) -> Chain | None:
    if root_idx is None:
        root_idx = _find_pot_root_node_index(pot)
    if target_idx is None:
        target_idx = _find_pot_target_node_index(pot)
    if (
        root_idx is not None
        and target_idx is not None
        and nx.has_path(pot.graph, root_idx, target_idx)
    ):
        best_path_nodes, _ = _best_path_by_apparent_barrier(
            pot,
            root_idx=root_idx,
            target_idx=target_idx,
        )
        if best_path_nodes:
            chain = _path_chain_from_pot(pot, [int(v) for v in best_path_nodes])
            if chain is not None:
                return chain

    first_edge = next(iter(pot.graph.edges), None)
    if first_edge is None:
        return None
    with contextlib.suppress(Exception):
        return _best_chain_for_directed_edge(
            pot, int(first_edge[0]), int(first_edge[1])
        ).copy()
    return None


def _load_best_path_chain_from_network_json(network_fp: Path) -> Chain | None:
    network_path = Path(network_fp).expanduser().resolve()
    if not network_path.exists():
        return None

    best_path_candidates: list[Path] = []
    if network_path.name.endswith("_network.json"):
        best_path_candidates.append(
            network_path.with_name(
                network_path.name.replace("_network.json", "_best_path.json")
            )
        )
    best_path_candidates.append(
        network_path.parent / f"{network_path.stem}_best_path.json"
    )
    # dedupe while preserving order
    seen = set()
    deduped_candidates = []
    for candidate in best_path_candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(candidate)

    for best_path_fp in deduped_candidates:
        if not best_path_fp.exists():
            continue
        try:
            payload = json.loads(best_path_fp.read_text(encoding="utf-8"))
            path = [int(v) for v in payload.get("path") or []]
            if len(path) < 2:
                continue
            pot = Pot.read_from_disk(network_path)
            chain = _path_chain_from_pot(pot, path)
            if chain is not None:
                return chain
        except Exception:
            continue

    try:
        pot = Pot.read_from_disk(network_path)
    except Exception:
        return None

    endpoint_hints = _load_network_endpoint_hints(network_path) or {}
    start_node, end_node = _load_network_endpoint_structures(network_path)
    connectivity_hints = _match_network_endpoint_indices_by_connectivity(
        pot,
        start_node=start_node,
        end_node=end_node,
    )
    if connectivity_hints:
        endpoint_hints.update(
            {k: v for k, v in connectivity_hints.items() if v is not None}
        )
    endpoint_hints = endpoint_hints or None

    root_idx = (
        int(endpoint_hints["root_index"])
        if endpoint_hints and endpoint_hints.get("root_index") is not None
        else _find_pot_root_node_index(pot)
    )
    target_idx = _find_pot_target_node_index(
        pot,
        target_idx_hint=(
            int(endpoint_hints["target_index"])
            if endpoint_hints and endpoint_hints.get("target_index") is not None
            else None
        ),
    )

    return _load_best_path_chain_from_pot(
        pot,
        root_idx=root_idx,
        target_idx=target_idx,
    )


def _load_best_path_chain_from_network_completion(
    *,
    network_fp: Path | None,
    output_dir: Path,
    base_name: str,
) -> Chain | None:
    if network_fp is None or not Path(network_fp).exists():
        return None
    best_path_fp = output_dir / f"{base_name}_best_path.json"
    if not best_path_fp.exists():
        return None
    try:
        payload = json.loads(best_path_fp.read_text(encoding="utf-8"))
        path = [int(v) for v in payload.get("path") or []]
        if len(path) < 2:
            return None
        pot = Pot.read_from_disk(network_fp)
        return _path_chain_from_pot(pot, path)
    except Exception:
        return None


def _dedupe_minima_and_sources(
    minima: list[StructureNode],
    sources: list[StructureNode],
    chain_inputs: ChainInputs,
) -> tuple[list[StructureNode], list[StructureNode]]:
    """Deduplicate minima while preserving same-index source mapping."""
    if len(minima) != len(sources):
        raise ValueError("minima and sources must have equal lengths.")

    unique_minima: list[StructureNode] = []
    unique_sources: list[StructureNode] = []
    for node, source in zip(minima, sources):
        duplicate = False
        for existing in unique_minima:
            if is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False,
            ):
                duplicate = True
                break
        if not duplicate:
            unique_minima.append(node.copy())
            unique_sources.append(source.copy())
    return unique_minima, unique_sources


def _run_single_geometry_optimization(engine, node: StructureNode) -> list[StructureNode]:
    try:
        return engine.compute_geometry_optimization(
            node, keywords={'coordsys': 'cart', 'maxiter': 500}
        )
    except TypeError:
        return engine.compute_geometry_optimization(node)


def _summarize_refine_exception(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        text = repr(exc)

    # ChemCloud task-output schema mismatch often appears when failed tasks are
    # materialized with a ProgramOutput layout incompatible with local qcio.
    if isinstance(exc, ExternalProgramError) and "ProgramOutput schema" in text:
        return (
            f"{text} "
            "Likely a ChemCloud/qcio schema mismatch while reading a failed task output."
        )

    one_line = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return f"{type(exc).__name__}: {one_line}" if one_line else type(exc).__name__


def _source_display_text(source: Any) -> str:
    if isinstance(source, Path):
        return str(source)
    if isinstance(source, str):
        return source
    return f"<{type(source).__name__}>"


def _infer_refine_base_name(source: Any, explicit_name: str | None) -> str:
    if explicit_name is not None:
        return explicit_name

    source_path: Path | None = None
    if isinstance(source, Path):
        source_path = source
    elif isinstance(source, str):
        with contextlib.suppress(Exception):
            source_path = Path(source).expanduser().resolve()

    if source_path is None:
        inferred_name = "mep_output"
    else:
        inferred_name = source_path.stem
        for suffix in ("_parallel_msmep", "_msmep", "_neb"):
            if inferred_name.endswith(suffix):
                inferred_name = inferred_name[: -len(suffix)]
                break
        if not inferred_name:
            inferred_name = "mep_output"
    return f"{inferred_name}_refine"


def _clone_chain_with_graph_fallback(chain: Chain) -> Chain:
    cloned_nodes: list[StructureNode] = []
    for node in chain.nodes:
        node_copy = node.copy()
        if getattr(node_copy, "graph", None) is None:
            with contextlib.suppress(Exception):
                _ = node_copy.graph
        cloned_nodes.append(node_copy)
    return Chain.model_validate(
        {"nodes": cloned_nodes, "parameters": chain.parameters}
    )


def _chains_to_refinement_pot(
    chains: list[Chain],
    *,
    source_label: str,
) -> Pot:
    usable_chains = [
        _clone_chain_with_graph_fallback(chain)
        for chain in chains
        if chain is not None and len(getattr(chain, "nodes", []) or []) >= 2
    ]
    if not usable_chains:
        raise ValueError("No valid chains available to construct a refinement network.")

    chain_inputs = usable_chains[0].parameters or ChainInputs()
    node_registry: list[StructureNode] = []
    graph = nx.DiGraph()

    root_idx: int | None = None
    target_idx: int | None = None
    for chain_idx, chain in enumerate(usable_chains):
        start_node = chain.nodes[0].copy()
        end_node = chain.nodes[-1].copy()
        start_idx = _register_recursive_split_node(
            node=start_node,
            registry=node_registry,
            chain_inputs=chain_inputs,
        )
        end_idx = _register_recursive_split_node(
            node=end_node,
            registry=node_registry,
            chain_inputs=chain_inputs,
        )
        if root_idx is None:
            root_idx = int(start_idx)
        target_idx = int(end_idx)

        for node_index, node in ((start_idx, start_node), (end_idx, end_node)):
            node_attrs = graph.nodes[node_index] if graph.has_node(node_index) else {}
            molecule = getattr(node, "graph", None) or node_attrs.get("molecule") or Molecule()
            graph.add_node(
                node_index,
                molecule=molecule,
                td=node.copy(),
                converged=True,
                endpoint_optimized=True,
                generated_by="refine_source",
            )
        edge_key = (int(start_idx), int(end_idx))
        if not graph.has_edge(*edge_key):
            graph.add_edge(
                edge_key[0],
                edge_key[1],
                reaction=f"source_chain_{chain_idx}",
                list_of_nebs=[],
                generated_by="refine_source",
            )
        graph.edges[edge_key].setdefault("list_of_nebs", [])
        graph.edges[edge_key]["list_of_nebs"].append(chain.copy())

    if root_idx is None or target_idx is None:
        raise ValueError("Could not determine root/target nodes from source chains.")

    for node_index in graph.nodes:
        graph.nodes[node_index]["root"] = int(node_index) == int(root_idx)
        graph.nodes[node_index]["requested_target"] = int(node_index) == int(target_idx)

    root_molecule = graph.nodes[root_idx].get("molecule") or Molecule()
    target_molecule = graph.nodes[target_idx].get("molecule") or Molecule()
    pot = Pot(
        root=root_molecule,
        target=target_molecule,
        multiplier=1,
        rxn_name=source_label,
    )
    pot.graph = graph
    return pot


def _source_obj_to_refinement_pot(
    source_obj: TreeNode | NEB | Chain | Pot,
    *,
    source_label: str,
) -> Pot:
    if isinstance(source_obj, Pot):
        return source_obj.model_copy(deep=True)
    if isinstance(source_obj, Chain):
        return _chains_to_refinement_pot([source_obj], source_label=source_label)
    if isinstance(source_obj, NEB):
        out_chain = (
            source_obj.chain_trajectory[-1]
            if source_obj.chain_trajectory
            else source_obj.optimized
        )
        if out_chain is None:
            raise ValueError("Loaded NEB source has no optimized chain.")
        return _chains_to_refinement_pot([out_chain], source_label=source_label)
    if isinstance(source_obj, TreeNode):
        chains: list[Chain] = []
        for leaf in source_obj.ordered_leaves:
            if not leaf.data:
                continue
            chain = (
                leaf.data.chain_trajectory[-1]
                if leaf.data.chain_trajectory
                else leaf.data.optimized
            )
            if chain is not None and len(chain.nodes) >= 2:
                chains.append(chain)
        if not chains:
            chains = [source_obj.output_chain]
        return _chains_to_refinement_pot(chains, source_label=source_label)
    raise TypeError(
        "source_obj must be TreeNode, NEB, Chain, or Pot."
    )


def _next_available_directory(base_dir: Path) -> Path:
    candidate = base_dir
    suffix = 1
    while candidate.exists():
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix}")
        suffix += 1
    return candidate


def _print_ts_irc_refine_summary(summary: dict[str, Any], *, title: str) -> None:
    table = Table(box=box.ROUNDED, border_style="green", show_header=False)
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("Source Workspace", str(summary["source_workspace"]))
    table.add_row("Refined Workspace", str(summary["workspace"]))
    table.add_row("Inputs", str(summary["inputs_fp"]))
    table.add_row("Edges Scanned", str(summary["edges_scanned"]))
    table.add_row("Edges With Chains", str(summary["edges_with_chains"]))
    table.add_row("TS Attempted", str(summary["ts_guesses_attempted"]))
    table.add_row("TS Submitted", str(summary.get("ts_jobs_submitted", summary["ts_guesses_attempted"])))
    table.add_row("TS Converged", str(summary["ts_converged"]))
    table.add_row("TS Failed", str(summary.get("ts_failed", 0)))
    if summary.get("ts_top_errors"):
        table.add_row("TS Top Error", str(summary.get("ts_top_errors", [""])[0]))
    if summary.get("ts_failure_log"):
        table.add_row("TS Failure Log", str(summary["ts_failure_log"]))
    table.add_row("IRC Submitted", str(summary.get("irc_jobs_submitted", summary["ts_converged"])))
    table.add_row("IRC Converged", str(summary["irc_converged"]))
    table.add_row("IRC Failed", str(summary.get("irc_failed", 0)))
    if summary.get("irc_top_errors"):
        table.add_row("IRC Top Error", str(summary.get("irc_top_errors", [""])[0]))
    if summary.get("irc_failure_log"):
        table.add_row("IRC Failure Log", str(summary["irc_failure_log"]))
    table.add_row("ChemCloud Parallel", str(bool(summary.get("chemcloud_parallel", False))))
    table.add_row("Use BigChem", str(bool(summary.get("use_bigchem", False))))
    table.add_row("Added Nodes", str(summary["added_nodes"]))
    table.add_row("Added Edges", str(summary["added_edges"]))
    table.add_row("Final Nodes", str(summary["final_nodes"]))
    table.add_row("Final Edges", str(summary["final_edges"]))
    table.add_row("Queue Items", str(summary["queue_items"]))
    table.add_row("Artifacts", str(summary["artifacts_dir"]))
    console.print(Panel(table, title=title, border_style="green"))


def _is_chemcloud_runinputs(run_inputs: RunInputs) -> bool:
    engine = getattr(run_inputs, "engine", None)
    engine_name = str(getattr(run_inputs, "engine_name", "") or "").lower()
    compute_program = str(getattr(engine, "compute_program", "") or "").lower()
    return engine_name == "chemcloud" or compute_program == "chemcloud"


def _reoptimize_minima_for_refinement(
    minima: list[StructureNode],
    expensive_input: RunInputs,
    *,
    source_geometry_label: str,
) -> tuple[list[StructureNode], list[StructureNode], int, int]:
    """Optimize minima at expensive level, with ChemCloud batch submission when available."""
    refined_minima: list[StructureNode] = []
    refined_source_minima: list[StructureNode] = []
    dropped_count = 0
    kept_unoptimized_count = 0

    engine = expensive_input.engine
    batch_trajectories: list[list[StructureNode]] | None = None
    batch_optimizer = getattr(engine, "compute_geometry_optimizations", None)
    if _is_chemcloud_runinputs(expensive_input) and callable(batch_optimizer):
        console.print(
            "[dim]Submitting batched expensive-level geometry optimizations for minima...[/dim]"
        )
        try:
            try:
                trajectories = batch_optimizer(
                    minima, keywords={'coordsys': 'cart', 'maxiter': 500}
                )
            except TypeError:
                trajectories = batch_optimizer(minima)
            if len(trajectories) != len(minima):
                console.print(
                    "[yellow]⚠ Batched geometry optimization returned an unexpected number of results; falling back to per-minimum optimization.[/yellow]"
                )
            else:
                batch_trajectories = trajectories
        except Exception as exc:
            console.print(
                "[yellow]⚠ Batched geometry optimization submission failed; falling back to per-minimum optimization.[/yellow]"
            )
            console.print(
                f"[yellow]Reason:[/yellow] {_summarize_refine_exception(exc)}"
            )

    for i, node in enumerate(minima):
        optimized_successfully = False
        failure_reason = ""
        if batch_trajectories is not None:
            traj = batch_trajectories[i] if i < len(batch_trajectories) else []
            if traj:
                opt_node = traj[-1]
                optimized_successfully = True
            else:
                opt_node = _clear_node_cached_properties(node)
                failure_reason = "Batch optimization returned no trajectory."
                kept_unoptimized_count += 1
        else:
            try:
                traj = _run_single_geometry_optimization(engine, node)
                opt_node = traj[-1]
                optimized_successfully = True
            except Exception as exc:
                opt_node = _clear_node_cached_properties(node)
                failure_reason = _summarize_refine_exception(exc)
                kept_unoptimized_count += 1

        if not optimized_successfully:
            console.print(
                f"[yellow]⚠ Failed to optimize minimum {i}; keeping {source_geometry_label} geometry for refinement.[/yellow]"
            )
            if failure_reason:
                console.print(f"[yellow]Reason:[/yellow] {failure_reason}")

        if optimized_successfully and node.has_molecular_graph and opt_node.has_molecular_graph:
            same_connectivity = _is_connectivity_identical(
                node, opt_node, verbose=False
            )
            if not same_connectivity:
                dropped_count += 1
                continue
        refined_minima.append(opt_node)
        refined_source_minima.append(node.copy())

    return refined_minima, refined_source_minima, dropped_count, kept_unoptimized_count


def _clear_node_cached_properties(node: StructureNode) -> StructureNode:
    clean = node.copy()
    clean._cached_result = None
    clean._cached_energy = None
    clean._cached_gradient = None
    return clean


def _clear_chain_cached_properties(chain: Chain, parameters: ChainInputs) -> Chain:
    """Return chain copy with cached energies/gradients removed from every node."""
    clean_nodes = [_clear_node_cached_properties(node) for node in chain.nodes]
    return Chain.model_validate({"nodes": clean_nodes, "parameters": parameters})


def _find_matching_node_index(
    target: StructureNode,
    chain: Chain,
    chain_inputs: ChainInputs,
) -> int | None:
    try:
        dists = [np.linalg.norm(node.coords - target.coords)
                 for node in chain.nodes]
        if len(dists) == 0:
            return None
        closest_idx = int(np.argmin(dists))
        if dists[closest_idx] < 1e-8:
            return closest_idx
    except Exception:
        dists = []

    candidates: list[tuple[int, float]] = []
    for i, node in enumerate(chain.nodes):
        if is_identical(
            target,
            node,
            fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
            kcal_mol_cutoff=chain_inputs.node_ene_thre,
            verbose=False,
        ):
            dist = np.linalg.norm(node.coords - target.coords)
            candidates.append((i, dist))

    if candidates:
        return min(candidates, key=lambda t: t[1])[0]

    if len(dists) > 0:
        return int(np.argmin(dists))
    return None


def _build_recycled_pair_chain(
    cheap_output_chain: Chain,
    cheap_start_ref: StructureNode,
    cheap_end_ref: StructureNode,
    expensive_start: StructureNode,
    expensive_end: StructureNode,
    cheap_chain_inputs: ChainInputs,
    expensive_chain_inputs: ChainInputs,
    expected_nimages: int,
) -> Chain | None:
    start_idx = _find_matching_node_index(
        cheap_start_ref, cheap_output_chain, cheap_chain_inputs
    )
    end_idx = _find_matching_node_index(
        cheap_end_ref, cheap_output_chain, cheap_chain_inputs
    )
    if start_idx is None or end_idx is None or start_idx == end_idx:
        return None

    if start_idx < end_idx:
        segment_nodes = [node.copy()
                         for node in cheap_output_chain.nodes[start_idx:end_idx + 1]]
    else:
        segment_nodes = [
            node.copy() for node in cheap_output_chain.nodes[end_idx:start_idx + 1]][::-1]

    if len(segment_nodes) != expected_nimages:
        return None

    segment_nodes[0] = expensive_start.copy()
    segment_nodes[-1] = expensive_end.copy()
    recycled = Chain.model_validate(
        {"nodes": segment_nodes, "parameters": expensive_chain_inputs}
    )
    return _clear_chain_cached_properties(recycled, expensive_chain_inputs)


def _reverse_chain(chain: Chain) -> Chain:
    return Chain.model_validate(
        {"nodes": [node.copy() for node in chain.nodes[::-1]],
         "parameters": chain.parameters}
    )


def _concat_chains(chains: list[Chain], parameters: ChainInputs) -> Chain:
    if len(chains) == 0:
        raise ValueError("Cannot concatenate an empty list of chains.")
    nodes = []
    for i, chain in enumerate(chains):
        chain_nodes = chain.nodes if i == 0 else chain.nodes[1:]
        nodes.extend([node.copy() for node in chain_nodes])
    return Chain.model_validate({"nodes": nodes, "parameters": parameters})


@dataclass
class _QueuedRecursivePairRequest:
    request_id: int
    start_node: StructureNode
    end_node: StructureNode
    start_index: int
    end_index: int
    parent_request_id: int | None = None
    reason: str = ""


def _find_registered_node_index(
    node: StructureNode,
    registry: list[StructureNode],
    chain_inputs: ChainInputs,
    connectivity_only: bool = False,
) -> int | None:
    for i, existing in enumerate(registry):
        try:
            if connectivity_only:
                if _is_connectivity_identical(
                    node,
                    existing,
                    verbose=False,
                    collect_comparison=False,
                ):
                    return i
                continue
            if is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False,
            ):
                return i
        except Exception:
            if (
                list(node.structure.symbols) == list(
                    existing.structure.symbols)
                and np.allclose(node.coords, existing.coords)
            ):
                return i
    return None


def _register_recursive_split_node(
    node: StructureNode,
    registry: list[StructureNode],
    chain_inputs: ChainInputs,
    connectivity_only: bool = False,
) -> int:
    existing_index = _find_registered_node_index(
        node=node,
        registry=registry,
        chain_inputs=chain_inputs,
        connectivity_only=connectivity_only,
    )
    if existing_index is not None:
        return existing_index
    registry.append(node.copy())
    return len(registry) - 1


def _ordered_leaf_path_nodes(
    history: TreeNode, chain_inputs: ChainInputs
) -> list[StructureNode]:
    def _nodes_match_for_path(a: StructureNode, b: StructureNode) -> bool:
        try:
            if (
                list(a.structure.symbols) == list(b.structure.symbols)
                and np.allclose(a.coords, b.coords)
            ):
                return True
        except Exception:
            pass
        try:
            a_cmp = a.copy()
            b_cmp = b.copy()
            setattr(a_cmp, "disable_smiles", True)
            setattr(b_cmp, "disable_smiles", True)
            return bool(
                is_identical(
                    a_cmp,
                    b_cmp,
                    fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                    kcal_mol_cutoff=chain_inputs.node_ene_thre,
                    verbose=False,
                )
            )
        except Exception:
            return False

    path_nodes: list[StructureNode] = []
    for leaf in history.ordered_leaves:
        if not leaf.data or not leaf.data.chain_trajectory:
            continue
        final_chain = leaf.data.chain_trajectory[-1]
        if len(final_chain.nodes) == 0:
            continue
        start_node = final_chain[0].copy()
        end_node = final_chain[-1].copy()
        if not path_nodes:
            path_nodes.append(start_node)
        elif not _nodes_match_for_path(path_nodes[-1], start_node):
            path_nodes.append(start_node)
        path_nodes.append(end_node)
    if len(path_nodes) <= 1:
        return path_nodes

    pruned: list[StructureNode] = []
    for node in path_nodes:
        match_index = None
        for i, existing in enumerate(pruned):
            if _nodes_match_for_path(existing, node):
                match_index = i
                break
        if match_index is None:
            pruned.append(node)
            continue
        pruned = pruned[: match_index + 1]
    return pruned


def _queue_follow_on_recursive_requests(
    path_nodes: list[StructureNode],
    parent_request_id: int,
    next_request_id: int,
    chain_inputs: ChainInputs,
    node_registry: list[StructureNode],
    attempted_pairs: set[tuple[int, int]],
    split_mode: NetworkCompletionMode = "linear",
    root_index: int | None = None,
    target_index: int | None = None,
) -> tuple[list[_QueuedRecursivePairRequest], int]:
    queued: list[_QueuedRecursivePairRequest] = []
    if len(path_nodes) < 3:
        return queued, next_request_id

    path_indices = [
        _register_recursive_split_node(
            node=node,
            registry=node_registry,
            chain_inputs=chain_inputs,
            connectivity_only=True,
        )
        for node in path_nodes
    ]

    def _queue_pair(start_index: int, end_index: int, reason: str) -> None:
        nonlocal next_request_id
        if start_index == end_index:
            return
        pair_key = (start_index, end_index)
        if pair_key in attempted_pairs:
            return
        if (
            start_index < 0
            or end_index < 0
            or start_index >= len(node_registry)
            or end_index >= len(node_registry)
        ):
            return
        attempted_pairs.add(pair_key)
        queued.append(
            _QueuedRecursivePairRequest(
                request_id=next_request_id,
                start_node=node_registry[start_index].copy(),
                end_node=node_registry[end_index].copy(),
                start_index=start_index,
                end_index=end_index,
                parent_request_id=parent_request_id,
                reason=reason,
            )
        )
        next_request_id += 1

    if split_mode == "all-to-all":
        for start_pos in range(0, len(path_nodes) - 2):
            start_index = path_indices[start_pos]
            for end_pos in range(start_pos + 2, len(path_nodes)):
                _queue_pair(
                    start_index,
                    path_indices[end_pos],
                    "all-to-all mode: connect non-adjacent nodes found on this path",
                )
        return queued, next_request_id

    resolved_root_index = path_indices[0] if root_index is None else root_index
    resolved_target_index = path_indices[-1] if target_index is None else target_index
    for intermediate_index in path_indices[1:-1]:
        _queue_pair(
            resolved_root_index,
            intermediate_index,
            "linear mode: connect discovered intermediate to input reactant",
        )
        _queue_pair(
            intermediate_index,
            resolved_target_index,
            "linear mode: connect discovered intermediate to input product",
        )
    return queued, next_request_id


def _network_completion_node_label(
    node_index: int,
    *,
    root_index: int,
    target_index: int,
) -> str:
    if int(node_index) == int(root_index):
        return f"node {node_index} (input reactant)"
    if int(node_index) == int(target_index):
        return f"node {node_index} (input product)"
    return f"node {node_index} (intermediate)"


def _network_completion_mode_description(split_mode: NetworkCompletionMode) -> str:
    if split_mode == "all-to-all":
        return (
            "all-to-all mode queues every non-adjacent pair on each discovered path; "
            "pairs already attempted by molecular graph and stereochemical label are skipped."
        )
    return (
        "linear mode queues each discovered intermediate back to the input reactant "
        "and forward to the input product; pairs already attempted by molecular graph "
        "and stereochemical label are skipped."
    )


def _print_network_completion_queue_update(
    *,
    title: str,
    new_requests: list[_QueuedRecursivePairRequest],
    queue_depth: int,
    split_mode: NetworkCompletionMode,
    root_index: int,
    target_index: int,
    path_node_count: int,
) -> None:
    table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
    table.add_column("Request", style="cyan", no_wrap=True)
    table.add_column("Parent", no_wrap=True)
    table.add_column("Connects")
    table.add_column("Why")
    for request in new_requests[:20]:
        table.add_row(
            str(request.request_id),
            "root" if request.parent_request_id is None else str(request.parent_request_id),
            (
                f"{_network_completion_node_label(request.start_index, root_index=root_index, target_index=target_index)} "
                f"→ {_network_completion_node_label(request.end_index, root_index=root_index, target_index=target_index)}"
            ),
            request.reason or _network_completion_mode_description(split_mode),
        )
    if len(new_requests) > 20:
        table.add_row(
            "...",
            "...",
            f"{len(new_requests) - 20} additional queued request(s)",
            "omitted from display",
        )

    summary = (
        f"{_network_completion_mode_description(split_mode)}\n"
        f"Path nodes considered: {path_node_count}. "
        f"New requests queued: {len(new_requests)}. "
        f"Queue depth after update: {queue_depth}."
    )
    console.print(Panel(summary, title=title, border_style="cyan"))
    if new_requests:
        console.print(table)
    else:
        console.print("[dim]No new network-completion requests were queued from this path.[/dim]")


def _print_network_completion_request_start(
    *,
    request: _QueuedRecursivePairRequest,
    queue_depth: int,
    root_index: int,
    target_index: int,
) -> None:
    console.print(
        "[cyan]Network completion request "
        f"{request.request_id}: "
        f"{_network_completion_node_label(request.start_index, root_index=root_index, target_index=target_index)} "
        "→ "
        f"{_network_completion_node_label(request.end_index, root_index=root_index, target_index=target_index)}"
        f" ({request.reason}; queued remaining: {queue_depth})[/cyan]"
    )


def _print_network_completion_final_summary(
    *,
    request_records: list[dict],
    node_registry: list[StructureNode],
    attempted_pairs: set[tuple[int, int]],
    split_mode: NetworkCompletionMode,
    followup_grad_calls_total: int,
    followup_geom_grad_calls: int,
) -> None:
    counts = _request_record_summary(request_records)
    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("Mode", split_mode)
    table.add_row("Why", _network_completion_mode_description(split_mode))
    table.add_row("Total requests", str(len(request_records)))
    table.add_row("Completed", str(counts.get("completed", 0)))
    table.add_row("Failed", str(counts.get("failed", 0)))
    table.add_row("Empty", str(counts.get("empty", 0)))
    table.add_row("Unique graph/stereo endpoints", str(len(node_registry)))
    table.add_row("Attempted graph/stereo pairs", str(len(attempted_pairs)))
    table.add_row("Follow-up gradient calls", str(int(followup_grad_calls_total)))
    table.add_row("Follow-up geom-opt gradient calls", str(int(followup_geom_grad_calls)))
    console.print(Panel(table, title="Network Split Summary", border_style="cyan"))


def _mark_path_pairs_attempted(
    path_nodes: list[StructureNode],
    *,
    chain_inputs: ChainInputs,
    node_registry: list[StructureNode],
    attempted_pairs: set[tuple[int, int]],
) -> None:
    if len(path_nodes) < 2:
        return
    path_indices = [
        _register_recursive_split_node(
            node=node,
            registry=node_registry,
            chain_inputs=chain_inputs,
            connectivity_only=True,
        )
        for node in path_nodes
    ]
    for start_index, end_index in zip(path_indices[:-1], path_indices[1:]):
        attempted_pairs.add((start_index, end_index))


def _build_attempted_pair_skip_payload(
    *,
    attempted_pairs: set[tuple[int, int]],
    node_registry: list[StructureNode],
    directed: bool,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    excluded = exclude_pairs or set()
    for start_index, end_index in sorted(attempted_pairs):
        if (start_index, end_index) in excluded:
            continue
        if (
            start_index < 0
            or end_index < 0
            or start_index >= len(node_registry)
            or end_index >= len(node_registry)
        ):
            continue
        start_node = node_registry[start_index]
        end_node = node_registry[end_index]
        if not hasattr(start_node, "to_serializable") or not hasattr(
            end_node, "to_serializable"
        ):
            continue
        try:
            start_payload = start_node.to_serializable()
            end_payload = end_node.to_serializable()
        except Exception:
            continue
        payload.append(
            {
                "start": start_payload,
                "end": end_payload,
                "directed": bool(directed),
            }
        )
    return payload


def _grad_calls_from_history(history: Any) -> tuple[int, int]:
    getter = getattr(history, "get_optimization_history", None)
    if not callable(getter):
        return 0, 0
    try:
        objects = [obj for obj in getter() if obj]
    except Exception:
        return 0, 0
    total = sum(int(getattr(obj, "grad_calls_made", 0)) for obj in objects)
    geom = sum(int(getattr(obj, "geom_grad_calls_made", 0)) for obj in objects)
    return total, geom


def _load_recursive_split_manifest_costs(manifest_fp: Path) -> dict[int, tuple[int, int]]:
    costs: dict[int, tuple[int, int]] = {}
    if not Path(manifest_fp).is_file():
        return costs
    try:
        manifest = json.loads(Path(manifest_fp).read_text())
    except Exception:
        return costs
    for record in manifest.get("requests", []):
        try:
            request_id = int(record["request_id"])
        except Exception:
            continue
        total = int(record.get("gradient_calls_total", 0) or 0)
        geom = int(record.get("gradient_calls_geometry_optimizations", 0) or 0)
        costs[request_id] = (total, geom)
    return costs


def _estimate_elapsed_seconds_from_paths(paths: list[Path]) -> float | None:
    timestamps: list[float] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            with contextlib.suppress(Exception):
                timestamps.append(float(path.stat().st_mtime))
            continue
        if path.is_dir():
            with contextlib.suppress(Exception):
                timestamps.append(float(path.stat().st_mtime))
            for child in path.rglob("*"):
                if not child.is_file():
                    continue
                with contextlib.suppress(Exception):
                    timestamps.append(float(child.stat().st_mtime))
    if len(timestamps) < 2:
        return None
    elapsed = max(timestamps) - min(timestamps)
    return float(max(0.0, elapsed))


def _estimate_grad_calls_from_history_dir(history_dir: Path) -> int:
    total = 0
    traj_files = sorted(history_dir.glob("traj_*.xyz"))
    for traj_fp in traj_files:
        chain: Chain | None = None
        with contextlib.suppress(Exception):
            chain = Chain.from_xyz(
                traj_fp,
                parameters=ChainInputs(),
                charge=None,
                spinmult=None,
            )
        if chain is None:
            with contextlib.suppress(Exception):
                chain = Chain.from_xyz(
                    traj_fp,
                    parameters=ChainInputs(),
                    charge=0,
                    spinmult=1,
                )
        if chain is not None:
            # Backfill assumption: endpoints remain frozen and no extra frozen
            # interior nodes are active, so each NEB step evaluates N-2 beads.
            total += int(max(0, len(chain.nodes) - 2))
    return int(total)


def _estimate_grad_calls_from_tree_folder(tree_dir: Path) -> int:
    total = 0
    for history_dir in sorted(tree_dir.glob("node_*_history")):
        if history_dir.is_dir():
            total += _estimate_grad_calls_from_history_dir(history_dir)
    return int(total)


def _estimate_grad_calls_from_network_completion_dir(
    network_dir: Path, *, include_request0: bool = True
) -> int:
    total = 0
    for tree_dir in sorted(network_dir.glob("request_*_msmep")):
        if tree_dir.is_dir():
            if not include_request0 and tree_dir.name == "request_0_msmep":
                continue
            total += _estimate_grad_calls_from_tree_folder(tree_dir)
    return int(total)


def _try_load_status_snapshot(path: Path) -> dict[str, Any] | None:
    with contextlib.suppress(Exception):
        return _load_status_snapshot(str(path))
    return None


def _is_neb_history_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not path.name.endswith("_history"):
        return False
    return any(path.glob("traj_*.xyz"))


def _is_msmep_tree_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not path.name.endswith("_msmep"):
        return False
    return any(path.glob("node_*_history"))


def _infer_refine_base_name_from_path(path: Path) -> str | None:
    stem = path.stem
    for suffix in ("_refined", "_refined_minima", "_cheap"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def _estimate_natoms_from_xyz(path: Path) -> int | None:
    chain: Chain | None = None
    with contextlib.suppress(Exception):
        chain = Chain.from_xyz(
            path,
            parameters=ChainInputs(),
            charge=None,
            spinmult=None,
        )
    if chain is None:
        with contextlib.suppress(Exception):
            chain = Chain.from_xyz(
                path,
                parameters=ChainInputs(),
                charge=0,
                spinmult=1,
            )
    if chain is None or len(chain.nodes) == 0:
        return None
    return int(len(chain.nodes[0].coords))


def _estimate_ts_steps_from_qcio(path: Path) -> int | None:
    with contextlib.suppress(Exception):
        output = ProgramOutput.open(path)
        data = getattr(output, "data", None)
        trajectory = getattr(data, "trajectory", None)
        if isinstance(trajectory, list):
            return int(len(trajectory))
    return None


def _estimate_natoms_from_qcio(path: Path) -> int | None:
    with contextlib.suppress(Exception):
        output = ProgramOutput.open(path)
        input_data = getattr(output, "input_data", None)
        structure = getattr(input_data, "structure", None)
        symbols = getattr(structure, "symbols", None)
        if symbols is not None:
            return int(len(symbols))
    return None


def _estimate_ts_irc_grad_calls_from_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    job_map: dict[str, dict[str, Path]] = {}

    for ts_xyz in artifacts_dir.glob("*.ts.xyz"):
        stem = ts_xyz.name[: -len(".ts.xyz")]
        job_map.setdefault(stem, {})["ts_xyz"] = ts_xyz
    for ts_qcio in artifacts_dir.glob("*.ts.qcio"):
        stem = ts_qcio.name[: -len(".ts.qcio")]
        job_map.setdefault(stem, {})["ts_qcio"] = ts_qcio
    for irc_xyz in artifacts_dir.glob("*.irc.xyz"):
        stem = irc_xyz.name[: -len(".irc.xyz")]
        job_map.setdefault(stem, {})["irc_xyz"] = irc_xyz

    ts_jobs_detected = 0
    irc_jobs_detected = 0
    ts_hessian_calls = 0
    ts_optimization_steps = 0
    irc_hessian_calls = 0
    irc_node_grad_calls = 0
    unknown_natoms_ts_jobs = 0
    unknown_natoms_irc_jobs = 0
    unknown_ts_step_jobs = 0
    unknown_irc_node_jobs = 0

    for _stem, files in sorted(job_map.items()):
        has_ts = "ts_xyz" in files or "ts_qcio" in files
        has_irc = "irc_xyz" in files

        natoms: int | None = None
        if "irc_xyz" in files:
            natoms = _estimate_natoms_from_xyz(files["irc_xyz"])
        if natoms is None and "ts_xyz" in files:
            natoms = _estimate_natoms_from_xyz(files["ts_xyz"])
        if natoms is None and "ts_qcio" in files:
            natoms = _estimate_natoms_from_qcio(files["ts_qcio"])

        hessian_calls = None if natoms is None else int(max(0, 6 * natoms))

        if has_ts:
            ts_jobs_detected += 1
            if hessian_calls is None:
                unknown_natoms_ts_jobs += 1
            else:
                ts_hessian_calls += hessian_calls

            ts_steps = None
            if "ts_qcio" in files:
                ts_steps = _estimate_ts_steps_from_qcio(files["ts_qcio"])
            if ts_steps is None:
                unknown_ts_step_jobs += 1
            else:
                ts_optimization_steps += int(max(0, ts_steps))

        if has_irc:
            irc_jobs_detected += 1
            if hessian_calls is None:
                unknown_natoms_irc_jobs += 1
            else:
                irc_hessian_calls += hessian_calls

            irc_nodes: int | None = None
            if "irc_xyz" in files:
                chain: Chain | None = None
                with contextlib.suppress(Exception):
                    chain = Chain.from_xyz(
                        files["irc_xyz"],
                        parameters=ChainInputs(),
                        charge=None,
                        spinmult=None,
                    )
                if chain is None:
                    with contextlib.suppress(Exception):
                        chain = Chain.from_xyz(
                            files["irc_xyz"],
                            parameters=ChainInputs(),
                            charge=0,
                            spinmult=1,
                        )
                if chain is not None:
                    irc_nodes = int(len(chain.nodes))
            if irc_nodes is None:
                unknown_irc_node_jobs += 1
            else:
                irc_node_grad_calls += int(max(0, irc_nodes))

    gradient_calls_total = int(
        ts_hessian_calls
        + ts_optimization_steps
        + irc_hessian_calls
        + irc_node_grad_calls
    )
    return {
        "gradient_calls_total": gradient_calls_total,
        "ts_jobs_detected": int(ts_jobs_detected),
        "irc_jobs_detected": int(irc_jobs_detected),
        "ts_hessian_calls_estimated": int(ts_hessian_calls),
        "ts_optimization_steps_estimated": int(ts_optimization_steps),
        "irc_hessian_calls_estimated": int(irc_hessian_calls),
        "irc_node_grad_calls_estimated": int(irc_node_grad_calls),
        "unknown_natoms_ts_jobs": int(unknown_natoms_ts_jobs),
        "unknown_natoms_irc_jobs": int(unknown_natoms_irc_jobs),
        "unknown_ts_step_jobs": int(unknown_ts_step_jobs),
        "unknown_irc_node_jobs": int(unknown_irc_node_jobs),
        "gradient_calls_assumption": (
            "TS: (6N) Hessian + 1 grad per optimization step; "
            "IRC: (6N) Hessian + 1 grad per IRC node"
        ),
    }


def _set_runner_attempted_pairs_payload(
    runner: Any,
    attempted_pairs_payload: list[dict[str, Any]],
) -> None:
    with contextlib.suppress(Exception):
        setattr(runner.inputs.path_min_inputs, "attempted_pairs_payload", attempted_pairs_payload)


def _write_recursive_split_request_artifacts(
    output_dir: Path,
    request_id: int,
    history: TreeNode,
    write_qcio: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_chain = history.output_chain

    output_chain.write_to_disk(
        output_dir / f"request_{request_id}.xyz", write_qcio=write_qcio)
    history.write_to_disk(
        output_dir / f"request_{request_id}_msmep", write_qcio=write_qcio)


def _load_recursive_split_request_history(
    output_dir: Path,
    request_id: int,
    *,
    chain_inputs: ChainInputs,
    engine,
    charge: int,
    multiplicity: int,
) -> TreeNode | None:
    request_dir = Path(output_dir) / f"request_{request_id}_msmep"
    if not request_dir.is_dir():
        return None
    try:
        return TreeNode.read_from_disk(
            request_dir,
            chain_parameters=chain_inputs,
            engine=engine,
            charge=charge,
            multiplicity=multiplicity,
        )
    except Exception:
        return None


def _maybe_resume_recursive_history(
    status_fp: Path,
    *,
    chain_inputs: ChainInputs,
    engine,
    charge: int,
    multiplicity: int,
) -> TreeNode | None:
    if not Path(status_fp).exists():
        return None
    try:
        snapshot = _load_status_snapshot(str(status_fp))
        run_status = snapshot.get("run_status") or {}
        if str(run_status.get("phase") or "") not in {"network_completion", "complete"}:
            return None
        tree_path = run_status.get("tree_path")
        if not tree_path:
            return None
        tree_dir = Path(tree_path)
        if not tree_dir.exists():
            return None
        return TreeNode.read_from_disk(
            tree_dir,
            chain_parameters=chain_inputs,
            engine=engine,
            charge=charge,
            multiplicity=multiplicity,
        )
    except Exception:
        return None


def _build_recursive_split_network_summary(
    output_dir: Path,
    base_name: str,
    chain_inputs: ChainInputs,
    root_index: int | None = None,
    target_index: int | None = None,
    root_node: StructureNode | None = None,
    target_node: StructureNode | None = None,
    source_tree_dir: Path | None = None,
    verbose: bool = False,
) -> Path | None:
    nb = NetworkBuilder(
        data_dir=output_dir,
        start=None,
        end=None,
        network_inputs=NetworkInputs(verbose=verbose),
        chain_inputs=chain_inputs,
    )
    nb.msmep_data_dir = output_dir

    msmep_paths = [p for p in output_dir.glob("*_msmep") if p.is_dir()]
    if source_tree_dir is None:
        sibling_source = output_dir.parent / base_name
        if sibling_source.is_dir():
            source_tree_dir = sibling_source
    if source_tree_dir is not None and source_tree_dir.is_dir():
        resolved_source = source_tree_dir.resolve()
        if all(p.resolve() != resolved_source for p in msmep_paths):
            msmep_paths.insert(0, source_tree_dir)
    if not msmep_paths:
        return None

    try:
        create_from_paths = getattr(nb, "create_rxn_network_from_paths", None)
        if callable(create_from_paths):
            pot = create_from_paths(msmep_paths)
        else:
            pot = nb.create_rxn_network(file_pattern="*_msmep")
    except (ValueError, IndexError) as exc:
        console.print(f"[yellow]⚠ Skipping network summary: {exc}[/yellow]")
        return None

    matched_indices = _match_network_endpoint_indices_by_connectivity(
        pot,
        start_node=root_node,
        end_node=target_node,
    )
    resolved_root_index = (
        matched_indices.get("root_index")
        if matched_indices and matched_indices.get("root_index") is not None
        else root_index
    )
    resolved_target_index = (
        matched_indices.get("target_index")
        if matched_indices and matched_indices.get("target_index") is not None
        else target_index
    )
    for node_idx in pot.graph.nodes:
        pot.graph.nodes[node_idx]["root"] = int(node_idx) == int(
            resolved_root_index) if resolved_root_index is not None else bool(pot.graph.nodes[node_idx].get("root"))
        pot.graph.nodes[node_idx]["requested_target"] = int(node_idx) == int(
            resolved_target_index) if resolved_target_index is not None else bool(pot.graph.nodes[node_idx].get("requested_target"))
    pot_fp = output_dir / f"{base_name}_network.json"
    pot.write_to_disk(pot_fp)

    try:
        plot_results_from_pot_obj(
            fp_out=(output_dir / f"{base_name}_network.png"),
            pot=pot,
            include_pngs=True,
        )
        plot_results_from_pot_obj(
            fp_out=(output_dir / f"{base_name}_network.png"),
            pot=pot,
            include_pngs=False,
        )
    except Exception:
        console.print(
            "[yellow]⚠ Failed to generate network plots. Continuing with JSON only.[/yellow]"
        )

    try:
        nodes = [pot.graph.nodes[x]["td"] for x in pot.graph.nodes]
        chain = Chain.model_validate({"nodes": nodes})
        chain.write_to_disk(output_dir / f"{base_name}_network_nodes.xyz")
    except Exception:
        console.print(
            "[yellow]⚠ Failed to export network node geometries. Continuing.[/yellow]"
        )

    try:
        best_path_nodes, _ = _best_path_by_apparent_barrier(
            pot,
            root_idx=resolved_root_index,
            target_idx=resolved_target_index,
        ) if resolved_root_index is not None and resolved_target_index is not None else (None, None)
        if best_path_nodes:
            best_path_chain = _path_chain_from_pot(pot, best_path_nodes)
            if best_path_chain is not None:
                _write_chain_with_nan_fallback(
                    best_path_chain,
                    output_dir / f"{base_name}_best_path.xyz",
                )
                _write_json_atomic(
                    output_dir / f"{base_name}_best_path.json",
                    {
                        "root_index": int(resolved_root_index),
                        "target_index": int(resolved_target_index),
                        "path": [int(v) for v in best_path_nodes],
                    },
                )
    except Exception:
        console.print(
            "[yellow]⚠ Failed to export best network path chain. Continuing.[/yellow]"
        )

    return pot_fp


def _run_recursive_network_completion(
    history: TreeNode,
    program_input: RunInputs,
    initial_start: StructureNode,
    initial_end: StructureNode,
    output_dir: Path,
    base_name: str,
    status_fp: Path | None = None,
    source_tree_dir: Path | None = None,
    parallel_recursive: bool = False,
    parallel_workers: int | None = None,
    overwrite: bool = False,
    overwrite_followups_only: bool = False,
    split_mode: NetworkCompletionMode = "linear",
) -> tuple[list[dict], Path | None, Path, dict[str, int]]:
    if split_mode not in {"linear", "all-to-all"}:
        raise ValueError(
            "split_mode must be either 'linear' or 'all-to-all'."
        )
    manifest_fp = _recursive_split_manifest_path(
        output_dir=output_dir, base_name=base_name)
    resume_mode = output_dir.exists() and (
        manifest_fp.exists() or (output_dir / "request_0_msmep").is_dir()
    )
    if overwrite and output_dir.exists():
        request0_dir = output_dir / "request_0_msmep"
        if overwrite_followups_only and request0_dir.is_dir():
            console.print(
                f"[yellow]⚠ Overwrite enabled: clearing existing follow-up network-completion artifacts under {output_dir} while preserving request_0_msmep.[/yellow]"
            )
            for child in output_dir.iterdir():
                if child.name.startswith("request_0"):
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    with contextlib.suppress(Exception):
                        child.unlink()
            resume_mode = True
        else:
            console.print(
                f"[yellow]⚠ Overwrite enabled: deleting existing network-completion output at {output_dir}[/yellow]"
            )
            shutil.rmtree(output_dir)
            resume_mode = False
    elif output_dir.exists() and not resume_mode:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_mode:
        console.print(
            f"[bold cyan]Resuming network completion from {output_dir}[/bold cyan]"
        )

    attempted_pairs: set[tuple[int, int]] = set()
    node_registry: list[StructureNode] = []
    request_records: list[dict] = []
    existing_manifest_costs = _load_recursive_split_manifest_costs(manifest_fp)
    followup_grad_calls_total = 0
    followup_geom_grad_calls = 0
    network_fp: Path | None = None
    write_qcio = bool(getattr(program_input, "write_qcio", False))
    charge = int(initial_start.structure.charge)
    multiplicity = int(initial_start.structure.multiplicity)

    start_index = _register_recursive_split_node(
        node=initial_start,
        registry=node_registry,
        chain_inputs=program_input.chain_inputs,
        connectivity_only=True,
    )
    end_index = _register_recursive_split_node(
        node=initial_end,
        registry=node_registry,
        chain_inputs=program_input.chain_inputs,
        connectivity_only=True,
    )
    attempted_pairs.add((start_index, end_index))

    initial_history = (
        _load_recursive_split_request_history(
            output_dir,
            0,
            chain_inputs=program_input.chain_inputs,
            engine=program_input.engine,
            charge=charge,
            multiplicity=multiplicity,
        )
        if resume_mode
        else None
    ) or history
    if not (output_dir / "request_0_msmep").is_dir():
        _write_recursive_split_request_artifacts(
            output_dir=output_dir,
            request_id=0,
            history=initial_history,
            write_qcio=write_qcio,
        )
    initial_path_nodes = _ordered_leaf_path_nodes(
        history=initial_history, chain_inputs=program_input.chain_inputs
    )
    _mark_path_pairs_attempted(
        initial_path_nodes,
        chain_inputs=program_input.chain_inputs,
        node_registry=node_registry,
        attempted_pairs=attempted_pairs,
    )
    _upsert_request_record(
        request_records,
        _create_recursive_request_record(
            request_id=0,
            parent_request_id=None,
            start_index=start_index,
            end_index=end_index,
            status="completed",
            n_path_nodes=len(initial_path_nodes),
            completed_at=datetime.now().isoformat(),
        ),
    )
    network_fp = _build_recursive_split_network_summary(
        output_dir=output_dir,
        base_name=base_name,
        chain_inputs=program_input.chain_inputs,
        root_index=start_index,
        target_index=end_index,
        root_node=initial_start,
        target_node=initial_end,
        source_tree_dir=source_tree_dir,
    )
    manifest_fp = _write_recursive_split_manifest(
        output_dir=output_dir,
        base_name=base_name,
        request_records=request_records,
        run_state="running",
        current_request_id=None,
        network_fp=network_fp,
    )
    if status_fp is not None:
        _write_run_status(
            status_fp,
            base_name=base_name,
            run_state="running",
            phase="network_completion",
            network_completion_dir=output_dir,
            manifest_fp=manifest_fp,
            network_fp=network_fp,
        )

    queue, next_request_id = _queue_follow_on_recursive_requests(
        path_nodes=initial_path_nodes,
        parent_request_id=0,
        next_request_id=1,
        chain_inputs=program_input.chain_inputs,
        node_registry=node_registry,
        attempted_pairs=attempted_pairs,
        split_mode=split_mode,
        root_index=start_index,
        target_index=end_index,
    )
    _print_network_completion_queue_update(
        title="Network Split Queue: initial path",
        new_requests=queue,
        queue_depth=len(queue),
        split_mode=split_mode,
        root_index=start_index,
        target_index=end_index,
        path_node_count=len(initial_path_nodes),
    )
    for request in queue:
        _upsert_request_record(
            request_records,
            _create_recursive_request_record(
                request_id=request.request_id,
                parent_request_id=request.parent_request_id,
                start_index=request.start_index,
                end_index=request.end_index,
                status="queued",
                reason=request.reason,
                queued_at=datetime.now().isoformat(),
            ),
        )
    manifest_fp = _write_recursive_split_manifest(
        output_dir=output_dir,
        base_name=base_name,
        request_records=request_records,
        run_state="running",
        current_request_id=None,
        network_fp=network_fp,
    )
    if status_fp is not None:
        _write_run_status(
            status_fp,
            base_name=base_name,
            run_state="running",
            phase="network_completion",
            network_completion_dir=output_dir,
            manifest_fp=manifest_fp,
            network_fp=network_fp,
        )

    msmep_runner = MSMEP(inputs=program_input)
    parallel_recursive = bool(parallel_recursive)
    resolved_parallel_workers = (
        None if parallel_workers is None else max(1, int(parallel_workers))
    )
    attempted_pair_directed = bool(
        getattr(program_input.path_min_inputs, "attempted_pair_skip_directed", False)
    )
    while queue:
        request = queue.pop(0)
        _print_network_completion_request_start(
            request=request,
            queue_depth=len(queue),
            root_index=start_index,
            target_index=end_index,
        )
        existing_history = _load_recursive_split_request_history(
            output_dir,
            request.request_id,
            chain_inputs=program_input.chain_inputs,
            engine=program_input.engine,
            charge=charge,
            multiplicity=multiplicity,
        )
        if existing_history is None:
            _upsert_request_record(
                request_records,
                _create_recursive_request_record(
                    request_id=request.request_id,
                    parent_request_id=request.parent_request_id,
                    start_index=request.start_index,
                    end_index=request.end_index,
                    status="running",
                    reason=request.reason,
                    started_at=datetime.now().isoformat(),
                ),
            )
            manifest_fp = _write_recursive_split_manifest(
                output_dir=output_dir,
                base_name=base_name,
                request_records=request_records,
                run_state="running",
                current_request_id=request.request_id,
                network_fp=network_fp,
            )
            if status_fp is not None:
                _write_run_status(
                    status_fp,
                    base_name=base_name,
                    run_state="running",
                    phase="network_completion",
                    network_completion_dir=output_dir,
                    manifest_fp=manifest_fp,
                    network_fp=network_fp,
                )
        request_chain = Chain.model_validate(
            {
                "nodes": [request.start_node.copy(), request.end_node.copy()],
                "parameters": program_input.chain_inputs,
            }
        )
        if existing_history is None:
            attempted_payload = _build_attempted_pair_skip_payload(
                attempted_pairs=attempted_pairs,
                node_registry=node_registry,
                directed=attempted_pair_directed,
                exclude_pairs={(request.start_index, request.end_index)},
            )
            _set_runner_attempted_pairs_payload(
                msmep_runner, attempted_pairs_payload=attempted_payload
            )
            try:
                if parallel_recursive:
                    try:
                        request_history = msmep_runner.run_parallel_recursive_minimize(
                            request_chain,
                            max_workers=resolved_parallel_workers,
                            attempted_pairs_payload=attempted_payload,
                        )
                    except TypeError as exc:
                        if "attempted_pairs_payload" not in str(exc):
                            raise
                        request_history = msmep_runner.run_parallel_recursive_minimize(
                            request_chain,
                            max_workers=resolved_parallel_workers,
                        )
                else:
                    try:
                        request_history = msmep_runner.run_recursive_minimize(
                            request_chain,
                            attempted_pairs_payload=attempted_payload,
                        )
                    except TypeError as exc:
                        if "attempted_pairs_payload" not in str(exc):
                            raise
                        request_history = msmep_runner.run_recursive_minimize(
                            request_chain
                        )
            except Exception:
                _upsert_request_record(
                    request_records,
                    _create_recursive_request_record(
                        request_id=request.request_id,
                        parent_request_id=request.parent_request_id,
                        start_index=request.start_index,
                        end_index=request.end_index,
                        status="failed",
                        reason=request.reason,
                        completed_at=datetime.now().isoformat(),
                        error=traceback.format_exc().strip(),
                    ),
                )
                manifest_fp = _write_recursive_split_manifest(
                    output_dir=output_dir,
                    base_name=base_name,
                    request_records=request_records,
                    run_state="running",
                    current_request_id=None,
                    network_fp=network_fp,
                )
                continue
        else:
            request_history = existing_history

        if not request_history.data:
            _upsert_request_record(
                request_records,
                _create_recursive_request_record(
                    request_id=request.request_id,
                    parent_request_id=request.parent_request_id,
                    start_index=request.start_index,
                    end_index=request.end_index,
                    status="empty",
                    reason=request.reason,
                    completed_at=datetime.now().isoformat(),
                ),
            )
            manifest_fp = _write_recursive_split_manifest(
                output_dir=output_dir,
                base_name=base_name,
                request_records=request_records,
                run_state="running",
                current_request_id=None,
                network_fp=network_fp,
            )
            continue

        if existing_history is None:
            _write_recursive_split_request_artifacts(
                output_dir=output_dir,
                request_id=request.request_id,
                history=request_history,
                write_qcio=write_qcio,
            )
        if existing_history is None:
            request_grad_total, request_grad_geom = _grad_calls_from_history(
                request_history
            )
        else:
            request_grad_total, request_grad_geom = existing_manifest_costs.get(
                int(request.request_id),
                (
                    _estimate_grad_calls_from_tree_folder(
                        output_dir / f"request_{request.request_id}_msmep"
                    ),
                    0,
                ),
            )
        followup_grad_calls_total += int(request_grad_total)
        followup_geom_grad_calls += int(request_grad_geom)
        request_path_nodes = _ordered_leaf_path_nodes(
            history=request_history, chain_inputs=program_input.chain_inputs
        )
        _mark_path_pairs_attempted(
            request_path_nodes,
            chain_inputs=program_input.chain_inputs,
            node_registry=node_registry,
            attempted_pairs=attempted_pairs,
        )
        _upsert_request_record(
            request_records,
            _create_recursive_request_record(
                request_id=request.request_id,
                parent_request_id=request.parent_request_id,
                start_index=request.start_index,
                end_index=request.end_index,
                status="completed",
                reason=request.reason,
                n_path_nodes=len(request_path_nodes),
                gradient_calls_total=int(request_grad_total),
                gradient_calls_geometry_optimizations=int(request_grad_geom),
                completed_at=datetime.now().isoformat(),
            ),
        )
        new_requests, next_request_id = _queue_follow_on_recursive_requests(
            path_nodes=request_path_nodes,
            parent_request_id=request.request_id,
            next_request_id=next_request_id,
            chain_inputs=program_input.chain_inputs,
            node_registry=node_registry,
            attempted_pairs=attempted_pairs,
            split_mode=split_mode,
            root_index=start_index,
            target_index=end_index,
        )
        _print_network_completion_queue_update(
            title=f"Network Split Queue: after request {request.request_id}",
            new_requests=new_requests,
            queue_depth=len(queue) + len(new_requests),
            split_mode=split_mode,
            root_index=start_index,
            target_index=end_index,
            path_node_count=len(request_path_nodes),
        )
        for new_request in new_requests:
            _upsert_request_record(
                request_records,
                _create_recursive_request_record(
                    request_id=new_request.request_id,
                    parent_request_id=new_request.parent_request_id,
                    start_index=new_request.start_index,
                    end_index=new_request.end_index,
                    status="queued",
                    reason=new_request.reason,
                    queued_at=datetime.now().isoformat(),
                ),
            )
        queue.extend(new_requests)
        network_fp = _build_recursive_split_network_summary(
            output_dir=output_dir,
            base_name=base_name,
            chain_inputs=program_input.chain_inputs,
            root_index=start_index,
            target_index=end_index,
            root_node=initial_start,
            target_node=initial_end,
            source_tree_dir=source_tree_dir,
        )
        manifest_fp = _write_recursive_split_manifest(
            output_dir=output_dir,
            base_name=base_name,
            request_records=request_records,
            run_state="running",
            current_request_id=None,
            network_fp=network_fp,
        )
        if status_fp is not None:
            _write_run_status(
                status_fp,
                base_name=base_name,
                run_state="running",
                phase="network_completion",
                network_completion_dir=output_dir,
                manifest_fp=manifest_fp,
                network_fp=network_fp,
            )

    network_fp = _build_recursive_split_network_summary(
        output_dir=output_dir,
        base_name=base_name,
        chain_inputs=program_input.chain_inputs,
        root_index=start_index,
        target_index=end_index,
        root_node=initial_start,
        target_node=initial_end,
        source_tree_dir=source_tree_dir,
    )
    manifest_fp = _write_recursive_split_manifest(
        output_dir=output_dir,
        base_name=base_name,
        request_records=request_records,
        run_state="completed",
        current_request_id=None,
        network_fp=network_fp,
    )
    if status_fp is not None:
        _write_run_status(
            status_fp,
            base_name=base_name,
            run_state="running",
            phase="network_completion",
            network_completion_dir=output_dir,
            manifest_fp=manifest_fp,
            network_fp=network_fp,
        )
    cost_summary = {
        "gradient_calls_total": int(followup_grad_calls_total),
        "gradient_calls_geometry_optimizations": int(followup_geom_grad_calls),
    }
    _print_network_completion_final_summary(
        request_records=request_records,
        node_registry=node_registry,
        attempted_pairs=attempted_pairs,
        split_mode=split_mode,
        followup_grad_calls_total=followup_grad_calls_total,
        followup_geom_grad_calls=followup_geom_grad_calls,
    )
    return request_records, network_fp, manifest_fp, cost_summary


def _section_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"value": obj}


def _flatten_params(data, prefix=""):
    if isinstance(data, dict):
        rows = []
        for key in sorted(data.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_params(data[key], next_prefix))
        return rows
    return [(prefix if prefix else "value", _format_param_value(data))]


def _format_param_value(value, max_str: int = 160):
    if isinstance(value, list):
        if len(value) > 20:
            head = ", ".join(str(v) for v in value[:8])
            tail = ", ".join(str(v) for v in value[-3:])
            return f"[{head}, ..., {tail}] (n={len(value)})"
        return str(value)
    s = str(value)
    if len(s) > max_str:
        return s[: max_str - 3] + "..."
    return s


def _render_runinputs(program_input: RunInputs):
    table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
    table.add_column("Section", style="bold cyan", no_wrap=True)
    table.add_column("Key", style="magenta")
    table.add_column("Value", style="white")

    path_min_inputs = _section_dict(program_input.path_min_inputs)
    # Hide legacy adaptive-trigger keys from the run summary table.
    for legacy_key in (
        "adaptive_segment_ratio",
        "adaptive_energy_ratio",
        "adaptive_use_energy",
    ):
        path_min_inputs.pop(legacy_key, None)

    sections = [
        ("General", {
            "engine_name": program_input.engine_name,
            "program": program_input.program,
            "path_min_method": program_input.path_min_method,
        }),
        ("ASE Engine", _section_dict(program_input.ase_engine_kwds)
         if str(getattr(program_input, "engine_name", "")).lower() == "ase" else {}),
        ("Path Minimizer", path_min_inputs),
        ("Chain", _section_dict(program_input.chain_inputs)),
        ("GI", _section_dict(program_input.gi_inputs)),
        ("Program Args", _section_dict(program_input.program_kwds)),
        ("Optimizer", _section_dict(program_input.optimizer_kwds)),
    ]

    for section_name, section_data in sections:
        flat_rows = _flatten_params(section_data)
        for i, (key, value) in enumerate(flat_rows):
            table.add_row(section_name if i == 0 else "", key, value)

    console.print(
        Panel(
            table,
            title="[bold cyan]Input Parameters[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def _load_endpoint_structure(
    path: str,
    charge: int,
    multiplicity: int,
) -> Structure:
    return Structure.open(path)


def _snap_assign_endpoint_nodes(nodes: list[StructureNode]) -> list[StructureNode]:
    if len(nodes) != 2:
        console.print(
            "[yellow]⚠ --snap-assign only applies to two endpoint structures; "
            "skipping snap assignment for multi-frame geometry input.[/yellow]"
        )
        return nodes

    start_node, end_node = nodes
    console.print(
        Panel(
            "[bold yellow]--snap-assign should ONLY be used for reactions where the start and end "
            "structures are supposed to be identical.[/bold yellow]\n"
            "[yellow]It will likely fail or produce invalid mappings when autosplitting across "
            "multiple graph changes.[/yellow]",
            title="[bold yellow]Snap Assignment Warning[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        )
    )

    if not (start_node.has_molecular_graph and end_node.has_molecular_graph):
        console.print(
            "[yellow]⚠ --snap-assign requested, but molecular graphs are unavailable; "
            "leaving endpoint atom order unchanged.[/yellow]"
        )
        return nodes

    if not _is_connectivity_identical(start_node, end_node, verbose=False):
        console.print(
            "[yellow]⚠ --snap-assign requested, but endpoint molecular graphs are not identical; "
            "leaving endpoint atom order unchanged.[/yellow]"
        )
        return nodes

    try:
        from qcdata import Structure as QCDataStructure
        from qcinf import snap_assign
    except Exception as exc:
        raise typer.BadParameter(
            "--snap-assign requires qcinf and qcdata to be importable."
        ) from exc

    try:
        start_qcdata = QCDataStructure(**start_node.structure.model_dump())
        end_qcdata = QCDataStructure(**end_node.structure.model_dump())
        assignment = list(snap_assign(start_qcdata, end_qcdata))
        expected_indices = list(range(len(end_node.structure.symbols)))
        if sorted(int(i) for i in assignment) != expected_indices:
            raise ValueError(
                f"snap_assign returned an invalid atom permutation: {assignment}"
            )
        assignment_array = np.array(assignment, dtype=int)
        reordered_structure = Structure(
            geometry=np.asarray(end_node.structure.geometry)[assignment_array],
            symbols=np.asarray(end_node.structure.symbols, dtype=str)[
                assignment_array
            ].tolist(),
            charge=end_node.structure.charge,
            multiplicity=end_node.structure.multiplicity,
        )
        reordered_end = StructureNode(
            structure=reordered_structure,
            has_molecular_graph=end_node.has_molecular_graph,
            converged=end_node.converged,
            comparison_atom_indices=end_node.comparison_atom_indices,
            disable_smiles=end_node.disable_smiles,
            graph_atom_indices_source=end_node.graph_atom_indices_source,
            graph_subset_atom_count=end_node.graph_subset_atom_count,
            graph_total_atom_count=end_node.graph_total_atom_count,
        )
    except Exception as exc:
        raise typer.BadParameter(f"--snap-assign failed: {exc}") from exc

    console.print(
        "[green]✓ Applied qcinf snap_assign to reorder the end endpoint onto the start endpoint atom mapping.[/green]"
    )
    return [start_node, reordered_end]


def _elem_step_result_payload(result, chain: Chain) -> dict[str, Any]:
    try:
        energies = [float(energy) for energy in chain.energies]
    except Exception:
        energies = []
    try:
        energies_kcalmol = [float(energy) for energy in chain.energies_kcalmol]
    except Exception:
        energies_kcalmol = []

    minimization_results = list(getattr(result, "minimization_results", None) or [])
    new_structures = list(getattr(result, "new_structures", None) or [])
    return {
        "is_elem_step": bool(getattr(result, "is_elem_step", False)),
        "is_concave": bool(getattr(result, "is_concave", False)),
        "splitting_criterion": getattr(result, "splitting_criterion", None),
        "number_grad_calls": int(getattr(result, "number_grad_calls", 0) or 0),
        "minimization_result_count": len(minimization_results),
        "new_structure_count": len(new_structures),
        "chain_node_count": len(chain.nodes),
        "chain_energies_hartree": energies,
        "chain_relative_energies_kcalmol": energies_kcalmol,
    }


def _geometry_optimizer_keywords(program_input: RunInputs, *, default_maxiter: int = 500) -> dict[str, Any]:
    keywords = {"coordsys": "cart", "maxiter": int(default_maxiter)}
    user_keywords = dict(getattr(program_input, "geometry_optimizer_kwds", {}) or {})
    keywords.update(user_keywords)
    return keywords


@app.command("check-elem-step")
def check_elem_step_cli(
    geometries: Annotated[
        str,
        typer.Argument(
            help="XYZ path containing the chain to classify as elementary or non-elementary."
        ),
    ],
    inputs: Annotated[
        str | None,
        typer.Option("--inputs", "-i", help="Path to RunInputs TOML file."),
    ] = None,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Path for the JSON elementary-step result."),
    ] = "elem_step_result.json",
    new_structures_output: Annotated[
        str | None,
        typer.Option(
            "--new-structures-output",
            help="Optional XYZ path for newly discovered structures in the result.",
        ),
    ] = None,
    charge: Annotated[int, typer.Option(help="Charge used when reading XYZ geometries.")] = 0,
    multiplicity: Annotated[
        int,
        typer.Option(help="Spin multiplicity used when reading XYZ geometries."),
    ] = 1,
    validate_minima_with_hessian: Annotated[
        bool,
        typer.Option(
            "--validate-minima-with-hessian",
            help="Hessian-check minima split candidates before accepting them.",
        ),
    ] = False,
    hessian_minimum_frequency_cutoff: Annotated[
        float,
        typer.Option(
            "--hessian-minimum-frequency-cutoff",
            help="Minimum frequency cutoff for --validate-minima-with-hessian.",
        ),
    ] = 0.0,
    hessian_minima_rescue_displacement: Annotated[
        float | None,
        typer.Option(
            "--hessian-minima-rescue-displacement",
            help=(
                "Rescue displacement in bohr applied along the lowest-frequency "
                "mode when a Hessian-validated minimum is rejected."
            ),
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose/--quiet", help="Show detailed elementary-step diagnostics."),
    ] = True,
):
    console.print(BANNER)
    geom_fp = Path(geometries).expanduser().resolve()
    if not geom_fp.exists():
        console.print(f"[bold red]x ERROR:[/bold red] Geometry file not found: {geom_fp}")
        raise typer.Exit(1)

    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        program_input = (
            RunInputs.open(inputs)
            if inputs is not None
            else RunInputs(program="xtb", engine_name="qcop")
        )

    with console.status(f"[bold cyan]Loading chain from {geom_fp}...[/bold cyan]"):
        try:
            chain = Chain.from_xyz(
                fp=geom_fp,
                parameters=program_input.chain_inputs,
                charge=charge,
                spinmult=multiplicity,
            )
        except ValueError:
            chain = Chain.from_xyz(
                fp=geom_fp,
                parameters=program_input.chain_inputs,
                charge=None,
                spinmult=None,
            )

    if not chain._energies_already_computed:
        with console.status("[bold cyan]Computing chain energies...[/bold cyan]"):
            program_input.engine.compute_energies(chain)
    if hessian_minima_rescue_displacement is not None:
        program_input.path_min_inputs.hessian_minima_rescue_displacement = float(
            hessian_minima_rescue_displacement
        )

    result = check_if_elem_step(
        inp_chain=chain,
        engine=program_input.engine,
        verbose=verbose,
        validate_minima_with_hessian=bool(validate_minima_with_hessian),
        hessian_minimum_frequency_cutoff=float(hessian_minimum_frequency_cutoff),
        hessian_minima_rescue_displacement=float(
            program_input.path_min_inputs.hessian_minima_rescue_displacement
        ),
    )
    payload = _elem_step_result_payload(result, chain)

    new_structures = list(getattr(result, "new_structures", None) or [])
    if new_structures_output is not None and new_structures:
        new_structures_fp = _write_nodes_xyz(new_structures, new_structures_output)
        payload["new_structures_xyz"] = str(new_structures_fp)

    output_fp = Path(output).expanduser().resolve()
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    output_fp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Elementary step", str(payload["is_elem_step"]))
    table.add_row("Splitting criterion", str(payload["splitting_criterion"]))
    table.add_row("Geometry opt grad calls", str(payload["number_grad_calls"]))
    table.add_row("New structures", str(payload["new_structure_count"]))
    table.add_row("Result JSON", str(output_fp))
    if "new_structures_xyz" in payload:
        table.add_row("New structures XYZ", str(payload["new_structures_xyz"]))
    console.print(
        Panel(
            table,
            title="[bold cyan]Elementary Step Result[/bold cyan]",
            border_style="cyan",
        )
    )


def _single_frame_xyz_text(structure: Structure) -> str:
    return structure.to_xyz().rstrip() + "\n"


def _all_to_all_run_name(base_name: str | None, start_index: int, end_index: int) -> str:
    suffix = f"start{start_index}_end{end_index}"
    if base_name is None:
        return f"mep_output_{suffix}"
    base_path = Path(base_name)
    return str(base_path.with_name(f"{base_path.stem}_{suffix}"))


def _all_to_all_pair_status_path(pair_name: str) -> Path:
    pair_path = Path(pair_name)
    if not pair_path.is_absolute():
        pair_path = Path.cwd() / pair_path
    return _run_status_path(pair_path.parent, pair_path.stem)


def _all_to_all_pair_is_completed(pair_name: str) -> bool:
    status_fp = _all_to_all_pair_status_path(pair_name)
    if not status_fp.exists():
        return False
    try:
        payload = json.loads(status_fp.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("run_state") or "").lower() == "completed"


def _maybe_run_all_to_all_xyz_endpoints(
    *,
    start: str | None,
    end: str | None,
    geometries: str | None,
    use_smiles: bool,
    charge: int,
    multiplicity: int,
    max_interpolations: int,
    recursive_run_kwargs: dict[str, Any],
) -> bool:
    if use_smiles or geometries is not None or start is None or end is None:
        return False
    if max_interpolations < 1:
        raise typer.BadParameter("--max-interpolations must be at least 1.")

    start_fp = Path(start).expanduser()
    end_fp = Path(end).expanduser()
    if start_fp.suffix.lower() != ".xyz" or end_fp.suffix.lower() != ".xyz":
        return False
    if not start_fp.exists() or not end_fp.exists():
        return False

    try:
        start_structures = _parse_xyz_text_to_structures(
            start_fp.read_text(encoding="utf-8"),
            charge=charge,
            multiplicity=multiplicity,
        )
        end_structures = _parse_xyz_text_to_structures(
            end_fp.read_text(encoding="utf-8"),
            charge=charge,
            multiplicity=multiplicity,
        )
    except Exception:
        return False

    if len(start_structures) == 1 and len(end_structures) == 1:
        return False

    pair_count = len(start_structures) * len(end_structures)
    if pair_count > max_interpolations:
        console.print(
            Panel(
                "[bold yellow]LOUD WARNING: all-to-all endpoint expansion exceeds the interpolation cap.[/bold yellow]\n"
                f"[white]{len(start_structures)} start structure(s) x {len(end_structures)} end structure(s) "
                f"would create {pair_count} interpolation(s).[/white]\n"
                f"[yellow]Only the first {max_interpolations} interpolation(s) will be run in all-to-all order.[/yellow]",
                title="[bold yellow]Interpolation Cap Applied[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )

    console.print(
        Panel(
            "[bold yellow]LOUD WARNING: interpreting multi-frame XYZ endpoints as an ALL-to-ALL query of interpolations.[/bold yellow]\n"
            f"[white]This will run {min(pair_count, max_interpolations)} of {pair_count} interpolation(s): every start frame against every end frame "
            f"(for example start-0 to end-0, start-0 to end-1, start-0 to end-2, ...).[/white]\n"
            f"[yellow]Cap: --max-interpolations={max_interpolations}.[/yellow]",
            title="[bold yellow]ALL-to-ALL Endpoint Expansion[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        )
    )

    base_name = recursive_run_kwargs.get("name")
    with tempfile.TemporaryDirectory(prefix="mepd-all-to-all-") as tmpdir:
        tmp_path = Path(tmpdir)
        for interpolation_index, (start_index, end_index) in enumerate(
            product(range(len(start_structures)), range(len(end_structures)))
        ):
            if interpolation_index >= max_interpolations:
                break
            pair_start_fp = tmp_path / f"start_{start_index}.xyz"
            pair_end_fp = tmp_path / f"end_{end_index}.xyz"
            pair_start_fp.write_text(
                _single_frame_xyz_text(start_structures[start_index]),
                encoding="utf-8",
            )
            pair_end_fp.write_text(
                _single_frame_xyz_text(end_structures[end_index]),
                encoding="utf-8",
            )
            pair_name = _all_to_all_run_name(base_name, start_index, end_index)
            if _all_to_all_pair_is_completed(pair_name):
                console.print(
                    f"[yellow]↷ Skipping completed ALL-to-ALL interpolation start-{start_index} to end-{end_index} -> {pair_name}[/yellow]"
                )
                continue
            console.print(
                f"[bold cyan]▶ ALL-to-ALL interpolation start-{start_index} to end-{end_index} -> {pair_name}[/bold cyan]"
            )
            run(
                start=str(pair_start_fp),
                end=str(pair_end_fp),
                geometries=None,
                use_smiles=False,
                charge=charge,
                multiplicity=multiplicity,
                max_interpolations=max_interpolations,
                **{**recursive_run_kwargs, "name": pair_name},
            )
    return True


@app.command("run")
def run(
        start: Annotated[str, typer.Option(
            help='path to start file, or a reactant smiles')] = None,
        end: Annotated[str, typer.Option(
            help='path to end file, or a product smiles')] = None,
        geometries:  Annotated[str, typer.Argument(help='file containing an approximate path between two endpoints, \
            or a *_network_completion directory containing one *_network.json. \
            Use this if you have precompted a path you want to use. Do not use this with smiles.')] = None,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to RunInputs toml file')] = None,
        use_smiles: bool = False,
        use_tsopt: Annotated[bool, typer.Option(
            help='whether to run a transition state optimization on each TS guess')] = False,
        minimize_ends: bool = False,
        recursive: bool = False,
        parallel: Annotated[bool, typer.Option(
            "--parallel",
            help="Run recursive autosplitting in parallel with bounded worker concurrency.",
        )] = False,
        parallel_workers: Annotated[int | None, typer.Option(
            "--parallel-workers",
            help="Maximum number of concurrent recursive split workers used by --parallel. Defaults to min(4, CPU count).",
        )] = None,
        name: str = None,
        charge: int = 0,
        multiplicity: int = 1,
        network_completion: Annotated[bool, typer.Option(
            "--network-completion",
            help="After a recursive MSMEP run, enqueue follow-on completion requests and build a reaction network from all attempted pairs.",
        )] = False,
        network_completion_mode: Annotated[NetworkCompletionMode, typer.Option(
            "--network-completion-mode",
            help="Follow-on completion request strategy: 'linear' connects every discovered intermediate to the original reactant/product endpoints; 'all-to-all' attempts every non-adjacent pair on each discovered path.",
            case_sensitive=False,
        )] = "linear",
        overwrite: Annotated[bool, typer.Option(
            "--overwrite",
            help="When writing follow-on network-completion data, delete any existing *_network_completion output directory and rebuild it from scratch.",
        )] = False,
        create_irc: Annotated[bool, typer.Option(
            help='whether to run and output an IRC chain. Need to set --use_tsopt also, otherwise\
                will attempt use the guess structure.')] = False,
        program: Annotated[str, typer.Option(
            "--program",
            help="IRC backend when running `mepd run irc`: auto, geometric, or native.",
        )] = "auto",
        irc_output: Annotated[str, typer.Option(
            "--output", "-o",
            help="Output IRC XYZ path when running `mepd run irc`.",
        )] = None,
        ts_index: Annotated[int | None, typer.Option(
            "--ts-index",
            help="TS guess index when running `mepd run irc`. Defaults to highest-energy node.",
        )] = None,
        irc_keyword: Annotated[list[str] | None, typer.Option(
            "--irc-keyword",
            help="IRC backend keyword override as KEY=VALUE when running `mepd run irc`. Can be repeated.",
        )] = None,
        max_interpolations: Annotated[int, typer.Option(
            "--max-interpolations",
            help="Maximum all-to-all start/end XYZ frame interpolations allowed when multi-frame endpoint XYZ files are provided.",
        )] = 25,
        snap_assign: Annotated[bool, typer.Option(
            "--snap-assign",
            help=(
                "When endpoint molecular graphs are identical, use qcinf.snap_assign "
                "to reorder the end endpoint onto the start endpoint atom mapping. "
                "Use only when start/end are supposed to be identical; likely unsafe "
                "for autosplitting across multiple graph changes."
            ),
        )] = False,
        validate_minima_with_hessian: Annotated[bool, typer.Option(
            "--validate-minima-with-hessian",
            help=(
                "During recursive autosplitting, Hessian-check minima split candidates "
                "and reject candidates with significant imaginary modes."
            ),
        )] = False,
        hessian_minimum_frequency_cutoff: Annotated[float, typer.Option(
            "--hessian-minimum-frequency-cutoff",
            help=(
                "Minimum frequency cutoff for --validate-minima-with-hessian. "
                "Defaults to 0 cm^-1; any negative frequency is rejected."
            ),
        )] = 0.0,
        use_bigchem: Annotated[bool, typer.Option(
            help='whether to use chemcloud to compute hessian for ts opt and irc jobs')] = False):

    if str(geometries or "").strip().lower() == "irc":
        run_irc(
            geometries="mep_output.xyz",
            inputs=inputs,
            program=program,
            output=irc_output,
            ts_index=ts_index,
            charge=charge,
            multiplicity=multiplicity,
            irc_keyword=irc_keyword,
            use_bigchem=use_bigchem,
        )
        return

    # Print header
    console.print(BANNER)

    if parallel and recursive:
        raise typer.BadParameter(
            "--parallel cannot be combined with --recursive. Use one mode."
        )

    cpu_cap = max(1, int(os.cpu_count() or 1))
    default_parallel_workers = min(4, cpu_cap)
    if parallel_workers is None:
        parallel_workers = default_parallel_workers
    if parallel_workers < 1:
        raise typer.BadParameter("--parallel-workers must be at least 1.")
    if parallel_workers > cpu_cap:
        console.print(
            f"[yellow]⚠ Requested {parallel_workers} parallel workers exceeds detected CPU count ({cpu_cap}). Continuing with requested value; tune based on your host capacity.[/yellow]"
        )

    if network_completion and not recursive and not parallel:
        console.print(
            Panel(
                "[bold yellow]--network-completion requires recursive MSMEP.[/bold yellow]\n"
                "[bold cyan]Automatically enabling --recursive for this run.[/bold cyan]",
                title="[bold yellow]Recursive Mode Forced[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        recursive = True
    if overwrite and not network_completion:
        console.print(
            "[yellow]⚠ --overwrite only applies when --network-completion is enabled; ignoring.[/yellow]"
        )

    if _maybe_run_all_to_all_xyz_endpoints(
        start=start,
        end=end,
        geometries=geometries,
        use_smiles=use_smiles,
        charge=charge,
        multiplicity=multiplicity,
        max_interpolations=max_interpolations,
        recursive_run_kwargs={
            "inputs": inputs,
            "use_tsopt": use_tsopt,
            "minimize_ends": minimize_ends,
            "recursive": recursive,
            "parallel": parallel,
            "parallel_workers": parallel_workers,
            "name": name,
            "network_completion": network_completion,
            "network_completion_mode": network_completion_mode,
            "overwrite": overwrite,
            "create_irc": create_irc,
            "snap_assign": snap_assign,
            "validate_minima_with_hessian": validate_minima_with_hessian,
            "hessian_minimum_frequency_cutoff": hessian_minimum_frequency_cutoff,
            "use_bigchem": use_bigchem,
        },
    ):
        return

    table = Table(box=None, show_header=False)
    table.add_column(style="dim")
    mode_label = "parallel" if parallel else ("recursive" if recursive else "regular")
    table.add_row("[bold cyan]Command:[/bold cyan]", "[white]run[/white]")
    table.add_row("[bold cyan]Method:[/bold cyan]",
                  f"[yellow]{mode_label}[/yellow]")
    table.add_row("[bold cyan]SMILES mode:[/bold cyan]",
                  f"[yellow]{use_smiles}[/yellow]")
    table.add_row("[bold cyan]Parallel:[/bold cyan]",
                  f"[yellow]{parallel}[/yellow]")
    if parallel:
        table.add_row("[bold cyan]Parallel workers:[/bold cyan]",
                      f"[yellow]{parallel_workers}[/yellow]")
    table.add_row("[bold cyan]Network completion:[/bold cyan]",
                  f"[yellow]{network_completion}[/yellow]")
    if network_completion:
        table.add_row("[bold cyan]Network completion mode:[/bold cyan]",
                      f"[yellow]{network_completion_mode}[/yellow]")
        table.add_row("[bold cyan]Overwrite splits:[/bold cyan]",
                      f"[yellow]{overwrite}[/yellow]")
    table.add_row("[bold cyan]Hessian minima validation:[/bold cyan]",
                  f"[yellow]{validate_minima_with_hessian}[/yellow]")
    table.add_row("[bold cyan]Snap assign:[/bold cyan]",
                  f"[yellow]{snap_assign}[/yellow]")
    console.print(table)
    console.print()

    start_time = time.time()
    filename: Path | None = None
    status_fp: Path | None = None
    tot_grad_calls: int | None = None
    geom_grad_calls: int | None = None
    network_completion_source_dir: Path | None = None
    network_completion_source_base_name: str | None = None
    network_completion_followup_grad_calls_estimate = 0
    network_completion_followup_geom_grad_calls = 0
    # load the structures
    if use_smiles:
        from mepd.nodes.nodehelpers import create_pairs_from_smiles
        from mepd.arbalign import align_structures

        console.print(
            "[yellow]⚠ WARNING:[/yellow] Using RXNMapper to create atomic mapping. Carefully check output to see how labels affected reaction path.")
        with console.status("[bold cyan]Creating structures from SMILES...[/bold cyan]") as status:
            start_structure, end_structure = create_pairs_from_smiles(
                smi1=start, smi2=end)

            console.print(
                "[cyan]Using arbalign to optimize index labelling for endpoints[/cyan]")
            end_structure = align_structures(
                start_structure, end_structure, distance_metric='RMSD')

        all_structs = [start_structure, end_structure]
    else:

        if geometries is not None:
            with console.status(f"[bold cyan]Loading geometries from {geometries}...[/bold cyan]"):
                geometries_path = Path(geometries).expanduser().resolve()
                if geometries_path.is_dir():
                    network_fps = sorted(geometries_path.glob("*_network.json"))
                    if len(network_fps) != 1:
                        console.print(
                            "[bold red]✗ ERROR:[/bold red] Network-completion directory input must contain exactly one *_network.json file."
                        )
                        raise typer.Exit(1)
                    best_chain = _load_best_path_chain_from_network_json(
                        network_fps[0]
                    )
                    if best_chain is None or len(best_chain.nodes) < 2:
                        console.print(
                            "[bold red]✗ ERROR:[/bold red] Could not load a best-path chain from the provided network-completion directory."
                        )
                        raise typer.Exit(1)
                    network_completion_source_dir = geometries_path
                    network_completion_source_base_name = network_fps[0].name
                    if network_completion_source_base_name.endswith("_network.json"):
                        network_completion_source_base_name = network_completion_source_base_name[
                            : -len("_network.json")
                        ]
                    all_structs = [node.structure for node in best_chain.nodes]
                else:
                    try:
                        all_structs = read_multiple_structure_from_file(
                            geometries, charge=charge, spinmult=multiplicity)
                    except ValueError:  # qcio does not allow an input for charge if file has a charge in it
                        all_structs = read_multiple_structure_from_file(
                            geometries, charge=None, spinmult=None)
        elif start is not None and end is not None:
            with console.status(f"[bold cyan]Loading structures...[/bold cyan]"):
                console.print(
                    f"[dim]Charge: {charge}, Multiplicity: {multiplicity}[/dim]")
                start_ref = _load_endpoint_structure(
                    start,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                end_ref = _load_endpoint_structure(
                    end,
                    charge=charge,
                    multiplicity=multiplicity,
                )

                if start_ref.charge != charge or start_ref.multiplicity != multiplicity:
                    console.print(
                        f"[yellow]⚠ WARNING:[/yellow] {start} has charge {start_ref.charge} and multiplicity {start_ref.multiplicity}. Using {charge} and {multiplicity} instead."
                    )
                    start_ref = Structure(geometry=start_ref.geometry,
                                          charge=charge,
                                          multiplicity=multiplicity,
                                          symbols=start_ref.symbols)
                if end_ref.charge != charge or end_ref.multiplicity != multiplicity:
                    console.print(
                        f"[yellow]⚠ WARNING:[/yellow] {end} has charge {end_ref.charge} and multiplicity {end_ref.multiplicity}. Using {charge} and {multiplicity} instead."
                    )
                    end_ref = Structure(geometry=end_ref.geometry,
                                        charge=charge,
                                        multiplicity=multiplicity,
                                        symbols=end_ref.symbols)

                all_structs = [start_ref, end_ref]
        else:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Either 'geometries' or 'start' and 'end' flags must be populated!")
            raise typer.Exit(1)

    # load the RunInputs
    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        if inputs is not None:
            program_input = RunInputs.open(inputs)
        else:
            program_input = RunInputs(program='xtb', engine_name='qcop')
    if validate_minima_with_hessian:
        setattr(program_input.path_min_inputs, "validate_minima_with_hessian", True)
        setattr(
            program_input.path_min_inputs,
            "hessian_minimum_frequency_cutoff",
            float(hessian_minimum_frequency_cutoff),
        )
        if not recursive and not parallel:
            console.print(
                "[yellow]⚠ --validate-minima-with-hessian only affects autosplitting "
                "elementary-step checks; enable --recursive or --parallel to use it for splitting.[/yellow]"
            )

    _render_runinputs(program_input)
    sys.stdout.flush()
    write_qcio = bool(getattr(program_input, "write_qcio", False))

    # minimize endpoints if requested
    all_nodes = [StructureNode(structure=s) for s in all_structs]
    if geometries is not None:
        _maybe_hydrate_nodes_from_xyz_sidecars(
            geometries=geometries,
            nodes=all_nodes,
            chain_inputs=program_input.chain_inputs,
            charge=charge,
            multiplicity=multiplicity,
        )
    if snap_assign:
        all_nodes = _snap_assign_endpoint_nodes(all_nodes)
    if minimize_ends:
        console.print("[bold cyan]⟳ Minimizing input endpoints...[/bold cyan]")
        batch_optimizer = getattr(
            program_input.engine, "compute_geometry_optimizations", None)
        endpoint_opt_keywords = _geometry_optimizer_keywords(program_input)
        endpoint_labels = ("start", "end")
        requested_endpoints = [all_nodes[0], all_nodes[-1]]

        def _endpoint_from_trajectory(
            trajectories: list[Any] | tuple[Any, ...], endpoint_index: int
        ) -> StructureNode:
            label = endpoint_labels[endpoint_index]
            if endpoint_index >= len(trajectories):
                console.print(
                    f"[yellow]⚠ {label.capitalize()} endpoint optimization did not return a trajectory; keeping input geometry.[/yellow]"
                )
                return requested_endpoints[endpoint_index]
            trajectory = trajectories[endpoint_index]
            if not trajectory:
                console.print(
                    f"[yellow]⚠ {label.capitalize()} endpoint optimization returned an empty trajectory; keeping input geometry.[/yellow]"
                )
                return requested_endpoints[endpoint_index]
            return trajectory[-1]

        if callable(batch_optimizer):
            console.print(
                "[dim]Submitting batched endpoint geometry optimizations...[/dim]")
            try:
                try:
                    trajectories = batch_optimizer(
                        [all_nodes[0], all_nodes[-1]],
                        keywords=endpoint_opt_keywords,
                    )
                except TypeError:
                    trajectories = batch_optimizer([all_nodes[0], all_nodes[-1]])
            except Exception as exc:
                console.print(
                    f"[yellow]⚠ Endpoint batch minimization failed ({type(exc).__name__}: {exc}); keeping input geometries.[/yellow]"
                )
            else:
                if not isinstance(trajectories, (list, tuple)):
                    console.print(
                        "[yellow]⚠ Endpoint batch minimization returned an unexpected result type; keeping input geometries.[/yellow]"
                    )
                else:
                    all_nodes[0] = _endpoint_from_trajectory(trajectories, 0)
                    all_nodes[-1] = _endpoint_from_trajectory(trajectories, 1)
        else:
            console.print("[dim]Minimizing start endpoint...[/dim]")
            try:
                start_tr = program_input.engine.compute_geometry_optimization(
                    all_nodes[0], keywords=endpoint_opt_keywords)
                if start_tr:
                    all_nodes[0] = start_tr[-1]
                else:
                    console.print(
                        "[yellow]⚠ Start endpoint optimization returned an empty trajectory; keeping input geometry.[/yellow]"
                    )
            except Exception as exc:
                console.print(
                    f"[yellow]⚠ Start endpoint minimization failed ({type(exc).__name__}: {exc}); keeping input geometry.[/yellow]"
                )
            console.print("[dim]Minimizing end endpoint...[/dim]")
            try:
                end_tr = program_input.engine.compute_geometry_optimization(
                    all_nodes[-1], keywords=endpoint_opt_keywords)
                if end_tr:
                    all_nodes[-1] = end_tr[-1]
                else:
                    console.print(
                        "[yellow]⚠ End endpoint optimization returned an empty trajectory; keeping input geometry.[/yellow]"
                    )
            except Exception as exc:
                console.print(
                    f"[yellow]⚠ End endpoint minimization failed ({type(exc).__name__}: {exc}); keeping input geometry.[/yellow]"
                )
        console.print("[bold green]✓ Done![/bold green]")

    # create Chain
    console.print(f"[dim]Loading {len(all_nodes)} nodes...[/dim]")
    chain = Chain.model_validate({
        "nodes": all_nodes,
        "parameters": program_input.chain_inputs}
    )

    # create MSMEP object
    m = MSMEP(inputs=program_input)

    # Run the optimization
    chain_for_profile = None
    fp = Path("mep_output")

    if parallel:
        if name is not None:
            name = Path(name)
            data_dir = Path(name).resolve().parent
            foldername = data_dir / name.stem
            filename = data_dir / (name.stem + ".xyz")
        else:
            data_dir = Path(os.getcwd())
            foldername = data_dir / f"{fp.stem}_parallel_msmep"
            filename = data_dir / f"{fp.stem}_parallel_msmep.xyz"
        status_fp = _run_status_path(data_dir, filename.stem)
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="running",
            phase="parallel_recursive_request",
            recursive=False,
            parallel=True,
            network_completion=network_completion,
            path_min_method=str(program_input.path_min_method),
        )

        if not program_input.path_min_inputs.do_elem_step_checks:
            console.print(
                "[yellow]⚠ WARNING: do_elem_step_checks is set to False. This may cause issues with recursive splitting.[/yellow]")
            console.print(
                "[yellow]Making it True to ensure proper functioning of recursive splitting.[/yellow]")
            program_input.path_min_inputs.do_elem_step_checks = True

        console.print(
            f"[bold magenta]▶ RUNNING PARALLEL AUTOSPLITTING {program_input.path_min_method} (max_workers={parallel_workers})[/bold magenta]"
        )
        try:
            history = m.run_parallel_recursive_minimize(
                chain,
                max_workers=parallel_workers,
            )
        except KeyboardInterrupt:
            stop_status()
            raise
        except BaseException:
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="failed",
                phase="parallel_recursive_request",
                recursive=False,
                parallel=True,
                network_completion=network_completion,
                path_min_method=str(program_input.path_min_method),
                error=traceback.format_exc().strip(),
            )
            stop_status()
            raise

        if not history.data:
            leaf_status = str(getattr(history, "leaf_status", "") or "unknown")
            if leaf_status == "identical_endpoints":
                empty_history_msg = (
                    "Program did not run because endpoints were classified as identical "
                    "under current thresholds (node_rms_thre/node_ene_thre)."
                )
            elif leaf_status == "electronic_structure_error":
                empty_history_msg = (
                    "Program did not run because electronic structure evaluation failed "
                    "during recursive minimization."
                )
            else:
                empty_history_msg = (
                    "Program did not run. Likely because your endpoints are conformers "
                    "of the same molecular graph."
                )
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="failed",
                phase="parallel_recursive_request",
                recursive=False,
                parallel=True,
                network_completion=network_completion,
                path_min_method=str(program_input.path_min_method),
                error=f"{empty_history_msg} leaf_status={leaf_status}",
            )
            stop_status()
            console.print(
                f"[bold red]✗ ERROR:[/bold red] {empty_history_msg} "
                "Tighten node_rms_thre/node_ene_thre in chain_inputs and try again."
            )
            raise typer.Exit(1)

        successful_leaf_chains = []
        recoverable_leaves = getattr(
            history, "recoverable_ordered_leaves", history.ordered_leaves
        )
        for leaf in recoverable_leaves:
            if not leaf.data:
                continue
            leaf_chain = None
            if getattr(leaf.data, "chain_trajectory", None):
                leaf_chain = leaf.data.chain_trajectory[-1]
            elif getattr(leaf.data, "optimized", None) is not None:
                leaf_chain = leaf.data.optimized
            if leaf_chain is not None:
                successful_leaf_chains.append(leaf_chain)

        parallel_failures = list(getattr(history, "parallel_failures", []) or [])

        if not successful_leaf_chains:
            root_chain = None
            if history.data is not None:
                if getattr(history.data, "chain_trajectory", None):
                    root_chain = history.data.chain_trajectory[-1]
                elif getattr(history.data, "optimized", None) is not None:
                    root_chain = history.data.optimized

            if root_chain is not None:
                console.print(
                    "[yellow]⚠ Parallel autosplitting produced no successful child leaves; "
                    "falling back to the root optimized chain.[/yellow]"
                )
                successful_leaf_chains = [root_chain]
            else:
                _write_run_status(
                    status_fp,
                    base_name=filename.stem,
                    run_state="failed",
                    phase="parallel_recursive_request",
                    recursive=False,
                    parallel=True,
                    network_completion=network_completion,
                    path_min_method=str(program_input.path_min_method),
                    error="Parallel autosplitting produced no successful leaf chains.",
                )
                console.print(
                    "[bold red]✗ ERROR:[/bold red] Parallel autosplitting did not yield any successful leaf chains."
                )
                if parallel_failures:
                    console.print(
                        f"[yellow]Captured {len(parallel_failures)} branch failure(s). "
                        "Showing the first one below.[/yellow]"
                    )
                    console.print(f"[dim]{parallel_failures[0]}[/dim]")
                raise typer.Exit(1)

        merged_chain = history.output_chain

        if parallel_failures:
            console.print(
                f"[yellow]⚠ {len(parallel_failures)} branch worker failure(s) occurred during "
                "parallel autosplitting; recovered branches were retained where possible.[/yellow]"
            )
            max_shown = min(3, len(parallel_failures))
            console.print(
                f"[yellow]Showing {max_shown} branch failure detail(s):[/yellow]"
            )
            for i, failure_text in enumerate(parallel_failures[:max_shown], start=1):
                console.print(f"[dim][parallel-failure {i}] {failure_text}[/dim]")
        identical_skipped_leaves = sum(
            1
            for leaf in history.depth_first_ordered_nodes
            if leaf.is_leaf
            and not bool(leaf.data)
            and getattr(leaf, "leaf_status", "") == "identical_endpoints"
        )
        failed_leaves = sum(
            1
            for leaf in history.depth_first_ordered_nodes
            if leaf.is_leaf
            and not bool(leaf.data)
            and getattr(leaf, "leaf_status", "") != "identical_endpoints"
        )
        if identical_skipped_leaves > 0:
            console.print(
                f"[yellow]⚠ {identical_skipped_leaves} parallel branch(es) were skipped because endpoints were identical.[/yellow]"
            )
        if failed_leaves > 0:
            console.print(
                f"[yellow]⚠ {failed_leaves} parallel branch(es) failed. Writing a partial merged chain from successful leaves.[/yellow]"
            )

        leaves_nebs = [obj for obj in history.get_optimization_history() if obj]
        end_time = time.time()
        merged_chain.write_to_disk(filename, write_qcio=write_qcio)
        history.write_to_disk(foldername, write_qcio=write_qcio)
        chain_for_profile = merged_chain
        if network_completion:
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="running",
                phase="network_completion",
                recursive=False,
                parallel=True,
                network_completion=True,
                path_min_method=str(program_input.path_min_method),
                output_chain_path=filename,
                tree_path=foldername,
            )
            network_dir = data_dir / f"{filename.stem}_network_completion"
            console.print(
                "[bold magenta]▶ RUNNING FOLLOW-ON NETWORK SPLIT REQUESTS[/bold magenta]"
            )
            request_records, network_fp, manifest_fp, network_cost_summary = _run_recursive_network_completion(
                history=history,
                program_input=program_input,
                initial_start=chain[0],
                initial_end=chain[-1],
                output_dir=network_dir,
                base_name=filename.stem,
                status_fp=status_fp,
                source_tree_dir=foldername,
                parallel_recursive=True,
                parallel_workers=parallel_workers,
                overwrite=overwrite,
                split_mode=network_completion_mode,
            )
            network_completion_followup_grad_calls_estimate = int(
                network_cost_summary.get("gradient_calls_total", 0)
            )
            network_completion_followup_geom_grad_calls = int(
                network_cost_summary.get("gradient_calls_geometry_optimizations", 0)
            )
            console.print(
                f"[cyan]Completed {len(request_records)} total recursive pair requests.[/cyan]"
            )
            if network_fp is not None:
                console.print(
                    f"[cyan]Network summary written to {network_fp}[/cyan]"
                )
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="completed",
                phase="complete",
                recursive=False,
                parallel=True,
                network_completion=True,
                path_min_method=str(program_input.path_min_method),
                output_chain_path=filename,
                tree_path=foldername,
                network_completion_dir=network_dir,
                manifest_fp=manifest_fp,
                network_fp=network_fp,
            )
        else:
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="completed",
                phase="complete",
                recursive=False,
                parallel=True,
                network_completion=False,
                path_min_method=str(program_input.path_min_method),
                output_chain_path=filename,
                tree_path=foldername,
            )

        if use_tsopt:
            for i, leaf in enumerate(history.ordered_leaves):
                if not leaf.data:
                    continue
                if not leaf.data.chain_trajectory:
                    console.print(
                        f"[yellow]⚠ Skipping TS optimization on leaf {i}: no chain trajectory.[/yellow]"
                    )
                    continue
                console.print(
                    f"[bold cyan]⟳ Running TS opt on leaf {i}...[/bold cyan]")
                try:
                    ts_node, tsres = _compute_ts_node(
                        engine=program_input.engine,
                        ts_guess=leaf.data.chain_trajectory[-1].get_ts_node(),
                    )
                except Exception:
                    console.print(
                        f"[yellow]⚠ TS optimization crashed on leaf {i}: {traceback.format_exc()}[/yellow]"
                    )
                    continue

                if tsres is not None and hasattr(tsres, "save"):
                    tsres.save(data_dir / (filename.stem+f"_tsres_{i}.qcio"))
                if ts_node is not None:
                    ts_node.structure.save(
                        data_dir / (filename.stem+f"_ts_{i}.xyz"))
                    if create_irc:
                        try:
                            irc = compute_irc_chain(
                                ts_node=ts_node,
                                engine=program_input.engine,
                            )
                            irc.write_to_disk(
                                filename.stem+f"_tsres_{i}_IRC.xyz")

                        except Exception:
                            console.print(
                                f"[yellow]⚠ IRC failed: {traceback.format_exc()}[/yellow]")
                            console.print(
                                "[yellow]IRC failed. Continuing...[/yellow]")
                else:
                    console.print(
                        f"[yellow]⚠ TS optimization did not converge on leaf {i}...[/yellow]")

        tot_grad_calls = sum(getattr(obj, "grad_calls_made", 0)
                             for obj in leaves_nebs)
        geom_grad_calls = sum(
            getattr(obj, "geom_grad_calls_made", 0) for obj in leaves_nebs
        )
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {tot_grad_calls} gradient calls total.[/cyan]")
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {geom_grad_calls} gradient for geometry optimizations.[/cyan]")

    elif recursive:
        if name is not None:
            name = Path(name)
            data_dir = Path(name).resolve().parent
            foldername = data_dir / name.stem
            filename = data_dir / (name.stem + ".xyz")
        else:
            data_dir = Path(os.getcwd())
            foldername = data_dir / f"{fp.stem}_msmep"
            filename = data_dir / f"{fp.stem}_msmep.xyz"
        status_fp = _run_status_path(data_dir, filename.stem)
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="running",
            phase="initial_recursive_request",
            recursive=True,
            parallel=False,
            network_completion=network_completion,
            path_min_method=str(program_input.path_min_method),
        )

        if not program_input.path_min_inputs.do_elem_step_checks:
            console.print(
                "[yellow]⚠ WARNING: do_elem_step_checks is set to False. This may cause issues with recursive splitting.[/yellow]")
            console.print(
                "[yellow]Making it True to ensure proper functioning of recursive splitting.[/yellow]")
            program_input.path_min_inputs.do_elem_step_checks = True
        console.print(
            f"[bold magenta]▶ RUNNING AUTOSPLITTING {program_input.path_min_method}[/bold magenta]")
        history = None
        source_split_history = None
        if (
            network_completion
            and network_completion_source_dir is not None
            and network_completion_source_dir.name.endswith("_network_completion")
        ):
            source_split_history = _load_recursive_split_request_history(
                network_completion_source_dir,
                0,
                chain_inputs=program_input.chain_inputs,
                engine=program_input.engine,
                charge=int(chain[0].structure.charge),
                multiplicity=int(chain[0].structure.multiplicity),
            )
            if source_split_history is not None and source_split_history.data is not None:
                history = source_split_history
                console.print(
                    f"[bold cyan]Resuming recursive history from source network-completion folder {network_completion_source_dir}[/bold cyan]"
                )
            else:
                console.print(
                    f"[yellow]⚠ Could not load request_0 history from {network_completion_source_dir}; running a fresh recursive autosplit from the provided best-path chain.[/yellow]"
                )
        if history is None:
            history = (
                _maybe_resume_recursive_history(
                    status_fp,
                    chain_inputs=program_input.chain_inputs,
                    engine=program_input.engine,
                    charge=int(chain[0].structure.charge),
                    multiplicity=int(chain[0].structure.multiplicity),
                )
                if network_completion
                else None
            )
            if history is not None:
                console.print(
                    f"[bold cyan]Resuming saved recursive history from {foldername}[/bold cyan]"
                )
        if history is None:
            try:
                history = m.run_recursive_minimize(chain)
            except KeyboardInterrupt:
                stop_status()
                raise
            except BaseException:
                _write_run_status(
                    status_fp,
                    base_name=filename.stem,
                    run_state="failed",
                    phase="initial_recursive_request",
                    recursive=True,
                    parallel=False,
                    network_completion=network_completion,
                    path_min_method=str(program_input.path_min_method),
                    error=traceback.format_exc().strip(),
                )
                stop_status()
                raise

        if not history.data:
            leaf_status = str(getattr(history, "leaf_status", "") or "unknown")
            if leaf_status == "identical_endpoints":
                empty_history_msg = (
                    "Program did not run because endpoints were classified as identical "
                    "under current thresholds (node_rms_thre/node_ene_thre)."
                )
            elif leaf_status == "electronic_structure_error":
                empty_history_msg = (
                    "Program did not run because electronic structure evaluation failed "
                    "during recursive minimization."
                )
            else:
                empty_history_msg = (
                    "Program did not run. Likely because your endpoints are conformers "
                    "of the same molecular graph."
                )
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="failed",
                phase="initial_recursive_request",
                recursive=True,
                parallel=False,
                network_completion=network_completion,
                path_min_method=str(program_input.path_min_method),
                error=f"{empty_history_msg} leaf_status={leaf_status}",
            )
            stop_status()
            console.print(
                f"[bold red]✗ ERROR:[/bold red] {empty_history_msg} "
                "Tighten node_rms_thre/node_ene_thre in chain_inputs and try again."
            )
            raise typer.Exit(1)

        leaves_nebs = [
            obj for obj in history.get_optimization_history() if obj]
        end_time = time.time()
        history.output_chain.write_to_disk(filename, write_qcio=write_qcio)
        history.write_to_disk(foldername, write_qcio=write_qcio)
        chain_for_profile = history.output_chain
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="running" if network_completion else "completed",
            phase="network_completion" if network_completion else "complete",
            recursive=True,
            parallel=False,
            network_completion=network_completion,
            path_min_method=str(program_input.path_min_method),
            output_chain_path=filename,
            tree_path=foldername,
        )

        if network_completion:
            source_network_dir = (
                network_completion_source_dir
                if (
                    network_completion_source_dir is not None
                    and network_completion_source_dir.name.endswith("_network_completion")
                )
                else None
            )
            network_dir = (
                source_network_dir
                if source_network_dir is not None
                else data_dir / f"{filename.stem}_network_completion"
            )
            split_base_name = (
                network_completion_source_base_name
                if source_network_dir is not None
                and network_completion_source_base_name is not None
                else filename.stem
            )
            console.print(
                "[bold magenta]▶ RUNNING FOLLOW-ON NETWORK SPLIT REQUESTS[/bold magenta]"
            )
            request_records, network_fp, manifest_fp, network_cost_summary = _run_recursive_network_completion(
                history=history,
                program_input=program_input,
                initial_start=chain[0],
                initial_end=chain[-1],
                output_dir=network_dir,
                base_name=split_base_name,
                status_fp=status_fp,
                source_tree_dir=foldername,
                overwrite=overwrite,
                overwrite_followups_only=source_network_dir is not None,
                split_mode=network_completion_mode,
            )
            network_completion_followup_grad_calls_estimate = int(
                network_cost_summary.get("gradient_calls_total", 0)
            )
            network_completion_followup_geom_grad_calls = int(
                network_cost_summary.get("gradient_calls_geometry_optimizations", 0)
            )
            console.print(
                f"[cyan]Completed {len(request_records)} total recursive pair requests.[/cyan]"
            )
            if network_fp is not None:
                console.print(
                    f"[cyan]Network summary written to {network_fp}[/cyan]"
                )
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="completed",
                phase="complete",
                recursive=True,
                parallel=False,
                network_completion=network_completion,
                path_min_method=str(program_input.path_min_method),
                output_chain_path=filename,
                tree_path=foldername,
                network_completion_dir=network_dir,
                manifest_fp=manifest_fp,
                network_fp=network_fp,
            )

        if use_tsopt:
            for i, leaf in enumerate(history.ordered_leaves):
                if not leaf.data:
                    continue
                if not leaf.data.chain_trajectory:
                    console.print(
                        f"[yellow]⚠ Skipping TS optimization on leaf {i}: no chain trajectory.[/yellow]"
                    )
                    continue
                console.print(
                    f"[bold cyan]⟳ Running TS opt on leaf {i}...[/bold cyan]")
                try:
                    ts_node, tsres = _compute_ts_node(
                        engine=program_input.engine,
                        ts_guess=leaf.data.chain_trajectory[-1].get_ts_node(),
                    )
                except Exception:
                    console.print(
                        f"[yellow]⚠ TS optimization crashed on leaf {i}: {traceback.format_exc()}[/yellow]"
                    )
                    continue

                if tsres is not None and hasattr(tsres, "save"):
                    tsres.save(data_dir / (filename.stem+f"_tsres_{i}.qcio"))
                if ts_node is not None:
                    ts_node.structure.save(
                        data_dir / (filename.stem+f"_ts_{i}.xyz"))
                    if create_irc:
                        try:
                            irc = compute_irc_chain(
                                ts_node=ts_node,
                                engine=program_input.engine,
                            )
                            irc.write_to_disk(
                                filename.stem+f"_tsres_{i}_IRC.xyz")

                        except Exception:
                            console.print(
                                f"[yellow]⚠ IRC failed: {traceback.format_exc()}[/yellow]")
                            console.print(
                                "[yellow]IRC failed. Continuing...[/yellow]")
                else:
                    console.print(
                        f"[yellow]⚠ TS optimization did not converge on leaf {i}...[/yellow]")

        tot_grad_calls = sum(getattr(obj, "grad_calls_made", 0)
                             for obj in leaves_nebs)
        geom_grad_calls = sum(
            getattr(obj, "geom_grad_calls_made", 0) for obj in leaves_nebs
        )
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {tot_grad_calls} gradient calls total.[/cyan]")
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {geom_grad_calls} gradient for geometry optimizations.[/cyan]")

    else:
        data_dir = Path(os.getcwd())
        if name is not None:
            filename = data_dir / (name + ".xyz")
        else:
            filename = data_dir / f"{fp.stem}_neb.xyz"
        status_fp = _run_status_path(data_dir, filename.stem)
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="running",
            phase="path_minimization",
            recursive=False,
            parallel=False,
            network_completion=False,
            path_min_method=str(program_input.path_min_method),
        )
        console.print(
            f"[bold magenta]▶ RUNNING REGULAR {program_input.path_min_method}[/bold magenta]")
        try:
            n, elem_step_results = m.run_minimize_chain(input_chain=chain)
        except Exception:
            _write_run_status(
                status_fp,
                base_name=filename.stem,
                run_state="failed",
                phase="path_minimization",
                recursive=False,
                parallel=False,
                network_completion=False,
                path_min_method=str(program_input.path_min_method),
                error=traceback.format_exc().strip(),
            )
            raise

        end_time = time.time()
        try:
            wrote_outputs = _write_neb_results_with_history(
                n, filename, write_qcio=write_qcio
            )
        except TypeError:
            wrote_outputs = _write_neb_results_with_history(n, filename)
        if n.chain_trajectory:
            chain_for_profile = n.chain_trajectory[-1]
        elif n.optimized is not None:
            chain_for_profile = n.optimized

        if not wrote_outputs:
            console.print(
                "[yellow]⚠ Skipping output write/profile because path minimization did not produce an optimized chain.[/yellow]"
            )
        _write_run_status(
            status_fp,
            base_name=filename.stem,
            run_state="completed",
            phase="complete",
            recursive=False,
            parallel=False,
            network_completion=False,
            path_min_method=str(program_input.path_min_method),
            output_chain_path=filename,
        )

        if use_tsopt and n.optimized is not None:
            console.print("[bold cyan]⟳ Running TS opt...[/bold cyan]")
            try:
                source_chain = n.chain_trajectory[-1] if n.chain_trajectory else n.optimized
                ts_node, tsres = _compute_ts_node(
                    engine=program_input.engine,
                    ts_guess=source_chain.get_ts_node(),
                )
            except Exception:
                console.print(
                    f"[yellow]⚠ TS optimization crashed: {traceback.format_exc()}[/yellow]"
                )
                ts_node, tsres = None, None
            if tsres is not None and hasattr(tsres, "save"):
                tsres.save(data_dir / (filename.stem+"_tsres.qcio"))
            if ts_node is not None:
                ts_node.structure.save(
                    data_dir / (filename.stem+"_ts.xyz"))

                if create_irc:
                    try:
                        irc = compute_irc_chain(
                            ts_node=ts_node, engine=program_input.engine
                        )
                        irc.write_to_disk(
                            filename.stem+"_tsres_IRC.xyz")

                    except Exception:
                        console.print(
                            f"[yellow]⚠ IRC failed: {traceback.format_exc()}[/yellow]")
                        console.print(
                            "[yellow]IRC failed. Continuing...[/yellow]")

            else:
                console.print("[yellow]⚠ TS optimization failed.[/yellow]")
        elif use_tsopt:
            console.print(
                "[yellow]⚠ Skipping TS optimization because path minimization did not converge.[/yellow]"
            )

        tot_grad_calls = n.grad_calls_made
        console.print(
            f"[bold green]✓[/bold green] [cyan]Made {tot_grad_calls} gradient calls total.[/cyan]")

    end_time = time.time()
    elapsed = end_time - start_time
    if filename is not None:
        primary_grad_calls = int(tot_grad_calls or 0)
        followup_grad_calls = int(network_completion_followup_grad_calls_estimate or 0)
        followup_geom_grad_calls = int(network_completion_followup_geom_grad_calls or 0)
        total_grad_calls = primary_grad_calls + followup_grad_calls
        total_geom_grad_calls = int(geom_grad_calls or 0) + followup_geom_grad_calls
        cost_fp = _cost_report_path(filename.parent, filename.stem)
        _write_cost_report(
            cost_fp,
            command="run",
            mode=mode_label,
            elapsed_seconds=elapsed,
            gradient_calls_total=total_grad_calls,
            gradient_calls_geometry_optimizations=total_geom_grad_calls,
            metadata={
                "output_chain_path": str(filename),
                "status_path": str(status_fp) if status_fp is not None else "",
                "recursive": bool(recursive),
                "parallel": bool(parallel),
                "network_completion": bool(network_completion),
                "gradient_calls_primary": primary_grad_calls,
                "gradient_calls_network_completion_followup_estimated": followup_grad_calls,
                "gradient_calls_geometry_optimizations_primary": int(
                    geom_grad_calls or 0
                ),
                "gradient_calls_geometry_optimizations_network_completion_followup": followup_geom_grad_calls,
                "path_min_method": str(program_input.path_min_method),
            },
        )
        console.print(f"[dim]Cost report: {cost_fp}[/dim]")

    # Print summary panel
    summary = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    if elapsed > 60:
        summary.add_row(
            "⏱ Walltime:", f"[yellow]{elapsed/60:.1f} min[/yellow]")
    else:
        summary.add_row("⏱ Walltime:", f"[yellow]{elapsed:.1f} s[/yellow]")
    summary.add_row("📁 Output:", f"[cyan]{filename}[/cyan]")
    console.print(Panel(
        summary, title="[bold green]✓ Complete![/bold green]", border_style="green"))

    if chain_for_profile is not None:
        _ascii_profile_for_chain(chain_for_profile)


@app.command("refine")
def refine(
        source: Annotated[str, typer.Argument(
            help="Source for refinement: TreeNode folder/object, NEB output/object, chain .xyz/object, network .json, or *_network_completion directory")] = None,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to expensive RunInputs toml file')] = None,
        mode: Annotated[Literal["neb", "ts-irc"], typer.Option(
            "--mode",
            help="Refinement mode: 'neb' for high-quality pairwise NEBs, or 'ts-irc' for TS optimization + IRC network refinement.",
            case_sensitive=False,
        )] = "neb",
        ts_irc_edge_scope: Annotated[Literal["best-path", "all-source-edges"], typer.Option(
            "--ts-irc-edge-scope",
            help="TS/IRC mode only: 'best-path' uses one inferred source path; 'all-source-edges' uses every edge available in the source workspace/network.",
            case_sensitive=False,
        )] = "all-source-edges",
        recycle_nodes: Annotated[bool, typer.Option(
            "--recycle-nodes",
            help="Reuse source path nodes as initial guess for expensive pair refinement.",
        )] = False,
        recursive: bool = False,
        parallel: Annotated[bool, typer.Option(
            "--parallel",
            help="Run recursive autosplitting in parallel with bounded worker concurrency.",
        )] = False,
        parallel_workers: Annotated[int | None, typer.Option(
            "--parallel-workers",
            help="Maximum number of concurrent recursive split workers used by --parallel. Defaults to min(4, CPU count).",
        )] = None,
        output_directory: Annotated[str, typer.Option(
            "--output-directory", "-o",
            help="TS/IRC mode only: directory for the refined workspace output.",
        )] = None,
        use_bigchem: Annotated[bool, typer.Option(
            "--use-bigchem/--no-use-bigchem",
            help="TS/IRC mode only: use BigChem for Hessian-backed TS/IRC steps when supported.",
        )] = False,
        write_status_html_output: Annotated[bool, typer.Option(
            "--write-status-html/--no-write-status-html",
            help="TS/IRC mode only: generate full status.html for the refined workspace (slower).",
        )] = False,
        irc_maxiter: Annotated[int | None, typer.Option(
            "--irc-maxiter",
            help="TS/IRC mode only: maximum IRC optimization steps per TS guess.",
        )] = None,
        ts_maxiter: Annotated[int | None, typer.Option(
            "--ts-maxiter",
            help="TS/IRC mode only: maximum TS optimization steps per TS guess.",
        )] = None,
        ts_keyword: Annotated[List[str], typer.Option(
            "--ts-keyword",
            help="TS/IRC mode only: override TS optimizer keyword as KEY=VALUE. Repeatable.",
        )] = [],
        name: str = None,
        charge: int = 0,
        multiplicity: int = 1):
    """Refine precomputed results using either NEB or TS/IRC workflows."""
    console.print(BANNER)

    if inputs is None:
        console.print(
            "[bold red]✗ ERROR:[/bold red] --inputs/-i is required for refine."
        )
        raise typer.Exit(1)
    if source is None:
        console.print(
            "[bold red]✗ ERROR:[/bold red] source path is required."
        )
        raise typer.Exit(1)

    mode_normalized = str(mode).strip().lower().replace("_", "-")
    if mode_normalized not in {"neb", "ts-irc"}:
        raise typer.BadParameter("--mode must be either 'neb' or 'ts-irc'.")
    if mode_normalized == "ts-irc":
        raise typer.BadParameter(
            "--mode ts-irc is not available in the public CLI; use --mode neb."
        )
    if irc_maxiter is not None and int(irc_maxiter) < 1:
        raise typer.BadParameter("--irc-maxiter must be at least 1.")
    if ts_maxiter is not None and int(ts_maxiter) < 1:
        raise typer.BadParameter("--ts-maxiter must be at least 1.")

    start_time = time.time()
    expensive_pair_grad_calls_total = 0
    expensive_pair_geom_grad_calls = 0
    base_name = _infer_refine_base_name(source, name)
    source_display = _source_display_text(source)
    refinement_inputs_path = Path(inputs).expanduser().resolve()

    if output_directory or use_bigchem or write_status_html_output or irc_maxiter is not None or ts_maxiter is not None or bool(ts_keyword) or ts_irc_edge_scope != "all-source-edges":
        console.print(
            "[yellow]⚠ --output-directory, --use-bigchem, --write-status-html, --irc-maxiter, --ts-maxiter, --ts-keyword, and --ts-irc-edge-scope apply only to --mode ts-irc and will be ignored.[/yellow]"
        )

    if parallel and recursive:
        raise typer.BadParameter(
            "--parallel cannot be combined with --recursive. Use one mode."
        )

    cpu_cap = max(1, int(os.cpu_count() or 1))
    default_parallel_workers = min(4, cpu_cap)
    if parallel_workers is None:
        parallel_workers = default_parallel_workers
    if parallel_workers < 1:
        raise typer.BadParameter("--parallel-workers must be at least 1.")
    if parallel_workers > cpu_cap:
        console.print(
            f"[yellow]⚠ Requested {parallel_workers} parallel workers exceeds detected CPU count ({cpu_cap}). Continuing with requested value; tune based on your host capacity.[/yellow]"
        )

    with console.status("[bold cyan]Loading source object and input parameters...[/bold cyan]"):
        source_obj = _load_precomputed_refine_source(
            source=source,
            charge=charge,
            multiplicity=multiplicity,
        )
        expensive_input = RunInputs.open(str(refinement_inputs_path))

    cheap_output_chain, cheap_minima, cheap_chain_inputs, source_kind = (
        _extract_refine_source_chain_and_minima(source_obj)
    )
    if len(cheap_minima) == 0:
        cheap_minima = _extract_minima_nodes_from_chain(cheap_output_chain)

    table = Table(box=None, show_header=False)
    table.add_column(style="dim")
    table.add_row("[bold cyan]Command:[/bold cyan]",
                  "[white]refine[/white]")
    table.add_row("[bold cyan]Mode:[/bold cyan]",
                  "[yellow]neb[/yellow]")
    table.add_row("[bold cyan]Source:[/bold cyan]",
                  f"[yellow]{source_display}[/yellow]")
    table.add_row("[bold cyan]Source Type:[/bold cyan]",
                  f"[yellow]{source_kind}[/yellow]")
    table.add_row("[bold cyan]Expensive Inputs:[/bold cyan]",
                  f"[yellow]{refinement_inputs_path}[/yellow]")
    mode_label = "parallel" if parallel else ("recursive" if recursive else "regular")
    table.add_row("[bold cyan]Method:[/bold cyan]",
                  f"[yellow]{mode_label}[/yellow]")
    table.add_row("[bold cyan]Parallel:[/bold cyan]",
                  f"[yellow]{parallel}[/yellow]")
    if parallel:
        table.add_row("[bold cyan]Parallel workers:[/bold cyan]",
                      f"[yellow]{parallel_workers}[/yellow]")
    table.add_row("[bold cyan]Recycle Nodes:[/bold cyan]",
                  f"[yellow]{recycle_nodes}[/yellow]")
    console.print(table)
    console.print()

    console.print("[bold cyan]Expensive-level Inputs[/bold cyan]")
    _render_runinputs(expensive_input)

    data_dir = Path(os.getcwd())
    cheap_chain_fp = data_dir / f"{base_name}_cheap.xyz"
    cheap_output_chain.write_to_disk(cheap_chain_fp)

    cheap_minima = _dedupe_minima_nodes(cheap_minima, cheap_chain_inputs)
    console.print(
        f"[cyan]Discovered {len(cheap_minima)} unique source minima (including endpoints).[/cyan]"
    )

    console.print(
        "[bold magenta]▶ REOPTIMIZING MINIMA AT EXPENSIVE LEVEL[/bold magenta]")
    refined_minima, refined_source_minima, dropped_count, kept_unoptimized_count = (
        _reoptimize_minima_for_refinement(
            cheap_minima,
            expensive_input,
            source_geometry_label="source",
        )
    )

    refined_minima, refined_source_minima = _dedupe_minima_and_sources(
        refined_minima, refined_source_minima, expensive_input.chain_inputs
    )
    if len(refined_minima) < 2:
        console.print(
            "[bold red]✗ ERROR:[/bold red] Fewer than 2 minima remain after expensive-level refinement."
        )
        raise typer.Exit(1)

    refined_minima_chain = Chain.model_validate(
        {"nodes": refined_minima, "parameters": expensive_input.chain_inputs}
    )
    refined_minima_fp = data_dir / f"{base_name}_refined_minima.xyz"
    refined_minima_chain.write_to_disk(refined_minima_fp)
    console.print(
        f"[cyan]Retained {len(refined_minima)} minima, dropped {dropped_count} due to connectivity changes, kept {kept_unoptimized_count} without expensive optimization.[/cyan]"
    )

    console.print(
        "[bold magenta]▶ EXPENSIVE PAIRWISE PATH MINIMIZATION[/bold magenta]")
    pair_dir = data_dir / f"{base_name}_refined_pairs"
    pair_dir.mkdir(exist_ok=True)
    expensive_msmep = MSMEP(inputs=expensive_input)
    if (recursive or parallel) and not expensive_input.path_min_inputs.do_elem_step_checks:
        console.print(
            "[yellow]⚠ WARNING: expensive do_elem_step_checks is False with --recursive/--parallel. Setting it to True.[/yellow]"
        )
        expensive_input.path_min_inputs.do_elem_step_checks = True

    pair_chains: list[Chain] = []
    pair_inds = list(zip(range(len(refined_minima) - 1),
                     range(1, len(refined_minima))))
    for i, j in pair_inds:
        endpoint_pair = Chain.model_validate(
            {"nodes": [refined_minima[i], refined_minima[j]],
                "parameters": expensive_input.chain_inputs}
        )
        pair = endpoint_pair
        if recycle_nodes:
            recycled_pair = _build_recycled_pair_chain(
                cheap_output_chain=cheap_output_chain,
                cheap_start_ref=refined_source_minima[i],
                cheap_end_ref=refined_source_minima[j],
                expensive_start=refined_minima[i],
                expensive_end=refined_minima[j],
                cheap_chain_inputs=cheap_chain_inputs,
                expensive_chain_inputs=expensive_input.chain_inputs,
                expected_nimages=expensive_input.gi_inputs.nimages,
            )
            if recycled_pair is not None:
                pair = recycled_pair
            else:
                console.print(
                    f"[yellow]⚠ Could not recycle source nodes for pair ({i}, {j}); using fresh interpolation.[/yellow]"
                )
        try:
            if recursive:
                pair_history = expensive_msmep.run_recursive_minimize(pair)
                if not pair_history.data:
                    console.print(
                        f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
                    )
                    continue
                out_chain = pair_history.output_chain
                pair_history.write_to_disk(pair_dir / f"pair_{i}_{j}_msmep")
                pair_grad_total, pair_grad_geom = _grad_calls_from_history(
                    pair_history
                )
                expensive_pair_grad_calls_total += pair_grad_total
                expensive_pair_geom_grad_calls += pair_grad_geom
            elif parallel:
                pair_history = expensive_msmep.run_parallel_recursive_minimize(
                    pair,
                    max_workers=parallel_workers,
                )
                if not pair_history.data:
                    console.print(
                        f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
                    )
                    continue
                out_chain = pair_history.output_chain
                pair_history.write_to_disk(pair_dir / f"pair_{i}_{j}_msmep")
                pair_grad_total, pair_grad_geom = _grad_calls_from_history(
                    pair_history
                )
                expensive_pair_grad_calls_total += pair_grad_total
                expensive_pair_geom_grad_calls += pair_grad_geom
            else:
                neb_obj, _ = expensive_msmep.run_minimize_chain(pair)
                out_chain = neb_obj.chain_trajectory[-1] if neb_obj.chain_trajectory else neb_obj.optimized
                expensive_pair_grad_calls_total += int(
                    getattr(neb_obj, "grad_calls_made", 0)
                )
                expensive_pair_geom_grad_calls += int(
                    getattr(neb_obj, "geom_grad_calls_made", 0)
                )
            if out_chain is None:
                console.print(
                    f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
                )
                continue
            pair_chains.append(out_chain)
            out_chain.write_to_disk(pair_dir / f"pair_{i}_{j}.xyz")
        except Exception:
            console.print(
                f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
            )
            continue

    if len(pair_chains) == 0:
        console.print(
            "[bold red]✗ ERROR:[/bold red] No expensive sequential pair paths converged."
        )
        raise typer.Exit(1)

    refined_chain = _concat_chains(
        pair_chains, expensive_input.chain_inputs)
    refined_chain_fp = data_dir / f"{base_name}_refined.xyz"
    refined_chain.write_to_disk(refined_chain_fp)

    elapsed = time.time() - start_time
    summary = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row(
        "⏱ Walltime:", f"[yellow]{elapsed/60:.1f} min[/yellow]" if elapsed > 60 else f"[yellow]{elapsed:.1f} s[/yellow]")
    summary.add_row("📁 Source chain:", f"[cyan]{cheap_chain_fp}[/cyan]")
    summary.add_row("📁 Refined minima:", f"[cyan]{refined_minima_fp}[/cyan]")
    summary.add_row("📁 Refined chain:", f"[cyan]{refined_chain_fp}[/cyan]")
    summary.add_row("🔗 Pair sequence:",
                    f"[white]{pair_inds}[/white]")
    console.print(Panel(
        summary, title="[bold green]✓ refine Complete![/bold green]", border_style="green"))
    cost_fp = _cost_report_path(data_dir, base_name)
    _write_cost_report(
        cost_fp,
        command="refine",
        mode=mode_label,
        elapsed_seconds=elapsed,
        gradient_calls_total=expensive_pair_grad_calls_total,
        gradient_calls_geometry_optimizations=expensive_pair_geom_grad_calls,
        metadata={
            "source_chain_path": str(cheap_chain_fp),
            "refined_minima_path": str(refined_minima_fp),
            "refined_chain_path": str(refined_chain_fp),
            "pair_count": int(len(pair_inds)),
            "recursive": bool(recursive),
            "parallel": bool(parallel),
        },
    )
    console.print(f"[dim]Cost report: {cost_fp}[/dim]")

    _ascii_profile_for_chain(refined_chain)


def _coerce_cli_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_key_value_options(options: list[str] | None, *, option_name: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in options or []:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid {option_name} value '{item}'. Use KEY=VALUE."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(
                f"Invalid {option_name} value '{item}'. Key cannot be empty."
            )
        parsed[key] = _coerce_cli_value(value.strip())
    return parsed


def _cache_irc_chain_energies(engine: Any, irc_chain: Chain) -> Chain:
    energies = np.asarray(engine.compute_energies(irc_chain), dtype=float).reshape(-1)
    if len(energies) != len(irc_chain.nodes):
        raise ValueError(
            "Engine returned a different number of energies than IRC chain nodes "
            f"({len(energies)} vs {len(irc_chain.nodes)})."
        )
    for node, energy in zip(irc_chain.nodes, energies):
        node._cached_energy = float(energy)
    return irc_chain


def _compute_native_irc_chain(
    *,
    ts_node: StructureNode,
    engine: Any,
    use_bigchem: bool = False,
    keywords: dict[str, Any] | None = None,
) -> Chain:
    irc_kwds = dict(keywords or {})
    engine.compute_energies([ts_node])
    if hasattr(engine, "compute_sd_irc"):
        irc_negative, irc_positive = engine.compute_sd_irc(
            ts=ts_node,
            use_bigchem=use_bigchem,
        )
        min_negative = engine.compute_geometry_optimization(
            irc_negative[-1],
            keywords=irc_kwds,
        )[-1]
        min_positive = engine.compute_geometry_optimization(
            irc_positive[-1],
            keywords=irc_kwds,
        )[-1]
        irc_negative.append(min_negative)
        irc_positive.append(min_positive)
        irc_negative.reverse()
        return Chain.model_validate({"nodes": irc_negative + [ts_node] + irc_positive})
    if hasattr(engine, "compute_irc_chain"):
        return engine.compute_irc_chain(ts_node=ts_node, keywords=irc_kwds)
    raise NotImplementedError(
        "Engine does not support native IRC: expected `compute_sd_irc` or `compute_irc_chain`."
    )


def _compute_irc_chain_for_program(
    *,
    ts_node: StructureNode,
    engine: Any,
    program: str,
    use_bigchem: bool = False,
    keywords: dict[str, Any] | None = None,
) -> Chain:
    normalized = str(program or "auto").strip().lower().replace("_", "-")
    if normalized in {"auto", "default"}:
        return compute_irc_chain(
            ts_node=ts_node,
            engine=engine,
            use_bigchem=use_bigchem,
            keywords=keywords,
        )
    if normalized in {"geometric", "geometric-optimize", "geometric-irc"}:
        raise typer.BadParameter(
            "The geomeTRIC IRC backend is not available in the public CLI; use --program native or --program auto."
        )
    if normalized in {"native", "engine"}:
        return _compute_native_irc_chain(
            ts_node=ts_node,
            engine=engine,
            use_bigchem=use_bigchem,
            keywords=keywords,
        )
    raise typer.BadParameter(
        f"Unsupported IRC program '{program}'. Supported values: auto, geometric, native."
    )


def _select_irc_ts_node(
    *,
    chain: Chain,
    engine: Any,
    ts_index: int | None = None,
) -> tuple[int, StructureNode]:
    if ts_index is not None:
        if ts_index < 0 or ts_index >= len(chain.nodes):
            raise typer.BadParameter(
                f"--ts-index must be between 0 and {len(chain.nodes) - 1}."
            )
        return ts_index, chain.nodes[ts_index]
    if len(chain.nodes) == 1:
        return 0, chain.nodes[0]
    if not chain._energies_already_computed:
        energies = np.asarray(engine.compute_energies(chain), dtype=float).reshape(-1)
        if len(energies) != len(chain.nodes):
            raise ValueError(
                "Engine returned a different number of energies than chain nodes "
                f"({len(energies)} vs {len(chain.nodes)})."
            )
        for node, energy in zip(chain.nodes, energies):
            node._cached_energy = float(energy)
    ts_index = int(np.asarray(chain.energies).argmax())
    return ts_index, chain.nodes[ts_index]


@app.command("run-irc")
def run_irc(
    geometries: Annotated[
        str,
        typer.Argument(
            help="XYZ/QCIO file containing a TS guess or a path. Defaults to mep_output.xyz.",
        ),
    ] = "mep_output.xyz",
    inputs: Annotated[
        str,
        typer.Option("--inputs", "-i", help="Path to RunInputs TOML file."),
    ] = None,
    program: Annotated[
        str,
        typer.Option(
            "--program",
            help="IRC backend to use. Supported values: auto, geometric, native.",
        ),
    ] = "auto",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output IRC XYZ path."),
    ] = None,
    ts_index: Annotated[
        int | None,
        typer.Option(
            "--ts-index",
            help="Index of the TS guess in the input path. Defaults to highest-energy node.",
        ),
    ] = None,
    charge: Annotated[int, typer.Option(help="Charge used when reading XYZ input.")] = 0,
    multiplicity: Annotated[
        int,
        typer.Option(help="Multiplicity used when reading XYZ input."),
    ] = 1,
    irc_keyword: Annotated[
        list[str] | None,
        typer.Option(
            "--irc-keyword",
            help="IRC backend keyword override as KEY=VALUE. Can be repeated.",
        ),
    ] = None,
    use_bigchem: Annotated[
        bool,
        typer.Option(
            help="Use BigChem for Hessian initialization when supported by the selected IRC backend.",
        ),
    ] = False,
):
    console.print(BANNER)
    input_fp = Path(geometries).expanduser().resolve()
    if not input_fp.exists():
        console.print(f"[bold red]✗ ERROR:[/bold red] IRC input not found: {input_fp}")
        raise typer.Exit(1)

    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        program_input = RunInputs.open(inputs) if inputs is not None else RunInputs(program="xtb", engine_name="qcop")
    _render_runinputs(program_input)

    try:
        try:
            chain = Chain.from_xyz(
                fp=input_fp,
                parameters=program_input.chain_inputs,
                charge=charge,
                spinmult=multiplicity,
            )
        except ValueError:
            chain = Chain.from_xyz(
                fp=input_fp,
                parameters=program_input.chain_inputs,
                charge=None,
                spinmult=None,
            )
    except Exception as exc:
        console.print(f"[bold red]✗ ERROR:[/bold red] Failed to load IRC input: {exc}")
        raise typer.Exit(1) from exc

    try:
        selected_index, ts_node = _select_irc_ts_node(
            chain=chain,
            engine=program_input.engine,
            ts_index=ts_index,
        )
    except Exception as exc:
        console.print(f"[bold red]✗ ERROR:[/bold red] Failed to select TS guess: {exc}")
        raise typer.Exit(1) from exc

    keywords = dict(getattr(program_input, "irc_kwds", {}) or {})
    keywords.update(_parse_key_value_options(irc_keyword, option_name="--irc-keyword"))

    output_fp = Path(output).expanduser().resolve() if output is not None else input_fp.with_name(f"{input_fp.stem}_irc.xyz")
    console.print(
        f"[bold magenta]▶ RUNNING IRC[/bold magenta] [dim](program={program}, ts_index={selected_index})[/dim]"
    )
    try:
        irc_chain = _compute_irc_chain_for_program(
            ts_node=ts_node,
            engine=program_input.engine,
            program=program,
            use_bigchem=use_bigchem,
            keywords=keywords,
        )
    except Exception:
        console.print(f"[bold red]✗ IRC failed:[/bold red]\n[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)

    write_qcio = bool(getattr(program_input, "write_qcio", False))
    irc_chain.write_to_disk(output_fp, write_qcio=write_qcio)
    console.print(f"[bold green]✓ IRC complete.[/bold green] Wrote [cyan]{output_fp}[/cyan]")
    _ascii_profile_for_chain(irc_chain)


@app.command("run-refine")
def run_refine(
        start: Annotated[str, typer.Option(
            help='path to start file, or a reactant smiles')] = None,
        end: Annotated[str, typer.Option(
            help='path to end file, or a product smiles')] = None,
        geometries:  Annotated[str, typer.Argument(help='file containing an approximate path between two endpoints. \
            Use this if you have precompted a path you want to use. Do not use this with smiles.')] = None,
        inputs: Annotated[str, typer.Option("--inputs", "-i",
                                            help='path to expensive RunInputs toml file')] = None,
        mode: Annotated[Literal["neb", "ts-irc"], typer.Option(
            "--mode",
            help="Refinement mode: 'neb' for high-quality pairwise NEBs, or 'ts-irc' for TS optimization + IRC network refinement.",
            case_sensitive=False,
        )] = "neb",
        ts_irc_edge_scope: Annotated[Literal["best-path", "all-source-edges"], typer.Option(
            "--ts-irc-edge-scope",
            help="TS/IRC mode only: 'best-path' refines the discovered best path; 'all-source-edges' keeps all available source edges for TS/IRC.",
            case_sensitive=False,
        )] = "best-path",
        cheap_inputs: Annotated[str, typer.Option("--cheap-inputs", "-ci",
                                                  help='optional path to cheaper RunInputs toml file for initial discovery')] = None,
        recycle_nodes: Annotated[bool, typer.Option(
            "--recycle-nodes",
            help="Reuse cheap-stage path nodes as initial guess for expensive pair refinement.",
        )] = False,
        network_completion: Annotated[bool, typer.Option(
            "--network-completion",
            help="For recursive cheap discovery, run follow-on completion requests and refine only the best path through the resulting network.",
        )] = False,
        network_completion_mode: Annotated[NetworkCompletionMode, typer.Option(
            "--network-completion-mode",
            help="Follow-on completion request strategy: 'linear' connects every discovered intermediate to the original reactant/product endpoints; 'all-to-all' attempts every non-adjacent pair on each discovered path.",
            case_sensitive=False,
        )] = "linear",
        use_smiles: bool = False,
        recursive: bool = False,
        minimize_ends: bool = False,
        name: str = None,
        charge: int = 0,
        multiplicity: int = 1,
        output_directory: Annotated[str, typer.Option(
            "--output-directory", "-o",
            help="TS/IRC mode only: directory for the refined workspace output.",
        )] = None,
        use_bigchem: Annotated[bool, typer.Option(
            "--use-bigchem/--no-use-bigchem",
            help="TS/IRC mode only: use BigChem for Hessian-backed TS/IRC steps when supported.",
        )] = False,
        write_status_html_output: Annotated[bool, typer.Option(
            "--write-status-html/--no-write-status-html",
            help="TS/IRC mode only: generate full status.html for the refined workspace (slower).",
        )] = False,
        irc_maxiter: Annotated[int | None, typer.Option(
            "--irc-maxiter",
            help="TS/IRC mode only: maximum IRC optimization steps per TS guess.",
        )] = None,
        ts_maxiter: Annotated[int | None, typer.Option(
            "--ts-maxiter",
            help="TS/IRC mode only: maximum TS optimization steps per TS guess.",
        )] = None,
        ts_keyword: Annotated[List[str], typer.Option(
            "--ts-keyword",
            help="TS/IRC mode only: override TS optimizer keyword as KEY=VALUE. Repeatable.",
        )] = []):
    """Two-stage refinement: cheap discovery -> configurable expensive refinement."""
    console.print(BANNER)

    if inputs is None:
        console.print(
            "[bold red]✗ ERROR:[/bold red] --inputs/-i is required for run-refine."
        )
        raise typer.Exit(1)

    mode_normalized = str(mode).strip().lower().replace("_", "-")
    if mode_normalized not in {"neb", "ts-irc"}:
        raise typer.BadParameter("--mode must be either 'neb' or 'ts-irc'.")
    if mode_normalized == "ts-irc":
        raise typer.BadParameter(
            "--mode ts-irc is not available in the public CLI; use --mode neb."
        )
    if irc_maxiter is not None and int(irc_maxiter) < 1:
        raise typer.BadParameter("--irc-maxiter must be at least 1.")
    if ts_maxiter is not None and int(ts_maxiter) < 1:
        raise typer.BadParameter("--ts-maxiter must be at least 1.")

    if network_completion and not recursive:
        recursive = True

    if mode_normalized == "ts-irc" and recycle_nodes:
        console.print(
            "[yellow]⚠ --recycle-nodes applies only to --mode neb and will be ignored.[/yellow]"
        )
    if mode_normalized == "neb" and (output_directory or use_bigchem or write_status_html_output or irc_maxiter is not None or ts_maxiter is not None or bool(ts_keyword) or ts_irc_edge_scope != "best-path"):
        console.print(
            "[yellow]⚠ --output-directory, --use-bigchem, --write-status-html, --irc-maxiter, --ts-maxiter, --ts-keyword, and --ts-irc-edge-scope apply only to --mode ts-irc and will be ignored.[/yellow]"
        )

    start_time = time.time()
    cheap_grad_calls_total = 0
    cheap_geom_grad_calls = 0
    expensive_pair_grad_calls_total = 0
    expensive_pair_geom_grad_calls = 0
    table = Table(box=None, show_header=False)
    table.add_column(style="dim")
    table.add_row("[bold cyan]Command:[/bold cyan]",
                  "[white]run-refine[/white]")
    table.add_row("[bold cyan]Mode:[/bold cyan]",
                  f"[yellow]{mode_normalized}[/yellow]")
    table.add_row("[bold cyan]SMILES mode:[/bold cyan]",
                  f"[yellow]{use_smiles}[/yellow]")
    table.add_row("[bold cyan]Method:[/bold cyan]",
                  f"[yellow]{'recursive' if recursive else 'regular'}[/yellow]")
    table.add_row("[bold cyan]Network completion:[/bold cyan]",
                  f"[yellow]{network_completion}[/yellow]")
    if network_completion:
        table.add_row("[bold cyan]Network completion mode:[/bold cyan]",
                      f"[yellow]{network_completion_mode}[/yellow]")
    table.add_row("[bold cyan]Cheap Inputs:[/bold cyan]",
                  f"[yellow]{cheap_inputs if cheap_inputs else inputs}[/yellow]")
    table.add_row("[bold cyan]Expensive Inputs:[/bold cyan]",
                  f"[yellow]{inputs}[/yellow]")
    table.add_row("[bold cyan]Recycle Nodes:[/bold cyan]",
                  f"[yellow]{recycle_nodes}[/yellow]")
    if mode_normalized == "ts-irc":
        table.add_row("[bold cyan]TS/IRC Edge Scope:[/bold cyan]",
                      f"[yellow]{ts_irc_edge_scope}[/yellow]")
        if output_directory:
            table.add_row("[bold cyan]Output Directory:[/bold cyan]",
                          f"[yellow]{Path(output_directory).expanduser().resolve()}[/yellow]")
        table.add_row("[bold cyan]Use BigChem:[/bold cyan]",
                      f"[yellow]{use_bigchem}[/yellow]")
        table.add_row("[bold cyan]Write status.html:[/bold cyan]",
                      f"[yellow]{write_status_html_output}[/yellow]")
        if irc_maxiter is not None:
            table.add_row("[bold cyan]IRC maxiter:[/bold cyan]",
                          f"[yellow]{int(irc_maxiter)}[/yellow]")
        if ts_maxiter is not None:
            table.add_row("[bold cyan]TS maxiter:[/bold cyan]",
                          f"[yellow]{int(ts_maxiter)}[/yellow]")
        if ts_keyword:
            table.add_row("[bold cyan]TS keyword overrides:[/bold cyan]",
                          f"[yellow]{len(ts_keyword)}[/yellow]")
    console.print(table)
    console.print()

    # load structures
    if use_smiles:
        from mepd.nodes.nodehelpers import create_pairs_from_smiles
        from mepd.arbalign import align_structures

        console.print(
            "[yellow]⚠ WARNING:[/yellow] Using RXNMapper to create atomic mapping. Carefully check output to see how labels affected reaction path.")
        with console.status("[bold cyan]Creating structures from SMILES...[/bold cyan]"):
            start_structure, end_structure = create_pairs_from_smiles(
                smi1=start, smi2=end)
            end_structure = align_structures(
                start_structure, end_structure, distance_metric='RMSD')
        all_structs = [start_structure, end_structure]
    else:
        if geometries is not None:
            with console.status(f"[bold cyan]Loading geometries from {geometries}...[/bold cyan]"):
                try:
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=charge, spinmult=multiplicity)
                except ValueError:
                    all_structs = read_multiple_structure_from_file(
                        geometries, charge=None, spinmult=None)
        elif start is not None and end is not None:
            with console.status(f"[bold cyan]Loading structures...[/bold cyan]"):
                start_ref = _load_endpoint_structure(
                    start,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                end_ref = _load_endpoint_structure(
                    end,
                    charge=charge,
                    multiplicity=multiplicity,
                )
                if start_ref.charge != charge or start_ref.multiplicity != multiplicity:
                    start_ref = Structure(
                        geometry=start_ref.geometry,
                        charge=charge,
                        multiplicity=multiplicity,
                        symbols=start_ref.symbols,
                    )
                if end_ref.charge != charge or end_ref.multiplicity != multiplicity:
                    end_ref = Structure(
                        geometry=end_ref.geometry,
                        charge=charge,
                        multiplicity=multiplicity,
                        symbols=end_ref.symbols,
                    )
                all_structs = [start_ref, end_ref]
        else:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Either 'geometries' or 'start' and 'end' flags must be populated!")
            raise typer.Exit(1)

    with console.status("[bold cyan]Loading input parameters...[/bold cyan]"):
        expensive_input = RunInputs.open(inputs)
        cheap_input = RunInputs.open(
            cheap_inputs) if cheap_inputs else RunInputs.open(inputs)

    console.print("[bold cyan]Cheap-level Inputs[/bold cyan]")
    _render_runinputs(cheap_input)
    console.print("[bold cyan]Expensive-level Inputs[/bold cyan]")
    _render_runinputs(expensive_input)

    all_nodes = [StructureNode(structure=s) for s in all_structs]
    if minimize_ends:
        console.print(
            "[bold cyan]⟳ Minimizing input endpoints at cheap level...[/bold cyan]")
        start_tr = cheap_input.engine.compute_geometry_optimization(
            all_nodes[0], keywords={'coordsys': 'cart', 'maxiter': 500})
        all_nodes[0] = start_tr[-1]
        end_tr = cheap_input.engine.compute_geometry_optimization(
            all_nodes[-1], keywords={'coordsys': 'cart', 'maxiter': 500})
        all_nodes[-1] = end_tr[-1]

    cheap_chain = Chain.model_validate(
        {"nodes": all_nodes, "parameters": cheap_input.chain_inputs}
    )
    cheap_msmep = MSMEP(inputs=cheap_input)

    console.print(
        f"[bold magenta]▶ CHEAP DISCOVERY RUN ({cheap_input.path_min_method})[/bold magenta]"
    )
    if recursive:
        if not cheap_input.path_min_inputs.do_elem_step_checks:
            console.print(
                "[yellow]⚠ WARNING: do_elem_step_checks is False with --recursive. Setting it to True.[/yellow]"
            )
            cheap_input.path_min_inputs.do_elem_step_checks = True
        cheap_history = cheap_msmep.run_recursive_minimize(cheap_chain)
        if not cheap_history.data:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Cheap run returned no valid history."
            )
            raise typer.Exit(1)
        cheap_output_chain = cheap_history.output_chain
        cheap_minima = _extract_minima_nodes(cheap_history)
        cheap_grad_total, cheap_grad_geom = _grad_calls_from_history(cheap_history)
        cheap_grad_calls_total += cheap_grad_total
        cheap_geom_grad_calls += cheap_grad_geom
    else:
        cheap_neb, _ = cheap_msmep.run_minimize_chain(cheap_chain)
        cheap_output_chain = cheap_neb.chain_trajectory[-1] if cheap_neb.chain_trajectory else cheap_neb.optimized
        cheap_grad_calls_total += int(getattr(cheap_neb, "grad_calls_made", 0))
        cheap_geom_grad_calls += int(
            getattr(cheap_neb, "geom_grad_calls_made", 0)
        )
        if cheap_output_chain is None:
            console.print(
                "[bold red]✗ ERROR:[/bold red] Cheap run produced no optimized chain."
            )
            raise typer.Exit(1)
        cheap_history = None
        cheap_minima = _extract_minima_nodes_from_chain(cheap_output_chain)

    base_name = name if name is not None else "mep_output"
    data_dir = Path(os.getcwd())
    cheap_chain_fp = data_dir / f"{base_name}_cheap.xyz"
    cheap_tree_dir = data_dir / f"{base_name}_cheap_msmep"
    cheap_output_chain.write_to_disk(cheap_chain_fp)
    if cheap_history is not None:
        cheap_history.write_to_disk(cheap_tree_dir)

    network_completion_pot: Pot | None = None
    if recursive and network_completion and cheap_history is not None:
        network_dir = data_dir / f"{base_name}_cheap_network_completion"
        console.print(
            "[bold magenta]▶ CHEAP DISCOVERY NETWORK SPLITS[/bold magenta]"
        )
        _request_records, network_fp, _manifest_fp, _network_cost_summary = _run_recursive_network_completion(
            history=cheap_history,
            program_input=cheap_input,
            initial_start=cheap_chain[0],
            initial_end=cheap_chain[-1],
            output_dir=network_dir,
            base_name=f"{base_name}_cheap",
            status_fp=None,
            source_tree_dir=cheap_tree_dir,
            split_mode=network_completion_mode,
        )
        if mode_normalized == "ts-irc" and ts_irc_edge_scope == "all-source-edges":
            with contextlib.suppress(Exception):
                network_completion_pot = Pot.read_from_disk(network_fp)
            if network_completion_pot is not None:
                console.print(
                    "[cyan]Using full network-completion graph as TS/IRC source "
                    f"({network_completion_pot.graph.number_of_nodes()} nodes, "
                    f"{network_completion_pot.graph.number_of_edges()} edges).[/cyan]"
                )

        if network_completion_pot is None:
            best_path_chain = _load_best_path_chain_from_network_completion(
                network_fp=network_fp,
                output_dir=network_dir,
                base_name=f"{base_name}_cheap",
            )
            if best_path_chain is None:
                console.print(
                    "[bold red]✗ ERROR:[/bold red] Network completion did not produce a best path for refinement."
                )
                raise typer.Exit(1)
            cheap_output_chain = best_path_chain
            cheap_minima = [node.copy() for node in best_path_chain.nodes]
            console.print(
                f"[cyan]Using best network path with {len(cheap_minima)} nodes for expensive refinement.[/cyan]"
            )

    cheap_minima = _dedupe_minima_nodes(cheap_minima, cheap_input.chain_inputs)
    console.print(
        f"[cyan]Discovered {len(cheap_minima)} unique cheap minima (including endpoints).[/cyan]"
    )

    console.print(
        "[bold magenta]▶ REOPTIMIZING MINIMA AT EXPENSIVE LEVEL[/bold magenta]")
    refined_minima, refined_source_minima, dropped_count, kept_unoptimized_count = (
        _reoptimize_minima_for_refinement(
            cheap_minima,
            expensive_input,
            source_geometry_label="cheap-level",
        )
    )

    refined_minima, refined_source_minima = _dedupe_minima_and_sources(
        refined_minima, refined_source_minima, expensive_input.chain_inputs
    )
    if len(refined_minima) < 2:
        console.print(
            "[bold red]✗ ERROR:[/bold red] Fewer than 2 minima remain after expensive-level refinement."
        )
        raise typer.Exit(1)

    refined_minima_chain = Chain.model_validate(
        {"nodes": refined_minima, "parameters": expensive_input.chain_inputs}
    )
    refined_minima_fp = data_dir / f"{base_name}_refined_minima.xyz"
    refined_minima_chain.write_to_disk(refined_minima_fp)
    console.print(
        f"[cyan]Retained {len(refined_minima)} minima, dropped {dropped_count} due to connectivity changes, kept {kept_unoptimized_count} without expensive optimization.[/cyan]"
    )

    console.print(
        "[bold magenta]▶ EXPENSIVE PAIRWISE PATH MINIMIZATION[/bold magenta]")
    pair_dir = data_dir / f"{base_name}_refined_pairs"
    pair_dir.mkdir(exist_ok=True)
    expensive_msmep = MSMEP(inputs=expensive_input)
    if recursive and not expensive_input.path_min_inputs.do_elem_step_checks:
        console.print(
            "[yellow]⚠ WARNING: expensive do_elem_step_checks is False with --recursive. Setting it to True.[/yellow]"
        )
        expensive_input.path_min_inputs.do_elem_step_checks = True

    pair_chains: list[Chain] = []
    pair_inds = list(zip(range(len(refined_minima) - 1),
                     range(1, len(refined_minima))))
    for i, j in pair_inds:
        endpoint_pair = Chain.model_validate(
            {"nodes": [refined_minima[i], refined_minima[j]],
                "parameters": expensive_input.chain_inputs}
        )
        pair = endpoint_pair
        if recycle_nodes:
            recycled_pair = _build_recycled_pair_chain(
                cheap_output_chain=cheap_output_chain,
                cheap_start_ref=refined_source_minima[i],
                cheap_end_ref=refined_source_minima[j],
                expensive_start=refined_minima[i],
                expensive_end=refined_minima[j],
                cheap_chain_inputs=cheap_input.chain_inputs,
                expensive_chain_inputs=expensive_input.chain_inputs,
                expected_nimages=expensive_input.gi_inputs.nimages,
            )
            if recycled_pair is not None:
                pair = recycled_pair
            else:
                console.print(
                    f"[yellow]⚠ Could not recycle cheap nodes for pair ({i}, {j}); using fresh interpolation.[/yellow]"
                )
        try:
            if recursive:
                pair_history = expensive_msmep.run_recursive_minimize(pair)
                if not pair_history.data:
                    console.print(
                        f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
                    )
                    continue
                out_chain = pair_history.output_chain
                pair_history.write_to_disk(pair_dir / f"pair_{i}_{j}_msmep")
                pair_grad_total, pair_grad_geom = _grad_calls_from_history(
                    pair_history
                )
                expensive_pair_grad_calls_total += pair_grad_total
                expensive_pair_geom_grad_calls += pair_grad_geom
            else:
                neb_obj, _ = expensive_msmep.run_minimize_chain(pair)
                out_chain = neb_obj.chain_trajectory[-1] if neb_obj.chain_trajectory else neb_obj.optimized
                expensive_pair_grad_calls_total += int(
                    getattr(neb_obj, "grad_calls_made", 0)
                )
                expensive_pair_geom_grad_calls += int(
                    getattr(neb_obj, "geom_grad_calls_made", 0)
                )
            if out_chain is None:
                console.print(
                    f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
                )
                continue
            pair_chains.append(out_chain)
            out_chain.write_to_disk(pair_dir / f"pair_{i}_{j}.xyz")
        except Exception:
            console.print(
                f"[yellow]⚠ Pair ({i}, {j}) failed at expensive level; skipping.[/yellow]"
            )
            continue

    if len(pair_chains) == 0:
        console.print(
            "[bold red]✗ ERROR:[/bold red] No expensive sequential pair paths converged."
        )
        raise typer.Exit(1)

    refined_chain = _concat_chains(
        pair_chains, expensive_input.chain_inputs)
    refined_chain_fp = data_dir / f"{base_name}_refined.xyz"
    refined_chain.write_to_disk(refined_chain_fp)

    elapsed = time.time() - start_time
    summary = Table(box=box.ROUNDED, border_style="green", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row(
        "⏱ Walltime:", f"[yellow]{elapsed/60:.1f} min[/yellow]" if elapsed > 60 else f"[yellow]{elapsed:.1f} s[/yellow]")
    summary.add_row("📁 Cheap chain:", f"[cyan]{cheap_chain_fp}[/cyan]")
    summary.add_row("📁 Refined minima:", f"[cyan]{refined_minima_fp}[/cyan]")
    summary.add_row("📁 Refined chain:", f"[cyan]{refined_chain_fp}[/cyan]")
    summary.add_row("🔗 Pair sequence:",
                    f"[white]{pair_inds}[/white]")
    console.print(Panel(
        summary, title="[bold green]✓ run-refine Complete![/bold green]", border_style="green"))
    cost_fp = _cost_report_path(data_dir, base_name)
    _write_cost_report(
        cost_fp,
        command="run-refine",
        mode="recursive" if recursive else "regular",
        elapsed_seconds=elapsed,
        gradient_calls_total=cheap_grad_calls_total + expensive_pair_grad_calls_total,
        gradient_calls_geometry_optimizations=(
            cheap_geom_grad_calls + expensive_pair_geom_grad_calls
        ),
        metadata={
            "cheap_chain_path": str(cheap_chain_fp),
            "refined_minima_path": str(refined_minima_fp),
            "refined_chain_path": str(refined_chain_fp),
            "pair_count": int(len(pair_inds)),
            "network_completion": bool(network_completion),
            "cheap_gradient_calls_total": int(cheap_grad_calls_total),
            "expensive_gradient_calls_total": int(expensive_pair_grad_calls_total),
        },
    )
    console.print(f"[dim]Cost report: {cost_fp}[/dim]")

    _ascii_profile_for_chain(refined_chain)


@app.command("ts")
def ts(
    geometry: Annotated[str, typer.Argument(help='path to geometry file to optimize')],
    inputs: Annotated[str, typer.Option("--inputs", "-i",
                                        help='path to RunInputs toml file')] = None,
    name: str = None,
    charge: int = 0,
    multiplicity: int = 1,
    bigchem: bool = False
):
    console.print(BANNER)

    # create output names
    fp = Path(geometry)
    data_dir = Path(os.getcwd())

    if name is not None:
        base_name = name
    else:
        base_name = fp.stem
    results_name = data_dir / f"{base_name}.qcio"
    filename = data_dir / f"{base_name}_optimized.xyz"

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')

    with console.status(f"[bold cyan]Optimizing transition state: {geometry}...[/bold cyan]") as status:
        sys.stdout.flush()
        try:
            struct = Structure.open(geometry)
            s_dict = struct.model_dump()
            s_dict["charge"], s_dict["multiplicity"] = charge, multiplicity
            struct = Structure(**s_dict)

            node = StructureNode(structure=struct)
            ts_node, output = _compute_ts_node(
                engine=program_input.engine,
                ts_guess=node,
                bigchem=bigchem,
            )

        except Exception:
            console.print(
                f"[bold red]✗ TS optimization failed:[/bold red] {traceback.format_exc()}"
            )
            raise typer.Exit(1)

    if output is not None and hasattr(output, "save"):
        output.save(results_name)
        console.print(f"[dim]Results: {results_name}[/dim]")
    if ts_node is None:
        console.print(
            "[bold red]✗ TS optimization did not converge.[/bold red]")
        raise typer.Exit(1)
    ts_node.structure.save(filename)
    console.print(f"[bold green]✓ TS optimization complete![/bold green]")
    console.print(f"[dim]Geometry: {filename}[/dim]")

@app.command("pseuirc")
def pseuirc(geometry: Annotated[str, typer.Argument(help='path to geometry file to optimize')],
            inputs: Annotated[str, typer.Option("--inputs", "-i",
                                                help='path to RunInputs toml file')] = None,
            name: str = None,
            charge: int = 0,
            multiplicity: int = 1,
            dr: float = 1.0):
    console.print(BANNER)

    # create output names
    fp = Path(geometry)
    data_dir = Path(os.getcwd())

    if name is not None:
        results_name = data_dir / (name + ".qcio")
    else:
        results_name = Path(fp.stem + ".qcio")

    # load the RunInputs
    if inputs is not None:
        program_input = RunInputs.open(inputs)
    else:
        program_input = RunInputs(program='xtb', engine_name='qcop')

    with console.status(f"[bold cyan]Computing hessian...[/bold cyan]"):
        sys.stdout.flush()
        try:
            struct = Structure.open(geometry)
            s_dict = struct.model_dump()
            s_dict["charge"], s_dict["multiplicity"] = charge, multiplicity
            struct = Structure(**s_dict)

            node = StructureNode(structure=struct)
            hessres = _compute_hessian_result_for_sampling(
                program_input.engine, node)

        except Exception as e:
            hessres = e.program_output

    hessres.save(results_name.parent / (results_name.stem+"_hessian.qcio"))

    console.print("[bold cyan]⟳ Minimizing TS(-)...[/bold cyan]")
    sys.stdout.flush()
    tsminus_raw = displace_by_dr(
        node=node, displacement=hessres.results.normal_modes_cartesian[0], dr=-dr)
    tsminus_res = program_input.engine._compute_geom_opt_result(
        tsminus_raw)
    tsminus_res.save(results_name.parent / (results_name.stem+"_minus.qcio"))

    console.print("[bold cyan]⟳ Minimizing TS(+)...[/bold cyan]")
    sys.stdout.flush()
    tsplus_raw = displace_by_dr(
        node=node, displacement=hessres.results.normal_modes_cartesian[0], dr=dr)
    tsplus_res = program_input.engine._compute_geom_opt_result(
        tsplus_raw)

    tsplus_res.save(results_name.parent / (results_name.stem+"_plus.qcio"))
    console.print(f"[bold green]✓ Pseudo-IRC complete![/bold green]")


@app.command("status")
def status(
    path: Annotated[str, typer.Argument(help="Path to a run artifact, .xyz output, status JSON, or *_network_completion directory")],
):
    console.print(BANNER)
    snapshot = _load_status_snapshot(path)
    run_status = snapshot.get("run_status") or {}
    manifest = snapshot.get("manifest") or {}
    root_info = run_status or manifest

    summary = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Artifact", snapshot["artifact_path"])
    summary.add_row("Base name", str(root_info.get("base_name", "unknown")))
    summary.add_row("Run state", str(root_info.get("run_state", "unknown")))
    if run_status:
        summary.add_row("Phase", str(run_status.get("phase", "unknown")))
        if "recursive" in run_status:
            summary.add_row("Recursive", str(run_status.get("recursive")))
        if "parallel" in run_status:
            summary.add_row("Parallel", str(run_status.get("parallel")))
        if "network_completion" in run_status:
            summary.add_row("Network completion", str(
                run_status.get("network_completion")))
        if "path_min_method" in run_status:
            summary.add_row("Path method", str(
                run_status.get("path_min_method")))
    console.print(
        Panel(summary, title="[bold cyan]MSMEP Status[/bold cyan]", border_style="cyan"))

    if manifest:
        counts = manifest.get("counts", {})
        counts_line = ", ".join(
            f"{key}={counts[key]}" for key in sorted(counts)) if counts else "none"
        console.print(
            f"[cyan]Requests:[/cyan] total={manifest.get('total_requests', 0)} [{counts_line}]")
        current_request_id = manifest.get("current_request_id")
        if current_request_id is not None:
            console.print(
                f"[yellow]Currently running request:[/yellow] {current_request_id}")

        request_table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
        request_table.add_column("ID", style="bold cyan", justify="right")
        request_table.add_column("Parent", style="dim", justify="right")
        request_table.add_column("Pair", style="magenta")
        request_table.add_column("Status", style="white")
        request_table.add_column("Path Nodes", style="white", justify="right")
        for record in manifest.get("requests", []):
            request_table.add_row(
                str(record.get("request_id", "")),
                "" if record.get("parent_request_id") is None else str(
                    record.get("parent_request_id")),
                f"{record.get('start_index', '?')} -> {record.get('end_index', '?')}",
                str(record.get("status", "")),
                "" if record.get("n_path_nodes") is None else str(
                    record.get("n_path_nodes")),
            )
        console.print(request_table)

        network_summary = manifest.get(
            "network_summary") or run_status.get("network_summary")
        if network_summary:
            network_table = Table(
                box=box.SIMPLE, show_header=True, pad_edge=False)
            network_table.add_column("Nodes", style="bold cyan")
            network_table.add_column("Edges", style="bold cyan")
            network_table.add_row(
                str(network_summary.get("node_count", 0)),
                str(network_summary.get("edge_count", 0)),
            )
            console.print(Panel(
                network_table, title="[bold cyan]Current Network[/bold cyan]", border_style="cyan"))
            edges = network_summary.get("edges") or []
            if edges:
                edge_text = ", ".join(f"{a}->{b}" for a, b in edges[:20])
                if len(edges) > 20:
                    edge_text += ", ..."
                console.print(f"[dim]{edge_text}[/dim]")


@app.command("backfill-cost")
def backfill_cost(
    path: Annotated[str, typer.Argument(help="Path to a run artifact, refine output, or directory to scan")],
    scan: Annotated[bool, typer.Option(
        "--scan",
        help="Recursively scan directory for recoverable runs/refinements and write missing *_cost.json files.",
    )] = False,
    overwrite: Annotated[bool, typer.Option(
        "--overwrite",
        help="Overwrite existing *_cost.json files.",
    )] = False,
):
    console.print(BANNER)
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise typer.BadParameter(f"Path does not exist: {root}")

    candidates: list[dict[str, Any]] = []
    seen_candidates: set[tuple[str, str]] = set()

    def _add_candidate(kind: str, candidate_path: Path) -> None:
        resolved = candidate_path.resolve()
        key = (kind, str(resolved))
        if key in seen_candidates:
            return
        seen_candidates.add(key)
        candidates.append({"kind": kind, "path": resolved})

    if scan:
        if not root.is_dir():
            raise typer.BadParameter("--scan requires a directory path.")
        for status_fp in sorted(root.rglob("*_status.json")):
            _add_candidate("status", status_fp)
        for refined_fp in sorted(root.rglob("*_refined.xyz")):
            _add_candidate("refined_xyz", refined_fp)
        for workspace_fp in sorted(root.rglob("workspace.json")):
            workspace_dir = workspace_fp.parent
            if (workspace_dir / "ts_irc_refinement").is_dir():
                _add_candidate("workspace", workspace_dir)
        for network_dir in sorted(root.rglob("*_network_completion")):
            if not network_dir.is_dir():
                continue
            stem = network_dir.name[: -len("_network_completion")]
            if not (network_dir / f"{stem}_request_manifest.json").exists():
                continue
            if (network_dir.parent / f"{stem}_status.json").exists():
                continue
            _add_candidate("network_completion", network_dir)
        for tree_dir in sorted(root.rglob("*_msmep")):
            if not _is_msmep_tree_dir(tree_dir):
                continue
            if (tree_dir.parent / f"{tree_dir.name}_status.json").exists():
                continue
            _add_candidate("msmep_tree", tree_dir)
        for history_dir in sorted(root.rglob("*_history")):
            if not _is_neb_history_dir(history_dir):
                continue
            if history_dir.name.startswith("node_"):
                continue
            base_name = history_dir.name[: -len("_history")]
            if (history_dir.parent / f"{base_name}_status.json").exists():
                continue
            _add_candidate("neb_history", history_dir)
    else:
        _add_candidate("auto", root)

    created = 0
    skipped = 0
    failed = 0
    result_table = Table(box=box.SIMPLE, show_header=True, pad_edge=False)
    result_table.add_column("Target", style="cyan")
    result_table.add_column("Cost File", style="white")
    result_table.add_column("Status", style="white")
    result_table.add_column("Notes", style="dim")

    for candidate in candidates:
        kind = str(candidate["kind"])
        candidate_path = Path(candidate["path"]).resolve()
        try:
            base_name: str | None = None
            out_dir: Path | None = None
            command_name = "unknown"
            mode_name: str | None = None
            metadata: dict[str, Any] = {}
            related_paths: list[Path] = [candidate_path]
            gradient_calls_total: int | None = None

            snapshot = (
                _try_load_status_snapshot(candidate_path)
                if kind in {"auto", "status", "network_completion"}
                else None
            )

            if kind == "status" or kind == "network_completion" or (kind == "auto" and snapshot is not None):
                if snapshot is None:
                    raise ValueError(
                        f"Could not load run status/manifest from: {candidate_path}"
                    )
                run_status = snapshot.get("run_status") or {}
                manifest = snapshot.get("manifest") or {}
                if not run_status and not manifest:
                    raise ValueError("No run_status or request_manifest payload found.")
                artifact_path = Path(str(snapshot.get("artifact_path") or candidate_path))

                inferred_base_name = (
                    run_status.get("base_name")
                    or manifest.get("base_name")
                )
                if inferred_base_name:
                    base_name = str(inferred_base_name)
                elif artifact_path.name.endswith("_status.json"):
                    base_name = artifact_path.stem.replace("_status", "")
                elif artifact_path.name.endswith("_request_manifest.json"):
                    base_name = artifact_path.stem.replace("_request_manifest", "")
                elif candidate_path.is_dir() and candidate_path.name.endswith("_network_completion"):
                    base_name = candidate_path.name[: -len("_network_completion")]
                else:
                    base_name = artifact_path.stem

                if run_status:
                    out_dir = artifact_path.parent
                elif artifact_path.parent.name.endswith("_network_completion"):
                    out_dir = artifact_path.parent.parent
                else:
                    out_dir = artifact_path.parent

                command_name = "run"
                recursive = bool(run_status.get("recursive", False))
                parallel = bool(run_status.get("parallel", False))
                if run_status:
                    mode_name = "parallel" if parallel else ("recursive" if recursive else "regular")
                else:
                    mode_name = "recursive_network_completion"

                related_paths.append(artifact_path)
                if run_status.get("output_chain_path"):
                    related_paths.append(Path(str(run_status["output_chain_path"])))
                if run_status.get("tree_path"):
                    tree_dir = Path(str(run_status["tree_path"]))
                    related_paths.append(tree_dir)
                    est = _estimate_grad_calls_from_tree_folder(tree_dir)
                    if est > 0:
                        gradient_calls_total = int(est)
                        metadata["gradient_calls_source"] = "estimated_from_tree_history"
                        metadata["gradient_calls_assumption"] = (
                            "per_neb_step=max(0,n_nodes-2); assumes endpoints frozen and no extra frozen interior nodes"
                        )
                if run_status.get("manifest_path"):
                    related_paths.append(Path(str(run_status["manifest_path"])))
                if run_status.get("network_path"):
                    related_paths.append(Path(str(run_status["network_path"])))
                if manifest.get("network_path"):
                    related_paths.append(Path(str(manifest["network_path"])))
                network_completion_dir = run_status.get("network_completion_dir")
                if network_completion_dir:
                    network_dir = Path(str(network_completion_dir))
                    related_paths.append(network_dir)
                    est = _estimate_grad_calls_from_network_completion_dir(network_dir)
                    if est > 0 and gradient_calls_total is None:
                        gradient_calls_total = int(est)
                        metadata["gradient_calls_source"] = "estimated_from_network_completion_histories"
                        metadata["gradient_calls_assumption"] = (
                            "per_neb_step=max(0,n_nodes-2); assumes endpoints frozen and no extra frozen interior nodes"
                        )
                elif artifact_path.parent.name.endswith("_network_completion"):
                    network_dir = artifact_path.parent
                    related_paths.append(network_dir)
                    est = _estimate_grad_calls_from_network_completion_dir(network_dir)
                    if est > 0 and gradient_calls_total is None:
                        gradient_calls_total = int(est)
                        metadata["gradient_calls_source"] = "estimated_from_network_completion_histories"
                        metadata["gradient_calls_assumption"] = (
                            "per_neb_step=max(0,n_nodes-2); assumes endpoints frozen and no extra frozen interior nodes"
                        )
                metadata["status_path"] = str(artifact_path)
                metadata["path_min_method"] = str(
                    run_status.get("path_min_method", "")
                )

            elif kind == "msmep_tree" or (kind == "auto" and _is_msmep_tree_dir(candidate_path)):
                base_name = candidate_path.name
                out_dir = candidate_path.parent
                command_name = "run"
                mode_name = "recursive"
                related_paths.append(out_dir / f"{base_name}.xyz")
                est = _estimate_grad_calls_from_tree_folder(candidate_path)
                if est > 0:
                    gradient_calls_total = int(est)
                    metadata["gradient_calls_source"] = "estimated_from_tree_history"
                    metadata["gradient_calls_assumption"] = (
                        "per_neb_step=max(0,n_nodes-2); assumes endpoints frozen and no extra frozen interior nodes"
                    )
                metadata["elapsed_source"] = "estimated_from_file_mtime"

            elif kind == "neb_history" or (kind == "auto" and _is_neb_history_dir(candidate_path)):
                base_name = candidate_path.name[: -len("_history")]
                out_dir = candidate_path.parent
                command_name = "run"
                mode_name = "regular"
                related_paths.append(out_dir / f"{base_name}.xyz")
                est = _estimate_grad_calls_from_history_dir(candidate_path)
                if est > 0:
                    gradient_calls_total = int(est)
                    metadata["gradient_calls_source"] = "estimated_from_neb_history"
                    metadata["gradient_calls_assumption"] = (
                        "per_neb_step=max(0,n_nodes-2); assumes endpoints frozen and no extra frozen interior nodes"
                    )
                metadata["elapsed_source"] = "estimated_from_file_mtime"

            elif kind == "workspace" or (kind == "auto" and candidate_path.is_dir() and (candidate_path / "workspace.json").exists()):
                workspace_dir = candidate_path if candidate_path.is_dir() else candidate_path.parent
                base_name = workspace_dir.name
                out_dir = workspace_dir
                command_name = "refine"
                mode_name = "ts-irc"
                artifacts_dir = workspace_dir / "ts_irc_refinement"
                related_paths.extend([workspace_dir / "workspace.json", artifacts_dir])
                ts_irc_estimate = _estimate_ts_irc_grad_calls_from_artifacts(
                    artifacts_dir
                ) if artifacts_dir.exists() else {
                    "gradient_calls_total": 0,
                    "ts_jobs_detected": 0,
                    "irc_jobs_detected": 0,
                    "gradient_calls_assumption": (
                        "TS: (6N) Hessian + 1 grad per optimization step; "
                        "IRC: (6N) Hessian + 1 grad per IRC node"
                    ),
                }
                gradient_calls_total = int(ts_irc_estimate["gradient_calls_total"])
                metadata.update(ts_irc_estimate)
                metadata["gradient_calls_source"] = "estimated_from_ts_irc_artifacts"
                metadata["elapsed_source"] = "estimated_from_file_mtime"

            else:
                if kind == "refined_xyz" or (
                    kind == "auto"
                    and candidate_path.is_file()
                    and candidate_path.name.endswith("_refined.xyz")
                ):
                    refined_fp = candidate_path
                else:
                    refined_fp = candidate_path if candidate_path.is_file() else None
                    if refined_fp is None and candidate_path.is_dir():
                        maybe = sorted(candidate_path.glob("*_refined.xyz"))
                        if len(maybe) == 1:
                            refined_fp = maybe[0]
                    if refined_fp is None:
                        raise ValueError(
                            "Could not infer refinement output. Pass a *_refined.xyz file or use --scan."
                        )
                inferred = _infer_refine_base_name_from_path(refined_fp)
                if not inferred:
                    raise ValueError(
                        f"Could not infer refinement base name from: {refined_fp.name}"
                    )
                base_name = inferred
                out_dir = refined_fp.parent
                command_name = "refine_or_run-refine"
                mode_name = "neb"
                pair_dir = out_dir / f"{base_name}_refined_pairs"
                related_paths.extend(
                    [
                        out_dir / f"{base_name}_cheap.xyz",
                        out_dir / f"{base_name}_refined_minima.xyz",
                        out_dir / f"{base_name}_refined.xyz",
                        pair_dir,
                    ]
                )
                pair_history_est = 0
                if pair_dir.exists():
                    for tree_dir in sorted(pair_dir.glob("pair_*_*_msmep")):
                        if tree_dir.is_dir():
                            pair_history_est += _estimate_grad_calls_from_tree_folder(
                                tree_dir
                            )
                if pair_history_est > 0:
                    gradient_calls_total = int(pair_history_est)
                    metadata["gradient_calls_source"] = "estimated_from_pair_histories"
                    metadata["gradient_calls_assumption"] = (
                        "per_neb_step=max(0,n_nodes-2); assumes endpoints frozen and no extra frozen interior nodes"
                    )
                metadata["elapsed_source"] = "estimated_from_file_mtime"

            assert base_name is not None
            assert out_dir is not None
            cost_fp = _cost_report_path(out_dir, base_name)
            if cost_fp.exists() and not overwrite:
                skipped += 1
                result_table.add_row(
                    str(candidate_path),
                    str(cost_fp),
                    "skipped",
                    "already exists (use --overwrite)",
                )
                continue

            elapsed_est = _estimate_elapsed_seconds_from_paths(related_paths)
            if elapsed_est is None:
                elapsed_est = 0.0
                metadata["elapsed_source"] = metadata.get(
                    "elapsed_source", "unavailable_defaulted_to_zero"
                )
            else:
                metadata["elapsed_source"] = metadata.get(
                    "elapsed_source", "estimated_from_file_mtime"
                )

            _write_cost_report(
                cost_fp,
                command=command_name,
                mode=mode_name,
                elapsed_seconds=float(elapsed_est),
                gradient_calls_total=gradient_calls_total,
                metadata=metadata,
            )
            created += 1
            notes = "estimated"
            if gradient_calls_total is None:
                notes = "time estimated; grad calls unavailable"
            result_table.add_row(str(candidate_path), str(cost_fp), "created", notes)
        except Exception as exc:
            failed += 1
            result_table.add_row(
                str(candidate_path),
                "-",
                "failed",
                f"{type(exc).__name__}: {exc}",
            )

    console.print(result_table)
    summary = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Created", str(created))
    summary.add_row("Skipped", str(skipped))
    summary.add_row("Failed", str(failed))
    console.print(
        Panel(summary, title="[bold cyan]Cost Backfill Summary[/bold cyan]", border_style="cyan")
    )


@app.command("visualize")
def visualize(
    result_path: Annotated[str, typer.Argument(help="Path to a NEB result .xyz, network .json, or TreeNode folder")],
    output_html: Annotated[str, typer.Option(
        "--output", "-o", help="Output HTML file path")] = None,
    atom_indices: Annotated[str, typer.Option(
        "--atom-indices", help="Comma/space-separated atom indices (e.g. '1,2,3' or '1 2 3')")] = None,
    charge: Annotated[int, typer.Option(
        help="Charge used when reading serialized geometries")] = 0,
    multiplicity: Annotated[int, typer.Option(
        help="Spin multiplicity used when reading serialized geometries")] = 1,
    no_open: Annotated[bool, typer.Option(
        "--no-open", help="Do not auto-open browser window")] = False,
):
    console.print(BANNER)
    src = Path(result_path).resolve()
    with console.status("[bold cyan]Loading result object...[/bold cyan]"):
        viz_data = _load_visualization_data(
            result_path=src,
            charge=charge,
            multiplicity=multiplicity,
        )
        selected = _parse_visualize_atom_indices(
            atom_indices=atom_indices
        )
        if selected is not None:
            viz_data.chain = _subset_chain_for_visualization(
                viz_data.chain, selected)
            if viz_data.chain_trajectory:
                viz_data.chain_trajectory = _subset_chain_trajectory_for_visualization(
                    viz_data.chain_trajectory, selected
                )
            if viz_data.tree_layers:
                viz_data.tree_layers = _subset_tree_layers_for_visualization(
                    viz_data.tree_layers, selected
                )
            console.print(
                f"[dim]Visualizing atom subset with {len(selected)} atoms.[/dim]"
            )

    with console.status("[bold cyan]Building interactive HTML...[/bold cyan]"):
        network_payload = None
        if viz_data.network_pot is not None:
            network_payload = _build_network_visualization_payload(
                viz_data.network_pot,
                atom_indices=selected,
                endpoint_hints=viz_data.network_endpoint_hints,
            )
        html = _build_chain_visualizer_html(
            chain=viz_data.chain,
            chain_trajectory=viz_data.chain_trajectory,
            tree_layers=viz_data.tree_layers,
            network_payload=network_payload,
        )

    if output_html is None:
        suffix = src.stem if src.is_file() else src.name
        out_fp = Path.cwd() / f"{suffix}_visualize.html"
    else:
        out_fp = Path(output_html).resolve()
    out_fp.write_text(html, encoding="utf-8")
    console.print(
        f"[bold green]✓ Visualization written:[/bold green] {out_fp}")

    if not no_open:
        webbrowser.open(out_fp.resolve().as_uri())
        console.print("[dim]Opened in default browser.[/dim]")


@app.command("irc-network")
def irc_network(
    directory: Annotated[
        str,
        typer.Argument(help="Directory containing IRC XYZ trajectories and matching .energies files"),
    ] = ".",
    output_json: Annotated[
        str | None,
        typer.Option("--output-json", help="Output MEPD network JSON path"),
    ] = None,
    output_html: Annotated[
        str | None,
        typer.Option("--output-html", "-o", help="Output interactive HTML path"),
    ] = None,
    pattern: Annotated[
        str,
        typer.Option("--pattern", help="XYZ glob to scan"),
    ] = "*.xyz",
    recursive: Annotated[
        bool,
        typer.Option("--recursive", "-r", help="Scan subdirectories recursively"),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Fail instead of skipping malformed XYZ/energy pairs"),
    ] = False,
    charge: Annotated[
        int,
        typer.Option(help="Molecular charge used while reading XYZ files"),
    ] = 0,
    multiplicity: Annotated[
        int,
        typer.Option(help="Spin multiplicity used while reading XYZ files"),
    ] = 1,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Do not auto-open the HTML viewer"),
    ] = False,
):
    """Build a connectivity-detected network from IRC trajectory files."""
    console.print(BANNER)
    source_dir = Path(directory).expanduser().resolve()
    with console.status("[bold cyan]Scanning IRC trajectories...[/bold cyan]"):
        scan = build_irc_network(
            source_dir,
            pattern=pattern,
            recursive=recursive,
            charge=charge,
            multiplicity=multiplicity,
            strict=strict,
        )

    json_fp = (
        Path(output_json).expanduser().resolve()
        if output_json
        else source_dir / f"{source_dir.name}_network.json"
    )
    html_fp = (
        Path(output_html).expanduser().resolve()
        if output_html
        else source_dir / f"{source_dir.name}_network.html"
    )
    json_fp.parent.mkdir(parents=True, exist_ok=True)
    html_fp.parent.mkdir(parents=True, exist_ok=True)
    scan.pot.write_to_disk(json_fp)

    with console.status("[bold cyan]Building interactive HTML...[/bold cyan]"):
        payload = _build_network_visualization_payload(scan.pot)
        first_chain = next(
            chain
            for _, _, attrs in scan.pot.graph.edges(data=True)
            for chain in attrs.get("list_of_nebs", [])
        )
        html = _build_chain_visualizer_html(
            chain=first_chain,
            network_payload=payload,
        )
        html_fp.write_text(html, encoding="utf-8")

    console.print(f"[bold green]✓ Network JSON:[/bold green] {json_fp}")
    console.print(f"[bold green]✓ Interactive HTML:[/bold green] {html_fp}")
    console.print(
        f"[dim]Loaded {len(scan.xyz_files)} IRC pair(s); "
        f"skipped {len(scan.skipped_xyz_files)} XYZ file(s).[/dim]"
    )
    if not no_open:
        webbrowser.open(html_fp.as_uri())


@app.command("extract-best-path")
def extract_best_path(
    network_json: Annotated[str, typer.Argument(help="Path to a network .json file")],
    output_xyz: Annotated[str, typer.Option(
        "--output", "-o", help="Output XYZ file path for the joined best path")] = None,
    start_node: Annotated[int, typer.Option(
        "--start-node", help="Explicit network node index to use as the path start")] = None,
    end_node: Annotated[int, typer.Option(
        "--end-node", help="Explicit network node index to use as the path end")] = None,
    charge: Annotated[int, typer.Option(
        help="Charge used when reading serialized geometries")] = 0,
    multiplicity: Annotated[int, typer.Option(
        help="Spin multiplicity used when reading serialized geometries")] = 1,
):
    console.print(BANNER)
    src = Path(network_json).resolve()
    with console.status("[bold cyan]Loading network...[/bold cyan]"):
        viz_data = _load_visualization_data(
            result_path=src,
            charge=charge,
            multiplicity=multiplicity,
        )
    if viz_data.network_pot is None:
        raise typer.BadParameter(
            "extract-best-path requires a network .json input."
        )
    endpoint_hints = dict(viz_data.network_endpoint_hints or {})
    if start_node is not None:
        endpoint_hints["root_index"] = int(start_node)
    if end_node is not None:
        endpoint_hints["target_index"] = int(end_node)
    if not endpoint_hints:
        endpoint_hints = None

    with console.status("[bold cyan]Finding best path...[/bold cyan]"):
        payload = _build_network_visualization_payload(
            viz_data.network_pot,
            endpoint_hints=endpoint_hints,
        )
        path_nodes = payload.get("highlighted_path") or []
        if not path_nodes:
            raise typer.BadParameter(
                "No best path could be inferred from this network.")
        chain = _path_chain_from_pot(viz_data.network_pot, path_nodes)
        if chain is None:
            raise typer.BadParameter(
                "Could not construct a chain for the inferred best path.")

    if output_xyz is None:
        base_name = src.stem.replace("_network", "")
        out_fp = src.with_name(f"{base_name}_best_path.xyz")
    else:
        out_fp = Path(output_xyz).resolve()

    _write_chain_with_nan_fallback(chain, out_fp)
    _write_json_atomic(
        out_fp.with_suffix(".json"),
        {
            "network_path": str(src),
            "root_index": payload.get("root_index"),
            "target_index": payload.get("target_index"),
            "path": path_nodes,
        },
    )
    console.print(f"[bold green]✓ Best path written:[/bold green] {out_fp}")


@app.command("make-default-inputs")
def make_default_inputs(
        name: Annotated[str, typer.Option(
            "--name", help='path to output toml file')] = None,
        path_min_method: Annotated[str, typer.Option("--path-min-method", "-pmm",
                                                     help='name of path minimization.\
                                                          Options are: [neb, fsm/fneb, mlpgi, neb-dlf, geometric-neb]')] = "neb"):
    console.print(BANNER)
    if name is None:
        name = Path(Path(os.getcwd()) / 'default_inputs')
    normalized_method = str(path_min_method or "").strip().lower().replace("_", "-")
    method_aliases = {
        "neb": "neb",
        "fsm": "fneb",
        "fneb": "fneb",
        "mlpgi": "mlpgi",
        "neb-dlf": "neb-dlf",
        "nebdlf": "neb-dlf",
        "geometric-neb": "geometric-neb",
        "geometric_neb": "geometric-neb",
        "geometricneb": "geometric-neb",
        "geometric": "geometric-neb",
    }
    resolved_method = method_aliases.get(normalized_method)
    if resolved_method is None:
        raise typer.BadParameter(
            "--path-min-method/-pmm must be one of: neb, fsm (alias: fneb), mlpgi, neb-dlf, geometric-neb."
        )

    ri = RunInputs(path_min_method=resolved_method)
    out = Path(name)
    ri.save(out.parent / (out.stem+".toml"))
    console.print(
        f"[bold green]✓ Default inputs saved to:[/bold green] {out.parent / (out.stem+'.toml')}")


if __name__ == "__main__":
    app()
