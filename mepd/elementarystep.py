"""
this whole module needs to be revamped and integrated with the qcio results objects probably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple
from mepd.chain import Chain
from mepd.nodes.node import Node
import numpy as np
from mepd.engines.engine import Engine
from mepd.nodes.nodehelpers import (
    _is_connectivity_identical,
    _print_all_comparisons,
    _render_molecule_ascii,
    _reset_comparison_results,
    displace_by_dr,
    is_identical,
)
from mepd.scripts.progress import stop_status, update_status, print_persistent
from mepd.errors import ElectronicStructureError, EnergiesNotComputedError
from qcinf import structure_to_smiles

# Rich imports for flashy CLI output
try:
    from rich.console import Console
    from rich.panel import Panel
    _console = Console()
    _rich_available = True
except ImportError:
    _console = None
    _rich_available = False


SLOPE_THRESH = 0.1
# SLOPE_THRESH = 20


def _get_ts_neighbor_pair_indices(chain: Chain) -> tuple[int, int] | None:
    """Return the highest-energy image and its highest-energy neighbor, ordered along the path."""
    if len(chain) < 3:
        return None

    energies = np.asarray(chain.energies, dtype=float)
    i_high = int(np.argmax(energies))
    if i_high == 0 or i_high == len(chain) - 1:
        return None

    neighs = [i for i in (i_high - 1, i_high + 1) if 0 <= i < len(chain)]
    if not neighs:
        return None

    i_neighbor = int(max(neighs, key=lambda idx: energies[idx]))
    return min(i_high, i_neighbor), max(i_high, i_neighbor)


def _get_ind_minima(chain: Chain) -> np.ndarray:
    """Return interior local minima indices for a chain energy profile."""
    if len(chain) < 3:
        return np.array([], dtype=int)
    energies = np.asarray(chain.energies, dtype=float)
    interior = np.arange(1, len(energies) - 1)
    minima_mask = (energies[interior] < energies[interior - 1]) & (
        energies[interior] < energies[interior + 1]
    )
    return interior[minima_mask]


@dataclass
class ElemStepResults:
    """
    Object to build report on minimization from elementary step checks.
    """

    is_elem_step: bool
    is_concave: bool
    splitting_criterion: str
    minimization_results: List[Node]
    number_grad_calls: int
    new_structures: list[Node] = field(default_factory=list)


@dataclass
class ConcavityResults:
    """
    Stores results on concavity checks (i.e. whether chain has a "dip" that could be\
        a new minimum)
    """

    is_concave: bool
    minimization_results: list[Node]
    number_grad_calls: int
    rejected_minimization_results: list[Node] = field(default_factory=list)

    @property
    def is_not_concave(self):
        return not self.is_concave


@dataclass
class HessianMinimaValidation:
    is_minimum: bool
    frequencies: list[float] = field(default_factory=list)
    min_frequency: float | None = None
    min_hessian_eigenvalue: float | None = None
    reason: str = ""
    hessian_result: Any | None = None


def _extract_hessian_matrix(hessian_result) -> np.ndarray | None:
    results = getattr(hessian_result, "results", None)
    hessian = getattr(results, "hessian", None)
    if hessian is None:
        hessian = getattr(hessian_result, "return_result", None)
    if hessian is None:
        return None
    hessian_arr = np.asarray(hessian, dtype=float)
    if hessian_arr.ndim != 2 or hessian_arr.shape[0] != hessian_arr.shape[1]:
        return None
    return 0.5 * (hessian_arr + hessian_arr.T)


def _extract_hessian_frequencies(hessian_result) -> list[float]:
    results = getattr(hessian_result, "results", None)
    freqs = getattr(results, "freqs_wavenumber", None)
    if freqs is None:
        freqs = getattr(results, "frequencies", None)
    if freqs is not None:
        return [float(freq) for freq in freqs]

    return []


def _extract_hessian_modes_and_frequencies(
    hessian_result,
    node: Node,
) -> tuple[list[np.ndarray], list[float]]:
    results = getattr(hessian_result, "results", None)
    modes = getattr(results, "normal_modes_cartesian", None)
    freqs = getattr(results, "freqs_wavenumber", None)
    if modes is not None and freqs is not None:
        try:
            return [np.asarray(mode, dtype=float) for mode in modes], [
                float(freq) for freq in freqs
            ]
        except Exception:
            pass

    try:
        from mepd.helper_functions import parse_nma_freq_data

        parsed_modes, parsed_freqs = parse_nma_freq_data(hessian_result)
        if parsed_modes and parsed_freqs:
            return [np.asarray(mode, dtype=float) for mode in parsed_modes], [
                float(freq) for freq in parsed_freqs
            ]
    except Exception:
        pass

    hessian = _extract_hessian_matrix(hessian_result)
    if hessian is None:
        return [], []
    eigvals, eigvecs = np.linalg.eigh(hessian)
    modes = []
    for mode_index in range(eigvecs.shape[1]):
        mode = np.asarray(eigvecs[:, mode_index], dtype=float)
        try:
            mode = mode.reshape(node.coords.shape)
        except ValueError:
            pass
        modes.append(mode)
    freqs = [float(np.sign(val) * np.sqrt(abs(val))) for val in eigvals]
    return modes, freqs


def _lowest_frequency_mode(
    hessian_result,
    node: Node,
) -> tuple[np.ndarray | None, float | None]:
    modes, freqs = _extract_hessian_modes_and_frequencies(hessian_result, node)
    if not modes or not freqs:
        return None, None
    count = min(len(modes), len(freqs))
    if count == 0:
        return None, None
    min_index = int(np.argmin(np.asarray(freqs[:count], dtype=float)))
    mode = np.asarray(modes[min_index], dtype=float)
    try:
        mode = mode.reshape(node.coords.shape)
    except ValueError:
        return None, float(freqs[min_index])
    if not np.all(np.isfinite(mode)) or np.linalg.norm(mode) == 0:
        return None, float(freqs[min_index])
    return mode, float(freqs[min_index])


def _validate_hessian_minimum(
    node: Node,
    engine: Engine,
    *,
    frequency_cutoff: float,
) -> HessianMinimaValidation:
    compute_result = getattr(engine, "_compute_hessian_result", None)
    try:
        if callable(compute_result):
            hessian_result = compute_result(node=node)
        else:
            hessian = engine.compute_hessian(node=node)
            from mepd.engines.engine import build_hessian_result_from_matrix

            hessian_result = build_hessian_result_from_matrix(node=node, hessian=hessian)
    except Exception as exc:
        return HessianMinimaValidation(
            is_minimum=False,
            reason=f"hessian calculation failed ({type(exc).__name__}: {exc})",
        )

    frequencies = _extract_hessian_frequencies(hessian_result)
    if not frequencies:
        return HessianMinimaValidation(
            is_minimum=False,
            reason="hessian calculation did not return reported frequencies",
            hessian_result=hessian_result,
        )

    min_frequency = float(min(frequencies))
    is_minimum = min_frequency >= float(frequency_cutoff)
    if is_minimum:
        reason = (
            f"minimum frequency {min_frequency:.3f} >= cutoff {float(frequency_cutoff):.3f}"
        )
    else:
        reason = (
            f"minimum frequency {min_frequency:.3f} < cutoff {float(frequency_cutoff):.3f}"
        )
    return HessianMinimaValidation(
        is_minimum=is_minimum,
        frequencies=frequencies,
        min_frequency=min_frequency,
        min_hessian_eigenvalue=None,
        reason=reason,
        hessian_result=hessian_result,
    )


def _emit_hessian_validation_message(
    msg: str,
    *,
    accepted: bool,
    verbose: bool,
):
    if verbose:
        if _rich_available:
            _console.print(Panel.fit(
                msg,
                border_style="green" if accepted else "yellow",
            ))
        else:
            print(msg)
    elif not accepted:
        stop_status()
        print_persistent(msg)
        update_status("Checking if elementary step")


def _hessian_rescue_failed_candidate(
    node: Node,
    engine: Engine,
    *,
    validation: HessianMinimaValidation,
    frequency_cutoff: float,
    rescue_displacement: float,
    verbose: bool,
    label: str,
) -> tuple[Node | None, HessianMinimaValidation | None, int]:
    mode, mode_frequency = _lowest_frequency_mode(validation.hessian_result, node)
    if mode is None:
        msg = (
            f"Could not rescue {label} split candidate after Hessian rejection: "
            "no usable unstable normal mode was available."
        )
        _emit_hessian_validation_message(msg, accepted=False, verbose=verbose)
        return None, None, 0

    msg = (
        f"Attempting Hessian rescue for {label} split candidate: displacing "
        f"±{float(rescue_displacement):.3f} bohr along lowest-frequency "
        f"mode ({mode_frequency:.3f} cm^-1) and reoptimizing."
    )
    _emit_hessian_validation_message(msg, accepted=False, verbose=verbose)

    rescue_grad_calls = 0
    best_validation: HessianMinimaValidation | None = None
    for signed_dr in (float(rescue_displacement), -float(rescue_displacement)):
        try:
            displaced = displace_by_dr(node=node, displacement=mode, dr=signed_dr)
            opt_traj = _run_geom_opt(displaced, engine=engine)
        except Exception as exc:
            msg = (
                f"Hessian rescue {label} candidate failed during "
                f"{signed_dr:+.3f} bohr reoptimization "
                f"({type(exc).__name__}: {exc})."
            )
            _emit_hessian_validation_message(msg, accepted=False, verbose=verbose)
            continue

        rescue_grad_calls += len(opt_traj)
        if not opt_traj:
            continue
        rescued = opt_traj[-1]
        rescued_validation = _validate_hessian_minimum(
            rescued,
            engine=engine,
            frequency_cutoff=frequency_cutoff,
        )
        if (
            best_validation is None
            or (
                rescued_validation.min_frequency is not None
                and (
                    best_validation.min_frequency is None
                    or rescued_validation.min_frequency > best_validation.min_frequency
                )
            )
        ):
            best_validation = rescued_validation
        if rescued_validation.is_minimum:
            msg = (
                f"Hessian rescue accepted {label} split candidate after "
                f"{signed_dr:+.3f} bohr displacement: {rescued_validation.reason}"
            )
            _emit_hessian_validation_message(msg, accepted=True, verbose=verbose)
            return rescued, rescued_validation, rescue_grad_calls

        msg = (
            f"Hessian rescue rejected {label} split candidate after "
            f"{signed_dr:+.3f} bohr displacement: {rescued_validation.reason}"
        )
        _emit_hessian_validation_message(msg, accepted=False, verbose=verbose)

    return None, best_validation, rescue_grad_calls


def _validate_hessian_split_candidates(
    nodes: list[Node],
    engine: Engine,
    *,
    frequency_cutoff: float,
    rescue_displacement: float,
    verbose: bool,
    label: str,
) -> tuple[list[Node], list[Node], int]:
    accepted = []
    rejected = []
    rescue_grad_calls = 0
    for node in nodes:
        validation = _validate_hessian_minimum(
            node,
            engine=engine,
            frequency_cutoff=frequency_cutoff,
        )
        if validation.is_minimum:
            accepted.append(node)
        else:
            rescued, rescued_validation, grad_calls = _hessian_rescue_failed_candidate(
                node,
                engine=engine,
                validation=validation,
                frequency_cutoff=frequency_cutoff,
                rescue_displacement=rescue_displacement,
                verbose=verbose,
                label=label,
            )
            rescue_grad_calls += grad_calls
            if rescued is not None:
                accepted.append(rescued)
                continue
            rejected.append(node)
            if rescued_validation is not None:
                validation = rescued_validation
        if validation.is_minimum:
            msg = (
                f"Hessian minima validation accepted {label} split candidate: "
                f"{validation.reason}"
            )
        elif label == "minima":
            msg = (
                "Rejecting minima split: apparent intermediate minimum failed "
                f"Hessian validation ({validation.reason}). This candidate will "
                "not be used for minima-based autosplitting; elementary-step "
                "checks will continue and may still request a maxima split if "
                "the path is non-elementary."
            )
        elif label == "maxima":
            msg = (
                "Rejecting maxima split: pseudo-IRC split candidate failed "
                f"Hessian minima validation ({validation.reason})."
            )
        else:
            msg = (
                f"Hessian minima validation rejected {label} split candidate: "
                f"{validation.reason}"
            )
        _emit_hessian_validation_message(
            msg,
            accepted=validation.is_minimum,
            verbose=verbose,
        )
    return accepted, rejected, rescue_grad_calls


@dataclass
class IRCResults:
    """Stores results on (pseudo)IRC checks"""

    found_reactant: Node
    found_product: Node
    number_grad_calls: int
    optimization_succeeded: bool = True


class CachedElementaryStepRequiresEngineError(RuntimeError):
    """Raised when a cached-only elementary-step check needs new optimization."""


class _CachedOnlyEngine:
    compute_program = "chemcloud"

    def _raise_requires_engine(self, *_args, **_kwargs):
        raise CachedElementaryStepRequiresEngineError(
            "This cached XYZ chain requires geometry optimization to finish the "
            "elementary-step check. Install a compute backend/full NEB stack and "
            "run with a real engine, or provide a chain whose cached energies are "
            "sufficient for the check."
        )

    compute_geometry_optimization = _raise_requires_engine
    steepest_descent = _raise_requires_engine
    compute_energies = _raise_requires_engine


def _is_backend_execution_error(exc: Exception) -> bool:
    if isinstance(exc, OSError):
        return True
    exc_name = exc.__class__.__name__
    return exc_name in {"ExternalProgramError", "ProgramNotFoundError"}


def _is_backend_unavailable_error(exc: Exception) -> bool:
    if isinstance(exc, OSError):
        return True
    return exc.__class__.__name__ == "ProgramNotFoundError"


def _backend_probe_failed_msg(exc: Exception) -> str:
    return (
        f"Auxiliary elementary-step optimization failed in the backend: {exc}. "
        "Treating this check as inconclusive."
    )


def check_cached_xyz_elem_step(
    xyz_path: str | Path,
    *,
    charge: int = 0,
    spinmult: int = 1,
    verbose: bool = False,
    cached_only: bool = False,
    program: str = "crest",
    geometry_optimizer: str = "geometric",
    method: str = "gfn2",
    basis: str = "gfn2",
) -> ElemStepResults:
    """
    Check a chain stored as an XYZ file with NEB sidecar energy/gradient files.

    This entry point is for users who already have a saved chain and want the
    elementary-step classification without installing the whole NEB workflow.
    By default it uses a qcop-backed engine for any geometry optimizations that
    the elementary-step report needs.
    """
    from mepd.inputs import ChainInputs

    chain = Chain.from_xyz(
        xyz_path,
        ChainInputs(),
        charge=charge,
        spinmult=spinmult,
    )
    if cached_only:
        if not chain._energies_already_computed:
            raise CachedElementaryStepRequiresEngineError(
                "This XYZ file does not have cached energies. Expected sidecar "
                "files next to the XYZ (`.energies`, `.gradients`, and "
                "`_grad_shapes.txt`) or run without --cached-only so qcop can "
                "compute the chain energies."
            )
        engine = _CachedOnlyEngine()
    else:
        from qcio.models.inputs import ProgramArgs
        from mepd.engines.qcop import QCOPEngine

        engine = QCOPEngine(
            program=program,
            geometry_optimizer=geometry_optimizer,
            compute_program="qcop",
            program_args=ProgramArgs(
                model={"method": method, "basis": basis},
                keywords={"threads": 1},
            ),
        )
        if not chain._energies_already_computed:
            engine.compute_energies(chain)
    return check_if_elem_step(chain, engine=engine, verbose=verbose)


def _print_new_structure(node: Node, message: str = "new structure found!") -> None:
    """Print a notification for a newly discovered structure."""
    if node is None:
        return
    if bool(getattr(node, "disable_smiles", False)):
        print_persistent(message=message, ascii_block="new structure")
        return
    if not getattr(node, "has_molecular_graph", False):
        print_persistent(message=message, ascii_block="new structure")
        return
    smi = ""
    try:
        if getattr(node, "graph", None) is not None:
            smi = str(node.graph.force_smiles())
    except Exception:
        smi = ""
    if not smi:
        try:
            if len(node.coords) < 100:
                smi = structure_to_smiles(node.structure)
        except Exception:
            smi = ""
    ascii_art = _render_molecule_ascii(
        smi, width=60, height=12) if smi else "new structure"
    print_persistent(message=message, ascii_block=ascii_art)


def _classify_new_structure(node: Node, reactant: Node, product: Node) -> str:
    """Classify a discovered structure relative to endpoint molecular graphs."""
    graphs_available = all(
        getattr(x, "has_molecular_graph", False)
        for x in (node, reactant, product)
    )
    if not graphs_available:
        return "new structure found!"

    same_as_reactant = _is_connectivity_identical(
        node, reactant, verbose=False, collect_comparison=False)
    same_as_product = _is_connectivity_identical(
        node, product, verbose=False, collect_comparison=False
    )

    if same_as_reactant and not same_as_product:
        return "new reactant conformer found!"
    if same_as_product and not same_as_reactant:
        return "new product conformer found!"
    if not same_as_reactant and not same_as_product:
        return "new molecule found!"
    return "new structure found!"


def _deduplicate_discoveries(nodes: list[Node], reactant: Node, product: Node) -> list[tuple[Node, str]]:
    """
    Collapse duplicate discovery reports.
    If both endpoint minimizations converge to connectivity-equivalent
    structures with the same classification, print only once.
    """
    unique: list[tuple[Node, str]] = []
    graphs_available = all(
        getattr(x, "has_molecular_graph", False)
        for x in [reactant, product, *nodes]
    )
    for node in nodes:
        msg = _classify_new_structure(
            node=node, reactant=reactant, product=product)
        duplicate = False
        for seen_node, seen_msg in unique:
            if seen_msg != msg:
                continue
            if graphs_available and _is_connectivity_identical(node, seen_node, verbose=False):
                duplicate = True
                break
        if not duplicate:
            unique.append((node, msg))
    return unique


def _write_nodes_xyz(nodes: list[Node], fp: str | Path) -> Path:
    path = Path(fp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for node in nodes:
            structure = getattr(node, "structure", None)
            if structure is None or not hasattr(structure, "to_xyz"):
                continue
            handle.write(structure.to_xyz().rstrip())
            handle.write("\n")
    return path


def _filter_new_structures(nodes: list[Node], reactant: Node, product: Node, chain: Chain) -> list[Node]:
    """Keep only nodes that are not identical to either endpoint."""
    new_nodes: list[Node] = []
    for node in nodes:
        is_r = is_identical(
            node,
            reactant,
            fragment_rmsd_cutoff=chain.parameters.node_rms_thre,
            kcal_mol_cutoff=chain.parameters.node_ene_thre,
            verbose=False,
            collect_comparison=False,
        )
        is_p = is_identical(
            node,
            product,
            fragment_rmsd_cutoff=chain.parameters.node_rms_thre,
            kcal_mol_cutoff=chain.parameters.node_ene_thre,
            verbose=False,
            collect_comparison=False,
        )
        if not (is_r or is_p):
            new_nodes.append(node)
    return new_nodes


def check_if_elem_step(
    inp_chain: Chain,
    engine: Engine,
    verbose: bool = True,
    validate_minima_with_hessian: bool = False,
    hessian_minimum_frequency_cutoff: float = 0.0,
    hessian_minima_rescue_displacement: float = 0.1,
) -> ElemStepResults:
    """Calculates whether an input chain is an elementary step.

    Args:
        inp_chain (Chain): input chain to check.
        verbose (bool): whether to print detailed output (default True)

    Returns:
        ElemStepResults: object containing report on chain.
    """
    # Reset comparison results collector for consolidated reporting
    _reset_comparison_results()

    if verbose:
        if _rich_available:
            _console.print(Panel.fit(
                "[bold cyan]🔍 Checking if chain is elementary step...[/bold cyan]",
                border_style="cyan"
            ))
        else:
            print("Checking if chain is elementary step...")
    else:
        update_status("Checking if elementary step")

    n_geom_opt_grad_calls = 0
    chain = inp_chain.copy()
    if len(inp_chain) <= 1:
        if verbose:
            if _rich_available:
                _console.print(Panel.fit(
                    "[bold green]✓ Chain has 1 or fewer nodes, automatically elementary step[/bold green]",
                    border_style="green",
                ))
            else:
                print("Chain has 1 or fewer nodes, automatically elementary step.")

        return ElemStepResults(
            is_elem_step=True,
            is_concave=True,
            splitting_criterion=None,
            minimization_results=None,
            number_grad_calls=0,
        )

    try:
        concavity_results = _chain_is_concave(
            chain=inp_chain,
            engine=engine,
            verbose=verbose,
            validate_minima_with_hessian=bool(validate_minima_with_hessian),
            hessian_minimum_frequency_cutoff=float(hessian_minimum_frequency_cutoff),
            hessian_minima_rescue_displacement=float(
                hessian_minima_rescue_displacement
            ),
        )
    except TypeError as exc:
        if (
            "validate_minima_with_hessian" not in str(exc)
            and "hessian_minimum_frequency_cutoff" not in str(exc)
            and "hessian_minima_rescue_displacement" not in str(exc)
        ):
            raise
        concavity_results = _chain_is_concave(
            chain=inp_chain, engine=engine, verbose=verbose
        )
    n_geom_opt_grad_calls += concavity_results.number_grad_calls

    if concavity_results.is_not_concave:
        new_structures = _filter_new_structures(
            nodes=concavity_results.minimization_results,
            reactant=chain[0],
            product=chain[-1],
            chain=inp_chain,
        )
        if new_structures:
            if not verbose:
                stop_status()
            for node, msg in _deduplicate_discoveries(
                nodes=new_structures, reactant=chain[0], product=chain[-1]
            ):
                _print_new_structure(node, message=msg)
            if not verbose:
                update_status("Checking if elementary step")
        if verbose:
            _print_all_comparisons()
        return ElemStepResults(
            is_elem_step=False,
            is_concave=concavity_results.is_concave,
            splitting_criterion="minima",
            minimization_results=concavity_results.minimization_results,
            number_grad_calls=n_geom_opt_grad_calls,
            new_structures=new_structures,
        )

    crude_irc_passed, ngc_approx_elem_step = is_approx_elem_step(
        chain=inp_chain, engine=engine, verbose=verbose
    )
    if verbose:
        if _rich_available:
            status = "[bold green]✓ Passed[/bold green]" if crude_irc_passed else "[bold red]✗ Failed[/bold red]"
            _console.print(Panel.fit(
                f"[bold]CrudeIRC:[/bold] {status}",
                border_style="green" if crude_irc_passed else "red",
            ))
        else:
            print("CrudeIRC: ", crude_irc_passed)
    n_geom_opt_grad_calls += ngc_approx_elem_step

    if crude_irc_passed:
        return ElemStepResults(
            is_elem_step=True,
            is_concave=concavity_results.is_concave,
            splitting_criterion=None,
            minimization_results=[inp_chain[0], inp_chain[-1]],
            number_grad_calls=n_geom_opt_grad_calls,
        )

    pseu_irc_results = pseudo_irc(chain=inp_chain, engine=engine)
    n_geom_opt_grad_calls += pseu_irc_results.number_grad_calls
    if not bool(getattr(pseu_irc_results, "optimization_succeeded", True)):
        raise ElectronicStructureError(
            msg=(
                "Pseudo-IRC endpoint optimization did not succeed; refusing to "
                "use raw high-energy path images as minima for a maxima split."
            ),
            obj=pseu_irc_results,
        )

    # Compare endpoints - results are collected for consolidated report
    found_r = is_identical(
        pseu_irc_results.found_reactant,
        chain[0],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
        verbose=False,  # Suppress individual prints, use consolidated report
    )

    found_p = is_identical(
        pseu_irc_results.found_product,
        chain[-1],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
        verbose=False,  # Suppress individual prints, use consolidated report
    )

    p_is_r = is_identical(
        pseu_irc_results.found_product,
        chain[0],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
        verbose=False,  # Suppress individual prints, use consolidated report
    )

    r_is_p = is_identical(
        pseu_irc_results.found_reactant,
        chain[-1],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
        verbose=False,  # Suppress individual prints, use consolidated report
    )

    if found_r and found_p:
        minimizing_gives_endpoints = True
    elif found_r and p_is_r:
        if verbose and _rich_available:
            _console.print(Panel.fit(
                "[bold yellow]⚠ Warning! Both geometries converged to reactant.[/bold yellow]",
                border_style="yellow",
            ))
        elif verbose:
            print("Warning! Both geometries converged to reactant.")
        minimizing_gives_endpoints = True
    elif found_p and r_is_p:
        if verbose and _rich_available:
            _console.print(Panel.fit(
                "[bold yellow]⚠ Warning! Both geometries converged to product.[/bold yellow]",
                border_style="yellow",
            ))
        elif verbose:
            print("Warning! Both geometries converged to product.")

        minimizing_gives_endpoints = True
    else:
        minimizing_gives_endpoints = False

    validated_maxima_nodes: list[Node] = []
    if (
        validate_minima_with_hessian
        and not minimizing_gives_endpoints
    ):
        validation_targets = []
        if not (found_r or r_is_p):
            validation_targets.append(pseu_irc_results.found_reactant)
        if not (found_p or p_is_r):
            validation_targets.append(pseu_irc_results.found_product)
        if validation_targets:
            accepted, rejected, rescue_grad_calls = _validate_hessian_split_candidates(
                validation_targets,
                engine=engine,
                frequency_cutoff=hessian_minimum_frequency_cutoff,
                rescue_displacement=hessian_minima_rescue_displacement,
                verbose=verbose,
                label="maxima",
            )
            n_geom_opt_grad_calls += rescue_grad_calls
            validated_maxima_nodes = list(accepted)

    if validated_maxima_nodes:
        new_structures = _filter_new_structures(
            nodes=validated_maxima_nodes,
            reactant=chain[0],
            product=chain[-1],
            chain=inp_chain,
        )
    else:
        new_structures = []

    if validated_maxima_nodes:
        elem_step = False
    else:
        elem_step = True

    if new_structures:
        if not verbose:
            stop_status()
        for node, msg in _deduplicate_discoveries(
            nodes=new_structures, reactant=chain[0], product=chain[-1]
        ):
            _print_new_structure(node, message=msg)
        if not verbose:
            update_status("Checking if elementary step")
    elif not minimizing_gives_endpoints and not validated_maxima_nodes:
        # We are splitting by maxima, but endpoint mapping did not produce explicit
        # new endpoint structures. Emit a clear notice so recursive runs still show
        # that a non-elementary path (possible new chemistry) was detected.
        if _rich_available:
            _console.print(
                "[bold yellow]new structure pattern found (maxima split)[/bold yellow]")
        else:
            print("new structure pattern found (maxima split)")

    if verbose:
        _print_all_comparisons()

    return ElemStepResults(
        is_elem_step=elem_step,
        is_concave=concavity_results.is_concave,
        splitting_criterion=("maxima" if validated_maxima_nodes else None),
        minimization_results=(
            validated_maxima_nodes
            if validated_maxima_nodes
            else [pseu_irc_results.found_reactant, pseu_irc_results.found_product]
        ),
        number_grad_calls=n_geom_opt_grad_calls,
        new_structures=new_structures,
    )


def _upsample_around_ts_guess(chain, ts_index):
    import mepd.chainhelpers as ch

    tang = ch.calculate_geodesic_tangent(
        list_of_nodes=chain, ref_node_ind=ts_index, dr=0.1)
    tang[0].converged = False
    tang[2].converged = False

    nodes = chain.nodes
    nodes.insert(ts_index, tang[0])
    nodes.insert(ts_index+2, tang[2])
    chain_for_opt = chain.model_copy(update={"nodes": nodes})
    return chain_for_opt


def is_approx_elem_step(
    chain: Chain,
    engine: Engine,
    slope_thresh=SLOPE_THRESH,
    verbose: bool = True,
) -> Tuple[bool, int]:
    """Will do at most 50 steepest descent steps  on geometries neighboring the transition state guess
    and check whether they are approaching the chain endpoints. If function returns False, the geoms
    will be fully optimized.

    Args:
        chain (Chain): chain to check on
        slope_thresh (float, optional): Steepest descent optimization will stop when the slope
        of the distances of the minimized geometry to the target endpoint is >= threshold.
        Defaults to 0.1.

    Returns:
        (bool, int): whether chain seems to be an elementary step, number grad calls it took to do this check

    """
    if chain.energies_are_monotonic:
        return True, 0

    pair_indices = _get_ts_neighbor_pair_indices(chain)
    if pair_indices is None:
        return True, 0
    # if len(chain) == 3 or arg_max == 1 or arg_max == len(chain)-2:
    #     print("Chain TS neighboring nodes need to be approximated. ")

    #     chain_for_opt = _upsample_around_ts_guess(
    #         chain=chain, ts_index=arg_max)

    #     arg_max = arg_max + 1  # now the TS index is different
    #     engine.compute_energies(
    #         [chain_for_opt.nodes[arg_max-1], chain_for_opt.nodes[arg_max+1]])

    # else:
    chain_for_opt = chain.copy()

    if hasattr(engine, "compute_program") and engine.compute_program.lower() == "chemcloud":
        if verbose and _rich_available:
            _console.print(Panel.fit(
                "[bold blue]☁ Chemcloud detected, skipping approx elem step check[/bold blue]\n[dim]Falling back to full geometry-optimization check[/dim]",
                border_style="blue",
            ))
        elif verbose:
            print("Chemcloud detected, skipping approx elem step check; falling back to full geometry-optimization check.")

        return False, 0

    try:
        r_index, p_index = pair_indices
        r_passes_opt, r_traj = _converges_to_an_endpoints(
            chain=chain_for_opt,
            engine=engine,
            node_index=r_index,
            direction=-1,
            slope_thresh=slope_thresh,
            verbose=verbose,
        )
        p_passes_opt, p_traj = _converges_to_an_endpoints(
            chain=chain_for_opt,
            engine=engine,
            node_index=p_index,
            direction=+1,
            slope_thresh=slope_thresh,
            verbose=verbose,
        )
    except CachedElementaryStepRequiresEngineError:
        raise
    except Exception as e:
        if _is_backend_execution_error(e):
            if _is_backend_unavailable_error(e):
                raise
            if verbose:
                print(_backend_probe_failed_msg(e))
        else:
            import traceback

            print(traceback.format_exc())
            print(
                f"Error in geometry optimization: {e}. Pretending this is an elem step.")
        return True, 0
    nodes_have_graph = chain.nodes[0].has_molecular_graph
    # if we have molecular graphs to work with, make sure the connectivities are
    # isomorphic to each other. Otherwise, we will decide only based on distance.
    # (which is bad!!)
    if nodes_have_graph:
        r_passes = r_passes_opt and _is_connectivity_identical(
            r_traj[-1], chain[0], verbose=verbose)
        p_passes = p_passes_opt and _is_connectivity_identical(
            p_traj[-1], chain[-1], verbose=verbose)
    else:
        r_passes = r_passes_opt
        p_passes = p_passes_opt

    n_grad_calls = len(r_traj) + len(p_traj)
    if r_passes and p_passes:
        return True, n_grad_calls
    else:
        return False, n_grad_calls


def _converges_to_an_endpoints(
    chain,
    node_index,
    direction,
    engine: Engine,
    slope_thresh: float,
    max_grad_calls=50,
    verbose: bool = True,
) -> Tuple[bool, List[Node]]:
    """helper function to `is_approx_elem_step`. Actually carries out the minimizations.

    Args:
        chain (_type_): chain with reference geometries.
        node_index (_type_): index of geometry to minimize.
        slope_thresh (float, optional): Threshold for exiting out of minimization early.. Defaults to 0.01.
        direction (int, optional): Direction minimization should be going towards if elem step. -1 refers to
        reactant. +1 refers to product.
        max_grad_calls (int, optional): Maximum number of steepest descent calls until exits out of check.
        Defaults to 50.

    Returns:
        Tuple[bool, List[Node]]: boolean describing whether geometry is minimizing in correct direction, and list of
        nodes containing minimization trajectory.
    """
    done = False
    total_traj = [chain[node_index]]
    endpoint = "reactant" if direction == -1 else "product"
    if verbose and _rich_available:
        _console.print(Panel.fit(
            f"[bold yellow]⚙ Checking if node {node_index} converges to endpoint {endpoint}...[/bold yellow]",
            border_style="yellow",
        ))
    elif verbose:
        print("Checking if node", node_index,
              "converges to endpoint", endpoint, "...")

    while not done:
        try:
            traj = engine.steepest_descent(node=total_traj[-1], max_steps=5)
            total_traj.extend(traj)
        except CachedElementaryStepRequiresEngineError:
            raise
        except Exception as e:
            if _is_backend_execution_error(e):
                if _is_backend_unavailable_error(e):
                    raise
                if verbose:
                    print(_backend_probe_failed_msg(e))
            else:
                import traceback
                print(traceback.format_exc())
                print(
                    f"Error in geometry optimization: {e}. Need to do more expensive check."
                )
            return False, total_traj

        distances = [
            _distances_to_refs(ref1=chain[0], ref2=chain[-1], raw_node=n)
            for n in total_traj
        ]

        slopes_to_ref1 = distances[-1][0] - distances[0][0]
        if np.isclose(distances[-1][0], 0, atol=0.001, rtol=0.001):
            slopes_to_ref1 = -np.inf

        slopes_to_ref2 = distances[-1][1] - distances[0][1]
        if np.isclose(distances[-1][1], 0, atol=0.001, rtol=0.001):
            slopes_to_ref2 = np.inf
        # print("slope1", slopes_to_ref1, "slope2", slopes_to_ref2)

        slope1_conv = abs(slopes_to_ref1) / slope_thresh > 1
        slope2_conv = abs(slopes_to_ref2) / slope_thresh > 1

        # print(f"{slope1_conv=} {slope2_conv=}")
        # slope1_conv = 1
        # slope2_conv = 1

        done = slope1_conv and slope2_conv
        if len(total_traj) - 1 >= max_grad_calls and not done:
            return False, total_traj

    if direction == -1:
        converged_to_reactant = slopes_to_ref1 < 0 and slopes_to_ref2 > 0
        if verbose and _rich_available:
            status = "[bold green]✓ Yes[/bold green]" if converged_to_reactant else "[bold red]✗ No[/bold red]"
            _console.print(Panel.fit(
                f"[bold]Converged to reactant:[/bold] {status}",
                border_style="green" if converged_to_reactant else "red",
            ))
        elif verbose:
            print("Converged to reactant:", converged_to_reactant)

        return converged_to_reactant, total_traj
    elif direction == 1:
        converged_to_product = slopes_to_ref1 > 0 and slopes_to_ref2 < 0
        if verbose and _rich_available:
            status = "[bold green]✓ Yes[/bold green]" if converged_to_product else "[bold red]✗ No[/bold red]"
            _console.print(Panel.fit(
                f"[bold]Converged to product:[/bold] {status}",
                border_style="green" if converged_to_product else "red",
            ))
        elif verbose:
            print("Converged to product:", converged_to_product)

        return converged_to_product, total_traj


def _distances_to_refs(ref1: Node, ref2: Node, raw_node: Node) -> List[float]:
    """
    Computes distances of `raw_node` to `ref1` and `ref2`.
    """
    if raw_node is None:
        return [np.inf, np.inf]
    dist_to_ref1 = np.linalg.norm(
        raw_node.coords - ref1.coords)/np.sqrt(len(raw_node.coords))

    dist_to_ref2 = np.linalg.norm(
        raw_node.coords - ref2.coords)/np.sqrt(len(raw_node.coords))
    return [dist_to_ref1, dist_to_ref2]


def _run_geom_opt(node: Node, engine: Engine):
    """
    will run a check on whether the Engine has implemented the
    geometry optimization function. If not, it will just run Steepest
    Descent.
    """
    # try:
    kwds = {}
    if getattr(engine, "geometry_optimizer", None) == "geometric":
        kwds = {'coordsys': "cart", 'maxiter': 1000}
    opt_traj = engine.compute_geometry_optimization(node, keywords=kwds)
    if not opt_traj:
        raise ElectronicStructureError(
            msg=(
                "Geometry optimization did not produce a converged trajectory; "
                "refusing to accept the input or a failed final frame as a minimum."
            )
        )
    # except AttributeError:
    #     opt_traj = engine.steepest_descent(node, max_steps=500, ss=0.001)

    return opt_traj


def _chain_is_concave(
    chain: Chain,
    engine: Engine,
    min_slope_thre=SLOPE_THRESH,
    verbose: bool = True,
    validate_minima_with_hessian: bool = False,
    hessian_minimum_frequency_cutoff: float = 0.0,
    hessian_minima_rescue_displacement: float = 0.1,
) -> ConcavityResults:
    """
    will assess+categorize the presence of minima on the chain.
    """
    if verbose and _rich_available:
        _console.print(Panel.fit(
            "[bold cyan]🔍 Checking if chain has intermediate minima...[/bold cyan]",
            border_style="cyan",
        ))
    elif verbose:
        print("Checking if chain has intermediate minima...")

    n_grad_calls = 0
    ind_minima = _get_ind_minima(chain=chain)
    if verbose and _rich_available:
        _console.print(Panel.fit(
            f"[bold green]✓ Found {len(ind_minima)} minima on chain[/bold green]",
            border_style="green",
        ))
    elif verbose:
        print(f"\tFound {len(ind_minima)} minima on chain.")

    minima_present = len(ind_minima) != 0
    opt_results = []
    rejected_opt_results = []
    if minima_present:
        minimas_is_r_or_p = []
        try:
            for i in ind_minima:
                # print("chemcloud" not in engine.engine_name.lower(), engine.engine_name.lower())
                compute_program = str(
                    getattr(engine, "compute_program", "") or ""
                ).lower()
                if compute_program != "chemcloud":

                    _, min_traj = _converges_to_an_endpoints(
                        chain=chain,
                        engine=engine,
                        node_index=i,
                        direction=-1,
                        slope_thresh=min_slope_thre,
                        verbose=verbose,
                    )

                    distances = [
                        _distances_to_refs(
                            ref1=chain[0], ref2=chain[-1], raw_node=n)
                        for n in min_traj
                    ]

                    slopes_to_ref1 = distances[-1][0] - distances[0][0]
                    slopes_to_ref2 = distances[-1][1] - distances[0][1]

                    slope1_conv = abs(slopes_to_ref1) / min_slope_thre > 1
                    slope2_conv = abs(slopes_to_ref2) / min_slope_thre > 1
                    # print(f"{slope1_conv=} {slope2_conv=}")

                    done = slope1_conv and slope2_conv
                else:
                    if verbose and _rich_available:
                        _console.print(Panel.fit(
                            "[bold blue]☁ Skipping concavity check for chemcloud[/bold blue]\n[dim]Not minimizing apparent minima, assuming it's real[/dim]",
                            border_style="blue",
                        ))
                    elif verbose:
                        print(
                            "\tSkipping concavity check for chemcloud, not minimizing apparent minima, assuming it's real.")

                    done = False  # chemcloud cannot do the crude irc check
                    kinked_chain = False

                if done:
                    is_r = slopes_to_ref1 < 0 and slopes_to_ref2 > 0
                    is_p = slopes_to_ref1 > 0 and slopes_to_ref2 < 0
                    kinked_chain = is_r or is_p
                    minimas_is_r_or_p.append(kinked_chain)

                elif not done or not kinked_chain:
                    opt_traj = _run_geom_opt(chain[i], engine=engine)
                    n_grad_calls += len(opt_traj)
                    opt = opt_traj[-1]
                    hessian_validated = True
                    if validate_minima_with_hessian:
                        (
                            accepted,
                            _rejected,
                            rescue_grad_calls,
                        ) = _validate_hessian_split_candidates(
                            [opt],
                            engine=engine,
                            frequency_cutoff=hessian_minimum_frequency_cutoff,
                            rescue_displacement=hessian_minima_rescue_displacement,
                            verbose=verbose,
                            label="minima",
                        )
                        n_grad_calls += rescue_grad_calls
                        hessian_validated = bool(accepted)
                    if hessian_validated:
                        opt_results.extend(
                            accepted if validate_minima_with_hessian else [opt]
                        )
                    else:
                        rejected_opt_results.append(opt)
                    is_r = is_identical(
                        opt,
                        chain[0],
                        fragment_rmsd_cutoff=chain.parameters.node_rms_thre,
                        kcal_mol_cutoff=chain.parameters.node_ene_thre,
                        verbose=False,
                    )
                    is_p = is_identical(
                        opt,
                        chain[-1],
                        fragment_rmsd_cutoff=chain.parameters.node_rms_thre,
                        kcal_mol_cutoff=chain.parameters.node_ene_thre,
                        verbose=False,
                    )
                minimas_is_r_or_p.append(is_r or is_p)
        except CachedElementaryStepRequiresEngineError:
            raise
        except Exception as e:
            if _is_backend_execution_error(e):
                if _is_backend_unavailable_error(e):
                    raise
                if verbose:
                    print(_backend_probe_failed_msg(e))
            else:
                import traceback

                print(traceback.format_exc())
                print(
                    f"Error in geometry optimization: {e}. Pretending this is an elem step."
                )

            return ConcavityResults(
                is_concave=True,
                minimization_results=[chain[0], chain[-1]],
                number_grad_calls=n_grad_calls,
                rejected_minimization_results=rejected_opt_results,
            )

        if all(minimas_is_r_or_p) or (validate_minima_with_hessian and not opt_results):
            return ConcavityResults(
                is_concave=True,
                minimization_results=[chain[0], chain[-1]],
                number_grad_calls=n_grad_calls,
                rejected_minimization_results=rejected_opt_results,
            )
        else:
            # assert len(
            #     opt_results) > 0, "chain is not elementary step but minima were not stored"
            return ConcavityResults(
                is_concave=False,
                minimization_results=opt_results,
                number_grad_calls=n_grad_calls,
                rejected_minimization_results=rejected_opt_results,
            )
    else:
        return ConcavityResults(
            is_concave=True,
            minimization_results=[chain[0], chain[-1]],
            number_grad_calls=n_grad_calls,
            rejected_minimization_results=rejected_opt_results,
        )


def pseudo_irc(chain: Chain, engine: Engine):
    n_grad_calls = 0
    arg_max = np.argmax(chain.energies)

    if arg_max == len(chain) - 1 or arg_max == 0:  # monotonically changing function,
        return IRCResults(
            found_reactant=chain[0],
            found_product=chain[len(chain) - 1],
            number_grad_calls=n_grad_calls,
        )
    elif len(chain) == 3 or arg_max == 1 or arg_max == len(chain)-2:
        chain_for_opt = _upsample_around_ts_guess(
            chain=chain, ts_index=arg_max)
        engine.compute_energies(
            [chain_for_opt.nodes[arg_max-1], chain_for_opt.nodes[arg_max+1]]
        )
        arg_max = arg_max+1

    else:
        chain_for_opt = chain
    pair_indices = _get_ts_neighbor_pair_indices(chain_for_opt)
    if pair_indices is None:
        return IRCResults(
            found_reactant=chain[0],
            found_product=chain[len(chain) - 1],
            number_grad_calls=n_grad_calls,
        )
    r_index, p_index = pair_indices
    candidate_r = chain_for_opt[r_index]
    candidate_p = chain_for_opt[p_index]

    try:
        r_traj = _run_geom_opt(candidate_r, engine=engine)
        r = r_traj[-1]
        n_grad_calls += len(r_traj)

        p_traj = _run_geom_opt(candidate_p, engine=engine)
        n_grad_calls += len(p_traj)
        p = p_traj[-1]

    except CachedElementaryStepRequiresEngineError:
        raise
    except Exception as e:
        if isinstance(e, ElectronicStructureError):
            raise
        raise ElectronicStructureError(
            msg=(
                "Pseudo-IRC endpoint optimization failed; no maxima split "
                "endpoints were accepted."
            ),
            obj=e,
        ) from e

    return IRCResults(
        found_reactant=r,
        found_product=p,
        number_grad_calls=n_grad_calls,
        optimization_succeeded=True,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mepd-elementarystep",
        description="Check whether a saved chain XYZ is an elementary step.",
    )
    parser.add_argument("xyz_path", help="Path to the chain XYZ file.")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spinmult", type=int, default=1)
    parser.add_argument(
        "--program",
        default="crest",
        help="qcop subprogram used for energies/gradients. Default: crest.",
    )
    parser.add_argument(
        "--geometry-optimizer",
        default="geometric",
        help="qcop program used for geometry optimizations. Default: geometric.",
    )
    parser.add_argument("--method", default="gfn2")
    parser.add_argument("--basis", default="gfn2")
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Do not run geometry optimizations; fail if the check needs them.",
    )
    parser.add_argument(
        "--new-structures-out",
        default=None,
        help=(
            "Path for writing newly discovered structures as multi-frame XYZ. "
            "Defaults to <input>_new_structures.xyz when new structures are found."
        ),
    )
    parser.add_argument(
        "--no-write-new-structures",
        action="store_true",
        help="Do not write newly discovered structures to disk.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed elementary-step diagnostics.",
    )
    args = parser.parse_args(argv)

    try:
        result = check_cached_xyz_elem_step(
            args.xyz_path,
            charge=args.charge,
            spinmult=args.spinmult,
            verbose=args.verbose,
            cached_only=args.cached_only,
            program=args.program,
            geometry_optimizer=args.geometry_optimizer,
            method=args.method,
            basis=args.basis,
        )
    except CachedElementaryStepRequiresEngineError as exc:
        print(f"error={exc}")
        return 2
    except Exception as exc:
        print(f"error={exc}")
        return 2

    print(f"is_elem_step={result.is_elem_step}")
    print(f"is_concave={result.is_concave}")
    print(f"splitting_criterion={result.splitting_criterion}")
    print(f"number_grad_calls={result.number_grad_calls}")
    print(f"new_structures_count={len(result.new_structures)}")
    if result.new_structures and not args.no_write_new_structures:
        out_fp = (
            Path(args.new_structures_out)
            if args.new_structures_out
            else Path(args.xyz_path).with_name(
                f"{Path(args.xyz_path).stem}_new_structures.xyz"
            )
        )
        written_fp = _write_nodes_xyz(result.new_structures, out_fp)
        print(f"new_structures_xyz={written_fp}")
    return 0 if result.is_elem_step else 1


if __name__ == "__main__":
    raise SystemExit(main())
