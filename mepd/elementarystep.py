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
from mepd.progress import stop_status, update_status, print_persistent
from mepd.errors import ElectronicStructureError
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
    modes, freqs, _source = _extract_hessian_modes_frequencies_and_source(hessian_result, node)
    return modes, freqs


def _extract_hessian_modes_frequencies_and_source(
    hessian_result,
    node: Node,
) -> tuple[list[np.ndarray], list[float], str]:
    """Same as `_extract_hessian_modes_and_frequencies`, but also reports
    which source actually produced the modes:

    - `"normal_modes_cartesian"`: whatever a QC program's own result object
      emitted (usually already excludes translations/rotations, in whatever
      order the program used).
    - `"parse_nma_freq_data"`: parsed from a mass-weighted-modes file (e.g.
      TeraChem), or -- indistinguishably from here -- that parser's own
      internal raw-Hessian-eigendecomposition fallback when no such file is
      available (ascending order, translations/rotations included).
    - `"hessian_eigendecomposition"`: this function's own bottom-of-the-line
      fallback, a raw `np.linalg.eigh` of the Cartesian Hessian (ascending
      order, translations/rotations included).
    - `"none"`: no Hessian data could be extracted at all.

    Used by `prepare_modes` for provenance, and to know that translation/
    rotation filtering is only ever a no-op (not a mistake) on the sources
    that already exclude them.
    """
    results = getattr(hessian_result, "results", None)
    modes = getattr(results, "normal_modes_cartesian", None)
    freqs = getattr(results, "freqs_wavenumber", None)
    if modes is not None and freqs is not None:
        try:
            return (
                [np.asarray(mode, dtype=float) for mode in modes],
                [float(freq) for freq in freqs],
                "normal_modes_cartesian",
            )
        except Exception:
            pass

    try:
        from mepd.helper_functions import parse_nma_freq_data

        parsed_modes, parsed_freqs = parse_nma_freq_data(hessian_result)
        if parsed_modes and parsed_freqs:
            return (
                [np.asarray(mode, dtype=float) for mode in parsed_modes],
                [float(freq) for freq in parsed_freqs],
                "parse_nma_freq_data",
            )
    except Exception:
        pass

    hessian = _extract_hessian_matrix(hessian_result)
    if hessian is None:
        return [], [], "none"
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
    return modes, freqs, "hessian_eigendecomposition"


_TRANS_ROT_CUTOFF_WAVENUMBER = 50.0
_MODE_NORM_MIN = 1e-8


def _expected_trans_rot_dof(coords: np.ndarray) -> int:
    """3 translational plus 0 (single atom), 2 (linear), or 3 (otherwise)
    rotational degrees of freedom. Linearity is detected by the rank of the
    centered coordinate matrix: colinear atoms span a 1-D subspace, so the
    second-largest singular value is ~0.

    Returns 0 for anything that isn't a genuine (n_atoms, 3) Cartesian
    geometry (e.g. a toy 2-D potential's coordinates) -- translation/
    rotation only has physical meaning for real 3-D molecular structures, so
    mode-dropping is simply skipped for anything else rather than guessing.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return 0
    natoms = coords.shape[0]
    if natoms <= 1:
        return 3 * natoms
    centered = coords - coords.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    is_linear = len(singular_values) < 2 or singular_values[1] < 1e-6 * max(singular_values[0], 1e-12)
    return 5 if is_linear else 6


@dataclass
class ModeSet:
    """Cartesian normal modes ready for displacement: translations/rotations
    filtered out, sorted ascending by signed frequency (imaginary modes --
    negative frequency, i.e. a reaction coordinate at a TS -- first), each
    mode L2-normalized over the full 3N Cartesian vector, with its harmonic
    force constant precomputed. Built by `prepare_modes`.
    """

    vectors: list[np.ndarray]        # each shaped like node.coords, L2-normalized over 3N
    freqs_wavenumber: list[float]    # signed; negative == imaginary
    force_constants: list[float]     # k_i = u_i^T H u_i, Hartree/bohr^2 (nan if Hessian unavailable)
    imaginary_indices: list[int]
    source: str
    n_dropped_trans_rot: int

    def __len__(self) -> int:
        return len(self.vectors)


def prepare_modes(
    hessian_result,
    node: Node,
    *,
    trans_rot_cutoff_wavenumber: float = _TRANS_ROT_CUTOFF_WAVENUMBER,
) -> ModeSet:
    """The single place every Hessian-mode consumer that wants to *displace*
    along modes (e.g. `run_hessian_sample`) should go through, instead of
    calling `_extract_hessian_modes_and_frequencies` directly.

    That function has four possible sources with inconsistent guarantees
    (see `_extract_hessian_modes_frequencies_and_source`): two of them
    return every 3N eigenvector, translations/rotations included, in
    ascending-eigenvalue order; the other two typically already exclude
    trans/rot but in whatever order the QC program emitted them. Code that
    just takes the first `max_candidates` modes in list order therefore
    silently wastes part of its budget on directions that displace nothing
    physical (a translation/rotation re-optimizes straight back to the seed)
    on two of the four sources, and gets an inconsistent ordering across all
    four.

    This normalizes all four into one contract:

    - Translations/rotations dropped when the source gave the full 3N set
      (the two raw-eigendecomposition sources), capped at the expected count
      (6, or 5 for a linear molecule, from `_expected_trans_rot_dof`) so a
      genuinely soft low-frequency vibration is never mistaken for one -- a
      mismatch between the cutoff-based count and the expected count is
      reported via `warnings.warn` rather than silently dropping more or
      fewer. A source that already gave fewer than 3N modes is trusted to
      have excluded trans/rot itself; nothing further is dropped from it.
    - Sorted ascending by signed frequency, so imaginary modes (the reaction
      coordinate at a TS seed) come first.
    - Each mode L2-normalized over the full 3N Cartesian vector -- the same
      normalization `displace_by_dr` applies, done once here so force
      constants are computed against the actual unit vector displacement
      uses.
    - Harmonic force constant `k_i = u_i^T H u_i` precomputed from the same
      Cartesian Hessian (Hartree/bohr^2). This needs no reduced masses (not
      reliably available across all four sources) and is exactly the
      curvature felt along the direction displacement actually moves in.
    """
    modes, freqs, source = _extract_hessian_modes_frequencies_and_source(hessian_result, node)
    if not modes or not freqs:
        return ModeSet(
            vectors=[], freqs_wavenumber=[], force_constants=[],
            imaginary_indices=[], source=source, n_dropped_trans_rot=0,
        )

    count = min(len(modes), len(freqs))
    coords = np.asarray(node.coords, dtype=float)

    reshaped_modes: list[np.ndarray] = []
    kept_freqs: list[float] = []
    for mode, freq in zip(modes[:count], freqs[:count]):
        mode = np.asarray(mode, dtype=float)
        if mode.shape != coords.shape:
            try:
                mode = mode.reshape(coords.shape)
            except ValueError:
                continue  # can't align this mode to the seed's geometry; drop it
        norm = float(np.linalg.norm(mode))
        if not np.isfinite(norm) or norm < _MODE_NORM_MIN:
            continue
        reshaped_modes.append(mode / norm)
        kept_freqs.append(float(freq))

    expected_trans_rot = _expected_trans_rot_dof(coords)
    natoms = coords.shape[0] if coords.ndim == 2 and coords.shape[1] == 3 else 0
    already_excluded = expected_trans_rot == 0 or len(kept_freqs) <= max(
        3 * natoms - expected_trans_rot, 0
    )
    if already_excluded:
        # Fewer modes were given than the full 3N (or this isn't a real 3-D
        # geometry at all) -- the source already looks like it excludes
        # translations/rotations, so trust it rather than second-guessing
        # via the cutoff, which would risk dropping a genuine soft mode.
        drop_indices: set[int] = set()
    else:
        candidate_drop = [
            i for i, f in enumerate(kept_freqs) if abs(f) < trans_rot_cutoff_wavenumber
        ]
        if len(candidate_drop) != expected_trans_rot:
            import warnings
            warnings.warn(
                f"prepare_modes: expected {expected_trans_rot} translation/rotation "
                f"mode(s) below {trans_rot_cutoff_wavenumber} cm^-1, found "
                f"{len(candidate_drop)} (source={source}). Capping at "
                f"{expected_trans_rot} lowest-magnitude candidates so a genuinely "
                "soft vibration isn't mistaken for one.",
                stacklevel=2,
            )
        drop_indices = set(
            sorted(candidate_drop, key=lambda i: abs(kept_freqs[i]))[:expected_trans_rot]
        )

    vectors = [m for i, m in enumerate(reshaped_modes) if i not in drop_indices]
    vibrational_freqs = [f for i, f in enumerate(kept_freqs) if i not in drop_indices]
    n_dropped = len(drop_indices)

    order = sorted(range(len(vibrational_freqs)), key=lambda i: vibrational_freqs[i])
    vectors = [vectors[i] for i in order]
    vibrational_freqs = [vibrational_freqs[i] for i in order]

    hessian = _extract_hessian_matrix(hessian_result)
    force_constants: list[float] = []
    for vec in vectors:
        if hessian is None:
            force_constants.append(float("nan"))
            continue
        flat = vec.reshape(-1)
        if flat.shape[0] != hessian.shape[0]:
            force_constants.append(float("nan"))
        else:
            force_constants.append(float(flat @ hessian @ flat))

    imaginary_indices = [i for i, f in enumerate(vibrational_freqs) if f < 0]

    return ModeSet(
        vectors=vectors,
        freqs_wavenumber=vibrational_freqs,
        force_constants=force_constants,
        imaginary_indices=imaginary_indices,
        source=source,
        n_dropped_trans_rot=n_dropped,
    )


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


# Pushes tried after the configured one, in bohr, when a rescue fails. A
# single small push is not always enough: eclipsed ethane re-optimizes
# straight back onto its torsional saddle from +-0.1 bohr but reaches
# staggered from +-0.3.
RESCUE_ESCALATION_BOHR = (0.3, 0.5)


def _rescue_schedule(first: float) -> list[float]:
    """The pushes to try, smallest first: the configured one, then each
    escalation step larger than it."""
    first = float(first)
    return [first] + [d for d in RESCUE_ESCALATION_BOHR if d > first + 1e-9]


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
    """Push a Hessian-rejected structure along its lowest mode, both ways,
    and reoptimize -- first by `rescue_displacement`, then by each larger step
    of RESCUE_ESCALATION_BOHR -- stopping at the first reoptimized geometry
    that passes the Hessian check. Returns (rescued node or None, best
    validation seen, gradient calls spent)."""
    mode, mode_frequency = _lowest_frequency_mode(validation.hessian_result, node)
    if mode is None:
        msg = (
            f"Could not rescue {label} split candidate after Hessian rejection: "
            "no usable unstable normal mode was available."
        )
        _emit_hessian_validation_message(msg, accepted=False, verbose=verbose)
        return None, None, 0

    schedule = _rescue_schedule(rescue_displacement)
    msg = (
        f"Attempting Hessian rescue for {label} split candidate: displacing "
        f"along lowest-frequency mode ({mode_frequency:.3f} cm^-1) by "
        f"{', '.join(f'±{d:.3f}' for d in schedule)} bohr (in that order, until one "
        "reoptimizes to a minimum)."
    )
    _emit_hessian_validation_message(msg, accepted=False, verbose=verbose)

    rescue_grad_calls = 0
    best_validation: HessianMinimaValidation | None = None
    for signed_dr in [s * d for d in schedule for s in (1.0, -1.0)]:
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
            rescued_validation.reason += f" (rescued with a {signed_dr:+.3f} bohr push)"
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


def validate_minimum_with_rescue(
    node: Node,
    engine: Engine,
    *,
    frequency_cutoff: float = 0.0,
    rescue_displacement: float = 0.1,
    label: str = "structure",
) -> tuple[Node, dict]:
    """Hessian-check that `node` is a minimum; if it is not, push it along
    its lowest mode (both directions) and reoptimize, escalating the push
    (see _hessian_rescue_failed_candidate) until one attempt succeeds.

    Returns (node to keep, record). `record` has is_minimum, min_frequency,
    rescued and validation (a human-readable reason). When the rescue
    succeeds the returned node is the rescued geometry; otherwise it is the
    input node, with is_minimum False (callers decide whether to drop it).
    """
    check = _validate_hessian_minimum(node, engine, frequency_cutoff=frequency_cutoff)
    rescued = False
    if not check.is_minimum and check.hessian_result is not None:
        new_node, new_check, _ = _hessian_rescue_failed_candidate(
            node, engine, validation=check, frequency_cutoff=frequency_cutoff,
            rescue_displacement=rescue_displacement, verbose=False, label=label,
        )
        if new_node is not None and new_check is not None and new_check.is_minimum:
            node, check, rescued = new_node, new_check, True
        elif (new_check is not None and check.min_frequency is not None
              and new_check.min_frequency is not None and new_check.min_frequency > check.min_frequency):
            check = new_check  # report the closest we got
    return node, {"is_minimum": bool(check.is_minimum), "min_frequency": check.min_frequency,
                  "rescued": rescued, "validation": check.reason}


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
    By default it uses a qccompute-backed engine for any geometry optimizations that
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
                "`_grad_shapes.txt`) or run without --cached-only so qccompute can "
                "compute the chain energies."
            )
        engine = _CachedOnlyEngine()
    else:
        from mepd.program_args import ProgramArgs
        from mepd.engines.qccompute import QCComputeEngine

        engine = QCComputeEngine(
            program=program,
            geometry_optimizer=geometry_optimizer,
            compute_program="qccompute",
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


def _classify_new_structure(
    node: Node,
    reactant: Node,
    product: Node,
    *,
    disregard_stereochem: bool = False,
) -> str:
    """Classify a discovered structure relative to endpoint molecular graphs."""
    graphs_available = all(
        getattr(x, "has_molecular_graph", False)
        for x in (node, reactant, product)
    )
    if not graphs_available:
        return "new structure found!"

    same_as_reactant = _is_connectivity_identical(
        node,
        reactant,
        verbose=False,
        collect_comparison=False,
        disregard_stereochem=disregard_stereochem,
    )
    same_as_product = _is_connectivity_identical(
        node,
        product,
        verbose=False,
        collect_comparison=False,
        disregard_stereochem=disregard_stereochem,
    )

    if same_as_reactant and not same_as_product:
        return "new reactant conformer found!"
    if same_as_product and not same_as_reactant:
        return "new product conformer found!"
    if not same_as_reactant and not same_as_product:
        return "new molecule found!"
    return "new structure found!"


def _deduplicate_discoveries(
    nodes: list[Node],
    reactant: Node,
    product: Node,
    *,
    disregard_stereochem: bool = False,
) -> list[tuple[Node, str]]:
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
            node=node,
            reactant=reactant,
            product=product,
            disregard_stereochem=disregard_stereochem,
        )
        duplicate = False
        for seen_node, seen_msg in unique:
            if seen_msg != msg:
                continue
            if graphs_available and _is_connectivity_identical(
                node,
                seen_node,
                verbose=False,
                disregard_stereochem=disregard_stereochem,
            ):
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


def _filter_new_structures(
    nodes: list[Node],
    reactant: Node,
    product: Node,
    chain: Chain,
    *,
    disregard_stereochem: bool = False,
) -> list[Node]:
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
            disregard_stereochem=disregard_stereochem,
        )
        is_p = is_identical(
            node,
            product,
            fragment_rmsd_cutoff=chain.parameters.node_rms_thre,
            kcal_mol_cutoff=chain.parameters.node_ene_thre,
            verbose=False,
            collect_comparison=False,
            disregard_stereochem=disregard_stereochem,
        )
        if not (is_r or is_p):
            new_nodes.append(node)
    return new_nodes


def _comparison_preparer(engine: Engine):
    prepare_node = getattr(engine, "prepare_node_for_comparison", None)
    return prepare_node if callable(prepare_node) else None


def _prepare_node_for_comparison(node: Node, prepare_node):
    if prepare_node is None:
        return node
    return prepare_node(node)


def _prepare_nodes_for_comparison(nodes: list[Node], prepare_node) -> list[Node]:
    if prepare_node is None:
        return nodes
    return [_prepare_node_for_comparison(node, prepare_node) for node in nodes]


def check_if_elem_step(
    inp_chain: Chain,
    engine: Engine,
    verbose: bool = True,
    validate_minima_with_hessian: bool = False,
    hessian_minimum_frequency_cutoff: float = 0.0,
    hessian_minima_rescue_displacement: float = 0.1,
    geodesic_kwargs: dict | None = None,
    disregard_stereochem: bool = False,
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
    prepare_node = _comparison_preparer(engine)
    if prepare_node is not None:
        chain = chain.model_copy(
            update={"nodes": [prepare_node(node) for node in chain.nodes]}
        )
        inp_chain = chain
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

    stereochem_kwargs = (
        {"disregard_stereochem": True} if disregard_stereochem else {}
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
            **stereochem_kwargs,
        )
    except TypeError as exc:
        if (
            "validate_minima_with_hessian" not in str(exc)
            and "hessian_minimum_frequency_cutoff" not in str(exc)
            and "hessian_minima_rescue_displacement" not in str(exc)
        ):
            raise
        concavity_results = _chain_is_concave(
            chain=inp_chain,
            engine=engine,
            verbose=verbose,
            **stereochem_kwargs,
        )
    n_geom_opt_grad_calls += concavity_results.number_grad_calls

    if concavity_results.is_not_concave:
        minimization_results = _prepare_nodes_for_comparison(
            list(concavity_results.minimization_results or []),
            prepare_node,
        )
        new_structures = _filter_new_structures(
            nodes=minimization_results,
            reactant=chain[0],
            product=chain[-1],
            chain=inp_chain,
            disregard_stereochem=disregard_stereochem,
        )
        if new_structures:
            if not verbose:
                stop_status()
            for node, msg in _deduplicate_discoveries(
                nodes=new_structures,
                reactant=chain[0],
                product=chain[-1],
                disregard_stereochem=disregard_stereochem,
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
            minimization_results=minimization_results,
            number_grad_calls=n_geom_opt_grad_calls,
            new_structures=new_structures,
        )

    # NOTE: the CrudeIRC shortcut (`is_approx_elem_step`) is disabled -- it was
    # found to be unreliable (e.g. it silently assumes "elementary step" on
    # any internal error) and could short-circuit the elem-step decision
    # before the full pseudo-IRC-based check below ever ran. Always fall
    # through to the full check instead. `is_approx_elem_step` itself is left
    # in place, just unused here, in case it's revisited later.

    if geodesic_kwargs is None:
        pseu_irc_results = pseudo_irc(chain=inp_chain, engine=engine)
    else:
        pseu_irc_results = pseudo_irc(
            chain=inp_chain, engine=engine, geodesic_kwargs=geodesic_kwargs)
    n_geom_opt_grad_calls += pseu_irc_results.number_grad_calls
    if not bool(getattr(pseu_irc_results, "optimization_succeeded", True)):
        raise ElectronicStructureError(
            msg=(
                "Pseudo-IRC endpoint optimization did not succeed; refusing to "
                "use raw high-energy path images as minima for a maxima split."
            ),
            obj=pseu_irc_results,
        )
    found_reactant = _prepare_node_for_comparison(
        pseu_irc_results.found_reactant,
        prepare_node,
    )
    found_product = _prepare_node_for_comparison(
        pseu_irc_results.found_product,
        prepare_node,
    )

    # Compare endpoints - results are collected for consolidated report
    found_r = is_identical(
        found_reactant,
        chain[0],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
        verbose=False,  # Suppress individual prints, use consolidated report
        disregard_stereochem=disregard_stereochem,
    )

    found_p = is_identical(
        found_product,
        chain[-1],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
        verbose=False,  # Suppress individual prints, use consolidated report
        disregard_stereochem=disregard_stereochem,
    )

    p_is_r = is_identical(
        found_product,
        chain[0],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
        verbose=False,  # Suppress individual prints, use consolidated report
        disregard_stereochem=disregard_stereochem,
    )

    r_is_p = is_identical(
        found_reactant,
        chain[-1],
        fragment_rmsd_cutoff=inp_chain.parameters.node_rms_thre,
        kcal_mol_cutoff=inp_chain.parameters.node_ene_thre,
        verbose=False,  # Suppress individual prints, use consolidated report
        disregard_stereochem=disregard_stereochem,
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

    maxima_split_nodes: list[Node] = []
    if not minimizing_gives_endpoints:
        validation_targets: list[Node] = []
        if not (found_r or r_is_p):
            validation_targets.append(found_reactant)
        if not (found_p or p_is_r):
            validation_targets.append(found_product)
        if validate_minima_with_hessian and validation_targets:
            accepted, rejected, rescue_grad_calls = _validate_hessian_split_candidates(
                validation_targets,
                engine=engine,
                frequency_cutoff=hessian_minimum_frequency_cutoff,
                rescue_displacement=hessian_minima_rescue_displacement,
                verbose=verbose,
                label="maxima",
            )
            n_geom_opt_grad_calls += rescue_grad_calls
            maxima_split_nodes = list(accepted)
        else:
            maxima_split_nodes = validation_targets

    if maxima_split_nodes:
        new_structures = _filter_new_structures(
            nodes=maxima_split_nodes,
            reactant=chain[0],
            product=chain[-1],
            chain=inp_chain,
            disregard_stereochem=disregard_stereochem,
        )
    else:
        new_structures = []

    if maxima_split_nodes:
        elem_step = False
    else:
        elem_step = True

    if new_structures:
        if not verbose:
            stop_status()
        for node, msg in _deduplicate_discoveries(
            nodes=new_structures,
            reactant=chain[0],
            product=chain[-1],
            disregard_stereochem=disregard_stereochem,
        ):
            _print_new_structure(node, message=msg)
        if not verbose:
            update_status("Checking if elementary step")
    elif not minimizing_gives_endpoints and not maxima_split_nodes:
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
        splitting_criterion=("maxima" if maxima_split_nodes else None),
        minimization_results=(
            maxima_split_nodes
            if maxima_split_nodes
            else [found_reactant, found_product]
        ),
        number_grad_calls=n_geom_opt_grad_calls,
        new_structures=new_structures,
    )


def _upsample_around_ts_guess(chain, ts_index, geodesic_kwargs: dict | None = None):
    import mepd.chainhelpers as ch

    geodesic_kwargs = geodesic_kwargs or {}
    tang = ch.calculate_geodesic_tangent(
        list_of_nodes=chain, ref_node_ind=ts_index, dr=0.1, **geodesic_kwargs)
    tang[0].converged = False
    tang[2].converged = False

    nodes = list(chain.nodes)
    nodes.insert(ts_index, tang[0])
    nodes.insert(ts_index+2, tang[2])
    chain_for_opt = chain.model_copy(update={"nodes": nodes})
    return chain_for_opt


def _cache_returned_energies(nodes: list[Node], energies) -> None:
    """Keep generated nodes usable when an engine returns energies without mutating."""
    if energies is None:
        return
    for node, energy in zip(nodes, energies):
        if getattr(node, "_cached_energy", None) is None:
            node._cached_energy = float(energy)


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


def _run_geom_opts(nodes: list[Node], engine: Engine) -> list[list[Node]]:
    kwds = {}
    if getattr(engine, "geometry_optimizer", None) == "geometric":
        kwds = {'coordsys': "cart", 'maxiter': 1000}
    if hasattr(engine, "compute_geometry_optimizations"):
        trajectories = engine.compute_geometry_optimizations(nodes, keywords=kwds)
    else:
        trajectories = [_run_geom_opt(node, engine=engine) for node in nodes]
    if len(trajectories) != len(nodes) or any(not traj for traj in trajectories):
        raise ElectronicStructureError(
            msg=(
                "Geometry optimization did not produce converged trajectories; "
                "refusing to accept the input or failed final frames as minima."
            ),
            obj=trajectories,
        )
    return trajectories


def _chain_is_concave(
    chain: Chain,
    engine: Engine,
    min_slope_thre=SLOPE_THRESH,
    verbose: bool = True,
    validate_minima_with_hessian: bool = False,
    hessian_minimum_frequency_cutoff: float = 0.0,
    hessian_minima_rescue_displacement: float = 0.1,
    disregard_stereochem: bool = False,
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
    prepare_node = _comparison_preparer(engine)
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
                        prepared_results = _prepare_nodes_for_comparison(
                            list(accepted) if validate_minima_with_hessian else [opt],
                            prepare_node,
                        )
                        opt_results.extend(prepared_results)
                        opt_for_comparison = prepared_results[0]
                    else:
                        rejected_opt_results.append(opt)
                        opt_for_comparison = _prepare_node_for_comparison(
                            opt,
                            prepare_node,
                        )
                    is_r = is_identical(
                        opt_for_comparison,
                        chain[0],
                        fragment_rmsd_cutoff=chain.parameters.node_rms_thre,
                        kcal_mol_cutoff=chain.parameters.node_ene_thre,
                        verbose=False,
                        disregard_stereochem=disregard_stereochem,
                    )
                    is_p = is_identical(
                        opt_for_comparison,
                        chain[-1],
                        fragment_rmsd_cutoff=chain.parameters.node_rms_thre,
                        kcal_mol_cutoff=chain.parameters.node_ene_thre,
                        verbose=False,
                        disregard_stereochem=disregard_stereochem,
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


def pseudo_irc(chain: Chain, engine: Engine, geodesic_kwargs: dict | None = None):
    n_grad_calls = 0
    arg_max = np.argmax(chain.energies)

    if arg_max == len(chain) - 1 or arg_max == 0:  # monotonically changing function,
        return IRCResults(
            found_reactant=chain[0],
            found_product=chain[len(chain) - 1],
            number_grad_calls=n_grad_calls,
        )
    elif len(chain) == 3 or arg_max == 1 or arg_max == len(chain)-2:
        update_status("Preparing pseudo-IRC tangent around TS guess")
        chain_for_opt = _upsample_around_ts_guess(
            chain=chain, ts_index=arg_max, geodesic_kwargs=geodesic_kwargs)
        arg_max = arg_max+1
        inserted_neighbors = [
            chain_for_opt.nodes[arg_max-1],
            chain_for_opt.nodes[arg_max+1],
        ]
        update_status("Computing pseudo-IRC neighbor energies")
        energies = engine.compute_energies(inserted_neighbors)
        _cache_returned_energies(inserted_neighbors, energies)

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
        if (
            getattr(engine, "compute_program", "").lower() == "chemcloud"
            and hasattr(engine, "compute_geometry_optimizations")
        ):
            update_status("Submitting pseudo-IRC endpoint optimizations to Chemcloud")
            r_traj, p_traj = _run_geom_opts([candidate_r, candidate_p], engine=engine)
        else:
            update_status("Optimizing pseudo-IRC reactant endpoint")
            r_traj = _run_geom_opt(candidate_r, engine=engine)
            update_status("Optimizing pseudo-IRC product endpoint")
            p_traj = _run_geom_opt(candidate_p, engine=engine)
        r = r_traj[-1]
        n_grad_calls += len(r_traj)
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
        help="qccompute subprogram used for energies/gradients. Default: crest.",
    )
    parser.add_argument(
        "--geometry-optimizer",
        default="geometric",
        help="qccompute program used for geometry optimizations. Default: geometric.",
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
