"""Hessian normal-mode sampling: explore minima near a seed structure by
displacing along each Hessian normal mode (in both directions) and
re-optimizing.

This is a network-agnostic port of the `hessian-sample` CLI command from the
upstream neb-dynamics workflow (`neb_dynamics/scripts/main_cli.py`): it drops
the reaction-graph/provenance merging (out of scope for this minimal
package), but otherwise mirrors its behavior and output, including batch
optimization when the engine supports it, per-candidate mode/frequency
metadata, and reporting every optimized minimum (deduped) rather than
silently filtering to only those lower in energy than the seed.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import qcconst.constants as _qcconst_constants

from mepd.elementarystep import prepare_modes
from mepd.engines.engine import Engine, build_hessian_result_from_matrix
from mepd.inputs import ChainInputs
from mepd.nodes.node import Node
from mepd.nodes.nodehelpers import displace_by_dr, is_identical

# kcal/(mol*K), derived from CODATA constants rather than hardcoded.
_R_GAS_KCAL_MOL_K = (
    float(_qcconst_constants.BOLTZMANN_CONSTANT)
    * float(_qcconst_constants.AVOGADRO_NUMBER)
    / float(_qcconst_constants.KCAL_TO_JOULE)
)

# Optional progress hook: called as `on_event(event_name, payload)` at
# various points during sampling/optimization, purely for callers (e.g. the
# CLI) to render live progress -- never load-bearing for the result itself.
OnEvent = Optional[Callable[[str, dict], None]]


def _emit(on_event: OnEvent, event: str, **payload) -> None:
    if on_event is not None:
        on_event(event, payload)


@dataclass
class HessianSampleCandidate:
    mode_index: int
    direction: str  # "+" or "-"
    frequency_wavenumber: Optional[float]
    dr: Optional[float]  # bohr; None under amplitude_policy="energy" (see target_energy_kcal instead)
    effective_dr: float  # bohr; the actual per-mode displacement magnitude, under any policy
    dr_scan_index: Optional[int] = None
    mode_source: Optional[str] = None  # which of prepare_modes' sources this mode came from
    is_reaction_coordinate: bool = False  # True for an imaginary (negative-frequency) mode
    displacement_energy_kcal: Optional[float] = None  # harmonic estimate, 1/2 k_i * effective_dr^2
    amplitude_policy: str = "fixed_cartesian"
    target_energy_kcal: Optional[float] = None  # kcal/mol; set only under amplitude_policy="energy"
    # Filled in after optimization completes:
    outcome: Optional[str] = None  # "new_minimum" | "known_minimum" | "returned_to_seed_basin" | "opt_failed"
    final_energy_kcal_rel_seed: Optional[float] = None


@dataclass
class HessianSampleResult:
    seed_energy: float
    hessian_result: object
    frequencies_wavenumber: List[float]
    displaced_nodes: List[Node] = field(default_factory=list)
    displaced_metadata: List[HessianSampleCandidate] = field(default_factory=list)
    candidates_clipped: bool = False
    optimization_submission_mode: str = "serial"
    optimized_nodes: List[Node] = field(default_factory=list)
    optimized_metadata: List[HessianSampleCandidate] = field(default_factory=list)
    failed_candidates: List[dict] = field(default_factory=list)
    unique_minima: List[Node] = field(default_factory=list)
    # With validate_minima_with_hessian: one record per unique_minima entry
    # ({is_minimum, min_frequency, rescued, validation}), plus the optimized
    # structures that were still not minima after the rescue push (dropped
    # from unique_minima) and their records.
    minima_validation: List[dict] = field(default_factory=list)
    rejected_minima: List[Node] = field(default_factory=list)
    rejected_validation: List[dict] = field(default_factory=list)


def _effective_dr(seed_node: Node, dr: float) -> float:
    """Scale the requested displacement by system size so the per-atom RMS
    displacement stays roughly constant across molecule sizes, instead of a
    fixed per-mode displacement in bohr becoming vanishingly small (in
    per-atom RMS terms) for large molecules.

    `displace_by_dr` normalizes the mode to unit norm across the full 3N
    Cartesian vector, then steps by this returned value -- so the resulting
    per-atom RMS displacement is `effective_dr / sqrt(N)`. For that to equal
    `dr` regardless of N, `effective_dr` must be `dr * sqrt(N)`, not `dr * N`
    (which would make the per-atom RMS grow with sqrt(N) instead).
    """
    if float(dr) <= 0:
        raise ValueError("The Hessian-sample displacement (dr) must be positive.")
    natoms = int(np.asarray(seed_node.coords, dtype=float).shape[0])
    if natoms <= 0:
        raise ValueError("Cannot resolve a Hessian-sample displacement for an empty structure.")
    return float(dr) * float(natoms) ** 0.5


def _dedupe_minima_nodes(nodes: List[Node], chain_inputs: ChainInputs) -> List[Node]:
    """Drop duplicate minima by geometry/graph identity, using the same
    node_rms_thre/node_ene_thre thresholds RunInputs uses elsewhere (e.g.
    network-completion) for "are these the same minimum" -- rather than an
    unrelated hardcoded cutoff.
    """
    unique_nodes: List[Node] = []
    for node in nodes:
        duplicate = any(
            is_identical(
                node,
                existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False,
                collect_comparison=False,
            )
            for existing in unique_nodes
        )
        if not duplicate:
            unique_nodes.append(node.copy())
    return unique_nodes


def _classify_optimized_outcomes(
    optimized_nodes: List[Node], seed_node: Node, chain_inputs: ChainInputs,
) -> List[str]:
    """One outcome label per successfully optimized candidate, in the same
    order as `optimized_nodes`: `"returned_to_seed_basin"` (re-optimized
    straight back to the seed -- no new chemistry found), `"known_minimum"`
    (matches a minimum an earlier candidate this run already found), or
    `"new_minimum"` (the first candidate this run to land here). Purely
    diagnostic -- does not affect `unique_minima`, which is computed
    separately by `_dedupe_minima_nodes` and unchanged by this.
    """
    outcomes: List[str] = []
    unique_so_far: List[Node] = []
    for node in optimized_nodes:
        if is_identical(
            node, seed_node,
            fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
            kcal_mol_cutoff=chain_inputs.node_ene_thre,
            verbose=False, collect_comparison=False,
        ):
            outcomes.append("returned_to_seed_basin")
            continue
        duplicate = any(
            is_identical(
                node, existing,
                fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                kcal_mol_cutoff=chain_inputs.node_ene_thre,
                verbose=False, collect_comparison=False,
            )
            for existing in unique_so_far
        )
        if duplicate:
            outcomes.append("known_minimum")
        else:
            outcomes.append("new_minimum")
            unique_so_far.append(node)
    return outcomes


def _harmonic_displacement_energy_kcal(force_constant: float, effective_dr: float) -> Optional[float]:
    """Harmonic estimate of the energy this displacement costs, 1/2 k a^2
    (Hartree/bohr^2 * bohr^2 = Hartree, converted to kcal/mol) -- purely from
    already-known quantities (no extra QM calls). Negative for an imaginary
    mode (displacing along negative curvature releases energy in the
    harmonic approximation -- expected, not a bug). None if the force
    constant isn't available (e.g. no Hessian matrix could be extracted)."""
    if force_constant is None or not np.isfinite(force_constant):
        return None
    hartree = 0.5 * float(force_constant) * float(effective_dr) ** 2
    return hartree * float(_qcconst_constants.HARTREE_TO_KCAL_PER_MOL)


_DEFAULT_FORCE_CONSTANT_MIN = 1.0e-4  # Hartree/bohr^2
_DEFAULT_IMAGINARY_MODE_AMPLITUDE = 0.3  # bohr


def _amplitudes_for_modes(
    mode_set,  # ModeSet, from mepd.elementarystep.prepare_modes
    *,
    seed_node: Node,
    amplitude_policy: str,
    scan_value: float,
    imaginary_mode_amplitude: float,
    force_constant_min: float,
) -> List[Optional[float]]:
    """One displacement amplitude (bohr; the full-3N-vector norm fed to
    `displace_by_dr`) per mode in `mode_set`, or None to skip that mode
    entirely.

    `"fixed_cartesian"`: every mode gets the same amplitude, `scan_value`
    (interpreted as a target per-atom RMS displacement in bohr) scaled by
    `_effective_dr` -- a uniform Cartesian kick, unaware of how stiff or
    soft any given mode is.

    `"energy"`: `scan_value` is a target harmonic displacement energy in
    kcal/mol; each mode's amplitude is calibrated so that displacing it
    costs (in the harmonic approximation) exactly that energy --
    `a_i = sqrt(2 * E_target / k_i)` -- instead of every mode getting the
    same Cartesian distance regardless of how stiff it is (which over-
    stretches stiff bonds and under-perturbs soft ones for the same `dr`).
    A mode with `k_i` below `force_constant_min` is skipped (its amplitude
    would otherwise diverge); an imaginary mode (the reaction coordinate at
    a TS seed, `k_i < 0`) instead gets the fixed `imaginary_mode_amplitude`,
    since harmonic energy calibration is undefined for negative curvature.
    """
    if amplitude_policy == "fixed_cartesian":
        effective = _effective_dr(seed_node, scan_value)
        return [effective for _ in mode_set.freqs_wavenumber]

    if amplitude_policy == "energy":
        target_hartree = float(scan_value) / float(_qcconst_constants.HARTREE_TO_KCAL_PER_MOL)
        amplitudes: List[Optional[float]] = []
        for force_constant, freq in zip(mode_set.force_constants, mode_set.freqs_wavenumber):
            if freq < 0:
                amplitudes.append(float(imaginary_mode_amplitude))
            elif (
                force_constant is None
                or not np.isfinite(force_constant)
                or force_constant < force_constant_min
            ):
                amplitudes.append(None)
            else:
                amplitudes.append(float((2.0 * target_hartree / force_constant) ** 0.5))
        return amplitudes

    raise ValueError(
        f"amplitude_policy must be 'fixed_cartesian' or 'energy', got {amplitude_policy!r}."
    )


def _generate_mode_displacements(
    seed_node: Node,
    mode_set,  # ModeSet, from mepd.elementarystep.prepare_modes
    *,
    amplitude_policy: str,
    scan_value: float,
    max_candidates: int,
    imaginary_mode_amplitude: float = _DEFAULT_IMAGINARY_MODE_AMPLITUDE,
    force_constant_min: float = _DEFAULT_FORCE_CONSTANT_MIN,
    dr_scan_index: Optional[int] = None,
) -> tuple[List[Node], List[HessianSampleCandidate], bool]:
    """Displace `seed_node` along every normal mode (both directions),
    capped at `max_candidates`. Shared by the single-value path and each
    value of a `--full-dr-scan`/energy-target scan (in which case
    `dr_scan_index` records which scan value this batch came from).

    `scan_value` is a target per-atom RMS displacement (bohr) under
    `amplitude_policy="fixed_cartesian"`, or a target harmonic displacement
    energy (kcal/mol) under `"energy"` -- see `_amplitudes_for_modes`.
    """
    amplitudes = _amplitudes_for_modes(
        mode_set, seed_node=seed_node, amplitude_policy=amplitude_policy,
        scan_value=scan_value, imaginary_mode_amplitude=imaginary_mode_amplitude,
        force_constant_min=force_constant_min,
    )
    is_fixed_cartesian = amplitude_policy == "fixed_cartesian"

    nodes: List[Node] = []
    metadata: List[HessianSampleCandidate] = []
    clipped = False
    for mode_index, (mode, amplitude) in enumerate(zip(mode_set.vectors, amplitudes)):
        if amplitude is None:
            continue  # e.g. a near-zero force constant under "energy" policy
        freq = (
            float(mode_set.freqs_wavenumber[mode_index])
            if mode_index < len(mode_set.freqs_wavenumber) else None
        )
        force_constant = (
            mode_set.force_constants[mode_index]
            if mode_index < len(mode_set.force_constants) else None
        )
        for direction, signed_amplitude in (("+", amplitude), ("-", -amplitude)):
            nodes.append(
                displace_by_dr(node=seed_node, displacement=np.asarray(mode), dr=signed_amplitude)
            )
            metadata.append(
                HessianSampleCandidate(
                    mode_index=mode_index,
                    direction=direction,
                    frequency_wavenumber=freq,
                    dr=abs(float(scan_value)) if is_fixed_cartesian else None,
                    effective_dr=abs(float(signed_amplitude)),
                    dr_scan_index=dr_scan_index,
                    mode_source=mode_set.source,
                    is_reaction_coordinate=freq is not None and freq < 0,
                    displacement_energy_kcal=_harmonic_displacement_energy_kcal(
                        force_constant, signed_amplitude
                    ),
                    amplitude_policy=amplitude_policy,
                    target_energy_kcal=None if is_fixed_cartesian else abs(float(scan_value)),
                )
            )
            if len(nodes) >= int(max_candidates):
                clipped = True
                break
        if clipped:
            break
    return nodes, metadata, clipped


_OUTCOME_LABEL = {
    "new_minimum": "new minimum",
    "known_minimum": "found before",
    "returned_to_seed_basin": "back to seed",
    "opt_failed": "failed",
}


def _live_stream(prefix: str, index: int) -> str:
    return f"{prefix}c{index:03d}"


def _live_begin(prefix: str, index: int, total: int, node: Node, meta: HessianSampleCandidate,
                seed_energy: Optional[float], label_prefix: str = "") -> None:
    """Live view (web UI): announce candidate `index` (1-based) starting."""
    from mepd import progress as _progress

    stream = _live_stream(prefix, index)
    if _progress._stream_path(stream) is None:
        return  # nobody is watching: cost nothing
    try:
        start_xyz = node.structure.to_xyz()
    except Exception:
        start_xyz = None
    amp = f", {meta.effective_dr:.2f} bohr" if meta.effective_dr is not None else ""
    _progress.begin_minimization(
        stream,
        label=f"{label_prefix}#{index} · mode {meta.mode_index}{meta.direction}",
        caption=f"Candidate {index}/{total}: mode {meta.mode_index} displaced {meta.direction}{amp}",
        reference_energy=seed_energy, start_xyz=start_xyz,
    )


def _optimize_candidates_serially(
    engine: Engine,
    candidates: List[Node],
    metadata: List[HessianSampleCandidate],
    keywords: dict,
    *,
    on_event: OnEvent = None,
    live_prefix: str = "",
    live_label: str = "",
    seed_energy: Optional[float] = None,
) -> tuple[List[Node], List[HessianSampleCandidate], List[dict]]:
    """Optimize each candidate one at a time, isolating failures per
    candidate rather than letting one blow up the rest -- used both when the
    engine has no batch optimizer, and as a fallback when the batch call
    itself raises (see `run_hessian_sample`)."""
    optimized_nodes: List[Node] = []
    optimized_metadata: List[HessianSampleCandidate] = []
    failed_candidates: List[dict] = []
    total = len(candidates)
    from mepd import progress as _progress

    for index, (candidate, meta) in enumerate(zip(candidates, metadata), start=1):
        _live_begin(live_prefix, index, total, candidate, meta, seed_energy, live_label)
        try:
            try:
                trajectory = engine.compute_geometry_optimization(candidate, keywords=keywords)
            except TypeError:
                trajectory = engine.compute_geometry_optimization(candidate)
            if not trajectory:
                raise ValueError("optimization returned an empty trajectory")
            optimized_nodes.append(trajectory[-1])
            optimized_metadata.append(meta)
            _progress.end_minimization(_live_stream(live_prefix, index), trajectory=trajectory)
        except Exception as exc:
            failed_candidates.append({"meta": meta, "error": f"{type(exc).__name__}: {exc}"})
            _progress.end_minimization(_live_stream(live_prefix, index), status="failed",
                                       error=type(exc).__name__)
        _emit(on_event, "candidate_done", index=index, total=total)
    return optimized_nodes, optimized_metadata, failed_candidates


def run_hessian_sample(
    seed_node: Node,
    engine: Engine,
    *,
    dr: float = 0.1,
    dr_values: Optional[List[float]] = None,
    amplitude_policy: str = "fixed_cartesian",
    target_energy_kcal: float = 25.0,
    target_energy_kcal_values: Optional[List[float]] = None,
    imaginary_mode_amplitude: float = _DEFAULT_IMAGINARY_MODE_AMPLITUDE,
    max_candidates: int = 100,
    maxiter: int = 500,
    chain_inputs: ChainInputs | None = None,
    on_event: OnEvent = None,
    validate_minima_with_hessian: bool = False,
    hessian_minimum_frequency_cutoff: float = 0.0,
    hessian_minima_rescue_displacement: float = 0.1,
    live_prefix: str = "",
    live_label: str = "",
) -> HessianSampleResult:
    """Explore minima near `seed_node` by displacing along Hessian normal modes.

    `max_candidates` caps the total number of displaced geometries generated
    (and thus geometry optimizations run) -- without this cap, sampling every
    normal mode in both directions for a large molecule (3N-6 modes) would run
    an unbounded number of optimizations. `maxiter` caps the optimization step
    budget for each individual candidate.

    `amplitude_policy` controls how far each mode is displaced:

    - `"fixed_cartesian"` (default): every mode gets the same Cartesian
      distance, `dr` (a target per-atom RMS displacement in bohr, scaled by
      `sqrt(n_atoms)` to stay size-invariant). Simple, but a fixed distance
      probes very different energies on a stiff mode (e.g. an X-H stretch)
      versus a soft one (e.g. a torsion) -- a uniform `dr` large enough to
      be useful on soft modes can over-stretch stiff bonds into unphysical
      geometries, or vice versa.
    - `"energy"`: `target_energy_kcal` is a target *harmonic displacement
      energy* (kcal/mol) instead -- every mode's amplitude is calibrated
      (`a_i = sqrt(2 * E_target / k_i)`, from the mode's own harmonic force
      constant) so displacing it costs roughly that much energy, regardless
      of stiffness. An imaginary mode (the reaction coordinate at a TS seed)
      instead gets the fixed `imaginary_mode_amplitude` (bohr), since
      harmonic calibration is undefined for negative curvature.

    `dr_values`/`target_energy_kcal_values`, when given (matching the
    active `amplitude_policy`), switch to a full scan (upstream's
    `--full-dr-scan`/`--dr-scan-values`, generalized to either policy):
    every value is displaced across both directions of every normal mode,
    each value capped independently at `max_candidates` -- trading a single
    fixed displacement per mode for a denser sweep of the local potential-
    energy surface. `dr`/`target_energy_kcal` are ignored when the
    corresponding `*_values` list is given.

    Every successfully optimized candidate is kept (deduped into
    `unique_minima`) -- not just ones lower in energy than the seed -- so the
    caller can inspect and filter by whatever criterion it wants afterward.

    `on_event`, if given, is called with `(event_name, payload)` at each
    stage (`hessian_computing`, `hessian_computed`, `candidates_generated`,
    `optimizing_candidates`, `candidate_done`, `candidates_optimized`) purely
    so a caller (e.g. the CLI) can render live progress; it never affects the
    result.
    """
    if int(max_candidates) <= 0:
        raise ValueError("max_candidates must be a positive integer.")
    if int(maxiter) <= 0:
        raise ValueError("maxiter must be a positive integer.")
    if amplitude_policy not in ("fixed_cartesian", "energy"):
        raise ValueError(
            f"amplitude_policy must be 'fixed_cartesian' or 'energy', got {amplitude_policy!r}."
        )
    if dr_values is not None and amplitude_policy != "fixed_cartesian":
        raise ValueError("dr_values is only used with amplitude_policy='fixed_cartesian'.")
    if target_energy_kcal_values is not None and amplitude_policy != "energy":
        raise ValueError(
            "target_energy_kcal_values is only used with amplitude_policy='energy'."
        )
    is_scan = dr_values is not None if amplitude_policy == "fixed_cartesian" else (
        target_energy_kcal_values is not None
    )
    if amplitude_policy == "fixed_cartesian":
        scan_values = list(dr_values) if dr_values is not None else [float(dr)]
    else:
        scan_values = (
            list(target_energy_kcal_values) if target_energy_kcal_values is not None
            else [float(target_energy_kcal)]
        )
    if not scan_values:
        raise ValueError("At least one scan/target value is required.")
    if any(float(value) <= 0 for value in scan_values):
        raise ValueError("All dr/dr_values/target_energy_kcal(_values) entries must be positive.")
    if chain_inputs is None:
        chain_inputs = ChainInputs()

    if not (hasattr(engine, "_compute_hessian_result") or hasattr(engine, "compute_hessian")):
        raise ValueError(
            "This engine does not expose Hessian computation "
            "(`_compute_hessian_result` or `compute_hessian`), which is required "
            "for normal-mode sampling."
        )

    seed_energy = float(engine.compute_energies([seed_node])[0])

    _emit(on_event, "hessian_computing")
    compute_result = getattr(engine, "_compute_hessian_result", None)
    if callable(compute_result):
        hessian_result = compute_result(node=seed_node)
    else:
        hessian = engine.compute_hessian(node=seed_node)
        hessian_result = build_hessian_result_from_matrix(node=seed_node, hessian=hessian)

    mode_set = prepare_modes(hessian_result, seed_node)
    modes, freqs = mode_set.vectors, mode_set.freqs_wavenumber
    if not modes:
        raise ValueError("No normal modes were returned from the Hessian result.")
    _emit(
        on_event, "hessian_computed", n_modes=len(modes),
        n_dropped_trans_rot=mode_set.n_dropped_trans_rot, mode_source=mode_set.source,
    )

    max_candidates = int(max_candidates)
    displaced_nodes: List[Node] = []
    displaced_metadata: List[HessianSampleCandidate] = []
    clipped = False
    for index, scan_value in enumerate(scan_values):
        scan_nodes, scan_metadata, scan_clipped = _generate_mode_displacements(
            seed_node, mode_set,
            amplitude_policy=amplitude_policy, scan_value=float(scan_value),
            max_candidates=max_candidates, imaginary_mode_amplitude=imaginary_mode_amplitude,
            dr_scan_index=index if is_scan else None,
        )
        displaced_nodes.extend(scan_nodes)
        displaced_metadata.extend(scan_metadata)
        clipped = clipped or scan_clipped
    _emit(on_event, "candidates_generated", n_candidates=len(displaced_nodes), clipped=clipped)

    result = HessianSampleResult(
        seed_energy=seed_energy,
        hessian_result=hessian_result,
        frequencies_wavenumber=[float(f) for f in freqs],
        displaced_nodes=displaced_nodes,
        displaced_metadata=displaced_metadata,
        candidates_clipped=clipped,
    )

    from mepd import progress as _progress

    keywords = {"coordsys": "cart", "maxiter": int(maxiter)}
    batch_optimizer = getattr(engine, "compute_geometry_optimizations", None)
    optimized_nodes: List[Node] = []
    optimized_metadata: List[HessianSampleCandidate] = []
    failed_candidates: List[dict] = []
    total_candidates = len(displaced_nodes)
    _emit(on_event, "optimizing_candidates", total=total_candidates)

    if callable(batch_optimizer):
        # Best-effort: engines whose batch call is really a sequential loop
        # under the hood (e.g. GXTBCalculator) can accept an optional
        # `progress_callback(completed, total)` to report live per-candidate
        # progress during that one blocking call; engines that don't support
        # it (a TypeError on the attempt) fall back silently -- their
        # progress just shows up in one shot when the whole call returns.
        # Live view: a sequential "batch" reports each candidate finishing,
        # so the next one's stream starts right then (a real remote batch
        # never calls back: its streams appear when the whole call returns).
        def _batch_progress_cb(completed: int, total: int = total_candidates) -> None:
            _progress.end_minimization(_live_stream(live_prefix, completed))
            if completed < total:
                _live_begin(live_prefix, completed + 1, total, displaced_nodes[completed],
                            displaced_metadata[completed], seed_energy, live_label)
            _emit(on_event, "candidate_done", index=completed, total=total)

        if displaced_nodes:
            _live_begin(live_prefix, 1, total_candidates, displaced_nodes[0], displaced_metadata[0],
                        seed_energy, live_label)

        try:
            try:
                trajectories = batch_optimizer(
                    displaced_nodes, keywords=keywords, progress_callback=_batch_progress_cb,
                )
            except TypeError:
                try:
                    trajectories = batch_optimizer(displaced_nodes, keywords=keywords)
                except TypeError:
                    trajectories = batch_optimizer(displaced_nodes)
        except Exception:
            trajectories = None

        if trajectories is None:
            # Some engines' "batch" optimizer is really just a sequential
            # loop under the hood (no true remote batching), so one
            # candidate's failure -- commonly a non-convergent geometry
            # optimization for an aggressive displacement -- raises and
            # aborts the whole call instead of isolating it, unlike a real
            # batch backend (e.g. ChemCloud) which reports per-candidate
            # failures as empty trajectories. Fall back to optimizing each
            # candidate individually so the rest of the batch isn't lost to
            # one bad candidate.
            result.optimization_submission_mode = "batch_fallback_serial"
            optimized_nodes, optimized_metadata, failed_candidates = _optimize_candidates_serially(
                engine, displaced_nodes, displaced_metadata, keywords, on_event=on_event,
                live_prefix=live_prefix, live_label=live_label, seed_energy=seed_energy,
            )
        else:
            result.optimization_submission_mode = "batch"
            if len(trajectories) != len(displaced_nodes):
                raise ValueError(
                    "Batch geometry optimization returned a trajectory count "
                    "different from the submitted candidate count."
                )
            for index, (node, meta, trajectory) in enumerate(
                zip(displaced_nodes, displaced_metadata, trajectories), start=1
            ):
                stream = _live_stream(live_prefix, index)
                if stream not in _progress._minimizations:
                    _live_begin(live_prefix, index, total_candidates, node, meta, seed_energy, live_label)
                if trajectory:
                    optimized_nodes.append(trajectory[-1])
                    optimized_metadata.append(meta)
                    _progress.end_minimization(stream, trajectory=trajectory)
                else:
                    failed_candidates.append(
                        {"meta": meta, "error": "optimization returned an empty trajectory"}
                    )
                    _progress.end_minimization(stream, status="failed", error="no trajectory")
                _emit(on_event, "candidate_done", index=index, total=total_candidates)
    else:
        result.optimization_submission_mode = "serial"
        optimized_nodes, optimized_metadata, failed_candidates = _optimize_candidates_serially(
            engine, displaced_nodes, displaced_metadata, keywords, on_event=on_event,
            live_prefix=live_prefix, live_label=live_label, seed_energy=seed_energy,
        )

    for failed in failed_candidates:
        failed["meta"].outcome = "opt_failed"

    hartree_to_kcal = float(_qcconst_constants.HARTREE_TO_KCAL_PER_MOL)
    outcomes = _classify_optimized_outcomes(optimized_nodes, seed_node, chain_inputs)
    for node, meta, outcome in zip(optimized_nodes, optimized_metadata, outcomes):
        meta.outcome = outcome
        meta.final_energy_kcal_rel_seed = (float(node.energy) - seed_energy) * hartree_to_kcal
    stream_of = {id(meta): _live_stream(live_prefix, i) for i, meta in enumerate(displaced_metadata, start=1)}
    for meta in displaced_metadata:
        if meta.outcome in _OUTCOME_LABEL:
            _progress.set_minimization_outcome(stream_of[id(meta)], _OUTCOME_LABEL[meta.outcome])

    result.optimized_nodes = optimized_nodes
    result.optimized_metadata = optimized_metadata
    result.failed_candidates = failed_candidates
    result.unique_minima = _dedupe_minima_nodes(optimized_nodes, chain_inputs)
    _emit(
        on_event, "candidates_optimized",
        n_optimized=len(optimized_nodes), n_failed=len(failed_candidates),
    )
    if validate_minima_with_hessian and result.unique_minima:
        _validate_unique_minima(
            result, engine, chain_inputs,
            frequency_cutoff=hessian_minimum_frequency_cutoff,
            rescue_displacement=hessian_minima_rescue_displacement,
            on_event=on_event,
        )

    return result


def _validate_unique_minima(
    result: HessianSampleResult,
    engine: Engine,
    chain_inputs: ChainInputs,
    *,
    frequency_cutoff: float,
    rescue_displacement: float,
    on_event: OnEvent = None,
) -> None:
    """Hessian-check every unique minimum (optimizers can stop on a saddle
    point, e.g. an eclipsed rotor). Rescued ones replace the original; the
    list is then re-deduplicated, since a rescue can land on a minimum that
    is already in it. Ones still not minima move to `rejected_minima`."""
    from mepd.elementarystep import validate_minimum_with_rescue

    total = len(result.unique_minima)
    _emit(on_event, "validating_minima", total=total)
    kept, kept_records = [], []
    for i, node in enumerate(result.unique_minima):
        node, record = validate_minimum_with_rescue(
            node, engine, frequency_cutoff=frequency_cutoff,
            rescue_displacement=rescue_displacement, label=f"minimum {i}",
        )
        if record["is_minimum"]:
            kept.append(node)
            kept_records.append(record)
        else:
            result.rejected_minima.append(node)
            result.rejected_validation.append(record)
        _emit(on_event, "minimum_validated", index=i + 1, total=total, is_minimum=record["is_minimum"])
    deduped = _dedupe_minima_nodes(kept, chain_inputs)
    # Keep each surviving node's own record (dedupe keeps first occurrences, as copies).
    records, used = [], set()
    for node in deduped:
        for j, original in enumerate(kept):
            if j not in used and np.allclose(np.asarray(original.coords), np.asarray(node.coords)):
                records.append(kept_records[j])
                used.add(j)
                break
    result.unique_minima = deduped
    result.minima_validation = records


def _boltzmann_acceptance_probability(delta_kcal: float, temperature_kelvin: float) -> float:
    """Classic Metropolis/Boltzmann acceptance probability. A non-uphill move
    (delta_kcal <= 0) is always accepted (probability 1); an uphill move is
    accepted with probability exp(-delta_kcal / (R*T))."""
    if delta_kcal <= 0:
        return 1.0
    exponent = -float(delta_kcal) / (_R_GAS_KCAL_MOL_K * float(temperature_kelvin))
    return float(min(1.0, np.exp(exponent)))


def _candidate_acceptance_draw(random_seed: Optional[int], generation_index: int) -> float:
    """A uniform [0, 1) draw for the acceptance test. When `random_seed` is
    given, the draw is a deterministic hash of (seed, generation_index) --
    not a stateful RNG stream -- so re-running with the same seed reproduces
    identical accept/reject outcomes regardless of how many candidates were
    evaluated before this one. Falls back to real randomness when no seed is
    given."""
    if random_seed is None:
        return random.random()
    digest = hashlib.blake2b(f"{random_seed}:{generation_index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


@dataclass
class HessianGlobalOptResult:
    start_energy: float
    rounds_run: int = 0
    stopped_reason: str = "queue_exhausted"  # or "max_rounds"
    accepted_minima: List[Node] = field(default_factory=list)
    round_summaries: List[dict] = field(default_factory=list)


def run_hessian_global_optimization(
    seed_node: Node,
    engine: Engine,
    *,
    dr: float = 0.1,
    dr_values: Optional[List[float]] = None,
    max_candidates: int = 100,
    maxiter: int = 500,
    temperature: float = 298.15,
    energy_tolerance_kcal: float = 1.0e-4,
    max_rounds: int = 100,
    random_seed: Optional[int] = None,
    chain_inputs: ChainInputs | None = None,
    acceptance_baseline: str = "connected",
    on_event: OnEvent = None,
    validate_minima_with_hessian: bool = False,
    hessian_minimum_frequency_cutoff: float = 0.0,
    hessian_minima_rescue_displacement: float = 0.1,
) -> HessianGlobalOptResult:
    """Basin-hopping-style global optimization built on repeated Hessian
    sampling.

    Each round Hessian-samples from every currently-queued accepted minimum
    (a breadth-first expansion -- not "chase a single best/last-accepted
    structure"). A newly-found minimum is accepted via the Metropolis/
    Boltzmann criterion (`temperature`): moves within `energy_tolerance_kcal`
    of the acceptance baseline are treated as flat (always accepted);
    uphill moves beyond it are accepted with probability
    exp(-deltaE / (R*T)). Every accepted, globally-unique minimum becomes a
    seed for the next round. Stops when the queue empties (`stopped_reason
    == "queue_exhausted"`) or after `max_rounds` rounds
    (`"max_rounds"`) -- the control that bounds this otherwise-unbounded
    search.

    `acceptance_baseline` selects what each candidate's energy is compared
    against:

    - `"connected"` (default): the specific source structure this candidate
      was Hessian-sampled from -- standard Metropolis basin-hopping
      semantics (accept/reject relative to the state you stepped from).
      This is the fix for a real bug in the old, only behavior (comparing
      every round to the *original* seed's energy): seeded from a
      high-energy structure (e.g. a transition-state guess), every later
      round's candidates would look unconditionally downhill relative to
      that fixed, stale baseline and the Metropolis test would never
      actually discriminate -- the search would just accumulate whatever it
      found until `max_rounds`, never using its own progress. `"connected"`
      makes the baseline track the search as it moves to new minima.
    - `"seed"`: always the original seed's energy (the old, only behavior).
      Kept for anyone who explicitly wants that.
    - `"running_best"`: the lowest energy found so far (seed included) --
      a more greedy variant that only rewards genuine improvement over the
      whole run rather than each local step.

    `dr_values`, when given, is forwarded to every round's Hessian sampling
    as a full displacement scan (see `run_hessian_sample`) instead of the
    single fixed `dr` -- matching upstream's `--full-dr-scan`.

    `on_event`, if given, is called with `(event_name, payload)` for live
    progress -- round/source-level events (`round_start`, `source_start`,
    `round_done`), `minimum_accepted` the instant each new minimum is
    accepted (payload includes the `node` itself and `rel_energy_kcal`, so a
    caller can stream/write results out without waiting for the whole
    search), plus every `run_hessian_sample` event forwarded as-is from
    whichever source is currently being sampled.
    """
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    if int(max_rounds) < 1:
        raise ValueError("max_rounds must be a positive integer.")
    if acceptance_baseline not in ("seed", "connected", "running_best"):
        raise ValueError(
            "acceptance_baseline must be 'seed', 'connected', or 'running_best', "
            f"got {acceptance_baseline!r}."
        )
    if chain_inputs is None:
        chain_inputs = ChainInputs()

    from qcconst.constants import HARTREE_TO_KCAL_PER_MOL

    hartree_to_kcal = float(HARTREE_TO_KCAL_PER_MOL)
    start_energy = float(engine.compute_energies([seed_node])[0])
    best_energy = start_energy  # only tracked/used for acceptance_baseline="running_best"

    accepted_minima: List[Node] = []
    # Dedup pool starts with the seed itself (not reported in accepted_minima,
    # which is "newly found" minima) -- otherwise the search can rediscover
    # the starting structure indefinitely: it is always isoenergetic with
    # itself, so it would pass the energy-tolerance auto-accept branch every
    # time a branch happens to lead back to it, re-queuing forever.
    known_structures: List[Node] = [seed_node]
    queue: List[Node] = [seed_node]
    round_summaries: List[dict] = []
    generation_index = 0
    rounds_run = 0
    stopped_reason = "max_rounds"

    for round_index in range(int(max_rounds)):
        if not queue:
            stopped_reason = "queue_exhausted"
            break
        rounds_run = round_index + 1
        current_queue, queue = queue, []
        candidates_optimized = 0
        accepted_this_round = 0
        source_errors: List[str] = []
        _emit(
            on_event, "round_start",
            round=round_index, max_rounds=int(max_rounds), n_sources=len(current_queue),
        )

        for source_index, source_node in enumerate(current_queue):
            _emit(
                on_event, "source_start",
                round=round_index, source_index=source_index, n_sources=len(current_queue),
            )
            try:
                sample = run_hessian_sample(
                    source_node, engine, dr=dr, dr_values=dr_values, max_candidates=max_candidates,
                    maxiter=maxiter, chain_inputs=chain_inputs, on_event=on_event,
                    # Only validated minima get accepted -- and so seed later rounds.
                    validate_minima_with_hessian=validate_minima_with_hessian,
                    hessian_minimum_frequency_cutoff=hessian_minimum_frequency_cutoff,
                    hessian_minima_rescue_displacement=hessian_minima_rescue_displacement,
                    # Live view: one stream per candidate of every round/source.
                    live_prefix=f"r{round_index + 1:02d}s{source_index + 1:02d}",
                    live_label=(f"round {round_index + 1} · " if len(current_queue) == 1
                                else f"round {round_index + 1} · source {source_index + 1} · "),
                )
            except Exception as exc:
                # A source failing outright (e.g. its Hessian computation
                # itself errors) shouldn't silently look like "no minima
                # found" -- record it so callers/summaries can surface it,
                # then move on to the other queued sources.
                source_errors.append(f"{type(exc).__name__}: {exc}")
                continue
            candidates_optimized += len(sample.optimized_nodes)

            if acceptance_baseline == "seed":
                baseline_energy = start_energy
            elif acceptance_baseline == "connected":
                baseline_energy = float(source_node.energy)
            else:  # "running_best"
                baseline_energy = best_energy

            for candidate in sample.unique_minima:
                # Always reported relative to the original seed (an
                # intuitive, fixed reference point for the caller), even
                # though acceptance itself is decided against
                # `baseline_energy`, which may be a different reference.
                report_delta_kcal = (float(candidate.energy) - start_energy) * hartree_to_kcal
                acceptance_delta_kcal = (float(candidate.energy) - baseline_energy) * hartree_to_kcal
                effective_delta = (
                    0.0 if abs(acceptance_delta_kcal) <= energy_tolerance_kcal
                    else acceptance_delta_kcal
                )
                if effective_delta <= 0:
                    accept = True
                else:
                    probability = _boltzmann_acceptance_probability(effective_delta, temperature)
                    accept = _candidate_acceptance_draw(random_seed, generation_index) < probability
                generation_index += 1
                if not accept:
                    continue

                is_duplicate = any(
                    is_identical(
                        candidate, existing,
                        fragment_rmsd_cutoff=chain_inputs.node_rms_thre,
                        kcal_mol_cutoff=chain_inputs.node_ene_thre,
                        verbose=False, collect_comparison=False,
                    )
                    for existing in known_structures
                )
                if is_duplicate:
                    continue

                accepted_minima.append(candidate)
                known_structures.append(candidate)
                queue.append(candidate)
                accepted_this_round += 1
                best_energy = min(best_energy, float(candidate.energy))
                _emit(
                    on_event, "minimum_accepted",
                    round=round_index, index=len(accepted_minima),
                    node=candidate, rel_energy_kcal=report_delta_kcal,
                )

        round_summaries.append({
            "round": round_index,
            "sources": len(current_queue),
            "candidates_optimized": candidates_optimized,
            "accepted": accepted_this_round,
            "source_errors": source_errors,
        })
        _emit(
            on_event, "round_done",
            round=round_index, max_rounds=int(max_rounds),
            candidates_optimized=candidates_optimized, accepted=accepted_this_round,
        )

    return HessianGlobalOptResult(
        start_energy=start_energy,
        rounds_run=rounds_run,
        stopped_reason=stopped_reason,
        accepted_minima=accepted_minima,
        round_summaries=round_summaries,
    )
