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

from mepd.elementarystep import _extract_hessian_modes_and_frequencies
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
    dr: float
    effective_dr: float
    dr_scan_index: Optional[int] = None


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


def _generate_dr_displacements(
    seed_node: Node,
    modes: List[np.ndarray],
    freqs: List[float],
    *,
    dr: float,
    max_candidates: int,
    dr_scan_index: Optional[int] = None,
) -> tuple[List[Node], List[HessianSampleCandidate], bool]:
    """Displace `seed_node` along every normal mode (both directions) by a
    single `dr`, capped at `max_candidates`. Shared by the single-dr path and
    each value of a `--full-dr-scan` sweep (in which case `dr_scan_index`
    records which scan value this batch came from)."""
    scaled_dr = _effective_dr(seed_node, dr)
    nodes: List[Node] = []
    metadata: List[HessianSampleCandidate] = []
    clipped = False
    for mode_index, mode in enumerate(modes):
        freq = float(freqs[mode_index]) if mode_index < len(freqs) else None
        for direction, signed_dr in (("+", scaled_dr), ("-", -scaled_dr)):
            nodes.append(
                displace_by_dr(node=seed_node, displacement=np.asarray(mode), dr=signed_dr)
            )
            metadata.append(
                HessianSampleCandidate(
                    mode_index=mode_index,
                    direction=direction,
                    frequency_wavenumber=freq,
                    dr=abs(float(dr)),
                    effective_dr=abs(float(signed_dr)),
                    dr_scan_index=dr_scan_index,
                )
            )
            if len(nodes) >= int(max_candidates):
                clipped = True
                break
        if clipped:
            break
    return nodes, metadata, clipped


def _optimize_candidates_serially(
    engine: Engine,
    candidates: List[Node],
    metadata: List[HessianSampleCandidate],
    keywords: dict,
    *,
    on_event: OnEvent = None,
) -> tuple[List[Node], List[HessianSampleCandidate], List[dict]]:
    """Optimize each candidate one at a time, isolating failures per
    candidate rather than letting one blow up the rest -- used both when the
    engine has no batch optimizer, and as a fallback when the batch call
    itself raises (see `run_hessian_sample`)."""
    optimized_nodes: List[Node] = []
    optimized_metadata: List[HessianSampleCandidate] = []
    failed_candidates: List[dict] = []
    total = len(candidates)
    for index, (candidate, meta) in enumerate(zip(candidates, metadata), start=1):
        try:
            try:
                trajectory = engine.compute_geometry_optimization(candidate, keywords=keywords)
            except TypeError:
                trajectory = engine.compute_geometry_optimization(candidate)
            if not trajectory:
                raise ValueError("optimization returned an empty trajectory")
            optimized_nodes.append(trajectory[-1])
            optimized_metadata.append(meta)
        except Exception as exc:
            failed_candidates.append({"meta": meta, "error": f"{type(exc).__name__}: {exc}"})
        _emit(on_event, "candidate_done", index=index, total=total)
    return optimized_nodes, optimized_metadata, failed_candidates


def run_hessian_sample(
    seed_node: Node,
    engine: Engine,
    *,
    dr: float = 0.1,
    dr_values: Optional[List[float]] = None,
    max_candidates: int = 100,
    maxiter: int = 500,
    chain_inputs: ChainInputs | None = None,
    on_event: OnEvent = None,
) -> HessianSampleResult:
    """Explore minima near `seed_node` by displacing along Hessian normal modes.

    `max_candidates` caps the total number of displaced geometries generated
    (and thus geometry optimizations run) -- without this cap, sampling every
    normal mode in both directions for a large molecule (3N-6 modes) would run
    an unbounded number of optimizations. `maxiter` caps the optimization step
    budget for each individual candidate.

    `dr_values`, when given, switches to a full displacement scan (matching
    upstream's `--full-dr-scan`/`--dr-scan-values`): every value is displaced
    across both directions of every normal mode, each value capped
    independently at `max_candidates` -- trading a single fixed displacement
    per mode for a denser sweep of the local potential-energy surface. `dr` is
    ignored when `dr_values` is given.

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
    if dr_values is not None:
        if len(dr_values) == 0:
            raise ValueError("dr_values must contain at least one positive value.")
        if any(float(value) <= 0 for value in dr_values):
            raise ValueError("dr_values entries must be positive.")
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

    modes, freqs = _extract_hessian_modes_and_frequencies(hessian_result, seed_node)
    if not modes:
        raise ValueError("No normal modes were returned from the Hessian result.")
    _emit(on_event, "hessian_computed", n_modes=len(modes))

    max_candidates = int(max_candidates)
    displaced_nodes: List[Node] = []
    displaced_metadata: List[HessianSampleCandidate] = []
    clipped = False
    if dr_values:
        for scan_index, scan_dr in enumerate(dr_values):
            scan_nodes, scan_metadata, scan_clipped = _generate_dr_displacements(
                seed_node, modes, freqs,
                dr=float(scan_dr), max_candidates=max_candidates, dr_scan_index=scan_index,
            )
            displaced_nodes.extend(scan_nodes)
            displaced_metadata.extend(scan_metadata)
            clipped = clipped or scan_clipped
    else:
        displaced_nodes, displaced_metadata, clipped = _generate_dr_displacements(
            seed_node, modes, freqs, dr=float(dr), max_candidates=max_candidates,
        )
    _emit(on_event, "candidates_generated", n_candidates=len(displaced_nodes), clipped=clipped)

    result = HessianSampleResult(
        seed_energy=seed_energy,
        hessian_result=hessian_result,
        frequencies_wavenumber=[float(f) for f in freqs],
        displaced_nodes=displaced_nodes,
        displaced_metadata=displaced_metadata,
        candidates_clipped=clipped,
    )

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
        def _batch_progress_cb(completed: int, total: int = total_candidates) -> None:
            _emit(on_event, "candidate_done", index=completed, total=total)

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
            )
        else:
            result.optimization_submission_mode = "batch"
            if len(trajectories) != len(displaced_nodes):
                raise ValueError(
                    "Batch geometry optimization returned a trajectory count "
                    "different from the submitted candidate count."
                )
            for index, (meta, trajectory) in enumerate(zip(displaced_metadata, trajectories), start=1):
                if trajectory:
                    optimized_nodes.append(trajectory[-1])
                    optimized_metadata.append(meta)
                else:
                    failed_candidates.append(
                        {"meta": meta, "error": "optimization returned an empty trajectory"}
                    )
                _emit(on_event, "candidate_done", index=index, total=total_candidates)
    else:
        result.optimization_submission_mode = "serial"
        optimized_nodes, optimized_metadata, failed_candidates = _optimize_candidates_serially(
            engine, displaced_nodes, displaced_metadata, keywords, on_event=on_event,
        )

    result.optimized_nodes = optimized_nodes
    result.optimized_metadata = optimized_metadata
    result.failed_candidates = failed_candidates
    result.unique_minima = _dedupe_minima_nodes(optimized_nodes, chain_inputs)
    _emit(
        on_event, "candidates_optimized",
        n_optimized=len(optimized_nodes), n_failed=len(failed_candidates),
    )

    return result


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
    on_event: OnEvent = None,
) -> HessianGlobalOptResult:
    """Basin-hopping-style global optimization built on repeated Hessian
    sampling.

    Each round Hessian-samples from every currently-queued accepted minimum
    (a breadth-first expansion -- not "chase a single best/last-accepted
    structure"). A newly-found minimum is accepted via the Metropolis/
    Boltzmann criterion (`temperature`), evaluated against the *original*
    seed's energy, not its immediate source -- moves within
    `energy_tolerance_kcal` of that baseline are treated as flat (always
    accepted); uphill moves beyond it are accepted with probability
    exp(-deltaE / (R*T)). Every accepted, globally-unique minimum becomes a
    seed for the next round. Stops when the queue empties (`stopped_reason
    == "queue_exhausted"`) or after `max_rounds` rounds
    (`"max_rounds"`) -- the control that bounds this otherwise-unbounded
    search.

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
    if chain_inputs is None:
        chain_inputs = ChainInputs()

    from qcconst.constants import HARTREE_TO_KCAL_PER_MOL

    hartree_to_kcal = float(HARTREE_TO_KCAL_PER_MOL)
    start_energy = float(engine.compute_energies([seed_node])[0])

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
                )
            except Exception as exc:
                # A source failing outright (e.g. its Hessian computation
                # itself errors) shouldn't silently look like "no minima
                # found" -- record it so callers/summaries can surface it,
                # then move on to the other queued sources.
                source_errors.append(f"{type(exc).__name__}: {exc}")
                continue
            candidates_optimized += len(sample.optimized_nodes)

            for candidate in sample.unique_minima:
                delta_kcal = (float(candidate.energy) - start_energy) * hartree_to_kcal
                effective_delta = 0.0 if abs(delta_kcal) <= energy_tolerance_kcal else delta_kcal
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
                _emit(
                    on_event, "minimum_accepted",
                    round=round_index, index=len(accepted_minima),
                    node=candidate, rel_energy_kcal=delta_kcal,
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
