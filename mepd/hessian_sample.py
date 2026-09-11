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

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from mepd.elementarystep import _extract_hessian_modes_and_frequencies
from mepd.engines.engine import Engine, build_hessian_result_from_matrix
from mepd.inputs import ChainInputs
from mepd.nodes.node import Node
from mepd.nodes.nodehelpers import displace_by_dr, is_identical


@dataclass
class HessianSampleCandidate:
    mode_index: int
    direction: str  # "+" or "-"
    frequency_wavenumber: Optional[float]
    dr: float
    effective_dr: float


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
    """Scale the requested displacement by system size, matching the
    upstream convention -- a fixed per-mode displacement in bohr becomes
    vanishingly small (in per-atom RMS terms) for large molecules otherwise.
    """
    if float(dr) <= 0:
        raise ValueError("The Hessian-sample displacement (dr) must be positive.")
    natoms = int(np.asarray(seed_node.coords, dtype=float).shape[0])
    if natoms <= 0:
        raise ValueError("Cannot resolve a Hessian-sample displacement for an empty structure.")
    return float(dr) * float(natoms)


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


def run_hessian_sample(
    seed_node: Node,
    engine: Engine,
    *,
    dr: float = 0.1,
    max_candidates: int = 100,
    maxiter: int = 500,
    chain_inputs: ChainInputs | None = None,
) -> HessianSampleResult:
    """Explore minima near `seed_node` by displacing along Hessian normal modes.

    `max_candidates` caps the total number of displaced geometries generated
    (and thus geometry optimizations run) -- without this cap, sampling every
    normal mode in both directions for a large molecule (3N-6 modes) would run
    an unbounded number of optimizations. `maxiter` caps the optimization step
    budget for each individual candidate.

    Every successfully optimized candidate is kept (deduped into
    `unique_minima`) -- not just ones lower in energy than the seed -- so the
    caller can inspect and filter by whatever criterion it wants afterward.
    """
    if int(max_candidates) <= 0:
        raise ValueError("max_candidates must be a positive integer.")
    if int(maxiter) <= 0:
        raise ValueError("maxiter must be a positive integer.")
    if chain_inputs is None:
        chain_inputs = ChainInputs()

    if not (hasattr(engine, "_compute_hessian_result") or hasattr(engine, "compute_hessian")):
        raise ValueError(
            "This engine does not expose Hessian computation "
            "(`_compute_hessian_result` or `compute_hessian`), which is required "
            "for normal-mode sampling."
        )

    seed_energy = float(engine.compute_energies([seed_node])[0])

    compute_result = getattr(engine, "_compute_hessian_result", None)
    if callable(compute_result):
        hessian_result = compute_result(node=seed_node)
    else:
        hessian = engine.compute_hessian(node=seed_node)
        hessian_result = build_hessian_result_from_matrix(node=seed_node, hessian=hessian)

    modes, freqs = _extract_hessian_modes_and_frequencies(hessian_result, seed_node)
    if not modes:
        raise ValueError("No normal modes were returned from the Hessian result.")

    scaled_dr = _effective_dr(seed_node, dr)
    max_candidates = int(max_candidates)
    displaced_nodes: List[Node] = []
    displaced_metadata: List[HessianSampleCandidate] = []
    clipped = False
    for mode_index, mode in enumerate(modes):
        freq = float(freqs[mode_index]) if mode_index < len(freqs) else None
        for direction, signed_dr in (("+", scaled_dr), ("-", -scaled_dr)):
            displaced_nodes.append(
                displace_by_dr(node=seed_node, displacement=np.asarray(mode), dr=signed_dr)
            )
            displaced_metadata.append(
                HessianSampleCandidate(
                    mode_index=mode_index,
                    direction=direction,
                    frequency_wavenumber=freq,
                    dr=abs(float(dr)),
                    effective_dr=abs(float(signed_dr)),
                )
            )
            if len(displaced_nodes) >= max_candidates:
                clipped = True
                break
        if clipped:
            break

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

    if callable(batch_optimizer):
        result.optimization_submission_mode = "batch"
        try:
            trajectories = batch_optimizer(displaced_nodes, keywords=keywords)
        except TypeError:
            trajectories = batch_optimizer(displaced_nodes)
        if len(trajectories) != len(displaced_nodes):
            raise ValueError(
                "Batch geometry optimization returned a trajectory count "
                "different from the submitted candidate count."
            )
        for meta, trajectory in zip(displaced_metadata, trajectories):
            if trajectory:
                optimized_nodes.append(trajectory[-1])
                optimized_metadata.append(meta)
            else:
                failed_candidates.append(
                    {"meta": meta, "error": "optimization returned an empty trajectory"}
                )
    else:
        result.optimization_submission_mode = "serial"
        for candidate, meta in zip(displaced_nodes, displaced_metadata):
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

    result.optimized_nodes = optimized_nodes
    result.optimized_metadata = optimized_metadata
    result.failed_candidates = failed_candidates
    result.unique_minima = _dedupe_minima_nodes(optimized_nodes, chain_inputs)

    return result
