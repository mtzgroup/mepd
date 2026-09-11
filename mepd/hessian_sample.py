"""Hessian normal-mode sampling: explore minima near a seed structure by
displacing along each Hessian normal mode (in both directions) and
re-optimizing, keeping the resulting minima that are lower in energy than the
seed and distinct from it and from each other.

This is a simplified, network-agnostic version of the `_run_hessian_sample`
routine from the upstream neb-dynamics retropaths workflow: it drops the
reaction-graph/provenance merging (out of scope for this minimal package) and
just reports the accepted minima.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from mepd.elementarystep import _extract_hessian_modes_and_frequencies, _run_geom_opt
from mepd.engines.engine import Engine, build_hessian_result_from_matrix
from mepd.nodes.node import Node
from mepd.nodes.nodehelpers import displace_by_dr, is_identical


@dataclass
class HessianSampleResult:
    seed_energy: float
    candidates_generated: int = 0
    candidates_clipped: bool = False
    minima: List[Node] = field(default_factory=list)
    skipped_not_lower_energy: int = 0
    skipped_duplicate: int = 0
    skipped_failed_optimization: int = 0


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


def run_hessian_sample(
    seed_node: Node,
    engine: Engine,
    *,
    dr: float = 0.1,
    max_candidates: int = 100,
) -> HessianSampleResult:
    """Explore minima near `seed_node` by displacing along Hessian normal modes.

    `max_candidates` caps the total number of displaced geometries generated
    (and thus geometry optimizations run) -- without this cap, sampling every
    normal mode in both directions for a large molecule (3N-6 modes) would run
    an unbounded number of optimizations.
    """
    if int(max_candidates) <= 0:
        raise ValueError("max_candidates must be a positive integer.")

    seed_energy = float(engine.compute_energies([seed_node])[0])

    compute_result = getattr(engine, "_compute_hessian_result", None)
    if callable(compute_result):
        hessian_result = compute_result(node=seed_node)
    else:
        hessian = engine.compute_hessian(node=seed_node)
        hessian_result = build_hessian_result_from_matrix(node=seed_node, hessian=hessian)

    modes, _freqs = _extract_hessian_modes_and_frequencies(hessian_result, seed_node)
    if not modes:
        raise ValueError("No normal modes were returned from the Hessian result.")

    scaled_dr = _effective_dr(seed_node, dr)
    max_candidates = int(max_candidates)
    candidates: List[Node] = []
    clipped = False
    for mode in modes:
        for signed_dr in (scaled_dr, -scaled_dr):
            candidates.append(
                displace_by_dr(node=seed_node, displacement=np.asarray(mode), dr=signed_dr)
            )
            if len(candidates) >= max_candidates:
                clipped = True
                break
        if clipped:
            break

    result = HessianSampleResult(
        seed_energy=seed_energy,
        candidates_generated=len(candidates),
        candidates_clipped=clipped,
    )

    accepted: List[Node] = []
    for candidate in candidates:
        try:
            optimized = _run_geom_opt(candidate, engine=engine)[-1]
        except Exception:
            result.skipped_failed_optimization += 1
            continue

        if optimized.energy >= seed_energy:
            result.skipped_not_lower_energy += 1
            continue

        if is_identical(optimized, seed_node, verbose=False, collect_comparison=False):
            result.skipped_duplicate += 1
            continue
        if any(
            is_identical(optimized, existing, verbose=False, collect_comparison=False)
            for existing in accepted
        ):
            result.skipped_duplicate += 1
            continue

        accepted.append(optimized)

    result.minima = accepted
    return result
