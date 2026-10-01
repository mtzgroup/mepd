"""Pick among several candidate atom-to-atom mappings (or no remapping at
all) between two reaction endpoints, using a geodesic-interpolated-path
metric -- generalizes the old single-mapping "reindex or not" veto
(`mepd/cli.py::_check_endpoint_atom_mapping`) into an N-way comparison
across {identity ordering, candidate mapping 1, ..., candidate mapping k}.

Which metric is actually the best predictor of a "correct" mapping is not
settled up front: `"gi-energy"` (an actual QM energy evaluation along the
geodesic path) is the most direct proxy but the most expensive per
candidate; `"geodesic-distance"` and `"path-rmsd"` are effectively free
byproducts of the same interpolation, but unvalidated as mapping-quality
signals; `"endpoint-rmsd"` skips the interpolation altogether, so it is
cheaper again and weaker again (see `score_candidate`). All four are kept
pluggable, and `score_candidate_all_metrics` lets a caller (see `cli.py`'s
`--debug-dump` handling) record every one of them for every candidate so
real runs double as comparative data.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from qcdata.models.structure import Structure

from mepd.atom_mapping import AtomMapping, realign_end_to_start
from mepd.atom_mapping_metrics import METRICS  # noqa: F401  (re-exported)
from mepd.chain import Chain


@dataclass
class MappingCandidate:
    label: str  # "identity" or "mapping_<i>"
    end_structure: Structure
    atom_map: Optional[AtomMapping]  # None for identity


@dataclass
class SelectionResult:
    winner: MappingCandidate
    scores: dict[str, float] = field(default_factory=dict)  # label -> score under the selection metric
    chains: dict[str, Chain] = field(default_factory=dict)  # label -> its geodesic-interpolated chain


def build_candidates(
    start_structure: Structure, end_structure: Structure, atom_maps: list[AtomMapping]
) -> list[MappingCandidate]:
    """`end_structure` unchanged ("identity"), plus one candidate per
    non-identity mapping in `atom_maps` -- de-duplicated if two mappings
    happen to produce the same atom order (SLAPMapper's tied candidates can
    coincide after symmetry, even though `_remove_isomorphic_results`
    already dedupes most of those upstream). Identity is left out when the
    end's element order differs from the start's: it would pair up
    mismatched atoms, so its score means nothing."""
    same_order = list(start_structure.symbols) == list(end_structure.symbols)
    candidates = [MappingCandidate(label="identity", end_structure=end_structure, atom_map=None)] if same_order else []
    seen_orders = {tuple(range(len(end_structure.symbols)))}
    i = 0
    for atom_map in atom_maps:
        if atom_map.is_identity:
            continue
        order = tuple(atom_map.as_order())
        if order in seen_orders:
            continue
        seen_orders.add(order)
        candidates.append(
            MappingCandidate(
                label=f"mapping_{i}",
                end_structure=realign_end_to_start(atom_map, end_structure),
                atom_map=atom_map,
            )
        )
        i += 1
    return candidates


def _aligned_rmsd(a: Structure, b: Structure) -> float:
    """Kabsch-aligned RMSD (bohr) between two identically-indexed
    structures -- no interpolation, no engine, microseconds not seconds.
    Backs the "endpoint-rmsd" metric: unlike the other three, it says
    nothing about what happens ALONG the path between `a` and `b`, only
    how far apart the two fixed endpoints are."""
    import numpy as np

    from mepd.rigid_alignment import kabsch_align

    x = np.asarray(a.geometry, dtype=float)
    y = np.asarray(b.geometry, dtype=float)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    aligned = kabsch_align(x, y)
    return float(np.sqrt(((aligned - y) ** 2).sum(axis=1).mean()))


def _interpolate(candidate: MappingCandidate, start_structure: Structure, run_inputs):
    import mepd.chainhelpers as ch
    from mepd.nodes.node import StructureNode

    start_node = StructureNode(structure=start_structure)
    end_node = StructureNode(structure=candidate.end_structure)
    seed_chain = Chain.model_validate({
        "nodes": [start_node, end_node],
        "parameters": copy.deepcopy(run_inputs.chain_inputs),
    })
    chain, smoother = ch.run_geodesic(
        chain=seed_chain,
        chain_inputs=copy.deepcopy(run_inputs.chain_inputs),
        nimages=run_inputs.gi_inputs.nimages,
        friction=run_inputs.gi_inputs.friction,
        nudge=run_inputs.gi_inputs.nudge,
        random_seed=run_inputs.gi_inputs.random_seed,
        return_smoother=True,
    )
    return chain, smoother


def rmsd_window(run_inputs) -> float:
    ami = getattr(run_inputs, "atom_mapping_inputs", None)
    return float(getattr(ami, "rmsd_window", 1.0) if ami is not None else 1.0)


def lowest_rmsd_variants(candidates: list, start_structure: Structure, run_inputs, *, always: tuple = ()) -> list:
    """rmsd-geodesic: the `gi_variant_cap` lowest-endpoint-RMSD candidates (plus `always`)."""
    cap = int(getattr(getattr(run_inputs, "atom_mapping_inputs", None), "gi_variant_cap", 20))
    if cap <= 0 or len(candidates) <= cap:
        return list(candidates)
    rmsd = [_aligned_rmsd(start_structure, c.end_structure) for c in candidates]
    keep = set(sorted(range(len(candidates)), key=rmsd.__getitem__)[:cap])
    keep |= {k for k, c in enumerate(candidates) if c.label in always}
    return [c for k, c in enumerate(candidates) if k in keep]


def near_lowest(values: list[float], window: float = 1.0, *, keep: int = 1) -> list[int]:
    """Indices of the values within `window` standard deviations of the
    lowest (the "degenerate" ones, as far as that score can tell), lowest
    first; at least `keep` of them (the next-lowest fill up), all when they
    are all equal."""
    import numpy as np

    if not values:
        return []
    v = np.asarray(values, dtype=float)
    order = [int(i) for i in np.argsort(v, kind="stable")]
    cut = float(v.min()) + window * float(v.std())
    inside = [i for i in order if v[i] <= cut + 1e-12]
    return inside if len(inside) >= keep else order[:keep]


def score_candidate(
    candidate: MappingCandidate, metric: str, start_structure: Structure, run_inputs,
) -> tuple[float, Optional[Chain]]:
    """Reduces `candidate` to a single score under `metric` -- lower is
    always "better". `endpoint-rmsd` is the odd one out: it skips the
    geodesic interpolation entirely (returns `chain=None`), so it's orders
    of magnitude cheaper than the other three but, per its own docstring
    (`_aligned_rmsd`), a weaker signal -- untested at scale against the
    others before relying on it for something consequential; see
    docs/channels_candidates.md's open-problem note on mapping cost."""
    if metric == "endpoint-rmsd":
        return _aligned_rmsd(start_structure, candidate.end_structure), None

    chain, smoother = _interpolate(candidate, start_structure, run_inputs)

    # "rmsd-geodesic" filters conformer PAIRS by endpoint RMSD (mepd channels);
    # a candidate that is scored at all is scored by its GI path.
    if metric in ("geodesic-distance", "rmsd-geodesic"):
        return float(smoother.length), chain
    if metric == "path-rmsd":
        return float(chain.path_length[-1]), chain
    if metric == "gi-energy":
        run_inputs.engine.compute_energies(chain)
        return float(max(chain.energies_kcalmol)), chain
    raise ValueError(f"Unknown atom-mapping selection metric '{metric}'. Known: {METRICS}.")


def score_candidate_all_metrics(
    candidate: MappingCandidate, start_structure: Structure, run_inputs,
) -> tuple[dict[str, float], Chain]:
    """Like `score_candidate`, but computes every metric from a single
    interpolation (geodesic-distance and path-rmsd are free byproducts of
    it; gi-energy is the only one requiring an extra QM evaluation;
    endpoint-rmsd needs no interpolation at all) -- used for `--debug-dump`
    so every candidate's data is directly comparable across metrics."""
    chain, smoother = _interpolate(candidate, start_structure, run_inputs)
    run_inputs.engine.compute_energies(chain)
    scores = {
        "geodesic-distance": float(smoother.length),
        "path-rmsd": float(chain.path_length[-1]),
        "gi-energy": float(max(chain.energies_kcalmol)),
        "endpoint-rmsd": _aligned_rmsd(start_structure, candidate.end_structure),
    }
    scores["rmsd-geodesic"] = scores["geodesic-distance"]   # a scored candidate is scored by its GI path
    return scores, chain


def select_best_candidate(
    candidates: list[MappingCandidate],
    metric: str,
    start_structure: Structure,
    run_inputs,
    veto_margin: float = 0.0,
) -> SelectionResult:
    """Scores every candidate under `metric` and picks the lowest-scoring
    one -- "identity" (don't reindex) is just one more candidate in the
    pool, not a privileged default. A non-identity winner is only adopted
    if it beats identity's score by more than `veto_margin`; otherwise
    identity is kept, even if some other candidate scored marginally
    better (a stability guard against metric noise, opt-in via
    `--atom-mapping-veto-margin`; the default of 0.0 is a pure best-of-N
    with identity winning exact ties)."""
    scores: dict[str, float] = {}
    chains: dict[str, Chain] = {}
    identity = next((c for c in candidates if c.label == "identity"), None)
    if metric == "rmsd-geodesic":
        candidates = lowest_rmsd_variants(candidates, start_structure, run_inputs,
                                          always=(identity.label,) if identity else ())
    for candidate in candidates:
        score, chain = score_candidate(candidate, metric, start_structure, run_inputs)
        scores[candidate.label] = score
        chains[candidate.label] = chain

    best = min(candidates, key=lambda c: scores[c.label])
    if identity and best is not identity and scores[identity.label] - scores[best.label] <= veto_margin:
        best = identity

    return SelectionResult(winner=best, scores=scores, chains=chains)


def maybe_realign_pair(
    start_structure: Structure, end_structure: Structure, run_inputs,
) -> tuple[Structure, bool]:
    """Best-of-N atom-mapping selection between an arbitrary (start, end)
    pair -- not necessarily the top-level --start/--end -- using
    `run_inputs.atom_mapping_inputs`. Returns `(chosen_end_structure,
    changed)`.

    No-ops (returns `(end_structure, False)`) if slapmapper is
    unavailable, atom counts differ, SLAPMapper finds no mapping, or every
    candidate it finds is the identity mapping. Unlike `check_atom_mapping`,
    this never warns -- meant for `mepd.msmep`'s `--atom-mapping-recheck-splits`,
    which may call this many times per run, deep in the recursion."""
    from mepd.atom_mapping import HAS_SLAPMAPPER, suggest_atom_mapping_candidates

    if not HAS_SLAPMAPPER or len(start_structure.symbols) != len(end_structure.symbols):
        return end_structure, False

    atom_mapping_inputs = run_inputs.atom_mapping_inputs
    try:
        atom_maps = suggest_atom_mapping_candidates(
            start_structure, end_structure, max_candidates=atom_mapping_inputs.n_candidates
        )
    except Exception:
        return end_structure, False

    candidates = build_candidates(start_structure, end_structure, atom_maps)
    if [c.label for c in candidates] in ([], ["identity"]):
        return end_structure, False

    try:
        result = select_best_candidate(
            candidates, atom_mapping_inputs.metric, start_structure, run_inputs,
            veto_margin=atom_mapping_inputs.veto_margin,
        )
    except Exception:
        return end_structure, False

    return result.winner.end_structure, result.winner.label != "identity"


@dataclass
class MechanismChoice:
    key: str  # `mepd.atom_mapping.mechanism_key`
    winner: MappingCandidate  # this mechanism's best-scoring symmetry variant
    score: float
    n_variants: int


def select_per_mechanism(
    start_structure: Structure, end_structure: Structure, metric: str, run_inputs,
    *, max_variants_per_mechanism: int = 200, only_keys: Optional[set] = None,
) -> list[MechanismChoice]:
    """For one (reactant, product) pair: every mechanism SLAPMapper's
    minimal-cost mappings allow, each represented by its best symmetry
    variant under `metric` -- i.e. the geodesic score chooses how to label a
    mechanism's equivalent atoms for THIS pair's geometry, but never chooses
    between mechanisms. Sorted best-scoring first.

    The current ordering ("identity") is scored as one more variant of
    whichever mechanism it implies, or as a mechanism of its own if it
    isn't one of SLAPMapper's. Returns [] if there is nothing to map
    (slapmapper missing, atom counts or compositions differ). `only_keys`:
    score only these mechanisms (the others are left out)."""
    from mepd.atom_mapping import (
        HAS_SLAPMAPPER, mechanism_key, realign_end_to_start, suggest_mechanism_candidates,
    )

    if not HAS_SLAPMAPPER or len(start_structure.symbols) != len(end_structure.symbols):
        return []
    groups = suggest_mechanism_candidates(
        start_structure, end_structure, max_variants_per_mechanism=max_variants_per_mechanism,
    )
    if not groups:
        return []

    identity_order = tuple(range(len(end_structure.symbols)))
    by_key: dict[str, list[MappingCandidate]] = {}
    for key, atom_maps in groups.items():
        by_key[key] = [
            MappingCandidate(
                label=f"{key} #{n}",
                end_structure=end_structure if tuple(m.as_order()) == identity_order
                else realign_end_to_start(m, end_structure),
                atom_map=None if tuple(m.as_order()) == identity_order else m,
            )
            for n, m in enumerate(atom_maps)
        ]
    if (list(start_structure.symbols) == list(end_structure.symbols)
            and not any(c.atom_map is None for cands in by_key.values() for c in cands)):
        key = mechanism_key(start_structure, end_structure)
        by_key.setdefault(key, []).append(
            MappingCandidate(label=f"{key} identity", end_structure=end_structure, atom_map=None)
        )

    choices = []
    for key, cands in by_key.items():
        if only_keys is not None and key not in only_keys:
            continue
        pool = lowest_rmsd_variants(cands, start_structure, run_inputs) if metric == "rmsd-geodesic" else cands
        scored = [(score_candidate(c, metric, start_structure, run_inputs)[0], c) for c in pool]
        score, best = min(scored, key=lambda sc: sc[0])
        choices.append(MechanismChoice(key=key, winner=best, score=score, n_variants=len(cands)))
    choices.sort(key=lambda ch: ch.score)
    return choices
