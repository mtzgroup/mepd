"""Atom mapping when one endpoint is built from the other (--pair-from).

With both endpoints as given, a candidate mapping is judged by the path to
the given product (mepd.atom_mapping_selection). When the product is built
in the reactant's frame instead (or the reactant in the product's), that
product is not the path search's endpoint: each mapping gives its own built
partner. So here every candidate mapping is turned into the bonds it implies
on the source's atoms, a partner with those bonds is embedded in the
source's frame and minimized (all of them, as one batch), partners whose
bonds or stereochemistry change are dropped, and the rest are ranked by the
path from the source to each minimized partner -- the pair that is then
searched. Candidates with the same bond set give the same partner and are
built once. Snap is not used: the built partner is already in the source's
frame, and which symmetric atoms react is compared by the paths themselves.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from itertools import islice
from typing import Callable, Optional

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR
from qcdata import Structure


MAX_SWAPS = 20000   # symmetry swaps of the source scanned for new bond sets


@dataclass
class BuiltPair:
    start: Structure       # in the start's atom order
    end: Structure         # in the start's atom order
    label: str             # the winning candidate
    scores: dict           # {label: the quantity compared}
    quantity: str          # what `scores` hold
    notes: list            # lines explaining the choice
    n_bond_sets: int       # distinct bond sets built
    n_kept: int            # of them, minimized with bonds and stereochemistry intact


def _edges(structure: Structure) -> set:
    from mepd.nodes.node import StructureNode

    return {tuple(sorted((int(u), int(v)))) for u, v in StructureNode(structure=structure).graph.edges()}


def _in_end_order(atom_map, start: Structure) -> Structure:
    """`start` renumbered to the end's atom order under `atom_map`
    (start atom i -> end atom mapping[i])."""
    from mepd.atom_mapping import reorder_structure

    inverse = [0] * len(atom_map.mapping)
    for i, j in atom_map.mapping.items():
        inverse[j] = i
    return reorder_structure(start, inverse)


def _minimize(guesses: list[Structure], run_inputs) -> list[Optional[Structure]]:
    """Each guess minimized with the run's engine (one batch when the
    engine can); None where an optimization failed."""
    from mepd.nodes.node import StructureNode

    nodes = [StructureNode(structure=g) for g in guesses]
    keywords = {"coordsys": "cart", "maxit": 500, **(getattr(run_inputs, "geometry_optimizer_kwds", None) or {})}
    batch = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    try:
        if callable(batch):
            try:
                trajectories = batch(nodes, keywords=keywords)
            except TypeError:
                trajectories = batch(nodes)
        else:
            raise AttributeError
    except Exception:
        trajectories = []
        for n in nodes:
            try:
                trajectories.append(run_inputs.engine.compute_geometry_optimization(n, keywords=keywords))
            except Exception:
                trajectories.append(None)
    return [traj[-1].structure if traj else None for traj in trajectories]


def _build(start: Structure, end: Structure, atom_maps: list, pair_from: str, run_inputs, echo) -> tuple:
    """Every candidate's partner built and minimized: (jobs, kept). jobs:
    (label, mapping, source, target, bonds) per distinct bond set; kept:
    (label, mapping, source, built) for those whose bonds and
    stereochemistry survived minimization."""
    from mepd.atom_mapping import AtomMapping, _automorphisms, realign_end_to_start
    from mepd.discovery.network_expansion import embed_product
    from mepd.nodes.node import StructureNode
    from mepd.nodes.nodehelpers import _is_connectivity_identical

    n = len(start.symbols)
    maps = [("identity", AtomMapping(mapping={i: i for i in range(n)}, cost=0, n_alternatives=1))] \
        if list(start.symbols) == list(end.symbols) else []
    maps += [(f"mapping_{k}", m) for k, m in enumerate(m for m in atom_maps if not m.is_identity)]
    cap = max(1, int(getattr(run_inputs.atom_mapping_inputs, "n_candidates", 200) or 200))

    # Which of the source's symmetric atoms react changes the built
    # partner's geometry (which of a methyl's H moves): each mapping also
    # with the source's symmetric atoms swapped. Swaps of atoms that do not
    # react give the same bonds and are built once.
    autos = list(islice(_automorphisms(start if pair_from == "start" else end), MAX_SWAPS))

    def variants(label, m):
        yield label, m
        for k, sigma in enumerate(autos):
            if pair_from == "start":
                mapping = {i: m.mapping[int(sigma[i])] for i in range(n)}
            else:
                mapping = {i: int(sigma[m.mapping[i]]) for i in range(n)}
            yield f"{label} sym{k}", dataclasses.replace(m, mapping=mapping, relabeling=True)

    # One build per bond set. Built from the start, a mapping's bonds are
    # the end's, renumbered onto the start's atoms; from the end, the
    # start's bonds renumbered onto the end's atoms.
    seen, jobs = set(), []
    for label0, m0 in maps:
        for label, m in variants(label0, m0):
            if len(jobs) >= cap:
                break
            if pair_from == "start":
                source, target = start, realign_end_to_start(m, end)
            else:
                source, target = end, _in_end_order(m, start)
            want = frozenset(_edges(target))
            if want in seen:
                continue
            seen.add(want)
            jobs.append((label, m, source, target, want))
    built_name = "product" if pair_from == "start" else "reactant"
    echo(f"Atom mapping: building the {built_name} of each of {len(jobs)} distinct bond set(s) "
         f"({len(maps)} candidate mapping(s) and swaps of the {'reactant' if pair_from == 'start' else 'product'}'s "
         f"symmetric atoms{', capped' if len(jobs) >= cap else ''}) in its frame, minimizing them, then comparing "
         f"the paths to them...")

    guesses, embedded = [], []
    for job in jobs:
        _, _, source, target, want = job
        try:
            xyz = embed_product(list(source.symbols), np.asarray(source.geometry).reshape(-1, 3) / ANGSTROM_TO_BOHR,
                                set(want))
        except Exception:
            continue
        guesses.append(target.model_copy(update={"geometry": np.asarray(xyz) * ANGSTROM_TO_BOHR}))
        embedded.append(job)
    minimized = _minimize(guesses, run_inputs) if guesses else []

    kept = []
    for (label, m, source, target, want), s in zip(embedded, minimized):
        if s is None or frozenset(_edges(s)) != want:
            continue
        if not _is_connectivity_identical(StructureNode(structure=s), StructureNode(structure=target),
                                          verbose=False, collect_comparison=False):
            continue   # stereochemistry changed
        kept.append((label, m, source, s))
    echo(f"  {len(kept)} of {len(jobs)} kept their bonds and stereochemistry on minimization.")
    return jobs, kept


def _rank(kept: list, start: Structure, pair_from: str, run_inputs) -> tuple:
    """`kept` (from `_build`) ranked as paths from the source (shared by
    all) to each built partner: (best label, scores, quantity, notes)."""
    from mepd.atom_mapping_selection import MappingCandidate, _rank_picks, score_candidate

    picks = [MappingCandidate(label=label, end_structure=s, atom_map=None if label == "identity" else m)
             for label, m, _, s in kept]
    source = kept[0][2] if pair_from == "end" else start
    metric = run_inputs.atom_mapping_inputs.metric
    if metric in ("snap", "snap-gi-xtb"):
        _, best, _, scores, quantity, notes = _rank_picks(picks, picks, source, run_inputs,
                                                          use_path=metric == "snap-gi-xtb")
    else:
        scores = {c.label: score_candidate(c, metric, source, run_inputs)[0] for c in picks}
        best, quantity, notes = min(picks, key=lambda c: scores[c.label]), metric, []
    if pair_from == "end" and quantity:
        quantity += ", from the product"
    return best.label, scores, quantity, notes


def _pair(pair_from: str, start: Structure, end: Structure, m, built: Structure) -> tuple:
    """(start, end) of a built pair, in the start's atom order."""
    from mepd.atom_mapping import realign_end_to_start

    if pair_from == "start":
        return start, built
    # the built start and the end are both in the end's order
    return realign_end_to_start(m, built), realign_end_to_start(m, end)


def build_and_rank(start: Structure, end: Structure, atom_maps: list, pair_from: str, run_inputs,
                   echo: Callable[[str], None] = print) -> Optional[BuiltPair]:
    """`pair_from` "start": the end is built in the start's frame (the start
    is kept); "end": the start in the end's frame. `atom_maps`: the
    candidate mappings (start atom i -> end atom mapping[i]); the current
    numbering competes too when the elements are in the same order. Returns
    the chosen pair in the start's atom order, or None if no partner kept
    its bonds and stereochemistry."""
    jobs, kept = _build(start, end, atom_maps, pair_from, run_inputs, echo)
    if not kept:
        return None
    best, scores, quantity, notes = _rank(kept, start, pair_from, run_inputs)
    label, m, _, built = next(k for k in kept if k[0] == best)
    changes = {job[0]: len(_edges(job[2]) ^ job[4]) for job in jobs}
    fewest = min(changes.values())
    if changes[label] > fewest:
        echo(f"WARNING: no {'product' if pair_from == 'start' else 'reactant'} with the fewest bond changes "
             f"({fewest}) survived minimization with its bonds and stereochemistry; the chosen one has "
             f"{changes[label]}.")
    pair_start, pair_end = _pair(pair_from, start, end, m, built)
    return BuiltPair(start=pair_start, end=pair_end, label=label, scores=scores, quantity=quantity, notes=notes,
                     n_bond_sets=len(jobs), n_kept=len(kept))


@dataclass
class MechanismPair:
    key: str               # mechanism_key: which bonds break and form, by element and symmetry class
    score: float           # xtb peak along its interpolated path, kcal/mol (else GI length)
    start: Structure       # in the start's atom order
    end: Structure
    n_variants: int        # built variants of this mechanism compared


def best_per_mechanism(start: Structure, end: Structure, atom_maps: list, pair_from: str, run_inputs,
                       echo: Callable[[str], None] = print) -> list[MechanismPair]:
    """Like `build_and_rank`, but one pair per mechanism instead of one in
    all: the built partners grouped by which bonds break and form, and each
    group's variants (which symmetric atoms react) ranked on their own.
    For channels, where every mechanism gets its own path searches."""
    from mepd.atom_mapping import mechanism_key

    _, kept = _build(start, end, atom_maps, pair_from, run_inputs, echo)
    groups: dict = {}
    for k in kept:
        label, m, _, built = k
        a, b = _pair(pair_from, start, end, m, built)
        groups.setdefault(mechanism_key(a, b), []).append(k)
    out = []
    for key, members in groups.items():
        best, _, _, _ = _rank(members, start, pair_from, run_inputs) if len(members) > 1 else (members[0][0],) * 4
        label, m, _, built = next(k for k in members if k[0] == best)
        a, b = _pair(pair_from, start, end, m, built)
        out.append(MechanismPair(key=key, score=_path_score(a, b, run_inputs), start=a, end=b,
                                 n_variants=len(members)))
    return out


def _path_score(start: Structure, end: Structure, run_inputs) -> float:
    """One number to compare a mechanism's pairs across conformers: the
    xtb peak along the interpolated path (kcal/mol above its start), else
    the path's length; inf if neither can be had."""
    import math

    from mepd.atom_mapping_selection import KCAL_PER_HARTREE, MappingCandidate, _interpolate, _xtb_engine

    try:
        chain, smoother = _interpolate(MappingCandidate(label="pair", end_structure=end, atom_map=None), start,
                                       run_inputs)
    except Exception:
        return math.inf
    engine = _xtb_engine()
    if engine is not None:
        try:
            e = [float(x) for x in engine.compute_energies(chain)]
            if e and all(math.isfinite(x) for x in e):
                return (max(e) - e[0]) * KCAL_PER_HARTREE
        except Exception:
            pass
    return float(smoother.length)
