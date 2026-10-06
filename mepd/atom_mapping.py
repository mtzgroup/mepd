"""Atom-to-atom mapping between two molecular structures.

Wraps SLAPMapper -- Koda, "General and scalable atom-to-atom mapping via
Weisfeiler-Lehman-like approximate graph matching"
(ChemRxiv, 2025, https://chemrxiv.org/doi/10.26434/chemrxiv-2025-hthwn) -- a
WL-style iterative label-refinement scheme coupled with sequential
linear-assignment problems, to:

  1. sanity-check that a pair of endpoint structures handed to us as
     "already atom-matched" (same index means the same atom on both sides
     of the reaction) actually is, warning when SLAPMapper's own suggested
     correspondence disagrees (`check_atom_mapping`); and
  2. build a consistently-indexed structure pair directly from two
     reaction-endpoint SMILES strings, where no such correspondence exists
     until a mapping is computed (`map_smiles_pair`).

Requires the optional `slapmapper` dependency (`pip install mepd[aam]`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from qcdata.models.structure import Structure

from mepd.helper_functions import symbol_to_atomic_number
from mepd.molecule import Molecule
from mepd.qcdata_structure_helpers import molecule_to_structure, structure_to_molecule

try:
    from slapmapper.core import LabeledGraph, SlapMapper

    HAS_SLAPMAPPER = True
except ImportError:  # pragma: no cover - exercised only without the optional dep
    HAS_SLAPMAPPER = False

# Mirrors mepd.qcdata_structure_helpers.molecule_to_structure's bond-order map.
_BOND_ORDER_WEIGHT = {"single": 1.0, "double": 2.0, "triple": 3.0, "aromatic": 1.5}


@dataclass
class AtomMapping:
    """The correspondence SLAPMapper suggests between two structures'
    atoms: `mapping[i]` is the index into the `end`-side structure that
    atom `i` of the `start`-side structure maps onto.
    """

    mapping: dict[int, int]
    cost: float
    n_alternatives: int
    # A relabeling of symmetric atoms added to fill the candidate budget,
    # not a mapping SLAPMapper itself returned.
    relabeling: bool = False
    explored: bool = False  # from `explore_mechanisms`, beyond SLAPMapper's minimal-edit mappings

    @property
    def is_identity(self) -> bool:
        return all(k == v for k, v in self.mapping.items())

    def as_order(self) -> list[int]:
        """`end`-side reordering such that `end[order[i]]` corresponds to
        `start`'s atom `i`."""
        return [self.mapping[i] for i in range(len(self.mapping))]


def _require_slapmapper() -> None:
    if not HAS_SLAPMAPPER:
        raise ImportError(
            "Atom-to-atom mapping requires the optional 'slapmapper' package. "
            "Install it with `pip install mepd[aam]`."
        )


def _atomic_numbers(mol: Molecule) -> list[int]:
    return [symbol_to_atomic_number(mol.nodes[n]["element"]) for n in mol.nodes]


def molecule_to_labeled_graph(mol: Molecule) -> "LabeledGraph":
    """Build a SLAPMapper `LabeledGraph` from a mepd `Molecule` graph.

    Node labels are atomic numbers; edge weights are bond orders. Assumes
    `mol`'s node indices are the dense range 0..N-1, in the same order as
    the `Structure` it was derived from -- true for
    `mepd.qcdata_structure_helpers.structure_to_molecule` output.
    """
    _require_slapmapper()

    nodes = sorted(mol.nodes)
    if nodes != list(range(len(nodes))):
        raise ValueError(
            "Molecule nodes must be a dense 0..N-1 range for atom mapping."
        )

    labels = _atomic_numbers(mol)
    graph: dict[int, dict[int, float]] = {i: {} for i in nodes}
    for u, v, data in mol.edges(data=True):
        weight = _BOND_ORDER_WEIGHT[data["bond_order"]]
        graph[u][v] = weight
        graph[v][u] = weight

    return LabeledGraph(graph, labels)


def suggest_atom_mapping(
    struct_start: Structure, struct_end: Structure, *, binary: bool = True
) -> Optional[AtomMapping]:
    """Suggest an atom mapping from `struct_start` onto `struct_end` using
    SLAPMapper's WL-refinement + sequential-LAP algorithm.

    `binary=True` (the default, matching SLAPMapper's own chemical-AAM
    default) ignores bond order and matches on adjacency alone -- important
    since bonds are expected to change order (or break/form) between a
    reaction's endpoints.

    Returns None if the two structures don't share the same multiset of
    atomic numbers (SLAPMapper doesn't support such "unbalanced" pairs) or
    if SLAPMapper finds no mapping.
    """
    _require_slapmapper()

    mol_start = structure_to_molecule(struct_start)
    mol_end = structure_to_molecule(struct_end)

    if sorted(_atomic_numbers(mol_start)) != sorted(_atomic_numbers(mol_end)):
        return None

    # With symmetry broken (suggest_atom_mapping_candidates), each label
    # class is one atom. An unbranched pass leaves classes of equivalent
    # atoms (e.g. acetone's six methyl H) paired by index order, which need
    # not be the cost-minimal pairing: acetone -> its enol came out as the
    # current numbering, a double H shift (4 bond changes, not 2).
    found = suggest_atom_mapping_candidates(struct_start, struct_end, binary=binary, max_candidates=1)
    if not found:
        return None
    best = found[0]
    n = len(best.mapping)
    if not best.is_identity and list(struct_start.symbols) == list(struct_end.symbols):
        identity = {i: i for i in range(n)}
        if _bond_changes_under(struct_start, struct_end, identity) <= _bond_changes_under(
                struct_start, struct_end, best.mapping):
            # The current numbering is as good: the bonds already correspond.
            return AtomMapping(mapping=identity, cost=best.cost, n_alternatives=best.n_alternatives)
    return best


def _bond_changes_under(struct_start: Structure, struct_end: Structure, mapping: dict[int, int]) -> int:
    """Bonds broken plus formed when start atom i becomes end atom mapping[i]."""
    from mepd.nodes.node import StructureNode

    def edges(s):
        return {tuple(sorted((int(u), int(v)))) for u, v in StructureNode(structure=s).graph.edges()}

    inverse = {j: i for i, j in mapping.items()}
    end_in_start = {tuple(sorted((inverse[u], inverse[v]))) for u, v in edges(struct_end)}
    return len(edges(struct_start) ^ end_in_start)


def _symmetry_orbits(structure: Structure) -> list[list[int]]:
    """Groups of atom indices in `structure` that are topologically
    interchangeable, via `_symmetry_ranks` (e.g. a methyl group's three Hs,
    or a symmetric =CH2's two Hs). Only groups of size > 1 are actual
    degrees of freedom. `[]` if RDKit can't parse/assign bonds (each atom
    then has its own unique rank, so no group exceeds size 1) -- never
    raises."""
    groups: dict[int, list[int]] = {}
    for idx, rank in enumerate(_symmetry_ranks(structure)):
        groups.setdefault(rank, []).append(idx)
    return [idxs for idxs in groups.values() if len(idxs) > 1]


def expand_mapping_by_symmetry(
    atom_map: "AtomMapping",
    start_structure: Structure,
    end_structure: Structure,
    *,
    max_expansions: int = 10,
) -> list["AtomMapping"]:
    """Generate additional candidate mappings from `atom_map` by permuting
    its correspondence within each side's own symmetry orbit (one orbit
    permuted at a time, holding the rest of the mapping fixed -- not the
    full cross-product across multiple orbits at once, to keep this
    tractable for highly symmetric molecules).

    These are cost-preserving relabelings (same molecule, provably the
    same LAP cost) -- not mappings from a higher-cost tier (SLAPMapper
    itself provides no supported way to retrieve those). SLAPMapper's own
    `_remove_isomorphic_results` already collapses these away as
    graph-redundant, but two symmetry-equivalent mappings are not
    necessarily geometrically equivalent for a *specific* input
    conformer's geodesic interpolation -- e.g. which specific methyl H
    ends up matched can change how well the resulting path aligns with
    the actual input reactant geometry, even though the mapping "cost" is
    identical -- so `mepd.atom_mapping_selection`'s geodesic-metric
    selection is given the chance to actually try them, rather than
    silently keeping whichever one SLAPMapper happened to return first.

    A start-side orbit is only permuted if its image under `atom_map`
    (`{atom_map.mapping[i] for i in orbit}`) is itself a subset of one of
    `end_structure`'s own orbits -- i.e. only when doing so is verified to
    still be a valid, cost-preserving relabeling, never guessed. Molecule
    pairs with no real symmetry return `[]` -- there's only one
    minimal-cost correspondence, period.
    """
    start_orbits = _symmetry_orbits(start_structure)
    end_orbits = _symmetry_orbits(end_structure)

    def _end_orbit_containing(idxs: set) -> Optional[list[int]]:
        for orbit in end_orbits:
            if idxs <= set(orbit):
                return orbit
        return None

    import itertools

    expansions: list[AtomMapping] = []
    for orbit in start_orbits:
        if len(expansions) >= max_expansions:
            break
        image = {atom_map.mapping[i] for i in orbit}
        end_orbit = _end_orbit_containing(image)
        if end_orbit is None:
            continue

        for perm in itertools.permutations(sorted(image)):
            if len(expansions) >= max_expansions:
                break
            new_mapping = dict(atom_map.mapping)
            for start_idx, end_idx in zip(sorted(orbit), perm):
                new_mapping[start_idx] = end_idx
            if new_mapping == atom_map.mapping:
                continue
            expansions.append(
                AtomMapping(
                    mapping=new_mapping,
                    cost=atom_map.cost,
                    n_alternatives=atom_map.n_alternatives,
                )
            )
    return expansions


def suggest_atom_mapping_candidates(
    struct_start: Structure, struct_end: Structure, *, binary: bool = True, max_candidates: int = 5
) -> list["AtomMapping"]:
    """Like `suggest_atom_mapping`, but keeps up to `max_candidates` of
    SLAPMapper's equal-minimal-cost candidate mappings instead of only the
    first (`mapper.results` is already filtered to ties at the minimum
    WL/LAP cost -- these candidates are NOT ranked by quality among
    themselves, SLAPMapper gives no further signal to order them by).

    Passes every atom index as a `break_sym_targets` candidate so
    SLAPMapper both (a) explores genuinely distinct tied-cost orderings
    its un-branched single pass would otherwise miss, and (b) dedupes
    results that are mere relabelings of each other under the same
    symmetry (`_remove_isomorphic_results`, only run when
    `break_sym_targets` is given) -- so this doesn't waste downstream
    geodesic-interpolation calls on redundant symmetry-equivalent
    candidates. `_break_sym` only actually branches on label groups that
    turn out to have size > 1, so this doesn't force unnecessary
    branching on atoms with no real symmetry.

    If SLAPMapper's own (deduped) results don't fill `max_candidates`, the
    remaining budget is filled with `expand_mapping_by_symmetry` expansions
    of the best candidate -- cost-preserving relabelings within a
    symmetry orbit that SLAPMapper's dedup already treats as redundant by
    graph/cost, but which can still correspond differently well to the
    actual input conformer's geometry (see that function's docstring).

    Returns an empty list under the same conditions `suggest_atom_mapping`
    returns `None` (unbalanced atomic-number multisets, or no mapping
    found at all).
    """
    _require_slapmapper()

    mol_start = structure_to_molecule(struct_start)
    mol_end = structure_to_molecule(struct_end)

    if sorted(_atomic_numbers(mol_start)) != sorted(_atomic_numbers(mol_end)):
        return []

    lg_start = molecule_to_labeled_graph(mol_start)
    lg_end = molecule_to_labeled_graph(mol_end)

    mapper = SlapMapper(binary=binary)
    try:
        mapper.get_maps([lg_start, lg_end], break_sym_targets=list(range(len(lg_start.labels))))
    except ValueError:
        # SLAPMapper's symmetry branching can prune every result and then
        # take min() of none (seen on a 50-atom pair). The un-branched pass
        # still finds the best mapping; the symmetry expansion below fills
        # the remaining candidates.
        mapper = SlapMapper(binary=binary)
        mapper.get_maps([lg_start, lg_end])
    if not mapper.results:
        return []

    n_alternatives = len(mapper.results)
    max_candidates = max(int(max_candidates), 0)
    candidates: list[AtomMapping] = []
    seen_orders: set[tuple[int, ...]] = set()
    for result in mapper.results:
        if len(candidates) >= max_candidates:
            break

        label2idxs_start = result["lgp"][0].label2idxs
        label2idxs_end = result["lgp"][1].label2idxs

        mapping: dict[int, int] = {}
        for label, idxs_start in label2idxs_start.items():
            idxs_end = label2idxs_end[label]
            for a, b in zip(sorted(idxs_start), sorted(idxs_end)):
                mapping[a] = b

        order = tuple(mapping[i] for i in range(len(mapping)))
        if order in seen_orders:
            continue
        seen_orders.add(order)

        candidates.append(
            AtomMapping(mapping=mapping, cost=result["val"], n_alternatives=n_alternatives)
        )

    # Fill any remaining budget with symmetry-orbit expansions of the best
    # candidate found so far -- cost-preserving relabelings SLAPMapper's own
    # dedup already discarded as graph-redundant, but which can still differ
    # geometrically for this specific input conformer (see
    # `expand_mapping_by_symmetry`'s docstring). No-op (and no extra
    # SLAPMapper/QM cost) whenever the budget's already full or there's no
    # real symmetry to expand.
    if candidates and len(candidates) < max_candidates:
        for expansion in expand_mapping_by_symmetry(
            candidates[0], struct_start, struct_end,
            max_expansions=max_candidates - len(candidates),
        ):
            order = tuple(expansion.mapping[i] for i in range(len(expansion.mapping)))
            if order in seen_orders:
                continue
            seen_orders.add(order)
            expansion.relabeling = True
            candidates.append(expansion)
            if len(candidates) >= max_candidates:
                break

    return candidates


def _symmetry_colors(structure: Structure) -> list:
    """Per atom, its element and RDKit symmetry class (`_symmetry_ranks`,
    which tells diastereotopic groups apart: a bare graph automorphism could
    swap them, inverting a stereocentre); the element alone where RDKit
    can't perceive the bonds."""
    ranks = _symmetry_ranks(structure)
    if len(set(ranks)) == len(ranks) and ranks == list(range(len(ranks))):
        return list(structure.symbols)
    return [f"{x}{r}" for x, r in zip(structure.symbols, ranks)]


def _automorphisms(structure: Structure):
    """The bond graph's automorphisms that keep every bond and every atom's
    `_symmetry_colors`, lazily, as arrays: sigma[i] is where atom i goes.
    The graph is the one SLAPMapper sees (`structure_to_molecule`).
    Identity alone if there is no symmetry or nauty can't be used."""
    n = len(structure.symbols)
    try:
        import pynauty
        from sympy.combinatorics import Permutation, PermutationGroup

        adj = {i: [] for i in range(n)}
        for u, v in structure_to_molecule(structure).edges():
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))
        by_color: dict = {}
        for i, c in enumerate(_symmetry_colors(structure)):
            by_color.setdefault(c, set()).add(i)
        gens = pynauty.autgrp(pynauty.Graph(n, adjacency_dict=adj, vertex_coloring=list(by_color.values())))[0]
    except Exception:
        gens = []
    if not gens:
        return iter([list(range(n))])
    return PermutationGroup([Permutation(g) for g in gens]).generate(af=True)


# snap's search is greedy; which of its settings lands lowest varies from
# molecule to molecule, and each run takes milliseconds, so all are kept.
SNAP_SETTINGS = ({}, {"align_per_component": True}, {"factor_depth": 2},
                 {"factor_depth": 2, "align_per_component": True})


def _snap_variants(atom_map: "AtomMapping", start_structure: Structure, end_structure: Structure) -> list:
    """Relabelings of `atom_map` by start-graph automorphisms that bring the
    end close to the start (endpoint RMSD), found the way snap-RMSD does
    (qcinf): the molecule is factored into a core and its symmetric groups,
    and each group's own permutations are scored in turn, so even an
    astronomically large symmetry group costs milliseconds. The end, put in
    the start's atom order, is given the start's bond graph so the two are
    isomorphic by construction. One per `SNAP_SETTINGS`; [] if snap can't be
    used."""
    try:
        from qcinf.algorithms.snap import snap_rmsd_align_assign

        conn = [(int(u), int(v), 1.0) for u, v in structure_to_molecule(start_structure).edges()]
        colors = _symmetry_colors(start_structure)
        aligned = realign_end_to_start(atom_map, end_structure)
    except Exception:
        return []
    out = []
    for settings in SNAP_SETTINGS:
        try:
            _, _, p = snap_rmsd_align_assign(start_structure, aligned, a_connectivity=conn, b_connectivity=conn,
                                             a_coloring=colors, b_coloring=colors, **settings)
        except Exception:
            continue
        out.append(AtomMapping(mapping={i: atom_map.mapping[int(p[i])] for i in range(len(p))},
                               cost=atom_map.cost, n_alternatives=atom_map.n_alternatives))
    return out


def expand_mapping_fully(
    atom_map: "AtomMapping",
    start_structure: Structure,
    end_structure: Structure,
    *,
    max_variants: int = 200,
) -> list["AtomMapping"]:
    """`atom_map` and its relabelings by the start graph's automorphisms,
    up to `max_variants` (`atom_map` first, then those with the lowest
    endpoint RMSD over the whole group: `_snap_variants`, then the group's
    elements in turn).

    These all describe the same mechanism (a start automorphism maps the
    broken and formed bonds onto symmetry-equivalent ones), but not the
    same path: which of a CH2's two hydrogens goes where decides whether the
    group has to rotate on the way, so each variant needs its own score
    against the actual pair of conformers. Only automorphisms: permuting a
    symmetry orbit freely would also swap hydrogens between the carbons of
    different methyls, a different mechanism (and 9! relabelings of a
    SiMe3's hydrogens where 1296 keep its bonds)."""
    variants = [atom_map]
    seen = {tuple(atom_map.as_order())}

    def add(mapping):
        order = tuple(mapping[i] for i in range(len(mapping)))
        if order not in seen and len(variants) < max_variants:
            seen.add(order)
            variants.append(AtomMapping(mapping=mapping, cost=atom_map.cost, n_alternatives=atom_map.n_alternatives))

    for best in _snap_variants(atom_map, start_structure, end_structure):
        add(best.mapping)
    for sigma in _automorphisms(start_structure):
        if len(variants) >= max_variants:
            break
        add({i: atom_map.mapping[int(sigma[i])] for i in range(len(sigma))})
    return variants


def _symmetry_ranks(structure: Structure) -> list[int]:
    """RDKit canonical rank per atom with ties kept (graph-equivalent atoms
    share a rank); plain atom indices if RDKit can't perceive the bonding."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    try:
        mol = Chem.MolFromXYZBlock(structure.to_xyz())
        rdDetermineBonds.DetermineBonds(mol, charge=structure.charge)
        return list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:
        return list(range(len(structure.symbols)))


def mechanism_key(start_structure: Structure, aligned_end_structure: Structure) -> str:
    """Name the mechanism an atom correspondence implies: the bonds broken
    and formed going from `start_structure` to `aligned_end_structure`
    (already reindexed so atom i is the same atom on both sides), each atom
    written as element + its symmetry class in the start structure.

    Symmetry classes rather than indices, so relabelings of one mechanism
    (a CH2's hydrogens swapped) and the same mechanism reached from a
    different pair of conformers get the same key. Bonds are perceived the
    way SLAPMapper sees them (`structure_to_molecule`, adjacency only)."""
    symbols = list(start_structure.symbols)
    ranks = _symmetry_ranks(start_structure)

    def _edges(structure):
        mol = structure_to_molecule(structure)
        return {tuple(sorted((u, v))) for u, v in mol.edges()}

    before, after = _edges(start_structure), _edges(aligned_end_structure)

    def _fmt(edges):
        names = sorted(
            "-".join(sorted(f"{symbols[a]}{ranks[a]}" for a in edge)) for edge in edges
        )
        return ",".join(names) or "none"

    return f"break {_fmt(before - after)} | form {_fmt(after - before)}"


def suggest_mechanism_candidates(
    struct_start: Structure, struct_end: Structure, *, binary: bool = True,
    max_variants_per_mechanism: int = 200, explore: int = 0,
) -> dict[str, list["AtomMapping"]]:
    """SLAPMapper's equal-minimal-cost mappings, grouped by the mechanism
    they imply (`mechanism_key`), each fully expanded into its symmetry
    variants (`expand_mapping_fully`).

    SLAPMapper's own symmetry dedup (`_remove_isomorphic_results`) is kept
    for what it is good at -- telling genuinely different correspondences
    apart -- and the relabelings it collapses are regenerated here for
    EVERY mechanism, so each can be scored against the specific pair's
    geometry rather than only whichever result SLAPMapper listed first.

    `explore`: also add up to this many mechanisms found beyond SLAPMapper's
    minimal-edit ones (`explore_mechanisms`: relays and exchanges through
    other molecules), each expanded into its symmetry variants the same way.
    Their keys start with "explored:", and name any molecule that takes
    part but is regenerated ("[catalytic: H2O]").

    Returns {} when `suggest_atom_mapping_candidates` would return []."""
    _require_slapmapper()

    mol_start = structure_to_molecule(struct_start)
    mol_end = structure_to_molecule(struct_end)
    if sorted(_atomic_numbers(mol_start)) != sorted(_atomic_numbers(mol_end)):
        return {}

    lg_start = molecule_to_labeled_graph(mol_start)
    lg_end = molecule_to_labeled_graph(mol_end)
    mapper = SlapMapper(binary=binary)
    mapper.get_maps([lg_start, lg_end], break_sym_targets=list(range(len(lg_start.labels))))

    # Each SLAPMapper result is a mechanism of its own, never merged with
    # another: only its own symmetry variants (expand_mapping_fully) join it.
    # A result that is one of an earlier result's variants is that same
    # correspondence, and is skipped.
    routes: list[tuple[AtomMapping, list[AtomMapping]]] = []
    seen: set[tuple[int, ...]] = set()
    for result in mapper.results:
        label2idxs_start = result["lgp"][0].label2idxs
        label2idxs_end = result["lgp"][1].label2idxs
        mapping = {
            a: b
            for label, idxs_start in label2idxs_start.items()
            for a, b in zip(sorted(idxs_start), sorted(label2idxs_end[label]))
        }
        base = AtomMapping(mapping=mapping, cost=result["val"], n_alternatives=len(mapper.results))
        if tuple(base.as_order()) in seen:
            continue
        variants = []
        for variant in expand_mapping_fully(
            base, struct_start, struct_end, max_variants=max_variants_per_mechanism,
        ):
            order = tuple(variant.as_order())
            if order not in seen and len(variants) < max_variants_per_mechanism:
                seen.add(order)
                variants.append(variant)
        routes.append((base, variants))
    keys = _route_keys(struct_start, struct_end, [b for b, _ in routes])
    out = dict(zip(keys, (v for _, v in routes)))
    if explore > 0 and routes:
        for base in explore_mechanisms(struct_start, struct_end, [b for b, _ in routes], max_new=explore):
            aligned = realign_end_to_start(base, struct_end)
            key = f"explored: {mechanism_key(struct_start, aligned)}"
            cats = catalytic_participants(struct_start, aligned)
            if cats:
                key += f" [catalytic: {', '.join(cats)}]"
            if key in out:  # same mechanism by symmetry class, through other atoms: its own route
                key += f" [route {sum(1 for k in out if k.startswith(key)) + 1}]"
            variants = [v for v in expand_mapping_fully(base, struct_start, struct_end,
                                                         max_variants=max_variants_per_mechanism)
                        if tuple(v.as_order()) not in seen]
            seen.update(tuple(v.as_order()) for v in variants)
            if variants:
                out[key] = variants
    return out


class _EditCount:
    """The exact number of bonds broken plus formed under a mapping, on the
    bond graphs SLAPMapper sees (`structure_to_molecule`), with O(degree)
    updates for a swap of two atoms' images."""

    def __init__(self, struct_start: Structure, struct_end: Structure):
        self.n = len(struct_start.symbols)
        self.sym = [str(s) for s in struct_start.symbols]
        g0, g1 = structure_to_molecule(struct_start), structure_to_molecule(struct_end)
        self.adj0 = {i: {int(k) for k in g0.neighbors(i)} for i in range(self.n)}
        self.adj1 = {j: {int(k) for k in g1.neighbors(j)} for j in range(self.n)}
        # terminal neighbours by element: what a group swap carries along
        self.terminal = {i: {} for i in range(self.n)}
        for i in range(self.n):
            for k in sorted(self.adj0[i]):
                if len(self.adj0[k]) == 1:
                    self.terminal[i].setdefault(self.sym[k], []).append(k)

    def cost(self, order) -> int:
        c = 0
        for i in range(self.n):
            for k in self.adj0[i]:
                c += k > i and order[k] not in self.adj1[order[i]]
        inv = {int(v): a for a, v in enumerate(order)}
        for j in range(self.n):
            for b in self.adj1[j]:
                c += b > j and inv[b] not in self.adj0[inv[j]]
        return c

    def _local(self, order, inv, atoms) -> int:
        c, seen = 0, set()
        for i in atoms:
            for k in self.adj0[i] | {inv[b] for b in self.adj1[order[i]]}:
                pair = (min(i, k), max(i, k))
                if pair not in seen:
                    seen.add(pair)
                    c += (k in self.adj0[i]) != (order[k] in self.adj1[order[i]])
        return c

    def swap_delta(self, order, inv, i, j) -> int:
        before = self._local(order, inv, (i, j))
        order[i], order[j] = order[j], order[i]
        inv[order[i]], inv[order[j]] = i, j
        after = self._local(order, inv, (i, j))
        order[i], order[j] = order[j], order[i]
        inv[order[i]], inv[order[j]] = i, j
        return after - before

    def route(self, order) -> tuple:
        """The bonds broken and formed, atoms named by index except that a
        terminal atom is named by the atom it starts on: routes through
        different molecules stay apart (a wire through water 1 is not one
        through water 4, however alike the waters), while which of a
        group's equivalent atoms moves does not split a route."""
        def name(i):
            if len(self.adj0[i]) == 1:
                (p,) = tuple(self.adj0[i])
                return f"{self.sym[i]}@{p}"
            return str(i)
        inv = {int(v): a for a, v in enumerate(order)}
        broken = sorted(tuple(sorted((name(i), name(k)))) for i in range(self.n) for k in self.adj0[i]
                        if k > i and order[k] not in self.adj1[order[i]])
        formed = sorted(tuple(sorted((name(inv[j]), name(inv[b])))) for j in range(self.n) for b in self.adj1[j]
                        if b > j and inv[b] not in self.adj0[inv[j]])
        return tuple(broken), tuple(formed)

    def group_swaps(self, order):
        """Two same-element atoms swapped together with their same-element
        terminal neighbours (a CH3's three H, not its halide): one move for
        a group handed from one atom to another, which atom-by-atom swaps
        would only reach through high-cost intermediates."""
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.sym[i] != self.sym[j] or len(self.adj0[i]) == 1 or len(self.adj0[j]) == 1:
                    continue
                carried = [(a, b) for el, ti in self.terminal[i].items()
                           if len(self.terminal[j].get(el, [])) == len(ti)
                           for a, b in zip(ti, self.terminal[j][el])]
                if not carried:
                    continue
                t = list(order)
                t[i], t[j] = order[j], order[i]
                for a, b in carried:
                    t[a], t[b] = order[b], order[a]
                yield tuple(t)


def explore_mechanisms(
    struct_start: Structure, struct_end: Structure, bases: list["AtomMapping"], *,
    max_new: int = 20, rounds: int = 5, step: int = 2, per_round: int = 400,
) -> list["AtomMapping"]:
    """Mechanisms beyond SLAPMapper's minimal-edit ones, for an unbiased look
    at routes that involve more atoms -- a proton or methyl relayed through
    a catalyst or solvent, a group exchanged with another molecule.

    Starting from `bases` (SLAPMapper's mappings), repeatedly swap the images
    of two same-element atoms, or of two groups (an atom with its terminal
    atoms). A swap is kept if it raises the exact bond-edit count by at most
    `step`: one relay hop costs +2 (a bond broken and one formed), while
    shuffling bystanders costs more, so the expansion follows shuttles and
    exchanges at the reaction without any rule about which atoms may move.
    `rounds` swaps deep, the `per_round` cheapest new mappings carried on per
    round. Returns one mapping per new route (`_EditCount.route`: bonds
    broken and formed, by atom, not among the bases'), lowest edit count
    first, at most `max_new`. Which of them
    are actually low in energy is for the caller's scoring to decide."""
    if not bases or max_new <= 0:
        return []
    ec = _EditCount(struct_start, struct_end)
    pairs = [(i, j) for i in range(ec.n) for j in range(i + 1, ec.n) if ec.sym[i] == ec.sym[j]]
    seen = {}
    frontier = []
    for b in bases:
        o = tuple(int(v) for v in b.as_order())
        if o not in seen:
            seen[o] = ec.cost(o)
            frontier.append(o)
    for _ in range(rounds):
        new = {}
        for st in frontier:
            order = list(st)
            inv = {v: a for a, v in enumerate(order)}
            c = seen[st]
            for i, j in pairs:
                d = ec.swap_delta(order, inv, i, j)
                if d > step:
                    continue
                t = list(order)
                t[i], t[j] = t[j], t[i]
                t = tuple(t)
                if t not in seen and t not in new:
                    new[t] = c + d
            for t in ec.group_swaps(order):
                if t not in seen and t not in new:
                    ct = ec.cost(t)
                    if ct - c <= step:
                        new[t] = ct
        kept = sorted(new.items(), key=lambda kv: kv[1])[:per_round]
        seen.update(kept)
        frontier = [t for t, _ in kept]
        if not frontier:
            break

    known = {ec.route(tuple(int(v) for v in b.as_order())) for b in bases}
    best: dict[tuple, tuple[int, tuple]] = {}
    for order, c in seen.items():
        key = ec.route(order)
        if key not in known and (key not in best or c < best[key][0]):
            best[key] = (c, order)
    ranked = sorted(best.values(), key=lambda t: (t[0], t[1]))[:max_new]
    return [AtomMapping(mapping=dict(enumerate(o)), cost=float(c), n_alternatives=len(best), explored=True)
            for c, o in ranked]


def catalytic_participants(struct_start: Structure, aligned_end_structure: Structure) -> list[str]:
    """Molecules of the start that take part in the reaction yet are
    regenerated: some bond touching them breaks or forms, and the end has a
    molecule with the same bond graph (elements included) holding at least
    one of their atoms -- possibly rebuilt from other atoms, as when a water
    passes a proton on and takes another, or a CH3I hands over its methyl
    and gains one. Formulas, e.g. ['H2O', 'C5H9NO2']: a catalyst, or solvent
    acting as one."""
    import networkx as nx

    g0 = structure_to_molecule(struct_start)
    g1 = structure_to_molecule(aligned_end_structure)
    sym = [str(s) for s in struct_start.symbols]
    nx.set_node_attributes(g0, {i: sym[i] for i in g0.nodes}, "el")
    nx.set_node_attributes(g1, {i: sym[i] for i in g1.nodes}, "el")
    ends = [set(c) for c in nx.connected_components(g1)]
    out = []
    for comp in nx.connected_components(g0):
        comp = set(comp)
        touched = any(g0.has_edge(i, k) != g1.has_edge(i, k)
                      for i in comp for k in set(g0.neighbors(i)) | set(g1.neighbors(i)))
        if not touched:
            continue
        h0 = g0.subgraph(comp)
        if any(e & comp and len(e) == len(comp)
               and nx.is_isomorphic(h0, g1.subgraph(e), node_match=lambda a, b: a["el"] == b["el"])
               for e in ends):
            counts = {}
            for i in comp:
                counts[sym[i]] = counts.get(sym[i], 0) + 1
            order = sorted(counts, key=lambda e: (e != "C", e != "H", e))
            out.append("".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order))
    return sorted(out)


def _route_keys(struct_start: Structure, struct_end: Structure, bases: list["AtomMapping"]) -> list[str]:
    """A distinct mechanism key per SLAPMapper result. `mechanism_key` (the
    bonds broken and formed, atoms named by element and symmetry class) where
    it is unique; results that share one are told apart by the shape of
    their reaction centre (which changed bonds share which atoms), then, if
    even that is the same, by an ordinal in a geometry-independent order --
    so the same reaction gets the same keys for every conformer pair."""
    aligned = [realign_end_to_start(b, struct_end) for b in bases]
    coarse = [mechanism_key(struct_start, e) for e in aligned]
    keys = list(coarse)
    for key in set(coarse):
        idx = [k for k, c in enumerate(coarse) if c == key]
        if len(idx) < 2:
            continue
        sig = {k: _centre_signature(struct_start, aligned[k]) for k in idx}
        for k in idx:
            same = sorted((m for m in idx if sig[m] == sig[k]), key=lambda m: tuple(bases[m].as_order()))
            keys[k] = f"{key} [centre {sig[k][:8]}" + (f" #{same.index(k) + 1}]" if len(same) > 1 else "]")
    return keys


def _centre_signature(start_structure: Structure, aligned_end_structure: Structure) -> str:
    """Canonical hash of the reaction centre: the atoms whose bonds change
    (element + symmetry class), joined by their broken and formed bonds."""
    import networkx as nx

    symbols = list(start_structure.symbols)
    ranks = _symmetry_ranks(start_structure)
    before = {tuple(sorted(e)) for e in structure_to_molecule(start_structure).edges()}
    after = {tuple(sorted(e)) for e in structure_to_molecule(aligned_end_structure).edges()}
    g = nx.Graph()
    for edges, kind in ((before - after, "break"), (after - before, "form")):
        for u, v in edges:
            for a in (u, v):
                g.add_node(a, label=f"{symbols[a]}{ranks[a]}")
            g.add_edge(u, v, kind=kind)
    return nx.weisfeiler_lehman_graph_hash(g, node_attr="label", edge_attr="kind")


def check_atom_mapping(
    struct_start: Structure, struct_end: Structure, *, binary: bool = True
) -> Optional[AtomMapping]:
    """Like `suggest_atom_mapping`, but also raises a `UserWarning` when the
    suggested mapping disagrees with the identity mapping implied by
    `struct_start`/`struct_end` sharing the same atom indexing.
    """
    atom_map = suggest_atom_mapping(struct_start, struct_end, binary=binary)
    if atom_map is not None and not atom_map.is_identity:
        warnings.warn(
            "SLAPMapper's suggested atom-to-atom mapping disagrees with the "
            "input structures' shared atom ordering "
            f"(WL/LAP cost={atom_map.cost}, {atom_map.n_alternatives} equally-good "
            f"mapping(s) found). Suggested mapping (start index -> end index): "
            f"{atom_map.mapping}",
            UserWarning,
            stacklevel=2,
        )
    return atom_map


def reorder_structure(structure: Structure, order: list[int]) -> Structure:
    """Return a copy of `structure` with atoms permuted to `order`, i.e.
    `reordered.symbols[i] == structure.symbols[order[i]]`.
    """
    symbols = np.asarray(structure.symbols)[order]
    geometry = np.asarray(structure.geometry)[order]
    return Structure(
        symbols=symbols,
        geometry=geometry,
        charge=structure.charge,
        multiplicity=structure.multiplicity,
    )


def realign_end_to_start(atom_map: AtomMapping, struct_end: Structure) -> Structure:
    """Reorder `struct_end`'s atoms to align with the `start` structure
    `atom_map` was computed against."""
    return reorder_structure(struct_end, atom_map.as_order())


def map_smiles_pair(
    smi_start: str,
    smi_end: str,
    *,
    charge_start: Optional[int] = None,
    charge_end: Optional[int] = None,
    multiplicity_start: int = 1,
    multiplicity_end: int = 1,
) -> tuple[Structure, Structure]:
    """Build a pair of 3D `Structure`s from two reaction-endpoint SMILES
    strings, with atom indices made consistent across the pair via
    SLAPMapper's atom-to-atom mapping.

    Parsing/embedding each SMILES independently would give each structure
    its own arbitrary (canonical-SMILES-order) atom indexing with no
    correspondence between the two -- exactly the case SLAPMapper's
    chemistry-aware extension (`slapmapper.aam.SlapAAM.map_smiles`) is for.
    """
    _require_slapmapper()
    from rdkit import Chem
    from slapmapper.aam import SlapAAM

    for smi in (smi_start, smi_end):
        mol = Chem.MolFromSmiles(smi)
        # RDKit parses "" as a valid 0-atom Mol rather than returning None;
        # SlapAAM's native mapper segfaults on an empty-molecule reaction, so
        # this must be rejected here rather than relying on the None check.
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"{smi!r} is not a valid (non-empty) SMILES string.")

    mapper = SlapAAM(binary=True)
    mapper.map_smiles(f"{smi_start}>>{smi_end}")
    if not mapper.results:
        raise ValueError(
            f"SLAPMapper could not find an atom mapping between "
            f"{smi_start!r} and {smi_end!r}."
        )

    mapped_smi_start, mapped_smi_end = mapper.results[0]["smiles"].split(">>")

    mol_start = Molecule.from_mapped_smiles(mapped_smi_start)
    mol_end = Molecule.from_mapped_smiles(mapped_smi_end)

    struct_start = molecule_to_structure(
        mol_start,
        charge=charge_start if charge_start is not None else mol_start.charge,
        spinmult=multiplicity_start,
    )
    struct_end = molecule_to_structure(
        mol_end,
        charge=charge_end if charge_end is not None else mol_end.charge,
        spinmult=multiplicity_end,
    )
    return struct_start, struct_end
