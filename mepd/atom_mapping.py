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

    lg_start = molecule_to_labeled_graph(mol_start)
    lg_end = molecule_to_labeled_graph(mol_end)

    mapper = SlapMapper(binary=binary)
    mapper.get_maps([lg_start, lg_end])
    if not mapper.results:
        return None

    result = mapper.results[0]
    label2idxs_start = result["lgp"][0].label2idxs
    label2idxs_end = result["lgp"][1].label2idxs

    mapping: dict[int, int] = {}
    for label, idxs_start in label2idxs_start.items():
        idxs_end = label2idxs_end[label]
        for a, b in zip(sorted(idxs_start), sorted(idxs_end)):
            mapping[a] = b

    return AtomMapping(
        mapping=mapping, cost=result["val"], n_alternatives=len(mapper.results)
    )


def _symmetry_orbits(structure: Structure) -> list[list[int]]:
    """Groups of atom indices in `structure` that are topologically
    interchangeable -- same RDKit canonical rank via
    `Chem.CanonicalRankAtoms(breakTies=False)`, which assigns identical
    ranks to graph-symmetric atoms (e.g. a methyl group's three Hs, or a
    symmetric =CH2's two Hs). Only groups of size > 1 are actual degrees
    of freedom. Returns `[]` if RDKit can't parse/assign bonds -- never
    raises."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    try:
        mol = Chem.MolFromXYZBlock(structure.to_xyz())
        rdDetermineBonds.DetermineBonds(mol, charge=structure.charge)
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:
        return []

    groups: dict[int, list[int]] = {}
    for idx, rank in enumerate(ranks):
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
    mapper.get_maps([lg_start, lg_end], break_sym_targets=list(range(len(lg_start.labels))))
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
            candidates.append(expansion)
            if len(candidates) >= max_candidates:
                break

    return candidates


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
