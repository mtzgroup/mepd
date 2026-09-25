"""Candidate generation for the VRI search.

The IRC scan can only confirm a bifurcation when it is handed the right
TS1:

- Endpoint-driven TS searches (NEB/GSM/channels) land on an asymmetric
  saddle next to a symmetric ridge TS (cyclopentadiene dimerization: the
  bispericyclic TS is never sampled). `symmetric_ts_candidates` averages TS1
  with its images under near-symmetries of its bond graph and re-optimizes.
- `ridge_connected` checks that a TS2 lies downhill from TS1 with no
  intermediate (part of TS2 verification).
"""

from __future__ import annotations

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from mepd.discovery.vri import (
    _ensure_energy_gradient,
    labeled_bond_set,
    node_masses,
    same_species,
    stationary_point_modes,
)
from mepd.engines.engine import Engine
from mepd.nodes.node import Node

logger = logging.getLogger(__name__)

BOHR_TO_ANGSTROM = 0.529177210903
H2K = 627.509474

# Covalent radii (Angstrom), Cordero et al. 2008.
_COVALENT_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39, "Li": 1.28, "Na": 1.66, "Mg": 1.41, "Al": 1.21,
}


def _radius(symbol: str) -> float:
    return _COVALENT_RADII.get(symbol, 1.2)


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and centroids so that (P - p) @ R + q best fits Q."""
    p, q = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - p).T @ (Q - q))
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt, p, q


def aligned_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    R, p, q = _kabsch(A, B)
    return float(np.sqrt((((A - p) @ R + q - B) ** 2).sum(1).mean()))


# --------------------------------------------------------------------------
# Symmetric TS candidates
# --------------------------------------------------------------------------


def _partial_bond_graph(node: Node, scale: float = 1.3):
    """Heavy-atom graph with bonds up to `scale` x covalent distance, so the
    partially formed bonds of a TS count."""
    import networkx as nx

    symbols = list(node.symbols)
    X = np.asarray(node.coords) * BOHR_TO_ANGSTROM
    heavy = [i for i, s in enumerate(symbols) if s != "H"]
    G = nx.Graph()
    for i in heavy:
        G.add_node(i, element=symbols[i])
    for i, j in itertools.combinations(heavy, 2):
        if np.linalg.norm(X[i] - X[j]) < scale * (_radius(symbols[i]) + _radius(symbols[j])):
            G.add_edge(i, j)
    return G


def _symmetrized_geometry(node: Node, heavy_perm: dict) -> Optional[np.ndarray]:
    """Average of the geometry and its image under the heavy-atom
    permutation (hydrogens follow their heavy atom, matched by distance
    after alignment). None if hydrogen counts do not match."""
    symbols = list(node.symbols)
    X = np.asarray(node.coords, dtype=float)
    heavy = sorted(heavy_perm)
    hydrogens = [i for i, s in enumerate(symbols) if s == "H"]
    owner = {h: min(heavy, key=lambda c: np.linalg.norm(X[h] - X[c])) for h in hydrogens}
    R, p, q = _kabsch(X[[heavy_perm[c] for c in heavy]], X[heavy])
    Y = (X - p) @ R + q  # image: atom perm[i] of Y should sit on atom i
    perm = dict(heavy_perm)
    for c in heavy:
        mine = [h for h in hydrogens if owner[h] == c]
        theirs = [h for h in hydrogens if owner[h] == heavy_perm[c]]
        if len(mine) != len(theirs):
            return None
        if mine:
            best = min(
                itertools.permutations(theirs),
                key=lambda t: sum(np.linalg.norm(Y[a] - X[b]) for a, b in zip(t, mine)),
            )
            for a, b in zip(mine, best):
                perm[a] = b
    return 0.5 * (X + np.array([Y[perm[i]] for i in range(len(symbols))]))


def symmetric_ts_candidates(
    ts_node: Node,
    engine: Engine,
    *,
    imaginary_cutoff: float = 50.0,
    rmsd_window: tuple = (0.05, 1.0),
    max_automorphisms: int = 200,
    workers: int = 1,
) -> List[Node]:
    """First-order saddles obtained by symmetrizing TS1 under near-symmetries
    of its partial-bond graph and re-optimizing, distinct from TS1 and from
    each other. `rmsd_window` (Angstrom): a permutation counts as a near-
    symmetry when the permuted geometry aligns onto TS1 within it (below
    the window TS1 is already symmetric under it)."""
    from networkx.algorithms.isomorphism import GraphMatcher

    if node_masses(ts_node) is None:
        return []
    G = _partial_bond_graph(ts_node)
    X = np.asarray(ts_node.coords, dtype=float)
    heavy = sorted(G.nodes)
    matcher = GraphMatcher(G, G, node_match=lambda a, b: a["element"] == b["element"])
    guesses: List[np.ndarray] = []
    for k, mapping in enumerate(matcher.isomorphisms_iter()):
        if k >= max_automorphisms:
            break
        if all(mapping[i] == i for i in heavy):
            continue
        rmsd = aligned_rmsd(X[[mapping[i] for i in heavy]], X[heavy]) * BOHR_TO_ANGSTROM
        if not (rmsd_window[0] < rmsd < rmsd_window[1]):
            continue
        Z = _symmetrized_geometry(ts_node, mapping)
        if Z is None:
            continue
        if any(aligned_rmsd(Z, g) * BOHR_TO_ANGSTROM < 0.05 for g in guesses):
            continue
        guesses.append(Z)
    if not guesses:
        return []

    def _opt(Z):
        try:
            result = engine.compute_transition_state(node=ts_node.update_coords(Z))
            modes = stationary_point_modes(result, engine)
            return result if modes.n_imaginary(imaginary_cutoff) == 1 else None
        except Exception as exc:
            logger.info("Symmetric TS candidate failed: %s", exc)
            return None

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        optimized = list(pool.map(_opt, guesses))
    kept: List[Node] = []
    for node in optimized:
        if node is None:
            continue
        coords = np.asarray(node.coords)
        if aligned_rmsd(coords, X) * BOHR_TO_ANGSTROM < 0.1:
            continue
        if any(aligned_rmsd(coords, np.asarray(k.coords)) * BOHR_TO_ANGSTROM < 0.1 for k in kept):
            continue
        kept.append(node)
    return kept


# --------------------------------------------------------------------------
# Ridge check
# --------------------------------------------------------------------------


def ridge_connected(
    ts1: Node,
    ts2: Node,
    engine: Engine,
    *,
    n_images: int = 15,
    tol_kcal: float = 1.0,
    save_path=None,
) -> dict:
    """Whether TS2 lies downhill from TS1 with nothing in between: along a
    geodesic TS1 -> TS2 path no interior point rises more than `tol_kcal`
    above TS1 and none dips more than `tol_kcal` below TS2. An unrelaxed
    geodesic path over-estimates barriers, so passing is a strong sign;
    dipping below TS2 means the path runs through a product valley (TS2 is
    an ordinary isomerization of P1, not the ridge saddle)."""
    import mepd.chainhelpers as ch
    from mepd.inputs import ChainInputs

    try:
        chain = ch.run_geodesic(chain=[ts1.copy(), ts2.copy()], chain_inputs=ChainInputs(), nimages=n_images)
        e = np.asarray(engine.compute_energies(chain.nodes), dtype=float)
    except Exception as exc:
        return {"connected": None, "error": f"{type(exc).__name__}: {exc}"}
    rel = (e - e[0]) * H2K
    inner = rel[1:-1]
    if save_path is not None:
        try:
            chain.write_to_disk(save_path)
        except Exception as exc:
            logger.info("could not save ridge path: %s", exc)
    connected = bool(inner.max() <= tol_kcal and inner.min() >= rel[-1] - tol_kcal)
    return {
        "connected": connected,
        "max_interior_kcal": float(inner.max()),
        "min_interior_kcal": float(inner.min()),
        "ts2_rel_ts1_kcal": float(rel[-1]),
        "profile_kcal": [float(x) for x in rel],
    }
