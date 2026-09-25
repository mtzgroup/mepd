"""`snap_rmsd` that also works for structures containing an atom with no
bonds -- a bare ion such as Br-, Cl- or Na+ in a complex.

qcinf <= 0.4.1 builds its molecular graph from the atoms that have bonds, so
any unbonded atom breaks it ("vertex N conflicts with number_of_vertices=N",
and deeper in SNAP "Not all indices got fixed!"): `qcinf.snap_rmsd` fails
even for a structure compared with itself. `mepd channels` read that as
"conformer not isomorphic" and discarded every CREST conformer of e.g.
[Br-].CO.O.O.O.

Here the bonded atoms go through qcinf unchanged (symmetry/permutation
aware), the rigid transform qcinf applied to them is recovered and applied
to the unbonded atoms, and those are matched element by element to the
other structure's unbonded atoms (least squares). Structures without
unbonded atoms take the plain qcinf path. Drop this once qcinf handles
isolated vertices upstream.
"""

from __future__ import annotations

import numpy as np
import qcinf
from qcconst import constants
from qcdata import Structure


def _isolated_atoms(structure: Structure, cov_factor: float) -> list[int]:
    from qcinf.algorithms.connectivity import compute_connectivity

    bonded = set()
    for i, j, _ in compute_connectivity(structure, cov_factor=cov_factor):
        bonded.add(int(i))
        bonded.add(int(j))
    return [i for i in range(len(structure.symbols)) if i not in bonded]


def _subset(structure: Structure, indices: list[int]) -> Structure:
    geom = np.asarray(structure.geometry)[indices]
    return Structure(symbols=[structure.symbols[i] for i in indices], geometry=geom,
                     charge=structure.charge, multiplicity=structure.multiplicity)


def _rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R, c_src, c_dst with dst ~= (src - c_src) @ R + c_dst (Kabsch)."""
    c_src, c_dst = src.mean(axis=0), dst.mean(axis=0)
    if len(src) < 2:
        return np.eye(3), c_src, c_dst
    h = (src - c_src).T @ (dst - c_dst)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt, c_src, c_dst


def _match_ions(a_pos: np.ndarray, a_sym: list[str], b_pos: np.ndarray, b_sym: list[str]) -> float:
    """Sum of squared distances of the best element-preserving assignment."""
    from scipy.optimize import linear_sum_assignment

    total = 0.0
    for element in sorted(set(a_sym)):
        ia = [k for k, s in enumerate(a_sym) if s == element]
        ib = [k for k, s in enumerate(b_sym) if s == element]
        cost = ((a_pos[ia][:, None, :] - b_pos[ib][None, :, :]) ** 2).sum(axis=-1)
        rows, cols = linear_sum_assignment(cost)
        total += float(cost[rows, cols].sum())
    return total


def snap_rmsd(a: Structure, b: Structure, **kwargs) -> float:
    """`qcinf.snap_rmsd(a, b, **kwargs)`, extended to unbonded atoms.
    Raises ValueError, like qcinf, when the two are not the same molecule."""
    cov_factor = kwargs.get("cov_factor", 1.2)
    a_iso, b_iso = _isolated_atoms(a, cov_factor), _isolated_atoms(b, cov_factor)
    if not a_iso and not b_iso:
        return qcinf.snap_rmsd(a, b, **kwargs)
    a_ion_sym = sorted(a.symbols[i] for i in a_iso)
    if a_ion_sym != sorted(b.symbols[i] for i in b_iso):
        raise ValueError("structures are not isomorphic (different unbonded atoms)")
    n = len(a.symbols)
    if len(b.symbols) != n:
        raise ValueError("structures are not isomorphic (different atom counts)")

    a_rest = [i for i in range(n) if i not in set(a_iso)]
    b_rest = [i for i in range(n) if i not in set(b_iso)]
    a_geom, b_geom = np.asarray(a.geometry, float), np.asarray(b.geometry, float)
    units = str(kwargs.get("units", "bohr")).lower()
    if a_rest:
        sub_kwargs = {**kwargs, "units": "bohr"}
        rmsd_rest, a_rest_aligned, _ = qcinf.snap_rmsd_align_assign(_subset(a, a_rest), _subset(b, b_rest), **sub_kwargs)
        rot, c_src, c_dst = _rigid_transform(a_geom[a_rest], np.asarray(a_rest_aligned.geometry, float))
        rest_sq = float(rmsd_rest) ** 2 * len(a_rest)
    else:  # nothing but unbonded atoms: superpose the centroids only
        rot, c_src, c_dst = np.eye(3), a_geom.mean(axis=0), b_geom.mean(axis=0)
        rest_sq = 0.0
    a_ions = (a_geom[a_iso] - c_src) @ rot + c_dst
    ion_sq = _match_ions(a_ions, [a.symbols[i] for i in a_iso], b_geom[b_iso], [b.symbols[i] for i in b_iso])
    rmsd = float(np.sqrt((rest_sq + ion_sq) / n))
    return rmsd * constants.BOHR_TO_ANGSTROM if units.startswith("ang") else rmsd
