"""Substituent scans: replace hydrogens of an existing reaction with
functional groups and see how its barrier (and, across channels, its
selectivity) responds.

A *site* is a hydrogen on a heavy atom that keeps its bonds through the
reaction (both IRC ends); symmetry-equivalent hydrogens (e.g. the three of
a methyl group) are one site. Each group is built once in 3D (RDKit) and
attached the same way to the reactant end, the TS and the product end:
along the old X-H direction at the covalent bond length, turned about that
bond to stay clear of the other atoms. The same atom index keeps the
attachment atom, so all three geometries stay aligned.

* fast (default): only the new group is relaxed, with every other atom
  held at the parent's geometries, then single points. The TS is not a
  saddle point of the new molecule any more, so this is a first estimate,
  like solvent single points.
* reoptimize: the substituted TS is optimized again (with its IRC, which
  must connect the same bonds as the parent's), and the reactant end is
  minimized; the parent is put through the same protocol so the shift
  compares like with like.

The electronic trend is read against Hammett's sigma_para (C. Hansch,
A. Leo, R. W. Taft, Chem. Rev. 91, 165 (1991), doi:10.1021/cr00002a004).
sigma_p is defined for para-substituted benzoic acids, so outside aryl
systems it is only a generic donor/acceptor scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

BOHR_PER_ANGSTROM = 1 / 0.529177210903

# Hammett sigma_para (Hansch, Leo, Taft 1991).
SIGMA_P = {
    "methyl": -0.17, "ethyl": -0.15, "isopropyl": -0.15, "tert-butyl": -0.20, "vinyl": -0.04,
    "ethynyl": 0.23, "phenyl": -0.01, "hydroxyl": -0.37, "methoxy": -0.27, "amino": -0.66,
    "dimethylamino": -0.83, "fluoro": 0.06, "chloro": 0.23, "bromo": 0.23, "iodo": 0.18,
    "trifluoromethyl": 0.54, "cyano": 0.66, "nitro": 0.78, "formyl": 0.42, "acetyl": 0.50,
    "carboxyl": 0.45, "ester (CO2Me)": 0.45, "amide": 0.36,
}
# From strong donor to strong acceptor, small enough to be quick.
DEFAULT_GROUPS = ("amino", "hydroxyl", "methoxy", "methyl", "fluoro", "chloro", "trifluoromethyl", "cyano", "nitro")
SHORT = {"methyl": "Me", "ethyl": "Et", "isopropyl": "iPr", "tert-butyl": "tBu", "vinyl": "vinyl",
         "ethynyl": "C≡CH", "phenyl": "Ph", "benzyl": "Bn", "hydroxyl": "OH", "methoxy": "OMe", "amino": "NH2",
         "dimethylamino": "NMe2", "fluoro": "F", "chloro": "Cl", "bromo": "Br", "iodo": "I",
         "trifluoromethyl": "CF3", "cyano": "CN", "nitro": "NO2", "formyl": "CHO", "acetyl": "COMe",
         "carboxyl": "CO2H", "ester (CO2Me)": "CO2Me", "amide": "CONH2"}


def group_smiles() -> dict:
    from mepd.web.design import GROUPS

    return {k: v for k, v in GROUPS.items() if k != "H"}


def _rcov(symbol: str) -> float:
    from rdkit import Chem

    return Chem.GetPeriodicTable().GetRcovalent(symbol)   # Å


def adjacency(symbols, coords_bohr, scale: float = 1.25) -> np.ndarray:
    c = np.asarray(coords_bohr) / BOHR_PER_ANGSTROM
    r = np.array([_rcov(s) for s in symbols])
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    adj = d < scale * (r[:, None] + r[None])
    np.fill_diagonal(adj, False)
    return adj


@dataclass
class Site:
    h: int              # the hydrogen replaced
    anchor: int         # the heavy atom it is on
    equivalent: list    # symmetry-equivalent hydrogens (the same site)

    def label(self, symbols) -> str:
        return f"{symbols[self.anchor]}{self.anchor}"


def find_sites(symbols, reactant, product, *, only=None) -> list[Site]:
    """Every hydrogen bonded to the same heavy atom at both ends (so not
    itself transferred), one per symmetry-equivalent set. `only`: restrict
    to these hydrogen or anchor indices."""
    a_r, a_p = adjacency(symbols, reactant), adjacency(symbols, product)
    ranks = _graph_ranks(symbols, a_r)
    out: dict[tuple, Site] = {}
    for h, s in enumerate(symbols):
        if s != "H":
            continue
        nr, np_ = set(np.nonzero(a_r[h])[0]), set(np.nonzero(a_p[h])[0])
        if len(nr) != 1 or nr != np_:
            continue            # a hydrogen that moves (or is not simply bonded) is part of the reaction
        anchor = next(iter(nr))
        if symbols[anchor] == "H":
            continue
        if only is not None and h not in only and anchor not in only:
            continue
        key = (ranks[anchor], ranks[h])
        if key in out:
            out[key].equivalent.append(h)
        else:
            out[key] = Site(h=h, anchor=anchor, equivalent=[h])
    return sorted(out.values(), key=lambda s: (s.anchor, s.h))


def _graph_ranks(symbols, adj, rounds: int = 4) -> list:
    """Weisfeiler-Lehman labels: atoms that are equivalent in the bond graph
    get the same label (so a methyl group's hydrogens are one site)."""
    labels = list(symbols)
    for _ in range(rounds):
        labels = [hash((labels[i], tuple(sorted(labels[j] for j in np.nonzero(adj[i])[0]))))
                  for i in range(len(symbols))]
    return labels


@lru_cache(maxsize=None)
def template(group: str) -> tuple:
    """(symbols, coords in Å, dummy index, attachment index) of `group`,
    embedded with RDKit; the dummy stands where the anchor atom will be."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smi = group_smiles()[group].replace("[*]", "[2H]")
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    if AllChem.EmbedMolecule(mol, randomSeed=7) != 0:
        raise ValueError(f"could not build a 3D {group} group")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    dummy = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H" and a.GetIsotope() == 2)
    attach = mol.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
    xyz = mol.GetConformer().GetPositions()
    return tuple(a.GetSymbol() for a in mol.GetAtoms()), xyz, dummy, attach


def _rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def attach(symbols, coords_bohr, site: Site, group: str) -> tuple[list, np.ndarray, list]:
    """(symbols, coords in Bohr, indices of the new group's atoms): the
    site's hydrogen becomes the group's attachment atom (same index), the
    group's other atoms are appended."""
    gs, gx, dummy, att = template(group)
    c = np.asarray(coords_bohr, dtype=float) / BOHR_PER_ANGSTROM
    anchor, h = c[site.anchor], c[site.h]
    u = (h - anchor) / np.linalg.norm(h - anchor)
    w = gx[dummy] - gx[att]
    R = _rotation(w / np.linalg.norm(w), -u)
    length = _rcov(symbols[site.anchor]) + _rcov(gs[att])
    placed = (gx - gx[att]) @ R.T + anchor + length * u
    others = [i for i in range(len(gs)) if i not in (dummy, att)]
    rest = np.array([c[i] for i in range(len(c)) if i not in (site.h, site.anchor)])
    best, best_d = placed, -1.0
    for k in range(12):
        ang = 2 * np.pi * k / 12
        K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        Rs = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        trial = (placed - placed[att]) @ Rs.T + placed[att]
        dmin = float(np.min(np.linalg.norm(trial[others][:, None] - rest[None], axis=-1))) if others and len(rest) \
            else 9.9
        if dmin > best_d:
            best, best_d = trial, dmin
    new_symbols = list(symbols)
    new_symbols[site.h] = gs[att]
    new = c.copy()
    new[site.h] = best[att]
    new = np.vstack([new, best[others]]) if others else new
    new_symbols += [gs[i] for i in others]
    group_atoms = [site.h] + list(range(len(c), len(c) + len(others)))
    return new_symbols, new * BOHR_PER_ANGSTROM, group_atoms


def relax_group(engine, node, free: list, *, fmax: float = 0.05, steps: int = 300):
    """Minimize only the atoms in `free` (the rest held fixed), through ASE
    on `engine`'s own energies and gradients."""
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS

    from mepd.engines.gxtb import _GXTBASEResultsCalculator
    from mepd.qcdata_structure_helpers import ase_atoms_to_structure, structure_to_ase_atoms
    from mepd.nodes.node import StructureNode

    atoms = structure_to_ase_atoms(node.structure)
    atoms.info["charge"], atoms.info["spin"] = node.structure.charge, node.structure.multiplicity
    atoms.set_constraint(FixAtoms(indices=[i for i in range(len(atoms)) if i not in set(free)]))
    atoms.calc = _GXTBASEResultsCalculator(engine, charge=node.structure.charge,
                                           multiplicity=node.structure.multiplicity)
    LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    atoms.set_constraint()
    out = StructureNode(structure=ase_atoms_to_structure(atoms=atoms, charge=node.structure.charge,
                                                         multiplicity=node.structure.multiplicity))
    engine.compute_energies([out])
    return out


def hammett(shifts: dict[str, float]) -> Optional[dict]:
    """Least-squares slope of the barrier shift against sigma_p over the
    groups that have one: {"slope", "r2", "n"} (kcal/mol per sigma unit)."""
    pts = [(SIGMA_P[g], v) for g, v in shifts.items() if g in SIGMA_P and v is not None]
    if len(pts) < 4:
        return None
    x, y = np.array(pts).T
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = slope * x + icpt
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - pred) ** 2)) / ss if ss > 1e-12 else 0.0
    return {"slope": float(slope), "intercept": float(icpt), "r2": r2, "n": len(pts)}
