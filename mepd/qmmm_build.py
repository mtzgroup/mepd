"""Setting up QM/MM systems: a solute in a shell of explicit solvent, a
region on a structure the user brings (XYZ/PDB), or an existing TeraChem
QM/MM input (tc.in + prmtop + rst7 + qmindices) converted for mepd.

The solvent shell is a droplet: solvent molecules at the liquid's number
density, placed at random orientations around the (fixed) solute without
close contacts, then relaxed with GFN-FF while the solute is held fixed.
It is a starting structure, not an equilibrated liquid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

from mepd.qmmm import QMMMRegion

# SMILES, density (g/cm^3, 25 °C) and molar mass (g/mol) of shell solvents.
SOLVENTS = {
    "water": ("O", 0.997, 18.015),
    "methanol": ("CO", 0.792, 32.04),
    "ethanol": ("CCO", 0.789, 46.07),
    "acetonitrile": ("CC#N", 0.786, 41.05),
    "dmso": ("CS(C)=O", 1.100, 78.13),
    "acetone": ("CC(C)=O", 0.784, 58.08),
    "thf": ("C1CCOC1", 0.889, 72.11),
    "dichloromethane": ("ClCCl", 1.326, 84.93),
    "chloroform": ("ClC(Cl)Cl", 1.489, 119.38),
    "benzene": ("c1ccccc1", 0.876, 78.11),
    "hexane": ("CCCCCC", 0.659, 86.18),
}
_AVOGADRO = 6.02214076e23


def number_density(solvent: str) -> float:
    """Molecules per Å^3."""
    _, rho, mw = SOLVENTS[solvent]
    return rho / mw * _AVOGADRO * 1e-24


def _rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def solvate(solute, solvent: str = "water", shell: float = 6.0, *, n_molecules: Optional[int] = None,
            seed: int = 0, min_distance: float = 2.1):
    """The solute (atoms first, in their order, centred) inside a droplet of
    `solvent` reaching `shell` Å beyond its outermost atom. Returns a
    qcdata Structure with the solute's charge and spin."""
    from scipy.spatial import cKDTree

    from mepd.cli import _load_structure_from_smiles_or_xyz

    if solvent not in SOLVENTS:
        raise ValueError(f"unknown solvent {solvent!r}; known: {', '.join(SOLVENTS)}")
    smiles = SOLVENTS[solvent][0]
    mol = _load_structure_from_smiles_or_xyz(smiles, 0, 1)
    mxyz = np.asarray(mol.geometry) / ANGSTROM_TO_BOHR
    mxyz -= mxyz.mean(axis=0)
    mext = float(np.max(np.linalg.norm(mxyz, axis=1)))
    sxyz = np.asarray(solute.geometry) / ANGSTROM_TO_BOHR
    center = sxyz.mean(axis=0)
    sxyz = sxyz - center
    r_solute = float(np.max(np.linalg.norm(sxyz, axis=1)))
    radius = r_solute + float(shell)
    # Volume left for solvent: the sphere minus roughly the solute's own.
    v_solute = 4 / 3 * np.pi * (r_solute + 1.2) ** 3 * 0.55
    target = n_molecules if n_molecules is not None else max(
        1, int(round((4 / 3 * np.pi * radius ** 3 - v_solute) * number_density(solvent))))
    rng = np.random.default_rng(seed)
    placed = [sxyz]
    tree_pts = sxyz.copy()
    tree = cKDTree(tree_pts)
    added, tries = 0, 0
    pending: list[np.ndarray] = []
    while added < target and tries < 400 * target:
        tries += 1
        c = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(c) > radius - mext * 0.5:
            continue
        trial = mxyz @ _rotation(rng).T + c
        if tree.query(trial)[0].min() < min_distance:
            continue
        if pending and np.min(np.linalg.norm(np.vstack(pending)[:, None] - trial[None], axis=2)) < min_distance:
            continue
        pending.append(trial)
        added += 1
        if len(pending) >= 20:   # rebuild the neighbour tree now and then
            tree_pts = np.vstack([tree_pts, *pending])
            placed.extend(pending)
            pending = []
            tree = cKDTree(tree_pts)
    placed.extend(pending)
    symbols = list(solute.symbols) + list(mol.symbols) * added
    xyz = np.vstack(placed)
    from qcdata import Structure

    return Structure(symbols=symbols, geometry=xyz * ANGSTROM_TO_BOHR, charge=int(solute.charge),
                     multiplicity=int(solute.multiplicity))


def relax_environment(structure, fixed: Sequence[int], *, maxiter: int = 300, fmax: float = 0.05):
    """GFN-FF minimization of everything but `fixed` (the solute)."""
    from mepd.engines.frozen import FrozenAtomsEngine
    from mepd.engines.gfnff import GFNFFEngine
    from mepd.nodes.node import StructureNode

    eng = FrozenAtomsEngine(base=GFNFFEngine(n_parallel=1), frozen=list(fixed))
    eng.base.topology_reference = structure.to_xyz()
    traj = eng.compute_geometry_optimization(StructureNode(structure=structure, has_molecular_graph=False),
                                             keywords={"maxiter": maxiter, "fmax": fmax})
    return traj[-1].structure


def solvated_system(solute, solvent: str = "water", shell: float = 6.0, *, qm_atoms=None,
                    active_radius: Optional[float] = 5.0, mm: str = "gfnff", relax: bool = True, seed: int = 0,
                    n_molecules: Optional[int] = None, name: str = "") -> tuple:
    """(system Structure, QMMMRegion): the solute is the QM region unless
    `qm_atoms` says otherwise."""
    system = solvate(solute, solvent, shell, seed=seed, n_molecules=n_molecules)
    nsolute = len(solute.symbols)
    if relax:
        system = relax_environment(system, list(range(nsolute)))
    qm = list(range(nsolute)) if qm_atoms is None else qm_atoms
    region = QMMMRegion.build(system, qm, qm_charge=int(solute.charge), qm_multiplicity=int(solute.multiplicity),
                              active_radius=active_radius, mm=mm, name=name or f"in {solvent}",
                              solute_atoms=list(range(nsolute)))
    return system, region


def _clear_contacts(y: np.ndarray, env: Optional[np.ndarray], min_distance: float = 1.8,
                    stiffness: float = 10.0) -> np.ndarray:
    """Move a molecule (Angstrom) out of contact with fixed atoms `env`
    while keeping its own geometry: its internal distances are stiff springs
    (a TS keeps its partial bonds), contacts closer than `min_distance` are
    pushed apart. Mostly a small rigid shift and turn."""
    from scipy.optimize import minimize

    if env is None or not len(env):
        return y
    n = len(y)
    iu, ju = np.triu_indices(n, 1)
    d0 = np.linalg.norm(y[iu] - y[ju], axis=1)

    def fg(flat):
        x = flat.reshape(n, 3)
        dv = x[iu] - x[ju]
        d = np.linalg.norm(dv, axis=1) + 1e-12
        e = stiffness * np.sum((d - d0) ** 2)
        g = np.zeros_like(x)
        c = (2 * stiffness * (d - d0) / d)[:, None] * dv
        np.add.at(g, iu, c)
        np.add.at(g, ju, -c)
        de = x[:, None, :] - env[None]
        r = np.linalg.norm(de, axis=2) + 1e-12
        short = np.minimum(r - min_distance, 0.0)
        e += np.sum(short ** 2)
        g += np.sum((2 * short / r)[:, :, None] * de, axis=1)
        return e, g.ravel()

    res = minimize(fg, y.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return res.x.reshape(n, 3)


def embed(structure, system, region, *, relax: bool = True):
    """Another geometry of the solute (a product, a TS, a conformer: the
    solute's atoms in the solute's order) put into a QM/MM system, in place
    of the solute as it is in `system` (a structure of that system, e.g. the
    solvated reactant): aligned onto the solute's atoms there, then (`relax`)
    nudged out of contact with the environment, which stays exactly as it
    is. Returns (the system with the new solute, a report dict). The result
    is a starting geometry: minimize it with QM/MM (or optimize it as a TS)
    before using its energy."""
    from mepd.qmmm import bonds, diagnose
    from mepd.rigid_alignment import kabsch_align

    solute = region.solute
    sym_sys = [str(system.symbols[i]) for i in solute]
    if [str(x) for x in structure.symbols] != sym_sys:
        raise ValueError(f"the structure has atoms {''.join(map(str, structure.symbols))[:60]}..., the solute in this "
                         f"system {''.join(sym_sys)[:60]}...: it must list the solute's atoms in the same order")
    if len(system.symbols) != region.natoms:
        raise ValueError(f"the system structure has {len(system.symbols)} atoms, the QM/MM system {region.natoms}")
    xs = np.asarray(system.geometry, dtype=float) / ANGSTROM_TO_BOHR
    y = kabsch_align(np.asarray(structure.geometry, dtype=float) / ANGSTROM_TO_BOHR, xs[solute])
    if relax:
        inside = set(solute)
        # Environment atoms near the solute, except those bonded to it (cut bonds).
        bonded = {j if i in inside else i for i, j in bonds([str(a) for a in system.symbols], xs)
                  if (i in inside) != (j in inside)}
        env_idx = np.array([i for i in range(len(xs)) if i not in inside and i not in bonded], dtype=int)
        if len(env_idx):
            d = np.linalg.norm(xs[env_idx][:, None] - y[None], axis=2).min(axis=1)
            env = xs[env_idx[d < 6.0]]
        else:
            env = None
        y = _clear_contacts(y, env)
    out = xs.copy()
    out[solute] = y
    result = system.model_copy(update={"geometry": out * ANGSTROM_TO_BOHR, "charge": int(system.charge),
                                       "multiplicity": int(system.multiplicity)})
    rep = diagnose(region, [system, result])
    return result, {"closest_contact": rep["frames"][1]["closest_contact"],
                    "rmsd_from_solute": float(np.sqrt(np.mean(np.sum((y - xs[solute]) ** 2, axis=1)))),
                    "warnings": [w for w in rep["warnings"] if "frozen" not in w]}


# ------------------------------------------------------------ AMBER / TeraChem
def from_amber(prmtop: Path, coords: Path, qm_atoms, *, qm_charge: int = 0, qm_multiplicity: int = 1,
               charge: Optional[int] = None, active_radius: Optional[float] = None, frozen_atoms=(),
               mm: str = "amber", tcin: Optional[Path] = None) -> tuple:
    """A system and region from AMBER files (prmtop + rst7/inpcrd, or a PDB
    for the coordinates)."""
    from qcdata import Structure

    from mepd.engines.terachem_qmmm import prmtop_symbols, rst7_coords

    symbols = prmtop_symbols(Path(prmtop).read_text())
    if Path(coords).suffix.lower() == ".pdb":
        import openmm.app as app

        pos = app.PDBFile(str(coords)).getPositions(asNumpy=True)
        xyz = np.asarray(pos._value) * 10.0
    else:
        xyz = rst7_coords(Path(coords).read_text())
    if len(xyz) != len(symbols):
        raise ValueError(f"{coords} has {len(xyz)} atoms, {prmtop} {len(symbols)}")
    system = Structure(symbols=symbols, geometry=xyz * ANGSTROM_TO_BOHR,
                       charge=int(qm_charge if charge is None else charge), multiplicity=int(qm_multiplicity))
    region = QMMMRegion.build(system, qm_atoms, qm_charge=qm_charge, qm_multiplicity=qm_multiplicity,
                              active_radius=active_radius, frozen_atoms=frozen_atoms, mm=mm,
                              prmtop=str(Path(prmtop).resolve()),
                              tcin=str(Path(tcin).resolve()) if tcin else None)
    return system, region


def from_terachem(tcin: Path, *, mm: str = "amber", active_radius: Optional[float] = None) -> tuple:
    """A system, region and QM settings from a TeraChem QM/MM input (its
    prmtop, coordinates and qmindices files are found next to it).
    `mm` = "amber" runs it now with any QM engine through OpenMM;
    "terachem" keeps TeraChem's own QM/MM (needs TeraChem/ChemCloud)."""
    from mepd.engines.terachem_qmmm import parse_tcin

    tcin = Path(tcin)
    p = parse_tcin(tcin.read_text())
    for key in ("prmtop", "coordinates", "qmindices"):
        if not p.get(key):
            raise ValueError(f"{tcin} has no `{key}` line: is it a QM/MM input?")
    here = tcin.parent
    qm = []
    for line in (here / p["qmindices"]).read_text().splitlines():
        line = line.split("#", 1)[0]
        qm.extend(int(t) for t in line.replace(",", " ").split() if t.lstrip("-").isdigit() and int(t) >= 0)
    system, region = from_amber(here / p["prmtop"], here / p["coordinates"], sorted(set(qm)),
                                qm_charge=int(p.get("charge", 0)), qm_multiplicity=int(p.get("spinmult", 1)),
                                active_radius=active_radius, frozen_atoms=p["frozen_atom_indices"], mm=mm,
                                tcin=tcin if mm == "terachem" else None)
    return system, region, {"method": p.get("method"), "basis": p.get("basis"), "keywords": p["keywords"]}
