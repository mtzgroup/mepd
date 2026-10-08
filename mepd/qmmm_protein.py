"""A species (molecule or complex) inside a protein, as a QM/MM system.

1. **Prepare** the protein (`prepare_protein`): chains kept, ligands and
   crystal waters removed, missing atoms and hydrogens added at a pH
   (PDBFixer), checked against the AMBER force field it will run with.
2. **Find sites** (`find_sites`): the species is docked as a rigid body (its
   own geometry, e.g. a reactant from QM, is kept) with AutoDock Vina in
   overlapping boxes covering the protein; the poses are grouped into sites,
   each with its best Vina score, how many boxes found it, and the residues
   lining it.
3. **Build** a QM/MM system at a site (`build_site_system`): the protein
   (AMBER ff14SB) and a TIP3P water shell around the species are the
   environment, the species (plus any residue side chains asked for) the QM
   region; the protein's and water's charges act on it (electrostatic
   embedding, so the QM level must take point charges: engine_name = "xtb"
   or "psi4"). The force-field atoms come first, in the PDB's order, the QM
   species last. Everything but the water and residues near the species is
   frozen, or the whole protein with `freeze_protein`.

The system then goes through the same QM/MM operations as a solvated one
(embedding a product, paths, TS searches).

AutoDock Vina: J. Eberhardt, D. Santos-Martins, A. F. Tillack, S. Forli, J.
Chem. Inf. Model. 61, 3891 (2021), doi:10.1021/acs.jcim.1c00203; O. Trott,
A. J. Olson, J. Comput. Chem. 31, 455 (2010), doi:10.1002/jcc.21334.
PDBFixer/OpenMM: P. Eastman et al., PLoS Comput. Biol. 13, e1005659 (2017),
doi:10.1371/journal.pcbi.1005659. ff14SB: J. A. Maier et al., J. Chem.
Theory Comput. 11, 3696 (2015), doi:10.1021/acs.jctc.5b00255.
"""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

FORCEFIELD = ["amber14-all.xml", "amber14/tip3p.xml"]
# Formal charge of a side chain cut at CA-CB (protonation states as PDBFixer
# assigns them; HIP is the doubly protonated histidine).
SIDE_CHAIN_CHARGE = {"ARG": 1, "LYS": 1, "HIP": 1, "ASP": -1, "GLU": -1}
INSTALL = "pip install vina pdbfixer (or pip install 'mepd[qmmm]')"


def _need(*mods):
    """Check without importing: vina imported before openbabel crashes the
    process (their SWIG runtimes clash), so vina is only imported in the
    docking workers."""
    import importlib.util

    for m in mods:
        if importlib.util.find_spec(m) is None:
            raise ImportError(f"the protein workflow needs {m}: {INSTALL}")


# ---------------------------------------------------------------- prepare
def fetch_pdb(code: str, dest: Path) -> Path:
    """Download a PDB entry from RCSB (`code`: 4 characters)."""
    code = code.strip().upper()
    if len(code) != 4 or not code.isalnum():
        raise ValueError(f"{code!r} is not a PDB ID (4 letters/digits)")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fp = dest / f"{code}.pdb"
    with urllib.request.urlopen(f"https://files.rcsb.org/download/{code}.pdb", timeout=60) as r:
        fp.write_bytes(r.read())
    return fp


def prepare_protein(source: Path, out_pdb: Path, *, chains: Optional[Sequence[str]] = None, ph: float = 7.0,
                    keep_water: bool = False) -> dict:
    """The protein of `source` (a PDB file) ready for the force field:
    `chains` kept (all if None), every other residue that is not standard
    protein removed (ligands, ions, crystal waters unless `keep_water`),
    missing heavy atoms and hydrogens added at `ph`. Missing loops are not
    rebuilt (their ends stay as they are). Writes `out_pdb`; returns a report."""
    _need("pdbfixer", "openmm")
    import openmm
    import openmm.app as app
    from pdbfixer import PDBFixer

    fx = PDBFixer(filename=str(source))
    all_chains = sorted({c.id for c in fx.topology.chains()})
    if chains:
        want = {c.strip() for c in chains}
        unknown = want - set(all_chains)
        if unknown:
            raise ValueError(f"no chain {', '.join(sorted(unknown))} in {Path(source).name} "
                             f"(it has {', '.join(all_chains)})")
        fx.removeChains([i for i, c in enumerate(fx.topology.chains()) if c.id not in want])
    fx.findMissingResidues()
    gaps = len(fx.missingResidues)
    fx.missingResidues = {}
    fx.findNonstandardResidues()
    replaced = [f"{r.name}{r.id}→{std}" for r, std in fx.nonstandardResidues]
    fx.replaceNonstandardResidues()
    removed = sorted({r.name for r in fx.topology.residues()
                      if r.name not in _PROTEIN and not (keep_water and r.name == "HOH")})
    fx.removeHeterogens(keepWater=keep_water)
    fx.findMissingAtoms()
    n_missing = sum(len(v) for v in fx.missingAtoms.values())
    fx.addMissingAtoms()
    fx.addMissingHydrogens(ph)
    ff = app.ForceField(*FORCEFIELD)
    system = ff.createSystem(fx.topology, nonbondedMethod=app.NoCutoff)
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    charge = sum(nb.getParticleParameters(i)[0].value_in_unit(openmm.unit.elementary_charge)
                 for i in range(nb.getNumParticles()))
    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeFile(fx.topology, fx.positions, fh, keepIds=True)
    return {"pdb": str(out_pdb), "atoms": fx.topology.getNumAtoms(), "residues": fx.topology.getNumResidues(),
            "chains": sorted({c.id for c in fx.topology.chains()}), "charge": int(round(charge)),
            "gaps_not_rebuilt": gaps, "heavy_atoms_added": n_missing, "replaced": replaced,
            "removed": removed, "ph": ph}


_PROTEIN = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
            "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN", "ACE", "NME"}


# ---------------------------------------------------------------- dock
def receptor_pdbqt(protein_pdb: Path, out: Path) -> Path:
    from openbabel import openbabel as ob, pybel

    ob.obErrorLog.SetOutputLevel(0)
    mol = next(pybel.readfile("pdb", str(protein_pdb)))
    mol.write("pdbqt", str(out), overwrite=True, opt={"r": None})
    return Path(out)


def ligand_pdbqt(symbols: Sequence[str], coords_angstrom: np.ndarray) -> tuple[str, list[int]]:
    """A rigid Vina ligand of these atoms (one rigid body, no torsions;
    several molecules stay in their arrangement). Returns (PDBQT text, the
    input index of each of its atoms: Vina's united atoms drop nonpolar H)."""
    from openbabel import openbabel as ob, pybel

    ob.obErrorLog.SetOutputLevel(0)
    x = np.asarray(coords_angstrom, dtype=float)
    xyz = f"{len(symbols)}\n\n" + "".join(f"{s} {a:.6f} {b:.6f} {c:.6f}\n" for s, (a, b, c) in zip(symbols, x))
    text = pybel.readstring("xyz", xyz).write("pdbqt", opt={"r": None, "n": None})
    atoms = [ln for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]
    pos = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in atoms])
    index = [int(np.argmin(np.linalg.norm(x - p, axis=1))) for p in pos]
    return "ROOT\n" + "\n".join(atoms) + "\nENDROOT\nTORSDOF 0\n", index


def _kabsch(a: np.ndarray, b: np.ndarray):
    """Rotation r and translation t with a @ r.T + t ≈ b."""
    ca, cb = a.mean(0), b.mean(0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cb - ca @ r.T


def box_centers(protein_xyz: np.ndarray, *, spacing: float = 12.0, min_atoms: int = 40,
                reach: float = 10.0) -> np.ndarray:
    """Box centres on a grid over the protein: those with at least
    `min_atoms` protein heavy atoms within `reach` Å (on or in the protein,
    not in open solvent)."""
    from scipy.spatial import cKDTree

    lo, hi = protein_xyz.min(0) - 4.0, protein_xyz.max(0) + 4.0
    axes = [np.arange(a, b + 1e-6, spacing) + ((b - a) % spacing) / 2 for a, b in zip(lo, hi)]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    tree = cKDTree(protein_xyz)
    counts = np.array([len(h) for h in tree.query_ball_point(grid, r=reach)])
    return grid[counts >= min_atoms]


def _dock_box(args) -> list[dict]:
    receptor, ligand, center, size, exhaustiveness, n_poses, seed = args
    from vina import Vina

    v = Vina(sf_name="vina", cpu=1, seed=int(seed), verbosity=0)
    v.set_receptor(str(receptor))
    v.set_ligand_from_string(ligand)
    v.compute_vina_maps(center=[float(c) for c in center], box_size=[float(size)] * 3)
    v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
    energies = v.energies(n_poses=int(n_poses))[:, 0]
    out, block = [], []
    for ln in v.poses(n_poses=int(n_poses)).splitlines():
        if ln.startswith("MODEL"):
            block = []
        elif ln.startswith(("ATOM", "HETATM")):
            block.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        elif ln.startswith("ENDMDL"):
            out.append(block)
    return [{"score": float(e), "coords": c, "box": [float(x) for x in center]} for e, c in zip(energies, out)]


@dataclass
class Site:
    id: int
    score: float                     # best Vina score, kcal/mol (lower binds better)
    centroid: list[float]            # Å
    coords: list[list[float]]        # the species placed there: every atom, input order, Å
    hits: int                        # poses (from all boxes) that landed here
    residues: list[str] = field(default_factory=list)   # e.g. "A:ARG90", within 4.5 Å
    scores: list[float] = field(default_factory=list)   # every pose's score here


def find_sites(protein_pdb: Path, symbols: Sequence[str], coords_angstrom: np.ndarray, workdir: Path, *,
               box: float = 20.0, spacing: float = 12.0, exhaustiveness: int = 8, poses_per_box: int = 3,
               cluster: float = 4.0, max_sites: int = 20, workers: int = 4, seed: int = 1,
               centers: Optional[np.ndarray] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> list[Site]:
    """Sites where the species docks (rigid, its own geometry), best first."""
    _need("vina", "openmm")
    import openmm.app as app
    import openmm.unit as unit

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pdb = app.PDBFile(str(protein_pdb))
    atoms = list(pdb.topology.atoms())
    pxyz = np.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    heavy = np.array([a.element is not None and a.element.symbol != "H" for a in atoms])
    receptor = receptor_pdbqt(protein_pdb, workdir / "receptor.pdbqt")
    x0 = np.asarray(coords_angstrom, dtype=float)
    ligand, index = ligand_pdbqt(symbols, x0)
    ref = x0[index]
    if centers is None:
        centers = box_centers(pxyz[heavy], spacing=spacing)
    jobs = [(receptor, ligand, c, box, exhaustiveness, poses_per_box, seed + k) for k, c in enumerate(centers)]
    poses = []
    import multiprocessing

    # Fresh interpreters: Vina and openbabel's native code do not survive a fork.
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=multiprocessing.get_context("spawn")) as pool:
        for k, res in enumerate(pool.map(_dock_box, jobs)):
            poses += res
            if progress:
                progress(k + 1, len(jobs))
    poses.sort(key=lambda p: p["score"])
    from scipy.spatial import cKDTree

    tree = cKDTree(pxyz)
    names = [f"{a.residue.chain.id}:{a.residue.name}{a.residue.id}" for a in atoms]
    sites: list[Site] = []
    for p in poses:
        placed = np.asarray(p["coords"])
        c = placed.mean(0)
        home = next((s for s in sites if np.linalg.norm(np.asarray(s.centroid) - c) < cluster), None)
        if home is not None:
            home.hits += 1
            home.scores.append(p["score"])
            continue
        r, t = _kabsch(ref, placed)
        full = x0 @ r.T + t
        near = sorted({names[i] for hits in tree.query_ball_point(full, r=4.5) for i in hits},
                      key=lambda s: (s.split(":")[0], int("".join(ch for ch in s.split(":")[1] if ch.isdigit()) or 0)))
        sites.append(Site(id=len(sites), score=p["score"], centroid=[float(v) for v in c],
                          coords=full.round(4).tolist(), hits=1, residues=near, scores=[p["score"]]))
    return sites[:max_sites]


# ---------------------------------------------------------------- build
def build_site_system(protein_pdb: Path, symbols: Sequence[str], coords_angstrom: np.ndarray, out_dir: Path, *,
                      ligand_charge: int = 0, ligand_multiplicity: int = 1, qm_residues: Sequence[str] = (),
                      water_shell: float = 8.0, active_radius: float = 6.0, freeze_protein: bool = False,
                      cutoff: Optional[float] = 12.0, name: str = "") -> dict:
    """The species at these coordinates (Å) in the protein, with a TIP3P water
    shell of `water_shell` Å around it, as a QM/MM system in `out_dir`:
    system.xyz, environment.pdb (the force-field atoms: protein, then water)
    and region.json. `qm_residues` ("A:ARG90" or "ARG90"): side chains (from
    CB on) put in the QM region too. Returns the paths and a summary."""
    _need("openmm")
    import openmm.app as app
    import openmm.unit as unit
    from qcdata import Structure
    from scipy.spatial import cKDTree

    from mepd.qmmm import QMMMRegion

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lig = np.asarray(coords_angstrom, dtype=float)
    pdb = app.PDBFile(str(protein_pdb))
    ff = app.ForceField(*FORCEFIELD)
    model = app.Modeller(pdb.topology, pdb.positions)
    n_protein = model.topology.getNumAtoms()
    # Water everywhere, then only a shell around the species, out of its way.
    model.addSolvent(ff, model="tip3p", padding=0.6 * unit.nanometer, neutralize=False)
    pos = np.asarray(model.getPositions().value_in_unit(unit.angstrom))
    tree = cKDTree(lig)
    drop = []
    for res in model.topology.residues():
        if res.name != "HOH":
            continue
        idx = [a.index for a in res.atoms()]
        d = tree.query(pos[idx])[0]
        if d.min() < 2.4 or d.max() > water_shell:
            drop.append(res)
    model.delete(drop)
    relax = _relax_environment(model, ff, symbols, lig, int(ligand_charge), int(ligand_multiplicity), cutoff)
    env_pdb = out_dir / "environment.pdb"
    with open(env_pdb, "w") as fh:
        app.PDBFile.writeFile(model.topology, model.positions, fh, keepIds=True)
    atoms = list(model.topology.atoms())
    env_xyz = np.asarray(model.getPositions().value_in_unit(unit.angstrom))
    n_env = len(atoms)
    n_water = (n_env - n_protein) // 3
    env_symbols = [a.element.symbol for a in atoms]
    # QM: the species (appended last) and any side chains asked for.
    want = {r.strip().upper() for r in qm_residues}
    qm_env, qm_side_charge = [], 0
    for res in model.topology.residues():
        tag = f"{res.chain.id}:{res.name}{res.id}".upper()
        if tag in want or f"{res.name}{res.id}".upper() in want:
            side = [a.index for a in res.atoms() if a.name not in ("N", "H", "CA", "HA", "C", "O", "OXT", "H1", "H2", "H3")]
            qm_env += side
            qm_side_charge += SIDE_CHAIN_CHARGE.get(res.name, 0)
            want.discard(tag)
            want.discard(f"{res.name}{res.id}".upper())
    if want:
        raise ValueError(f"no residue {', '.join(sorted(want))} in the protein")
    sym = env_symbols + [str(s) for s in symbols]
    xyz = np.vstack([env_xyz, lig])
    qm = sorted(qm_env) + list(range(n_env, n_env + len(symbols)))
    system_charge = int(round(_ff_charge(model.topology, ff))) + int(ligand_charge)
    structure = Structure(symbols=sym, geometry=xyz * ANGSTROM_TO_BOHR, charge=system_charge,
                          multiplicity=int(ligand_multiplicity))
    frozen = ()
    if freeze_protein:
        frozen = [i for i in range(n_protein) if i not in set(qm_env)]
    region = QMMMRegion.build(structure, qm, qm_charge=int(ligand_charge) + qm_side_charge,
                              qm_multiplicity=int(ligand_multiplicity), active_radius=active_radius,
                              frozen_atoms=frozen, mm="amber", embedding="electrostatic",
                              pdb=env_pdb.name, forcefield=list(FORCEFIELD), mm_cutoff=cutoff,
                              solute_atoms=list(range(n_env, n_env + len(symbols))),
                              name=name or "in protein")
    sys_fp = out_dir / "system.xyz"
    sys_fp.write_text(structure.to_xyz())
    reg_fp = out_dir / "region.json"
    reg_fp.write_text(json.dumps(region.to_dict(), indent=1))
    return {"system": str(sys_fp), "region": str(reg_fp), "environment_pdb": str(env_pdb),
            "atoms": len(sym), "protein_atoms": n_protein, "waters": n_water, "qm_atoms": len(qm),
            "links": len(region.links), "moving": len(region.active_atoms), "frozen": len(region.frozen_atoms),
            "charge": system_charge, "qm_charge": region.qm_charge, "summary": region.summary(), "relax": relax,
            "structure": structure, "region_obj": region}


def _relax_environment(model, ff, symbols, lig_angstrom, charge: int, multiplicity: int,
                       cutoff: Optional[float]) -> dict:
    """Minimize the protein's hydrogens and the water around the species (in
    place), the protein's heavy atoms and the species held: PDBFixer's
    hydrogens and freshly placed water are far from relaxed, which would
    otherwise be paid for in every QM/MM step. The species carries GFN2-xTB
    charges (Lennard-Jones from UFF) for this."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    from mepd.engines.qmmm import _UFF_LJ

    q = np.zeros(len(symbols))
    try:
        from mepd.engines.gfnff import XTBEngine

        xtb = XTBEngine(method="gfn2")
        xtb.energy_gradient([str(s) for s in symbols], np.asarray(lig_angstrom) * ANGSTROM_TO_BOHR, charge, multiplicity)
        if xtb.last_charges is not None and len(xtb.last_charges) == len(symbols):
            q = np.asarray(xtb.last_charges, dtype=float)
    except Exception:
        pass
    kw = dict(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    if cutoff:
        kw.update(nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=float(cutoff) * unit.angstrom)
    system = ff.createSystem(model.topology, **kw)
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    n_env = system.getNumParticles()
    for s, qi in zip(symbols, q):
        system.addParticle(0.0)                               # held
        x, d = _UFF_LJ.get(str(s), (3.5, 0.1))
        nb.addParticle(float(qi), x / 2 ** (1 / 6) / 10.0, d * 4.184)
    for a in model.topology.atoms():
        if a.residue.name != "HOH" and a.element is not None and a.element.symbol != "H":
            system.setParticleMass(a.index, 0.0)              # protein heavy atoms held
    pos = np.vstack([np.asarray(model.getPositions().value_in_unit(unit.nanometer)), np.asarray(lig_angstrom) / 10.0])
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CPU"),
                         {"Threads": "1"})
    ctx.setPositions(pos * unit.nanometer)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    openmm.LocalEnergyMinimizer.minimize(ctx, 10.0, 5000)
    st = ctx.getState(getEnergy=True, getPositions=True)
    e1 = st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    new = np.asarray(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))[:n_env]
    model.positions = [openmm.Vec3(*p) for p in new] * unit.nanometer
    return {"energy_drop_kcal": float(e0 - e1), "species_charge_sum": float(q.sum())}


def _ff_charge(topology, ff) -> float:
    import openmm
    import openmm.app as app

    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff)
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    return sum(nb.getParticleParameters(i)[0].value_in_unit(openmm.unit.elementary_charge)
               for i in range(nb.getNumParticles()))


def sites_to_json(sites: list[Site]) -> list[dict]:
    return [asdict(s) for s in sites]
