"""QM/MM: a QM engine for a region of a system, embedded in a cheaper
description of the rest.

`QMMMEngine(base=<any mepd engine>, region=<mepd.qmmm.QMMMRegion>)`
computes, for the whole system x,

* **subtractive** (ONIOM) embedding, low level = GFN-FF / GFN1 / GFN2 (xtb):

      E(x) = E_QM(model) + E_low(real) - E_low(model)

  where "model" is the QM atoms capped with link hydrogens and "real" is the
  whole system. No force-field parameters are needed (GFN-FF covers almost
  every element), so any structure can be embedded as it is.

* **additive** embedding, low level = AMBER (OpenMM, from a prmtop or a PDB
  of standard residues):

      E(x) = E_QM(model) + E_MM(x; no MM terms inside the QM region)

  QM-QM bonds, angles, torsions and nonbonded pairs are switched off in the
  force field; every term with at least one MM atom stays.

Both are *mechanical* embeddings: the QM region feels its environment
through the low level's forces (sterics, dispersion, and the low level's
charges), its electrons are not polarized by it. Link-atom gradients are
spread onto the two atoms of the cut bond (see mepd.qmmm).

Geometry optimizations, TS searches and IRCs run through ASE on this
engine's own surface; wrap it in mepd.engines.frozen.FrozenAtomsEngine to
hold the environment's outer part fixed (mepd.inputs does this).

References: ONIOM, M. Svensson et al., J. Phys. Chem. 100, 19357 (1996),
doi:10.1021/jp962071j; QM/MM review, H. M. Senn, W. Thiel, Angew. Chem.
Int. Ed. 48, 1198 (2009), doi:10.1002/anie.200802019; OpenMM, P. Eastman
et al., PLoS Comput. Biol. 13, e1005659 (2017),
doi:10.1371/journal.pcbi.1005659; GFN-FF, S. Spicher, S. Grimme, Angew.
Chem. Int. Ed. 59, 15665 (2020), doi:10.1002/anie.202004239.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.engines.gfnff import XTBEngine
from mepd.engines.modified import ModifiedEngine, fresh
from mepd.errors import ElectronicStructureError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache
from mepd.qmmm import QMMMRegion

HARTREE_KJ_MOL = 2625.4996394799
BOHR_NM = 0.0529177210903


class _XTBLow:
    """Subtractive low level through the xtb program (GFN-FF, GFN1, GFN2).
    GFN-FF topologies come from the region's reference geometry (the same
    for every structure of the system)."""

    kind = "subtractive"

    def __init__(self, region: QMMMRegion, method: str, n_parallel: int):
        from mepd.engines.gfnff import GFNFFEngine

        self.region = region
        self.engine = GFNFFEngine(method=method, n_parallel=n_parallel)
        ref = region.reference_structure()
        self.ref_real = ref.to_xyz() if ref is not None else None
        self.ref_model = region.model_structure(ref).to_xyz() if ref is not None else None

    def terms(self, coords_list: list[np.ndarray], charge: int) -> list[dict]:
        r = self.region
        uhf = int(r.qm_multiplicity) - 1
        jobs = []
        for x in coords_list:
            jobs.append((list(r.symbols), x, int(charge), self.ref_real, uhf))
            jobs.append((r.model_symbols, r.model_coords(x), int(r.qm_charge), self.ref_model, uhf))
        res = self.engine.energy_gradients(jobs)
        out = []
        for k in range(len(coords_list)):
            (e_real, g_real), (e_model, g_model) = res[2 * k], res[2 * k + 1]
            out.append({"energy": e_real - e_model,
                        "gradient": np.asarray(g_real) - r.model_gradient_to_full(g_model),
                        "low_real": e_real, "low_model": e_model})
        return out


class _OpenMMLow:
    """Additive AMBER environment through OpenMM, with every force-field term
    inside the QM region switched off."""

    kind = "additive"

    def __init__(self, region: QMMMRegion, base_dir: Optional[Path] = None):
        try:
            import openmm
            import openmm.app as app
            import openmm.unit as unit
        except ImportError as exc:
            raise ImportError("an AMBER (OpenMM) QM/MM environment needs openmm (pip install openmm)") from exc
        self.region = region
        self._unit = unit
        base_dir = Path(base_dir or ".")

        def path(p):
            fp = Path(p)
            return fp if fp.is_absolute() else base_dir / fp

        if region.mm == "tip3p":
            system = tip3p_system(region, openmm)
        elif region.prmtop:
            top = app.AmberPrmtopFile(str(path(region.prmtop)))
        elif region.pdb:
            pdb = app.PDBFile(str(path(region.pdb)))
            top = app.ForceField(*region.forcefield)
            self._pdb_topology = pdb.topology
        else:
            raise ValueError("qmmm.mm = 'amber' needs `prmtop` (or `pdb` of standard residues)")
        if region.mm != "tip3p":
            kw = dict(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
            system = top.createSystem(**kw) if region.prmtop else top.createSystem(self._pdb_topology, **kw)
        if system.getNumParticles() != region.natoms:
            raise ValueError(f"the force field has {system.getNumParticles()} atoms, the system {region.natoms}")
        self.electrostatic = region.embedding == "electrostatic"
        self.charges = self._mm_charges(system, openmm)
        self.removed = self._switch_off_qm_terms(system, openmm)
        integrator = openmm.VerletIntegrator(0.001)
        # Double precision by default: single-precision forces are too noisy
        # for tight optimizations and finite-difference Hessians.
        name = str(getattr(region, "mm_platform", None) or "Reference")
        platform = openmm.Platform.getPlatformByName(name)
        props = {"Threads": "1"} if name == "CPU" else {"Precision": "double"} if name in ("CUDA", "OpenCL") else {}
        self.context = openmm.Context(system, integrator, platform, props)
        self._lock = threading.Lock()

    def _switch_off_qm_terms(self, system, openmm) -> dict:
        qm = set(self.region.qm_atoms)
        removed = {"bonds": 0, "angles": 0, "torsions": 0, "pairs": 0}
        for force in system.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                for k in range(force.getNumBonds()):
                    i, j, r0, kb = force.getBondParameters(k)
                    if i in qm and j in qm:
                        force.setBondParameters(k, i, j, r0, 0.0)
                        removed["bonds"] += 1
            elif isinstance(force, openmm.HarmonicAngleForce):
                for k in range(force.getNumAngles()):
                    i, j, l, a0, ka = force.getAngleParameters(k)
                    if {i, j, l} <= qm:
                        force.setAngleParameters(k, i, j, l, a0, 0.0)
                        removed["angles"] += 1
            elif isinstance(force, openmm.PeriodicTorsionForce):
                for k in range(force.getNumTorsions()):
                    i, j, l, m, n, ph, kt = force.getTorsionParameters(k)
                    if {i, j, l, m} <= qm:
                        force.setTorsionParameters(k, i, j, l, m, n, ph, 0.0)
                        removed["torsions"] += 1
            elif isinstance(force, openmm.NonbondedForce):
                q = sorted(qm)
                for a in range(len(q)):
                    for b in range(a + 1, len(q)):
                        force.addException(q[a], q[b], 0.0, 0.1, 0.0, True)
                        removed["pairs"] += 1
                if self.electrostatic:
                    # QM-MM electrostatics come from the QM calculation in the
                    # field of the MM charges: none at the force-field level
                    # (QM atoms keep their Lennard-Jones terms).
                    for i in q:
                        _, sig, eps = force.getParticleParameters(i)
                        force.setParticleParameters(i, 0.0, sig, eps)
                    for k in range(force.getNumExceptions()):
                        i, j, qq, sig, eps = force.getExceptionParameters(k)
                        if i in qm or j in qm:
                            force.setExceptionParameters(k, i, j, 0.0, sig, eps)
        return removed

    def _mm_charges(self, system, openmm) -> np.ndarray:
        """Charge of every atom (e), before any is switched off."""
        out = np.zeros(system.getNumParticles())
        for force in system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for i in range(force.getNumParticles()):
                    out[i] = force.getParticleParameters(i)[0].value_in_unit(self._unit.elementary_charge)
        return out

    def point_charges(self) -> tuple[np.ndarray, np.ndarray]:
        """(atom indices, charges) the QM region feels: every MM atom with a
        charge, except those bonded to the QM region (their charge would sit
        on top of a link atom; the "Z1" scheme)."""
        qm = set(self.region.qm_atoms)
        z1 = {m for _, m in self.region.links}
        idx = np.array([i for i in range(self.region.natoms) if i not in qm and i not in z1
                        and abs(self.charges[i]) > 1e-12], dtype=int)
        return idx, self.charges[idx]

    def terms(self, coords_list: list[np.ndarray], charge: int) -> list[dict]:
        u = self._unit
        out = []
        with self._lock:
            for x in coords_list:
                self.context.setPositions(np.asarray(x, dtype=float) * BOHR_NM * u.nanometer)
                st = self.context.getState(getEnergy=True, getForces=True)
                e = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole) / HARTREE_KJ_MOL
                f = np.asarray(st.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer))
                out.append({"energy": e, "gradient": -f * BOHR_NM / HARTREE_KJ_MOL, "low_real": e, "low_model": 0.0})
        return out


# UFF Lennard-Jones (x_i in Angstrom, D_i in kcal/mol; A. K. Rappe et al.,
# J. Am. Chem. Soc. 114, 10024 (1992), doi:10.1021/ja00051a040) for the QM
# atoms' contacts with a TIP3P environment.
_UFF_LJ = {"H": (2.886, 0.044), "C": (3.851, 0.105), "N": (3.660, 0.069), "O": (3.500, 0.060),
           "F": (3.364, 0.050), "P": (4.147, 0.305), "S": (4.035, 0.274), "Cl": (3.947, 0.227),
           "Br": (4.189, 0.251), "I": (4.500, 0.339), "B": (4.083, 0.180), "Si": (4.295, 0.402),
           "Li": (2.451, 0.025), "Na": (2.983, 0.030), "K": (3.812, 0.035), "Mg": (3.021, 0.111),
           "Ca": (3.399, 0.238), "Zn": (2.763, 0.124), "Fe": (2.912, 0.013), "Cu": (3.495, 0.005)}


def tip3p_system(region: QMMMRegion, openmm):
    """An OpenMM System for a QM region in water: every environment molecule
    must be a water (TIP3P: charges -0.834/+0.417, flexible O-H bonds and
    H-O-H angle as in AMBER's tip3p.xml without rigid water, hydrogens with
    CHARMM's small Lennard-Jones term); the QM atoms get UFF Lennard-Jones
    only. W. L. Jorgensen et al., J. Chem. Phys. 79, 926 (1983),
    doi:10.1063/1.445869; CHARMM TIP3P: A. D. MacKerell Jr. et al., J. Phys.
    Chem. B 102, 3586 (1998), doi:10.1021/jp973084f."""
    from mepd.qmmm import bonds, molecules

    import openmm.unit as u

    ref = region.reference_structure()
    xyz = np.asarray(ref.geometry) / 1.8897259886
    symbols = region.symbols
    qm = set(region.qm_atoms)
    bl = bonds(symbols, xyz)
    system = openmm.System()
    for s in symbols:
        system.addParticle(openmm.app.element.Element.getBySymbol(s).mass)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    hb, ha = openmm.HarmonicBondForce(), openmm.HarmonicAngleForce()
    for i, s in enumerate(symbols):
        if i in qm:
            x, d = _UFF_LJ.get(s, (3.5, 0.1))
            nb.addParticle(0.0, x / 2 ** (1 / 6) / 10.0, d * 4.184)
        elif s == "O":
            nb.addParticle(-0.834, 0.315061, 0.636386)
        else:
            # CHARMM's TIP3P hydrogen Lennard-Jones (Rmin/2 0.2245 A, eps 0.046
            # kcal/mol): without it nothing keeps a water H off a negative QM atom.
            nb.addParticle(0.417, 0.04000, 0.046 * 4.184)
    for mol in molecules(len(symbols), bl):
        if qm.intersection(mol):
            continue
        if sorted(symbols[i] for i in mol) != ["H", "H", "O"]:
            raise ValueError("mm = 'tip3p' needs every environment molecule to be a water; this one is "
                             + "".join(symbols[i] for i in mol))
        o = next(i for i in mol if symbols[i] == "O")
        h1, h2 = [i for i in mol if symbols[i] == "H"]
        hb.addBond(o, h1, 0.09572, 462750.4)
        hb.addBond(o, h2, 0.09572, 462750.4)
        ha.addAngle(h1, o, h2, 104.52 * np.pi / 180, 836.8)
        for a, b in ((o, h1), (o, h2), (h1, h2)):
            nb.addException(a, b, 0.0, 0.1, 0.0)
    for f in (nb, hb, ha):
        system.addForce(f)
    del u
    return system


def make_low(region: QMMMRegion, n_parallel: int = 4, base_dir: Optional[Path] = None):
    if region.mm in ("gfnff", "gfn2", "gfn1"):
        return _XTBLow(region, region.mm, n_parallel)
    if region.mm in ("amber", "tip3p"):
        return _OpenMMLow(region, base_dir)
    raise ValueError(f"QMMMEngine has no low level {region.mm!r} (TeraChem QM/MM is its own engine)")


@dataclass
class QMMMEngine(ModifiedEngine):
    """`base`: the QM engine. `region`: the partition. `n_parallel`: low-level
    processes at once."""

    base: Engine
    region: QMMMRegion
    n_parallel: int = 4
    base_dir: Optional[str] = None     # where relative prmtop/pdb paths live
    _low: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.region.embedding == "electrostatic" and not getattr(self.base, "supports_point_charges", False):
            raise ValueError(f"electrostatic embedding needs a QM engine that takes point charges (engine_name = "
                             f"\"xtb\" or \"psi4\"); {type(self.base).__name__} does not")
        self._low = make_low(self.region, self.n_parallel, Path(self.base_dir) if self.base_dir else None)
        self._inherit()

    @property
    def scheme(self) -> str:
        return self._low.kind

    def _check(self, node) -> None:
        if len(node.symbols) != self.region.natoms:
            raise ElectronicStructureError(
                msg=f"QM/MM region is for {self.region.natoms} atoms, this structure has {len(node.symbols)}")

    def evaluate(self, nodes: list[StructureNode]) -> list[dict]:
        """Energy, gradient and their parts for each node (nothing cached)."""
        r = self.region
        for n in nodes:
            self._check(n)
        low = self._low.terms([np.asarray(n.coords, dtype=float) for n in nodes], int(nodes[0].structure.charge))
        if getattr(self._low, "electrostatic", False):
            return self._evaluate_electrostatic(nodes, low)
        models = [StructureNode(structure=r.model_structure(n.structure), has_molecular_graph=False) for n in nodes]
        g_qm = np.asarray(self.base.compute_gradients(models), dtype=float)
        e_qm = np.array([m.energy for m in models], dtype=float)
        out = []
        for eq, gq, lw in zip(e_qm, g_qm, low):
            out.append({"energy": float(eq + lw["energy"]),
                        "gradient": r.model_gradient_to_full(gq) + lw["gradient"],
                        "qm": float(eq), "environment": float(lw["energy"]),
                        "low_real": float(lw["low_real"]), "low_model": float(lw["low_model"])})
        return out

    def _evaluate_electrostatic(self, nodes, low) -> list[dict]:
        """Electrostatic embedding: the QM region computed in the field of the
        environment's point charges (which then feel the QM density and
        nuclei); the force field has no QM-MM electrostatics."""
        r = self.region
        idx, q = self._low.point_charges()
        kw = {"elements": [r.symbols[i] for i in idx]} if isinstance(self.base, XTBEngine) else {}

        def one(n, lw):
            x = np.asarray(n.coords, dtype=float)
            model = r.model_structure(n.structure)
            e_qm, g_model, g_pc = self.base.energy_gradient(
                list(model.symbols), np.asarray(model.geometry), int(model.charge), int(model.multiplicity),
                point_charges=list(zip(q.tolist(), x[idx].tolist())), **kw)
            grad = r.model_gradient_to_full(g_model) + lw["gradient"]
            if g_pc is not None and len(idx):
                grad[idx] += np.asarray(g_pc).reshape(-1, 3)
            return {"energy": float(e_qm + lw["energy"]), "gradient": grad, "qm": float(e_qm),
                    "environment": float(lw["energy"]), "low_real": float(lw["low_real"]),
                    "low_model": float(lw["low_model"]),
                    "qm_charges": getattr(self.base, "last_charges", None)}

        # Psi4 runs one at a time (one worker); xtb as many as it may.
        width = max(1, min(len(nodes), int(getattr(self.base, "n_parallel", 1) or 1)))
        if width == 1:
            return [one(n, lw) for n, lw in zip(nodes, low)]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=width) as pool:
            return list(pool.map(one, nodes, low))

    def _run(self, chain: Union[Chain, List]) -> list[StructureNode]:
        nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [n for n in nodes if n._cached_energy is None or n._cached_gradient is None]
        if todo:
            res = self.evaluate([fresh(n) for n in todo])
            update_node_cache(node_list=todo, results=[
                FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                    {"energy": r["energy"], "gradient": r["gradient"]})}) for r in res])
        return nodes

    def decompose(self, structures) -> list[dict]:
        """Energy parts (Hartree) of each structure, for checks and plots."""
        return [{k: v for k, v in r.items() if k != "gradient"} | {"max_force_qm": float(np.abs(
            r["gradient"][self.region.qm_atoms]).max())} for r in self.evaluate(
            [StructureNode(structure=s, has_molecular_graph=False) for s in structures])]

    def prepare_node_for_comparison(self, node: StructureNode) -> StructureNode:
        """Graphs (species identity, elementary-step checks) of the QM atoms only."""
        return qm_comparison_node(node, self.region)


def qm_comparison_node(node: StructureNode, region: QMMMRegion) -> StructureNode:
    if getattr(node, "graph_atom_indices_source", None) == "qmmm_qm_atoms" or len(node.symbols) != region.natoms:
        return node
    from mepd.qcdata_structure_helpers import structure_to_molecule

    try:
        graph, has_graph = structure_to_molecule(region.qm_structure(node.structure)), True
    except Exception:
        graph, has_graph = None, False
    payload = node.__dict__.copy()
    payload.update({"has_molecular_graph": has_graph, "graph": graph,
                    "comparison_atom_indices": list(region.qm_atoms), "disable_smiles": True,
                    "graph_atom_indices_source": "qmmm_qm_atoms",
                    "graph_subset_atom_count": len(region.qm_atoms),
                    "graph_total_atom_count": region.natoms})
    return StructureNode(**payload)
