"""A species in a protein as a QM/MM system (mepd/qmmm_protein.py): the
protein prepared, the species docked rigid, the system built at a site and
computed with the protein's charges acting on the QM region."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from qcconst.constants import ANGSTROM_TO_BOHR

pytest.importorskip("openmm")
if importlib.util.find_spec("pdbfixer") is None:
    pytest.skip("needs pdbfixer", allow_module_level=True)

FRAGMENT = Path(__file__).parent / "data" / "qmmm_protein" / "fragment.pdb"   # 2CHT chain A, residues 2-12
# Acetate, a little away from the fragment's Arg4/Arg7.
ACETATE = (["C", "C", "O", "O", "H", "H", "H"],
           np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.15, 1.07, 0.0], [2.12, -1.10, 0.0],
                     [-0.37, 1.02, 0.0], [-0.37, -0.51, 0.89], [-0.37, -0.51, -0.89]]))


@pytest.fixture(scope="module")
def protein(tmp_path_factory):
    from mepd.qmmm_protein import prepare_protein

    out = tmp_path_factory.mktemp("protein") / "protein.pdb"
    rep = prepare_protein(FRAGMENT, out)
    assert rep["atoms"] == 174 and rep["charge"] == 2 and rep["chains"] == ["A"]
    return out


def _site(protein):
    """The acetate placed 4 Å off the fragment's surface, near its arginines."""
    import openmm.app as app

    pdb = app.PDBFile(str(protein))
    xyz = np.asarray(pdb.getPositions(asNumpy=True)._value) * 10.0
    cz = [a.index for a in pdb.topology.atoms() if a.name == "CZ"]
    target = xyz[cz].mean(0)
    away = target - xyz.mean(0)
    sym, x = ACETATE
    return sym, x - x.mean(0) + target + 4.0 * away / np.linalg.norm(away)


def test_species_in_protein_has_exact_qmmm_gradients(protein, tmp_path):
    """The species is appended after the force-field atoms (UFF Lennard-Jones
    only), the protein's and water's charges act on it through xtb's
    embedding: QM/MM gradients on QM and protein atoms are exact."""
    from qcdata import Structure

    from mepd.engines.gfnff import XTBEngine
    from mepd.engines.qmmm import QMMMEngine
    from mepd.nodes.node import StructureNode
    from mepd.qmmm import QMMMRegion
    from mepd.qmmm_protein import build_site_system

    sym, x = _site(protein)
    rep = build_site_system(protein, sym, x, tmp_path, ligand_charge=-1, water_shell=5.0, active_radius=5.0)
    assert rep["charge"] == 1 and rep["qm_charge"] == -1 and rep["protein_atoms"] == 174
    region = QMMMRegion.open(tmp_path / "region.json")
    assert region.qm_atoms == list(range(rep["atoms"] - 7, rep["atoms"])) and region.embedding == "electrostatic"
    s = Structure.open(str(tmp_path / "system.xyz"))
    eng = QMMMEngine(base=XTBEngine(method="gfn2"), region=region, base_dir=str(tmp_path))
    node = lambda y: StructureNode(structure=s.model_copy(update={"geometry": y}), has_molecular_graph=False)
    x0 = np.asarray(s.geometry, dtype=float)
    out = eng.evaluate([node(x0)])[0]
    h = 1e-3
    for a in (region.qm_atoms[2], region.active_mm_atoms[0]):     # an acetate O, a protein/water atom
        ys = []
        for sign in (1, -1):
            y = x0.copy()
            y[a, 0] += sign * h
            ys.append(node(y))
        ep, em = (o["energy"] for o in eng.evaluate(ys))
        assert abs((ep - em) / (2 * h) - out["gradient"][a, 0]) < 2e-5


def test_reduced_force_field_gives_the_moving_atoms_their_full_forces(protein, tmp_path):
    """Microiterations relax the environment on the atoms near the moving ones
    only (a cut-off force field): the same forces on the moving atoms."""
    from qcdata import Structure

    from mepd.engines.gfnff import XTBEngine
    from mepd.engines.microiter import MicroIterations
    from mepd.engines.qmmm import QMMMEngine
    from mepd.qmmm import QMMMRegion
    from mepd.qmmm_protein import build_site_system

    sym, x = _site(protein)
    build_site_system(protein, sym, x, tmp_path, ligand_charge=-1, water_shell=5.0, active_radius=4.0, cutoff=6.0)
    region = QMMMRegion.open(tmp_path / "region.json")
    eng = QMMMEngine(base=XTBEngine(method="gfn2"), region=region, base_dir=str(tmp_path))
    micro = MicroIterations(eng)
    reduced = eng._low.reduced(sorted(set(micro.inner) | set(micro.outer)))
    assert reduced is not None and len(reduced.keep) < region.natoms
    x0 = np.asarray(Structure.open(str(tmp_path / "system.xyz")).geometry, dtype=float)
    _, g = reduced.terms(x0)
    full = eng._low.terms([x0], 0)[0]["gradient"]
    assert np.abs(g[micro.inner] - full[micro.inner]).max() < 1e-5


def test_rigid_docking_keeps_the_species_geometry_and_names_the_site(protein, tmp_path):
    if importlib.util.find_spec("vina") is None:
        pytest.skip("needs vina")
    from mepd.qmmm_protein import find_sites

    sym, x = _site(protein)
    sites = find_sites(protein, sym, x, tmp_path, centers=np.array([x.mean(0)]), workers=1, exhaustiveness=4)
    assert sites and sites[0].score < 0
    placed = np.asarray(sites[0].coords)
    d = lambda y: np.linalg.norm(y[:, None] - y[None], axis=2)
    assert np.abs(d(placed) - d(x)).max() < 1e-3          # rigid: every interatomic distance kept
    assert sites[0].residues and all(r.startswith("A:") for r in sites[0].residues)
