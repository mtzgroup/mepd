"""QM/MM (mepd.qmmm, engines/qmmm.py, engines/frozen.py) and frozen atoms."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from qcconst.constants import ANGSTROM_TO_BOHR
from qcdata import Structure

from mepd.nodes.node import StructureNode
from mepd.qmmm import QMMMRegion, active_shell, bonds, diagnose, format_indices, parse_indices

DATA = Path(__file__).parent / "data" / "qmmm_amber"

# 1-propanol: C0 C1 C2 O3, H4-6 on C0, H7-8 on C1, H9-10 on C2, H11 on O3.
PROPANOL = """12

C  -1.9031  0.2557  0.0177
C  -0.5130 -0.3528 -0.0410
C   0.5631  0.7155 -0.1206
O   1.8380  0.1186 -0.1778
H  -1.9566  0.9583  0.8556
H  -2.6513 -0.5304  0.1557
H  -2.1508  0.8031 -0.8975
H  -0.2985 -0.9198  0.8723
H  -0.4646 -1.0520 -0.8826
H   0.3977  1.3426 -1.0045
H   0.4986  1.3546  0.7656
H   2.4995  0.8061 -0.2236
"""
QM_PROPANOL = [2, 3, 9, 10, 11]


def _xtb():
    from mepd.programs import xtb_executable

    exe = xtb_executable(download=False)
    if not exe:
        pytest.skip("needs the xtb program")
    return exe


def _propanol():
    return Structure.from_xyz(PROPANOL)


def test_indices_round_trip():
    assert parse_indices("0-3 7, 9") == [0, 1, 2, 3, 7, 9]
    assert format_indices([9, 0, 1, 2, 3, 7]) == "0-3 7 9"
    assert parse_indices([3, 1, 1]) == [1, 3]


def test_region_cuts_the_c_c_bond_and_places_a_link_hydrogen():
    s = _propanol()
    r = QMMMRegion.build(s, QM_PROPANOL, active_radius=None)
    assert r.links == [(2, 1)]
    assert r.model_symbols == ["C", "O", "H", "H", "H", "H"]
    x = np.asarray(s.geometry)
    link = r.link_positions(x)[0]
    d_ch = np.linalg.norm(link - x[2]) / ANGSTROM_TO_BOHR
    assert 1.0 < d_ch < 1.2                                    # a C-H bond length along the cut C-C bond
    assert r.check(s) == []
    # The model gradient goes back onto the cut bond's two atoms, conserving the total force.
    g = np.zeros((len(r.model_symbols), 3))
    g[-1] = [1.0, 2.0, 3.0]
    full = r.model_gradient_to_full(g)
    assert np.allclose(full.sum(axis=0), [1.0, 2.0, 3.0])
    assert np.abs(full[[i for i in range(12) if i not in (1, 2)]]).max() == 0


def test_region_problems_are_reported():
    s = _propanol()
    r = QMMMRegion.build(s, [2, 3, 9, 10], active_radius=None)   # cuts the O-H: the H stays MM
    assert any("hydrogen 11" in p for p in r.check(s))
    odd = QMMMRegion.build(s, QM_PROPANOL, qm_charge=1, active_radius=None)
    assert any("electrons" in p for p in odd.check(s))


def test_active_shell_takes_small_molecules_whole():
    water = "3\n\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"
    w = Structure.from_xyz(water)
    xyz = np.asarray(w.geometry) / ANGSTROM_TO_BOHR
    far = xyz + [3.5, 0, 0]       # its O within 3.6 Å of the QM O, its H's beyond
    symbols = ["O", "H", "H"] * 2
    shell = active_shell(symbols, np.vstack([xyz, far]), [0, 1, 2], 3.6)
    assert shell == [0, 1, 2, 3, 4, 5]
    assert bonds(symbols, np.vstack([xyz, far])) == [(0, 1), (0, 2), (3, 4), (3, 5)]


def test_region_json_round_trip(tmp_path):
    r = QMMMRegion.build(_propanol(), QM_PROPANOL, active_radius=None, frozen_atoms="4-6", name="propanol")
    r.save(tmp_path / "r.json")
    back = QMMMRegion.open(tmp_path / "r.json")
    assert back.to_dict() == r.to_dict()
    assert back.signature() == r.signature()
    assert set(back.frozen_atoms).isdisjoint(back.qm_atoms) and back.frozen_atoms


def test_subtractive_qmmm_matches_xtb_oniom_and_its_gradient(tmp_path, monkeypatch):
    exe = _xtb()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    from mepd.engines.gfnff import GFNFFEngine
    from mepd.engines.qmmm import QMMMEngine

    s = _propanol()
    r = QMMMRegion.build(s, QM_PROPANOL, active_radius=None)
    eng = QMMMEngine(base=GFNFFEngine(method="gfn2", executable=exe, n_parallel=1), region=r, n_parallel=1)
    res = eng.evaluate([StructureNode(structure=s, has_molecular_graph=False)])[0]
    s.save(str(tmp_path / "p.xyz"))
    out = subprocess.run([exe, "p.xyz", "--oniom", "gfn2:gfnff", ",".join(str(i + 1) for i in QM_PROPANOL)],
                         cwd=tmp_path, capture_output=True, text=True).stdout
    ref = float(next(ln for ln in out.splitlines() if "ONIOM TOTAL ENERGY" in ln).split()[4])
    assert abs(res["energy"] - ref) < 1e-4              # link placement differs slightly from xtb's
    x0 = np.asarray(s.geometry, dtype=float)
    h = 1e-4
    for a, c in [(2, 0), (1, 1), (0, 2)]:               # QM atom, cut-bond MM atom, plain MM atom
        e = []
        for sign in (1, -1):
            x = x0.copy()
            x[a, c] += sign * h
            e.append(eng.evaluate([StructureNode(structure=s.model_copy(update={"geometry": x}),
                                                 has_molecular_graph=False)])[0]["energy"])
        assert abs((e[0] - e[1]) / (2 * h) - res["gradient"][a, c]) < 2e-5


def _amber_region():
    pytest.importorskip("openmm")
    from mepd.qmmm_build import from_terachem

    return from_terachem(DATA / "tc.in")


def test_terachem_input_converts_and_runs_with_openmm():
    system, region, qm = _amber_region()
    assert qm["method"] == "b3lyp" and region.mm == "amber"
    assert region.qm_atoms == [0, 1, 2, 3, 4, 5]
    assert region.frozen_atoms == [12, 13, 14]           # TeraChem's $constraints, 1-based there
    assert len(system.symbols) == 15


def test_additive_openmm_environment_switches_off_qm_terms_and_has_exact_gradients():
    _, region, _ = _amber_region()
    from mepd.engines.qmmm import _OpenMMLow

    low = _OpenMMLow(region)
    assert low.removed == {"bonds": 4, "angles": 2, "torsions": 0, "pairs": 15}
    x0 = np.asarray(region.reference_structure().geometry, dtype=float)
    t = low.terms([x0], 0)[0]
    h = 1e-4
    for a, c in [(0, 0), (7, 1)]:
        e = []
        for sign in (1, -1):
            x = x0.copy()
            x[a, c] += sign * h
            e.append(low.terms([x], 0)[0]["energy"])
        assert abs((e[0] - e[1]) / (2 * h) - t["gradient"][a, c]) < 1e-6
    # Every atom QM: nothing is left for the force field.
    whole = QMMMRegion.from_dict({**region.to_dict(), "qm_atoms": list(range(15)), "frozen_atoms": [],
                                  "links": []})
    assert abs(_OpenMMLow(whole).terms([x0], 0)[0]["energy"]) < 1e-12


def test_terachem_output_parsing():
    from mepd.engines.terachem_qmmm import (TeraChemQMMMEngine, parse_tcin, rst7_coords, structure_to_rst7)

    p = parse_tcin((DATA / "tc.in").read_text())
    assert p["charge"] == 0 and p["frozen_atom_indices"] == [12, 13, 14] and "scrdir" not in p["keywords"]
    x = np.arange(15.0).reshape(5, 3) * ANGSTROM_TO_BOHR
    assert np.allclose(rst7_coords(structure_to_rst7(x)), np.arange(15.0).reshape(5, 3))
    eng = TeraChemQMMMEngine(tcin=(DATA / "tc.in").read_text(), prmtop="", qm_atoms=[1, 2])
    assert "$constraints" not in eng.tcin and "run gradient" in eng.tcin
    stdout = "\n".join([
        "Number of link atoms: 1 link atoms", "dE/dX dE/dY dE/dZ",
        "0.1 0.2 0.3", "0.4 0.5 0.6", "9.0 9.0 9.0", "---------",
        "MM / Point charge part", "1.0 1.1 1.2", "2.0 2.1 2.2", "3.0 3.1 3.2", "---------",
        "FINAL ENERGY: -76.123 a.u."])
    e, g = eng.energy_gradient_from_stdout(stdout, 5)
    assert e == -76.123
    assert np.allclose(g[[1, 2]], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])     # link atom's row dropped
    assert np.allclose(g[[0, 3, 4]], [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]])


# ------------------------------------------------------------------ frozen
def _lj_engine():
    from ase.calculators.lj import LennardJones

    from mepd.engines.ase import ASEEngine

    return ASEEngine(calculator=LennardJones(sigma=3.4, epsilon=0.0104, rc=10.0), geometry_optimizer="LBFGS")


def _argon(n=5, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=2.2, size=(n, 3))
    return Structure(symbols=["Ar"] * n, geometry=x * ANGSTROM_TO_BOHR)


def test_frozen_atoms_engine_holds_atoms_in_optimizations_and_takes_partial_hessians():
    from mepd.engines.frozen import FrozenAtomsEngine

    eng = FrozenAtomsEngine(base=_lj_engine(), frozen=[0, 1])
    s = _argon()
    node = StructureNode(structure=s, has_molecular_graph=False)
    g = eng.compute_gradients([node])[0]
    assert np.all(g[[0, 1]] == 0) and np.abs(g[2:]).max() > 0
    traj = eng.compute_geometry_optimization(node, keywords={"fmax": 1e-3, "maxiter": 300})
    x0, x1 = np.asarray(s.geometry), np.asarray(traj[-1].coords)
    assert np.abs(x1[[0, 1]] - x0[[0, 1]]).max() < 1e-10
    assert np.abs(x1[2:] - x0[2:]).max() > 1e-3
    res = eng._compute_hessian_result(traj[-1])
    assert len(res.results.freqs_wavenumber) == 9          # 3 moving atoms: no rigid-body modes to drop
    assert np.asarray(res.results.hessian).shape == (15, 15)
    assert min(res.results.freqs_wavenumber) > -1          # a minimum


def test_paths_hold_frozen_atoms(monkeypatch):
    from mepd import interpolation
    from mepd.chainhelpers import run_geodesic
    from mepd.helper_functions import project_rigid_body_forces

    a = _argon()
    xb = np.asarray(a.geometry).copy()
    xb[3:] += 1.5
    b = a.model_copy(update={"geometry": xb})
    monkeypatch.setattr(interpolation, "_GLOBAL_FROZEN", None)
    interpolation.set_global_frozen_atoms([0, 1, 2], 5)
    chain = run_geodesic([StructureNode(structure=a, has_molecular_graph=False),
                          StructureNode(structure=b, has_molecular_graph=False)], nimages=6)
    for node in chain:
        assert np.abs(np.asarray(node.coords)[:3] - np.asarray(a.geometry)[:3]).max() < 1e-9
    f = project_rigid_body_forces(np.asarray(a.geometry), np.ones((5, 3)))
    assert np.all(f[:3] == 0) and np.all(f[3:] == 1)
    interpolation.set_global_frozen_atoms([])


def test_neb_runs_with_frozen_atoms(monkeypatch):
    from mepd import interpolation
    from mepd.chain import Chain
    from mepd.engines.frozen import FrozenAtomsEngine
    from mepd.inputs import ChainInputs, NEBInputs
    from mepd.neb import NEB
    from mepd.optimizers.cg import ConjugateGradient

    monkeypatch.setattr(interpolation, "_GLOBAL_FROZEN", None)
    a = _argon(4, seed=3)
    xb = np.asarray(a.geometry).copy()
    xb[3] += [2.0, 0.0, 0.0]
    b = a.model_copy(update={"geometry": xb})
    nodes = [StructureNode(structure=a.model_copy(update={"geometry": np.asarray(a.geometry) + t * (xb - np.asarray(a.geometry))}),
                           has_molecular_graph=False) for t in np.linspace(0, 1, 5)]
    chain = Chain.model_validate({"nodes": nodes, "parameters": ChainInputs(frozen_atom_indices="0 1")})
    neb = NEB(initial_chain=chain, engine=FrozenAtomsEngine(base=_lj_engine(), frozen=[0, 1]),
              parameters=NEBInputs(max_steps=5, v=0, do_elem_step_checks=False), optimizer=ConjugateGradient())
    try:
        neb.optimize_chain()
    except Exception as exc:     # not converging in 5 steps is fine; crashing is not
        assert "converge" in type(exc).__name__.lower() or "converge" in str(exc).lower(), exc
    for c in neb.chain_trajectory:
        for node in c:
            assert np.abs(np.asarray(node.coords)[:2] - np.asarray(a.geometry)[:2]).max() < 1e-10


# --------------------------------------------------------------- profile
def test_profile_qmmm_table_embeds_the_engine_and_freezes_the_environment(tmp_path, monkeypatch):
    _xtb()
    from mepd.engines.frozen import FrozenAtomsEngine
    from mepd.engines.qmmm import QMMMEngine
    from mepd.inputs import RunInputs

    r = QMMMRegion.build(_propanol(), QM_PROPANOL, active_radius=None, frozen_atoms=[4, 5, 6])
    r.save(tmp_path / "region.json")
    (tmp_path / "p.toml").write_text('engine_name = "ase"\n[ase_engine_kwds]\ncalculator = "ase.calculators.emt:EMT"\n'
                                     '[qmmm]\nfile = "region.json"\n')
    try:
        ri = RunInputs.open(tmp_path / "p.toml")
        assert isinstance(ri.engine, FrozenAtomsEngine) and isinstance(ri.engine.base, QMMMEngine)
        assert ri.engine.frozen == r.frozen_atoms == list(ri.chain_inputs.frozen_atom_indices)
        assert ri.engine.region is ri.engine.base.region        # reachable through the wrapper
        node = StructureNode(structure=_propanol())
        assert node.comparison_atom_indices == QM_PROPANOL      # graphs of the QM atoms only
        assert len(node.graph.nodes) == len(QM_PROPANOL)
    finally:
        StructureNode.set_global_graph_atoms(None)
        from mepd.interpolation import set_global_frozen_atoms

        set_global_frozen_atoms([])


def test_diagnose_flags_frozen_drift_and_mm_bond_changes():
    s = _propanol()
    r = QMMMRegion.build(s, QM_PROPANOL, active_radius=None, frozen_atoms=[4, 5, 6])
    x = np.asarray(s.geometry).copy()
    x[4] += 0.5 * ANGSTROM_TO_BOHR                # a frozen H moves...
    x[0] += [3.0 * ANGSTROM_TO_BOHR, 0, 0]        # ...and C0 leaves C1 (an MM bond breaks)
    rep = diagnose(r, [s, s.model_copy(update={"geometry": x})])
    assert rep["worst"]["frozen"] > 0.4
    assert rep["worst"]["mm_changes"] >= 1
    assert len(rep["warnings"]) >= 2
    json.dumps(rep)


def test_expansion_view_works_on_the_capped_qm_region():
    from mepd.discovery.network_expansion import Proposal, _View

    s = _propanol()
    r = QMMMRegion.build(s, QM_PROPANOL, active_radius=None)
    view = _View(r)
    node = StructureNode(structure=s, has_molecular_graph=False)
    assert view.symbols(node) == ["C", "O", "H", "H", "H", "H"]
    assert (0, 5) in view.edges(node)              # the C-link H bond is a real bond of the model
    keep = view.keep([Proposal(0, ((0, 5),), (), "x"), Proposal(0, ((1, 4),), (), "y")])
    assert [p.smiles for p in keep] == ["y"]       # link hydrogens never take part
    guess = view.coords(node) + 0.1
    full = view.full_structure(node, guess)
    moved = np.linalg.norm(np.asarray(full.geometry) - np.asarray(s.geometry), axis=1)
    assert set(np.nonzero(moved > 1e-9)[0]) == set(QM_PROPANOL)


def test_embed_puts_another_geometry_of_the_solute_into_the_system():
    from mepd.qmmm_build import embed

    s = _propanol()
    region = QMMMRegion.build(s, QM_PROPANOL, active_radius=None, solute_atoms=QM_PROPANOL)
    x = np.asarray(s.geometry)
    solute = Structure(symbols=[s.symbols[i] for i in QM_PROPANOL], geometry=x[QM_PROPANOL] + 5.0)  # moved away
    placed, rep = embed(solute, s, region)
    y = np.asarray(placed.geometry)
    rest = [i for i in range(12) if i not in QM_PROPANOL]
    assert np.allclose(y[rest], x[rest])                         # the environment stays exactly as it was
    assert rep["rmsd_from_solute"] < 0.3                         # aligned back onto the solute's place
    with pytest.raises(ValueError, match="same order"):
        embed(Structure(symbols=["O", "C", "H", "H", "H"], geometry=x[QM_PROPANOL]), s, region)


def test_reaction_maps_an_end_given_in_another_atom_order():
    pytest.importorskip("slapmapper")
    from mepd.cli_qmmm import _end_onto_start, _reorder

    a = _propanol()
    perm = [0, 1, 3, 2] + list(range(4, 12))          # the end lists O before C2
    b = _reorder(a, perm)
    order = _end_onto_start(a, b)
    assert order is not None
    back = _reorder(b, order)
    assert list(back.symbols) == list(a.symbols)
    assert np.allclose(np.asarray(back.geometry)[[0, 1, 2, 3]], np.asarray(a.geometry)[[0, 1, 2, 3]])
    assert _end_onto_start(a, a) == list(range(12)) or _end_onto_start(a, a) is not None


WATERS_AROUND_METHANOL = None


def _cluster():
    from test_qmmm_web import SYSTEM

    return Structure.from_xyz(SYSTEM)


def test_tip3p_goes_with_electrostatic_embedding():
    s = _cluster()
    r = QMMMRegion.build(s, "0-5", active_radius=None, mm="tip3p")
    assert r.embedding == "electrostatic"
    with pytest.raises(ValueError, match="electrostatic"):
        QMMMRegion.build(s, "0-5", active_radius=None, mm="tip3p", embedding="mechanical")
    with pytest.raises(ValueError, match="fixed environment charges"):
        QMMMRegion.build(s, "0-5", active_radius=None, mm="gfnff", embedding="electrostatic")
    assert r.signature() != QMMMRegion.build(s, "0-5", active_radius=None, mm="gfnff").signature()


def test_electrostatic_embedding_needs_a_point_charge_engine():
    pytest.importorskip("openmm")
    from mepd.engines.gfnff import GFNFFEngine
    from mepd.engines.qmmm import QMMMEngine

    r = QMMMRegion.build(_cluster(), "0-5", active_radius=None, mm="tip3p")
    with pytest.raises(ValueError, match="point charges"):
        QMMMEngine(base=GFNFFEngine(method="gfn2", executable="xtb"), region=r)


def test_tip3p_environment_has_no_qm_charges_and_exact_gradients():
    pytest.importorskip("openmm")
    import openmm

    from mepd.engines.qmmm import _OpenMMLow

    s = _cluster()
    r = QMMMRegion.build(s, "0-5", active_radius=None, mm="tip3p")
    low = _OpenMMLow(r)
    idx, q = low.point_charges()
    assert set(idx) == set(range(6, 15)) and abs(q.sum()) < 1e-9          # three neutral waters
    x0 = np.asarray(s.geometry, dtype=float)
    t = low.terms([x0], 0)[0]
    h = 1e-4
    for a, c in [(1, 0), (6, 1)]:
        e = []
        for sign in (1, -1):
            x = x0.copy()
            x[a, c] += sign * h
            e.append(low.terms([x], 0)[0]["energy"])
        assert abs((e[0] - e[1]) / (2 * h) - t["gradient"][a, c]) < 1e-6
    del openmm


def test_electrostatic_embedding_gradients_with_psi4():
    from mepd.engines.psi4 import psi4_python

    if not psi4_python():
        pytest.skip("needs Psi4")
    pytest.importorskip("openmm")
    from mepd.engines.psi4 import Psi4Engine
    from mepd.engines.qmmm import QMMMEngine
    from mepd.nodes.node import StructureNode

    s = _cluster()
    r = QMMMRegion.build(s, "0-5", active_radius=None, mm="tip3p")
    qm = Psi4Engine(method="hf", basis="sto-3g", threads=1, memory="500 MB")
    try:
        eng = QMMMEngine(base=qm, region=r)
        out = eng.evaluate([StructureNode(structure=s, has_molecular_graph=False)])[0]
        x0 = np.asarray(s.geometry, dtype=float)
        h = 2e-3
        for a, c in [(1, 0), (6, 1), (7, 2)]:     # a QM atom, a water O, a water H (point-charge forces)
            e = []
            for sign in (1, -1):
                x = x0.copy()
                x[a, c] += sign * h
                e.append(eng.evaluate([StructureNode(structure=s.model_copy(update={"geometry": x}),
                                                     has_molecular_graph=False)])[0]["energy"])
            assert abs((e[0] - e[1]) / (2 * h) - out["gradient"][a, c]) < 2e-5
    finally:
        qm.close()


def test_electrostatic_embedding_gradients_with_gfn2_xtb():
    """GFN2-xTB in TIP3P water: xtb's own point-charge embedding, exact
    gradients on the QM atoms and the water (the charges' forces)."""
    pytest.importorskip("openmm")
    from mepd.engines.gfnff import XTBEngine
    from mepd.engines.qmmm import QMMMEngine
    from mepd.nodes.node import StructureNode

    s = _cluster()
    r = QMMMRegion.build(s, "0-5", active_radius=None, mm="tip3p")
    eng = QMMMEngine(base=XTBEngine(method="gfn2", executable="xtb"), region=r)
    node = lambda x: StructureNode(structure=s.model_copy(update={"geometry": x}), has_molecular_graph=False)
    x0 = np.asarray(s.geometry, dtype=float)
    out = eng.evaluate([node(x0)])[0]
    assert out["qm_charges"] is not None and len(out["qm_charges"]) == 6
    h = 1e-3
    for a, c in [(1, 0), (6, 1), (7, 2)]:     # a QM atom, a water O, a water H
        xs = []
        for sign in (1, -1):
            x = x0.copy()
            x[a, c] += sign * h
            xs.append(node(x))
        ep, em = (o["energy"] for o in eng.evaluate(xs))
        assert abs((ep - em) / (2 * h) - out["gradient"][a, c]) < 2e-5
    # The water's charges change the QM energy (unlike mechanical embedding).
    mech = QMMMEngine(base=XTBEngine(method="gfn2", executable="xtb"),
                      region=QMMMRegion.build(s, "0-5", active_radius=None, mm="gfnff"))
    assert abs(mech.evaluate([node(x0)])[0]["qm"] - out["qm"]) > 1e-4


def test_microiteration_inner_energy_has_exact_gradients():
    pytest.importorskip("openmm")
    from types import SimpleNamespace

    from mepd.engines.microiter import MicroIterations
    from mepd.engines.qmmm import _OpenMMLow

    s = _cluster()
    r = QMMMRegion.build(s, "0-5", active_radius=None, mm="tip3p")
    fake = SimpleNamespace(region=r, _low=_OpenMMLow(r))
    micro = MicroIterations(fake)
    assert micro.inner == list(range(6, 15)) and micro.outer == list(range(6)) and micro.electrostatic
    q = np.array([-0.3, -0.5, 0.1, 0.1, 0.1, 0.5])          # QM charges (the methanol)
    x0 = np.asarray(s.geometry, dtype=float)
    e0, g0 = micro._inner_energy(x0, q, 0)
    h = 1e-4
    for a, c in [(6, 0), (8, 2)]:
        e = []
        for sign in (1, -1):
            x = x0.copy()
            x[a, c] += sign * h
            e.append(micro._inner_energy(x, q, 0)[0])
        assert abs((e[0] - e[1]) / (2 * h) - g0[a, c]) < 1e-7
    relaxed = micro.relax(x0, q, 0)
    assert np.allclose(relaxed[:6], x0[:6])                  # the QM region stays put
    assert micro._inner_energy(relaxed, q, 0)[0] < e0


def test_microiterative_minimization_with_electrostatic_embedding():
    from mepd.engines.psi4 import psi4_python

    if not psi4_python():
        pytest.skip("needs Psi4")
    pytest.importorskip("openmm")
    from mepd.engines.frozen import FrozenAtomsEngine
    from mepd.engines.psi4 import Psi4Engine
    from mepd.engines.qmmm import QMMMEngine
    from mepd.nodes.node import StructureNode

    s = _cluster()
    qm = Psi4Engine(method="hf", basis="sto-3g", threads=1, memory="500 MB")
    try:
        r = QMMMRegion.build(s, "0-5", active_radius=None, mm="tip3p", frozen_atoms=[12, 13, 14])
        eng = FrozenAtomsEngine(base=QMMMEngine(base=qm, region=r), frozen=r.frozen_atoms)
        micro = eng._micro()
        assert micro is not None and micro.outer == list(range(6)) and micro.inner == list(range(6, 12))
        traj = eng.compute_geometry_optimization(StructureNode(structure=s, has_molecular_graph=False),
                                                 keywords={"fmax": 0.05, "maxiter": 15})
        x = np.asarray(traj[-1].coords)
        assert np.abs(x[[12, 13, 14]] - np.asarray(s.geometry)[[12, 13, 14]]).max() < 1e-9   # frozen
        assert np.allclose(x[[12, 13, 14]], np.asarray(s.geometry)[[12, 13, 14]])
        g = np.asarray(traj[-1].gradient)
        # The moving water is relaxed against the true QM/MM forces (the
        # gradient-corrected cheap surface), within the per-step move cap.
        assert np.abs(g[6:12]).max() * 51.42 < 0.2
        assert traj[-1].energy < traj[0].energy
        assert len(traj) <= 16                     # one QM calculation per outer step
    finally:
        qm.close()


def test_qm_hessian_is_exact_and_climbs_the_reaction_mode():
    """The microiterative TS search's starting Hessian: finite differences of
    the QM/MM gradient over the outer atoms (environment held), in Sella's
    units, keeping only the negative mode along the given direction."""
    from types import SimpleNamespace

    from mepd.engines.microiter import HARTREE_EV, MicroIterations

    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    w = np.array([-0.2, -0.05, 0.3, 0.4, 0.5, 0.6])          # Eh/bohr², two negative modes
    A = (q * w) @ q.T

    class Quadratic:
        region = SimpleNamespace(links=[], qm_atoms=[0, 1], active_mm_atoms=[2])
        _low = SimpleNamespace(electrostatic=False)

        def evaluate(self, nodes):
            x = np.asarray(nodes[0].coords, dtype=float)
            g = np.zeros_like(x)
            g[:2] = (A @ x[:2].ravel()).reshape(2, 3)
            return [{"energy": 0.0, "gradient": g}]

    micro = MicroIterations(Quadratic())
    s = Structure(symbols=["H", "H", "He"], geometry=rng.normal(size=(3, 3)))
    to_ev = HARTREE_EV * ANGSTROM_TO_BOHR ** 2
    H = micro.qm_hessian(s, s.geometry)
    assert np.allclose(H, A * to_ev, atol=1e-6)
    tangent = np.zeros((3, 3))
    tangent[:2] = q[:, 1].reshape(2, 3)                    # the reaction is the second negative mode
    H = micro.qm_hessian(s, s.geometry, tangent)
    ev = np.linalg.eigvalsh(H) / to_ev
    assert np.allclose(np.sort(ev), np.sort([0.2, -0.05, 0.3, 0.4, 0.5, 0.6]), atol=1e-6)
