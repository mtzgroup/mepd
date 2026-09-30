"""Design tab: 3D molecule edits (RDKit), and moving designs to and from the graph."""
import json

import pytest
from fastapi.testclient import TestClient

from mepd.web import design as d
from mepd.web.app import apply_design_optimization, create_app
from mepd.web.workspace import Workspace, WorkspaceError


def _h_on(mb, idx):
    mol = d.read(mb)
    return next(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1)


def test_edits_keep_valences_and_hydrogens():
    acid = d.from_smiles("CC(=O)O")
    assert acid["smiles"] == "CC(=O)O" and acid["natoms"] == 8
    mb = acid["molblock"]
    assert d.edit(mb, {"op": "group", "atom": _h_on(mb, 0), "group": "phenyl"})["smiles"] == "O=C(O)Cc1ccccc1"
    assert d.edit(mb, {"op": "group", "atom": 0, "group": "tert-butyl"})["smiles"] == "CC(C)(C)C(=O)O"
    assert d.edit(mb, {"op": "element", "atom": 3, "element": "S"})["smiles"] == "CC(=O)S"
    assert d.edit(mb, {"op": "element", "atom": 1, "element": "P"})["smiles"] == "C[PH](=O)O"
    assert d.edit(mb, {"op": "bond", "a": 1, "b": 2, "order": 1})["smiles"] == "CC(O)O"
    assert d.edit(mb, {"op": "add", "atom": 0, "element": "N"})["smiles"] == "NCC(=O)O"
    assert d.edit(mb, {"op": "delete", "atom": 3})["smiles"] == "CC=O"
    charged = d.edit(mb, {"op": "charge", "atom": 3, "delta": -1})
    assert charged["smiles"] == "CC(=O)[O-]" and charged["charge"] == -1
    with pytest.raises(WorkspaceError, match="would have"):
        d.edit(mb, {"op": "bond", "a": 0, "b": 3, "order": 3})
    with pytest.raises(WorkspaceError, match="terminal group"):
        d.edit(mb, {"op": "group", "atom": 1, "group": "methyl"})     # the carbonyl C is not terminal


def test_group_swap_keeps_the_rest_of_the_geometry_in_place():
    import numpy as np

    mb = d.from_smiles("CCO")["molblock"]
    before = d.read(mb).GetConformer().GetPositions()
    after = d.read(d.edit(mb, {"op": "group", "atom": _h_on(mb, 2), "group": "methyl"})["molblock"]).GetConformer().GetPositions()
    # C, C, O and the carbons' hydrogens did not move.
    assert np.allclose(before[:3], after[:3], atol=1e-4)


def test_structures_without_bond_orders_load_with_a_warning():
    # Five hydrogens on one carbon (like an SN2 transition state's centre): no Lewis structure.
    ts_like = ("6\n\nC 0 0 0\nH 1.09 0 0\nH -1.09 0 0\nH 0 1.09 0\nH 0 -1.09 0\nH 0 0 1.09\n")
    info, warnings = d.from_xyz(d.to_xyz(d.from_smiles("CCO")["molblock"]))
    assert info["smiles"] == "CCO" and not warnings
    info, warnings = d.from_xyz(ts_like)
    assert warnings and info["natoms"] == 6


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        yield c


def test_design_round_trip_through_the_graph(client):
    new = client.post("/api/design/new", json={"smiles": "CCO"}).json()
    assert new["name"] == "CCO" and new["charge"] == 0 and new["multiplicity"] == 1
    edited = client.post("/api/design/edit", json={"op": {"op": "element", "atom": 2, "element": "S"}}).json()
    assert edited["smiles"] == "CCS" and edited["name"] == "CCS"          # an automatic name follows the molecule
    undone = client.put("/api/design", json={"molblock": new["molblock"]}).json()
    assert undone["smiles"] == "CCO"
    rec = client.post("/api/design/to-graph").json()
    assert rec["name"] == "CCO" and not rec["merged"] and rec["origin"]["kind"] == "design"
    assert client.post("/api/design/to-graph").json()["duplicate"]         # same geometry again: nothing new
    loaded = client.post("/api/design/load", json={"structure": rec["id"]}).json()
    assert loaded["smiles"] == "CCO" and loaded["source"]["structure"] == rec["id"]
    moved = client.post("/api/design/clean").json()
    back = client.post("/api/design/to-graph").json()
    assert back["id"] == rec["id"] and back["merged"]                       # the same molecule: a new conformer
    assert moved["energy"] is None
    (job,) = client.post("/api/jobs", json={"op": "design-optimize", "dry_run": True, "profile": "default"}).json()
    assert job["argv"][0] == "optimize" and job["design_rev"] == client.get("/api/state").json()["workspace"]["design"]["rev"]
    assert client.delete("/api/design").json()["ok"]
    assert client.post("/api/design/edit", json={"op": {"op": "hydrogens"}}).status_code == 400


def test_minimized_design_replaces_the_geometry_unless_edited_since(tmp_path):
    ws = Workspace(tmp_path / "ws")
    info = d.from_smiles("CCO")
    ws.set_design({"molblock": info["molblock"], "charge": 0, "multiplicity": 1, "name": "x"})
    rev = ws.design["rev"]
    out = tmp_path / "out"
    out.mkdir()
    moved = d.to_xyz(d.clean(info["molblock"])["molblock"])
    (out / "opt_0.xyz").write_text(moved)
    (out / "summary.json").write_text(json.dumps({"structures": [{"index": 0, "converged": True, "energy": -155.0}]}))
    job = {"id": "j1", "status": "done", "output_dir": str(out), "design_rev": rev, "level": {"key": "k", "label": "gxtb"}}
    apply_design_optimization(ws, job)
    assert ws.design["energy"] == -155.0 and ws.design["level"]["label"] == "gxtb"
    job["design_rev"] = rev                                                # stale now: the design moved on
    ws.set_design({**ws.design, "energy": None})
    apply_design_optimization(ws, job)
    assert ws.design["energy"] is None


def test_placed_species_and_their_charge(client):
    d.edit(d.from_smiles("CC=O")["molblock"], {"op": "place", "atom": 2, "species": "water", "count": 3})
    client.post("/api/design/new", json={"smiles": "CC=O"})
    ion = client.post("/api/design/edit", json={"op": {"op": "place", "atom": 2, "species": "Mg2+"}}).json()
    assert ion["smiles"] == "CC=O.[Mg+2]" and ion["charge"] == 2
    # A charge set by hand is kept on top of the formal charges through later edits.
    client.put("/api/design", json={"charge": 3})
    more = client.post("/api/design/edit", json={"op": {"op": "place", "atom": 2, "species": "water"}}).json()
    assert more["charge"] == 3 and more["charge_offset"] == 1


def test_ts_search_from_the_design_snaps_back_when_it_fails(tmp_path):
    from mepd.web.app import apply_design_tsopt

    ws = Workspace(tmp_path / "ws")
    info = d.from_smiles("CC=O")
    ws.set_design({"molblock": info["molblock"], "charge": 0, "multiplicity": 1, "name": "guess"})
    submitted = dict(ws.design)
    # The user can't edit while it runs, but say something moved the design: a failure restores what was sent.
    ws.set_design({**ws.design, "molblock": d.clean(info["molblock"])["molblock"]})
    job = {"id": "j1", "status": "failed", "error": "TS optimization did not converge", "design_rev": ws.design["rev"],
           "design_snapshot": submitted, "level": {"key": "k", "label": "gxtb"}}
    apply_design_tsopt(ws, job, {"groups": [], "headline": "No TS converged"})
    assert ws.design["molblock"] == submitted["molblock"]
    assert ws.design["last_ts"] == {"job": "j1", "ok": False, "headline": "No TS converged",
                                    "error": "TS optimization did not converge"}
    # Converged: the design becomes the TS, with its barrier.
    moved = d.to_xyz(d.clean(info["molblock"])["molblock"])
    result = {"headline": "1 TS optimized", "groups": [
        {"kind": "ts", "entries": [{"id": "ts", "barrier_kcal": 40.0, "frames": [{"xyz": moved, "energy_hartree": -153.7}]}]},
        {"kind": "irc", "entries": [{"id": "ts_irc", "note": "A ⇌ B", "frames": []}]}]}
    ok = {**job, "status": "done", "error": None, "design_rev": ws.design["rev"]}
    apply_design_tsopt(ws, ok, result)
    assert ws.design["role"] == "ts" and ws.design["energy"] == -153.7
    assert ws.design["last_ts"]["ok"] and ws.design["last_ts"]["barrier_kcal"] == 40.0


def test_any_species_or_group_as_smiles():
    mb = d.from_smiles("CC=O")["molblock"]
    assert d.edit(mb, {"op": "place", "atom": 2, "smiles": "[Cl-]"})["charge"] == -1
    assert d.edit(mb, {"op": "place", "atom": 2, "smiles": "[Pd]"})["smiles"] == "CC=O.[Pd]"
    assert d.edit(mb, {"op": "group", "atom": 0, "smiles": "[*]C(=O)N(C)C"})["smiles"] == "CN(C)C(=O)C=O"
    with pytest.raises(WorkspaceError, match=r"\[\*\]"):
        d.edit(mb, {"op": "group", "atom": 0, "smiles": "CC"})
    with pytest.raises(WorkspaceError, match="could not read"):
        d.edit(mb, {"op": "place", "atom": 2, "smiles": "not(a smiles"})


def test_elements_outside_the_level_of_theory_are_flagged():
    from mepd.web.coverage import element_warnings

    assert element_warnings(None, ["C", "H", "Pd"]) == []                         # g-xTB covers H-Lr
    assert "not parametrized for Og" in element_warnings(None, ["C", "Og"])[0]
    xtb = 'engine_name = "qccompute"\nprogram = "xtb"\n'
    assert "GFN2-xTB is not parametrized for U" in element_warnings(xtb, ["C", "U"])[0]
    mlip = 'engine_name = "mlip"\n[mlip_engine_kwds]\nmodel = "aimnet2-rxn"\n'
    assert "Mg" in element_warnings(mlip, ["C", "H", "Mg"])[0]
    assert "Check that psi4" in element_warnings('engine_name = "qccompute"\nprogram = "psi4"\n', ["C", "Pd"])[0]


def test_design_minimize_skips_the_hessian_check_unless_asked(client):
    client.post("/api/design/new", json={"smiles": "CCO"})
    assert client.get("/api/design/coverage").json()["method"] == "g-xTB"
    for flag in (False, True):
        (job,) = client.post("/api/jobs", json={"op": "design-optimize", "dry_run": True,
                                               "params": {"validate_minima_with_hessian": flag}}).json()
        assert ("--validate-minima-with-hessian" in job["argv"]) == flag


def test_a_bond_beyond_an_atoms_valence_says_why_instead_of_crashing(client):
    """Bonding AlCl3 to an alkene carbon was a 500 "Internal Server Error"
    (the valence message itself raised under RDKit 2026.03), so the click
    seemed to do nothing. It is a 400 that names the atom and the way out."""
    assert client.post("/api/design/new", json={"smiles": "C1=CCCCC1.[Cl][Al]([Cl])[Cl]"}).status_code == 200
    r = client.post("/api/design/edit", json={"op": {"op": "bond", "a": 7, "b": 0, "order": 1}})
    assert r.status_code == 400, r.text
    assert "Al8 would have 4 bonds" in r.json()["detail"] and "Charge" in r.json()["detail"]
    # Following the hint: Al- first, then the bond forms.
    assert client.post("/api/design/edit", json={"op": {"op": "charge", "atom": 7, "delta": -1}}).status_code == 200
    r = client.post("/api/design/edit", json={"op": {"op": "bond", "a": 7, "b": 0, "order": 1}})
    assert r.status_code == 200, r.text
    assert "[Al-]" in r.json()["smiles"]


def test_xyz_files_start_a_design_and_can_be_placed_as_they_are(client):
    import numpy as np

    water = "3\nwater\nO 0.000 0.000 0.117\nH 0.000 0.757 -0.470\nH 0.000 -0.757 -0.470\n"
    new = client.post("/api/design/new", json={"xyz": water, "name": "water"}).json()
    assert new["smiles"] == "O" and new["name"] == "water" and new["charge"] == 0
    # Bare "El x y z" lines (no header) and a charge for the bond orders.
    hydroxide = client.post("/api/design/new", json={"xyz": "O 0 0 0\nH 0 0 0.97\n", "charge": -1}).json()
    assert hydroxide["smiles"] == "[OH-]" and hydroxide["charge"] == -1
    assert client.post("/api/design/new", json={"xyz": "2\n\nO 0 0 0\n"}).status_code == 400

    # An uploaded structure placed next to an atom keeps its own geometry
    # (no force-field relaxation), and brings the charge it was given.
    client.post("/api/design/new", json={"smiles": "CC=O"})
    bent = "3\n\nO 0 0 0\nH 0.96 0 0\nH -0.2 0.94 0\n"          # a deliberately odd H-O-H angle
    placed = client.post("/api/design/edit", json={"op": {"op": "place", "atom": 2, "xyz": bent, "charge": 0}}).json()
    assert placed["smiles"] == "CC=O.O" and placed["charge"] == 0
    pos = d.read(placed["molblock"]).GetConformer().GetPositions()[-3:]
    ref = np.array([[0, 0, 0], [0.96, 0, 0], [-0.2, 0.94, 0]])
    dist = lambda x: np.linalg.norm(x[:, None] - x[None], axis=-1)
    assert np.allclose(dist(pos), dist(ref), atol=1e-3)
    ion = client.post("/api/design/edit", json={"op": {"op": "place", "atom": 2, "xyz": "1\n\nNa 0 0 0\n", "charge": 1}}).json()
    assert ion["charge"] == 1


def test_a_reaction_design_edits_both_sides_and_keeps_their_atoms_matched(client):
    new = client.post("/api/design/new", json={"smiles": "CC(=O)C>>CC(O)=C"}).json()
    rx = new["reaction"]
    assert new["smiles"] == "CC(C)=O" and rx["product"]["smiles"] == "C=C(C)O" and rx["amap"] == list(range(10))
    assert rx["balanced"] and rx["mapping"]["source"] == "slapmapper"
    # Swap a methyl H on the reactant for a phenyl: the product gets it too, on the matching carbon.
    mol = d.read(new["molblock"])
    h = next(a.GetIdx() for a in mol.GetAtomWithIdx(0).GetNeighbors() if a.GetAtomicNum() == 1)
    out = client.post("/api/design/edit", json={"op": {"op": "group", "atom": h, "group": "phenyl"}, "side": "reactant"}).json()
    assert out["smiles"] == "CC(=O)Cc1ccccc1" and out["reaction"]["product"]["smiles"] in ("C=C(O)Cc1ccccc1", "OC(=C)Cc1ccccc1")
    assert out["reaction"]["balanced"] and sorted(out["reaction"]["amap"]) == list(range(out["natoms"]))
    # Matched atoms are the same element, and the ring sits on the same carbon on both sides.
    r, p = d.read(out["molblock"]), d.read(out["reaction"]["product"]["molblock"])
    amap = out["reaction"]["amap"]
    assert all(r.GetAtomWithIdx(i).GetSymbol() == p.GetAtomWithIdx(j).GetSymbol() for i, j in enumerate(amap))
    for b in r.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if r.GetAtomWithIdx(i).GetIsAromatic() and r.GetAtomWithIdx(j).GetIsAromatic():
            assert p.GetBondBetweenAtoms(amap[i], amap[j]) is not None
    # One side alone: no longer balanced, so it can't go to the graph.
    ph = next(a.GetIdx() for a in p.GetAtoms() if a.GetAtomicNum() == 1)
    lone = client.post("/api/design/edit", json={"op": {"op": "element", "atom": ph, "element": "Cl"}, "side": "product",
                                                 "linked": False}).json()
    assert lone["reaction"]["balanced"] is False
    assert client.post("/api/design/to-graph").status_code == 400
    # Undo restores both sides.
    back = client.put("/api/design", json={"molblock": out["molblock"], "reaction": {
        "product_molblock": out["reaction"]["product"]["molblock"], "amap": amap}}).json()
    assert back["reaction"]["balanced"]
    added = client.post("/api/design/to-graph").json()
    ws = client.get("/api/state").json()["workspace"]
    edge = ws["edges"][added["edge"]]
    assert edge["source"] == added["reactant"]["id"] and len(edge["conformers"]) == 2
    (job,) = client.post("/api/jobs", json={"op": "ts", "edges": [added["edge"]], "dry_run": True, "profile": "default"}).json()
    assert job["op"] == "ts"
    assert client.post("/api/design/minimize", json={}).status_code == 400
    assert client.get("/api/design/xyz?side=product").text.startswith(str(out["natoms"]))


def test_a_molecule_built_from_scratch(client):
    """No SMILES or xyz: start empty, drop atoms in space, grow them by clicking hydrogens."""
    new = client.post("/api/design/new", json={"scratch": True}).json()
    assert new["natoms"] == 0 and new["name"] == "New molecule"
    assert client.post("/api/design/to-graph").status_code == 400          # nothing to add yet
    c = client.post("/api/design/edit", json={"op": {"op": "drop", "element": "C", "xyz": [0, 0, 0]}}).json()
    assert c["smiles"] == "C" and c["natoms"] == 5
    h = next(a.GetIdx() for a in d.read(c["molblock"]).GetAtoms() if a.GetSymbol() == "H")
    cc = client.post("/api/design/edit", json={"op": {"op": "add", "atom": h, "element": "C"}}).json()
    assert cc["smiles"] == "CC"                                             # the new atom took that H's place
    ion = client.post("/api/design/edit", json={"op": {"op": "drop", "element": "Na", "xyz": [5, 0, 0],
                                                        "hydrogens": False}}).json()
    assert ion["smiles"] == "CC.[Na]"
    assert client.post("/api/design/to-graph").status_code == 200
