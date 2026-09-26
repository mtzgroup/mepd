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
