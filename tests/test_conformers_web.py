"""One node per molecule: conformers merge into it, the lowest represents it,
edges can pick a conformer at either end, and sampling fills the list."""
import json

import pytest
from fastapi.testclient import TestClient

from mepd.web import chem
from mepd.web.app import attach_conformers, create_app
from mepd.web.workspace import Workspace
from tests.test_web import HCN_XYZ, WATER_BENT_XYZ, WATER_XYZ, _add, _wait  # noqa: F401

WATER_2 = WATER_XYZ.replace("0.758 0.000 0.504", "0.700 0.100 0.520", 1)
LEVEL = {"profile": None, "key": "k1", "label": "gxtb"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        yield c


def _s(xyz):
    return chem.structures_from_xyz_text(xyz, 0, 1)[0]


def test_conformers_merge_into_one_node_and_the_lowest_represents_it(tmp_path):
    ws = Workspace(tmp_path / "ws")
    a = ws.add_or_merge(_s(WATER_XYZ), origin={"kind": "xyz"}, energy=-76.0, level=LEVEL)
    b = ws.add_or_merge(_s(WATER_2), origin={"kind": "xyz"}, energy=-76.1, level=LEVEL)
    again = ws.add_or_merge(_s(WATER_2), origin={"kind": "xyz"}, energy=-76.1, level=LEVEL)
    other = ws.add_or_merge(_s(WATER_BENT_XYZ), origin={"kind": "xyz"})
    rec = ws.structure(a["rec"]["id"])
    assert b["merged"] and b["rec"]["id"] == rec["id"] and again["duplicate"]
    assert not other["merged"] and other["rec"]["id"] != rec["id"]
    assert len(rec["conformers"]) == 2 and rec["conformer"] == b["conformer"] and rec["energy"] == -76.1
    # The node's geometry file is its lowest conformer's.
    assert ws.structure_path(rec["id"]).read_text() == ws.conformer_path(rec["id"], b["conformer"]).read_text()
    # A TS is never merged.
    ts = ws.add_or_merge(_s(WATER_XYZ), origin={"kind": "xyz"}, role="ts")
    assert not ts["merged"]
    # Energies from another level are never compared with these.
    ws.add_conformer(rec["id"], _s(WATER_XYZ.replace("0.504", "0.530")), energy=-99.0,
                     level={"key": "other", "label": "x"})
    assert ws.structure(rec["id"])["energy"] == -76.1
    # Deleting the representative hands over to the next lowest.
    ws.delete_conformer(rec["id"], b["conformer"])
    assert ws.structure(rec["id"])["energy"] == -76.0


def test_older_workspace_gets_conformers_and_duplicates_merge(tmp_path):
    ws = Workspace(tmp_path / "ws")
    one = ws.add_structure(_s(WATER_XYZ), origin={"kind": "xyz"}, merge=False)
    two = ws.add_structure(_s(WATER_2), origin={"kind": "xyz"}, merge=False, energy=-1.0, level=LEVEL)
    three = ws.add_structure(_s(WATER_BENT_XYZ), origin={"kind": "xyz"})
    ws.add_edge(two["id"], three["id"])
    ws.add_edge(one["id"], two["id"])             # a "conformer change" edge: dropped by the merge
    # Simulate a workspace saved before conformers existed.
    data = json.loads(ws._fp.read_text())
    for rec in data["structures"].values():
        rec.pop("conformers"), rec.pop("conformer")
    ws._fp.write_text(json.dumps(data))
    ws = Workspace(tmp_path / "ws")
    assert all(len(r["conformers"]) == 1 for r in ws.snapshot()["structures"].values())
    assert ws.merge_duplicates() == {"merged": 1}
    snap = ws.snapshot()
    assert set(snap["structures"]) == {one["id"], three["id"]}
    assert len(snap["structures"][one["id"]]["conformers"]) == 2
    (edge,) = snap["edges"].values()
    assert {edge["source"], edge["target"]} == {one["id"], three["id"]}


def test_edge_conformer_choice_reaches_the_job(client):
    first = _add(client, WATER_XYZ)[0]
    conf = client.post("/api/structures", json={"text": WATER_2, "optimize": False}).json()[0]
    assert conf["merged"] and conf["id"] == first["id"]
    (b,) = _add(client, WATER_BENT_XYZ)
    edge = client.post("/api/edges", json={"source": first["id"], "target": b["id"]}).json()
    cid = conf["added_conformer"]
    edge = client.patch(f"/api/edges/{edge['id']}", json={"conformers": {first["id"]: cid}}).json()
    assert edge["conformers"] == {first["id"]: cid}
    (job,) = client.post("/api/jobs", json={"op": "ts", "edges": [edge["id"]], "dry_run": True,
                                           "params": {"endpoints": "xyz"}}).json()
    assert job["target_conformers"][0] == cid
    # Back to the default (lowest) conformer.
    edge = client.patch(f"/api/edges/{edge['id']}", json={"conformers": {first["id"]: None}}).json()
    assert edge["conformers"] == {}
    assert client.patch(f"/api/edges/{edge['id']}", json={"conformers": {first["id"]: "nope"}}).status_code == 400


def test_sampled_conformers_attach_to_their_node(tmp_path):
    ws = Workspace(tmp_path / "ws")
    rec = ws.add_structure(_s(WATER_XYZ), origin={"kind": "xyz"})
    frames = [{"xyz": WATER_2, "energy_hartree": -76.2}, {"xyz": WATER_BENT_XYZ, "energy_hartree": -76.1}]
    result = {"groups": [{"kind": "conformers", "entries": [
        {"id": f"conf_{i}", "label": f"Conformer {i}", "frames": [f], "validation": None}
        for i, f in enumerate(frames)]}]}
    job = {"id": "j1", "op": "conformers", "targets": {"structures": [rec["id"]], "edges": []},
           "params": {"minimize": True}, "level": LEVEL, "charge": 0, "multiplicity": 1}
    assert attach_conformers(ws, job, result) == 1     # the pulled-apart frame is another molecule: skipped
    rec = ws.structure(rec["id"])
    assert len(rec["conformers"]) == 2 and rec["energy"] == -76.2 and rec["level"] == LEVEL


def test_job_input_is_the_chosen_conformer(tmp_path):
    from mepd.web.operations import JobContext

    ws = Workspace(tmp_path / "ws")
    rec = ws.add_structure(_s(WATER_XYZ), origin={"kind": "xyz"}, energy=-76.0, level=LEVEL)
    cid, _ = ws.add_conformer(rec["id"], _s(WATER_2), energy=-75.0, level=LEVEL)
    ctx = JobContext(ws, tmp_path / "job", tmp_path / "job" / "output", [], None)
    lowest = ctx.snapshot_structure(ws.structure_view(rec["id"]), "a").read_text()
    chosen = ctx.snapshot_structure(ws.structure_view(rec["id"], cid), "b").read_text()
    assert lowest == ws.structure_path(rec["id"]).read_text()
    assert chosen == ws.conformer_path(rec["id"], cid).read_text() != lowest


def test_path_search_conformers_join_their_endpoint_nodes(tmp_path):
    ws = Workspace(tmp_path / "ws")
    start = ws.add_structure(_s(WATER_XYZ), origin={"kind": "xyz"}, energy=-76.0, level=LEVEL)
    end = ws.add_structure(_s(WATER_BENT_XYZ), origin={"kind": "xyz"})
    new_conf = WATER_XYZ.replace("0.758 0.000 0.504", "0.600 0.200 0.600", 1)
    result = {"groups": [{"kind": "path", "entries": [
        {"id": "mep", "label": "MEP", "frames": [
            {"xyz": new_conf, "energy_hartree": -76.3},           # the start's molecule, a new conformer
            {"xyz": HCN_XYZ, "energy_hartree": -75.0},            # (not an endpoint's molecule)
            {"xyz": WATER_XYZ, "energy_hartree": -76.0}]}]}]}     # the start's existing conformer, again
    job = {"id": "j1", "op": "ts", "targets": {"structures": [start["id"], end["id"]], "edges": []},
           "params": {}, "level": LEVEL, "charge": 0, "multiplicity": 1}
    assert attach_conformers(ws, job, result) == 1
    rec = ws.structure(start["id"])
    assert len(rec["conformers"]) == 2 and rec["energy"] == -76.3   # the new one is lower: it represents the node
    assert rec["conformers"][-1]["origin"]["label"] == "MEP (start)"


def test_chosen_endpoint_conformers_are_saved_on_the_edge(client):
    first = _add(client, WATER_XYZ)[0]
    conf = client.post("/api/structures", json={"text": WATER_2, "optimize": False}).json()[0]
    (b,) = _add(client, WATER_BENT_XYZ)
    cid = conf["added_conformer"]
    (job,) = client.post("/api/jobs", json={"op": "ts", "structures": [first["id"], b["id"]], "profile": "default",
                                           "params": {"endpoints": "xyz"}, "conformers": {first["id"]: cid}}).json()
    assert job["target_conformers"][0] == cid
    edge = client.get("/api/state").json()["workspace"]["edges"][job["targets"]["edges"][0]]
    assert edge["conformers"] == {first["id"]: cid}
    client.post(f"/api/jobs/{job['id']}/cancel")


def test_level_fingerprint_is_the_effective_level_and_old_records_migrate(tmp_path):
    from mepd.web.workspace import _legacy_level_key, level_key

    restated = 'engine_name = "gxtb"\nprogram = "xtb"\n[gxtb_engine_kwds]\nn_threads = 8\nn_parallel = 4\n'
    assert level_key(restated) == level_key(None)                       # same energies: same level
    assert level_key('engine_name = "gxtb"\n[gxtb_engine_kwds]\nextra_args = "--alpb water"\n') != level_key(None)
    mlip = 'engine_name = "mlip"\n[mlip_engine_kwds]\nmodel = "{}"\n'
    assert level_key(mlip.format("aimnet2")) != level_key(mlip.format("ani-2x"))   # were equal before
    # A structure recorded under the old fingerprint of a profile gets the new one.
    ws = Workspace(tmp_path / "ws")
    ws.write_profile("p", restated)
    rec = ws.add_structure(_s(WATER_XYZ), origin={"kind": "xyz"}, energy=-1.0,
                           level={"profile": "p", "key": _legacy_level_key(restated), "label": "gxtb"})
    data = json.loads(ws._fp.read_text())
    data.pop("level_keys", None)
    ws._fp.write_text(json.dumps(data))
    ws = Workspace(tmp_path / "ws")
    assert ws.structure(rec["id"])["level"]["key"] == level_key(None)
