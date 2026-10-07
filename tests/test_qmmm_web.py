"""QM/MM systems in the web workspace (mepd/web/qmmm.py)."""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mepd.web.app import create_app  # noqa: E402

# Methanol (QM, atoms 0-5) between three waters.
SYSTEM = """15

C   0.000  0.000  0.000
O   1.410  0.000  0.000
H  -0.360  1.030  0.000
H  -0.360 -0.510  0.890
H  -0.360 -0.510 -0.890
H   1.730  0.910  0.000
O   3.900  1.300  0.000
H   4.300  2.150  0.000
H   4.600  0.650  0.000
O  -2.900  0.300  2.300
H  -3.300  1.150  2.300
H  -3.600 -0.350  2.300
O   0.500 -3.200  0.400
H   0.900 -3.700 -0.300
H  -0.400 -3.500  0.500
"""


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture
def client(tmp_path):
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        yield c


def _upload(client, text=SYSTEM, qm="0-5", radius="3.0"):
    r = client.post("/api/qmmm/upload", files={"file": ("cluster.xyz", text)},
                    data={"qm_atoms": qm, "charge": "0", "active_radius": radius, "optimize": "false"})
    assert r.status_code == 200, r.text
    return r.json()


def _ws(client):
    return client.get("/api/state").json()["workspace"]


def test_upload_makes_one_qmmm_node_named_by_its_qm_region(client):
    out = _upload(client)
    ws = _ws(client)
    rec = ws["structures"][out["structure"]]
    assert rec["qmmm"] == out["system"]["id"]
    assert rec["role"] == "minimum" and "members" not in rec          # not split into molecules
    assert rec["smiles"] == "CO" and rec["name"].startswith("CO · ")
    assert len(ws["structures"]) == 1
    sysv = client.get(f"/api/qmmm/systems/{rec['qmmm']}").json()
    assert sysv["qm_atoms"] == [0, 1, 2, 3, 4, 5] and sysv["natoms"] == 15 and sysv["links"] == []
    # The same atoms again (another geometry) belong to the same system, as a conformer of that node.
    moved = SYSTEM.replace("O   3.900  1.300", "O   3.950  1.300")
    added = client.post("/api/structures", json={"text": moved, "optimize": False}).json()
    assert added[0]["id"] == rec["id"] and added[0]["merged"]
    # Plain methanol is a different node (a different world).
    plain = client.post("/api/structures", json={"text": "CO", "optimize": False}).json()
    assert plain[0]["id"] != rec["id"] and not plain[0].get("qmmm")
    client.post("/api/structures/merge-duplicates")
    assert len([s for s in _ws(client)["structures"].values() if s["smiles"] == "CO"]) == 2


def test_jobs_on_qmmm_structures_run_embedded_at_their_own_level(client, tmp_path):
    out = _upload(client)
    sid = out["structure"]
    jobs = client.post("/api/jobs", json={"op": "optimize", "structures": [sid], "params": {}, "dry_run": True})
    assert jobs.status_code == 200, jobs.text
    job = jobs.json()[0]
    assert job["qmmm"] == out["system"]["id"]
    assert job["level"]["key"].endswith("+qmmm:" + _ws(client)["qmmm_systems"][job["qmmm"]]["sig"])
    refused = client.post("/api/jobs", json={"op": "conformers", "structures": [sid], "params": {}, "dry_run": True})
    assert refused.status_code == 400 and "QM/MM" in refused.json()["detail"]
    plain = client.post("/api/structures", json={"text": "CO", "optimize": False}).json()[0]["id"]
    mixed = client.post("/api/jobs", json={"op": "optimize", "structures": [sid, plain], "params": {}, "dry_run": True})
    assert mixed.status_code == 400 and "QM/MM" in mixed.json()["detail"]

    # The job's profile is the QM level with the system's region added.
    from mepd.web.operations import JobContext

    ws = client.app.state.sessions.session(None).ws
    jdir = tmp_path / "job"
    ctx = JobContext(ws, jdir, jdir / "output", [ws.structure_view(sid)], "default")
    flags = ctx.common_flags()
    prof = Path(flags[flags.index("--inputs") + 1])
    data = tomllib.loads(prof.read_text())
    assert data["qmmm"] == {"file": "qmmm_region.json"} and data["engine_name"] == "gxtb"
    assert (prof.parent / "qmmm_region.json").exists()


def test_preview_update_and_checks(client):
    out = _upload(client)
    sid, sysid = out["structure"], out["system"]["id"]
    p = client.post("/api/qmmm/preview", json={"structure": sid, "qm_atoms": "0-8", "active_radius": None}).json()
    assert len(p["qm_atoms"]) == 9 and p["frozen_atoms"] == []
    old_sig = _ws(client)["qmmm_systems"][sysid]["sig"]
    upd = client.put(f"/api/qmmm/systems/{sysid}", json={"qm_atoms": "0-8"}).json()
    assert upd["n_qm"] == 9 and upd["sig"] != old_sig
    assert _ws(client)["structures"][sid]["smiles"] == "CO.O"
    chk = client.get(f"/api/qmmm/check?structure={sid}").json()
    assert chk["worst"]["frozen"] == 0 and chk["worst"]["mm_changes"] == 0


def test_operations_say_which_run_on_qmmm(client):
    ops = {o["key"]: o for o in client.get("/api/state").json()["operations"]}
    assert ops["ts"]["qmmm"] and ops["optimize"]["qmmm"] and ops["graph-enumeration"]["qmmm"]
    assert not ops["channels"]["qmmm"] and not ops["conformers"]["qmmm"]
    assert "qmmm-build" in ops and ops["qmmm-build"]["target"] == "structure"


def test_finished_build_job_becomes_a_qmmm_node(tmp_path):
    from qcdata import Structure

    from mepd.qmmm import QMMMRegion
    from mepd.web.qmmm import adopt_build
    from mepd.web.workspace import Workspace

    ws = Workspace(tmp_path / "ws")
    s = Structure.from_xyz(SYSTEM)
    out = tmp_path / "job" / "output"
    out.mkdir(parents=True)
    s.save(str(out / "system.xyz"))
    QMMMRegion.build(s, "0-5", active_radius=3.0).save(out / "region.json")
    sid = adopt_build(ws, {"id": "j_x", "output_dir": str(out), "targets": {"structures": []},
                           "params": {"solvent": "water"}, "charge": 0, "multiplicity": 1})
    rec = ws.structure(sid)
    assert rec["qmmm"] and rec["name"] == "CO · in water"
    region = ws.qmmm_region(rec["qmmm"])
    assert region.qm_atoms == [0, 1, 2, 3, 4, 5]
    assert np.allclose(np.asarray(ws.load_structure(sid).geometry), np.asarray(s.geometry))



def test_energy_split_follow_up_emits_real_cli_flags(client):
    """The split runs `mepd qmmm inspect` on one entry of a finished QM/MM job."""
    import json

    from mepd.web.operations import _cli_options

    out = _upload(client)
    (cmd,) = client.post("/api/jobs", json={"op": "optimize", "structures": [out["structure"]], "profile": "default",
                                           "dry_run": True}).json()
    manager = client.app.state.sessions.current.jobs
    job = {**cmd, "id": "j_qmmmfake", "status": "done"}
    manager.jobs[job["id"]] = job
    manager._write(job)
    xyz = client.get(f"/api/structures/{out['structure']}/xyz").text
    (manager.job_dir(job["id"])).mkdir(parents=True, exist_ok=True)
    (manager.job_dir(job["id"]) / "result.json").write_text(json.dumps(
        {"groups": [{"kind": "minima", "entries": [{"id": "e0", "frames": [{"xyz": xyz}, {"xyz": xyz}]}]}]}))
    r = client.post("/api/jobs", json={"op": "qmmm-inspect", "source_job": job["id"], "params": {"entry": "e0"},
                                       "dry_run": True})
    assert r.status_code == 200, r.text
    argv = r.json()[0]["argv"]
    assert argv[:2] == ["qmmm", "inspect"]
    known = _cli_options(("qmmm", "inspect"))
    assert not [w for w in argv if w.startswith("--") and w not in known]


def test_putting_a_gas_phase_structure_into_a_system(client):
    from mepd.web.operations import _cli_options

    out = _upload(client)
    host = out["structure"]
    gas = client.post("/api/structures", json={"text": "CO", "optimize": False}).json()[0]["id"]
    r = client.post("/api/jobs", json={"op": "qmmm-embed", "structures": [host, gas], "profile": "default",
                                       "dry_run": True})
    assert r.status_code == 200, r.text
    job = r.json()[0]
    assert job["argv"][:2] == ["qmmm", "embed"] and job["qmmm"] == out["system"]["id"]
    known = _cli_options(("qmmm", "embed"))
    assert not [w for w in job["argv"] if w.startswith("--") and w not in known]
    # Two gas-phase ends (same atom order, from a reaction SMILES): the whole reaction into solvent.
    ends = [a["id"] for a in client.post("/api/structures", json={"text": "CC=O>>C=CO", "optimize": False}).json()]
    r = client.post("/api/jobs", json={"op": "qmmm-reaction", "structures": ends, "dry_run": True})
    assert r.status_code == 200, r.text
    argv = r.json()[0]["argv"]
    assert argv[:2] == ["qmmm", "reaction"] and "--start" in argv and "--end" in argv
    known = _cli_options(("qmmm", "reaction"))
    assert not [w for w in argv if w.startswith("--") and w not in known]
    # ...but not on structures already in a QM/MM system.
    r = client.post("/api/jobs", json={"op": "qmmm-reaction", "structures": [host, ends[0]], "dry_run": True})
    assert r.status_code == 400


def test_finished_reaction_job_becomes_a_qmmm_edge_and_ts(tmp_path):
    from qcdata import Structure

    from mepd.qmmm import QMMMRegion
    from mepd.web.qmmm import adopt_reaction
    from mepd.web.workspace import Workspace

    ws = Workspace(tmp_path / "ws")
    s = Structure.from_xyz(SYSTEM)
    out = tmp_path / "job" / "output"
    out.mkdir(parents=True)
    s.save(str(out / "system.xyz"))
    x = np.asarray(s.geometry).copy()
    x[5] = x[0] + (x[5] - x[0]) * 0.0 + np.array([0.0, 0.0, 2.05])   # the O-H hydrogen onto the carbon
    s.model_copy(update={"geometry": x}).save(str(out / "product.xyz"))
    s.save(str(out / "ts.xyz"))
    QMMMRegion.build(s, "0-5", active_radius=3.0, solute_atoms=list(range(6))).save(out / "region.json")
    made = adopt_reaction(ws, {"id": "j_r", "output_dir": str(out), "targets": {"structures": []},
                               "params": {"solvent": "water"}, "charge": 0, "multiplicity": 1})
    assert made["start"] and made["end"] and made["start"] != made["end"] and made["edge"] and made["ts"]
    assert ws.structure(made["ts"])["role"] == "ts"
    assert all(ws.structure(made[k])["qmmm"] for k in ("start", "end", "ts"))


def test_a_ts_reoptimized_in_solvent_sets_the_qmmm_edge_barrier(tmp_path):
    from qcdata import Structure

    from mepd.qmmm import QMMMRegion
    from mepd.web.qmmm import attach_ts, refresh_edges_of
    from mepd.web.workspace import Workspace

    ws = Workspace(tmp_path / "ws")
    s = Structure.from_xyz(SYSTEM)
    region = QMMMRegion.build(s, "0-5", active_radius=3.0)
    sysid = ws.put_qmmm_system(region, name="in water")["id"]
    x = np.asarray(s.geometry).copy()
    x[5] = x[0] + np.array([0.0, 0.0, 2.05])                 # methanol -> its O-H hydrogen on carbon
    p = s.model_copy(update={"geometry": x})
    a = ws.add_or_merge(s, origin={"kind": "xyz"})["rec"]["id"]
    b = ws.add_or_merge(p, origin={"kind": "xyz"})["rec"]["id"]
    assert ws.structure(a)["smiles"] != ws.structure(b)["smiles"]
    eid = ws.add_edge(a, b, origin={"kind": "job", "job": "j_r"})["id"]
    level = {"profile": "default", "key": "k+qmmm:x", "label": "gxtb / QM/MM"}
    out = tmp_path / "tsopt"
    out.mkdir()
    (out / "ts.energies").write_text("-10.0\n")
    (out / "irc.xyz").write_text(p.to_xyz() + s.to_xyz())
    job = {"id": "j_t", "op": "tsopt", "qmmm": sysid, "output_dir": str(out), "level": level, "status": "done"}
    assert attach_ts(ws, job, {}) == [eid]
    assert ws.edge(eid)["origin"].get("barrier_kcal") is None      # ends not minimized at that level yet
    # IRC ends added as conformers at the TS's level (not minimized) already give a barrier, said to be so.
    for sid, e in ((a, -10.08), (b, -10.04)):
        ws.structure(sid).setdefault("conformers", []).append({"id": f"c_{sid}", "energy": e, "level": level,
                                                               "optimized": False})
    refresh_edges_of(ws, {}, [a, b])
    o = ws.edge(eid)["origin"]
    assert abs(o["barrier_kcal"] - 0.08 * 627.509474) < 1e-6 and not o["ends_minimized"]
    assert "not all minimized" in o["headline"]
    for sid, e in ((a, -10.1), (b, -10.05)):
        rec = ws.structure(sid)
        rec.update(energy=e, level=level, optimized=True)
    refresh_edges_of(ws, {}, [a, b])
    o = ws.edge(eid)["origin"]
    assert abs(o["barrier_kcal"] - 0.1 * 627.509474) < 1e-6 and o["job"] == "j_t"
    assert abs(o["reaction_kcal"] - 0.05 * 627.509474) < 1e-6
    assert o["ends_minimized"] and "from the minimized ends" in o["headline"]


def test_electrostatic_system_at_a_level_without_point_charges_is_refused_before_it_runs(client, tmp_path):
    out = _upload(client)
    sid, sysid = out["structure"], out["system"]["id"]
    pytest.importorskip("openmm")
    # The builder warns: no profile here can run TIP3P's electrostatic embedding.
    p = client.post("/api/qmmm/preview", json={"structure": sid, "qm_atoms": "0-5", "mm": "tip3p"}).json()
    assert any("Psi4" in x for x in p["problems"])
    client.put(f"/api/qmmm/systems/{sysid}", json={"mm": "tip3p"})
    r = client.post("/api/jobs", json={"op": "optimize", "structures": [sid], "profile": "default",
                                       "params": {}, "dry_run": True})
    assert r.status_code == 400 and 'engine_name = "xtb" or "psi4"' in r.json()["detail"]
    # With an xtb (GFN2) profile in the workspace, the builder no longer warns.
    (tmp_path / "ws" / "profiles" / "gfn2.toml").write_text('engine_name = "xtb"\n')
    p = client.post("/api/qmmm/preview", json={"structure": sid, "qm_atoms": "0-5", "mm": "tip3p"}).json()
    assert not any("Psi4" in x for x in p["problems"])


def test_demo_visitors_get_qmmm_within_limits(tmp_path):
    from mepd.web.demo import DemoPolicy

    root = tmp_path / "demo"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    policy = DemoPolicy(max_atoms=10, qmmm={"max_atoms": 14, "max_active_radius": 6.0})
    with TestClient(create_app(root, demo=policy, demo_password="pw"), follow_redirects=False) as c:
        r = c.post("/login", data={"password": "pw"})
        c.cookies.set("mepd_visitor", r.cookies.get("mepd_visitor"))
        ops = {o["key"]: o for o in c.get("/api/state").json()["operations"]}
        assert all(ops[k]["available"] for k in ("qmmm-build", "qmmm-reaction", "qmmm-embed"))

        def upload(text=SYSTEM, qm="0-5", radius="3.0"):
            return c.post("/api/qmmm/upload", files={"file": ("cluster.xyz", text)},
                          data={"qm_atoms": qm, "charge": "0", "active_radius": radius, "optimize": "false"})

        r = upload()                                                     # 15 atoms > 14
        assert r.status_code == 400 and "at most 14 atoms" in r.json()["detail"]
        small = "\n".join(SYSTEM.splitlines()[:2] + SYSTEM.splitlines()[2:14]).replace("15", "12", 1) + "\n"
        assert upload(small, qm="0-11").status_code == 400               # 12 QM atoms > max_atoms 10
        assert upload(small, radius="0").status_code == 400              # everything moving
        sysrec = upload(small)
        assert sysrec.status_code == 200, sysrec.text
        sysid = sysrec.json()["system"]["id"]
        r = c.put(f"/api/qmmm/systems/{sysid}", json={"qm_atoms": "0-11"})
        assert r.status_code == 400 and "QM atoms" in r.json()["detail"]
        r = c.post("/api/qmmm/systems", json={"terachem": "/etc/hosts"})
        assert r.status_code == 403
