"""mepd web: workspace/graph API, operation argv building, the job queue
(real subprocesses), and reading results back into the graph."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mepd.web import operations as ops  # noqa: E402
from mepd.web.app import create_app  # noqa: E402

WATER_XYZ = """3
water
O 0.000 0.000 0.000
H 0.758 0.000 0.504
H -0.758 0.000 0.504
"""

# A different species with water's atoms (one H pulled off), so it is its own
# node: two geometries of the *same* molecule are one node with two conformers.
WATER_BENT_XYZ = """3
water, one H pulled off
O 0.000 0.000 0.000
H 2.600 0.000 0.450
H -0.758 0.000 0.504
"""

HCN_XYZ = """3
hydrogen cyanide
H 0.000 0.000 -1.066
C 0.000 0.000 0.000
N 0.000 0.000 1.156
"""


@pytest.fixture(autouse=True)
def _isolated_recent_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Don't spawn `mepd defaults` (builds an engine) for every test.
    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    app = create_app(tmp_path / "ws", max_concurrent=1)
    with TestClient(app) as c:
        yield c


def _add(client, text, **kw):
    kw.setdefault("optimize", False)  # tests that aren't about optimize-on-add skip the QM job
    r = client.post("/api/structures", json={"text": text, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def _wait(client, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{jid}").json()["job"]
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {jid} still {job['status']}")


def test_structures_from_smiles_and_multiframe_xyz(client):
    (claisen,) = _add(client, "C=CCOC=C allyl_vinyl_ether")
    assert claisen["name"] == "allyl_vinyl_ether"
    assert claisen["origin"] == {"kind": "smiles", "input": "C=CCOC=C"}
    assert claisen["natoms"] == 14 and claisen["charge"] == 0

    frames = _add(client, WATER_XYZ + WATER_BENT_XYZ, name="w")
    assert [f["name"] for f in frames] == ["w 0", "w 1"]
    assert all(f["formula"] == "H2O" for f in frames)

    xyz = client.get(f"/api/structures/{frames[0]['id']}/xyz").text
    assert xyz.splitlines()[0].strip() == "3"

    state = client.get("/api/state").json()
    assert len(state["workspace"]["structures"]) == 3
    assert {op["key"] for op in state["operations"]} >= {"ts", "channels", "hessian-sample", "network-splits"}

    r = client.get("/api/depict", params={"smiles": "C=CCOC=C"})
    assert r.status_code == 200 and "<svg" in r.text


def test_bad_smiles_is_a_400_not_a_crash(client):
    r = client.post("/api/structures", json={"text": "not_a_smiles(("})
    assert r.status_code == 400
    assert "could not embed" in r.json()["detail"]


def test_edges_conserve_atoms_and_cascade_on_delete(client):
    a, b = _add(client, WATER_XYZ + WATER_BENT_XYZ)
    (c,) = _add(client, "CCO")
    r = client.post("/api/edges", json={"source": a["id"], "target": c["id"]})
    assert r.status_code == 400 and "conserve atoms" in r.json()["detail"]

    edge = client.post("/api/edges", json={"source": a["id"], "target": b["id"]}).json()
    again = client.post("/api/edges", json={"source": b["id"], "target": a["id"]}).json()
    assert again["id"] == edge["id"]  # undirected identity, no duplicates

    rev = client.patch(f"/api/edges/{edge['id']}", json={"reverse": True}).json()
    assert (rev["source"], rev["target"]) == (b["id"], a["id"])

    client.delete(f"/api/structures/{a['id']}")
    assert client.get("/api/state").json()["workspace"]["edges"] == {}


def test_dry_run_commands_match_cli_flags(client):
    (s,) = _add(client, "C=CCOC=C")
    (e,) = _add(client, "C=CCCC=O")

    def cmd(op, params=None, structures=None, edges=None, profile="default"):
        r = client.post("/api/jobs", json={
            "op": op, "structures": structures or [], "edges": edges or [],
            "params": params or {}, "profile": profile, "dry_run": True})
        assert r.status_code == 200, r.text
        return r.json()

    (ts,) = cmd("ts", structures=[s["id"], e["id"]])
    argv = ts["argv"]
    # Both endpoints were entered as SMILES -> SMILES go straight to mepd.
    assert argv[:5] == ["run", "--start", "C=CCOC=C", "--end", "C=CCCC=O"]
    assert "--recursive" in argv and "--use-tsopt" in argv and "--irc" in argv
    assert "--validate-minima-with-hessian" in argv
    # SMILES embeddings are not minima: auto asks mepd to relax them.
    assert "--minimize-ends" in argv
    assert argv[argv.index("--inputs") + 1].endswith("inputs/profile.toml")
    # A dry run creates neither jobs nor edges.
    st = client.get("/api/state").json()
    assert st["jobs"] == [] and st["workspace"]["edges"] == {}

    (ts_xyz,) = cmd("ts", {"endpoints": "xyz", "path_mode": "single", "irc": False, "use_tsopt": False,
                           "validate_minima_with_hessian": False}, structures=[s["id"], e["id"]])
    argv = ts_xyz["argv"]
    assert argv[2].endswith("start.xyz") and argv[4].endswith("end.xyz")
    assert "--recursive" not in argv and "--use-tsopt" not in argv
    # MSMEP-only knobs are dropped for a single path search...
    assert not any("validate-minima" in a for a in argv)
    assert "--network-completion-mode" not in argv  # ...and so are knobs of disabled features
    (ts_noh,) = cmd("ts", {"validate_minima_with_hessian": False}, structures=[s["id"], e["id"]])
    assert "--no-validate-minima-with-hessian" in ts_noh["argv"]

    (ch,) = cmd("channels", {"backend": "crest", "workers": 8}, structures=[s["id"], e["id"]])
    assert ch["argv"][0] == "channels"
    assert ch["argv"][ch["argv"].index("--backend") + 1] == "crest"
    assert ch["argv"][ch["argv"].index("--workers") + 1] == "8"
    assert "--crest-threads" not in ch["argv"]  # CREST stays single-threaded (CLI default)
    assert "--crest-ewin" in ch["argv"] and "--n-embed" not in ch["argv"]

    batch = cmd("hessian-sample", {"amplitude_policy": "energy"}, structures=[s["id"], e["id"]], profile=None)
    assert len(batch) == 2
    assert batch[0]["argv"][:2] == ["discovery", "hessian-sample"]
    assert "--inputs" not in batch[0]["argv"]


def test_invalid_params_are_reported(client):
    (s,) = _add(client, "C=CCOC=C")
    (e,) = _add(client, "C=CCCC=O")
    body = {"op": "ts", "structures": [s["id"], e["id"]], "dry_run": True}
    r = client.post("/api/jobs", json={**body, "params": {"irc": True, "use_tsopt": False}})
    assert r.status_code == 400 and "Optimize TS" in r.json()["detail"]
    r = client.post("/api/jobs", json={**body, "params": {"no_such_knob": 1}})
    assert r.status_code == 400 and "no_such_knob" in r.json()["detail"]
    r = client.post("/api/jobs", json={"op": "nanoreactor", "structures": [s["id"]]})
    assert r.status_code == 400 and "not available" in r.json()["detail"]
    r = client.post("/api/jobs", json={"op": "ts", "structures": [s["id"]]})
    assert r.status_code == 400


def test_incomplete_profile_is_refused_before_queueing(client):
    (s,) = _add(client, "C=CCOC=C")
    (e,) = _add(client, "C=CCCC=O")
    # The ASE engine with no calculator can only fail once the job loads it.
    client.put("/api/profiles/noasecalc",
               json={"text": 'engine_name = "ase"\nprogram = "terachem"\n[ase_engine_kwds]\ncalculator = ""\n'})
    r = client.post("/api/jobs", json={"op": "optimize", "structures": [s["id"]], "profile": "noasecalc"})
    assert r.status_code == 400 and "calculator" in r.json()["detail"]
    assert client.get("/api/state").json()["jobs"] == []
    # A path-method problem only blocks path searches, not an optimization.
    client.put("/api/profiles/dlf", json={"text": 'engine_name = "gxtb"\npath_min_method = "NEB-DLF"\n'})
    r = client.post("/api/jobs", json={"op": "optimize", "structures": [s["id"]], "profile": "dlf", "dry_run": True})
    assert r.status_code == 200, r.text
    r = client.post("/api/jobs", json={"op": "ts", "structures": [s["id"], e["id"]], "profile": "dlf",
                                       "dry_run": True})
    assert r.status_code == 400 and "DL-FIND" in r.json()["detail"]


@pytest.fixture
def quick_op(monkeypatch):
    """A registry entry whose command finishes in about a second: exercises
    the real subprocess pipeline (queue, logs, result collection) without
    any QM."""

    class Quick(ops.Params):
        name: str = ops.P("out.toml", "File name", cli="--output")

    def build(ctx, p):
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        return ["init", "--output", str(ctx.output_dir / p.name)]

    op = ops.Operation("quick", "Quick", "test", "structure", "Test", Quick, build)
    monkeypatch.setitem(ops.OPERATIONS, "quick", op)
    return op


def test_job_runs_as_subprocess_and_can_be_resumed(client, quick_op):
    (s,) = _add(client, WATER_XYZ)
    (job,) = client.post("/api/jobs", json={"op": "quick", "structures": [s["id"]], "profile": None}).json()
    assert job["status"] == "queued"
    done = _wait(client, job["id"])
    assert done["status"] == "done", done.get("error")
    files = client.get(f"/api/jobs/{job['id']}/files").json()
    assert "out.toml" in files
    log = client.get(f"/api/jobs/{job['id']}/log").text
    assert "$ mepd init" in log

    # retry = rerun into the same folder
    client.post(f"/api/jobs/{job['id']}/retry")
    assert _wait(client, job["id"])["status"] == "done"

    client.delete(f"/api/jobs/{job['id']}")
    assert client.get("/api/state").json()["jobs"] == []


def test_failed_job_reports_log_tail(client, monkeypatch):
    (s,) = _add(client, WATER_XYZ)
    op = ops.Operation("broken", "Broken", "test", "structure", "Test", None,
                       lambda ctx, p: ["no-such-command"])
    monkeypatch.setitem(ops.OPERATIONS, "broken", op)
    (job,) = client.post("/api/jobs", json={"op": "broken", "structures": [s["id"]]}).json()
    failed = _wait(client, job["id"])
    assert failed["status"] == "failed"
    assert "no-such-command" in failed["error"]


def _write_chain(fp: Path, xyz_frames: list[str], energies: list[float]) -> None:
    fp.write_text("".join(xyz_frames))
    np.savetxt(fp.with_suffix(".energies"), energies)


def test_import_existing_output_and_pull_irc_ends_into_graph(client, tmp_path):
    out = tmp_path / "mepd_ts_output"
    out.mkdir()
    _write_chain(out / "ts.xyz", [WATER_BENT_XYZ], [-76.30])
    _write_chain(out / "irc.xyz", [WATER_XYZ, WATER_XYZ.replace("0.758 0.000 0.504", "1.300 0.000 0.480", 1),
                                   WATER_BENT_XYZ], [-76.40, -76.30, -76.38])

    job = client.post("/api/jobs/import", json={"path": str(out)}).json()
    assert job["op"] == "tsopt" and job["status"] == "done" and job["external"]

    result = client.get(f"/api/jobs/{job['id']}/result").json()
    titles = [g["title"] for g in result["groups"]]
    assert titles == ["Transition states", "IRC paths"]
    irc = result["groups"][1]["entries"][0]
    assert irc["ts_index"] == 1
    assert irc["frames"][0]["energy_kcal"] == 0.0
    ts = result["groups"][0]["entries"][0]
    assert ts["barrier_kcal"] == pytest.approx(0.10 * 627.509474, abs=1e-3)

    r = client.post(f"/api/jobs/{job['id']}/import-entry", json={"entry": irc["id"], "frames": "endpoints"}).json()
    assert len(r["added"]) == 2 and r["edge"] is not None
    assert r["edge"]["origin"]["barrier_kcal"] == pytest.approx(0.10 * 627.509474, abs=1e-3)
    # Importing the same ends again reuses them (same connectivity + energy).
    r2 = client.post(f"/api/jobs/{job['id']}/import-entry", json={"entry": irc["id"], "frames": "endpoints"}).json()
    assert r2["added"] == [] and len(r2["reused"]) == 2
    assert len(client.get("/api/state").json()["workspace"]["edges"]) == 1

    # Deleting an imported job never touches the directory it points at.
    client.delete(f"/api/jobs/{job['id']}")
    assert (out / "ts.xyz").exists()


def test_references_tab_lists_every_cited_method(client):
    refs = client.get("/api/references").json()
    features = {g["feature"]: g for g in refs}
    assert {"Initial path", "Reaction network expansion", "Valley-ridge inflection"} <= set(features)
    cites = [c for g in refs for item in g["items"] for c in item["cite"]]
    assert all(c["text"] for c in cites)
    # DOIs become links; network expansion's come from the method's own REFERENCES.
    urls = {c["url"] for c in cites}
    assert "https://doi.org/10.1063/1.4878664" in urls        # IDPP
    assert "https://doi.org/10.1002/jcc.23271" in urls        # ZStruct
    guess = next(i for i in features["Reaction network expansion"]["items"] if i["what"] == "Guess geometry")
    assert guess["cite"] == [] and guess["note"]
    # ...and the operation forms explain features without inline citations.
    ops = {o["key"]: o for o in client.get("/api/state").json()["operations"]}
    assert "Zimmerman" not in ops["graph-enumeration"]["summary"]


def test_profiles_crud_and_toml_check(client):
    assert client.put("/api/profiles/fast", json={"text": 'engine_name = "gxtb"\n'}).status_code == 200
    assert "fast" in client.get("/api/state").json()["profiles"]
    r = client.put("/api/profiles/bad", json={"text": "engine_name = "})
    assert r.status_code == 400 and "TOML" in r.json()["detail"]
    assert client.put("/api/profiles/..%2Fescape", json={"text": ""}).status_code in (400, 404)
    client.delete("/api/profiles/fast")
    assert "fast" not in client.get("/api/state").json()["profiles"]


def test_interrupted_jobs_survive_a_restart(tmp_path, quick_op):
    ws = tmp_path / "ws"
    (ws / "profiles").mkdir(parents=True)
    (ws / "profiles" / "default.toml").write_text("")
    with TestClient(create_app(ws)) as c:
        (s,) = _add(c, WATER_XYZ)
        (job,) = c.post("/api/jobs", json={"op": "quick", "structures": [s["id"]]}).json()
        _wait(c, job["id"])
    # Simulate a server killed mid-run.
    import json
    fp = ws / "jobs" / job["id"] / "job.json"
    rec = json.loads(fp.read_text())
    rec["status"] = "running"
    fp.write_text(json.dumps(rec))
    with TestClient(create_app(ws)) as c:
        again = c.get(f"/api/jobs/{job['id']}").json()["job"]
        assert again["status"] == "interrupted"
        assert "resume" in again["error"]


def test_auto_minimize_skips_only_known_minima(client, tmp_path):
    """A sampled minimum imported from a result is already relaxed; paired
    with a SMILES structure (passed as xyz) the pair must still be minimized,
    and only a pair of known minima skips it."""
    out = tmp_path / "mepd_hessian_sample_output"
    out.mkdir()
    _write_chain(out / "unique.xyz", [WATER_XYZ, WATER_BENT_XYZ], [-76.40, -76.35])
    (out / "summary.json").write_text('{"seed_energy": -76.40}')
    job = client.post("/api/jobs/import", json={"path": str(out)}).json()
    assert job["op"] == "hessian-sample"
    a = client.post(f"/api/jobs/{job['id']}/import-entry", json={"entry": "min_0", "frames": "one", "frame": 0}).json()["added"][0]
    b = client.post(f"/api/jobs/{job['id']}/import-entry", json={"entry": "min_1", "frames": "one", "frame": 0}).json()["added"][0]
    # Minima from an imported (external) output: their level of theory is
    # unknown, so they are not trusted as minima at the job's level.
    assert a["optimized"] and a["level"] is None
    r = client.post("/api/jobs", json={"op": "ts", "structures": [a["id"], b["id"]], "profile": "default",
                                       "dry_run": True})
    assert "--minimize-ends" in r.json()[0]["argv"]


def test_optimize_on_add_puts_structures_at_the_workspace_level(client):
    """SMILES/xyz input is minimized at the workspace profile (a real g-xTB
    `mepd optimize` job) and tagged with that level; pair jobs at the same
    level then skip re-minimization, jobs at another level redo it."""
    (w1,) = client.post("/api/structures", json={"text": WATER_XYZ, "name": "w1"}).json()
    (w2,) = client.post("/api/structures", json={"text": WATER_BENT_XYZ, "name": "w2"}).json()
    assert w1["status"] == "optimizing" and not w1["optimized"]
    opt_jobs = [j for j in client.get("/api/state").json()["jobs"] if j["op"] == "optimize"]
    assert len(opt_jobs) == 2
    for j in opt_jobs:
        assert _wait(client, j["id"], timeout=120)["status"] == "done"
    for _ in range(50):  # results are applied right after the job finishes
        st = client.get("/api/state").json()
        recs = [st["workspace"]["structures"][x["id"]] for x in (w1, w2)]
        if all(r["status"] == "ready" for r in recs):
            break
        time.sleep(0.2)
    level = st["levels"]["default"]
    for r in recs:
        assert r["optimized"] and r["energy"] is not None and r["level"]["key"] == level["key"], r
        # Hessian-validated on entry (workspace default): a real minimum.
        assert r["validation"]["is_minimum"] and r["validation"]["min_frequency"] > 0, r
    assert st["level_profile"] == "default"

    def argv(profile):
        r = client.post("/api/jobs", json={"op": "ts", "structures": [w1["id"], w2["id"]], "profile": profile,
                                           "dry_run": True})
        assert r.status_code == 200, r.text
        return r.json()[0]["argv"]

    assert "--no-minimize-ends" in argv("default")
    client.put("/api/profiles/other", json={"text": 'engine_name = "gxtb"\n[gxtb_engine_kwds]\nmethod = "x"\n'})
    assert "--minimize-ends" in argv("other")  # another level of theory: re-minimize

    # Switching the workspace level makes them off-level; reoptimize queues jobs.
    client.put("/api/level", json={"profile": "other"})
    st = client.get("/api/state").json()
    assert st["level_profile"] == "other" and st["levels"]["other"]["key"] != level["key"]
    r = client.post("/api/structures/reoptimize", json={"structures": [w1["id"]]}).json()
    assert r[0]["op"] == "optimize" and r[0]["profile"] == "other"


def test_bulk_delete_and_downloads(client, tmp_path, quick_op):
    a, b, c = _add(client, WATER_XYZ + WATER_BENT_XYZ + HCN_XYZ)
    e1 = client.post("/api/edges", json={"source": a["id"], "target": b["id"]}).json()
    e2 = client.post("/api/edges", json={"source": b["id"], "target": c["id"]}).json()

    xyz = client.get("/api/structures-export", params={"ids": f"{a['id']},{c['id']}"}).text
    assert xyz.count("\n3\n") + xyz.startswith("3\n") == 2 and "charge=0 mult=1" in xyz

    r = client.post("/api/delete", json={"structures": [a["id"]], "edges": [e2["id"]]}).json()
    assert r["structures"] == [a["id"]] and set(r["edges"]) == {e1["id"], e2["id"]}  # a's edge goes too
    ws = client.get("/api/state").json()["workspace"]
    assert set(ws["structures"]) == {b["id"], c["id"]} and ws["edges"] == {}

    (job,) = client.post("/api/jobs", json={"op": "quick", "structures": [b["id"]]}).json()
    _wait(client, job["id"])
    import io
    import zipfile
    r = client.get(f"/api/jobs/{job['id']}/archive")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert "output/out.toml" in names and "job.json" in names and "stdout.log" in names


def test_sessions_new_open_and_isolation(client, tmp_path):
    (first,) = _add(client, WATER_XYZ)
    home = client.get("/api/sessions").json()
    assert home["current"].endswith("/ws")

    r = client.post("/api/sessions/new", json={"name": "second"})
    assert r.status_code == 200, r.text
    st = client.get("/api/state").json()
    assert st["workspace"]["root"].endswith("/second") and st["workspace"]["structures"] == {}
    assert st["profiles"] == ["default"]  # profiles carry over from the session we came from
    _add(client, "CCO")

    # A new session must not clobber an existing one.
    assert client.post("/api/sessions/new", json={"path": str(tmp_path / "ws")}).status_code == 400
    assert client.post("/api/sessions/open", json={"path": str(tmp_path / "nope")}).status_code == 400

    client.post("/api/sessions/open", json={"path": str(tmp_path / "ws")})
    st = client.get("/api/state").json()
    assert list(st["workspace"]["structures"]) == [first["id"]]
    listing = client.get("/api/sessions").json()["sessions"]
    assert [x["name"] for x in listing][:2] == ["ws", "second"]
    assert listing[0]["current"] and listing[1]["structures"] == 1


def test_ts_structures_are_never_minimized(client, tmp_path):
    out = tmp_path / "mepd_ts_output"
    out.mkdir()
    _write_chain(out / "ts.xyz", [WATER_BENT_XYZ], [-76.30])
    job = client.post("/api/jobs/import", json={"path": str(out)}).json()
    ts = client.post(f"/api/jobs/{job['id']}/import-entry", json={"entry": "ts", "frames": "one", "frame": 0}).json()["added"][0]
    assert ts["role"] == "ts" and ts["name"].endswith("[TS]")
    r = client.post("/api/structures/reoptimize", json={"structures": [ts["id"]]})
    assert r.status_code == 400 and "never minimized" in r.json()["detail"]
    r = client.post("/api/jobs", json={"op": "optimize", "structures": [ts["id"]], "dry_run": True})
    assert r.status_code == 400 and "not minimized" in r.json()["detail"]


def test_legacy_ts_records_are_recognized_by_name():
    from mepd.web.workspace import is_ts

    assert is_ts({"name": "C#N [TS]"})  # written before `role` existed
    assert not is_ts({"name": "C#N [TS]", "role": "minimum"})
    assert is_ts({"name": "x", "role": "ts"})


def test_saddle_point_is_flagged_not_trusted(client):
    """Eclipsed ethane optimizes onto the rotational saddle; the Hessian check
    catches it and the escalating rescue (0.1 fails, 0.3 bohr works) relaxes it
    to staggered."""
    eclipsed = """8
eclipsed ethane
C 0.000000 0.000000 0.765000
C 0.000000 0.000000 -0.765000
H 1.018000 0.000000 1.160000
H -0.509000 0.881614 1.160000
H -0.509000 -0.881614 1.160000
H 1.018000 0.000000 -1.160000
H -0.509000 0.881614 -1.160000
H -0.509000 -0.881614 -1.160000
"""
    (s,) = client.post("/api/structures", json={"text": eclipsed, "name": "ethane"}).json()
    job = [j for j in client.get("/api/state").json()["jobs"] if j["op"] == "optimize"][0]
    assert "--validate-minima-with-hessian" in job["argv"]
    # starts at mepd's 0.1 bohr; the rescue itself escalates to 0.3 (which is what frees eclipsed ethane)
    assert job["argv"][job["argv"].index("--hessian-minima-rescue-displacement") + 1] == "0.1"
    assert _wait(client, job["id"], timeout=180)["status"] == "done"
    for _ in range(50):
        rec = client.get("/api/state").json()["workspace"]["structures"][s["id"]]
        if rec["status"] != "optimizing":
            break
        time.sleep(0.2)
    assert rec["status"] == "ready" and rec["optimized"], rec
    assert rec["validation"]["rescued"] and rec["validation"]["min_frequency"] > 0

    # With the check off, the same input stays on the saddle unflagged --
    # which is exactly why it is on by default.
    client.put("/api/level", json={"validate_minima": False})
    assert client.get("/api/state").json()["validate_minima"] is False
    (s2,) = client.post("/api/structures", json={"text": eclipsed, "name": "ethane2"}).json()
    job2 = [j for j in client.get("/api/state").json()["jobs"] if j["op"] == "optimize" and s2["id"] in j["targets"]["structures"]][0]
    assert "--no-validate-minima-with-hessian" in job2["argv"]


def test_app_modules_are_revalidated_vendor_cached(client):
    """Stale cached JS modules next to fresh ones fail to link -> blank page."""
    assert client.get("/static/js/api.js").headers["cache-control"] == "no-cache"
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert "max-age" in client.get("/static/vendor/preact-htm.module.js").headers["cache-control"]


def test_hessian_sample_validates_minima_and_marks_rejects(client, tmp_path):
    (s,) = _add(client, WATER_XYZ)
    (cmd,) = client.post("/api/jobs", json={"op": "hessian-sample", "structures": [s["id"]], "dry_run": True}).json()
    assert "--validate-minima-with-hessian" in cmd["argv"]
    assert cmd["argv"][cmd["argv"].index("--hessian-minima-rescue-displacement") + 1] == "0.1"
    (cmd,) = client.post("/api/jobs", json={"op": "hessian-global", "structures": [s["id"]], "dry_run": True}).json()
    assert "--validate-minima-with-hessian" in cmd["argv"]

    out = tmp_path / "mepd_hessian_sample_output"
    out.mkdir()
    _write_chain(out / "unique.xyz", [WATER_XYZ], [-76.40])
    _write_chain(out / "rejected.xyz", [WATER_BENT_XYZ], [-76.35])
    (out / "summary.json").write_text(json.dumps({
        "seed_energy": -76.40, "hessian_validation": {"enabled": True},
        "unique_minima_validation": [{"is_minimum": True, "min_frequency": 1600.0, "rescued": False, "validation": "ok"}],
        "rejected_minima_validation": [{"is_minimum": False, "min_frequency": -50.0, "rescued": False, "validation": "saddle"}],
    }))
    job = client.post("/api/jobs/import", json={"path": str(out)}).json()
    r = client.get(f"/api/jobs/{job['id']}/result").json()
    assert "Hessian-validated, 1 rejected" in r["headline"]
    kinds = {g["kind"]: g for g in r["groups"]}
    assert "Hessian ✓" in kinds["minima"]["entries"][0]["note"]
    # min_0 is the seed's own molecule: it joins that node as a conformer.
    good = client.post(f"/api/jobs/{job['id']}/import-entry", json={"entry": "min_0", "frames": "one", "frame": 0}).json()["reused"][0]
    assert good["id"] == s["id"] and len(good["conformers"]) == 2
    bad = client.post(f"/api/jobs/{job['id']}/import-entry", json={"entry": "rejected_0", "frames": "one", "frame": 0}).json()["added"][0]
    assert good["optimized"] and good["validation"]["is_minimum"]
    assert not bad["optimized"]  # a rejected structure is never trusted as a minimum


def test_gsm_nnodes_ignored_when_seeded_is_surfaced(client):
    """GSM seeded from the geodesic path uses gi_inputs.nimages nodes; an
    explicit nnodes is silently ignored by mepd -- the UI must say so."""
    client.put("/api/profiles/gsm20", json={"text": 'engine_name = "gxtb"\npath_min_method = "GSM"\n[path_min_inputs]\nnnodes = 20\n'})
    client.put("/api/profiles/gsm20grow", json={"text": 'path_min_method = "GSM"\n[path_min_inputs]\nnnodes = 20\nseed_with_geodesic_interpolation = false\n'})
    client.put("/api/profiles/gsm20img", json={"text": 'path_min_method = "GSM"\n[gi_inputs]\nnimages = 20\n'})
    sums = client.get("/api/state").json()["path_summaries"]
    assert "10-node string" in sums["gsm20"]["text"] and "nnodes = 20 is ignored" in sums["gsm20"]["warnings"][0]
    assert sums["gsm20grow"]["text"].startswith("GSM · grows its own string to 20 nodes") and not sums["gsm20grow"]["warnings"]
    assert "20-node string" in sums["gsm20img"]["text"]


def test_auth_token_login(tmp_path, monkeypatch):
    """With a token, everything but /login needs the session cookie; the
    login link sets it and redirects so the token leaves the URL."""
    ws = tmp_path / "ws_auth"
    (ws / "profiles").mkdir(parents=True)
    (ws / "profiles" / "default.toml").write_text("")
    token = "t" * 32
    with TestClient(create_app(ws, auth_token=token), follow_redirects=False) as c:
        assert c.get("/api/state").status_code == 401
        assert "access token" in c.get("/").text                         # login page, not the app
        assert c.get("/static/js/app.js").status_code == 401
        assert c.post("/login", data={"token": "wrong"}).status_code == 401
        r = c.get(f"/login?token={token}")
        assert r.status_code == 303 and r.headers["location"] == "/"
        cookie = r.cookies.get("mepd_session")
        assert cookie and token not in cookie                            # cookie is an HMAC, not the token
        assert "samesite=lax" in r.headers["set-cookie"].lower() and "httponly" in r.headers["set-cookie"].lower()
        c.cookies.set("mepd_session", cookie)
        st = c.get("/api/state")
        assert st.status_code == 200 and st.json()["auth"] is True
        c.cookies.clear()
        assert c.get("/api/state", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        # throttling after repeated failures
        for _ in range(10):
            c.post("/login", data={"token": "nope"})
        assert c.post("/login", data={"token": token}).status_code == 429


def test_token_file_is_private_and_persistent(tmp_path, monkeypatch):
    import stat

    from mepd.web.auth import load_or_create_token, token_path

    t1 = load_or_create_token()
    assert load_or_create_token() == t1                                    # survives restarts
    assert stat.S_IMODE(token_path().stat().st_mode) == 0o600
    assert load_or_create_token(rotate=True) != t1


@pytest.fixture
def demo_app(tmp_path):
    from mepd.web.demo import DemoPolicy

    root = tmp_path / "demo"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    policy = DemoPolicy(max_atoms=10, max_active_jobs=1, max_sessions=2)
    return root, create_app(root, demo=policy, demo_password="pw-for-tests")


def _visitor(app):
    c = TestClient(app, follow_redirects=False)
    c.__enter__()
    r = c.post("/login", data={"password": "pw-for-tests"})
    assert r.status_code == 303
    c.cookies.set("mepd_visitor", r.cookies.get("mepd_visitor"))
    return c


def test_demo_visitors_are_isolated_and_locked_down(demo_app, tmp_path):
    root, app = demo_app
    anon = TestClient(app, follow_redirects=False)
    with anon:
        assert anon.get("/api/state").status_code == 401
        assert anon.post("/login", data={"password": "wrong"}).status_code == 401
        alice, bob = _visitor(app), _visitor(app)
        try:
            st = alice.get("/api/state").json()
            assert st["demo"]["max_atoms"] == 10 and st["workspace"]["root"] == "main"
            assert "/" not in st["workspace"]["root"]            # no server paths
            (a1,) = alice.post("/api/structures", json={"text": WATER_XYZ, "optimize": False}).json()
            assert bob.get("/api/state").json()["workspace"]["structures"] == {}   # private
            assert bob.get(f"/api/structures/{a1['id']}/xyz").status_code == 400    # can't reach alice's

            # locked down
            assert alice.put("/api/profiles/default", json={"text": 'engine_name = "x"'}).status_code == 403
            assert alice.post("/api/profiles/validate", json={"text": ""}).status_code == 403
            assert alice.post("/api/jobs/import", json={"path": str(tmp_path)}).status_code == 403
            r = alice.post("/api/sessions/open", json={"path": str(tmp_path)})
            assert r.status_code == 400
            r = alice.post("/api/sessions/new", json={"path": str(tmp_path / "x")})
            assert r.status_code == 400

            # limits
            r = alice.post("/api/structures", json={"text": "CCCCCCCC", "optimize": False})
            assert r.status_code == 400 and "10 atoms" in r.json()["detail"]
            r = alice.post("/api/jobs", json={"op": "channels", "structures": [a1["id"], a1["id"]],
                                              "params": {"workers": 8}, "dry_run": True})
            assert r.status_code == 400 and "demo limit" in r.json()["detail"]

            # sessions by name, capped
            assert alice.post("/api/sessions/new", json={"name": "second"}).status_code == 200
            assert alice.get("/api/state").json()["workspace"]["root"] == "second"
            assert alice.post("/api/sessions/new", json={"name": "third"}).status_code == 400   # max 2
            names = [x["path"] for x in alice.get("/api/sessions").json()["sessions"]]
            assert set(names) == {"main", "second"}
            assert alice.post("/api/sessions/open", json={"path": "main"}).status_code == 200
            assert list(alice.get("/api/state").json()["workspace"]["structures"]) == [a1["id"]]
            # on disk: two visitors, each under visitors/<id>/
            assert len(list((root / "visitors").iterdir())) == 2
        finally:
            alice.__exit__(None, None, None)
            bob.__exit__(None, None, None)


def test_demo_visitor_cookie_is_signed(demo_app):
    root, app = demo_app
    with TestClient(app, follow_redirects=False) as c:
        c.cookies.set("mepd_visitor", "someoneelse1234.deadbeef")
        assert c.get("/api/state").status_code == 401


def test_long_poll_fallback_delivers_events(client):
    """/api/poll: same events as SSE, for proxies that buffer streams."""
    first = client.get("/api/poll").json()
    assert first["events"] == [{"event": "hello", "data": {}}] and first["cid"]
    cid = first["cid"]
    _add(client, WATER_XYZ)
    r = client.get("/api/poll", params={"cid": cid, "wait": 5}).json()
    assert r["cid"] == cid and any(e["event"] == "workspace" for e in r["events"])
    empty = client.get("/api/poll", params={"cid": cid, "wait": 0.2}).json()
    assert empty["events"] == []


def test_demo_form_defaults_respect_limits(demo_app):
    """Submitting any operation with its (served) defaults must never hit a
    demo limit -- that silently stalled Quick start's exploration."""
    root, app = demo_app
    with TestClient(app, follow_redirects=False) as anon:
        c = _visitor(app)
        try:
            ops = {o["key"]: o for o in c.get("/api/state").json()["operations"]}
            hs = ops["hessian-sample"]["schema"]["properties"]["max_candidates"]
            assert hs["default"] == 60 and hs["maximum"] == 60
            assert ops["channels"]["schema"]["properties"]["workers"]["default"] == 2
            (s1,) = c.post("/api/structures", json={"text": WATER_XYZ, "optimize": False}).json()
            expand = ops["graph-enumeration"]
            assert expand["available"] and expand["schema"]["properties"]["max_products"]["default"] == 30
            for key in ("hessian-sample", "hessian-global", "graph-enumeration"):
                props = ops[key]["schema"]["properties"]
                params = {k: v.get("default") for k, v in props.items()}
                r = c.post("/api/jobs", json={"op": key, "structures": [s1["id"]], "params": params, "dry_run": True})
                assert r.status_code == 200, (key, r.text)
        finally:
            c.__exit__(None, None, None)


def test_demo_forces_https_behind_proxy(demo_app):
    root, app = demo_app
    with TestClient(app, follow_redirects=False) as c:
        r = c.get("/login", headers={"x-forwarded-proto": "http"})
        assert r.status_code == 301 and r.headers["location"].startswith("https://")
        r = c.get("/login", headers={"x-forwarded-proto": "https"})
        assert r.status_code == 200 and "max-age" in r.headers["strict-transport-security"]
        r = c.get("/login", headers={"x-forwarded-proto": "https", "cf-visitor": '{"scheme":"http"}'})
        assert r.status_code == 301                                   # Cloudflare's own signal wins


def test_barrier_needs_an_irc_connecting_start_and_end(tmp_path):
    """The edge barrier must come from a TS whose IRC connects the path's two
    ends -- not from a saddle the TS search slid into (e.g. a conformer
    rotation of the reactant, whose IRC returns to the reactant)."""
    from mepd.web.results import collect_ts, summarize

    ethane_a = """8
ethane
C 0.000 0.000 0.765
C 0.000 0.000 -0.765
H 1.018 0.000 1.160
H -0.509 0.882 1.160
H -0.509 -0.882 1.160
H -1.018 0.000 -1.160
H 0.509 -0.882 -1.160
H 0.509 0.882 -1.160
"""
    # "product": H moved from C1 to C2 (different connectivity)
    ethane_b = ethane_a.replace("H 1.018 0.000 1.160", "H 0.900 0.000 -1.600", 1)
    out = tmp_path / "run"
    out.mkdir()
    _write_chain(out / "mep_output.xyz", [ethane_a, ethane_a, ethane_b], [-79.80, -79.60, -79.70])
    # the only TS found: a conformer change of the reactant (both IRC ends = reactant)
    _write_chain(out / "ts_leaf_0.xyz", [ethane_a], [-79.795])
    _write_chain(out / "ts_leaf_0_irc.xyz", [ethane_a, ethane_a, ethane_a], [-79.80, -79.795, -79.80])
    r = collect_ts(out, 0, 1)
    assert r["barrier_verified"] is False
    assert r["barrier_kcal"] == pytest.approx(0.20 * 627.509474, abs=1e-2)       # path max, not the 3 kcal saddle
    assert "not verified" in r["headline"]
    other = {g["kind"]: g for g in r["groups"]}["ts_other"]["entries"][0]
    assert "conformer change" in other["note"]
    assert summarize(r)["barrier_verified"] is False

    # now a TS whose IRC does connect reactant and product -> verified
    _write_chain(out / "ts_leaf_0.xyz", [ethane_a], [-79.62])
    _write_chain(out / "ts_leaf_0_irc.xyz", [ethane_a, ethane_a, ethane_b], [-79.80, -79.62, -79.70])
    r = collect_ts(out, 0, 1)
    assert r["barrier_verified"] is True
    assert r["barrier_kcal"] == pytest.approx(0.18 * 627.509474, abs=1e-2)
    assert "IRC-verified" in r["headline"]


def test_bulk_add_to_graph(client, tmp_path):
    """Several sampled minima in one request; already-added ones are reused."""
    out = tmp_path / "mepd_hessian_sample_output"
    out.mkdir()
    frames = [WATER_XYZ, WATER_BENT_XYZ, WATER_XYZ.replace("0.758 0.000 0.504", "0.700 0.100 0.520", 1)]
    _write_chain(out / "unique.xyz", frames, [-76.40, -76.38, -76.35])
    (out / "summary.json").write_text(json.dumps({"seed_energy": -76.40}))
    job = client.post("/api/jobs/import", json={"path": str(out)}).json()
    r = client.post(f"/api/jobs/{job['id']}/import-entries", json={"entries": ["min_0", "min_2"]}).json()
    # Two conformers of water: one node holding both.
    assert len(r["added"]) == 1 and len(r["reused"]) == 1
    assert len(r["added"][0]["conformers"]) == 2 or len(client.get("/api/state").json()["workspace"]["structures"][r["added"][0]["id"]]["conformers"]) == 2
    r = client.post(f"/api/jobs/{job['id']}/import-entries", json={"entries": ["min_0", "min_1"]}).json()
    assert len(r["added"]) == 1 and len(r["reused"]) == 1          # min_0 was already in the Graph
    assert len(client.get("/api/state").json()["workspace"]["structures"]) == 2
    assert client.post(f"/api/jobs/{job['id']}/import-entries", json={"entries": ["nope"]}).status_code == 404


def test_overflowing_client_gets_resync_instead_of_silent_loss():
    import asyncio

    from mepd.web.jobs import Broadcaster

    async def run():
        bus = Broadcaster()
        bus.bind(asyncio.get_running_loop())
        q = bus.subscribe()
        for i in range(1005):          # queue holds 1000
            bus.publish("progress", {"i": i})
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        return items

    items = asyncio.run(run())
    assert ("resync", {}, None) in items and len(items) < 1000


def test_gsm_early_stop_is_off_by_default_and_flagged_when_on():
    from mepd.inputs import RunInputs
    from mepd.web.workspace import path_summary

    assert RunInputs(path_min_method="GSM").path_min_inputs.early_stop_on_minima is False
    assert "early stop off" in path_summary('path_min_method = "GSM"\n')["text"]
    on = path_summary('path_min_method = "GSM"\n[path_min_inputs]\nearly_stop_on_minima = true\n')
    assert "early stop on" in on["text"] and any("early_stop_on_minima" in w for w in on["warnings"])


def test_ts_uses_minimized_geometries_not_the_typed_smiles(client):
    (s,) = _add(client, "C=CCOC=C")
    (e,) = _add(client, "C=CCCC=O")
    ws = client.app.state.sessions.current.ws
    # Once an endpoint holds a minimized geometry, re-embedding its SMILES
    # would throw that minimum away.
    ws._data["structures"][s["id"]]["level"] = {"profile": "default", "key": "k", "label": "test"}
    (ts,) = client.post("/api/jobs", json={"op": "ts", "structures": [s["id"], e["id"]], "params": {},
                                           "profile": "default", "dry_run": True}).json()
    argv = ts["argv"]
    assert argv[2].endswith("start.xyz") and argv[4].endswith("end.xyz")


def test_structure_that_reacts_on_minimization_is_renamed_and_flagged(client):
    (s,) = _add(client, "[OH-].CBr")
    assert s["name"] == "[OH-].CBr"
    ws = client.app.state.sessions.current.ws
    # Minimized to methanol + bromide (the gas-phase SN2 product).
    product = client.post("/api/structures", json={"text": "CO.[Br-]", "optimize": False}).json()[0]
    xyz = (ws.structures_dir / f"{product['id']}.xyz").read_text()
    from mepd.web import chem

    (geom,) = chem.structures_from_xyz_text(xyz)
    rec = ws.replace_geometry(s["id"], geom, energy=-1.0, level={"profile": "default", "key": "k", "label": "test"})
    assert chem.canonical_key(rec["smiles"]) == chem.canonical_key("CO.[Br-]")
    assert rec["name"] == rec["smiles"]  # the automatic name follows the new species
    assert rec["reacted"]["from"] == "[OH-].CBr"
    assert "reacted" in rec["status_error"]
    # A name the user chose is kept.
    ws._data["structures"][product["id"]]["name"] = "my product"
    (geom2,) = chem.structures_from_xyz_text((ws.structures_dir / f"{s['id']}.xyz").read_text())
    rec2 = ws.replace_geometry(product["id"], geom2, energy=-1.0, level={"profile": "default", "key": "k", "label": "test"})
    assert rec2["name"] == "my product" and "reacted" not in rec2


def test_job_revisions_only_increase(client, quick_op):
    # The browser keeps whichever copy of a job has the larger rev, so a
    # full-state reload racing a live 'done' event can't roll it back.
    (s,) = _add(client, WATER_XYZ)
    (job,) = client.post("/api/jobs", json={"op": "quick", "structures": [s["id"]], "profile": None}).json()
    done = _wait(client, job["id"])
    assert done["status"] == "done"
    assert done["rev"] > job["rev"] > 0


def _vri_folder(root: Path) -> Path:
    """A small, synthetic `mepd discovery vri` output: one bifurcating branch."""
    d = root / "vri_out"
    d.mkdir()
    branch = {"verdict": "bifurcation",
              "vrt": {"s": 1.2, "rel_ts1_kcal_mol": -8.0},
              "products": {"p1_rel_ts1_kcal_mol": -30.0, "p2_rel_ts1_kcal_mol": -31.5,
                           "ts2_rel_ts1_kcal_mol": -20.0, "ts2_verified": True}}
    (d / "summary.json").write_text(json.dumps({
        "verdict": "bifurcation", "ts1_energy": -76.0, "ts1_n_imaginary": 1,
        "branches": {"forward": branch, "reverse": {"verdict": "no_vrt"}},
        "ts1_candidates": [{"label": "input", "dir": str(d)}]}))
    (d / "projected_freqs.json").write_text(json.dumps({"ts_index": 1, "branches": {}}))
    (d / "irc.xyz").write_text(WATER_BENT_XYZ + WATER_XYZ + WATER_BENT_XYZ)
    for name in ("p1_forward", "p2_forward", "ts2_forward", "vrt_forward"):
        (d / f"{name}.xyz").write_text(WATER_XYZ)
    (d / "checks_forward.json").write_text(json.dumps({
        "branch": "forward", "vri": {"converged": True, "iterations": 5},
        "basin": {"counts": {"P1": 7, "P2": 5}}, "trajectories": {"counts": {"P1": 30, "P2": 20}, "n": 50}}))
    return d


def _skip_unless_available(client, *keys):
    """VRI operations switch on only with an mepd whose VRI CLI has every flag they use."""
    ops = {o["key"]: o for o in client.get("/api/state").json()["operations"]}
    for key in keys:
        if not ops[key]["available"]:
            pytest.skip(f"{key}: {ops[key]['unavailable_reason']}")


def test_operations_whose_cli_is_missing_are_shown_unavailable(monkeypatch):
    from mepd.web import operations as ops

    monkeypatch.setattr(ops, "_cli_options", lambda path: frozenset({"--charge"}))
    problem = ops.OPERATIONS["vri"].cli_problem()
    assert problem and "lacks" in problem and "--stride" in problem
    assert not ops.OPERATIONS["vri"].describe()["available"]
    monkeypatch.setattr(ops, "_cli_options", lambda path: None)
    assert "no `mepd discovery vri-check` command" in ops.OPERATIONS["vri-check"].cli_problem()
    with pytest.raises(ops.WorkspaceError):
        ops.get_operation("vri-check")
    assert ops.OPERATIONS["ts"].cli_problem() is None   # operations without a declared command are unaffected


def _fake_ts_job(client, sids, eids=(), barrier=12.3):
    """A finished TS search on this pair whose IRC-verified TS is recorded
    (what results.collect_ts writes: summary.route_ts + route_ts.xyz)."""
    jobs = client.app.state.sessions.current.jobs
    jid = "j_fakets" + str(len(jobs.jobs))
    (jobs.job_dir(jid)).mkdir(parents=True)
    (jobs.job_dir(jid) / "route_ts.xyz").write_text(WATER_BENT_XYZ)
    jobs.jobs[jid] = {"id": jid, "op": "ts", "status": "done", "title": "TS", "created": 0, "finished": 1,
                      "targets": {"structures": list(sids), "edges": list(eids)}, "output_dir": "/nonexistent",
                      "summary": {"headline": "", "barrier_kcal": barrier, "barrier_verified": True, "counts": {},
                                  "route_ts": {"label": "ts_leaf_0", "barrier_kcal": barrier}}}
    return jid


def test_vri_runs_on_an_edge_from_its_verified_ts(client):
    (a,) = _add(client, WATER_XYZ)
    (b,) = _add(client, WATER_BENT_XYZ)
    edge = client.post("/api/edges", json={"source": a["id"], "target": b["id"]}).json()
    body = {"op": "vri", "edges": [edge["id"]], "params": {"find_ts2": False}, "dry_run": True}
    _skip_unless_available(client, "vri")
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 400 and "no transition state connects" in r.json()["detail"]

    _fake_ts_job(client, [a["id"], b["id"]], [edge["id"]], barrier=20.0)
    _fake_ts_job(client, [b["id"], a["id"]], [], barrier=9.5)     # same pair, other direction
    (job,) = client.post("/api/jobs", json=body).json()
    argv = job["argv"]
    assert argv[:2] == ["discovery", "vri"] and argv[2].endswith("inputs/ts1.xyz")
    assert "--skip-ts2" in argv and argv[argv.index("--stride") + 1] == "2" and argv[-2] == "--output"
    ops = {o["key"]: o for o in client.get("/api/state").json()["operations"]}
    assert ops["vri"]["target"] == "pair" and ops["vri"]["needs_route_ts"]
    assert ops["vri-check"]["target"] == "job" and ops["vri-check"]["source_ops"] == ["vri"]


def test_ts_result_records_the_route_ts(tmp_path):
    from mepd.web import results

    job = {"id": "j_x", "op": "ts", "status": "done", "finished": 1.0, "output_dir": str(tmp_path / "out")}
    fake = {"headline": "h", "groups": [], "summary": [], "warnings": [], "barrier_kcal": 5.0,
            "route_ts": {"label": "ts_leaf_0", "barrier_kcal": 5.0, "xyz": WATER_XYZ}}
    orig = results.collect
    results.collect = lambda j, job_dir=None: dict(fake)
    try:
        out = results.collect_cached(job, tmp_path / "jobdir")
    finally:
        results.collect = orig
    assert (tmp_path / "jobdir" / "route_ts.xyz").read_text() == WATER_XYZ
    assert results.summarize(out)["route_ts"] == {"label": "ts_leaf_0", "barrier_kcal": 5.0}


def test_vri_follow_ups_work_in_the_source_folder(client, tmp_path):
    _skip_unless_available(client, "vri-check", "vri-surface")
    d = _vri_folder(tmp_path)
    src = client.post("/api/jobs/import", json={"path": str(d), "op": "vri"}).json()
    (chk,) = client.post("/api/jobs", json={"op": "vri-check", "params": {"trajectories": 20},
                                            "source_job": src["id"], "dry_run": True}).json()
    assert chk["argv"][:3] == ["discovery", "vri-check", str(d)]
    assert chk["argv"][chk["argv"].index("--trajectories") + 1] == "20"
    assert chk["output_dir"] == str(d) and chk["source_job"] == src["id"]
    (srf,) = client.post("/api/jobs", json={"op": "vri-surface", "source_job": src["id"], "dry_run": True}).json()
    assert srf["argv"][:3] == ["discovery", "vri-surface", str(d)]

    # A follow-up needs a finished job of the right kind.
    (s,) = _add(client, WATER_XYZ)
    r = client.post("/api/jobs", json={"op": "vri-check", "source_job": "j_nope", "dry_run": True})
    assert r.status_code == 400
    other = client.post("/api/jobs/import", json={"path": str(d), "op": "tsopt"}).json()
    r = client.post("/api/jobs", json={"op": "vri-check", "source_job": other["id"], "dry_run": True})
    assert r.status_code == 400 and "follows up on vri" in r.json()["detail"]


def test_vri_result_reads_verdicts_products_and_checks(tmp_path):
    from mepd.web.results import collect_vri

    r = collect_vri(_vri_folder(tmp_path), 0, 1)
    assert r["headline"].startswith("Post-TS bifurcation") and "forward branch" in r["headline"]
    assert {k: r["vri"][k] for k in ("verdict", "checked", "surface")} == {
        "verdict": "bifurcation", "checked": True, "surface": False}
    kinds = {g["kind"]: [e["label"] for e in g["entries"]] for g in r["groups"]}
    assert kinds["ts"] == ["TS1", "TS2 (forward)"]
    assert kinds["minima"] == ["P1 (forward)", "P2 (forward)"]
    assert kinds["points"] == ["VRT (forward)"]
    assert kinds["irc"] == ["IRC through TS1"]
    summary = {s["label"]: s["value"] for s in r["summary"]}
    assert summary["Basin test (forward)"] == "P1 7 · P2 5"
    assert summary["Reverse"] == "no valley-ridge transition"


def _cli_options(argv: list[str]) -> set[str]:
    """Every option the mepd CLI command that `argv` runs accepts."""
    import click
    import typer

    from mepd.cli import app

    cmd = typer.main.get_command(app)
    words = list(argv)
    while hasattr(cmd, "get_command") and words and not words[0].startswith("-"):
        sub = cmd.get_command(click.Context(cmd), words[0])
        if sub is None:
            break
        cmd, words = sub, words[1:]
    opts = set()
    for prm in cmd.params:
        opts.update(getattr(prm, "opts", []))
        opts.update(getattr(prm, "secondary_opts", []))
    return opts


def test_every_web_operation_emits_only_real_cli_flags(client, tmp_path):
    # The web UI drives mepd through its CLI; a renamed or removed flag
    # would otherwise only show up as a failed job (it happened: VRI's
    # --enumerate-products). Build every operation's command with default
    # parameters and with every on/off option flipped, and check each flag.
    from qcdata import Structure

    ws = client.app.state.sessions.current.ws
    (a,) = _add(client, "C=CCOC=C")
    (b,) = _add(client, "C=CCCC=O")
    ts = ws.add_structure(Structure.from_xyz(WATER_BENT_XYZ), name="TS1", origin={"kind": "xyz"}, role="ts")
    _fake_ts_job(client, [a["id"], b["id"]])      # so edge operations that need a TS (VRI) can build
    src = client.post("/api/jobs/import", json={"path": str(_vri_folder(tmp_path)), "op": "vri"}).json()
    client.post("/api/design/new", json={"smiles": "CCO"})   # so the Design tab's minimization can build
    ops = client.get("/api/state").json()["operations"]
    checked = 0
    for op in ops:
        if not op["available"]:
            continue
        body = {"op": op["key"], "dry_run": True, "profile": "default"}
        if op["target"] == "pair":
            body["structures"] = [a["id"], b["id"]]
        elif op["target"] == "set":
            body["structures"] = [a["id"], b["id"]]
        elif op["target"] == "job":
            body["source_job"] = src["id"]
        elif op["target"] == "design":
            pass
        else:
            body["structures"] = [ts["id"] if op.get("structure_role") == "ts" else a["id"]]
        props = (op["schema"] or {}).get("properties", {})
        flipped = {k: not v.get("default") for k, v in props.items() if v.get("type") == "boolean"}
        for params in ({}, flipped):
            r = client.post("/api/jobs", json={**body, "params": params})
            if r.status_code == 400 and params:
                continue  # an invalid combination (e.g. IRC without TS optimization)
            assert r.status_code == 200, (op["key"], r.text)
            for job in r.json():
                known = _cli_options(job["argv"])
                unknown = [w for w in job["argv"] if w.startswith("--") and w not in known]
                assert not unknown, f"{op['key']}: mepd {' '.join(job['argv'][:2])} has no {unknown}"
                checked += 1
    assert checked >= 10


def test_atom_mapping_metric_choices_track_the_metric_registry(client):
    """The UI spells no metric out by hand any more: the list it offered was
    duplicated per form and had already drifted a metric behind the registry,
    leaving `endpoint-rmsd` unreachable from the browser for a while."""
    from mepd.atom_mapping_metrics import METRICS
    from mepd.web.operations import ChannelsParams, TsParams

    for model in (ChannelsParams, TsParams):
        choices = model.model_json_schema()["properties"]["atom_mapping_metric"]["enum"]
        assert tuple(choices) == METRICS, f"{model.__name__} drifted from METRICS"

    # ...and the odd one out survives the trip to a command line.
    (s,) = _add(client, "C=CCOC=C")
    (e,) = _add(client, "C=CCCC=O")
    r = client.post("/api/jobs", json={
        "op": "channels", "structures": [s["id"], e["id"]], "edges": [],
        "params": {"atom_mapping_metric": "endpoint-rmsd"},
        "profile": "default", "dry_run": True})
    assert r.status_code == 200, r.text
    argv = r.json()[0]["argv"]
    assert argv[argv.index("--atom-mapping-metric") + 1] == "endpoint-rmsd"


def test_live_species_are_spawned_into_the_graph_connected_to_their_parent(tmp_path):
    """A running network expansion's live/events.jsonl: each new species
    becomes a node joined to the one it came from, as it is reported, once."""
    from mepd.web import chem
    from mepd.web.jobs import JobManager
    from mepd.web.workspace import Workspace

    class Bus:
        def __init__(self):
            self.events = []

        def publish(self, event, data):
            self.events.append(event)

    ws = Workspace(tmp_path / "ws")
    (water,) = chem.structures_from_xyz_text(WATER_XYZ, 0, 1)
    seed = ws.add_structure(water, name="seed", origin={"kind": "smiles"})
    bus = Bus()
    jobs = JobManager(ws, bus)
    job = {"id": "j_live", "op": "graph-enumeration", "targets": {"structures": [seed["id"]], "edges": []},
           "charge": 0, "multiplicity": 1, "level": None}
    live = jobs.job_dir("j_live") / "live"
    live.mkdir(parents=True)
    events = [
        {"event": "species", "index": 1, "parent": 0, "xyz": WATER_BENT_XYZ, "energy_hartree": -1.0, "caption": "+O0–H1"},
        {"event": "species", "index": 2, "parent": 1, "xyz": HCN_XYZ, "energy_hartree": -2.0,
         "validation": {"is_minimum": True}},
        {"event": "reaction", "source": 0, "target": 2, "caption": ""},
        # Another geometry of the seed's own molecule: a conformer of the seed, no new node or edge.
        {"event": "species", "index": 3, "parent": 1, "xyz": WATER_XYZ.replace("0.504", "0.604"), "energy_hartree": -3.0},
    ]
    fp = live / "events.jsonl"
    fp.write_text(json.dumps(events[0]) + "\n" + json.dumps(events[1])[:20])   # second line half-written
    jobs._adopt_live_events(job)
    snap = ws.snapshot()
    assert len(snap["structures"]) == 2 and len(snap["edges"]) == 1 and bus.events == ["workspace"]
    (edge,) = snap["edges"].values()
    assert edge["source"] == seed["id"] and edge["origin"]["proposed"]
    first = snap["structures"][edge["target"]]
    assert first["origin"]["parent"] == seed["id"] and first["origin"]["entry"] == "min_1"

    fp.write_text(json.dumps(events[0]) + "\n" + "\n".join(json.dumps(e) for e in events[1:]) + "\n")
    jobs._adopt_live_events(job)
    jobs._adopt_live_events(job)          # nothing new: nothing added twice
    snap = ws.snapshot()
    assert len(snap["structures"]) == 3
    pairs = {(e["source"], e["target"]) for e in snap["edges"].values()}
    second = job["live_nodes"]["2"]
    assert pairs == {(seed["id"], first["id"]), (first["id"], second), (seed["id"], second)}
    assert snap["structures"][second]["origin"]["parent"] == first["id"]
    assert job["live_nodes"]["3"] == seed["id"] and len(snap["structures"][seed["id"]]["conformers"]) == 2
    # The seed is represented by its lower-energy conformer.
    assert snap["structures"][seed["id"]]["energy"] == -3.0
