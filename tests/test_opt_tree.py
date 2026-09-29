"""The optimization tree of an MSMEP run: recorded as it runs and with the
finished tree (mepd/tree_log.py), and read back for the web UI
(mepd/web/opt_tree.py)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mepd import tree_log
from mepd.web import opt_tree
from mepd.web.app import create_app

from test_msmep import _flower_split_msmep


@pytest.fixture(autouse=True)
def _main_stream(monkeypatch):
    from mepd import progress

    monkeypatch.setattr(progress, "_live_stream", "main")   # other tests leave theirs set


def _live_rows(live: Path) -> list[dict]:
    return [json.loads(ln) for ln in (live / "trees" / "main" / "nodes.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("parallel", [False, True])
def test_every_node_is_recorded_as_it_starts_and_finishes(tmp_path, monkeypatch, parallel):
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(tmp_path / "live"))
    msmep, chain = _flower_split_msmep()
    tree = msmep.run_parallel_recursive_minimize(chain, max_workers=3) if parallel else \
        msmep.run_recursive_minimize(chain, max_depth=2)
    rows = _live_rows(tmp_path / "live")
    started = {r["key"]: r for r in rows if r["event"] == "started"}
    finished = {r["key"]: r for r in rows if r["event"] == "finished"}
    final = tree_log.tree_records(tree)
    assert set(started) == set(finished) and len(finished) == len(final)
    # Same outcomes as the finished tree (a parallel run renumbers only at the end).
    assert Counter(r["outcome"] for r in finished.values()) == Counter(r["outcome"] for r in final)
    assert {r["outcome"] for r in final} >= {"split"}
    for r in final:
        if r["outcome"] == "split":
            assert r["split_criterion"] in ("minima", "maxima") and r["why"]
    # Every child points at a node that split.
    for k, r in started.items():
        if r["parent"] is not None:
            assert finished[r["parent"]]["outcome"] == "split"


def test_a_resumed_search_starts_a_fresh_live_record(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(tmp_path / "live"))
    for _ in range(2):
        msmep, chain = _flower_split_msmep()
        msmep.run_recursive_minimize(chain, max_depth=2)
    rows = _live_rows(tmp_path / "live")
    assert sum(r["event"] == "started" and r["parent"] is None for r in rows) == 1


def test_leaf_outcomes_say_why():
    from mepd.msmep import _empty_leaf, _failed_leaf

    assert tree_log.node_record(_empty_leaf(3, "identical_endpoints"))["outcome"] == "skipped"
    rec = tree_log.node_record(_empty_leaf(3, "time_budget"))
    assert rec["outcome"] == "unresolved" and "budget" in rec["why"]
    try:
        raise RuntimeError("SCF did not converge")
    except RuntimeError as exc:
        failed = tree_log.node_record(_failed_leaf(4, "electronic_structure_error", exc))
    assert failed["outcome"] == "failed" and failed["error"] == "RuntimeError: SCF did not converge"
    assert "Traceback" in failed["traceback"]


# ------------------------------------------------------------ reading trees back
def _xyz(frames):
    return "".join(f"2\nFrame\nH 0 0 0\nH {x:.4f} 0 0\n" for x in frames)


def _chain(fp: Path, xs, energies):
    fp.write_text(_xyz(xs))
    np.savetxt(fp.with_suffix(".energies"), energies)


def _old_style_tree(d: Path):
    """A tree as runs before tree.json wrote it: 0 split into 1 (a search)
    and 2 (failed)."""
    d.mkdir(parents=True)
    adj = np.zeros((3, 3))
    adj[0, 0] = adj[1, 1] = 1
    adj[0, 1] = adj[0, 2] = 1
    np.savetxt(d / "adj_matrix.txt", adj)
    for i, steps in ((0, 3), (1, 2)):
        _chain(d / f"node_{i}.xyz", [0.7, 1.0, 1.4], [-1.0, -0.98, -1.01])
        (d / f"node_{i}_history").mkdir()
        for k in range(steps):
            _chain(d / f"node_{i}_history" / f"traj_{k}.xyz", [0.7, 1.0 + 0.05 * k, 1.4], [-1.0, -0.97 - 0.005 * k, -1.01])
    _chain(d / "node_2_failed.xyz", [1.4, 1.7], [-1.01, -1.0])
    (d / "node_2_failed.txt").write_text("path_minimization_error: ValueError: bad chain\n\nTraceback (most recent call last): ...")


def test_old_trees_without_a_record_are_inferred_from_their_files(tmp_path):
    job, out = tmp_path / "job", tmp_path / "job" / "output"
    _old_style_tree(out / "tree")
    tree = opt_tree.load_tree([job], out, "main", job_running=False)
    by = {n["key"]: n for n in tree["nodes"]}
    assert (by[0]["outcome"], by[1]["outcome"], by[2]["outcome"]) == ("split", "elementary", "failed")
    assert by[2]["error"] == "path_minimization_error: ValueError: bad chain" and by[2]["traceback"].startswith("Traceback")
    assert by[0]["n_steps"] == 3 and by[1]["parent"] == 0 and by[1]["depth"] == 1
    assert by[0]["barrier_kcal"] == pytest.approx(0.02 * opt_tree.HARTREE_TO_KCAL)

    detail = opt_tree.node_detail([job], out, "main", 0, False)
    assert detail["kind"] == "history" and len(detail["steps"]) == 3 and detail["step"] == 2
    first, last = detail["steps"][0], detail["steps"][-1]
    assert first["moved"] is None and last["moved"] > 0
    assert last["e"][0] == pytest.approx(0) and last["barrier"] == pytest.approx(0.02 * opt_tree.HARTREE_TO_KCAL)
    assert first["s"][0] == 0 and first["s"][-1] == pytest.approx(1)
    assert len(detail["frames"]) == 3
    assert opt_tree.node_detail([job], out, "main", 0, False, step=0)["frames"][1].count("\n") == 4
    assert opt_tree.node_detail([job], out, "main", 2, False)["kind"] == "failed"


def test_a_run_that_died_mid_search_shows_its_live_record(tmp_path):
    job, out = tmp_path / "job", tmp_path / "job" / "output"
    live = job / "live" / "trees" / "pair_0_1"
    live.mkdir(parents=True)
    rows = [{"event": "started", "key": 0, "parent": None, "depth": 0},
            {"event": "finished", "key": 0, "outcome": "split", "why": "split", "n_steps": 4},
            {"event": "started", "key": 1, "parent": 0, "depth": 1}]
    (live / "nodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"event": "star')   # half a line
    listing = opt_tree.list_trees([job], out, job_running=True)
    assert listing[0]["id"] == "pair_0_1" and listing[0]["label"] == "Pair 0 → 1" and listing[0]["running"]
    assert opt_tree.load_tree([job], out, "pair_0_1", True)["nodes"][1]["outcome"] == "running"
    dead = opt_tree.load_tree([job], out, "pair_0_1", False)
    assert not dead["running"] and dead["nodes"][1]["outcome"] == "interrupted"


def test_tree_api(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "state"))
    out = tmp_path / "run"
    _old_style_tree(out / "tree")
    _chain(out / "mep_output.xyz", [0.7, 1.0, 1.4], [-1.0, -0.98, -1.01])
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        job = c.post("/api/jobs/import", json={"path": str(out), "op": "ts"}).json()
        trees = c.get(f"/api/jobs/{job['id']}/trees").json()
        assert [t["id"] for t in trees] == ["main"] and trees[0]["counts"] == {"split": 1, "elementary": 1, "failed": 1}
        tree = c.get(f"/api/jobs/{job['id']}/trees/main").json()
        assert len(tree["nodes"]) == 3
        node = c.get(f"/api/jobs/{job['id']}/trees/main/nodes/1?step=0").json()
        assert node["step"] == 0 and len(node["steps"]) == 2
        assert c.get(f"/api/jobs/{job['id']}/trees/nope").status_code == 404
        assert c.get(f"/api/jobs/{job['id']}/trees/bad%20id").status_code == 400
