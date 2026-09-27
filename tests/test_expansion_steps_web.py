"""Verified steps of a flux-steered expansion fill in their graph edges:
barrier (in the edge's direction), TS and IRC, instead of staying "proposed"."""
import json

from mepd.web import chem
from mepd.web.jobs import JobManager
from mepd.web.workspace import Workspace
from tests.test_web import HCN_XYZ, WATER_BENT_XYZ, WATER_XYZ


class Bus:
    def publish(self, event, data):
        pass


def _ws(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ids = [ws.add_structure(chem.structures_from_xyz_text(x, 0, 1)[0], origin={"kind": "xyz"})["id"]
           for x in (WATER_XYZ, WATER_BENT_XYZ, HCN_XYZ)]
    return ws, JobManager(ws, Bus()), ids


def test_step_event_turns_a_proposed_edge_into_a_computed_one(tmp_path):
    ws, jobs, (s0, s1, s2) = _ws(tmp_path)
    ws.add_edge(s0, s1, origin={"kind": "job", "job": "j", "proposed": True})
    job = {"id": "j", "live_nodes": {"0": s0, "1": s1, "2": s2}}
    known = ws.snapshot()["structures"]
    step = {"event": "step", "a": 1, "b": 0, "label": "pair_0_1_leaf_0", "barrier_kcal": [30.0, 20.0]}
    assert jobs._adopt_step(job, step, job["live_nodes"], known)
    (edge,) = ws.snapshot()["edges"].values()
    o = edge["origin"]
    # The edge runs s0 -> s1, the step 1 -> 0: its barrier from s0 is the step's reverse one.
    assert not o.get("proposed") and o["barrier_kcal"] == 20.0 and o["entry"] == "pair_0_1_leaf_0_irc" and o["has_ts"]
    # A higher-barrier TS for the same pair does not replace it; a lower one does.
    assert not jobs._adopt_step(job, {**step, "label": "other", "barrier_kcal": [40.0, 35.0]}, job["live_nodes"], known)
    assert jobs._adopt_step(job, {**step, "label": "better", "barrier_kcal": [15.0, 5.0]}, job["live_nodes"], known)
    assert ws.snapshot()["edges"][edge["id"]]["origin"]["barrier_kcal"] == 5.0
    # A step between species with no edge yet gets one.
    assert jobs._adopt_step(job, {"event": "step", "a": 1, "b": 2, "label": "x", "barrier_kcal": [9.0, 8.0]},
                            job["live_nodes"], known)
    assert len(ws.snapshot()["edges"]) == 2


def test_finished_expansion_backfills_steps_from_its_summary(tmp_path):
    from mepd.web.app import adopt_expansion_steps

    ws, jobs, (s0, s1, _) = _ws(tmp_path)
    ws.add_edge(s0, s1, origin={"kind": "job", "job": "j", "proposed": True})
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"steps": [
        {"a": 0, "b": 1, "label": "pair_0_1_leaf_0", "barrier_kcal": [12.0, 3.0], "files": {}}]}))
    job = {"id": "j", "output_dir": str(out), "live_nodes": {"0": s0, "1": s1}}
    assert adopt_expansion_steps(jobs, job)
    (edge,) = ws.snapshot()["edges"].values()
    assert edge["origin"]["barrier_kcal"] == 12.0 and not edge["origin"].get("proposed")
    assert not adopt_expansion_steps(jobs, job)      # nothing new the second time


def test_the_finished_summary_replaces_live_barriers(tmp_path):
    from mepd.web.app import adopt_expansion_steps

    ws, jobs, (s0, s1, _) = _ws(tmp_path)
    job = {"id": "j", "live_nodes": {"0": s0, "1": s1}}
    live = {"event": "step", "a": 0, "b": 1, "label": "early", "barrier_kcal": [5.0, 1.0]}
    jobs._adopt_step(job, live, job["live_nodes"], ws.snapshot()["structures"])
    out = tmp_path / "out"
    out.mkdir()
    # Species 0 was found lower later: the same TS is now 8 kcal/mol above it. Two TSs join the pair;
    # the lower one (by TS energy) is kept.
    (out / "summary.json").write_text(json.dumps({"steps": [
        {"a": 0, "b": 1, "label": "early", "ts_energy": -1.00, "barrier_kcal": [8.0, 1.0]},
        {"a": 0, "b": 1, "label": "high", "ts_energy": -0.90, "barrier_kcal": [70.0, 63.0]}]}))
    assert adopt_expansion_steps(jobs, {**job, "output_dir": str(out)})
    (edge,) = ws.snapshot()["edges"].values()
    assert edge["origin"]["barrier_kcal"] == 8.0 and edge["origin"]["label"] == "early"
