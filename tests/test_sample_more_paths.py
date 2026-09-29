"""Sample more paths from a TS search: the channels run it starts adds to
the search's page (one result, map, tree), instead of a page of its own."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from mepd.web.results import HARTREE_TO_KCAL, collect, collect_ts_extended  # noqa: E402

ETHANE = """8
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
MOVED = ETHANE.replace("H 1.018 0.000 1.160", "H 0.900 0.000 -1.600", 1)
# a TS geometry of its own (not the first search's)
OTHER_TS = ETHANE.replace("H 1.018 0.000 1.160", "H 0.950 0.000 0.000", 1)


def _write(fp: Path, frames: list[str], energies: list[float]) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text("".join(frames))
    np.savetxt(fp.with_suffix(".energies"), energies)


def _ts_search(out: Path) -> None:
    """A `mepd ts` output whose one TS (-79.62) IRC-connects start and end."""
    _write(out / "mep_output.xyz", [ETHANE, ETHANE, MOVED], [-79.80, -79.60, -79.70])
    _write(out / "ts_leaf_0.xyz", [ETHANE], [-79.62])
    _write(out / "ts_leaf_0_irc.xyz", [ETHANE, ETHANE, MOVED], [-79.80, -79.62, -79.70])


def _channels(out: Path, ts_xyz: str, ts_e: float, floor: float) -> None:
    """A channels output with one direct channel and a reactant conformer at `floor`."""
    _write(out / "conformers" / "start.xyz", [ETHANE], [floor])
    _write(out / "conformers" / "end.xyz", [MOVED], [-79.70])
    ch = out / "channels" / "channel_0"
    _write(ch / "ts.xyz", [ts_xyz], [ts_e])
    _write(ch / "irc.xyz", [ETHANE, ts_xyz, MOVED], [floor, ts_e, -79.70])
    (ch / "members.txt").write_text("connects: start -> end\nts_pair_0_0_leaf_0\n")
    _write(out / "ts" / "ts_pair_0_0_leaf_0.xyz", [ts_xyz], [ts_e])


def test_the_first_search_joins_the_channels_on_one_floor(tmp_path):
    _ts_search(tmp_path / "ts")
    # Sampling found a lower reactant conformer (-79.85) and another, lower, channel.
    _channels(tmp_path / "ch", OTHER_TS, -79.70, floor=-79.85)
    r = collect_ts_extended(tmp_path / "ts", tmp_path / "ch", 0, 1)
    chans = {g["kind"]: g for g in r["groups"]}["channel"]["entries"]
    by_label = {e["label"]: e["barrier_kcal"] for e in chans}
    # Both barriers from the lowest reactant either run found.
    assert by_label["First search · ts_leaf_0"] == pytest.approx(0.23 * HARTREE_TO_KCAL, abs=1e-2)
    assert by_label["Channel 0"] == pytest.approx(0.15 * HARTREE_TO_KCAL, abs=1e-2)
    assert r["barrier_kcal"] == pytest.approx(0.15 * HARTREE_TO_KCAL, abs=1e-2)
    assert r["headline"].startswith("2 direct channel(s)")
    kinds = [g["kind"] for g in r["groups"]]
    assert "path" in kinds and "conformers" in kinds          # the first search's path stays in view
    ids = [e["id"] for g in r["groups"] for e in g["entries"]]
    assert len(ids) == len(set(ids))                          # entries stay addressable (Add to Explore)


def test_a_channel_the_first_search_also_found_is_listed_once(tmp_path):
    _ts_search(tmp_path / "ts")
    _channels(tmp_path / "ch", ETHANE, -79.62, floor=-79.80)
    r = collect_ts_extended(tmp_path / "ts", tmp_path / "ch", 0, 1)
    chans = {g["kind"]: g for g in r["groups"]}["channel"]["entries"]
    assert [e["label"] for e in chans] == ["Channel 0"]
    assert "also the first search's TS" in chans[0]["note"]


def test_before_the_channels_run_writes_anything_the_first_search_stands(tmp_path):
    _ts_search(tmp_path / "ts")
    job = {"id": "j", "op": "ts", "output_dir": str(tmp_path / "ts"), "charge": 0, "multiplicity": 1,
           "status": "done", "external": True,
           "extension": {"output_dir": str(tmp_path / "ch"), "status": "running", "finished": None}}
    r = collect(job)
    assert "IRC-verified" in r["headline"] and r["barrier_verified"] is True
    assert r["barrier_kcal"] == pytest.approx(0.18 * HARTREE_TO_KCAL, abs=1e-2)


def test_sample_more_paths_runs_share_the_page_of_the_search(tmp_path):
    from mepd.web.jobs import JobManager
    from mepd.web.workspace import Workspace

    class Bus:
        def publish(self, *a, **k):
            pass

    jobs = JobManager(Workspace(tmp_path / "ws"), Bus())
    base = {"id": "j_ts", "op": "ts", "created": 1, "status": "done", "finished": 10, "output_dir": "/o/ts"}
    chan = {"id": "j_ch", "op": "channels", "created": 2, "status": "done", "finished": 20,
            "output_dir": "/o/ch", "extends": "j_ts"}
    more = {"id": "j_more", "op": "channels-more", "created": 3, "status": "running", "finished": None,
            "output_dir": "/o/ch", "source_job": "j_ch"}
    other = {"id": "j_vri", "op": "vri-check", "created": 4, "status": "done", "output_dir": "/o/x",
             "source_job": "j_x"}
    for j in (base, chan, more, other):
        jobs.jobs[j["id"]] = j
    assert jobs.page_job(more) is base and jobs.page_job(chan) is base and jobs.page_job(other) is other
    assert [j["id"] for j in jobs.family(more)] == ["j_ts", "j_ch", "j_more"]
    view = jobs.result_view(base)
    assert view["extension"]["output_dir"] == "/o/ch" and view["extension"]["status"] == "running"
    more["status"], more["finished"] = "done", 30
    assert jobs.result_view(base)["extension"] == {"output_dir": "/o/ch", "charge": None, "multiplicity": None,
                                                   "status": "done", "finished": 30}
    assert "extension" not in jobs.result_view(other)
    # A later Sample more paths run that was cancelled leaves the finished one in view.
    jobs.jobs["j_ch2"] = {"id": "j_ch2", "op": "channels", "created": 5, "status": "cancelled", "finished": 40,
                          "output_dir": "/o/ch2", "extends": "j_ts"}
    assert jobs.result_view(base)["extension"]["output_dir"] == "/o/ch"


def test_only_a_channels_run_extends_a_ts_search(tmp_path):
    from mepd.web import chem
    from mepd.web.jobs import JobManager
    from mepd.web.workspace import Workspace, WorkspaceError

    class Bus:
        def publish(self, *a, **k):
            pass

    ws = Workspace(tmp_path / "ws")
    (ws.root / "profiles").mkdir(parents=True, exist_ok=True)
    (ws.root / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    a = ws.add_structure(chem.structures_from_xyz_text(ETHANE, 0, 1)[0], name="a", origin={"kind": "xyz"})
    b = ws.add_structure(chem.structures_from_xyz_text(MOVED, 0, 1)[0], name="b", origin={"kind": "xyz"})
    jobs = JobManager(ws, Bus())
    jobs.jobs["j_ts"] = {"id": "j_ts", "op": "ts", "created": 1, "status": "done", "output_dir": "/o",
                         "targets": {"structures": [a["id"], b["id"]], "edges": []}}
    (rec,) = jobs.submit("channels", structure_ids=[a["id"], b["id"]], edge_ids=[], params={}, profile="default",
                         dry_run=True, extends_job_id="j_ts")
    assert rec["extends"] == "j_ts" and rec["source_job"] is None
    with pytest.raises(WorkspaceError):
        jobs.submit("ts", structure_ids=[a["id"], b["id"]], edge_ids=[], params={}, profile="default",
                    dry_run=True, extends_job_id="j_ts")
    with pytest.raises(WorkspaceError):
        jobs.submit("channels", structure_ids=[a["id"], a["id"]], edge_ids=[], params={}, profile="default",
                    dry_run=True, extends_job_id="j_ts")
