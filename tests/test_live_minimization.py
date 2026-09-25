"""Live view of geometry minimizations (Hessian-sampling candidates):
progress stream files, g-xTB step-log reading, and the web reduction."""

from __future__ import annotations

import json
from types import SimpleNamespace

from mepd import progress
from mepd.engines.gxtb import _read_xtbopt_progress

WATER = "3\n{c}\nO 0.000 0.000 0.000\nH 0.758 0.000 0.504\nH -0.758 0.000 0.504\n"


def test_read_xtbopt_progress_skips_a_half_written_frame(tmp_path):
    log = tmp_path / "xtbopt.log"
    log.write_text(WATER.format(c=" energy: -5.10 gnorm: 0.1") + WATER.format(c=" energy: -5.12 gnorm: 0.01")
                   + "3\n energy: -5.13 gnorm: 0.001\nO 0 0 0\n")
    energies, frames = _read_xtbopt_progress(log, 3)
    assert energies == [-5.10, -5.12] and len(frames) == 2
    assert frames[-1].splitlines()[0] == "3" and "-5.12" in frames[-1].splitlines()[1]
    assert _read_xtbopt_progress(tmp_path / "missing.log", 3) == ([], [])


def _node(e):
    return SimpleNamespace(energy=e, structure=SimpleNamespace(to_xyz=lambda e=e: WATER.format(c=f"e={e}")))


def test_minimization_stream_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(tmp_path))
    progress.begin_minimization("c001", label="#1 · mode 3+", caption="Candidate 1/2", reference_energy=-5.0,
                                start_xyz=WATER.format(c="start"))
    fp = tmp_path / "c001.json"
    d = json.loads(fp.read_text())
    assert d["kind"] == "minimization" and not d["finished"] and len(d["geometry"]["frames"]) == 1

    sink = progress.minimization_sink()
    assert sink is not None
    sink([-5.0, -5.01, -5.02], [WATER.format(c=f"step {i}") for i in (1, 2, 3)])
    d = json.loads(fp.read_text())
    assert [round(v, 2) for v in d["plot"]["y"]] == [0.0, -6.28, -12.55]  # kcal/mol vs the seed
    assert d["geometry"]["frame_steps"] == [2]

    traj = [_node(-5.0 - 0.001 * i) for i in range(100)]
    progress.end_minimization("c001", trajectory=traj)
    progress.set_minimization_outcome("c001", "new minimum")
    d = json.loads(fp.read_text())
    assert d["finished"] and d["outcome"] == "new minimum"
    assert len(d["plot"]["y"]) == 100
    steps = d["geometry"]["frame_steps"]
    assert len(d["geometry"]["frames"]) == len(steps) == progress._MAX_REPLAY_FRAMES
    assert steps[0] == 0 and steps[-1] == 99            # replay spans the whole run
    short = [_node(-5.0 - 0.01 * i) for i in range(6)]
    progress.begin_minimization("c002", reference_energy=-5.0)
    progress.end_minimization("c002", trajectory=short)
    assert json.loads((tmp_path / "c002.json").read_text())["geometry"]["frame_steps"] == [0, 1, 2, 3, 4, 5]
    assert progress.minimization_sink() is None          # nothing active any more


def test_no_viewer_means_no_files(tmp_path, monkeypatch):
    monkeypatch.delenv("MEPD_DRIVE_CHAIN_DIR", raising=False)
    progress.begin_minimization("c001", start_xyz=WATER.format(c=""))
    assert progress.minimization_sink() is None


def test_web_sends_only_the_final_frame_of_a_finished_minimization():
    from mepd.web.jobs import _reduce_chain_payload

    data = {"kind": "minimization", "finished": True, "label": "#1", "outcome": "back to seed",
            "plot": {"x": [0, 1], "y": [0, -1]},
            "geometry": {"frames": ["a", "b", "c"], "frame_steps": [0, 5, 9], "ts_index": None}}
    small = _reduce_chain_payload(data)
    assert small["geometry"]["frames"] == ["c"] and small["geometry"]["truncated"]
    assert small["kind"] == "minimization" and small["outcome"] == "back to seed"
    assert _reduce_chain_payload(data, full=True)["geometry"]["frames"] == ["a", "b", "c"]


def test_replay_from_live_steps_when_no_trajectory_is_handed_over(tmp_path, monkeypatch):
    # Inside an engine's batch call the candidate ends before its trajectory
    # is returned: the frames reported live become the replay right away.
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(tmp_path))
    progress.begin_minimization("c001", reference_energy=-5.0)
    sink = progress.minimization_sink()
    sink([-5.0 - 0.001 * i for i in range(8)], [WATER.format(c=f"s{i}") for i in range(8)], final=True)
    running = json.loads((tmp_path / "c001.json").read_text())
    assert len(running["geometry"]["frames"]) == 1 and running["geometry"]["frame_steps"] == [7]
    progress.end_minimization("c001")
    done = json.loads((tmp_path / "c001.json").read_text())
    assert done["finished"] and done["geometry"]["frame_steps"] == list(range(8))
