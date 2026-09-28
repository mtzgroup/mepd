"""A negative barrier is a symptom (mixed levels, endpoints not minimized, a
path too sparse): it must always be reported, never silently turned into 0."""
from mepd.discovery.kinetics import Step, barriers, negative_barrier_warnings, raw_barriers
from mepd.web.results import _flag_negative_barriers, summarize

HARTREE = 1 / 627.509474


def test_kinetics_reports_the_raw_barrier_and_warns():
    steps = [Step(0, 1, -1.0 - 2 * HARTREE, "s1", {})]      # TS 2 kcal/mol below species 0
    energies = [-1.0, -1.0 - 10 * HARTREE]
    (fwd, rev), = raw_barriers(steps, energies)
    assert round(fwd, 3) == -2.0 and round(rev, 3) == 8.0
    assert barriers(steps, energies)[0][0] == 0.0             # the rates' clamp, still there...
    (w,) = negative_barrier_warnings(steps, energies)          # ...but never silent
    assert "Negative barrier" in w and "2.0 kcal/mol below species 0" in w
    assert negative_barrier_warnings([Step(0, 1, -0.9, "ok", {})], energies) == []


def test_results_flag_negative_barriers_loudly():
    result = {"headline": "x", "barrier_kcal": -3.0, "warnings": ["some other warning"], "groups": [
        {"kind": "ts", "entries": [{"id": "ts", "label": "ts", "note": "A ⇌ B", "barrier_kcal": -3.0, "frames": []},
                                   {"id": "ts2", "label": "ts2", "note": "", "barrier_kcal": 12.0, "frames": []}]}]}
    _flag_negative_barriers(result)
    bad, good = result["groups"][0]["entries"]
    assert bad["negative_barrier"] and bad["note"].startswith("⚠ negative barrier") and "negative_barrier" not in good
    assert len(result["barrier_warnings"]) == 2 and result["warnings"] == ["some other warning"]
    assert "cannot happen" in summarize(result)["barrier_warning"]


def test_an_edge_shows_a_negative_barrier_with_its_warning(tmp_path):
    from tests.test_expansion_steps_web import _ws

    ws, jobs, (s0, s1, _) = _ws(tmp_path)
    job = {"id": "j", "live_nodes": {"0": s0, "1": s1}}
    ev = {"event": "step", "a": 0, "b": 1, "label": "s", "barrier_kcal": [-2.5, 6.0]}
    assert jobs._adopt_step(job, ev, job["live_nodes"], ws.snapshot()["structures"])
    (edge,) = ws.snapshot()["edges"].values()
    assert edge["origin"]["barrier_kcal"] == -2.5                    # as computed, not 0
    assert "negative" in edge["origin"]["barrier_warning"]
