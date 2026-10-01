"""Whole-network properties and their sensitivities (mepd.network_model,
mepd.web.setups.build_model / analyze / compare_setups)."""
from __future__ import annotations

import json
import math

import pytest

from mepd.network_model import NetworkModel, Species, Step, compare, rate_to, yield_of

H = 627.509474


def _toy(t=60.0):
    sp = [Species("A", "A", 0.0), Species("B", "B", -2.0), Species("C", "C", -10.0)]
    st = [Step("ab", "A", "B", 18.0), Step("ac", "A", "C", 22.0)]
    return NetworkModel(sp, st, 298.15, t, {"A": 1.0})


def test_kinetic_then_thermodynamic_control():
    short, long = _toy(60.0).properties(), _toy(3600 * 24 * 30).properties()
    assert short["final"]["B"] > 0.9 and short["control"] == "kinetic"
    assert long["final"]["C"] > 0.99 and long["control"] == "thermodynamic"
    assert short["equilibrium"]["C"] > 0.99
    assert len(short["timescales_s"]) == 2             # 3 species: two relaxations, no fake zero mode
    assert short["formation_time_s"]["B"] < 10 < short["formation_time_s"]["C"]
    assert short["bottleneck"]["C"]["bottleneck_step"] == "ac"
    assert short["bottleneck"]["C"]["effective_barrier"] == pytest.approx(22.0)


def test_amounts_stay_finite_at_any_time():
    m = _toy(1e40)
    c = m.amounts_at(1e40)
    assert all(math.isfinite(x) for x in c) and c.sum() == pytest.approx(1.0)


def test_degree_of_rate_control():
    m = _toy()
    x = m.sensitivity(rate_to("C"))
    # C forms from A, which pre-equilibrates with B: its own TS controls the
    # rate fully; the A<->B TS not at all; B's energy (holding A back) does.
    assert x["ts"]["ac"] == pytest.approx(1.0, abs=0.02) and abs(x["ts"]["ab"]) < 0.02
    assert x["species"]["B"] < -0.5


def test_compare_attributes_the_change_to_the_step_that_moved():
    base = _toy(3600.0)
    other = base.copy_with(G_ts=[18.0, 19.0])
    out = compare(base, other, {"rate": rate_to("C")})["quantities"]["rate"]
    # ~3 kcal/mol / RT: the formation time is almost, not exactly, set by that one TS
    assert out["dlnP"] == pytest.approx(3.0 / (8.314462618 * 298.15 / 4184), rel=0.05)
    assert out["contributions"][0]["id"] == "ac"


def _snapshot():
    level = {"key": "k1", "label": "gxtb"}
    s = {sid: {"name": sid, "energy": e / H, "level": level, "role": "minimum"}
         for sid, e in (("A", 0.0), ("B", -5.0), ("C", -8.0), ("D", -20.0))}
    edges = {"e1": {"id": "e1", "source": "A", "target": "B", "origin": {}},
             "e2": {"id": "e2", "source": "A", "target": "C", "origin": {}},
             "e3": {"id": "e3", "source": "B", "target": "D", "origin": {}}}
    return {"structures": s, "edges": edges}


def _jobs(tmp_path):
    lvl = {"key": "k1", "label": "gxtb", "profile": "default"}
    jobs = {}
    for jid, eid, b in (("t1", "e1", 20.0), ("t2", "e2", 21.0), ("t3", "e3", 15.0)):
        jobs[jid] = {"id": jid, "op": "ts", "status": "done", "level": lvl, "finished": 1,
                     "targets": {"edges": [eid], "structures": []},
                     "summary": {"barrier_kcal": b, "barrier_verified": True}}
    # DMSO lowers e2 by 6 kcal/mol.
    jobs["s1"] = {"id": "s1", "op": "solvent", "status": "done", "source_job": "t2", "finished": 2,
                  "targets": {"edges": ["e2"], "structures": []},
                  "summary": {"barrier_kcal": None, "conditions": {"mode": "single-point", "solvents": {
                      "dmso": {"barrier_kcal": 15.0, "reaction_kcal": -8.0}}}}}
    # A nitro group on atom 3 raises e1 by 4 kcal/mol.
    out = tmp_path / "sub"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({
        "mode": "fast", "channels": [{"id": "c", "barrier_kcal": 20.0}],
        "variants": [{"channel": "c", "site": 3, "h": 9, "group": "nitro", "status": "ok", "shift": 4.0,
                      "reaction_shift": 0.0}]}))
    jobs["x1"] = {"id": "x1", "op": "substituents", "status": "done", "source_job": "t1", "finished": 3,
                  "output_dir": str(out), "targets": {"edges": ["e1"], "structures": []},
                  "summary": {"barrier_kcal": None}}
    return jobs


def test_workspace_network_analysis_and_comparison(tmp_path):
    from mepd.web import setups

    snap, jobs = _snapshot(), _jobs(tmp_path)
    gas = setups.normalize({"id": "g", "start": ["A"], "time_s": 3600})
    dmso = setups.normalize({"id": "d", "solvent": "dmso", "start": ["A"], "time_s": 3600})
    nitro = setups.normalize({"id": "n", "start": ["A"], "time_s": 3600, "variant": {"site": 3, "group": "nitro"}})
    a = setups.analyze(snap, jobs, gas)
    assert {s["id"] for s in a["steps"]} == {"e1", "e2", "e3"}
    assert a["properties"]["products"][0] in ("D", "C", "B")
    phases = {s["id"]: s["phase"] for s in setups.analyze(snap, jobs, dmso)["steps"]}
    assert phases == {"e1": "gas", "e2": "dmso", "e3": "gas"}
    c = setups.compare_setups(snap, jobs, gas, dmso)
    rate_c = c["quantities"]["rate:C"]
    assert rate_c["dlnP"] > 5 and rate_c["contributions"][0]["id"] == "e2"
    c = setups.compare_setups(snap, jobs, gas, nitro)
    assert c["dG_ts"]["e1"] == pytest.approx(4.0) and c["dG_ts"]["e2"] == pytest.approx(0.0)
    assert any("gas-phase values" in w for w in setups.analyze(snap, jobs, dmso)["warnings"])


def test_compare_says_when_the_first_order_split_does_not_hold():
    # The product is saturated at equilibrium in the base; the other model
    # raises it by 16 kcal/mol, so its yield falls through thermodynamics
    # the base's (near-zero) sensitivities cannot see.
    base = NetworkModel([Species("A", "A", 0.0), Species("B", "B", -10.0)], [Step("ab", "A", "B", 15.0)],
                        298.15, 86400.0, {"A": 1.0})
    other = base.copy_with(G_species=[0.0, 1.0], G_ts=[19.0])
    q = compare(base, other, {"yield": yield_of("B")})["quantities"]["yield"]
    assert q["other"] < 0.5 < q["base"] and q["linear_ok"] is False
