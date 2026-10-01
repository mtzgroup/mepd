"""Implicit solvent (mepd.solvation), condition insights (mepd.conditions),
`mepd solvent`, and the web's experimental setups (mepd.web.setups)."""
from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np
import pytest
from qcdata import Structure

from mepd import conditions
from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.qcdata_structure_helpers import read_multiple_structure_from_file

HARTREE_TO_KCAL = 627.509474


def _node(shift: float) -> StructureNode:
    geom = np.array([[0.0, 0.0, 0.0], [1.43 + shift, 0.0, 0.95], [-1.43, 0.0, 0.95]])
    return StructureNode(structure=Structure(symbols=["O", "H", "H"], geometry=geom, charge=0, multiplicity=1))


class _Base(Engine):
    """E = x of atom 1 (Hartree), gradient 1 there; counts calls."""
    def __init__(self):
        self.calls = 0

    def _run(self, nodes):
        from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
        from mepd.nodes.nodehelpers import update_node_cache

        todo = [n for n in nodes if n._cached_energy is None]
        self.calls += len(todo)
        res = []
        for n in todo:
            g = np.zeros_like(n.coords)
            g[1, 0] = 1.0
            res.append(FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                {"energy": float(n.coords[1, 0]), "gradient": g})}))
        update_node_cache(todo, res)
        return nodes

    def compute_energies(self, chain):
        return np.array([n.energy for n in self._run(list(chain))])

    def compute_gradients(self, chain):
        return np.array([n.gradient for n in self._run(list(chain))])


class _FakeCorrection:
    """dG_solv = -0.01 * x of atom 1, gradient -0.01 there."""
    def __init__(self, solvent, model="alpb", **kw):
        self.solvent, self.model = solvent, model
        self.fallbacks = []

    def compute(self, nodes):
        dg = np.array([-0.01 * n.coords[1, 0] for n in nodes])
        grads = []
        for n in nodes:
            g = np.zeros_like(n.coords)
            g[1, 0] = -0.01
            grads.append(g)
        return dg, np.array(grads)


def test_solvents_and_models():
    from mepd.solvation import get_solvent

    assert get_solvent("DCM").key == "ch2cl2" and get_solvent("H2O").key == "water"
    with pytest.raises(ValueError, match="unknown solvent"):
        get_solvent("unobtainium")


def test_solvated_engine_adds_the_correction_once(monkeypatch):
    import mepd.solvation as S

    monkeypatch.setattr(S, "SolvationCorrection", _FakeCorrection)
    base = _Base()
    eng = S.SolvatedEngine(base, "water")
    n = _node(0.1)
    x = n.coords[1, 0]
    e1 = eng.compute_energies([n])[0]
    assert e1 == pytest.approx(x - 0.01 * x)
    # Cached on the node: asking again neither recomputes nor adds it twice.
    assert eng.compute_energies([n])[0] == pytest.approx(e1)
    assert base.calls == 1
    # The base engine only ever sees fresh copies, so it can never hand back
    # (and the wrapper re-correct) a solvated energy cached on the node.
    m = _node(0.2)
    g = eng.compute_gradients([m])[0]
    assert g[1, 0] == pytest.approx(1.0 - 0.01)
    assert eng.compute_energies([m])[0] == pytest.approx(m.coords[1, 0] * 0.99)


def test_a_solvent_model_with_no_effect_is_refused(monkeypatch):
    import mepd.solvation as S

    class Same(_Base):
        pass

    corr = S.SolvationCorrection.__new__(S.SolvationCorrection)
    corr.solvent, corr.model, corr._checked = "water", "cpcmx", False
    import threading
    corr._lock = threading.Lock()
    corr._gas, corr._solv = Same(), Same()
    from mepd.errors import ElectronicStructureError

    with pytest.raises(ElectronicStructureError, match="did not change"):
        corr.compute([_node(0.0)])


def test_eyring_half_life_and_temperature_round_trip():
    t = conditions.half_life(20.0, 298.15)
    assert 0.5 < t / 60 < 5              # 20 kcal/mol: minutes at room temperature
    T = conditions.temperature_for_half_life(25.0, 3600.0)
    assert conditions.half_life(25.0, T) == pytest.approx(3600.0, rel=1e-3)
    assert conditions.format_duration(math.inf) == "> 10¹⁰ yr"


def _row(key, label, kind, eps, bp, mp, barrier, gas, **kw):
    return {"key": key, "label": label, "kind": kind, "epsilon": eps, "bp_c": bp, "mp_c": mp,
            "barrier_kcal": barrier, "shift_kcal": barrier - gas,
            "t_1h_c": (conditions.temperature_for_half_life(barrier) or 0) - 273.15, **kw}


def test_insights_say_which_solvents_accelerate_and_warn_about_single_points():
    gas = {"barrier_kcal": 50.0, "reaction_kcal": 5.0}
    rows = [_row("dmso", "DMSO", "polar aprotic", 46.7, 189, 19, 20.0, 50.0, reaction_kcal=-3.0),
            _row("acetonitrile", "Acetonitrile", "polar aprotic", 37.5, 82, -45, 22.0, 50.0),
            _row("hexane", "n-Hexane", "nonpolar", 1.9, 69, -95, 45.0, 50.0)]
    out = conditions.solvent_insights(gas, rows, 298.15, mode="single-point")
    text = " ".join(i["text"] for i in out)
    top = next(i for i in out if i["level"] == "accelerates")
    assert "DMSO" in top["text"] and "50.0 → 20.0" in top["text"]
    assert any(i["level"] == "trend" for i in out)                    # polar solvents help
    assert any(i["level"] == "caution" and "re-optimize" in i["text"].lower() for i in out)
    assert "changes sign" in text
    assert out[0]["level"] == "caution"                                # cautions come first


def test_negative_barriers_are_reported_not_clamped():
    gas = {"barrier_kcal": 40.0}
    rows = [_row("water", "Water", "protic", 78.4, 100, 0, -3.0, 40.0)]
    out = conditions.solvent_insights(gas, rows, 298.15, mode="single-point")
    assert out[0]["level"] == "caution" and "Negative ΔE‡" in out[0]["text"] and "-3.0" in out[0]["text"]
    assert not any(i["level"] == "accelerates" for i in out)


def _tsopt_folder(tmp_path):
    """A `mepd ts`-shaped folder: ts.xyz + irc.xyz along atom 1's x. The
    IRC rises 30 kcal/mol to the TS (frame 5) and ends 5 kcal/mol below
    its start: the low end, written last, is the reactant (as the web draws
    a TS-only result), so the gas-phase barrier is 35 kcal/mol."""
    out = tmp_path / "gas"
    out.mkdir()
    shifts = np.linspace(0.0, 1.0, 11)
    e = [30 / HARTREE_TO_KCAL * (1 - ((i - 5) / 5) ** 2) for i in range(11)]
    e[-1] = -5 / HARTREE_TO_KCAL
    nodes = [_node(x) for x in shifts]
    for n, x in zip(nodes, e):
        n._cached_energy, n._cached_gradient = x, np.zeros((3, 3))
    Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()}).write_to_disk(out / "irc.xyz")
    Chain.model_validate({"nodes": [nodes[5]], "parameters": ChainInputs()}).write_to_disk(out / "ts.xyz")
    return out


def test_cli_solvent_single_points(tmp_path, monkeypatch):
    import mepd.solvation as S
    from mepd.cli_solvent import solvent

    monkeypatch.setattr(S, "SolvationCorrection", _FakeCorrection)
    src = _tsopt_folder(tmp_path)
    out = tmp_path / "solv"
    solvent(source=src, solvents=["water", "hexane"], model="alpb", mode="single-point", temperature=298.15,
            inputs=None, charge=0, multiplicity=1, output=out)
    data = json.loads((out / "summary.json").read_text())
    gas = data["gas"]
    assert gas["barrier_kcal"] == pytest.approx(35.0, abs=1e-6)
    water = data["solvents"][0]
    # dG = -0.01 * x (Hartree): the TS (x = 1.93) is solvated less than
    # the reactant end (x = 2.43), so the barrier rises by 0.005 Eh.
    xs = {"ts": 1.43 + 0.5, "low": 1.43 + 1.0}
    expected = 35.0 + (-0.01 * xs["ts"] + 0.01 * xs["low"]) * HARTREE_TO_KCAL
    assert water["barrier_kcal"] == pytest.approx(expected, abs=1e-4)
    assert water["shift_kcal"] == pytest.approx(expected - 35.0, abs=1e-4)
    assert (out / "water" / "irc.xyz").exists() and (out / "water" / "irc.energies").exists()
    assert any("Single points on gas-phase geometries" in w for w in data["warnings"])
    assert any("composite model" in w.lower() for w in data["warnings"])


def test_collect_solvent_never_reports_a_gas_barrier(tmp_path, monkeypatch):
    import mepd.solvation as S
    from mepd.cli_solvent import solvent
    from mepd.web import results

    monkeypatch.setattr(S, "SolvationCorrection", _FakeCorrection)
    src = _tsopt_folder(tmp_path)
    out = tmp_path / "solv"
    solvent(source=src, solvents=["water"], model="alpb", mode="single-point", temperature=298.15,
            inputs=None, charge=0, multiplicity=1, output=out)
    res = results.collect_solvent(out, 0, 1)
    assert res["barrier_kcal"] is None                     # never mixed into edge badges
    assert [e["id"] for e in res["groups"][0]["entries"]] == ["gas", "solvent_water"]
    summ = results.summarize(res)
    assert summ["barrier_kcal"] is None and summ["conditions"]["solvents"]["water"]["barrier_kcal"] is not None


def test_setup_prediction_uses_solvent_barriers_and_says_when_it_falls_back():
    from mepd.web import setups

    snap = {"structures": {"a": {"name": "A", "energy": 0.0, "level": {"key": "k"}},
                           "b": {"name": "B", "energy": -10 / HARTREE_TO_KCAL, "level": {"key": "k"}},
                           "c": {"name": "C", "energy": 0.0, "level": {"key": "k"}}},
            "edges": {"e1": {"id": "e1", "source": "a", "target": "b", "origin": {}},
                      "e2": {"id": "e2", "source": "a", "target": "c", "origin": {}}}}
    ts1 = {"id": "t1", "op": "ts", "status": "done", "targets": {"edges": ["e1"], "structures": ["a", "b"]},
           "summary": {"barrier_kcal": 35.0, "barrier_verified": True}}
    ts2 = {"id": "t2", "op": "ts", "status": "done", "targets": {"edges": ["e2"], "structures": ["a", "c"]},
           "summary": {"barrier_kcal": 22.0, "barrier_verified": True}}
    sv = {"id": "s1", "op": "solvent", "status": "done", "source_job": "t1", "finished": 2,
          "targets": {"edges": ["e1"], "structures": ["a", "b"]},
          "summary": {"barrier_kcal": None, "conditions": {"mode": "single-point", "solvents": {
              "dmso": {"barrier_kcal": 18.0, "reaction_kcal": -12.0}}}}}
    jobs = {j["id"]: j for j in (ts1, ts2, sv)}
    gas = setups.normalize({"solvent": None, "start": ["a"]})
    dmso = setups.normalize({"solvent": "dmso", "start": ["a"]})
    assert gas["name"] == "Gas phase, 25 °C" and dmso["name"].startswith("DMSO")
    info = setups.edge_under(snap["edges"]["e1"], snap["structures"], jobs, dmso)
    assert info["phase"] == "solvent" and info["barrier_kcal"] == 18.0 and info["reverse_kcal"] == 30.0
    p_gas = setups.predict(snap, jobs, gas)
    p_dmso = setups.predict(snap, jobs, dmso)
    final = lambda p: {s["id"]: s["final"] for s in p["species"]}
    assert final(p_gas).get("b", 0) < 0.01                  # 35 kcal/mol: nothing in an hour
    assert final(p_dmso)["b"] > 0.9                          # 18 kcal/mol in DMSO: done
    assert any("gas-phase barrier" in w.lower() for w in p_dmso["warnings"])   # e2 has no DMSO value
    assert setups.fill_requests(snap, jobs, dmso) == [{"source_job": "t2", "edge": "e2"}]


@pytest.mark.skipif(not (shutil.which("xtb") or os.getenv("XTB_EXECUTABLE") or os.getenv("GXTB_EXECUTABLE")),
                    reason="needs an xtb executable")
def test_real_alpb_correction_on_water(tmp_path):
    from mepd.solvation import SolvationCorrection

    n = _node(0.0)
    dg, grad = SolvationCorrection("water").compute([n])
    assert -20 < dg[0] * HARTREE_TO_KCAL < -3               # water is well solvated in water
    dg_hex, _ = SolvationCorrection("hexane").compute([n])
    assert dg_hex[0] > dg[0]                                 # ...and less so in hexane


@pytest.fixture
def client(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        yield c


def test_web_solvent_followup_runs_in_its_own_folder(client, tmp_path):
    jobs = client.app.state.sessions.current.jobs
    jid = "j_fakets"
    jobs.job_dir(jid).mkdir(parents=True)
    jobs.jobs[jid] = {"id": jid, "op": "ts", "status": "done", "title": "TS", "created": 0, "finished": 1,
                      "targets": {"structures": [], "edges": []}, "output_dir": str(tmp_path / "gas"),
                      "charge": 0, "multiplicity": 1, "profile": "default",
                      "summary": {"headline": "", "barrier_kcal": 30.0, "barrier_verified": True, "counts": {}}}
    r = client.post("/api/jobs", json={"op": "solvent", "source_job": jid, "dry_run": True,
                                       "params": {"solvents": ["water", "dmso"], "mode": "reoptimize"}})
    assert r.status_code == 200, r.text
    (job,) = r.json()
    argv = job["argv"]
    assert argv[:2] == ["solvent", str(tmp_path / "gas")]
    assert argv.count("--solvent") == 2 and "reoptimize" in argv
    out = argv[argv.index("--output") + 1]
    assert out == job["output_dir"] != str(tmp_path / "gas")      # its own folder, not the source's
    state = client.get("/api/state").json()
    assert any(s["key"] == "dmso" for s in state["solvents"])


def test_web_setups_round_trip_and_predict(client):
    r = client.put("/api/setups", json={"setups": [{"id": "s_a", "solvent": "DMSO", "temperature": 333.15,
                                                     "time_s": 7200}], "active": "s_a"})
    assert r.status_code == 200, r.text
    (s,) = r.json()["setups"]
    assert s["solvent"] == "dmso" and s["name"] == "DMSO, 60 °C" and r.json()["active"] == "s_a"
    r = client.post("/api/setups/s_a/predict")
    assert r.status_code == 400 and "starts from" in r.json()["detail"]
    bad = client.put("/api/setups", json={"setups": [{"solvent": "unobtainium"}]})
    assert bad.status_code == 400
