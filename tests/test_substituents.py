"""Substituent scans (mepd.substituents, `mepd substituents`)."""
from __future__ import annotations

import numpy as np
import pytest

from mepd import substituents as X

A = 1 / 0.529177210903


def _methane_chloride():
    """CH4 ... Cl, in Bohr: C0, H1-H4, Cl5."""
    c = np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63],
                  [0.63, -0.63, -0.63], [3.0, 0, 0]]) * A
    return ["C", "H", "H", "H", "H", "Cl"], c


def test_sites_skip_transferred_hydrogens_and_merge_equivalent_ones():
    sym, r = _methane_chloride()
    p = r.copy()
    p[1] = r[5] + (r[1] - r[5]) / np.linalg.norm(r[1] - r[5]) * 1.27 * A   # H1 moves onto Cl
    sites = X.find_sites(sym, r, p)
    assert [(s.anchor, sorted(s.equivalent)) for s in sites] == [(0, [2, 3, 4])]


def test_attach_keeps_indices_and_bond_length():
    sym, c = _methane_chloride()
    site = X.find_sites(sym, c, c)[0]
    sy, xyz, free = X.attach(sym, c, site, "fluoro")
    assert sy[site.h] == "F" and len(sy) == len(sym) and free == [site.h]
    d = np.linalg.norm(xyz[site.h] - xyz[0]) / A
    assert d == pytest.approx(X._rcov("C") + X._rcov("F"), abs=1e-6)
    sy, xyz, free = X.attach(sym, c, site, "methyl")
    assert sy[site.h] == "C" and len(sy) == len(sym) + 3 and len(free) == 4
    # The new hydrogens stay clear of the other atoms.
    new = xyz[len(sym):] / A
    old = np.delete(xyz[: len(sym)], [0, site.h], axis=0) / A
    assert np.min(np.linalg.norm(new[:, None] - old[None], axis=-1)) > 1.5


def test_hammett_slope():
    shifts = {g: 5.0 * X.SIGMA_P[g] + 1.0 for g in ("amino", "methyl", "fluoro", "cyano", "nitro")}
    fit = X.hammett(shifts)
    assert fit["slope"] == pytest.approx(5.0) and fit["r2"] == pytest.approx(1.0)
    assert X.hammett({"methyl": 1.0}) is None


@pytest.fixture
def client(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mepd.web.app import create_app

    (tmp_path / "ws" / "profiles").mkdir(parents=True)
    (tmp_path / "ws" / "profiles" / "default.toml").write_text('engine_name = "gxtb"\n')
    with TestClient(create_app(tmp_path / "ws", max_concurrent=1)) as c:
        yield c


def test_substituents_and_force_ops_follow_up_on_a_ts_job(client, tmp_path):
    jobs = client.app.state.sessions.current.jobs
    jid = "j_fakets"
    jobs.job_dir(jid).mkdir(parents=True)
    jobs.jobs[jid] = {"id": jid, "op": "ts", "status": "done", "title": "TS", "created": 0, "finished": 1,
                      "targets": {"structures": [], "edges": []}, "output_dir": str(tmp_path / "gas"),
                      "charge": 0, "multiplicity": 1, "profile": "default",
                      "summary": {"headline": "", "barrier_kcal": 30.0, "barrier_verified": True, "counts": {}}}
    r = client.post("/api/jobs", json={"op": "substituents", "source_job": jid, "dry_run": True,
                                       "params": {"groups": ["nitro", "amino"], "sites": "4, 7"}})
    assert r.status_code == 200, r.text
    argv = r.json()[0]["argv"]
    assert argv[:2] == ["substituents", str(tmp_path / "gas")]
    assert argv.count("--group") == 2 and argv.count("--site") == 2
    bad = client.post("/api/jobs", json={"op": "substituents", "source_job": jid, "dry_run": True,
                                         "params": {"sites": "C4"}})
    assert bad.status_code == 400
    # Mechanical force is switched off in the web UI (the CLI `mepd force` still runs it).
    r = client.post("/api/jobs", json={"op": "mechanochem", "source_job": jid, "dry_run": True,
                                       "params": {"mode": "reoptimize", "pair": "0,5", "forces": "0.5, 1"}})
    assert r.status_code == 400 and "command line" in r.json()["detail"]


@pytest.mark.skipif(not __import__("os").getenv("GXTB_EXECUTABLE"), reason="needs g-xTB")
def test_cli_substituents_end_to_end(tmp_path):
    import json

    from tests.test_solvation import _tsopt_folder

    from mepd.cli_substituents import substituents
    from mepd.web import results

    src = _tsopt_folder(tmp_path)
    out = tmp_path / "sub"
    substituents(source=src, groups=["methyl", "fluoro"], sites=None, mode="fast", max_sites=8, workers=2,
                 temperature=298.15, inputs=None, charge=0, multiplicity=1, output=out)
    data = json.loads((out / "summary.json").read_text())
    # In the fake IRC atom 1 leaves its oxygen, so only H2 is a site.
    assert [s["h"] for s in data["sites"]] == [2]
    assert {v["group"] for v in data["variants"]} == {"methyl", "fluoro"}
    assert all(v["status"] in ("ok", "reacted", "failed") for v in data["variants"])
    res = results.collect_substituents(out, 0, 1)
    assert res["barrier_kcal"] is None and res["substituents"]["sites"][0]["h"] == 2
