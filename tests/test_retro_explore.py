"""A retrosynthesis route can be put (back) into Explore from its results:
what was deleted since comes back, the rest is reused, and the route's
structures and edges are returned to select."""

from __future__ import annotations

import pytest

from test_retro import _routes  # noqa: E402  (tests/ is on sys.path)


def test_a_deleted_route_comes_back(tmp_path):
    pytest.importorskip("fastapi")
    from mepd.retro.steps import network, unique_steps
    from mepd.web.chem import structure_from_smiles
    from mepd.web.retro import adopt_retro, route_elements
    from mepd.web.workspace import Workspace, WorkspaceError

    out = tmp_path / "out"
    routes = _routes()
    network(routes[0]["tree"]["smiles"], routes, unique_steps(routes), out)
    ws = Workspace(tmp_path / "ws")
    target = ws.add_structure(structure_from_smiles("O=C(Nc1ccccc1)c1ccccc1"), name="benzanilide",
                              smiles="O=C(Nc1ccccc1)c1ccccc1", origin={"kind": "smiles"})
    job = {"id": "j1", "op": "retrosynthesis", "output_dir": str(out), "targets": {"structures": [target["id"]]}}
    assert adopt_retro(ws, job)
    aniline = job["retro_nodes"]["Nc1ccccc1"]
    ws.delete_structure(aniline)
    assert aniline not in ws.snapshot()["structures"]

    assert adopt_retro(ws, job, route=1)             # it comes back
    back = job["retro_nodes"]["Nc1ccccc1"]
    snap = ws.snapshot()
    assert back in snap["structures"] and snap["structures"][back]["smiles"] == "Nc1ccccc1"
    got = route_elements(ws, job, 1)
    assert target["id"] in got["structures"] and back in got["structures"] and len(got["edges"]) == 1
    assert not adopt_retro(ws, job, route=1)         # nothing missing now
    with pytest.raises(WorkspaceError):
        adopt_retro(ws, job, route=2)


def test_a_finished_job_brings_only_its_best_route(tmp_path):
    """Every species of every route was a wall of unoptimized nodes: the
    best route comes in by itself, the others from the result page."""
    pytest.importorskip("fastapi")
    from mepd.retro.steps import network, unique_steps
    from mepd.web.chem import structure_from_smiles
    from mepd.web.retro import adopt_best_route, adopt_retro
    from mepd.web.workspace import Workspace

    from mepd.retro.search import route_record

    out = tmp_path / "out"
    target_smi = "O=C(Nc1ccccc1)c1ccccc1"
    acid = {"smiles": target_smi, "in_stock": None, "children": [{
        "product": target_smi, "reactants": [{"smiles": "Nc1ccccc1", "in_stock": "stock"},
                                             {"smiles": "O=C(O)c1ccccc1", "in_stock": "stock"}],
        "score": 0.2, "method": "templates"}]}
    routes = _routes() + [route_record(acid, 0.2)]
    network(target_smi, routes, unique_steps(routes), out)
    ws = Workspace(tmp_path / "ws")
    target = ws.add_structure(structure_from_smiles(target_smi), name="benzanilide",
                              smiles=target_smi, origin={"kind": "smiles"})
    job = {"id": "j1", "op": "retrosynthesis", "output_dir": str(out), "targets": {"structures": [target["id"]]}}
    assert adopt_best_route(ws, job)
    smiles = {s["smiles"] for s in ws.snapshot()["structures"].values() if s.get("smiles")}
    assert {"Nc1ccccc1", "O=C(Cl)c1ccccc1"} <= smiles and "O=C(O)c1ccccc1" not in smiles   # route 2's acid
    assert adopt_retro(ws, job, route=2)
    assert "O=C(O)c1ccccc1" in {s["smiles"] for s in ws.snapshot()["structures"].values() if s.get("smiles")}

    empty = {"id": "j2", "op": "retrosynthesis", "output_dir": str(tmp_path / "none"), "targets": {"structures": []}}
    assert not adopt_best_route(ws, empty)
