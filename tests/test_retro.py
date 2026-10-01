"""Retrosynthesis (mepd.retro, `mepd retro`, and its web operation)."""
import json

import pytest

pytest.importorskip("rdkit")

from mepd.retro import chem  # noqa: E402
from mepd.retro.proposers import Step, parse_llm_steps  # noqa: E402
from mepd.retro.search import RetroSearch  # noqa: E402
from mepd.retro.stock import Stock, _pack  # noqa: E402


def test_balance_finds_the_byproducts_a_template_leaves_out():
    assert chem.balance(["CC(=O)Cl", "Nc1ccccc1"], "CC(=O)Nc1ccccc1") == ["Cl"]
    assert chem.balance(["CC(=O)O", "CCO"], "CCOC(C)=O") == ["O"]
    assert chem.balance(["C=CC=C", "C=C"], "C1=CCCCC1") == []
    assert chem.balance(["C"], "CC") is None       # the product has atoms the precursors lack


def test_balance_step_adds_what_a_reduction_or_hydrolysis_consumes():
    assert chem.balance_step(["CC(C)C(=O)c1ccccc1"], "CC(C)Cc1ccccc1") == (["[H][H]", "[H][H]"], ["O"])
    assert chem.balance_step(["COC(=O)c1ccccc1"], "O=C(O)c1ccccc1") == (["O"], ["CO"])
    assert chem.balance_step(["CC(=O)Cl", "Nc1ccccc1"], "CC(=O)Nc1ccccc1") == ([], ["Cl"])


def test_stock_matches_without_stereo_and_counts_small_molecules():
    st = Stock(max_heavy=2)
    st.add(["C[C@H](N)C(=O)O", "CC(=O)Nc1ccc(O)cc1"])
    assert st.why("C[C@@H](N)C(=O)O") == "stock"     # the other enantiomer: same first InChIKey block
    assert st.why("CC(=O)Nc1ccc(O)cc1") == "stock"
    assert st.why("CO") == "small" and st.why("CCCO") is None


def test_packed_keys_are_exact_on_the_first_block():
    keys = [chem.inchikey(s)[:14].encode() for s in ("CCO", "CCN", "c1ccccc1")]
    import numpy as np

    h = _pack(np.array(keys, dtype="S14"))
    assert len(set(h.tolist())) == 3
    st = Stock()
    st._packed.append(np.unique(h))
    assert st.why("OCC") == "stock" and st.why("CCC") is None


def _toy_proposer(table):
    def propose(smiles, k):
        return [Step(smiles, tuple(sorted(r)), p, "toy") for r, p in table.get(smiles, [])][:k]
    return propose


def test_search_finds_the_cheapest_route_and_alternatives():
    # T <= A + B (0.5) or T <= C (0.6); A <= D (0.9); B, C, D in stock.
    table = {"CCCCCCO": [(("CCCCO", "CC"), 0.5), (("CCCCCC=O",), 0.6)],
             "CCCCO": [(("CCCC=O",), 0.9)]}
    st = Stock()
    st.add(["CC", "CCCCCC=O", "CCCC=O"])
    res = RetroSearch(_toy_proposer(table), st, value="zero").run("CCCCCCO", max_iterations=20, routes=5)
    assert res.solved
    costs = [r["cost"] for r in res.routes]
    assert costs == sorted(costs)
    best = res.routes[0]
    assert best["n_steps"] == 1 and best["steps"][0]["reactants"] == ["CCCCCC=O"]   # -ln 0.6 < -ln 0.5 - ln 0.9
    assert {tuple(l["smiles"] for l in r["leaves"]) for r in res.routes} >= {("CCCCCC=O",)}
    assert any(r["n_steps"] == 2 for r in res.routes)


def test_search_never_cycles_back_to_an_ancestor():
    table = {"CCO": [(("CC=O",), 0.9)], "CC=O": [(("CCO",), 0.9)]}
    res = RetroSearch(_toy_proposer(table), Stock(), value="zero", max_depth=4).run("CCO", max_iterations=10)
    assert not res.solved and res.routes == []


def test_llm_answers_are_checked_by_rdkit():
    text = """<think>amide coupling</think>
    {"steps": [{"reaction": "amide coupling", "reactants": ["Nc1ccccc1", "O=C(Cl)c1ccccc1"], "confidence": 0.8},
               {"reaction": "nonsense", "reactants": ["C1CC"], "confidence": 0.9},
               {"reaction": "missing atoms", "reactants": ["CC"], "confidence": 0.9}]}"""
    steps = parse_llm_steps(text, "O=C(Nc1ccccc1)c1ccccc1", "test")
    assert len(steps) == 1 and steps[0].reactants == ("Nc1ccccc1", "O=C(Cl)c1ccccc1")


def test_templates_propose_known_disconnections():
    from mepd.retro.data import path

    if path("uspto_unique_templates.csv.gz", download=False) is None:
        pytest.skip("template data not downloaded (mepd retro setup)")
    pytest.importorskip("rdchiral")
    from mepd.retro.proposers import make

    steps = make("templates", policy=False, filter=False, download=False)("O=C(Nc1ccccc1)c1ccccc1", 10)
    assert ("Nc1ccccc1", "O=C(Cl)c1ccccc1") in {s.reactants for s in steps}


def _routes(target="O=C(Nc1ccccc1)c1ccccc1"):
    from mepd.retro.search import route_record

    tree = {"smiles": target, "in_stock": None, "children": [{
        "product": target, "reactants": [{"smiles": "Nc1ccccc1", "in_stock": "stock"},
                                         {"smiles": "O=C(Cl)c1ccccc1", "in_stock": "stock"}],
        "score": 0.7, "method": "templates"}]}
    return [route_record(tree, 0.36)]


def test_network_file_and_results(tmp_path):
    from mepd.retro.steps import network, unique_steps

    routes = _routes()
    steps = unique_steps(routes)
    data = network(routes[0]["tree"]["smiles"], routes, steps, tmp_path)
    assert [s["role"] for s in data["species"]] == ["target", "stock", "stock", "byproduct"]   # + HCl
    (rx,) = data["reactions"]
    assert rx["balanced"] and len(rx["products"]) == 2
    (tmp_path / "routes.json").write_text(json.dumps({"target": data["target"], "routes": routes}))
    (tmp_path / "summary.json").write_text(json.dumps({"kind": "retro", "method": "templates", "stats": {}}))

    pytest.importorskip("fastapi")
    from mepd.web.results import collect_retro, detect_operation

    assert detect_operation(tmp_path) == "retrosynthesis"
    res = collect_retro(tmp_path, 0, 1)
    assert res["retro"]["solved"] and res["retro"]["routes"][0]["steps"][0]["byproducts"] == ["Cl"]


def test_routes_join_explore_as_reactions(tmp_path):
    pytest.importorskip("fastapi")
    from mepd.retro.steps import network, unique_steps
    from mepd.web.chem import structure_from_smiles
    from mepd.web.retro import adopt_retro
    from mepd.web.workspace import Workspace

    out = tmp_path / "out"
    routes = _routes()
    network(routes[0]["tree"]["smiles"], routes, unique_steps(routes), out)
    ws = Workspace(tmp_path / "ws")
    target = ws.add_structure(structure_from_smiles("O=C(Nc1ccccc1)c1ccccc1"), name="benzanilide",
                              smiles="O=C(Nc1ccccc1)c1ccccc1", origin={"kind": "smiles"})
    job = {"id": "j1", "op": "retrosynthesis", "output_dir": str(out), "targets": {"structures": [target["id"]]}}
    assert adopt_retro(ws, job)
    snap = ws.snapshot()
    (rx,) = snap["reactions"].values()
    assert target["id"] in rx["products"] and len(rx["reactants"]) == 2
    assert snap["edges"][rx["edge"]]["origin"]["proposed"]
    species = [s for s in snap["structures"].values() if s["role"] != "complex"]
    assert len(species) == 4                       # target, aniline, benzoyl chloride, HCl
    assert not adopt_retro(ws, job)                # again: nothing new
    assert len(ws.snapshot()["reactions"]) == 1


def test_web_operation_builds_the_command(tmp_path):
    pytest.importorskip("fastapi")
    from mepd.web.operations import OPERATIONS, RetroParams, _build_retro

    op = OPERATIONS["retrosynthesis"]
    assert op.family["key"] == "network" and {m["fixed"]["method"] for m in op.methods} >= {"templates", "reactiont5", "local-llm", "aizynthfinder"}

    class Ctx:
        structures = [{"smiles": "CCO", "charge": 0, "multiplicity": 1}]
        output_dir = tmp_path

        def common_flags(self):
            return ["--charge", "0", "--multiplicity", "1"]

    argv = _build_retro(Ctx(), RetroParams(method="local-llm", llm_model="qwen3:8b", stock="paroutes"))
    assert argv[:4] == ["retro", "plan", "CCO", "--method"] and "model=qwen3:8b" in argv
    assert argv.count("--stock") == 2 and argv[-2:] == ["--output", str(tmp_path)]


def test_local_llm_talks_to_an_openai_compatible_server():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from mepd.retro.proposers import make

    answer = {"steps": [{"reaction": "Fischer esterification", "reactants": ["CC(=O)O", "CCO"], "confidence": 0.7},
                        {"reaction": "hallucinated", "reactants": ["C(C"], "confidence": 0.9}]}
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"data": [{"id": "toy-model"}]})

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._send({"choices": [{"message": {"content": json.dumps(answer)}}]})

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = make("local-llm", url=f"http://127.0.0.1:{srv.server_port}/v1")
        steps = llm("CCOC(C)=O", 5)
    finally:
        srv.shutdown()
    assert seen[0]["model"] == "toy-model" and "CCOC(C)=O" in seen[0]["messages"][0]["content"]
    assert [s.reactants for s in steps] == [("CC(=O)O", "CCO")] and steps[0].score == 1.0
