"""A TS whose IRC was run but failed is reported as such -- not as "no IRC
was run", and not filed under saddle points that do not connect start and
end: what it connects is unknown."""

from __future__ import annotations

from types import SimpleNamespace

from test_route_edges import A, I  # noqa: E402  (tests/ is on sys.path)
from test_web import _write_chain  # noqa: E402


def test_the_cli_leaves_a_note_when_an_irc_fails_and_clears_it_when_one_runs(tmp_path):
    from mepd.cli_common import _optimize_ts_and_irc
    from mepd.cli_common import _load_structure_from_smiles_or_xyz as load
    from mepd.nodes.node import StructureNode

    (tmp_path / "a.xyz").write_text(A)
    node = StructureNode(structure=load(str(tmp_path / "a.xyz"), 0, 1))

    def broken_irc(ts):
        raise RuntimeError("Sella gave up")

    engine = SimpleNamespace(compute_transition_state=lambda node: node, compute_irc_chain=broken_irc)
    out = tmp_path / "out"
    r = _optimize_ts_and_irc(node, SimpleNamespace(engine=engine), out, run_irc=True, label="ts_leaf_0")
    assert r.irc_chain is None
    assert (out / "ts_leaf_0_irc_failed.txt").read_text().startswith("RuntimeError: Sella gave up")

    from mepd.chain import Chain

    engine.compute_irc_chain = lambda ts: Chain.model_validate({"nodes": [ts, ts]})
    _optimize_ts_and_irc(node, SimpleNamespace(engine=engine), out, run_irc=True, label="ts_leaf_0")
    assert not (out / "ts_leaf_0_irc_failed.txt").exists()


def test_a_failed_irc_is_reported_and_its_ts_not_called_off_route(tmp_path):
    from mepd.web.results import collect_ts, collect_tsopt, summarize

    out = tmp_path / "run"
    out.mkdir()
    _write_chain(out / "mep_output.xyz", [A, I], [-79.80, -79.70])
    _write_chain(out / "ts_leaf_0.xyz", [A], [-79.75])
    (out / "ts_leaf_0_irc_failed.txt").write_text("ElectronicStructureError: ASE IRC computation failed.\n")
    r = collect_ts(out, 0, 1)
    groups = {g["title"]: g for g in r["groups"]}
    assert not any(t.startswith("Other saddle points") for t in groups)
    (entry,) = groups["Saddle points without an IRC (what they connect is unknown)"]["entries"]
    assert entry["note"].startswith("the IRC failed (ElectronicStructureError")
    assert any("1 IRC(s) failed" in w for w in r["warnings"])
    assert summarize(r)["counts"]["ts_other"] == 1

    t = collect_tsopt(out, 0, 1)
    assert t["headline"].endswith("· IRC failed")
    assert t["groups"][0]["entries"][0]["note"].startswith("IRC failed (ElectronicStructureError")
