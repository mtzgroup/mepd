"""CREST msreact as a network-expansion generator, and the web's network
exploration methods."""
import shutil

import pytest
from qcdata import Structure

from mepd.discovery.generators import get_generator
from mepd.web.operations import OPERATIONS

ETHANOL = ("9\n\nC -0.047 0.667 0.000\nC -1.264 -0.230 0.000\nO 1.119 -0.154 0.000\nH -0.047 1.310 0.884\n"
           "H -0.047 1.310 -0.884\nH -2.180 0.367 0.000\nH -1.237 -0.866 0.884\nH -1.237 -0.866 -0.884\n"
           "H 1.910 0.388 0.000\n")


def test_msreact_is_a_structure_generator_that_needs_crest_and_xtb(monkeypatch):
    gen = get_generator("crest-msreact")
    assert gen.kind == "structures" and set(gen.programs) == {"crest", "xtb"} and gen.references
    monkeypatch.setattr(shutil, "which", lambda prog: None)
    assert not gen.available()
    with pytest.raises(RuntimeError, match="crest, xtb"):
        gen.check()
    with pytest.raises(ValueError, match="options are"):
        gen.propose(None, max_products=5, options={"colour": "red"})


@pytest.mark.skipif(not (shutil.which("crest") and shutil.which("xtb")), reason="needs crest and xtb")
def test_msreact_fragments_keep_the_atoms_in_order(tmp_path):
    seed = Structure.from_xyz(ETHANOL)
    products = get_generator("crest-msreact").propose(seed, max_products=4, options={"mode": "fragments"})
    assert 1 <= len(products) <= 4
    assert all(list(p.symbols) == list(seed.symbols) and p.charge == 0 for p in products)


def test_the_network_card_offers_four_methods_and_msreact_builds_its_command():
    methods = [(op, m) for op in OPERATIONS.values() if (op.family or {}).get("key") == "network" for m in op.methods]
    labels = [m["label"] for _, m in sorted(methods, key=lambda x: x[1]["rank"])]
    assert labels == ["Bond rules", "CREST msreact", "Hessian sampling", "Basin hopping"]
    assert "nanoreactor" not in OPERATIONS                        # the placeholder is now this method
    op = OPERATIONS["graph-enumeration"]
    p = op.parse_params({"generator": "crest-msreact", "msreact_mode": "isomers", "msreact_nshifts": 2})

    class Ctx:
        output_dir = "/out"

        def snapshot_structure(self, sid, name):
            return f"/in/{name}.xyz"

        def common_flags(self):
            return ["--charge", "0"]

        structures = ["s_1"]

    argv = op.build(Ctx(), p)
    i = argv.index("--generator")
    assert argv[i + 1] == "crest-msreact" and "mode=isomers" in argv and "nshifts=2" in argv
    assert "--n-break" not in argv and argv[-2:] == ["--output", "/out"]
    assert "--generator" not in op.build(Ctx(), op.parse_params({}))   # bond rules: the CLI default


def test_ts_and_channels_are_one_card():
    """A TS from these endpoints and a sampled reaction-channels search are
    the two methods of one "Transition state" card."""
    ts, ch = OPERATIONS["ts"], OPERATIONS["channels"]
    assert ts.family == ch.family and ts.family["title"] == "Transition state"
    assert [m["label"] for m in ts.methods + ch.methods] == ["These endpoints", "Sample conformers and mappings"]
