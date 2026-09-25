import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qcconst.constants import ANGSTROM_TO_BOHR
from typer.testing import CliRunner

import mepd.cli_common as cli_common
from mepd.cli import app
from mepd.cli_common import _load_structure_from_smiles_or_xyz
from mepd.discovery.network_expansion import (
    _perceived_edges, embed_product, enumerate_bond_changes, expand_network, graph_edges, lewis_smiles,
)
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode


def _node(smiles):
    return StructureNode(structure=_load_structure_from_smiles_or_xyz(smiles, None, None))


def _ang(node):
    return np.asarray(node.coords) / ANGSTROM_TO_BOHR


class _GraphEngine:
    """Optimization returns the input geometry; the energy is minus the
    number of bonds (mHa), so the fake surface still orders species."""

    def __init__(self):
        self.optimized = 0

    def _energize(self, node):
        node = node.copy()
        node._cached_energy = -1e-3 * len(graph_edges(node))
        return node

    def compute_energies(self, nodes):
        for n in nodes:
            n._cached_energy = self._energize(n)._cached_energy
        return [n._cached_energy for n in nodes]

    def compute_geometry_optimization(self, node, keywords=None):
        self.optimized += 1
        return [self._energize(node)]


def test_lewis_filter_keeps_closed_shell_products_only_by_default():
    assert lewis_smiles(["C", "H", "H", "H", "H"], [(0, 1), (0, 2), (0, 3), (0, 4)]) == "C"
    carbene = ["C", "H", "H"], [(0, 1), (0, 2)]
    assert lewis_smiles(*carbene) is None
    assert lewis_smiles(*carbene, allow_radicals=True) is not None
    hnc = ["C", "N", "H"], [(0, 1), (1, 2)]
    assert lewis_smiles(*hnc) is None  # C-≡N+-H needs separated charges
    assert lewis_smiles(*hnc, allow_zwitterions=True) == "[C-]#[NH+]"


def test_enumeration_finds_ethanol_decompositions_fewest_changes_first():
    n = _node("CCO")
    props, stats = enumerate_bond_changes(list(n.symbols), _ang(n), graph_edges(n))
    smiles = [p.smiles for p in props]
    assert {"C=C.O", "CC=O.[H][H]", "COC"} <= set(smiles)
    assert "CCO" not in smiles and len(set(smiles)) == len(smiles)
    changes = [len(p.broken) + len(p.formed) for p in props]
    assert changes == sorted(changes)
    assert not stats["clipped"]
    assert len(enumerate_bond_changes(list(n.symbols), _ang(n), graph_edges(n), max_products=2)[0]) == 2
    # Nothing to form within 0.5 A, and bare bond breaks leave radicals.
    assert enumerate_bond_changes(list(n.symbols), _ang(n), graph_edges(n), form_distance=0.5)[0] == []


@pytest.mark.parametrize("smiles, product", [("CCO", "C=C.O"), ("C#N", "[C-]#[NH+]")])
def test_embedded_guess_has_the_proposed_bonds(smiles, product):
    n = _node(smiles)
    props, _ = enumerate_bond_changes(list(n.symbols), _ang(n), graph_edges(n), allow_zwitterions=True)
    p = next(p for p in props if p.smiles == product)
    bonds = (graph_edges(n) - set(p.broken)) | set(p.formed)
    assert _perceived_edges(n.symbols, embed_product(list(n.symbols), _ang(n), bonds)) == bonds


def test_expansion_rounds_add_species_and_link_known_ones_without_reoptimizing():
    eng = _GraphEngine()
    result = expand_network(_node("C#N"), eng, rounds=2, allow_zwitterions=True)
    assert [s.smiles for s in result.species] == ["C#N", "[C-]#[NH+]"]
    assert [(e.source, e.target, e.outcome) for e in result.edges] == [(0, 1, "new_species"), (1, 0, "known_species")]
    assert eng.optimized == 1  # the back-reaction to HCN is not optimized again
    assert result.connections() == [(0, 1)]


def test_energy_window_stops_expansion_of_high_species():
    result = expand_network(_node("C#N"), _GraphEngine(), rounds=2, allow_zwitterions=True, energy_window_kcal=-1.0)
    assert len(result.rounds) == 1 and result.rounds[0]["expanded_next"] == []


def test_products_from_another_tool_by_file_must_be_atom_mapped(tmp_path):
    seed = _node("CCO")
    props, _ = enumerate_bond_changes(list(seed.symbols), _ang(seed), graph_edges(seed))
    p = next(p for p in props if p.smiles == "C=C.O")
    bonds = (graph_edges(seed) - set(p.broken)) | set(p.formed)
    xyz = embed_product(list(seed.symbols), _ang(seed), bonds)
    fp = tmp_path / "products.xyz"
    fp.write_text(f"{len(xyz)}\n\n" + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(seed.symbols, xyz)))
    result = expand_network(seed, _GraphEngine(), products_file=str(fp))
    assert [s.smiles for s in result.species] == ["CCO", "C=C.O"]
    bad = tmp_path / "bad.xyz"
    bad.write_text("3\n\nO 0 0 0\nH 0 0 1\nH 0 1 0\n")
    with pytest.raises(ValueError, match="atom-mapped"):
        expand_network(seed, _GraphEngine(), products_file=str(bad))


def test_custom_generator_by_import_path(monkeypatch):
    seen = {}

    def generate(structure, scale=1.0):
        seen["scale"] = scale
        geom = np.array(structure.geometry) * scale
        return [structure.model_copy(update={"geometry": geom})]

    monkeypatch.setitem(sys.modules, "my_generators", SimpleNamespace(propose=generate))
    result = expand_network(_node("CCO"), _GraphEngine(), generator="my_generators:propose",
                            generator_options={"scale": 1.0})
    assert seen == {"scale": 1.0}
    assert [e.outcome for e in result.edges] == ["reverted"]  # the "product" is ethanol again


def test_cli_writes_species_and_reactions(tmp_path, monkeypatch):
    def run_inputs(_path):
        ri = RunInputs()
        ri.engine = _GraphEngine()
        return ri

    monkeypatch.setattr(cli_common, "_open_run_inputs", run_inputs)
    out = tmp_path / "out"
    res = CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions", "--rounds", "2", "-o", str(out)])
    assert res.exit_code == 0, res.output
    summary = json.loads((out / "summary.json").read_text())
    assert [s["smiles"] for s in summary["species"]] == ["C#N", "[C-]#[NH+]"]
    assert summary["outcomes"] == {"new_species": 1, "known_species": 1}
    assert summary["connections"] == [[0, 1]]
    assert (out / "species.xyz").exists() and (out / "species" / "species_1.xyz").exists()


def test_web_operation_runs_the_cli_and_reads_its_output(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from mepd.web.operations import get_operation
    from mepd.web.results import collect, detect_operation

    op = get_operation("graph-enumeration")
    assert op.available and op.cli_problem() is None

    def run_inputs(_path):
        ri = RunInputs()
        ri.engine = _GraphEngine()
        return ri

    monkeypatch.setattr(cli_common, "_open_run_inputs", run_inputs)
    out = tmp_path / "out"
    CliRunner().invoke(app, ["discovery", "expand", "C#N", "--allow-zwitterions", "-o", str(out)])
    assert detect_operation(out) == "graph-enumeration"
    result = collect({"op": "graph-enumeration", "output_dir": str(out), "external": True})
    assert result["headline"].startswith("2 species")
    assert "[C-]#[NH+]" in result["groups"][0]["entries"][1]["note"]


def test_live_view_animates_each_reaction_from_source_to_product(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(tmp_path))
    expand_network(_node("C#N"), _GraphEngine(), rounds=2, allow_zwitterions=True)
    streams = {fp.stem: json.loads(fp.read_text()) for fp in sorted(tmp_path.glob("rxn*.json"))}
    assert list(streams) == ["rxn0001", "rxn0002"]
    forward, back = streams["rxn0001"], streams["rxn0002"]
    assert forward["kind"] == "morph" and forward["finished"] and forward["outcome"] == "new species"
    assert back["outcome"] == "known species"  # linked without re-optimizing, still animated
    frames = forward["geometry"]["frames"]
    assert len(frames) > 2  # a geodesic interpolation, not just the two ends
    first, last = (np.array([[float(v) for v in line.split()[1:]] for line in f.strip().splitlines()[2:]])
                   for f in (frames[0], frames[-1]))
    h_near = lambda x: np.argmin(np.linalg.norm(x[:2] - x[2], axis=1))  # H next to C (0) or N (1)
    assert (h_near(first), h_near(last)) == (0, 1)
    assert forward["plot"]["reactant_smiles"] == "C#N" and forward["plot"]["product_smiles"] == "[C-]#[NH+]"
