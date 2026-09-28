"""Rerunning channels into the same folder with more pairs per mechanism:
every earlier (reactant, product, mechanism) keeps its pair label -- its
finished work is found and skipped by that label -- and new ones get new
labels (never one an earlier, different pair had)."""
import json
from types import SimpleNamespace

from mepd import cli_channels
from mepd.cli_common import _load_structure_from_smiles_or_xyz
from mepd.nodes.node import StructureNode


def _expand(tmp_path, monkeypatch, cap):
    s = _load_structure_from_smiles_or_xyz("CCO", None, None)
    structures = [StructureNode(structure=s) for _ in range(5)]      # 2 start + 3 end conformers
    candidates = [(i, j) for i in (0, 1) for j in (2, 3, 4)]

    def choices(start, end, metric, run_inputs, max_variants_per_mechanism=None):
        # Two mechanisms per pair; one always needs a remapped product. Scores fixed per pair.
        i = next(k for k, st in enumerate(structures) if st.structure is start)
        j = next(k for k, st in enumerate(structures) if st.structure is end and k >= 2)
        base = 0.1 * i + 0.01 * j
        return [SimpleNamespace(key="mech A", score=base, n_variants=1,
                                winner=SimpleNamespace(end_structure=end.model_copy(), atom_map=[1])),
                SimpleNamespace(key="mech B", score=1 - base, n_variants=2,
                                winner=SimpleNamespace(end_structure=end.model_copy(), atom_map=[0]))]

    monkeypatch.setattr("mepd.atom_mapping_selection.select_per_mechanism", choices)
    run_inputs = SimpleNamespace(atom_mapping_inputs=SimpleNamespace(metric="endpoint-rmsd", n_candidates=4))
    _, cands, _ = cli_channels._expand_pairs_by_mechanism(structures, candidates, True, run_inputs,
                                                           pairs_per_mechanism=cap, workers=1, output=tmp_path)
    table = json.loads((tmp_path / "pair_mechanisms.json").read_text())
    return {(r["start_conformer"], r["end_structure"], r["mechanism"]): r["pair"]
            for r in table if not r.get("not_in_this_run")}


def test_more_pairs_per_mechanism_keeps_earlier_labels(tmp_path, monkeypatch):
    first = _expand(tmp_path, monkeypatch, 1)
    second = _expand(tmp_path, monkeypatch, 3)
    assert len(first) == 2 and len(second) == 6
    for key, label in first.items():
        assert second[key] == label                                  # the same pair, the same folder
    new = {k: v for k, v in second.items() if k not in first}
    assert not set(new.values()) & set(first.values())             # no earlier label is re-used
    # Going back down keeps the dropped rows' labels reserved in the table.
    _expand(tmp_path, monkeypatch, 1)
    table = json.loads((tmp_path / "pair_mechanisms.json").read_text())
    assert {r["pair"] for r in table} >= set(second.values())
