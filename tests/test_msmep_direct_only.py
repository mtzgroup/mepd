"""path_min_inputs.direct_only: a piece of a split runs if at least one of
its ends is a queried species (any conformer or stereo variant); a piece
between two other species is not run."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.TreeNode import TreeNode
from tests.test_msmep_same_pair_split_limit import (
    _FakeMsmepEngine, _chain_from, _fake_elem_step_result, _msmep, _topology_structure,
)


def _other_species() -> Structure:
    # H on C instead of O: neither "a" (C-O-H) nor "b" (C-O ... H).
    return Structure(geometry=np.array([[0.0, 0.0, 0.0], [2.30, 0.0, 0.0], [-2.05, 0.0, 0.0]]),
                     symbols=["C", "O", "H"], charge=0, multiplicity=1)


def _run(monkeypatch, direct_only: bool, pieces, root=None):
    msmep = _msmep()
    msmep.inputs.engine = _FakeMsmepEngine()
    msmep.inputs.path_min_inputs.direct_only = direct_only
    root = root or _chain_from(_topology_structure("a"), _topology_structure("b"))
    monkeypatch.setattr(msmep, "run_minimize_chain",
                        lambda input_chain: (SimpleNamespace(chain_trajectory=[input_chain]),
                                             _fake_elem_step_result([])))
    calls = []

    def split(chain, split_method, minimization_results):
        calls.append(1)
        return pieces() if len(calls) == 1 else []   # only the root splits

    monkeypatch.setattr(msmep, "make_sequence_of_chains", split)
    return msmep.run_recursive_minimize(root, max_depth=1)


def test_a_split_whose_pieces_all_avoid_the_queried_species_ends_the_branch(monkeypatch, tmp_path):
    c = _other_species()
    d = Structure(geometry=np.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0], [9.0, 0.0, 0.0]]),  # all apart
                  symbols=["C", "O", "H"], charge=0, multiplicity=1)
    tree = _run(monkeypatch, True, lambda: [_chain_from(c, d)])
    assert tree.leaf_status == "offtarget_split_rejected"
    assert tree.children == [] and tree.data is None       # nothing left to TS-optimize
    tree.write_to_disk(tmp_path / "tree")
    assert (tmp_path / "tree" / "node_0_rejected.xyz").exists()
    assert not (tmp_path / "tree" / "node_0.xyz").exists()


def test_legs_with_one_queried_end_run(monkeypatch):
    c = _other_species()
    tree = _run(monkeypatch, True, lambda: [_chain_from(_topology_structure("a"), c),
                                             _chain_from(c, _topology_structure("b"))])
    assert getattr(tree, "leaf_status", None) != "offtarget_split_rejected"
    assert len(tree.children) == 2 and not getattr(tree, "rejected_chains", None)


def test_splits_through_conformers_of_the_queried_endpoints_are_kept(monkeypatch):
    a_conf = _topology_structure("a", offset=0.05)   # same bonds as "a"
    tree = _run(monkeypatch, True, lambda: [_chain_from(_topology_structure("a"), a_conf),
                                             _chain_from(a_conf, _topology_structure("b"))])
    assert getattr(tree, "leaf_status", None) != "offtarget_split_rejected"
    assert len(tree.children) == 2


def test_without_direct_only_the_intermediate_is_followed(monkeypatch):
    c = _other_species()
    tree = _run(monkeypatch, False, lambda: [_chain_from(_topology_structure("a"), c),
                                              _chain_from(c, _topology_structure("b"))])
    assert len(tree.children) == 2


def test_deeper_splits_are_judged_against_the_root_query():
    msmep = _msmep()
    msmep.inputs.path_min_inputs.direct_only = True
    a, b, c = (StructureNode(structure=s) for s in
               (_topology_structure("a"), _topology_structure("b"), _other_species()))
    child = _chain_from(_topology_structure("a", offset=0.05), _topology_structure("b"))
    child._split_ancestors = [(a, b)]                   # queried pair: a -> b
    kept, off = msmep._split_pieces_on_target(child, [child])
    assert kept == [child] and off == []
    via_c = _chain_from(a.structure, c.structure)
    d = Structure(geometry=np.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0], [9.0, 0.0, 0.0]]),
                  symbols=["C", "O", "H"], charge=0, multiplicity=1)
    c_to_d = _chain_from(c.structure, d)
    kept, off = msmep._split_pieces_on_target(child, [via_c, c_to_d, child])
    assert kept == [via_c, child] and len(off) == 2   # only the leg between two other species is dropped


def test_only_the_leg_between_two_other_species_is_dropped(monkeypatch, tmp_path):
    """A -> B splitting into A -> C, C -> D, D -> A', A' -> B runs A -> C,
    D -> A' and A' -> B; only C -> D is not run."""
    a, b = _topology_structure("a"), _topology_structure("b")
    a_conf = _topology_structure("a", offset=0.05)
    c = _other_species()
    d = Structure(geometry=np.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0], [9.0, 0.0, 0.0]]),  # all apart
                  symbols=["C", "O", "H"], charge=0, multiplicity=1)
    legs = lambda: [_chain_from(a, c), _chain_from(c, d), _chain_from(d, a_conf), _chain_from(a_conf, b)]
    tree = _run(monkeypatch, True, legs)
    assert len(tree.children) == 3
    starts = [np.asarray(ch.data.chain_trajectory[-1][0].coords) for ch in tree.children]
    expected = [StructureNode(structure=s).coords for s in (a, d, a_conf)]
    assert all(np.allclose(x, y) for x, y in zip(starts, expected))
    assert len(tree.rejected_chains) == 1
    record = TreeNode(data=None, children=[], index=0)   # the fake search data can't be written
    record.rejected_chains = tree.rejected_chains
    record.write_to_disk(tmp_path / "tree")
    assert sorted(p.name for p in (tmp_path / "tree").glob("node_0_rejected_*.xyz")) == ["node_0_rejected_0.xyz"]


def _fake_tree(root, adj, files):
    root.mkdir(parents=True)
    np.savetxt(root / "adj_matrix.txt", np.array(adj, dtype=float))
    for name in files:
        (root / name).write_text("1\n\nH 0 0 0\n")


def test_trees_with_rejected_branches_reload_without_the_parent_becoming_a_leaf(tmp_path, monkeypatch):
    """Root 0 split into 1 and 2; both branches were not run (direct_only).
    The rejected files must not be read as searches, and the root must not
    turn into a leaf (it would be TS-optimized as if it were a step)."""
    import mepd.TreeNode as tn

    tree = tmp_path / "pair" / "tree"
    _fake_tree(tree, [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
               ["node_0.xyz", "node_1_rejected.xyz", "node_1_rejected_0.xyz", "node_2_rejected.xyz",
                "node_0_rejected_0.xyz"])
    read = []
    monkeypatch.setattr(tn.NEB, "read_from_disk", staticmethod(lambda fp, **kw: read.append(fp.name) or "search"))
    t = TreeNode.read_from_disk(tree)
    assert read == ["node_0.xyz"]
    assert [c.leaf_status for c in t.children] == ["offtarget_split_rejected"] * 2
    assert not t.is_leaf and t.ordered_leaves == []


def test_direct_only_counts_pairs_left_with_nothing_to_search(tmp_path):
    from mepd.cli_common import _completed_tree_dirs, direct_only_counts

    pairs = tmp_path / "pairs"
    _fake_tree(pairs / "none_left" / "tree", [[1, 1], [0, 0]],
               ["node_0.xyz", "node_1_rejected.xyz", "node_1_rejected_0.xyz"])          # root kept, its only leg dropped
    _fake_tree(pairs / "root_rejected" / "tree", [[0]], ["node_0_rejected.xyz", "node_0_rejected_0.xyz"])
    _fake_tree(pairs / "partly" / "tree", [[1, 1, 1], [0, 1, 0], [0, 0, 0]],
               ["node_0.xyz", "node_1.xyz", "node_0_rejected_0.xyz"])                    # one leg ran, one dropped
    _fake_tree(pairs / "plain" / "tree", [[1]], ["node_0.xyz"])
    assert direct_only_counts(pairs) == {"legs_not_run": 3, "pairs_not_characterized": 2,
                                         "pairs_partly_characterized": 1}
    assert sorted(t.parent.name for t in _completed_tree_dirs(pairs)) == ["partly", "plain"]


def test_legs_to_a_stereo_variant_of_an_endpoint_still_run():
    """A split point with the endpoint's bonds but other stereochemistry
    (e.g. trans- vs cis-cyclohexene) can bound the leg holding the direct
    TS, so that leg runs; a genuinely different species does not."""
    from mepd.cli_common import _load_structure_from_smiles_or_xyz

    def node(smi):
        return StructureNode(structure=_load_structure_from_smiles_or_xyz(smi, None, None))

    trans, cis, other = node("C/C=C/C"), node("C/C=C\\C"), node("C=CCC")
    msmep = _msmep()
    msmep.inputs.path_min_inputs.direct_only = True
    root = Chain.model_validate({"nodes": [trans, other], "parameters": ChainInputs()})
    to_variant = Chain.model_validate({"nodes": [trans, cis], "parameters": ChainInputs()})
    kept, off = msmep._split_pieces_on_target(root, [to_variant])
    assert kept == [to_variant] and off == []

    root_b = Chain.model_validate({"nodes": [trans, cis], "parameters": ChainInputs()})
    to_other = Chain.model_validate({"nodes": [trans, other], "parameters": ChainInputs()})
    kept, off = msmep._split_pieces_on_target(root_b, [to_other])
    assert kept == [to_other] and off == []           # one queried end is enough
    between_others = Chain.model_validate({"nodes": [other, node("C=C(C)C")], "parameters": ChainInputs()})
    kept, off = msmep._split_pieces_on_target(root_b, [between_others])
    assert kept == [] and len(off) == 2
