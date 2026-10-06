"""Atom mapping with a built endpoint (--pair-from start/end): every
candidate's partner is built and minimized, and the pairs are compared, not
the given product (mepd.built_pair_mapping). Also: SLAPMapper's single
suggestion pairs equivalent atoms correctly (acetone -> its enol is one H
moving, not two)."""

from __future__ import annotations

import pytest

pytest.importorskip("slapmapper")
pytest.importorskip("rdkit")


def _embedded(smiles: str, seed: int):
    from qcconst.constants import ANGSTROM_TO_BOHR
    from qcdata import Structure
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return Structure(symbols=[a.GetSymbol() for a in m.GetAtoms()],
                     geometry=m.GetConformer().GetPositions() * ANGSTROM_TO_BOHR, charge=0, multiplicity=1)


def _acetone_and_enol():
    # Same element order (C C C O H...): the enol's numbering moves two H
    # (one between the carbons, one onto O) where one suffices.
    return _embedded("CC(C)=O", 1), _embedded("C=C(C)O", 7)


def test_slapmapper_suggestion_moves_one_h_not_two():
    from mepd.atom_mapping import _bond_changes_under, check_atom_mapping

    start, end = _acetone_and_enol()
    n = len(start.symbols)
    assert _bond_changes_under(start, end, {i: i for i in range(n)}) == 4
    m = check_atom_mapping(start, end)
    assert not m.is_identity and _bond_changes_under(start, end, m.mapping) == 2


def _no_optimizer(monkeypatch):
    """Minimization returns the embedded guess."""
    import mepd.built_pair_mapping as bp

    monkeypatch.setattr(bp, "_minimize", lambda guesses, run_inputs: list(guesses))


def _rank_by_bond_changes(monkeypatch, seen: list):
    """The path comparison stubbed out: fewest bond changes from the source
    wins (records the compared picks)."""
    import mepd.atom_mapping_selection as sel

    def rank(picks, fallback, start_structure, run_inputs, use_path=True):
        seen.extend(picks)
        score = {c.label: sel._bond_changes(start_structure, c.end_structure) for c in picks}
        best = min(picks, key=lambda c: score[c.label])
        return 0.0, best, "stub", score, "bond changes", []

    monkeypatch.setattr(sel, "_rank_picks", rank)


def test_each_methyl_h_gives_its_own_built_product(monkeypatch):
    """Built from acetone, which of the six methyl H moves changes the
    product's geometry: six distinct 2-change bond sets are built and
    compared; the pair comes back in the start's order with one H moved."""
    from mepd.atom_mapping import check_atom_mapping
    from mepd.atom_mapping_selection import _bond_changes
    from mepd.built_pair_mapping import build_and_rank
    from mepd.inputs import RunInputs

    _no_optimizer(monkeypatch)
    seen: list = []
    _rank_by_bond_changes(monkeypatch, seen)
    start, end = _acetone_and_enol()
    ri = RunInputs(engine_name="gxtb")
    out = build_and_rank(start, end, [check_atom_mapping(start, end)], "start", ri, echo=lambda s: None)
    assert out is not None and out.start is start
    assert list(out.end.symbols) == list(start.symbols) and _bond_changes(out.start, out.end) == 2
    one_h = [c for c in seen if _bond_changes(start, c.end_structure) == 2]
    assert len(one_h) == 6     # one per methyl H
    assert len({tuple(map(tuple, c.end_structure.geometry.round(3))) for c in one_h}) == 6


def test_from_the_product_the_pair_comes_back_in_start_order(monkeypatch):
    from mepd.atom_mapping import check_atom_mapping
    from mepd.atom_mapping_selection import _bond_changes
    from mepd.built_pair_mapping import _edges, build_and_rank
    from mepd.inputs import RunInputs

    _no_optimizer(monkeypatch)
    _rank_by_bond_changes(monkeypatch, [])
    start, end = _acetone_and_enol()
    out = build_and_rank(start, end, [check_atom_mapping(start, end)], "end", RunInputs(engine_name="gxtb"),
                         echo=lambda s: None)
    assert out is not None
    assert _bond_changes(out.start, out.end) == 2
    assert _edges(out.start) == _edges(start)     # the built reactant has the reactant's bonds, in its order


def test_partners_whose_bonds_change_are_dropped_with_a_warning(monkeypatch):
    """A built partner that minimizes back to something else is dropped; if
    every fewest-change one is lost, the log says so."""
    import mepd.built_pair_mapping as bp
    from mepd.atom_mapping import check_atom_mapping
    from mepd.inputs import RunInputs

    start, end = _acetone_and_enol()

    def minimize(guesses, run_inputs):     # every one-H product fails to minimize
        return [None if len(bp._edges(start) ^ bp._edges(g)) == 2 else g for g in guesses]

    monkeypatch.setattr(bp, "_minimize", minimize)
    _rank_by_bond_changes(monkeypatch, [])
    lines: list = []
    out = bp.build_and_rank(start, end, [check_atom_mapping(start, end)], "start", RunInputs(engine_name="gxtb"),
                            echo=lines.append)
    assert out is not None and out.n_kept < out.n_bond_sets
    assert any("fewest bond changes (2)" in ln for ln in lines)


@pytest.mark.parametrize("src", ["start", "end"])
def test_channels_keeps_one_built_pair_per_mechanism(monkeypatch, src):
    """channels --pair-from with --atom-mapping: every conformer's partners
    are built for each candidate mapping; --pairs-per-mechanism caps each
    mechanism over conformers and variants. The pairs come back in the
    start's atom order, the sources first like build_partners."""
    import mepd.built_pair_mapping as bp
    from mepd.atom_mapping_selection import _bond_changes
    from mepd.cli_channels import _built_pairs_by_mechanism
    from mepd.inputs import RunInputs
    from mepd.nodes.node import StructureNode

    _no_optimizer(monkeypatch)
    _rank_by_bond_changes(monkeypatch, [])
    monkeypatch.setattr(bp, "_path_score", lambda a, b, ri: float(_bond_changes(a, b)))
    start, end = _acetone_and_enol()
    conformers = [StructureNode(structure=start if src == "start" else end)] * 2
    sources, built = _built_pairs_by_mechanism(conformers, src, StructureNode(structure=start),
                                               StructureNode(structure=end), RunInputs(engine_name="gxtb"),
                                               pairs_per_mechanism=1)
    pairs = list(zip(sources, built)) if src == "start" else list(zip(built, sources))
    changes = sorted(_bond_changes(a.structure, b.structure) for a, b in pairs)
    assert changes[0] == 2                     # the one-H transfer is among them
    assert len(pairs) == len(set(changes))     # one pair per mechanism (capped at 1 across the 2 conformers)
    assert all(list(a.structure.symbols) == list(start.symbols) for a, _ in pairs)


def test_each_variant_of_a_mechanism_is_its_own_search(monkeypatch):
    """Which methyl H moves builds a different product: those are separate
    path searches of one mechanism (up to --pairs-per-mechanism), not one
    best variant per conformer. Repeats -- the same geometry up to atom
    numbering, e.g. a second conformer that is the same -- are dropped."""
    import mepd.built_pair_mapping as bp
    from mepd.atom_mapping_selection import _bond_changes
    from mepd.cli_channels import _built_pairs_by_mechanism
    from mepd.inputs import RunInputs
    from mepd.nodes.node import StructureNode

    _no_optimizer(monkeypatch)
    _rank_by_bond_changes(monkeypatch, [])
    monkeypatch.setattr(bp, "_path_score", lambda a, b, ri: float(_bond_changes(a, b)))
    start, end = _acetone_and_enol()
    same = [StructureNode(structure=start)] * 2       # one conformer twice: its pairs are repeats
    sources, built = _built_pairs_by_mechanism(same, "start", StructureNode(structure=start),
                                               StructureNode(structure=end), RunInputs(engine_name="gxtb"),
                                               pairs_per_mechanism=3)
    pairs = list(zip(sources, built))
    one_h = [(a, b) for a, b in pairs if _bond_changes(a.structure, b.structure) == 2]
    # Several of the six methyl H build different paths (mirror-image ones
    # are one search; how many differ depends on the conformer's symmetry).
    assert 2 <= len(one_h) <= 3
    sigs = [bp._signature(a.structure, b.structure) for a, b in one_h]
    assert all(abs(x - y).max() > 0.05 for i, x in enumerate(sigs) for y in sigs[i + 1:])   # the repeat conformer adds none
    assert len(pairs) - len(one_h) == 3                # the double shift: its 3 best of more
