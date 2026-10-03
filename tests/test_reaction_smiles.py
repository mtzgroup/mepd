"""Reaction SMILES ("reactants>>products") as one input for both endpoints:
the parser (mepd/reaction_smiles.py) and `--reaction` on `mepd run` and
`mepd channels`."""
import pytest
import typer
from rdkit import Chem

from mepd.reaction_smiles import (
    ReactionSmilesError, is_reaction_smiles, reaction_pair, reaction_structures, split_reaction,
)

from test_cli_channels import _call_channels
from test_cli_run import _call_run


def _heavy_smiles(mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def _bonded(mol, i, j) -> bool:
    return mol.GetBondBetweenAtoms(i, j) is not None


def test_split_and_detect():
    assert split_reaction("CC>>C=C.[H][H]") == ("CC", "", "C=C.[H][H]")
    assert split_reaction("A>B>C") == ("A", "B", "C")
    assert is_reaction_smiles("CC>>C=C") and is_reaction_smiles("CC>[Pd]>C=C")
    assert not is_reaction_smiles("CCO") and not is_reaction_smiles("C>C")
    with pytest.raises(ReactionSmilesError, match="not a reaction SMILES"):
        split_reaction("C>C")
    with pytest.raises(ReactionSmilesError, match="both the reactant and the product"):
        split_reaction("C>>")


def test_unmapped_reaction_gets_one_atom_order_on_both_sides():
    pair = reaction_pair("CC(=O)C>>CC(O)=C")
    r, p = pair.reactant, pair.product
    assert pair.source == "slapmapper" and pair.rmsd >= 0
    assert _heavy_smiles(r) == "CC(C)=O" and _heavy_smiles(p) == "C=C(C)O"
    assert [a.GetSymbol() for a in r.GetAtoms()] == [a.GetSymbol() for a in p.GetAtoms()]
    # The carbonyl O stays bonded to the same carbon on both sides.
    o = next(a.GetIdx() for a in r.GetAtoms() if a.GetSymbol() == "O")
    c = r.GetAtomWithIdx(o).GetNeighbors()[0].GetIdx()
    assert _bonded(p, o, c)
    # Exactly one hydrogen moves (C -> O): every other H keeps its carbon.
    moved = [a.GetIdx() for a in r.GetAtoms() if a.GetSymbol() == "H"
             and r.GetAtomWithIdx(a.GetIdx()).GetNeighbors()[0].GetIdx()
             != p.GetAtomWithIdx(a.GetIdx()).GetNeighbors()[0].GetIdx()]
    assert len(moved) == 1 and p.GetAtomWithIdx(moved[0]).GetNeighbors()[0].GetIdx() == o


def test_map_numbers_in_the_input_are_used():
    # The enol's CH2= is the carbon mapped :1 (written first in the reactant); a
    # mapper could as well have picked the other methyl. The given map wins.
    pair = reaction_pair("[CH3:1][C:2](=[O:3])[CH3:4]>>[CH3:4][C:2]([OH:3])=[CH2:1]")
    assert pair.source == "given"
    r, p = pair.reactant, pair.product
    assert [a.GetSymbol() for a in r.GetAtoms()][:4] == ["C", "C", "O", "C"]    # input order
    assert p.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble() == 2.0
    assert p.GetBondBetweenAtoms(3, 1).GetBondTypeAsDouble() == 1.0
    assert all(a.GetAtomMapNum() == 0 for m in (r, p) for a in m.GetAtoms())


def test_agents_are_reported_not_used():
    pair = reaction_pair("CC(=O)O.OCC>[H+]>CC(=O)OCC.O")
    assert pair.agents == "[H+]" and any("[H+]" in n for n in pair.notes)
    assert r"[H+]" not in _heavy_smiles(pair.reactant)


def test_unbalanced_and_charge_mismatch_are_refused():
    with pytest.raises(ReactionSmilesError, match="not balanced: reactants are C2H6, products C3H8"):
        reaction_pair("CC>>CCC")
    with pytest.raises(ReactionSmilesError, match="different charges"):
        reaction_pair("[NH4+]>>N.[H]")
    with pytest.raises(ReactionSmilesError, match="could not read"):
        reaction_pair("C1CC>>CCC")


def test_several_molecules_on_one_side_are_packed_not_lined_up():
    """'C=C.O.O': three molecules, whose centres a row would put on one line."""
    import networkx as nx
    import numpy as np

    from mepd.nodes.node import StructureNode

    start, *_ = reaction_structures("C=C.O.O>>CCO.O")
    xyz = np.asarray(start.geometry).reshape(-1, 3)
    frags = list(nx.connected_components(StructureNode(structure=start).graph))
    assert sorted(len(f) for f in frags) == [3, 3, 6]
    a, b, c = (xyz[sorted(f)].mean(axis=0) for f in frags)
    area = np.linalg.norm(np.cross(b - a, c - a)) / 2
    assert area > 0.5                                   # bohr^2: the centres are not collinear


def test_structures_share_symbols_and_charge():
    start, end, pair = reaction_structures("C[O-].CBr>>COC.[Br-]")
    assert start.symbols == end.symbols and start.charge == end.charge == -1 == pair.charge
    assert reaction_structures("C=C.[H][H]>>CC", multiplicity=1)[0].multiplicity == 1


# ------------------------------------------------------------ CLI
class _Stop(Exception):
    pass


def _stop_after_endpoints(monkeypatch, module):
    seen = {}

    def fake_minimize(start_node, end_node, run_inputs):
        seen["start"], seen["end"] = start_node.structure, end_node.structure
        raise _Stop

    monkeypatch.setattr(module, "_minimize_endpoints", fake_minimize)
    return seen


def test_run_takes_a_reaction_and_minimizes_its_embedded_ends(tmp_path, monkeypatch):
    import mepd.cli as cli_module

    seen = _stop_after_endpoints(monkeypatch, cli_module)
    with pytest.raises(_Stop):
        _call_run(reaction="CC(=O)C>>CC(O)=C", minimize_ends=None, output=tmp_path / "out")
    assert seen["start"].symbols == seen["end"].symbols and len(seen["start"].symbols) == 10


def test_channels_takes_a_reaction(tmp_path, monkeypatch):
    import mepd.cli_channels as channels_module

    seen = _stop_after_endpoints(monkeypatch, channels_module)
    with pytest.raises(_Stop):
        _call_channels(reaction="C=CCOC=C>>C=CCCC=O", minimize_ends=True, output=tmp_path / "out")
    assert seen["start"].symbols == seen["end"].symbols


@pytest.mark.parametrize("call", [_call_run, _call_channels])
def test_endpoint_options_are_either_a_pair_or_a_reaction(tmp_path, call):
    with pytest.raises(typer.BadParameter, match="not both"):
        call(start="CCO", end="CC=O", reaction="CCO>>CC=O.[H][H]", output=tmp_path / "out")
    with pytest.raises(typer.BadParameter, match="Give both --start and --end"):
        call(start="CCO", output=tmp_path / "out")
    with pytest.raises(typer.BadParameter, match="not balanced"):
        call(reaction="CC>>CCC", minimize_ends=False, output=tmp_path / "out")
