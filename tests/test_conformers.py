from __future__ import annotations

import numpy as np
import pytest
from qcdata.models.structure import Structure
from rdkit import Chem
from rdkit.Chem import AllChem

from mepd.conformers import ConformerInputs, generate_conformers
from mepd.helper_functions import RMSD
from mepd.nodes.node import StructureNode


def _butane_node(seed: int = 1) -> StructureNode:
    """Butane has real conformational freedom (anti/gauche about the central
    C-C bond) -- a good, cheap real molecule for exercising dedup, unlike a
    rigid toy system that would only ever produce one distinct conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    from qcconst.constants import ANGSTROM_TO_BOHR

    positions = mol.GetConformer().GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    structure = Structure(
        geometry=positions * ANGSTROM_TO_BOHR, symbols=symbols, charge=0, multiplicity=1
    )
    return StructureNode(structure=structure)


def test_generate_conformers_rdkit_finds_multiple_distinct_conformers():
    node = _butane_node()
    confs = generate_conformers(
        node, ConformerInputs(n_conformers=10, n_embed=30, random_seed=0)
    )
    assert 1 < len(confs) <= 10


def test_generate_conformers_respects_n_conformers_cap():
    node = _butane_node()
    confs = generate_conformers(
        node, ConformerInputs(n_conformers=1, n_embed=30, random_seed=0)
    )
    assert len(confs) == 1


def test_generate_conformers_preserves_atom_order_and_reuses_graph():
    node = _butane_node()
    confs = generate_conformers(
        node, ConformerInputs(n_conformers=10, n_embed=30, random_seed=0)
    )
    for conf in confs:
        assert list(conf.structure.symbols) == list(node.structure.symbols)
        assert conf.graph is node.graph


def test_generate_conformers_are_pairwise_distinct_by_rmsd():
    node = _butane_node()
    inputs = ConformerInputs(n_conformers=10, n_embed=30, rmsd_cutoff=0.5, random_seed=0)
    confs = generate_conformers(node, inputs)
    for i in range(len(confs)):
        for j in range(i + 1, len(confs)):
            rmsd = RMSD(confs[i].coords, confs[j].coords)[0]
            assert rmsd >= inputs.rmsd_cutoff


def test_generate_conformers_unknown_backend_raises():
    node = _butane_node()
    with pytest.raises(ValueError):
        generate_conformers(node, ConformerInputs(backend="not-a-backend"))


def test_generate_conformers_crest_backend_not_implemented():
    node = _butane_node()
    with pytest.raises(NotImplementedError):
        generate_conformers(node, ConformerInputs(backend="crest"))
