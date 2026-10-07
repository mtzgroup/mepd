"""Small things a first-time user ran into: SMILES errors in one line,
everyday names for small molecules, a headline rounded like its edge."""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")


def test_smiles_problems_in_one_line():
    from mepd.web.chem import smiles_problem

    assert smiles_problem("CCO") is None
    assert smiles_problem("C1CC(") == "syntax error around position 5"
    assert smiles_problem("C1CC") == "unclosed ring"
    assert smiles_problem("C(C)(C)(C)(C)C") == "atom 1 (C) has 5 bonds, more than it can"
    assert "aromatic ring" in smiles_problem("c1cccc1")


def test_small_molecules_get_their_everyday_names():
    from mepd.web.chem import common_name, smiles_problem

    assert common_name("Cl") == "HCl" and common_name("O") == "water" and common_name("[H]O[H]") == "water"
    assert common_name("OP(=O)(O)O") == "phosphoric acid"
    assert common_name("CC(=O)Oc1ccccc1C(=O)O") is None and common_name("C1CC(") is None
    assert smiles_problem("C1CC(") is not None    # naming does not silence RDKit's messages for good


def test_headline_rounds_like_the_page():
    pytest.importorskip("fastapi")
    from mepd.web.results import _one_decimal

    assert _one_decimal(65.74996) == "65.8"   # stored as 65.75, shown 65.8 on the edge
    assert _one_decimal(65.25) == "65.3"      # ties up, as JavaScript's toFixed
