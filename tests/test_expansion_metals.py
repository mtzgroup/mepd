"""Network expansion with metals and ions: metals are ions bound by dative
bonds, kept out of the main-group Lewis check."""
import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

from mepd.cli_common import _load_structure_from_smiles_or_xyz
from mepd.discovery.network_expansion import enumerate_bond_changes, graph_edges, lewis_smiles
from mepd.nodes.node import StructureNode


def _proposals(smiles):
    s = _load_structure_from_smiles_or_xyz(smiles, None, None)
    node = StructureNode(structure=s)
    props, _ = enumerate_bond_changes(list(s.symbols), np.asarray(s.geometry) / ANGSTROM_TO_BOHR, graph_edges(node),
                                      charge=int(s.charge), multiplicity=int(s.multiplicity))
    return [p.smiles for p in props]


def test_metal_ions_get_a_lewis_structure_with_dative_bonds():
    # Acetaldehyde (C0 C1 O2, H3-H6) with Mg2+ (7) on its O, then free; ethylene (C0 C1, H2-H5) on Cu+ (6).
    ald = ["C", "C", "O", "H", "H", "H", "H", "Mg"]
    ald_bonds = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6)]
    assert "->[Mg+2]" in lewis_smiles(ald, ald_bonds + [(2, 7)], charge=2)
    assert lewis_smiles(ald, ald_bonds, charge=2) == "CC=O.[Mg+2]"
    eth = ["C", "C", "H", "H", "H", "H", "Cu"]
    eth_bonds = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    assert "->[Cu+]" in lewis_smiles(eth, eth_bonds + [(0, 6)], charge=1)
    # Not a coordination bond: a metal on an sp3 carbon's hydrogen.
    assert lewis_smiles(["C", "H", "H", "H", "H", "Mg"], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)], charge=2) is None


def test_expansion_proposes_products_for_systems_with_metals():
    assert _proposals("CC=O.[Mg+2]")               # was empty: every product failed the Lewis check
    assert _proposals("[Li+].CC(=O)OC")
    cu = _proposals("[Cu+].C=C")
    assert any("->[Cu+]" in s for s in cu)
    assert _proposals("CC(=O)[O-]")                # organic ions kept working
