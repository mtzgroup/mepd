from __future__ import annotations

import numpy as np
from qcdata import Structure

from mepd.nodes.node import StructureNode
from mepd.pathminimizers.gsm import GSM, _inpfileq_text
from qcconst.constants import ANGSTROM_TO_BOHR
from types import SimpleNamespace


def _hcn_node(c_pos: float, n_pos: float, energy: float) -> StructureNode:
    """A toy linear HCN/HNC-migration node: H fixed at the origin, C and N
    on the z-axis at the given (Angstrom) positions."""
    node = StructureNode(
        structure=Structure(
            symbols=["H", "C", "N"],
            geometry=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, c_pos],
                    [0.0, 0.0, n_pos],
                ],
                dtype=float,
            )
            * ANGSTROM_TO_BOHR,
            charge=0,
            multiplicity=1,
        )
    )
    node._cached_energy = energy
    return node


def test_write_string_blocks_format_round_trips(tmp_path):
    """The block format `_write_string_blocks` writes must be exactly what
    the compiled `gsm` binary's `read_string` (GSM/gstring.cpp) expects, and
    what `_parse_string_blocks` already reads for `stringfile.xyz` output --
    no blank lines between blocks, each block is `<natoms>` /
    `<energy kcal/mol>` / `<natoms> "SYMBOL x y z"` lines."""
    reactant = _hcn_node(1.064, 2.220, energy=-0.5)
    midpoint = _hcn_node(1.6135, 1.6070, energy=-0.3)
    product = _hcn_node(2.163, 0.994, energy=-0.52)
    nodes = [reactant, midpoint, product]
    e_reference = float(reactant.energy)

    fp = tmp_path / "restart.xyz0000"
    GSM._write_string_blocks(fp, nodes, e_reference)

    blocks = GSM._parse_string_blocks(fp.read_text(), natoms_expected=3)
    assert len(blocks) == 3

    _KCAL_PER_HARTREE = 627.5
    for node, (v_kcal, coords) in zip(nodes, blocks):
        expected_kcal = (float(node.energy) - e_reference) * _KCAL_PER_HARTREE
        assert abs(v_kcal - expected_kcal) < 1e-6

        coord_angstrom = np.asarray(node.coords, dtype=float) / ANGSTROM_TO_BOHR
        assert np.allclose(np.asarray(coords), coord_angstrom, atol=1e-6)


def test_write_string_blocks_reactant_energy_is_zero_reference(tmp_path):
    """The reactant's own energy line should read as 0.0 kcal/mol when it is
    used as the reference energy (mirrors how the caller's energy
    reconstruction expects node 0 to sit at `e_reactant + 0`)."""
    reactant = _hcn_node(1.064, 2.220, energy=-0.5)
    product = _hcn_node(2.163, 0.994, energy=-0.5)
    e_reference = float(reactant.energy)

    fp = tmp_path / "restart.xyz0000"
    GSM._write_string_blocks(fp, [reactant, product], e_reference)

    v_kcal, _ = GSM._parse_string_blocks(fp.read_text(), natoms_expected=3)[0]
    assert abs(v_kcal) < 1e-8


def test_inpfileq_text_restart_and_nnodes_override():
    params = SimpleNamespace(nnodes=9)
    text = _inpfileq_text(params, restart=1, nnodes_override=5)
    assert "RESTART                 1" in text
    assert "NNODES                  5" in text

    default_text = _inpfileq_text(params)
    assert "RESTART                 0" in default_text
    assert "NNODES                  9" in default_text
