from __future__ import annotations

import numpy as np
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.pathminimizers.gsm import GSM

SYMBOLS = ["H", "H"]


def _node(z: float) -> StructureNode:
    node = StructureNode(structure=Structure(symbols=SYMBOLS, geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, z]]),
                                             charge=0, multiplicity=1))
    node._cached_energy = -1.0
    return node


def _block(it: int, energies_kcal: list[float], complete: bool = True) -> str:
    """One iteration in the patched molecularGSM's scratch/iterations.xyz format."""
    lines = [f"ITER {it} NODES {len(energies_kcal)} GRADRMS 0.01"]
    for k, v in enumerate(energies_kcal):
        lines += [" 2", f" {v}", "  H 0.0 0.0 0.0", f"  H 0.0 0.0 {0.7 + 0.1 * k + 0.01 * it}"]
    if complete:
        lines.append(f"END_ITER {it}")
    return "\n".join(lines) + "\n"


def test_iteration_strings_reads_complete_iterations_in_order_once(tmp_path):
    (tmp_path / "scratch").mkdir()
    fp = tmp_path / "scratch" / "iterations.xyz"
    fp.write_text(_block(1, [0.0, 5.0, 2.0]) + _block(2, [0.0, 4.0, 2.0]) + _block(3, [0.0, 3.0, 2.0], complete=False))
    gsm = GSM.__new__(GSM)
    r, p, state = _node(0.7), _node(0.9), {"seen": 0}
    got = gsm._iteration_strings(tmp_path, "", state, r, p, -1.0, ChainInputs())
    assert [it for it, _ in got] == [1, 2]            # the half-written iteration 3 waits
    it2 = got[1][1]
    assert isinstance(it2, Chain) and len(it2) == 3
    assert it2[0] is r and it2[-1] is p              # real endpoints pinned
    assert abs((it2[1]._cached_energy - (-1.0)) * 627.509474 - 4.0) < 1e-3
    assert np.isclose(it2[1].coords[1][2] * 0.529177210903, 0.82, atol=1e-6)
    # nothing is returned twice; iteration 3 once it is complete
    fp.write_text(_block(1, [0.0, 5.0, 2.0]) + _block(2, [0.0, 4.0, 2.0]) + _block(3, [0.0, 3.0, 2.0]))
    assert [it for it, _ in gsm._iteration_strings(tmp_path, "", state, r, p, -1.0, ChainInputs())] == [3]


def test_iteration_strings_empty_without_the_patched_binary_file(tmp_path):
    gsm = GSM.__new__(GSM)
    assert gsm._iteration_strings(tmp_path, " opt_iter:  1 totalgrad: 1 gradrms: 0.1 tgrads: 20", {"seen": 0},
                                  _node(0.7), _node(0.9), -1.0, ChainInputs()) == []
