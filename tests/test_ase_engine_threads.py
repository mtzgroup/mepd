from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from qcdata import Structure

from mepd.engines.ase import ASEEngine
from mepd.nodes.node import StructureNode


class _SlowCalculator(Calculator):
    """Energy = sum of coordinates, forces = -1. Sleeps after storing its
    results, so a second thread sharing the instance overwrites them before
    the first reads them back (the race a shared ASE calculator has)."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": float(self.atoms.positions.sum()),
                        "forces": -np.ones_like(self.atoms.positions)}
        time.sleep(0.002)


class _UncopyableCalculator(_SlowCalculator):
    def __deepcopy__(self, memo):
        raise TypeError("holds a handle that cannot be copied")


def _nodes(n):
    return [StructureNode(structure=Structure(symbols=["H", "H"], geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4 + 0.01 * k]]),
                                              charge=0, multiplicity=1)) for k in range(n)]


def _energies_from_threads(engine, nodes):
    with ThreadPoolExecutor(max_workers=8) as pool:
        return np.array(list(pool.map(lambda nd: float(engine.compute_energies([nd])[0]), nodes)))


def _expected(nodes):
    from ase.units import Bohr, Hartree

    return np.array([np.asarray(nd.coords).sum() * Bohr / Hartree for nd in nodes])


def test_ase_engine_gives_each_thread_its_own_calculator():
    nodes = _nodes(40)
    got = _energies_from_threads(ASEEngine(calculator=_SlowCalculator()), nodes)
    assert np.allclose(got, _expected(nodes))


def test_ase_engine_serializes_a_calculator_that_cannot_be_copied():
    nodes = _nodes(24)
    got = _energies_from_threads(ASEEngine(calculator=_UncopyableCalculator()), nodes)
    assert np.allclose(got, _expected(nodes))
