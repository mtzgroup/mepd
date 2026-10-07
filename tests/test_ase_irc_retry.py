"""Sella's IRC can stop on its first steps off the saddle
(IRCInnerLoopConvergenceFailure -- every g-xTB IRC tried did). The ASE
engine then reruns that direction from the TS with keep_going, which steps
on past it; any other failure still raises."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")


class IRCInnerLoopConvergenceFailure(Exception):
    pass


def _fake_irc(calls: list, fail_always: bool = False):
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import Trajectory

    class FakeIRC:
        def __init__(self, atoms, logfile=None, trajectory=None, keep_going=False, **kw):
            self.atoms, self.trajectory, self.keep_going = atoms, trajectory, keep_going

        def run(self, fmax, steps, direction):
            calls.append((direction, self.keep_going))
            if fail_always or not self.keep_going:
                raise IRCInnerLoopConvergenceFailure()
            with Trajectory(self.trajectory, "w") as t:
                for d in (0.05, 0.10):
                    a = self.atoms.copy()
                    a.positions[0] += d if direction == "forward" else -d
                    a.calc = SinglePointCalculator(a, energy=-1.0, forces=np.zeros((len(a), 3)))
                    t.write(a)

    return FakeIRC


def _ts():
    from qcconst.constants import ANGSTROM_TO_BOHR
    from qcdata import Structure

    from mepd.nodes.node import StructureNode

    xyz = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]) * ANGSTROM_TO_BOHR
    return StructureNode(structure=Structure(symbols=["O", "H", "H"], geometry=xyz, charge=0, multiplicity=1))


def test_an_inner_loop_failure_is_retried_with_keep_going(monkeypatch):
    from ase.calculators.emt import EMT

    import mepd.engines.ase as ase_engine

    calls: list = []
    monkeypatch.setattr(ase_engine, "SellaIRC", _fake_irc(calls))
    chain = ase_engine.ASEEngine(calculator=EMT()).compute_irc_chain(_ts())
    assert calls == [("reverse", False), ("reverse", True), ("forward", False), ("forward", True)]
    assert len(chain) == 5                     # 2 frames each way, and the TS


def test_a_failure_that_keep_going_cannot_fix_still_raises(monkeypatch):
    from ase.calculators.emt import EMT

    import mepd.engines.ase as ase_engine
    from mepd.errors import ElectronicStructureError

    calls: list = []
    monkeypatch.setattr(ase_engine, "SellaIRC", _fake_irc(calls, fail_always=True))
    with pytest.raises(ElectronicStructureError, match="IRCInnerLoopConvergenceFailure"):
        ase_engine.ASEEngine(calculator=EMT()).compute_irc_chain(_ts())
    assert calls == [("reverse", False), ("reverse", True)]
