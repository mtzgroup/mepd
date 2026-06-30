import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import Trajectory
from qcio import Structure

from mepd.chain import Chain
from mepd.engines.ase import ASEEngine
from mepd.helper_functions import compute_irc_chain
from mepd.nodes.node import StructureNode


class _CountingHarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, k: float = 2.0):
        super().__init__()
        self.k = float(k)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, list(properties), system_changes)
        positions = np.asarray(self.atoms.positions, dtype=float)
        if "energy" in properties:
            self.results["energy"] = 0.5 * self.k * float(np.sum(positions**2))
        if "forces" in properties:
            self.results["forces"] = -self.k * positions


def _ts_node() -> StructureNode:
    structure = Structure(
        symbols=["H"],
        geometry=np.array([[0.15, -0.05, 0.02]], dtype=float),
        charge=0,
        multiplicity=1,
    )
    return StructureNode(structure=structure)


def test_ase_engine_irc_chain_uses_sella_irc(monkeypatch):
    calc = _CountingHarmonicCalculator(k=1.5)
    eng = ASEEngine(calculator=calc)

    class _FakeIRC:
        directions: list[str] = []

        def __init__(self, atoms, logfile=None, trajectory=None, **kwargs):
            self.atoms = atoms
            self.logfile = logfile
            self.trajectory = trajectory
            self.kwargs = kwargs

        def run(self, fmax=0.1, steps=1000, direction="forward"):
            _ = (fmax, steps)
            self.__class__.directions.append(str(direction))

            start = self.atoms.copy()
            start.calc = self.atoms.calc
            _ = start.get_potential_energy()
            _ = start.get_forces()

            shift = 0.04 if str(direction).lower() == "forward" else -0.04
            end = self.atoms.copy()
            end.positions = end.positions + shift
            end.calc = self.atoms.calc
            _ = end.get_potential_energy()
            _ = end.get_forces()

            with Trajectory(self.trajectory, "w") as traj:
                traj.write(
                    start,
                    energy=start.get_potential_energy(),
                    forces=start.get_forces(),
                )
                traj.write(
                    end,
                    energy=end.get_potential_energy(),
                    forces=end.get_forces(),
                )

    monkeypatch.setattr("mepd.engines.ase.SellaIRC", _FakeIRC)

    ts = _ts_node()
    irc_chain = eng.compute_irc_chain(
        ts_node=ts,
        keywords={"maxiter": 15, "fmax": 1e-4, "dx": 0.1},
    )

    assert _FakeIRC.directions == ["reverse", "forward"]
    assert len(irc_chain.nodes) == 3
    assert np.allclose(np.asarray(irc_chain.nodes[1].coords), np.asarray(ts.coords), atol=1e-8)
    assert all(node._cached_energy is not None for node in irc_chain.nodes)


def test_compute_irc_chain_dispatches_to_engine_irc_method():
    ts = _ts_node()
    expected_chain = Chain.model_validate({"nodes": [ts.copy(), ts.copy()]})

    class _Engine:
        def __init__(self):
            self.called = False
            self.received_keywords = None

        def compute_energies(self, chain):
            return np.zeros(len(chain), dtype=float)

        def compute_irc_chain(self, ts_node, keywords=None):
            _ = ts_node
            self.called = True
            self.received_keywords = dict(keywords or {})
            return expected_chain

    engine = _Engine()
    out = compute_irc_chain(ts_node=ts, engine=engine, keywords={"maxiter": 25})

    assert out is expected_chain
    assert engine.called is True
    assert engine.received_keywords == {"maxiter": 25}

