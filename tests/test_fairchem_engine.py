import copy
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure

pytest.importorskip("ase")
from ase.calculators.emt import EMT  # noqa: E402
from ase.units import Hartree  # noqa: E402

from mepd.chain import Chain  # noqa: E402
from mepd.cli_common import _unsafe_to_fork  # noqa: E402
from mepd.engines.ase import ASEEngine  # noqa: E402
from mepd.engines.fairchem import FAIRChemEngine  # noqa: E402
from mepd.inputs import RunInputs  # noqa: E402
from mepd.msmep import _clone_run_inputs_for_worker  # noqa: E402
from mepd.nodes.node import StructureNode  # noqa: E402

ANG = 1.8897259886
K, R0 = 2.0, 1.1  # eV/Angstrom^2, Angstrom


def _pair_potential(pos):
    """Energy (eV) and forces (eV/Angstrom) of a harmonic pair potential."""
    e, f = 0.0, np.zeros_like(pos)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = pos[i] - pos[j]
            r = np.linalg.norm(d)
            e += 0.5 * K * (r - R0) ** 2
            fij = -K * (r - R0) * d / r
            f[i] += fij
            f[j] -= fij
    return e, f


class _FakeModelEngine(FAIRChemEngine):
    """FAIRChemEngine with the model forward pass replaced by the pair
    potential, recording batch sizes."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.batches = []

    def _batch_predict(self, atoms_chunk):
        self.batches.append(len(atoms_chunk))
        es, fs = zip(*(_pair_potential(a.positions) for a in atoms_chunk))
        return np.array(es), np.concatenate(fs)


def _node(xyz_ang, symbols=("H", "H", "H")):
    return StructureNode(structure=Structure(
        symbols=list(symbols), geometry=np.asarray(xyz_ang) * ANG, charge=0, multiplicity=2,
    ))


def _nodes(n):
    rng = np.random.default_rng(0)
    base = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.9, 1.9]])
    return [_node(base + 0.05 * rng.standard_normal(base.shape)) for _ in range(n)]


def test_fairchem_engine_is_built_lazily_from_run_inputs():
    ri = RunInputs(engine_name="fairchem", fairchem_engine_kwds={"model": "uma-s-1p2p1", "device": "cpu", "batch_size": 4})
    assert isinstance(ri.engine, FAIRChemEngine)
    assert ri.engine._predictor is None  # no fairchem import, no download, until first use
    assert (ri.engine.model, ri.engine.device, ri.engine.batch_size) == ("uma-s-1p2p1", "cpu", 4)


def test_fairchem_engine_batches_images_and_converts_units():
    eng = _FakeModelEngine(batch_size=3)
    nodes = _nodes(7)
    grads = eng.compute_gradients(nodes)
    energies = eng.compute_energies(nodes)
    assert eng.batches == [3, 3, 1]
    for node, e, g in zip(nodes, energies, grads):
        e_ev, f_ev = _pair_potential(np.asarray(node.coords) / ANG)
        assert e == pytest.approx(e_ev / Hartree)
        assert np.allclose(g, -f_ev / (Hartree * ANG))


def test_fairchem_engine_skips_cached_images():
    eng = _FakeModelEngine(batch_size=8)
    nodes = _nodes(5)
    eng.compute_gradients(nodes[:2])
    eng.compute_gradients(nodes)
    assert eng.batches == [2, 3]


def test_fairchem_engine_hessian_matches_the_model_gradients():
    eng = _FakeModelEngine(batch_size=64)
    node = _nodes(1)[0]
    h = eng.compute_hessian(node, step_size=1e-4)
    x = np.asarray(node.coords, dtype=float).reshape(-1)

    def grad(xf):
        _, f = _pair_potential(xf.reshape(-1, 3) / ANG)
        return (-f / (Hartree * ANG)).reshape(-1)

    fd = np.array([(grad(x + 1e-5 * e) - grad(x - 1e-5 * e)) / 2e-5 for e in np.eye(x.size)])
    assert np.allclose(h, 0.5 * (fd + fd.T), atol=1e-6)
    assert sum(eng.batches) == 2 * x.size


def test_fairchem_engine_pickles_without_the_model_and_copies_share_it():
    eng = _FakeModelEngine()
    eng._predictor = SimpleNamespace(name="loaded model")
    restored = pickle.loads(pickle.dumps(eng))
    assert restored._predictor is None and restored.model == eng.model
    assert copy.deepcopy(eng)._predictor is eng._predictor
    ri = RunInputs(engine_name="fairchem")
    ri.engine._predictor = eng._predictor
    assert _clone_run_inputs_for_worker(ri).engine._predictor is eng._predictor


def test_ase_engine_takes_any_calculator_by_import_path():
    ri = RunInputs(engine_name="ase", ase_engine_kwds={"calculator": "ase.calculators.emt:EMT", "geometry_optimizer": "BFGS"})
    assert isinstance(ri.engine, ASEEngine) and isinstance(ri.engine.calculator, EMT)
    node = StructureNode(structure=Structure(symbols=["C", "O"], geometry=np.array([[0, 0, 0], [0, 0, 2.2]]), charge=0, multiplicity=1))
    expected = ASEEngine(calculator=EMT()).compute_energies([node.copy()])
    assert np.allclose(ri.engine.compute_energies([node.copy()]), expected)


def test_legacy_ase_omol25_program_uses_the_fairchem_engine():
    ri = RunInputs(engine_name="ase", program="omol25", program_kwds={"model_path": "/models/uma.pt", "device": "cpu"})
    assert isinstance(ri.engine, FAIRChemEngine)
    assert (ri.engine.checkpoint, ri.engine.device, ri.engine.task) == ("/models/uma.pt", "cpu", "omol")


def test_forking_is_refused_once_cuda_is_initialised(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
    assert "CUDA" in _unsafe_to_fork()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    if sys.platform != "darwin":
        assert _unsafe_to_fork() is None


def test_fairchem_engine_runs_a_neb_step_on_a_chain():
    eng = _FakeModelEngine(batch_size=16)
    chain = Chain.model_validate({"nodes": _nodes(6)})
    assert eng.compute_gradients(chain).shape == (6, 3, 3)
    assert eng.batches == [6]
