import numpy as np
from types import SimpleNamespace
from qcio import Structure

import mepd.neb as neb_module
from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.neb import _endpoint_energy_inversion_warning_text, NEB


def test_endpoint_energy_inversion_warning_triggers_for_endpoint_peak():
    energies = np.array([0.020, 0.000, 0.005, 0.018])
    msg = _endpoint_energy_inversion_warning_text(energies=energies)
    assert msg is not None
    assert "Endpoint energies are higher" in msg


def test_endpoint_energy_inversion_warning_not_triggered_when_ts_interior():
    energies = np.array([0.000, 0.020, 0.005, 0.001])
    msg = _endpoint_energy_inversion_warning_text(energies=energies)
    assert msg is None


def test_neb_warning_path_handles_parameters_without_frozen_indices(monkeypatch):
    def _node(x: float, e: float) -> StructureNode:
        node = StructureNode(
            structure=Structure(
                geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
                symbols=["H", "H"],
                charge=0,
                multiplicity=1,
            )
        )
        node._cached_energy = e
        node._cached_gradient = np.zeros((2, 3))
        node.has_molecular_graph = False
        node.graph = None
        return node

    # Endpoint is highest energy to force warning branch evaluation.
    prepared_chain = Chain.model_validate(
        {
            "nodes": [_node(0.8, 0.020), _node(1.0, 0.000), _node(1.2, 0.010)],
            "parameters": ChainInputs(),
        }
    )

    params = SimpleNamespace(
        max_steps=2,
        v=False,
        climb=False,
        do_elem_step_checks=False,
        negative_steps_thre=10,
        positive_steps_thre=10,
    )
    optimizer = SimpleNamespace(timestep=0.1, g_old=None, reset=lambda: None)
    neb = NEB(
        initial_chain=prepared_chain.copy(),
        optimizer=optimizer,
        parameters=params,
        engine=SimpleNamespace(),
    )

    monkeypatch.setattr(NEB, "update_chain", lambda self, chain: prepared_chain.copy())
    monkeypatch.setattr(neb_module, "chain_converged", lambda **kwargs: True)
    monkeypatch.setattr(neb_module.ch, "_gradient_correlation", lambda a, b: 1.0)
    monkeypatch.setattr(neb_module, "format_neb_caption", lambda **kwargs: "")
    monkeypatch.setattr(neb_module, "print_chain_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(neb_module, "update_status", lambda *args, **kwargs: None)

    # Regression: this used to crash with
    # AttributeError: 'types.SimpleNamespace' object has no attribute 'frozen_atom_indices'
    result = neb.optimize_chain()
    assert result.is_elem_step is True


def test_early_stop_requires_both_ts_gperp_and_ts_triplet_spring(monkeypatch):
    class _DummyChain:
        def __init__(self, ts_triplet_spring: float):
            self.energies = np.array([0.0, 1.0, 0.2], dtype=float)
            self.ts_triplet_gspring_infnorm = float(ts_triplet_spring)
            # Keep RMS tiny to ensure legacy RMS-only logic would have triggered.
            self.rms_gradients = np.array([1e-6, 1e-6, 1e-6], dtype=float)
            self.springgradients = np.zeros((1, 2, 3), dtype=float)

        def __len__(self):
            return 3

    monkeypatch.setattr(
        neb_module.ch,
        "get_g_perps",
        lambda chain: [
            np.zeros((2, 3), dtype=float),
            np.full((2, 3), 0.05, dtype=float),
            np.zeros((2, 3), dtype=float),
        ],
    )

    neb = object.__new__(NEB)
    neb.parameters = SimpleNamespace(early_stop_force_thre=0.1)

    calls = {"count": 0}

    def _fake_early_stop(_chain):
        calls["count"] += 1
        return True, SimpleNamespace(number_grad_calls=0)

    neb._do_early_stop_check = _fake_early_stop

    # TS gperp is below threshold, but TS-triplet spring is above: no early stop.
    stop_early, _ = neb._check_early_stop(_DummyChain(ts_triplet_spring=0.2))
    assert stop_early is False
    assert calls["count"] == 0

    # Once both are below threshold, early stop check is triggered.
    stop_early, _ = neb._check_early_stop(_DummyChain(ts_triplet_spring=0.05))
    assert stop_early is True
    assert calls["count"] == 1
