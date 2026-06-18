from types import SimpleNamespace

import numpy as np
import pytest

from mepd import convergence_helpers as conv


class _DummyChain:
    def __init__(self, n_nodes: int, ts_index: int):
        self._n = n_nodes
        self._energies = np.zeros(n_nodes)
        self._energies[ts_index] = 1.0
        self.nodes = [
            SimpleNamespace(converged=False, _cached_gradient=np.zeros((2, 3)), _cached_energy=0.0)
            for _ in range(n_nodes)
        ]
        self.parameters = SimpleNamespace(frozen_atom_indices="", node_freezing=False)

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return self._n

    @property
    def gradients(self):
        return [np.zeros((2, 3)) for _ in range(self._n)]

    @property
    def springgradients(self):
        return [np.zeros((2, 3)) for _ in range(self._n - 2)]

    @property
    def ts_triplet_gspring_infnorm(self):
        return 0.0

    @property
    def energies(self):
        return self._energies

    @property
    def gperps(self):
        raise AssertionError("chain_converged should use chainhelpers.get_g_perps, not chain.gperps")

    def get_eA_chain(self):
        return float(self._energies.max())


def test_chain_converged_uses_endpoint_padded_gperps_for_ts_index(monkeypatch):
    chain_prev = _DummyChain(n_nodes=15, ts_index=13)
    chain_new = _DummyChain(n_nodes=15, ts_index=13)

    # Endpoint-padded gperps shape: len == n_nodes.
    monkeypatch.setattr(
        "mepd.chainhelpers.get_g_perps",
        lambda chain: [np.zeros((2, 3)) for _ in range(len(chain))],
    )

    parameters = SimpleNamespace(
        rms_grad_thre=0.001,
        max_rms_grad_thre=0.01,
        ts_grad_thre=0.05,
        ts_spring_thre=0.02,
        barrier_thre=0.1,
    )

    # Regression: this used to raise IndexError when TS index was near end.
    out = conv.chain_converged(
        chain_prev=chain_prev,
        chain_new=chain_new,
        parameters=parameters,
        verbose=False,
    )
    assert isinstance(out, (bool, np.bool_))


def test_chain_converged_handles_large_frozen_atom_indices(monkeypatch):
    chain_prev = _DummyChain(n_nodes=5, ts_index=2)
    chain_new = _DummyChain(n_nodes=5, ts_index=2)
    # Simulate many frozen atom indices, all > n_nodes.
    chain_new.parameters = SimpleNamespace(
        frozen_atom_indices=list(range(100, 200)),
        node_freezing=False,
    )

    monkeypatch.setattr(
        "mepd.chainhelpers.get_g_perps",
        lambda chain: [np.zeros((2, 3)) for _ in range(len(chain))],
    )

    parameters = SimpleNamespace(
        rms_grad_thre=0.001,
        max_rms_grad_thre=0.01,
        ts_grad_thre=0.05,
        ts_spring_thre=0.02,
        barrier_thre=0.1,
    )

    out = conv.chain_converged(
        chain_prev=chain_prev,
        chain_new=chain_new,
        parameters=parameters,
        verbose=False,
    )
    assert isinstance(out, (bool, np.bool_))


def test_chain_converged_uses_fraction_freeze_from_chain_inputs(monkeypatch):
    chain_prev = _DummyChain(n_nodes=5, ts_index=2)
    chain_new = _DummyChain(n_nodes=5, ts_index=2)
    chain_new.parameters = SimpleNamespace(
        frozen_atom_indices="",
        node_freezing=False,
        fraction_freeze=0.5,
    )

    monkeypatch.setattr(
        "mepd.chainhelpers.get_g_perps",
        lambda chain: [np.zeros((2, 3)) for _ in range(len(chain))],
    )

    thresholds = {}
    original_rms = conv._check_rms_grad_converged
    original_gperps = conv._check_gperps_converged
    original_spring = conv._check_springgrad_converged

    def _capture_rms(gradients, threshold):
        thresholds["rms"] = threshold
        return original_rms(gradients, threshold)

    def _capture_gperps(pe_grads, threshold):
        thresholds["gperps"] = threshold
        return original_gperps(pe_grads, threshold)

    def _capture_spring(spring_forces, threshold):
        thresholds["spring"] = threshold
        return original_spring(spring_forces, threshold)

    monkeypatch.setattr(conv, "_check_rms_grad_converged", _capture_rms)
    monkeypatch.setattr(conv, "_check_gperps_converged", _capture_gperps)
    monkeypatch.setattr(conv, "_check_springgrad_converged", _capture_spring)

    parameters = SimpleNamespace(
        rms_grad_thre=0.02,
        max_rms_grad_thre=0.05,
        ts_grad_thre=0.05,
        ts_spring_thre=0.04,
        barrier_thre=0.1,
    )

    conv.chain_converged(
        chain_prev=chain_prev,
        chain_new=chain_new,
        parameters=parameters,
        verbose=False,
    )

    assert thresholds["rms"] == pytest.approx(0.01)
    assert thresholds["gperps"] == pytest.approx(0.01)
    assert thresholds["spring"] == pytest.approx(0.02)


def test_chain_converged_emits_monitor_detail_lines(monkeypatch):
    chain_prev = _DummyChain(n_nodes=7, ts_index=3)
    chain_new = _DummyChain(n_nodes=7, ts_index=3)

    def _gperps(_chain):
        arr = [np.zeros((2, 3)) for _ in range(len(_chain))]
        arr[3] = np.ones((2, 3)) * 0.2
        return arr

    monkeypatch.setattr("mepd.chainhelpers.get_g_perps", _gperps)

    captured = {}

    def _capture(lines):
        captured["lines"] = list(lines or [])

    monkeypatch.setattr(conv, "set_monitor_details", _capture)

    parameters = SimpleNamespace(
        rms_grad_thre=0.001,
        max_rms_grad_thre=0.01,
        ts_grad_thre=0.05,
        ts_spring_thre=0.02,
        barrier_thre=0.1,
    )

    conv.chain_converged(
        chain_prev=chain_prev,
        chain_new=chain_new,
        parameters=parameters,
        verbose=True,
    )

    lines = captured.get("lines", [])
    assert len(lines) == 6
    assert any(line.startswith("NO TS_GRAD:") for line in lines)
    assert any("<= 0.05" in line for line in lines)


def test_chain_converged_preserves_detail_prefix_lines(monkeypatch):
    chain_prev = _DummyChain(n_nodes=7, ts_index=3)
    chain_new = _DummyChain(n_nodes=7, ts_index=3)

    def _gperps(_chain):
        arr = [np.zeros((2, 3)) for _ in range(len(_chain))]
        arr[3] = np.ones((2, 3)) * 0.2
        return arr

    monkeypatch.setattr("mepd.chainhelpers.get_g_perps", _gperps)

    captured = {}
    monkeypatch.setattr(
        conv,
        "set_monitor_details",
        lambda lines: captured.setdefault("lines", list(lines or [])),
    )

    parameters = SimpleNamespace(
        rms_grad_thre=0.001,
        max_rms_grad_thre=0.01,
        ts_grad_thre=0.05,
        ts_spring_thre=0.02,
        barrier_thre=0.1,
    )

    conv.chain_converged(
        chain_prev=chain_prev,
        chain_new=chain_new,
        parameters=parameters,
        verbose=True,
        detail_prefix_lines=["dt=0.25"],
    )

    lines = captured.get("lines", [])
    assert lines[0] == "dt=0.25"
    assert any(line.endswith("MAX(RMS_GPERP): 0 <= 0.01") for line in lines[1:])


def test_check_springgrad_converged_uses_infinity_norm():
    spring_forces = [
        np.array([[0.20, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
    ]
    converged_indices, components = conv._check_springgrad_converged(
        spring_forces=spring_forces,
        threshold=0.15,
    )

    assert components == [0.20]
    assert converged_indices[0].size == 0
