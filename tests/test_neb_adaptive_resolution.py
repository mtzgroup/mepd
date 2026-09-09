import numpy as np
import pytest

from mepd import neb as neb_module
from mepd.chain import Chain
from mepd.errors import NoneConvergedException
from mepd.inputs import ChainInputs, NEBInputs
from mepd.neb import NEB
from mepd.nodes.node import XYNode
from mepd.optimizers.cg import ConjugateGradient


class _FakeOptimizer:
    def __init__(self, timestep: float = 0.1):
        self.timestep = timestep
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _XYTestEngine:
    def compute_gradients(self, chain):
        grads = []
        for i, node in enumerate(chain):
            grad = np.zeros_like(np.array(node.coords, dtype=float))
            if 0 < i < len(chain) - 1:
                grad = np.array([2.0, 0.0], dtype=float)
            node._cached_gradient = grad
            grads.append(grad)
        return np.array(grads, dtype=float)

    def compute_energies(self, chain):
        enes = []
        for i, node in enumerate(chain):
            x = float(node.coords[0])
            ene = -((x - 2.5) ** 2) + 6.0
            if i in (0, len(chain) - 1):
                ene = 0.0
            node._cached_energy = ene
            enes.append(ene)
        return np.array(enes, dtype=float)


class _JumpEnergyEngine(_XYTestEngine):
    def compute_energies(self, chain):
        profile = [0.0, 0.1, 10.0, 10.2]
        enes = []
        for i, node in enumerate(chain):
            ene = profile[i] if i < len(profile) else profile[-1]
            node._cached_energy = ene
            enes.append(ene)
        return np.array(enes, dtype=float)


def _make_chain(coords):
    return Chain.model_validate(
        {
            "nodes": [XYNode(structure=np.array(c, dtype=float)) for c in coords],
            "parameters": ChainInputs(use_geodesic_interpolation=False),
        }
    )


def test_adaptive_resolution_inserts_around_max_rms_gperp(monkeypatch):
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [8.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)
    monkeypatch.setattr(
        Chain,
        "rms_gradients",
        property(lambda _chain: np.array([0.0, 0.2, 3.0, 0.1, 0.0])),
    )

    optimizer = _FakeOptimizer()
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=optimizer,
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_segment_ratio=2.0,
            adaptive_max_images=10,
            adaptive_cooldown_steps=0,
            adaptive_plateau_window=0,
            max_rms_grad_thre=1.0,
            rms_grad_thre=2.0,
            ts_spring_thre=1.0,
        ),
        engine=engine,
    )

    refined, inserted = neb._maybe_adapt_chain_resolution(chain, step=3)

    assert inserted is True
    assert len(refined) == len(chain) + 1
    assert np.isclose(refined[2].coords[0], 1.5)
    assert optimizer.reset_calls == 1
    assert neb._last_resolution_insert_step == 3


def test_adaptive_resolution_respects_max_images():
    chain = _make_chain([[0.0, 0.0], [0.1, 0.0], [4.5, 0.0], [5.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    optimizer = _FakeOptimizer()
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=optimizer,
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_segment_ratio=1.1,
            adaptive_max_images=len(chain),
            adaptive_cooldown_steps=0,
            adaptive_plateau_window=0,
        ),
        engine=engine,
    )

    refined, inserted = neb._maybe_adapt_chain_resolution(chain, step=3)

    assert inserted is False
    assert len(refined) == len(chain)
    assert optimizer.reset_calls == 0


def test_adaptive_resolution_inserts_in_largest_energy_gap_for_mean_rms_gperp():
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    engine = _JumpEnergyEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    optimizer = _FakeOptimizer()
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=optimizer,
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_segment_ratio=10.0,
            adaptive_use_energy=True,
            adaptive_energy_ratio=2.0,
            adaptive_max_images=10,
            adaptive_cooldown_steps=0,
            adaptive_plateau_window=0,
            max_rms_grad_thre=1.0,
            rms_grad_thre=1.0,
            ts_spring_thre=1.0,
        ),
        engine=engine,
    )

    refined, inserted = neb._maybe_adapt_chain_resolution(chain, step=4)

    assert inserted is True
    assert len(refined) == len(chain) + 1
    assert np.isclose(refined[2].coords[0], 1.5)
    assert optimizer.reset_calls == 1


def test_adaptive_resolution_uses_largest_energy_gap_for_mean_rms_gperp_even_without_adaptive_use_energy():
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    engine = _JumpEnergyEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    optimizer = _FakeOptimizer()
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=optimizer,
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_segment_ratio=10.0,
            adaptive_use_energy=False,
            adaptive_energy_ratio=2.0,
            adaptive_max_images=10,
            adaptive_cooldown_steps=0,
            adaptive_plateau_window=0,
            max_rms_grad_thre=1.0,
            rms_grad_thre=1.0,
            ts_spring_thre=1.0,
        ),
        engine=engine,
    )

    refined, inserted = neb._maybe_adapt_chain_resolution(chain, step=4)

    assert inserted is True
    assert len(refined) == len(chain) + 1
    assert np.isclose(refined[2].coords[0], 1.5)
    assert optimizer.reset_calls == 1


def test_adaptive_resolution_waits_for_plateau_before_inserting():
    chain = _make_chain([[0.0, 0.0], [0.1, 0.0], [4.5, 0.0], [5.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    optimizer = _FakeOptimizer()
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=optimizer,
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_segment_ratio=2.0,
            adaptive_max_images=10,
            adaptive_cooldown_steps=0,
            adaptive_plateau_window=1,
            max_rms_grad_thre=1.0,
            rms_grad_thre=1.0,
            ts_spring_thre=1.0,
        ),
        engine=engine,
    )

    first_chain, first_inserted = neb._maybe_adapt_chain_resolution(chain, step=3)
    second_chain, second_inserted = neb._maybe_adapt_chain_resolution(first_chain, step=4)

    assert first_inserted is False
    assert len(first_chain) == len(chain)
    assert second_inserted is True
    assert len(second_chain) == len(chain) + 1
    assert optimizer.reset_calls == 1


def test_adaptive_resolution_does_not_insert_when_mean_rms_gperp_is_limiting():
    chain = _make_chain([[0.0, 0.0], [0.1, 0.0], [4.5, 0.0], [5.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    optimizer = _FakeOptimizer()
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=optimizer,
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_max_images=10,
            adaptive_cooldown_steps=0,
            adaptive_plateau_window=0,
            max_rms_grad_thre=1.0,
            rms_grad_thre=0.5,
            ts_spring_thre=1.0,
        ),
        engine=engine,
    )

    refined, inserted = neb._maybe_adapt_chain_resolution(chain, step=3)

    assert inserted is False
    assert len(refined) == len(chain)
    assert optimizer.reset_calls == 0


def test_adaptive_resolution_does_not_insert_when_ts_grad_is_limiting(monkeypatch):
    chain = _make_chain([[0.0, 0.0], [0.1, 0.0], [4.5, 0.0], [5.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    optimizer = _FakeOptimizer()
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=optimizer,
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_max_images=10,
            adaptive_cooldown_steps=0,
            adaptive_plateau_window=0,
            max_rms_grad_thre=1.0,
            rms_grad_thre=1.0,
            ts_grad_thre=0.05,
            ts_spring_thre=1.0,
        ),
        engine=engine,
    )

    metrics = neb._collect_adaptive_convergence_metrics(chain)
    metrics["ts_grad"] = 0.1
    monkeypatch.setattr(neb, "_collect_adaptive_convergence_metrics", lambda _chain: metrics)

    refined, inserted = neb._maybe_adapt_chain_resolution(chain, step=3)

    assert inserted is False
    assert len(refined) == len(chain)
    assert optimizer.reset_calls == 0


def test_adaptive_resolution_does_not_insert_when_spring_infnorm_is_limiting(monkeypatch):
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [8.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)
    monkeypatch.setattr(
        Chain,
        "springgradients",
        property(
            lambda _chain: [
                np.array([0.01, 0.0]),
                np.array([0.20, 0.0]),
                np.array([0.01, 0.0]),
            ]
        ),
    )

    optimizer = _FakeOptimizer()
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=optimizer,
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_max_images=10,
            adaptive_cooldown_steps=0,
            adaptive_plateau_window=0,
            max_rms_grad_thre=1.0,
            rms_grad_thre=1.0,
            ts_grad_thre=1.0,
            ts_spring_thre=0.05,
        ),
        engine=engine,
    )

    metrics = neb._collect_adaptive_convergence_metrics(chain)
    metrics["max_rms_gperp"] = 0.5
    metrics["mean_rms_gperp"] = 0.5
    metrics["ts_grad"] = 0.5
    metrics["ts_triplet_gspring"] = 0.01
    metrics["max_spring"] = 0.1
    monkeypatch.setattr(neb, "_collect_adaptive_convergence_metrics", lambda _chain: metrics)

    refined, inserted = neb._maybe_adapt_chain_resolution(chain, step=3)

    assert inserted is False
    assert len(refined) == len(chain)
    assert optimizer.reset_calls == 0


def test_plateau_exit_triggers_when_adaptive_resolution_is_off():
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=_FakeOptimizer(),
        parameters=NEBInputs(
            adaptive_resolution=False,
            adaptive_plateau_window=2,
            adaptive_plateau_rtol=0.05,
        ),
        engine=engine,
    )

    assert neb._plateau_exit_triggered(chain) is False
    assert neb._plateau_exit_triggered(chain) is False
    assert neb._plateau_exit_triggered(chain) is True


def test_plateau_exit_waits_when_adaptive_resolution_can_still_add_images():
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=_FakeOptimizer(),
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_max_images=len(chain) + 1,
            adaptive_plateau_window=0,
        ),
        engine=engine,
    )

    assert neb._plateau_exit_triggered(chain) is False


def test_plateau_exit_triggers_when_adaptive_resolution_is_exhausted():
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=_FakeOptimizer(),
        parameters=NEBInputs(
            adaptive_resolution=True,
            adaptive_max_images=len(chain),
            adaptive_plateau_window=0,
        ),
        engine=engine,
    )

    assert neb._plateau_exit_triggered(chain) is True


def test_optimize_chain_raises_explicit_plateau_exit_message(monkeypatch):
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=_FakeOptimizer(),
        parameters=NEBInputs(
            adaptive_resolution=False,
            adaptive_plateau_window=1,
            max_steps=5,
            do_elem_step_checks=False,
            negative_steps_thre=10,
            positive_steps_thre=10,
        ),
        engine=engine,
    )

    monkeypatch.setattr(NEB, "update_chain", lambda self, chain: self.initial_chain.copy())
    monkeypatch.setattr(neb_module, "chain_converged", lambda **kwargs: False)
    monkeypatch.setattr(neb_module.ch, "_gradient_correlation", lambda a, b: 1.0)
    monkeypatch.setattr(neb_module, "print_chain_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(neb_module, "update_status", lambda *args, **kwargs: None)

    with pytest.raises(NoneConvergedException) as exc_info:
        neb.optimize_chain()

    msg = str(exc_info.value)
    assert "will likely not converge with more minimization steps" in msg
    assert "lower spring constants or higher bead density" in msg
    assert neb.optimized is not None


def test_optimize_chain_status_includes_timestep_for_conjugate_gradient(monkeypatch):
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    engine = _XYTestEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=ConjugateGradient(timestep=0.25),
        parameters=NEBInputs(
            max_steps=2,
            do_elem_step_checks=False,
            negative_steps_thre=10,
            positive_steps_thre=10,
            adaptive_resolution=False,
            adaptive_plateau_window=0,
        ),
        engine=engine,
    )

    messages = []
    monkeypatch.setattr(NEB, "update_chain", lambda self, chain: chain.copy())
    monkeypatch.setattr(neb_module, "chain_converged", lambda **kwargs: False)
    monkeypatch.setattr(neb_module.ch, "_gradient_correlation", lambda a, b: 1.0)
    monkeypatch.setattr(neb_module.ch, "_get_ind_minima", lambda chain: [])
    monkeypatch.setattr(neb_module.ch, "get_g_perps", lambda chain: np.zeros((len(chain), 2)))
    monkeypatch.setattr(NEB, "_plateau_exit_triggered", lambda self, chain: False)
    monkeypatch.setattr(neb_module, "print_chain_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(neb_module, "update_status", lambda message: messages.append(message))

    with pytest.raises(NoneConvergedException):
        neb.optimize_chain()

    assert "Optimizing path... Step 2 | dt=0.25" in messages
