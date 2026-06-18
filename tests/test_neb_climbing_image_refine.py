import numpy as np
from types import SimpleNamespace

import mepd.neb as neb_module
from mepd.chain import Chain
from mepd.elementarystep import ElemStepResults
from mepd.inputs import ChainInputs, NEBInputs
from mepd.neb import CLIMBING_IMAGE_MAX_STEPS, NEB
from mepd.nodes.node import XYNode


class _FakeOptimizer:
    def __init__(self, timestep: float = 0.1):
        self.timestep = timestep
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class _SimpleEngine:
    def compute_gradients(self, chain):
        grads = []
        for node in chain:
            grad = np.zeros_like(np.array(node.coords, dtype=float))
            node._cached_gradient = grad
            grads.append(grad)
        return np.array(grads, dtype=float)

    def compute_energies(self, chain):
        enes = []
        for node in chain:
            ene = float(node.coords[0])
            node._cached_energy = ene
            enes.append(ene)
        return np.array(enes, dtype=float)


def _make_chain(coords):
    chain = Chain.model_validate(
        {
            "nodes": [XYNode(structure=np.array(c, dtype=float)) for c in coords],
            "parameters": ChainInputs(use_geodesic_interpolation=False),
        }
    )
    return chain


def test_climbing_refinement_inserts_and_freezes_except_climber(monkeypatch):
    chain = _make_chain([[0.0, 0.0], [3.0, 0.0], [2.5, 0.0], [0.0, 0.0]])
    engine = _SimpleEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=_FakeOptimizer(),
        parameters=NEBInputs(climb=True, ts_grad_thre=0.05),
        engine=engine,
    )

    call_counter = {"gperp": 0, "updates": 0}

    def fake_get_g_perps(curr_chain):
        call_counter["gperp"] += 1
        grads = np.zeros((len(curr_chain), 2), dtype=float)
        if call_counter["gperp"] <= 2:
            grads[2] = np.array([0.2, 0.0], dtype=float)
        else:
            grads[2] = np.array([0.01, 0.0], dtype=float)
        return grads

    def fake_update_chain(curr_chain):
        call_counter["updates"] += 1
        unconverged = [i for i, node in enumerate(curr_chain.nodes) if not node.converged]
        climbing = [i for i, node in enumerate(curr_chain.nodes) if node.do_climb]
        assert unconverged == [2]
        assert climbing == [2]
        return curr_chain.copy()

    monkeypatch.setattr("mepd.chainhelpers.get_g_perps", fake_get_g_perps)
    monkeypatch.setattr(NEB, "update_chain", lambda self, chain: fake_update_chain(chain))

    refined = neb._run_post_convergence_climbing_refinement(chain)

    assert len(refined) == len(chain) + 1
    assert np.isclose(refined[2].coords[0], 2.75)
    assert [i for i, node in enumerate(refined.nodes) if node.do_climb] == [2]
    assert [i for i, node in enumerate(refined.nodes) if not node.converged] == [2]
    assert call_counter["updates"] == 2


def test_climbing_pair_uses_highest_energy_image_and_highest_energy_neighbor():
    chain = _make_chain(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [1.0, 0.0],
            [4.0, 0.0],
            [0.0, 0.0],
        ]
    )
    engine = _SimpleEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=_FakeOptimizer(),
        parameters=NEBInputs(climb=True, ts_grad_thre=0.05),
        engine=engine,
    )

    assert neb._get_climbing_pair_indices(chain) == (1, 2)


def test_climbing_refinement_stops_at_max_steps(monkeypatch):
    chain = _make_chain([[0.0, 0.0], [3.0, 0.0], [2.5, 0.0], [0.0, 0.0]])
    engine = _SimpleEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=_FakeOptimizer(),
        parameters=NEBInputs(climb=True, ts_grad_thre=0.05),
        engine=engine,
    )

    call_counter = {"updates": 0}

    def fake_get_g_perps(curr_chain):
        grads = np.zeros((len(curr_chain), 2), dtype=float)
        grads[2] = np.array([0.2, 0.0], dtype=float)
        return grads

    def fake_update_chain(curr_chain):
        call_counter["updates"] += 1
        return curr_chain.copy()

    monkeypatch.setattr("mepd.chainhelpers.get_g_perps", fake_get_g_perps)
    monkeypatch.setattr(NEB, "update_chain", lambda self, chain: fake_update_chain(chain))

    refined = neb._run_post_convergence_climbing_refinement(chain)

    assert len(refined) == len(chain) + 1
    assert call_counter["updates"] == CLIMBING_IMAGE_MAX_STEPS


def test_climbing_neb_checks_elementary_step_before_ci_and_not_after(monkeypatch):
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    engine = _SimpleEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    params = SimpleNamespace(
        max_steps=1,
        v=False,
        climb=True,
        do_elem_step_checks=True,
        negative_steps_thre=10,
        positive_steps_thre=10,
    )
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=_FakeOptimizer(),
        parameters=params,
        engine=engine,
    )

    events = []

    def fake_check_if_elem_step(inp_chain, engine, verbose=False):
        events.append(("elem", len(inp_chain)))
        return ElemStepResults(
            is_elem_step=True,
            is_concave=True,
            splitting_criterion=None,
            minimization_results=None,
            number_grad_calls=3,
        )

    def fake_climbing_refinement(self, converged_chain):
        events.append(("ci", len(converged_chain)))
        refined = converged_chain.copy()
        refined.nodes.insert(2, converged_chain[1].copy())
        return refined

    monkeypatch.setattr(NEB, "update_chain", lambda self, chain: chain.copy())
    monkeypatch.setattr(NEB, "_run_post_convergence_climbing_refinement", fake_climbing_refinement)
    monkeypatch.setattr(neb_module, "chain_converged", lambda **kwargs: True)
    monkeypatch.setattr(neb_module, "check_if_elem_step", fake_check_if_elem_step)
    monkeypatch.setattr(neb_module.ch, "get_g_perps", lambda chain: np.zeros((len(chain), 2)))
    monkeypatch.setattr(neb_module.ch, "_gradient_correlation", lambda a, b: 1.0)
    monkeypatch.setattr(neb_module, "format_neb_caption", lambda **kwargs: "")
    monkeypatch.setattr(neb_module, "print_chain_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(neb_module, "update_status", lambda *args, **kwargs: None)

    result = neb.optimize_chain()

    assert result.is_elem_step is True
    assert events == [("elem", 3), ("ci", 3)]
    assert len(neb.optimized) == 4


def test_climbing_neb_refines_when_unconverged_chain_is_elementary(monkeypatch):
    chain = _make_chain([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    engine = _SimpleEngine()
    engine.compute_energies(chain)
    engine.compute_gradients(chain)

    params = SimpleNamespace(
        max_steps=0,
        v=False,
        climb=True,
        do_elem_step_checks=True,
    )
    neb = NEB(
        initial_chain=chain.copy(),
        optimizer=_FakeOptimizer(),
        parameters=params,
        engine=engine,
    )

    events = []

    def fake_check_if_elem_step(inp_chain, engine, verbose=False):
        events.append(("elem", len(inp_chain)))
        return ElemStepResults(
            is_elem_step=True,
            is_concave=True,
            splitting_criterion=None,
            minimization_results=None,
            number_grad_calls=5,
        )

    def fake_climbing_refinement(self, unconverged_chain):
        events.append(("ci", len(unconverged_chain)))
        refined = unconverged_chain.copy()
        refined.nodes.insert(2, unconverged_chain[1].copy())
        return refined

    monkeypatch.setattr(NEB, "update_chain", lambda self, chain: chain.copy())
    monkeypatch.setattr(NEB, "_run_post_convergence_climbing_refinement", fake_climbing_refinement)
    monkeypatch.setattr(neb_module, "chain_converged", lambda **kwargs: False)
    monkeypatch.setattr(neb_module, "check_if_elem_step", fake_check_if_elem_step)

    result = neb.optimize_chain()

    assert result.is_elem_step is True
    assert events == [("elem", 3), ("ci", 3)]
    assert neb.geom_grad_calls_made == 5
    assert len(neb.optimized) == 4
