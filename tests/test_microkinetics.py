"""Microkinetics with steps of any order: equilibria, shuttles, steady
states and degrees of rate control."""
import math

import numpy as np
import pytest

from mepd import microkinetics as mk


def test_isomerization_reaches_its_equilibrium_constant():
    net = mk.Network(["A", "B"], [0.0, -1.0], [mk.Step([0], [1], 15.0)])
    res = mk.simulate(net, [1.0, 0.0], 298.15, 1e6)
    K = math.exp(1.0 / mk.rt_kcal(298.15))
    assert res.c[-1, 1] / res.c[-1, 0] == pytest.approx(K, rel=1e-3)
    assert res.c[-1].sum() == pytest.approx(1.0, rel=1e-6)       # mass conserved
    assert res.extent[0] == pytest.approx(res.c[-1, 1], rel=1e-4)


def test_association_obeys_mass_action_at_one_molar_standard_state():
    net = mk.Network(["A", "B", "C"], [0.0, 0.0, -2.0], [mk.Step([0, 1], [2], 10.0)])
    res = mk.simulate(net, [0.5, 0.3, 0.0], 298.15, 1e6)
    a, b, c = res.c[-1]
    assert c / (a * b) == pytest.approx(math.exp(2.0 / mk.rt_kcal(298.15)), rel=1e-3)


def test_a_shuttle_speeds_its_step_and_is_not_consumed():
    # A -> B directly (barrier 25) or with water W (A + W -> B + W, barrier 18).
    direct = mk.Network(["A", "B", "W"], [0.0, -5.0, 0.0], [mk.Step([0], [1], 25.0)])
    both = mk.Network(["A", "B", "W"], [0.0, -5.0, 0.0], [mk.Step([0], [1], 25.0), mk.Step([0, 2], [1, 2], 18.0)])
    t = 1.0
    slow = mk.simulate(direct, [1.0, 0.0, 1.0], 298.15, t).c[-1, 1]
    fast = mk.simulate(both, [1.0, 0.0, 1.0], 298.15, t)
    assert fast.c[-1, 1] > 100 * slow
    assert fast.c[-1, 2] == pytest.approx(1.0, rel=1e-9)
    assert fast.extent[1] > 50 * fast.extent[0]                   # the material went through the shuttle


def test_constant_feed_reaches_a_steady_state_and_rate_control_sums_to_one():
    # A (fed) -> I -> P, P drained to a held-empty sink: a steady formation rate of P.
    net = mk.Network(["A", "I", "P", "S"], [0.0, -2.0, -10.0, -40.0],
                     [mk.Step([0], [1], 18.0), mk.Step([1], [2], 20.0), mk.Step([2], [3], 5.0)])
    held = [0, 3]
    res = mk.simulate(net, [1.0, 0.0, 0.0, 0.0], 298.15, 1e8, held=held)
    assert res.steady
    drc = mk.degree_of_control(net, [1.0, 0.0, 0.0, 0.0], 298.15, 1e8, target=3, what="rate", held=held)
    assert sum(drc["steps"]) == pytest.approx(1.0, abs=0.03)
    assert np.argmax(drc["steps"]) == 1                            # I -> P is the rate-determining TS


def test_temperature_sweep_gives_an_arrhenius_activation_energy():
    net = mk.Network(["A", "B"], [0.0, -10.0], [mk.Step([0], [1], 20.0)])
    out = mk.sweep(net, [1.0, 0.0], [300, 320, 340], 1e-6, target=1)
    # at very short times the formation rate is k [A]: apparent Ea = barrier + RT
    assert out["apparent_ea_kcal"][1] == pytest.approx(20.0 + mk.rt_kcal(320), abs=0.3)
