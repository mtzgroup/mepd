"""Explore's edges tell a direct step from a route: a TS search whose
IRC-verified route goes through an intermediate reports how many steps it
took and the lowest direct (one-step) barrier, if any TS joins start and end
by itself."""

from __future__ import annotations

import pytest

from test_web import _write_chain  # noqa: E402  (tests/ is on sys.path)

A = """8
ethane
C 0.000 0.000 0.765
C 0.000 0.000 -0.765
H 1.018 0.000 1.160
H -0.509 0.882 1.160
H -0.509 -0.882 1.160
H -1.018 0.000 -1.160
H 0.509 -0.882 -1.160
H 0.509 0.882 -1.160
"""
I = A.replace("H 1.018 0.000 1.160", "H 0.900 0.000 -1.600", 1)       # one H moved to the other carbon
B = I.replace("H -0.509 0.882 1.160", "H -0.450 -0.100 -1.950", 1)    # a second one


def test_a_two_step_route_has_no_direct_barrier_until_a_ts_joins_the_ends(tmp_path):
    from mepd.web.results import collect_ts, summarize

    out = tmp_path / "run"
    out.mkdir()
    _write_chain(out / "mep_output.xyz", [A, I, B], [-79.80, -79.70, -79.75])
    _write_chain(out / "ts_leaf_0.xyz", [A], [-79.75])
    _write_chain(out / "ts_leaf_0_irc.xyz", [A, A, I], [-79.80, -79.75, -79.70])
    _write_chain(out / "ts_leaf_1.xyz", [I], [-79.65])
    _write_chain(out / "ts_leaf_1_irc.xyz", [I, I, B], [-79.70, -79.65, -79.75])
    r = collect_ts(out, 0, 1)
    assert r["barrier_verified"] is True and r["n_steps"] == 2
    assert r["barrier_kcal"] == pytest.approx(0.15 * 627.509474, abs=1e-2)     # the route's highest step
    assert r["direct_barrier_kcal"] is None
    s = summarize(r)
    assert s["n_steps"] == 2 and s["direct_barrier_kcal"] is None

    # A TS whose IRC joins A and B directly (higher than the route): a direct path.
    _write_chain(out / "ts_leaf_2.xyz", [A], [-79.60])
    _write_chain(out / "ts_leaf_2_irc.xyz", [A, A, B], [-79.80, -79.60, -79.75])
    r = collect_ts(out, 0, 1)
    assert r["direct_barrier_kcal"] == pytest.approx(0.20 * 627.509474, abs=1e-2)
    assert r["barrier_kcal"] == pytest.approx(0.15 * 627.509474, abs=1e-2)     # the route is still lower
