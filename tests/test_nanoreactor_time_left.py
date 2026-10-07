"""The nanoreactor's MD status line says about how long the rest will take,
from the simulated time per second since the first segment."""

from __future__ import annotations


def test_time_left_from_the_rate_so_far(monkeypatch):
    import time

    from mepd.discovery.cli_nanoreactor import _time_left

    clock = iter([1000.0, 1060.0, 1120.0])
    monkeypatch.setattr(time, "time", lambda: next(clock))
    live: dict = {}
    assert _time_left(live, 0.75, 20.0) == ""                       # first segment: no rate yet
    assert _time_left(live, 1.75, 20.0) == " · about 18 min left"   # 1 ps a minute, 18.25 ps to go
    assert _time_left(live, 19.75, 20.0) == " · about 2 s left"     # 19 ps in 120 s: 0.25 ps to go
