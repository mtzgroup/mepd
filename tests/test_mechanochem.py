"""Mechanical force on transition states (mepd.mechanochem, `mepd force`)."""
from __future__ import annotations

import numpy as np
import pytest

from mepd import mechanochem as M

A = 1 / M.BOHR_TO_ANGSTROM   # Å -> Bohr


def _line(*xs):
    """Atoms on the x axis at these positions (Å), in Bohr."""
    return np.array([[x * A, 0.0, 0.0] for x in xs])


def test_units():
    assert M.NN_ANGSTROM_KCAL == pytest.approx(14.393, abs=1e-3)
    assert M.NN_TO_HARTREE_PER_BOHR == pytest.approx(0.012138, abs=1e-5)


def test_levers_and_force_for_a_one_hour_half_life():
    # C0 ... C1 stretches by 1 Å at the TS, C1 ... C2 shortens by 0.5 Å.
    ch = M.Channel("c", "the reaction", 30.0, reactant=_line(0, 1.5, 3.5), ts=_line(0, 2.5, 4.0))
    out = M.analyze(["C", "C", "C"], [ch], top=3)
    row = out["channels"][0]
    fav = row["favor"][0]
    assert fav["label"] == "C0–C1" and fav["dq"] == pytest.approx(1.0)
    assert fav["barrier_at_1nN"] == pytest.approx(30.0 - 14.393, abs=1e-3)
    bench = M._bench_barrier(298.15)
    assert fav["force_for_1h"] == pytest.approx((30.0 - bench) / 14.393, rel=1e-3)
    assert row["disfavor"][0]["label"] == "C1–C2"
    assert out["insights"][0]["level"] == "accelerates" and "C0–C1" in out["insights"][0]["text"]


def test_a_pulling_pair_can_switch_the_selectivity():
    # Channel B is 5 kcal/mol higher, but its TS stretches C0-C2 by 1 Å while
    # A's does not: above ~0.35 nN (5 / 14.4 plus the margin) B wins.
    r = _line(0, 1.5, 3.0)
    a = M.Channel("a", "A", 20.0, reactant=r, ts=_line(0, 1.5, 3.0), kind="offtarget")
    b = M.Channel("b", "B", 25.0, reactant=r, ts=_line(0, 1.5, 4.0))
    out = M.analyze(["C", "C", "C"], [a, b], pairs=[(0, 2)])
    (s,) = out["selectivity"]
    assert s["channel"] == "b" and s["pair"] == [0, 2]
    assert 5.0 / 14.393 < s["force_nN"] < 5.6 / 14.393 + 0.02
    steer = [i for i in out["insights"] if i["level"] == "selectivity"]
    assert "intended" in steer[0]["text"]          # B is the direct channel, A leads elsewhere


def test_forced_engine_energy_and_gradient():
    from tests.test_solvation import _Base, _node

    eng = M.ForcedEngine(_Base(), pairs=[(0, 1)], force_nN=1.0)
    n = _node(0.0)
    f = M.NN_TO_HARTREE_PER_BOHR
    d = np.linalg.norm(n.coords[0] - n.coords[1])
    assert eng.compute_energies([n])[0] == pytest.approx(n.coords[1, 0] - f * d)
    # Finite differences of the forced energy match its gradient.
    x0, h = n.coords, 1e-4
    for k in range(3):
        dx = np.zeros_like(x0)
        dx[1, k] = h
        num = (eng.compute_energies([n.update_coords(x0 + dx)])[0]
               - eng.compute_energies([n.update_coords(x0 - dx)])[0]) / (2 * h)
        assert eng.compute_gradients([n.update_coords(x0)])[0][1, k] == pytest.approx(num, abs=1e-6)


def test_cli_force_on_a_ts_folder(tmp_path):
    import json

    from tests.test_solvation import _tsopt_folder

    from mepd.cli_force import force

    src = _tsopt_folder(tmp_path)
    out = tmp_path / "force"
    force(source=src, pairs=None, forces=None, mode="bell", top=3, hydrogens=True, max_force=2.5,
          temperature=298.15, inputs=None, charge=0, multiplicity=1, output=out)
    data = json.loads((out / "summary.json").read_text())
    (ch,) = data["channels"]
    assert ch["label"] == "the reaction" and ch["barrier_kcal"] == pytest.approx(35.0, abs=1e-6)
    # Atom 1 moves along x toward atom 2 from the reactant (written last, the
    # low end) to the TS: H1-H2 shortens by the full 0.5 bohr, so pulling it
    # apart holds the reaction back most.
    lever = ch["disfavor"][0]
    assert lever["label"] == "H1–H2" and lever["dq"] == pytest.approx(-0.5 * M.BOHR_TO_ANGSTROM, abs=1e-4)
    from mepd.web import results

    res = results.collect_mechanochem(out, 0, 1)
    assert res["barrier_kcal"] is None and res["mechano"]["channels"][0]["id"] == ch["id"]
