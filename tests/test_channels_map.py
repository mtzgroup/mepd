"""The channels path map: every live path placed in bond-progress or
mapping-independent distance coordinates, over a fitted surface."""
import json

import numpy as np

from mepd.web.channels_pes import channels_map

H2K = 627.509474


def _xyz(coords, symbols):
    return f"{len(symbols)}\n\n" + "".join(f"{s} {x:.4f} {y:.4f} {z:.4f}\n" for s, (x, y, z) in zip(symbols, coords))


def _write_stream(live, name, frames, energies_kcal, ref_hartree=-1.0, active=True):
    live.mkdir(parents=True, exist_ok=True)
    (live / f"{name}.json").write_text(json.dumps({"finished": False, "monitors": {"branch-0": {
        "active": active, "geometry": {"frames": frames},
        "plot": {"y": energies_kcal, "energy_ref_hartree": ref_hartree}}}}))


def _exchange(n=7):
    # H1-H2 + H3 -> H1 + H2-H3 along a line.
    out = []
    for t in np.linspace(0, 1, n):
        out.append(_xyz([(0, 0, 0), (0.74 + 1.2 * t, 0, 0), (0.74 + 1.2 + 0.74, 0, 0)], ["H", "H", "H"]))
    return out


def test_bond_progress_runs_from_reactant_to_product(tmp_path):
    job = tmp_path / "job"
    _write_stream(job / "live", "pair_0_1", _exchange(), [0, 5, 12, 15, 12, 6, -1])
    _write_stream(job / "live", "pair_1_1", _exchange(), [0, 4, 9, 11, 9, 4, -2], ref_hartree=-1.0 + 2 / H2K)
    m = channels_map(job, "bonds", running=True)
    assert len(m["paths"]) == 2
    assert any("no gradient" in w for w in m["warnings"])        # these streams carry no gradients: said so
    p = next(p for p in m["paths"] if p["pair"] == "pair_0_1")
    assert p["points"][0][:2] == [0.0, 0.0] and p["points"][-1][:2] == [1.0, 1.0]
    assert p["bonds"]["kind"] == "break-form" and p["bonds"]["x"] == ["H1–H2"] and p["bonds"]["y"] == ["H2–H3"]
    # One common energy scale: the second pair starts 2 kcal/mol higher.
    q = next(p for p in m["paths"] if p["pair"] == "pair_1_1")
    assert abs(q["points"][0][2] - 2.0) < 1e-6 and p["active"]
    assert len(m["E"]) == len(m["y"]) and len(m["E"][0]) == len(m["x"])
    assert not channels_map(job, "bonds", running=False)["paths"][0]["active"]   # a stopped run relaxes nothing


def test_only_formed_bonds_are_split_against_each_other(tmp_path):
    # Two bonds form (C0-C2 then C1-C3): asynchronous -- the first closes before the second.
    frames = []
    for t in np.linspace(0, 1, 6):
        d1 = 3.0 - 1.5 * min(1.0, 2 * t)
        d2 = 3.0 - 1.5 * max(0.0, 2 * t - 1)
        frames.append(_xyz([(0, 0, 0), (0, 1.5, 0), (d1, 0, 0), (d2, 1.5, 0)], ["C", "C", "C", "C"]))
    job = tmp_path / "job"
    _write_stream(job / "live", "pair_0_0", frames, [0, 10, 20, 15, 5, -3])
    m = channels_map(job, "bonds")
    (p,) = m["paths"]
    assert p["bonds"]["kind"] == "formed" and "first bond(s) formed" in m["axes"]["x"]
    xs = [q[0] for q in p["points"]]
    ys = [q[1] for q in p["points"]]
    assert xs[2] > 0.7 and ys[2] < 0.1           # bowed toward an edge: stepwise


def test_distance_coordinates_ignore_atom_numbering(tmp_path):
    frames = _exchange()
    job = tmp_path / "job"
    _write_stream(job / "live", "pair_0_1", frames, [0, 5, 12, 15, 12, 6, -1])
    renumbered = ["\n".join([f.splitlines()[0], "", f.splitlines()[4], f.splitlines()[2], f.splitlines()[3]]) + "\n"
                  for f in frames]
    _write_stream(job / "live", "pair_0_2", renumbered, [0, 5, 12, 15, 12, 6, -1])
    m = channels_map(job, "distance")
    a, b = (p["points"] for p in m["paths"])
    assert np.allclose([q[:2] for q in a], [q[:2] for q in b], atol=1e-6)   # same structures, same place


def test_gradients_are_projected_exactly_onto_the_bond_coordinates(tmp_path):
    """E = a*d12 + b*d23 is linear in the two bond lengths, so dE/dq along
    each normalized bond coordinate is exactly a*(d_end - d_start) (and b*...)."""
    from mepd.web.channels_pes import BOHR, _ddist, _projected_gradient

    X = np.array([[0.0, 0, 0], [1.3, 0.1, 0], [2.6, -0.1, 0.2]])
    a, b = 0.02, -0.03                        # Hartree / Angstrom
    s12, e12, s23, e23 = 0.74, 1.94, 1.94, 0.74
    B = np.array([(_ddist(X, 0, 1) / (e12 - s12)).reshape(-1), (_ddist(X, 1, 2) / (e23 - s23)).reshape(-1)])
    g_angstrom = (a * _ddist(X, 0, 1) + b * _ddist(X, 1, 2)).reshape(-1)
    g_q = _projected_gradient(B, g_angstrom * BOHR)            # payload units: Hartree / bohr
    assert np.allclose(g_q, [a * (e12 - s12) * H2K, b * (e23 - s23) * H2K])


def test_live_gradients_reach_the_fit(tmp_path):
    frames = _exchange()
    job = tmp_path / "job"
    live = job / "live"
    live.mkdir(parents=True)
    grads = [[0.001 * (k - 3)] * 9 for k in range(len(frames))]
    (live / "pair_0_1.json").write_text(json.dumps({"finished": False, "monitors": {"branch-0": {
        "active": True, "geometry": {"frames": frames, "gradients": grads},
        "plot": {"y": [0, 5, 12, 15, 12, 6, -1], "energy_ref_hartree": -1.0}}}}))
    m = channels_map(job, "bonds")
    assert not any("no gradient" in w for w in m["warnings"]) and m["E"]
