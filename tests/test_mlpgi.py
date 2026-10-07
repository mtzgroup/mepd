"""MLP-GI path minimization (mepd.mlp_geodesic, mepd.pathminimizers.mlpgi)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from qcdata import Structure

pytest.importorskip("ase")

from mepd import mlp_geodesic as gi
from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.inputs import ChainInputs, RunInputs
from mepd.nodes.node import StructureNode
from mepd.pathminimizers.mlpgi import KCAL_MOL_TO_EV, MLPGI, optimizer_config, surrogate_kwds


# ------------------------------------------------------------------ the geodesic math

def _quad_energy(x):
    """A double well along atom 0's x, harmonic elsewhere (eV, Angstrom)."""
    x = np.asarray(x, float)
    q = x[:, 0, 0]
    e = 0.5 * (q ** 2 - 1.0) ** 2 + 0.5 * (x[:, 1:] ** 2).sum(axis=(1, 2)) + 0.5 * (x[:, 0, 1:] ** 2).sum(axis=1)
    f = -x.copy()
    f[:, 0, 0] = -2.0 * q * (q ** 2 - 1.0)
    return e, f


def test_segment_length_is_the_arc_length_of_the_energy_parabola():
    e = np.array([0.0, 1.0, 0.3])
    e_mid = np.array([0.8, 0.9])
    got = gi.segment_lengths(e, e_mid)
    t = np.linspace(0.0, 1.0, 20001)
    for k in range(2):
        a, b, c = np.polyfit([0, 0.5, 1], [e[k], e_mid[k], e[k + 1]], 2)
        slope = 2 * a * t + b
        ref = np.trapezoid(np.sqrt(slope ** 2 + gi.EPS), t)
        assert got[k] == pytest.approx(ref, rel=1e-5)


def test_loss_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    nodes = np.stack([np.array([[q, 0.0, 0.0], [0.0, 1.0, 0.0]]) for q in np.linspace(-1, 1, 5)])
    nodes[1:-1] += rng.normal(scale=0.1, size=nodes[1:-1].shape)
    beta = 0.05

    def loss(r):
        mids = 0.5 * (r[:-1] + r[1:])
        e, _ = _quad_energy(r)
        em, _ = _quad_energy(mids)
        s = gi.segment_lengths(e, em)
        return s.sum() + beta * ((s / s.mean() - 1) ** 2).sum()

    e, f = _quad_energy(nodes)
    em, fm = _quad_energy(0.5 * (nodes[:-1] + nodes[1:]))
    pd = gi.PathData(nodes, e, f, em, fm)
    grad = gi.loss_gradient(pd, gi.segment_lengths(e, em), beta, tangent_project=False, climb=False, alpha_climb=0.5)
    h = 1e-6
    for i, a, d in [(1, 0, 0), (2, 0, 0), (2, 1, 1), (3, 0, 2)]:
        up, dn = nodes.copy(), nodes.copy()
        up[i, a, d] += h
        dn[i, a, d] -= h
        assert grad[i, a, d] == pytest.approx((loss(up) - loss(dn)) / (2 * h), rel=1e-4, abs=1e-7)


def test_optimizer_lowers_the_barrier_and_keeps_the_ends():
    # A path bowed over the ridge: the optimizer pulls it down to the saddle (q = 0, E = 0.5).
    q = np.linspace(-1, 1, 7)
    frames = np.stack([np.array([[x, 0.8 * (1 - x ** 2), 0.0], [0.0, 0.0, 0.0]]) for x in q])
    e0, _ = _quad_energy(frames)
    seen = []
    # (No alignment: this toy surface is not rotation-invariant.)
    opt = gi.GeodesicOptimizer(frames, ["C", "H"], _quad_energy, on_step=lambda pd, c: seen.append(c), align=False)
    final = opt.optimize()
    assert final.energies.max() < e0.max()
    assert final.energies.max() == pytest.approx(0.5, abs=0.05)
    np.testing.assert_allclose(final.nodes[0], frames[0])
    assert seen and seen[0].startswith("MLP-GI stage 1")
    assert opt.evaluations > 0


# ------------------------------------------------------------------ settings

def test_runinputs_mlpgi_defaults_and_aliases():
    for name in ("MLPGI", "mlpgi", "mlp-gi"):
        pmi = RunInputs(path_min_method=name).path_min_inputs
        assert pmi.fire_stage1_iter == 200 and pmi.fire_stage2_iter == 500
        assert pmi.mlp_model is None and pmi.climb is True


def test_optimizer_config_units_and_paper_names():
    cfg = optimizer_config({"fire_conv_geolen_tol": 0.5, "beta": 2.0, "cutoff": 10, "tau_refine": 5})
    assert cfg.fire_conv_geolen_tol == pytest.approx(0.5 * KCAL_MOL_TO_EV)
    assert cfg.variance_penalty_weight == pytest.approx(2.0 * KCAL_MOL_TO_EV)
    assert cfg.refinement_dynamic_threshold_fraction == pytest.approx(0.1)   # 10 (%) -> 0.1
    assert cfg.refinement_step_interval == 5
    assert optimizer_config({}).fire_conv_erelpeak_tol == pytest.approx(0.25 * KCAL_MOL_TO_EV, rel=1e-3)


def test_surrogate_choice():
    assert surrogate_kwds({}) is None
    assert surrogate_kwds({"backend": "engine"}) is None
    assert surrogate_kwds({"mlp_model": "aimnet2", "mlp_device": "cpu"}) == {"model": "aimnet2", "device": "cpu"}
    # neb-dynamics' fairchem checkpoint name, not on disk here: the mepd model of that name.
    assert surrogate_kwds({"backend": "fairchem", "model_path": "esen_sm_conserving_all.pt"}) == {
        "model": "esen-sm-conserving-all-omol"}


# ------------------------------------------------------------------ the path method

class _ProtonTransferEngine(Engine):
    """A-H-B (bohr, hartree): Morse A-H and H-B plus a stiff A-B spring --
    a symmetric double well with the barrier at the midpoint."""

    def _eg(self, x):
        def morse(i, j, d=0.1, a=1.5, r0=1.9):
            v = x[i] - x[j]
            r = np.linalg.norm(v)
            ex = np.exp(-a * (r - r0))
            return d * (1 - ex) ** 2, 2 * d * (1 - ex) * a * ex * v / r
        e1, g1 = morse(0, 1)
        e2, g2 = morse(1, 2)
        v = x[0] - x[2]
        r = np.linalg.norm(v)
        e3, g3 = 0.5 * (r - 6.0) ** 2, (r - 6.0) * v / r
        g = np.zeros_like(x)
        g[0] += g1 + g3
        g[1] += -g1 + g2
        g[2] += -g2 - g3
        return e1 + e2 + e3, g

    def compute_energies(self, nodes):
        for n in nodes:
            n._cached_energy, n._cached_gradient = self._eg(np.asarray(n.coords, float))
        return np.array([n._cached_energy for n in nodes])

    def compute_gradients(self, nodes):
        self.compute_energies(nodes)
        return np.array([n._cached_gradient for n in nodes])


def _node(h_x):
    geom = np.array([[0.0, 0.0, 0.0], [h_x, 0.3, 0.0], [6.0, 0.0, 0.0]])
    return StructureNode(structure=Structure(symbols=["O", "H", "O"], geometry=geom, charge=-1, multiplicity=1),
                         has_molecular_graph=False)


def test_mlpgi_path_method_on_an_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("MEPD_DRIVE_CHAIN_DIR", str(tmp_path / "live"))
    chain = Chain.model_validate({"nodes": [_node(x) for x in np.linspace(1.9, 4.1, 7)],
                                  "parameters": ChainInputs()})
    params = RunInputs(path_min_method="MLPGI", path_min_inputs={"do_elem_step_checks": False}).path_min_inputs
    engine = _ProtonTransferEngine()
    mini = MLPGI(initial_chain=chain, engine=engine, parameters=params)
    mini.optimize_chain()
    out = mini.optimized
    e = out.energies
    assert out.nodes[0] is chain.nodes[0] or np.allclose(out.nodes[0].coords, chain.nodes[0].coords)
    assert np.allclose(out.nodes[-1].coords, chain.nodes[-1].coords)
    peak = out.nodes[int(np.argmax(e))].coords
    r_ah, r_hb = np.linalg.norm(peak[1] - peak[0]), np.linalg.norm(peak[2] - peak[1])
    assert abs(r_ah - r_hb) < 0.3                       # the barrier sits halfway
    assert e.max() <= mini.chain_trajectory[0].energies.max() + 1e-9
    assert mini.grad_calls_made > len(chain) and mini.surrogate_calls_made == 0
    assert len(mini.chain_trajectory) > 2
    live = json.loads(next((tmp_path / "live").glob("*.json")).read_text())
    assert "MLP-GI" in json.dumps(live)
