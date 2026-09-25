from __future__ import annotations

import numpy as np

from mepd.discovery import vri, vri_checks
from tests.test_vri import _AnalyticEngine, _Surface, _find_ts, _irc, _xy


def _vri_residual(surface, p):
    """det H and g . adj(H) . g: both vanish at a VRI (2D)."""
    H, g = surface.H(p), surface.g(p)
    adj = np.array([[H[1, 1], -H[0, 1]], [-H[1, 0], H[0, 0]]])
    return np.linalg.det(H), g @ adj @ g


def test_refine_vri_symmetric_surface_converges_to_the_origin():
    surface = _Surface("sym")
    engine = _AnalyticEngine(surface)
    res = vri_checks.refine_vri(_xy(surface, [0.15, 0.0]), np.array([0.2, 1.0]), engine)
    assert res.converged
    np.testing.assert_allclose(res.node.coords, [0.0, 0.0], atol=1e-5)
    assert abs(res.v[1]) > 0.999  # zero-curvature direction is y, perpendicular to g


def test_refine_vri_asymmetric_surface_lands_on_the_vri_manifold():
    surface = _Surface("sym", eps=0.05)
    engine = _AnalyticEngine(surface)
    ts = _find_ts(surface, [-1.0, 0.0])
    scan = vri.scan_irc_for_vrt(_irc(surface, ts), engine, vrt_threshold=0.05, persist=2)
    vrt = scan.branches["forward"].vrt
    res = vri_checks.refine_vri(vrt.node, vrt.ridge_mode_cart, engine)
    assert res.converged
    det, gag = _vri_residual(surface, np.asarray(res.node.coords))
    assert abs(det) < 1e-6 and abs(gag) < 1e-6
    assert res.v_dot_g < 1e-4 and res.grad_norm > 0.1  # not a stationary point
    # the VRI is off the IRC but close to the VRT
    assert 0.0 < np.linalg.norm(np.asarray(res.node.coords) - np.asarray(vrt.node.coords)) < 0.2


def test_basin_test_splits_on_a_bifurcating_surface_and_not_on_a_valley():
    for kind, expect_split in (("sym", True), ("dead", False)):
        surface = _Surface(kind, eps=0.02 if kind == "sym" else 0.0)
        engine = _AnalyticEngine(surface)
        ts = _find_ts(surface, [-1.0, 0.0])
        nodes = _irc(surface, ts)
        scan = vri.scan_irc_for_vrt(nodes, engine, vrt_threshold=0.05, persist=2)
        br = scan.branches["forward"]
        pts = [p for p in br.points if 0.3 < p.s < 2.2][::8]
        p1 = _xy(surface, [1.2808, -1.1317]) if kind == "sym" else _xy(surface, [1.0, 0.0])
        p2 = _xy(surface, [1.2808, 1.1317]) if kind == "sym" else _xy(surface, [9.0, 9.0])
        res = vri_checks.basin_test([p.node for p in pts], [p.s for p in pts], [np.array([0.0, 1.0])] * len(pts),
                                    p1, p2, engine, displacements=(0.05, 0.1, 0.2))
        split = res.counts.get("P1", 0) > 0 and res.counts.get("P2", 0) > 0
        assert split == expect_split, (kind, res.counts)


def test_product_registry_names_further_products_and_the_reactant():
    surface = _Surface("sym")
    p1, p2, r = _xy(surface, [1.28, 1.13]), _xy(surface, [1.28, -1.13]), _xy(surface, [-2.0, 0.0])
    reg = vri_checks.ProductRegistry(p1, p2, [r])
    labels = [reg.label(n) for n in (_xy(surface, [1.28, 1.13]), _xy(surface, [5.0, 5.0]), _xy(surface, [-2.0, 0.0]),
                                    _xy(surface, [5.0, 5.0]), _xy(surface, [7.0, 0.0]), None)]
    assert labels == ["P1", "P3", "R", "P3", "P4", "failed"]
    assert [e["label"] for e in reg.describe()] == ["P1", "P2", "P3", "P4"]


def test_basin_paths_are_recorded_and_saved_with_the_other_kind_kept(tmp_path):
    surface = _Surface("sym", eps=0.02)
    engine = _AnalyticEngine(surface)
    ts = _find_ts(surface, [-1.0, 0.0])
    nodes = _irc(surface, ts)
    scan = vri.scan_irc_for_vrt(nodes, engine, vrt_threshold=0.05, persist=2)
    pt = [p for p in scan.branches["forward"].points if 1.0 < p.s][0]
    p1, p2 = _xy(surface, [1.2808, -1.1317]), _xy(surface, [1.2808, 1.1317])
    res = vri_checks.basin_test([pt.node], [pt.s], [np.array([0.0, 1.0])], p1, p2, engine, displacements=(0.2,))
    assert len(res.paths) == len(res.starts) == 2
    for path, start in zip(res.paths, res.starts):
        assert len(path) >= 2
        assert path[0][1] > path[-2][1]  # descends
        end = np.asarray(path[-1][0]).reshape(-1)
        target = p1 if start["outcome"] == "P1" else p2
        assert np.allclose(end, np.asarray(target.coords).reshape(-1), atol=0.05)

    fp = tmp_path / "paths_forward.npz"
    vri_checks.save_paths(fp, "trajectory", [[(np.zeros(2), -1.0, np.ones(2))]], ["P2"], [{"index": 0}])
    vri_checks.save_paths(fp, "basin", res.paths, [s["outcome"] for s in res.starts], res.starts)
    z = np.load(fp)
    assert list(z["trajectory_labels"]) == ["P2"]
    assert z["basin_offsets"][-1] == len(z["basin_coords"]) == sum(len(p) for p in res.paths)
    assert sorted(z["basin_labels"]) == ["P1", "P2"]
