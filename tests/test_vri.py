from __future__ import annotations

import numpy as np
import pytest
from qcdata import Structure
from scipy.optimize import minimize

from mepd.discovery import vri
from mepd.elementarystep import prepare_modes
from mepd.engines.engine import Engine, build_hessian_result_from_matrix
from mepd.helper_functions import get_mass
from mepd.nodes.node import StructureNode, XYNode


# --------------------------------------------------------------------------
# Analytic 2D surfaces
# --------------------------------------------------------------------------


class _Surface:
    """`sym`: V = x^3/3 - x - x y^2/2 + y^4/4 + eps*y. With eps=0: TS1 at
    (-1, 0), VRI exactly at the origin (g = (-1, 0), H_yy = 0), TS2 at
    (1, 0), minima at (1.2808, +/-1.1317). eps != 0 tilts it (asymmetric).

    `dead`: V = x^3/3 - x + y^2/2 (1 - a exp(-x^2/w)) + y^4/4. The transverse
    curvature goes negative only for |x| < ~0.2, then the valley re-forms and
    ends in a single minimum at (1, 0) -- a VRT with no second product."""

    def __init__(self, kind: str = "sym", eps: float = 0.0, a: float = 1.5, w: float = 0.1):
        self.kind, self.eps, self.a, self.w = kind, eps, a, w

    def _f(self, x):
        e = np.exp(-x * x / self.w)
        a, w = self.a, self.w
        return 1 - a * e, a * (2 * x / w) * e, a * e * (2 / w - 4 * x * x / w**2)

    def V(self, p):
        x, y = p
        if self.kind == "sym":
            return x**3 / 3 - x - x * y * y / 2 + y**4 / 4 + self.eps * y
        f, _, _ = self._f(x)
        return x**3 / 3 - x + y * y / 2 * f + y**4 / 4

    def g(self, p):
        x, y = p
        if self.kind == "sym":
            return np.array([x * x - 1 - y * y / 2, -x * y + y**3 + self.eps])
        f, f1, _ = self._f(x)
        return np.array([x * x - 1 + y * y / 2 * f1, y * f + y**3])

    def H(self, p):
        x, y = p
        if self.kind == "sym":
            return np.array([[2 * x, -y], [-y, -x + 3 * y * y]])
        f, f1, f2 = self._f(x)
        return np.array([[2 * x + y * y / 2 * f2, y * f1], [y * f1, f + 3 * y * y]])


class _AnalyticEngine(Engine):
    def __init__(self, surface: _Surface):
        self.surface = surface
        self.hessian_calls = 0

    def compute_energies(self, nodes):
        return [self.surface.V(np.asarray(n.coords)) for n in nodes]

    def compute_gradients(self, nodes):
        return [self.surface.g(np.asarray(n.coords)) for n in nodes]

    def compute_hessian(self, node, step_size=None):
        self.hessian_calls += 1
        return self.surface.H(np.asarray(node.coords))

    def compute_geometry_optimization(self, node, keywords=None):
        s = self.surface
        res = minimize(s.V, np.asarray(node.coords, float), jac=s.g, method="BFGS", options={"gtol": 1e-9})
        return [_xy(s, res.x)]


def _xy(surface, p) -> XYNode:
    node = XYNode(structure=np.array(p, dtype=float))
    node._cached_energy = float(surface.V(node.coords))
    node._cached_gradient = surface.g(node.coords)
    return node


def _find_ts(surface, guess):
    p = np.array(guess, dtype=float)
    for _ in range(50):
        p = p - np.linalg.solve(surface.H(p), surface.g(p))
    return p


def _irc(surface, ts, h=0.01, n_forward=250, n_reverse=60):
    """Fixed-step steepest descent from TS1 along +/- its imaginary mode,
    ordered reverse-end -> TS -> forward-end like mepd's IRC backends."""
    _, vecs = np.linalg.eigh(surface.H(ts))
    d = vecs[:, 0] if vecs[0, 0] > 0 else -vecs[:, 0]
    branches = []
    for sign, nmax in ((1, n_forward), (-1, n_reverse)):
        p = ts + sign * h * d
        pts, e = [p.copy()], surface.V(p)
        for _ in range(nmax):
            g = surface.g(p)
            if np.linalg.norm(g) < 1e-3:
                break
            q = p - h * g / np.linalg.norm(g)
            if surface.V(q) > e:
                break
            p, e = q, surface.V(q)
            pts.append(p.copy())
        branches.append(pts)
    forward, reverse = branches
    return (
        [_xy(surface, p) for p in reverse[::-1]]
        + [_xy(surface, ts)]
        + [_xy(surface, p) for p in forward]
    )


_TOY = dict(vrt_threshold=0.05, persist=2)


def _run(surface, **products_kwargs):
    engine = _AnalyticEngine(surface)
    ts = _find_ts(surface, [-1.0, 0.0])
    nodes = _irc(surface, ts)
    scan = vri.scan_irc_for_vrt(nodes, engine, **_TOY)
    reactant = scan.branches["reverse"].nodes[-1]
    products = None
    if scan.branches["forward"].vrt is not None:
        products = vri.find_bifurcation_products(
            scan, "forward", engine, reference_nodes=[reactant],
            push_amplitude=0.3, imaginary_cutoff=0.05, **products_kwargs,
        )
    return scan, products


def test_symmetric_surface_vrt_is_the_vri_and_irc_ends_on_ts2():
    surface = _Surface("sym")
    scan, products = _run(surface)

    forward = scan.branches["forward"]
    assert forward.vrt is not None
    assert np.linalg.norm(forward.vrt.node.coords) < 0.02  # the VRI is at the origin
    assert abs(forward.vrt.ridge_mode_cart[1]) > 0.99      # ridge mode is along y
    assert not forward.valley_reforms
    assert scan.branches["reverse"].vrt is None

    assert products.endpoint_n_imaginary == 1
    assert products.ts2_source == "irc_endpoint"
    np.testing.assert_allclose(products.ts2.coords, [1.0, 0.0], atol=1e-5)
    got = sorted(float(n.coords[1]) for n in (products.p1, products.p2))
    np.testing.assert_allclose(got, [-1.1317, 1.1317], atol=1e-3)
    for n in (products.p1, products.p2):
        assert float(n.coords[0]) == pytest.approx(1.2808, abs=1e-3)
    assert products.ts2_checks == {"n_imaginary": 1, "below_ts1": True, "connects_p1_p2": True}
    # the verdict waits for the basin test
    assert vri.branch_verdict(forward, products) == "second_product_untested"


def test_asymmetric_surface_irc_bypasses_vri_and_push_finds_second_product():
    surface = _Surface("sym", eps=0.05)
    scan, products = _run(surface)

    forward = scan.branches["forward"]
    assert forward.vrt is not None
    assert abs(float(forward.vrt.node.coords[0])) < 0.2
    assert products.endpoint_n_imaginary == 0
    assert products.ts2 is None
    # Tilt favours y < 0: the IRC ends there, the push finds the y > 0 product.
    assert float(products.p1.coords[1]) < -1.0
    assert float(products.p2.coords[1]) > 1.0
    assert vri.branch_verdict(forward, products) == "second_product_untested"


def test_dead_end_ridge_reports_vrt_without_second_product():
    scan, products = _run(_Surface("dead"))

    forward = scan.branches["forward"]
    assert forward.vrt is not None
    assert float(forward.vrt.node.coords[0]) == pytest.approx(-0.2, abs=0.02)
    assert forward.valley_reforms
    assert products.p2 is None
    assert vri.branch_verdict(forward, products) == "vrt_no_second_product"


def test_plain_valley_has_no_vrt():
    surface = _Surface("dead", a=0.5)  # transverse curvature never goes negative
    scan, products = _run(surface)
    assert all(b.vrt is None for b in scan.branches.values())
    assert products is None
    verdicts = [vri.branch_verdict(b, None) for b in scan.branches.values()]
    assert vri.overall_verdict(verdicts) == "no_vrt"


def test_verify_ts2_checks_connectivity_energy_and_curvature():
    surface = _Surface("sym")
    engine = _AnalyticEngine(surface)
    ts1, ts2 = _xy(surface, [-1, 0]), _xy(surface, [1, 0])
    p1, p2 = _xy(surface, [1.2808, 1.1317]), _xy(surface, [1.2808, -1.1317])

    good = vri.verify_ts2(ts2, [p2, ts2, p1], p1, p2, ts1, engine, imaginary_cutoff=0.05)
    assert good["verified"]

    bad = vri.verify_ts2(ts2, [p2, ts2, p2], p1, p2, ts1, engine, imaginary_cutoff=0.05)
    assert not bad["verified"] and bad["connects_p1_p2"] is False


# --------------------------------------------------------------------------
# VRT detection
# --------------------------------------------------------------------------


def test_detect_vrt_depth_window_limits_the_onset_not_the_run():
    # rxn_072: the ridge begins inside the window and keeps going past it
    freqs = [30, 12, -56, -198, -280, -300]
    ok = [True, True, True, False, False, False]
    first, transients, _ = vri._detect_vrt(freqs, threshold=50, persist=2, may_start=ok)
    assert first == 2 and transients == []
    # a dip that only begins past the window is not a VRT
    first, _, _ = vri._detect_vrt([30, 20, 10, -80, -90], threshold=50, persist=2,
                                  may_start=[True, True, True, False, False])
    assert first is None


def test_detect_vrt_rejects_single_point_noise_and_flags_reformed_valley():
    freqs = [100, 80, -30, 60, 40, 10, -25, -60, -90, -40, 30]
    first, transients, reforms = vri._detect_vrt(freqs, threshold=20, persist=2)
    assert first == 6
    assert transients == [2]
    assert reforms

    first, transients, _ = vri._detect_vrt([100, -30, 50, 40], threshold=20, persist=2)
    assert first is None and transients == [1]

    # A dip reaching the end of the branch counts even if shorter than `persist`.
    first, _, reforms = vri._detect_vrt([100, 50, -30], threshold=20, persist=3)
    assert first == 2 and not reforms


def test_transient_only_branch_verdict():
    branch = vri.BranchScan(name="forward", nodes=[], s=np.zeros(0), transient_dips=[0.3])
    assert vri.branch_verdict(branch, None) == "transient_softening"


# --------------------------------------------------------------------------
# Projected Hessian math
# --------------------------------------------------------------------------


def _water() -> StructureNode:
    return StructureNode(structure=Structure(
        symbols=["O", "H", "H"],
        geometry=np.array([[0.0, 0.0, 0.2], [1.43, 0.0, -0.9], [-1.43, 0.1, -0.95]]),
        charge=0, multiplicity=1,
    ))


def _rotation(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _invariant_hessian(node, seed=1):
    """A random Hessian with exact zero rigid-body modes (like a real one)."""
    masses = np.array([get_mass(s) for s in node.symbols])
    B = vri.rigid_body_basis(node.coords, masses)
    n = node.coords.size
    A = np.random.default_rng(seed).normal(size=(n, n))
    A = A @ A.T - 2.0 * np.eye(n)
    P = np.eye(n) - B @ B.T
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    return (P @ A @ P) * np.outer(sqrt_m, sqrt_m)


def test_rigid_body_basis_is_orthonormal_and_detects_linear_molecules():
    water = _water()
    masses = np.array([get_mass(s) for s in water.symbols])
    B = vri.rigid_body_basis(water.coords, masses)
    assert B.shape == (9, 6)
    np.testing.assert_allclose(B.T @ B, np.eye(6), atol=1e-10)

    co2 = np.array([[0.0, 0, 0], [0, 0, 2.2], [0, 0, -2.2]])
    B_lin = vri.rigid_body_basis(co2, np.array([12.0, 16.0, 16.0]))
    assert B_lin.shape == (9, 5)

    assert vri.rigid_body_basis(np.array([0.3, -0.2]), None).shape == (2, 0)


def test_projected_frequencies_match_prepare_modes_at_a_stationary_point():
    water = _water()
    H = _invariant_hessian(water)
    masses = np.array([get_mass(s) for s in water.symbols])
    ours = vri.projected_frequencies(H, water.coords, masses, gradient=None)
    ref = prepare_modes(build_hessian_result_from_matrix(water, H), water)
    np.testing.assert_allclose(ours.freqs, sorted(ref.freqs_wavenumber), rtol=1e-6)


def test_projected_frequencies_invariant_under_rigid_rotation():
    water = _water()
    masses = np.array([get_mass(s) for s in water.symbols])
    H = _invariant_hessian(water)
    g = np.random.default_rng(3).normal(size=9) * 1e-2
    R = _rotation()
    Rb = np.kron(np.eye(3), R)
    coords_rot = water.coords @ R.T
    a = vri.projected_frequencies(H, water.coords, masses, gradient=g)
    b = vri.projected_frequencies(Rb @ H @ Rb.T, coords_rot, masses, gradient=Rb @ g)
    assert a.tangent_source == b.tangent_source == "gradient"
    assert len(a.freqs) == 9 - 7
    np.testing.assert_allclose(a.freqs, b.freqs, rtol=1e-8)


def test_product_of_projected_eigenvalues_is_g_adjH_g():
    rng = np.random.default_rng(7)
    n = 5
    A = rng.normal(size=(n, n))
    H = A + A.T
    g = rng.normal(size=n)
    modes = vri.projected_frequencies(H, np.zeros(n), None, gradient=g)
    adj = np.linalg.det(H) * np.linalg.inv(H)
    assert np.prod(modes.eigvals) == pytest.approx(g @ adj @ g / (g @ g), rel=1e-8)


def test_small_gradient_falls_back_to_supplied_tangent():
    H = np.diag([-1.0, 2.0, 3.0])
    tangent = np.array([1.0, 0.0, 0.0])
    modes = vri.projected_frequencies(
        H, np.zeros(3), None, gradient=np.array([1e-7, 0, 0]), tangent_mw=tangent,
    )
    assert modes.tangent_source == "supplied"
    np.testing.assert_allclose(modes.eigvals, [2.0, 3.0])


def test_labeled_bond_sets_distinguish_degenerate_products():
    # C2 + C: same molecule, different carbon bonded -- atom-distinct but isomorphic.
    def c3(bonded_to_first: bool):
        geom = np.array([[0.0, 0, 0], [2.5, 0, 0], [15.0, 0, 0]]) if bonded_to_first else \
            np.array([[0.0, 0, 0], [15.0, 0, 0], [17.5, 0, 0]])
        return StructureNode(structure=Structure(symbols=["C", "C", "C"], geometry=geom, charge=0, multiplicity=1))

    a, b = c3(True), c3(False)
    assert not vri.same_species(a, b)
    assert vri.same_species(a, c3(True))
    assert vri.products_isomorphic(a, b) is True


def test_parallel_scan_matches_serial():
    surface = _Surface("sym", eps=0.05)
    ts = _find_ts(surface, [-1.0, 0.0])
    serial = vri.scan_irc_for_vrt(_irc(surface, ts), _AnalyticEngine(surface), **_TOY)
    parallel = vri.scan_irc_for_vrt(_irc(surface, ts), _AnalyticEngine(surface), workers=4, **_TOY)
    a, b = serial.branches["forward"], parallel.branches["forward"]
    assert [p.lowest_eigval for p in a.points] == [p.lowest_eigval for p in b.points]
    # k-section with 4 interior points per round brackets tighter, same crossing.
    assert abs(a.vrt.s - b.vrt.s) < 0.01
    products = vri.find_bifurcation_products(
        parallel, "forward", _AnalyticEngine(surface),
        reference_nodes=[parallel.branches["reverse"].nodes[-1]],
        push_amplitude=0.3, imaginary_cutoff=0.05, workers=4,
    )
    assert products.p2 is not None and float(products.p2.coords[1]) > 1.0


def test_late_dips_beyond_energy_window_are_not_vrts():
    # The dead-end surface's dip sits ~0.6 units below TS1 (hundreds of
    # kcal/mol once read as Hartree), so a 1 kcal/mol window excludes it.
    surface = _Surface("dead")
    ts = _find_ts(surface, [-1.0, 0.0])
    scan = vri.scan_irc_for_vrt(_irc(surface, ts), _AnalyticEngine(surface), max_vrt_depth=1.0, **_TOY)
    forward = scan.branches["forward"]
    assert forward.vrt is None
    assert forward.late_dips
    assert vri.branch_verdict(forward, None) == "no_vrt"


def test_basin_test_decides_the_verdict_for_a_second_product():
    branch = vri.BranchScan(name="forward", nodes=[], s=np.zeros(0))
    products = vri.BranchProducts(branch="forward", p1=object(), p2=object(), evidence="vrt_push")
    assert vri.branch_verdict(branch, products) == "second_product_untested"
    products.checks = {"basin": {"counts": {"P1": 30}}}
    assert vri.branch_verdict(branch, products) == "second_product_no_split"
    products.checks = {"basin": {"counts": {"P1": 21, "P2": 9}}}
    assert vri.branch_verdict(branch, products) == "bifurcation"
    products.ts2_verified = False  # a missing/unverified TS2 does not overrule the basin test
    assert vri.branch_verdict(branch, products) == "bifurcation"
    # a third, symmetry-equivalent product counts; returning to the reactant does not
    products.checks = {"basin": {"counts": {"P1": 15, "P3": 15}}}
    assert vri.branch_verdict(branch, products) == "bifurcation"
    products.checks = {"basin": {"counts": {"P1": 20, "R": 10}}}
    assert vri.branch_verdict(branch, products) == "second_product_no_split"


def test_species_identity_ignores_which_equivalent_hydrogen_moved():
    # H3 chain vs same chain with the two terminal H's swapped: identical.
    geom = np.array([[0.0, 0, 0], [1.4, 0, 0], [8.0, 0, 0]])
    a = StructureNode(structure=Structure(symbols=["C", "H", "H"], geometry=geom, charge=0, multiplicity=3))
    b = StructureNode(structure=Structure(symbols=["C", "H", "H"], geometry=geom[[0, 2, 1]], charge=0, multiplicity=3))
    assert vri.labeled_bond_set(a) != vri.labeled_bond_set(b)
    assert vri.same_species(a, b)


def test_gradient_hessian_engine_reproduces_an_analytic_hessian_and_caps_concurrency():
    import threading
    import time

    class _Quartic(Engine):
        """E = 1/2 x.A.x + 1/4 sum(x^4): an anharmonic surface with a known Hessian."""

        def __init__(self):
            rng = np.random.default_rng(4)
            M = rng.normal(size=(9, 9))
            self.A = M @ M.T
            self.active = 0
            self.peak = 0
            self.lock = threading.Lock()

        def compute_energies(self, nodes):
            return [float(0.5 * x @ self.A @ x + 0.25 * np.sum(x**4)) for x in (n.coords.reshape(-1) for n in nodes)]

        def compute_gradients(self, nodes):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.001)
            out = [(self.A @ x + x**3).reshape(n.coords.shape) for n, x in ((n, n.coords.reshape(-1)) for n in nodes)]
            with self.lock:
                self.active -= 1
            return out

        def hessian(self, x):
            return self.A + np.diag(3 * x**2)

    base = _Quartic()
    node = _water()
    wrapped = vri.GradientHessianEngine(base, max_concurrent=3)
    H = wrapped.compute_hessian(node)
    np.testing.assert_allclose(H, base.hessian(node.coords.reshape(-1)), atol=1e-4)
    assert base.peak <= 3
    assert wrapped.compute_energies([node]) == base.compute_energies([node])  # delegation


def test_verify_ts2_rejects_a_ts2_that_is_ts1_itself():
    surface = _Surface("sym")
    engine = _AnalyticEngine(surface)
    ts1 = _xy(surface, [-1, 0])
    p1, p2 = _xy(surface, [1.2808, 1.1317]), _xy(surface, [1.2808, -1.1317])
    same = vri.verify_ts2(_xy(surface, [-1, 0]), [p2, ts1, p1], p1, p2, ts1, engine, imaginary_cutoff=0.05)
    assert same["distinct_from_ts1"] is False and not same["verified"]


def test_scan_hessians_are_kept_and_saved(tmp_path):
    surface = _Surface("sym")
    ts = _find_ts(surface, [-1.0, 0.0])
    scan = vri.scan_irc_for_vrt(_irc(surface, ts), _AnalyticEngine(surface), **_TOY)
    forward = scan.branches["forward"]
    assert len(forward.hessians) == len(forward.points)
    assert len(forward.bisection_hessians) == len(forward.bisection_points)
    np.testing.assert_allclose(forward.hessians[3], surface.H(forward.points[3].node.coords), rtol=1e-6)
    written = vri.save_hessians(scan, tmp_path)
    data = np.load(tmp_path / "hessians_forward.npz")
    assert {fp.name for fp in written} == {"hessians_forward.npz", "hessians_reverse.npz"}
    n = len(forward.points) + len(forward.bisection_points)
    assert data["hessian"].shape == (n, 2, 2) and data["bisection"].sum() == len(forward.bisection_points)
