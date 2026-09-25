from __future__ import annotations

import numpy as np
from qcdata import Structure

from mepd.discovery import qct, vri
from mepd.engines.engine import Engine
from mepd.helper_functions import get_mass
from mepd.nodes.node import StructureNode


def _water() -> StructureNode:
    return StructureNode(structure=Structure(
        symbols=["O", "H", "H"],
        geometry=np.array([[0.0, 0.0, 0.2], [1.43, 0.0, -0.9], [-1.43, 0.1, -0.95]]),
        charge=0, multiplicity=1,
    ))


class _QuadraticEngine(Engine):
    """E = 1/2 dx^T H dx around a reference geometry, with a Hessian that
    has exact rigid-body zero modes, one imaginary mode (a 'TS') and real
    vibrations."""

    def __init__(self, ref: StructureNode):
        masses = np.array([get_mass(s) for s in ref.symbols])
        B = vri.rigid_body_basis(ref.coords, masses)
        n = ref.coords.size
        rng = np.random.default_rng(0)
        V, _ = np.linalg.qr(np.column_stack([B, rng.normal(size=(n, n - B.shape[1]))]))
        vib = V[:, B.shape[1]:]
        lam = np.array([-1e-4, 0.05, 0.3])  # Eh/(bohr^2 amu), mass-weighted; barrier ~ 50i cm^-1
        sqrt_m = np.repeat(np.sqrt(masses), 3)
        self.H = (vib @ np.diag(lam) @ vib.T) * np.outer(sqrt_m, sqrt_m)
        self.x0 = ref.coords.reshape(-1)

    def compute_energies(self, nodes):
        out = []
        for n in nodes:
            d = n.coords.reshape(-1) - self.x0
            n._cached_energy = 0.5 * float(d @ self.H @ d)
            out.append(n._cached_energy)
        return out

    def compute_gradients(self, nodes):
        out = []
        for n in nodes:
            g = self.H @ (n.coords.reshape(-1) - self.x0)
            n._cached_gradient = g.reshape(n.coords.shape)
            out.append(n._cached_gradient)
        return out

    def compute_hessian(self, node, step_size=None):
        return self.H


def test_zero_temperature_sampling_puts_exactly_zero_point_energy_in_each_mode():
    ts = _water()
    engine = _QuadraticEngine(ts)
    modes = vri.stationary_point_modes(ts, engine)
    assert modes.n_imaginary(10.0) == 1
    init = qct.sample_initial_conditions(
        ts, modes, modes.modes_mw[:, 0], np.random.default_rng(1), temperature=0.0,
    )
    omegas = np.sqrt(modes.eigvals[1:] / qct.AMU_TO_ME)
    np.testing.assert_allclose(init.mode_energies, 0.5 * omegas)
    # Total energy = sum of ZPEs (harmonic surface, no reaction-mode energy at 0 K).
    masses = np.array([get_mass(s) for s in ts.symbols]) * qct.AMU_TO_ME
    kinetic = 0.5 * float(np.sum(masses[:, None] * init.velocities**2))
    potential = engine.compute_energies([ts.update_coords(init.coords)])[0]
    np.testing.assert_allclose(kinetic + potential, 0.5 * omegas.sum(), rtol=1e-6)


def test_velocity_verlet_conserves_energy_and_leaves_along_the_requested_side():
    ts = _water()
    engine = _QuadraticEngine(ts)
    modes = vri.stationary_point_modes(ts, engine)
    reaction = modes.modes_mw[:, 0]
    result = qct.run_qct(ts, modes, engine, reaction, n_trajectories=4, max_fs=100.0, workers=2, seed=3)
    masses = np.repeat(np.array([get_mass(s) for s in ts.symbols]), 3)
    for traj in result.trajectories:
        assert traj.error is None
        drift = abs(traj.energy_drift) * 627.5
        # Verlet energy error is bounded, ~(omega*dt)^2/8 of the mode energy: ~0.1
        # kcal/mol for a 12 fs vibration at dt = 1 fs, like C-H stretches.
        assert drift < 0.5, drift
        disp = (traj.final_node.coords - ts.coords).reshape(-1) * np.sqrt(masses)
        assert float(disp @ reaction) > 0  # rolled off the barrier toward +reaction mode


def test_trajectory_stops_once_committed_to_a_product():
    ts = _water()
    engine = _QuadraticEngine(ts)
    modes = vri.stationary_point_modes(ts, engine)
    reaction = modes.modes_mw[:, 0]
    calls = []

    def commit_to(x):  # everything past the barrier counts as "P1"
        calls.append(1)
        return "P1"

    result = qct.run_qct(ts, modes, engine, reaction, n_trajectories=2, max_fs=200.0, seed=1, commit_to=commit_to)
    for t in result.trajectories:
        if t.committed:
            assert t.committed == "P1" and t.commit_fs < 200.0
            assert len(t.frames) < 21  # stopped early


def test_bond_pattern_and_classifier():
    from mepd.discovery.qct import bond_pattern, pattern_classifier

    ts = _water()
    pat = bond_pattern(ts.symbols, ts.coords)
    assert pat == frozenset({(0, 1), (0, 2)})
    far = ts.update_coords(ts.coords * np.array([[1, 1, 1], [3, 3, 3], [1, 1, 1]]))
    classify = pattern_classifier(list(ts.symbols), {"P1": far}, ts.coords)
    assert classify(far.coords) == "P1" and classify(ts.coords) is None
