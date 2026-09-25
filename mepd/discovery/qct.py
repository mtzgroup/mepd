"""Quasiclassical trajectories from a transition state.

A dynamics probe for post-TS bifurcations the IRC scan cannot see: when the
VRI lies off the IRC, the projected frequencies along the IRC need not ever
turn imaginary, yet trajectories leaving TS1 still split between two
products. This samples initial conditions at TS1 the standard way
(Progdyn/Milo-style normal-mode sampling), integrates with velocity Verlet,
and relaxes each trajectory's final frame to a minimum for classification.

Initial conditions:
- every real mode gets a quantum level n drawn from its Boltzmann
  distribution at `temperature` (mostly n=0, i.e. zero-point energy),
  energy (n + 1/2) hbar*omega, with a random phase;
- the reaction mode (the TS imaginary mode) gets no displacement and a
  kinetic energy drawn from the Boltzmann distribution, pointing toward the
  requested branch.

Atomic units throughout: bohr, Hartree, electron masses, hbar = 1.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from mepd.discovery.vri import ProjectedModes, node_masses
from mepd.engines.engine import Engine
from mepd.nodes.node import Node

logger = logging.getLogger(__name__)

AMU_TO_ME = 1822.888486209
FS_TO_AU = 41.341373335
KB_HARTREE_PER_K = 3.166811563e-6


@dataclass
class InitialConditions:
    coords: np.ndarray       # bohr, node.coords shape
    velocities: np.ndarray   # bohr / au time
    mode_energies: np.ndarray  # Hartree per real mode (incl. ZPE)
    reaction_energy: float   # Hartree along the reaction mode


def sample_initial_conditions(
    ts_node: Node,
    ts_modes: ProjectedModes,
    direction_mw: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float = 298.15,
    imaginary_index: int = 0,
) -> InitialConditions:
    """Normal-mode sampling at a TS. `ts_modes` are the stationary-point
    modes at the TS (rigid-body motion projected out); `direction_mw` is
    any mass-weighted vector pointing toward the branch the trajectory
    should head into (only the sign of its overlap with the reaction mode
    is used)."""
    masses = node_masses(ts_node)
    if masses is None:
        raise ValueError("Quasiclassical sampling needs a molecular node with atomic masses.")
    m_e = np.repeat(masses * AMU_TO_ME, 3)
    sqrt_m = np.sqrt(m_e)
    shape = np.asarray(ts_node.coords).shape
    kT = KB_HARTREE_PER_K * float(temperature)

    Q = np.zeros(sqrt_m.size)   # mass-weighted displacement, bohr * m_e^1/2
    P = np.zeros(sqrt_m.size)   # mass-weighted velocity
    energies = []
    for i, lam in enumerate(ts_modes.eigvals):
        L = ts_modes.modes_mw[:, i]
        if i == imaginary_index:
            continue
        if lam <= 0:
            continue  # extra (spurious) imaginary modes get nothing
        omega = float(np.sqrt(lam / AMU_TO_ME))
        if kT > 0:
            # Boltzmann over levels: P(n) ~ exp(-n*omega/kT) -> geometric.
            p_excited = np.exp(-omega / kT)
            n = int(rng.geometric(1.0 - p_excited) - 1) if p_excited > 1e-12 else 0
        else:
            n = 0
        energy = (n + 0.5) * omega
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = np.sqrt(2.0 * energy) / omega
        Q += amplitude * np.cos(phase) * L
        P += -np.sqrt(2.0 * energy) * np.sin(phase) * L
        energies.append(energy)

    L_rc = ts_modes.modes_mw[:, imaginary_index]
    sign = 1.0 if float(L_rc @ np.asarray(direction_mw).reshape(-1)) >= 0 else -1.0
    e_rc = -kT * np.log(1.0 - rng.uniform()) if kT > 0 else 0.0
    P += sign * np.sqrt(2.0 * e_rc) * L_rc

    coords = np.asarray(ts_node.coords, dtype=float) + (Q / sqrt_m).reshape(shape)
    velocities = (P / sqrt_m).reshape(shape)
    return InitialConditions(coords, velocities, np.array(energies), float(e_rc))


@dataclass
class Trajectory:
    index: int
    frames: List[Node] = field(default_factory=list, repr=False)  # every `save_every` steps
    total_energies: List[float] = field(default_factory=list)
    potential_energies: List[float] = field(default_factory=list)                   # per saved frame
    frame_gradients: List[np.ndarray] = field(default_factory=list, repr=False)     # per saved frame
    final_node: Optional[Node] = field(default=None, repr=False)
    product: Optional[Node] = field(default=None, repr=False)     # optimized final frame
    label: Optional[str] = None                                   # set by the caller's classification
    error: Optional[str] = None
    reaction_coordinate: List[float] = field(default_factory=list)  # per saved frame, amu^1/2 bohr, + = toward branch
    recrossed: bool = False                                       # went back across the TS1 dividing surface
    committed: Optional[str] = None                               # bond pattern it committed to (see `commit_to`)
    commit_fs: Optional[float] = None                             # time of commitment

    @property
    def energy_drift(self) -> Optional[float]:
        if len(self.total_energies) < 2:
            return None
        return float(self.total_energies[-1] - self.total_energies[0])


def _gradient(engine: Engine, node: Node) -> tuple[float, np.ndarray]:
    grad = np.asarray(engine.compute_gradients([node])[0], dtype=float)
    energy = float(engine.compute_energies([node])[0])
    return energy, grad


def run_trajectory(
    engine: Engine,
    ts_node: Node,
    initial: InitialConditions,
    *,
    index: int = 0,
    dt_fs: float = 1.0,
    max_fs: float = 400.0,
    save_every: int = 10,
    reaction_mode_mw: Optional[np.ndarray] = None,
    recross_tolerance: float = 0.1,
    commit_to: Optional[Callable[[np.ndarray], Optional[str]]] = None,
    commit_checks: int = 3,
    commit_min_rc: float = 0.5,
) -> Trajectory:
    """Velocity Verlet from the sampled initial conditions.

    With `reaction_mode_mw` (unit, amu-weighted, oriented toward the
    branch), records the displacement from the TS along it at every saved
    frame and flags the trajectory as `recrossed` once it comes back more
    than `recross_tolerance` (amu^1/2 bohr) to the other side of TS1.

    With `commit_to` (coordinates in bohr -> a bond-pattern label, or None),
    the trajectory stops at its first product: once it is at least
    `commit_min_rc` (amu^1/2 bohr) past TS1 along the reaction mode and the
    label has stayed the same for `commit_checks` saved frames. Without it,
    a hot product keeps reacting until `max_fs` and the final frame no longer
    tells which way the trajectory went at the ridge."""
    masses = node_masses(ts_node)
    m = (masses * AMU_TO_ME)[:, None]
    dt = float(dt_fs) * FS_TO_AU
    n_steps = int(round(float(max_fs) / float(dt_fs)))
    traj = Trajectory(index=index)
    # Skip OpenBabel bond perception on every MD frame; products are
    # perceived once, after optimization.
    base = ts_node.copy()
    if type(base).__name__ == "StructureNode":
        base.has_molecular_graph = False
        base.graph = None
    ts_node = base
    x0 = np.asarray(ts_node.coords, dtype=float).reshape(-1)
    sqrt_amu = np.repeat(np.sqrt(masses), 3)
    x = initial.coords.copy()
    v = initial.velocities.copy()
    streak_label, streak = None, 0
    try:
        node = ts_node.update_coords(x)
        energy, g = _gradient(engine, node)
        for step in range(n_steps + 1):
            if step % save_every == 0 or step == n_steps:
                traj.frames.append(node)
                traj.total_energies.append(energy + 0.5 * float(np.sum(m * v * v)))
                traj.potential_energies.append(energy)
                traj.frame_gradients.append(np.asarray(g, dtype=float).reshape(-1).copy())
                if reaction_mode_mw is not None:
                    q = float(((x.reshape(-1) - x0) * sqrt_amu) @ reaction_mode_mw)
                    traj.reaction_coordinate.append(q)
                    if q < -abs(recross_tolerance):
                        traj.recrossed = True
                if commit_to is not None and step > 0:
                    past = (not traj.reaction_coordinate) or abs(traj.reaction_coordinate[-1]) >= commit_min_rc
                    label = commit_to(x) if past else None
                    if label is not None and label == streak_label:
                        streak += 1
                    else:
                        streak_label, streak = label, (1 if label is not None else 0)
                    if streak >= commit_checks:
                        traj.committed = label
                        traj.commit_fs = step * float(dt_fs)
                        break
            if step == n_steps:
                break
            a = -g / m
            x = x + v * dt + 0.5 * a * dt * dt
            node = ts_node.update_coords(x)
            energy, g_new = _gradient(engine, node)
            v = v + 0.5 * (a - g_new / m) * dt
            g = g_new
        traj.final_node = node
    except Exception as exc:
        traj.error = f"{type(exc).__name__}: {exc}"
    return traj


@dataclass
class QCTResult:
    trajectories: List[Trajectory]
    temperature: float
    dt_fs: float
    max_fs: float

    def counts(self) -> dict:
        out: dict = {}
        for t in self.trajectories:
            key = t.label or ("failed" if t.error or t.product is None else "unclassified")
            out[key] = out.get(key, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "n_trajectories": len(self.trajectories),
            "temperature": self.temperature,
            "dt_fs": self.dt_fs,
            "max_fs": self.max_fs,
            "counts": self.counts(),
            "trajectories": [
                {
                    "index": t.index,
                    "label": t.label,
                    "recrossed": t.recrossed,
                    "committed": t.committed,
                    "commit_fs": t.commit_fs,
                    "error": t.error,
                    "energy_drift_kcal_mol": None if t.energy_drift is None else t.energy_drift * 627.5095,
                    "product_energy": None if t.product is None else t.product._cached_energy,
                }
                for t in self.trajectories
            ],
        }


def run_qct(
    ts_node: Node,
    ts_modes: ProjectedModes,
    engine: Engine,
    direction_mw: np.ndarray,
    *,
    n_trajectories: int = 16,
    temperature: float = 298.15,
    dt_fs: float = 1.0,
    max_fs: float = 400.0,
    save_every: int = 10,
    workers: int = 1,
    seed: int = 0,
    opt_keywords: Optional[dict] = None,
    on_done: Optional[Callable[[int, int], None]] = None,
    commit_to: Optional[Callable[[np.ndarray], Optional[str]]] = None,
) -> QCTResult:
    """Run `n_trajectories` from TS1 toward `direction_mw`, `workers` at a
    time, then optimize every final frame (`Trajectory.product`).
    Classification against known species is left to the caller."""
    from mepd.discovery.vri import _optimize_nodes

    rng = np.random.default_rng(int(seed))
    L_rc = ts_modes.modes_mw[:, 0]
    if float(L_rc @ np.asarray(direction_mw).reshape(-1)) < 0:
        L_rc = -L_rc
    initials = [
        sample_initial_conditions(ts_node, ts_modes, direction_mw, rng, temperature=temperature)
        for _ in range(int(n_trajectories))
    ]
    done = [0]

    def _one(item):
        i, init = item
        traj = run_trajectory(
            engine, ts_node, init, index=i, dt_fs=dt_fs, max_fs=max_fs, save_every=save_every,
            reaction_mode_mw=L_rc, commit_to=commit_to,
        )
        done[0] += 1
        if on_done is not None:
            on_done(done[0], len(initials))
        return traj

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        trajectories = list(pool.map(_one, enumerate(initials)))

    finals = [t for t in trajectories if t.final_node is not None]
    products = _optimize_nodes(engine, [t.final_node for t in finals], opt_keywords, workers)
    for traj, product in zip(finals, products):
        traj.product = product
        if product is not None and product._cached_energy is None:
            product._cached_energy = float(engine.compute_energies([product])[0])
    return QCTResult(trajectories=trajectories, temperature=temperature, dt_fs=dt_fs, max_fs=max_fs)


_RADII = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07,
          "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39}


def bond_pattern(symbols, coords_bohr, scale: float = 1.25) -> frozenset:
    """Atom-indexed bonds by distance (< `scale` x sum of covalent radii):
    cheap enough to run on every saved MD frame."""
    X = np.asarray(coords_bohr, dtype=float).reshape(-1, 3) * 0.529177210903
    r = np.array([_RADII.get(s, 1.2) for s in symbols])
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    i, j = np.where(np.triu(d < scale * (r[:, None] + r[None]), 1))
    return frozenset(zip(i.tolist(), j.tolist()))


def pattern_classifier(symbols, references: dict, ts_coords) -> Callable[[np.ndarray], Optional[str]]:
    """`commit_to` for `run_trajectory`: a frame whose bond pattern matches a
    reference structure's (e.g. {"P1": ..., "P2": ..., "R": ...}) gets its
    name; any other pattern that differs from TS1's gets a label of its own,
    so a stable unknown product also counts as a commitment."""
    refs = {name: bond_pattern(symbols, np.asarray(node.coords)) for name, node in references.items()}
    ts_pattern = bond_pattern(symbols, ts_coords)

    def classify(x):
        pat = bond_pattern(symbols, x)
        for name, ref in refs.items():
            if pat == ref:
                return name
        if pat == ts_pattern:
            return None
        return "new:" + str(hash(pat))

    return classify
