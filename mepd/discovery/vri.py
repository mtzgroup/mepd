"""Valley-ridge inflection (VRI) search along an IRC.

The IRC is steepest descent, so it never splits at a VRI -- it can only
branch at a stationary point. What *is* visible along an IRC is the
valley-ridge transition (VRT): the point where the lowest vibrational
frequency orthogonal to the path (from the gradient-projected, mass-weighted
Hessian of Miller, Handy & Adams, JCP 72, 99 (1980)) turns imaginary. On a
symmetric surface the VRT is the VRI; on an asymmetric one the VRI lies off
the IRC nearby.

Pipeline (see `docs`/the VRI writeup for background):

1. `scan_irc_for_vrt`: Hessian at points along each IRC branch, projected
   frequencies, bisect the sign change of the lowest one -> VRT.
2. `find_bifurcation_products`: optimize the IRC endpoint. If it is a saddle
   the IRC ran down a symmetric ridge onto TS2, and pushing along its
   imaginary mode gives P1/P2. Otherwise push along the ridge mode at
   points past the VRT to find a second product P2.
3. `verify_ts2`: TS2 (from step 2, or found by the caller via NEB between
   P1 and P2) must have one imaginary mode, lie below TS1, and have an IRC
   connecting P1 and P2.
4. `branch_verdict` / `overall_verdict`.

Products of a bifurcation are often symmetry-degenerate (e.g. the two
[4+2] adducts of cyclopentadiene dimerization), so species identity here is
atom-indexed bond sets, not graph isomorphism.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from mepd.engines.engine import Engine, _HESSIAN_EIGENVALUE_TO_CM2
from mepd.helper_functions import get_mass
from mepd.nodes.node import Node

logger = logging.getLogger(__name__)

OnEvent = Optional[Callable[[str, dict], None]]


def _emit(on_event: OnEvent, event: str, **payload) -> None:
    if on_event is not None:
        on_event(event, payload)


# --------------------------------------------------------------------------
# Projected Hessian
# --------------------------------------------------------------------------


def _is_molecular(node: Node) -> bool:
    coords = np.asarray(node.coords)
    return coords.ndim == 2 and coords.shape[1] == 3 and hasattr(node, "symbols")


def node_masses(node: Node) -> Optional[np.ndarray]:
    """Atomic masses (amu) for a molecular node; None for toy potentials,
    which are treated as unit-mass with no rigid-body motion."""
    if not _is_molecular(node):
        return None
    return np.array([get_mass(s) for s in node.symbols], dtype=float)


def rigid_body_basis(coords: np.ndarray, masses: Optional[np.ndarray]) -> np.ndarray:
    """Orthonormal mass-weighted translation/rotation vectors, shape
    (3N, k): k=6 in general, 5 for a linear molecule, 3 for one atom.
    Linear geometries fall out of the SVD rank cut automatically.

    Returns an empty (ndof, 0) basis when `masses` is None (a toy potential
    whose coordinates are not Cartesian atoms)."""
    coords = np.asarray(coords, dtype=float)
    ndof = coords.size
    if masses is None:
        return np.zeros((ndof, 0))
    coords = coords.reshape(-1, 3)
    masses = np.asarray(masses, dtype=float)
    sqrt_m = np.sqrt(masses)
    com = np.average(coords, axis=0, weights=masses)
    r = coords - com
    vectors = []
    for axis in np.eye(3):
        vectors.append((sqrt_m[:, None] * axis[None, :]).reshape(-1))
    for axis in np.eye(3):
        vectors.append((sqrt_m[:, None] * np.cross(axis[None, :], r)).reshape(-1))
    mat = np.array(vectors).T
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    keep = s > 1e-6 * s.max()
    return u[:, keep]


@dataclass
class ProjectedModes:
    """Frequencies orthogonal to the reaction path (and rigid-body motion).

    `freqs` are signed (negative == imaginary). For molecular nodes they are
    in cm^-1; for toy potentials (unit masses, no conversion) they are
    sign(lambda) * sqrt(|lambda|) in the potential's own units."""

    eigvals: np.ndarray            # mass-weighted projected eigenvalues, ascending
    freqs: np.ndarray              # signed frequencies, ascending
    modes_mw: np.ndarray           # (ndof, n) mass-weighted unit eigenvectors
    modes_cart: List[np.ndarray]   # Cartesian displacement vectors, node.coords shape, unit norm
    grad_norm: float
    tangent_source: str            # "gradient" | "supplied" | "none"
    tangent_mw: Optional[np.ndarray] = None

    @property
    def lowest_freq(self) -> float:
        return float(self.freqs[0])

    @property
    def lowest_mode_cart(self) -> np.ndarray:
        return self.modes_cart[0]

    def n_imaginary(self, cutoff: float) -> int:
        return int(np.sum(self.freqs < -abs(cutoff)))


def projected_frequencies(
    hessian: np.ndarray,
    coords: np.ndarray,
    masses: Optional[np.ndarray] = None,
    gradient: Optional[np.ndarray] = None,
    *,
    tangent_mw: Optional[np.ndarray] = None,
    grad_floor: float = 1e-4,
) -> ProjectedModes:
    """Diagonalize K = (1 - P) H~ (1 - P), with H~ the mass-weighted Hessian
    and P the projector onto rigid-body motion plus the path tangent.

    The tangent is the mass-weighted gradient direction when |g| >=
    `grad_floor` (Eh/bohr). Below that -- near a stationary point, where the
    gradient direction is noise -- `tangent_mw` is used if given (e.g. the
    TS imaginary mode, or a finite-difference chain tangent), otherwise only
    rigid-body motion is projected out (ordinary frequencies at a
    stationary point).

    The projected-out directions are dropped by overlap with P, not by
    smallest |lambda|, so a genuinely soft vibration is never discarded.
    """
    coords = np.asarray(coords, dtype=float)
    shape = coords.shape
    ndof = coords.size
    H = np.asarray(hessian, dtype=float).reshape(ndof, ndof)
    H = 0.5 * (H + H.T)

    if masses is None:
        sqrt_m = np.ones(ndof)
        conversion = 1.0
    else:
        sqrt_m = np.repeat(np.sqrt(np.asarray(masses, dtype=float)), 3)
        conversion = _HESSIAN_EIGENVALUE_TO_CM2
    H_mw = H / np.outer(sqrt_m, sqrt_m)

    basis = rigid_body_basis(coords, masses)

    grad_norm = 0.0
    tangent = None
    tangent_source = "none"
    if gradient is not None:
        g = np.asarray(gradient, dtype=float).reshape(-1)
        grad_norm = float(np.linalg.norm(g))
        if grad_norm >= grad_floor:
            tangent = g / sqrt_m
            tangent_source = "gradient"
    if tangent is None and tangent_mw is not None:
        tangent = np.asarray(tangent_mw, dtype=float).reshape(-1)
        tangent_source = "supplied"
    if tangent is not None:
        tangent = tangent - basis @ (basis.T @ tangent)
        norm = np.linalg.norm(tangent)
        if norm < 1e-12:
            tangent = None
            tangent_source = "none"
        else:
            tangent = tangent / norm

    if tangent is not None:
        Q = np.column_stack([basis, tangent])
    else:
        Q = basis
    k = Q.shape[1]

    if k:
        P_perp = np.eye(ndof) - Q @ Q.T
        K = P_perp @ H_mw @ P_perp
    else:
        K = H_mw
    eigvals, eigvecs = np.linalg.eigh(K)

    if k:
        overlap = np.sum((Q.T @ eigvecs) ** 2, axis=0)
        drop = set(np.argsort(overlap)[-k:].tolist())
        keep = [i for i in range(ndof) if i not in drop]
        eigvals = eigvals[keep]
        eigvecs = eigvecs[:, keep]

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    freqs = np.sign(eigvals) * np.sqrt(np.abs(eigvals) * conversion)

    modes_cart = []
    for i in range(eigvecs.shape[1]):
        cart = eigvecs[:, i] / sqrt_m
        cart = cart / np.linalg.norm(cart)
        modes_cart.append(cart.reshape(shape))

    return ProjectedModes(
        eigvals=eigvals,
        freqs=freqs,
        modes_mw=eigvecs,
        modes_cart=modes_cart,
        grad_norm=grad_norm,
        tangent_source=tangent_source,
        tangent_mw=tangent,
    )


class GradientHessianEngine:
    """Wraps an engine so `compute_hessian` is central differences of the
    engine's own gradients (6N gradient calls); everything else delegates.

    Why: g-xTB's `--hess` matrix is not consistent with its own gradients
    (differences up to ~0.02 Eh/bohr^2, even at stationary points), and at
    non-stationary IRC points that flips the sign of soft projected
    frequencies. A Hessian built from the gradients agrees with energy
    finite differences to ~1e-5, i.e. it describes the surface the IRC and
    the optimizations actually run on.

    At most `max_concurrent` gradient calls run at once across all threads
    (a shared semaphore), so parallel Hessians never oversubscribe cores.
    """

    def __init__(self, engine, *, step: float = 0.005, max_concurrent: int = 1):
        import threading

        self.engine = engine
        self.step = float(step)
        self._slots = threading.Semaphore(max(1, int(max_concurrent)))
        self._max = max(1, int(max_concurrent))

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def _gradient(self, node) -> np.ndarray:
        with self._slots:
            return np.asarray(self.engine.compute_gradients([node])[0], dtype=float).reshape(-1)

    def compute_hessian(self, node, step_size=None) -> np.ndarray:
        h = float(step_size or self.step)
        base = node.copy()
        if type(base).__name__ == "StructureNode":
            # skip OpenBabel perception on each of the 6N displaced copies
            base.has_molecular_graph = False
            base.graph = None
        x = np.asarray(base.coords, dtype=float)
        shape, flat = x.shape, x.reshape(-1)

        def column(i):
            e = np.zeros_like(flat)
            e[i] = h
            gp = self._gradient(base.update_coords((flat + e).reshape(shape)))
            gm = self._gradient(base.update_coords((flat - e).reshape(shape)))
            return (gp - gm) / (2.0 * h)

        with ThreadPoolExecutor(max_workers=self._max) as pool:
            H = np.array(list(pool.map(column, range(flat.size))))
        return 0.5 * (H + H.T)


_warned_fd_engines: set = set()


def compute_cartesian_hessian(node: Node, engine: Engine) -> np.ndarray:
    """Raw Cartesian Hessian (Eh/bohr^2) from the engine.

    Deliberately not `_compute_hessian_result`: engines like g-xTB return
    their own projected frequencies from it, which are only meaningful at
    stationary points."""
    compute = getattr(engine, "compute_hessian", None)
    if not callable(compute):
        raise ValueError(f"Engine {type(engine).__name__} cannot compute Hessians.")
    if (
        getattr(type(engine), "compute_hessian", None) is Engine.compute_hessian
        and type(engine) not in _warned_fd_engines
    ):
        _warned_fd_engines.add(type(engine))
        warnings.warn(
            f"{type(engine).__name__} has no analytic or gradient-based Hessian; the "
            "VRI scan will use energy finite differences ((3N)^2 energies per point).",
            RuntimeWarning,
            stacklevel=2,
        )
    hessian = np.asarray(compute(node), dtype=float)
    ndof = np.asarray(node.coords).size
    return hessian.reshape(ndof, ndof)


def compute_hessians(
    nodes: Sequence[Node],
    engine: Engine,
    workers: int = 1,
    on_done: Optional[Callable[[int], None]] = None,
) -> List[np.ndarray]:
    """Cartesian Hessians for `nodes`, in order, `workers` at a time.

    Threads suffice: engines like g-xTB run each Hessian as its own
    subprocess in its own temp directory. Keep each engine call
    single-threaded and parallelize across points instead -- for g-xTB that
    is ~10x faster than one multi-threaded call at a time."""
    nodes = list(nodes)
    if int(workers) <= 1 or len(nodes) <= 1:
        out = []
        for i, node in enumerate(nodes, start=1):
            out.append(compute_cartesian_hessian(node, engine))
            if on_done is not None:
                on_done(i)
        return out
    out: List[Optional[np.ndarray]] = [None] * len(nodes)
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(compute_cartesian_hessian, node, engine): i for i, node in enumerate(nodes)}
        done = 0
        from concurrent.futures import as_completed

        for future in as_completed(futures):
            out[futures[future]] = future.result()
            done += 1
            if on_done is not None:
                on_done(done)
    return out


def _ensure_energy_gradient(nodes: Sequence[Node], engine: Engine) -> None:
    missing = [n for n in nodes if n._cached_gradient is None or n._cached_energy is None]
    if not missing:
        return
    grads = engine.compute_gradients(missing)
    energies = engine.compute_energies(missing)
    for node, g, e in zip(missing, grads, energies):
        if node._cached_gradient is None:
            node._cached_gradient = np.asarray(g, dtype=float)
        if node._cached_energy is None:
            node._cached_energy = float(e)


def _ensure_energy(nodes: Sequence[Node], engine: Engine) -> None:
    missing = [n for n in nodes if n._cached_energy is None]
    if not missing:
        return
    energies = engine.compute_energies(missing)
    for node, e in zip(missing, energies):
        if node._cached_energy is None:
            node._cached_energy = float(e)


def _mass_weighted(node: Node) -> np.ndarray:
    masses = node_masses(node)
    coords = np.asarray(node.coords, dtype=float).reshape(-1)
    if masses is None:
        return coords
    return coords * np.repeat(np.sqrt(masses), 3)


def stationary_point_modes(node: Node, engine: Engine) -> ProjectedModes:
    """Ordinary frequencies (rigid-body motion projected out, no path) at a
    stationary point."""
    hessian = compute_cartesian_hessian(node, engine)
    return projected_frequencies(hessian, node.coords, node_masses(node), gradient=None)


# --------------------------------------------------------------------------
# VRT scan
# --------------------------------------------------------------------------


@dataclass
class ProjectedPoint:
    s: float                      # mass-weighted arc length from TS1 (amu^1/2 bohr; bohr for toys)
    energy: float
    grad_norm: float
    lowest_freqs: List[float]     # up to 5 lowest projected frequencies
    lowest_eigval: float
    soft_overlap: Optional[float]  # |<soft mode here | soft mode at previous point>|
    tangent_source: str
    chain_index: Optional[int]    # index within the branch; None for a bisection point
    soft_mode_cart: Optional[np.ndarray] = field(default=None, repr=False)
    node: Optional[Node] = field(default=None, repr=False)
    second_mode_cart: Optional[np.ndarray] = field(default=None, repr=False)  # next-softest perpendicular mode

    @property
    def lowest_freq(self) -> float:
        return float(self.lowest_freqs[0])

    def to_dict(self) -> dict:
        return {
            "s": self.s,
            "energy": self.energy,
            "grad_norm": self.grad_norm,
            "lowest_freqs": self.lowest_freqs,
            "lowest_eigval": self.lowest_eigval,
            "soft_overlap": self.soft_overlap,
            "tangent_source": self.tangent_source,
            "chain_index": self.chain_index,
        }


@dataclass
class VRT:
    s: float
    node: Node
    ridge_mode_cart: np.ndarray
    freq_before: float
    freq_after: float
    bracket_chain_indices: tuple  # (last sampled index with real freq, first with imaginary)

    def to_dict(self) -> dict:
        return {
            "s": self.s,
            "energy": float(self.node._cached_energy) if self.node._cached_energy is not None else None,
            "freq_before": self.freq_before,
            "freq_after": self.freq_after,
            "bracket_chain_indices": list(self.bracket_chain_indices),
        }


@dataclass
class BranchScan:
    name: str                       # "forward" | "reverse"
    nodes: List[Node]               # TS-outward; nodes[0] is TS1
    s: np.ndarray
    points: List[ProjectedPoint] = field(default_factory=list)
    bisection_points: List[ProjectedPoint] = field(default_factory=list)
    vrt: Optional[VRT] = None
    transient_dips: List[float] = field(default_factory=list)  # s of noise-level dips
    valley_reforms: bool = False    # soft mode turned real again after the VRT
    late_dips: List[float] = field(default_factory=list)  # s of dips deeper than max_vrt_depth below TS1
    # s of points where the two softest perpendicular modes are both imaginary:
    # a possible higher-order VRI (the ridge may open into more than two valleys)
    double_ridge: List[float] = field(default_factory=list)
    # Raw Cartesian Hessians (Eh/bohr^2) per scanned point / bisection point,
    # parallel to `points` / `bisection_points`; see `save_hessians`.
    hessians: List[np.ndarray] = field(default_factory=list, repr=False)
    bisection_hessians: List[np.ndarray] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_nodes": len(self.nodes),
            "path_length": float(self.s[-1]) if len(self.s) else 0.0,
            "points": [p.to_dict() for p in self.points],
            "bisection_points": [p.to_dict() for p in self.bisection_points],
            "vrt": self.vrt.to_dict() if self.vrt else None,
            "transient_dips_s": self.transient_dips,
            "valley_reforms": self.valley_reforms,
            "late_dips_s": self.late_dips,
            "double_ridge_s": self.double_ridge,
        }


@dataclass
class VRTScan:
    ts_index: int
    ts_modes: ProjectedModes
    branches: Dict[str, BranchScan]
    n_hessians: int
    freq_units: str

    def to_dict(self) -> dict:
        return {
            "ts_index": self.ts_index,
            "ts_freqs_lowest": [float(f) for f in self.ts_modes.freqs[:6]],
            "n_hessians": self.n_hessians,
            "freq_units": self.freq_units,
            "branches": {k: b.to_dict() for k, b in self.branches.items()},
        }


def split_irc_branches(irc_nodes: Sequence[Node], ts_index: int) -> Dict[str, List[Node]]:
    """Split an IRC chain ordered reverse-end -> TS -> forward-end into two
    TS-outward branches, each starting at the TS. Consecutive
    (near-)duplicate frames are dropped."""
    def dedupe(nodes: List[Node]) -> List[Node]:
        out = [nodes[0]]
        for node in nodes[1:]:
            if np.linalg.norm(_mass_weighted(node) - _mass_weighted(out[-1])) > 1e-6:
                out.append(node)
        return out

    forward = dedupe(list(irc_nodes[ts_index:]))
    reverse = dedupe(list(irc_nodes[: ts_index + 1])[::-1])
    return {"forward": forward, "reverse": reverse}


def _arc_length(nodes: Sequence[Node]) -> np.ndarray:
    q = np.array([_mass_weighted(n) for n in nodes])
    if len(q) < 2:
        return np.zeros(len(q))
    steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _detect_vrt(freqs: Sequence[float], threshold: float, persist: int):
    """Scan a TS-outward sequence of lowest projected frequencies.

    Returns (first_index, transient_indices, reforms):
    - first_index: first point whose frequency is below -threshold and stays
      there for `persist` consecutive points (or until the branch ends), or
      None.
    - transient_indices: starts of dips below -threshold that did not
      persist (numerical noise).
    - reforms: the frequency rose back above +threshold after the VRT.
    """
    freqs = list(freqs)
    n = len(freqs)
    persist = max(1, int(persist))
    transients: List[int] = []
    i = 0
    while i < n:
        if freqs[i] < -threshold:
            run_end = i
            while run_end + 1 < n and freqs[run_end + 1] < -threshold:
                run_end += 1
            run_len = run_end - i + 1
            if run_len >= persist or run_end == n - 1:
                reforms = any(f > threshold for f in freqs[run_end + 1:])
                return i, transients, reforms
            transients.append(i)
            i = run_end + 1
        else:
            i += 1
    return None, transients, False


def _chain_tangent_mw(nodes: Sequence[Node], k: int) -> Optional[np.ndarray]:
    if len(nodes) < 2:
        return None
    lo, hi = max(k - 1, 0), min(k + 1, len(nodes) - 1)
    t = _mass_weighted(nodes[hi]) - _mass_weighted(nodes[lo])
    norm = np.linalg.norm(t)
    return None if norm < 1e-12 else t / norm


def _interpolate_node(a: Node, b: Node, tau: float) -> Node:
    coords = (1.0 - tau) * np.asarray(a.coords, dtype=float) + tau * np.asarray(b.coords, dtype=float)
    return a.update_coords(coords)


def scan_irc_for_vrt(
    irc_nodes: Sequence[Node],
    engine: Engine,
    *,
    ts_index: Optional[int] = None,
    branches: Sequence[str] = ("forward", "reverse"),
    stride: int = 1,
    n_bisect: int = 4,
    vrt_threshold: float = 20.0,
    persist: int = 2,
    grad_floor: float = 1e-4,
    max_vrt_depth: Optional[float] = None,
    workers: int = 1,
    on_event: OnEvent = None,
) -> VRTScan:
    """Projected-frequency scan along each IRC branch; locate the VRT.

    `irc_nodes` is ordered reverse-end -> TS -> forward-end (what every mepd
    IRC backend returns). `ts_index` defaults to the energy maximum.
    `vrt_threshold` is in the frequency units of `ProjectedModes` (cm^-1
    for molecules). The branch endpoint is not scanned (the projection is
    meaningless at a minimum); it is handled by `find_bifurcation_products`.

    `max_vrt_depth` (kcal/mol): imaginary dips more than this far below
    TS1 are recorded as `late_dips`, not as a VRT. Deep in a product valley
    a soft torsion going slightly imaginary is not a bifurcation of the
    TS1 reaction (the VRI of a PTSB lies above TS2, which is below TS1 but
    typically not far below). None disables the window.

    `workers` Hessians run concurrently. Bisection then becomes a k-section:
    each of the `n_bisect` rounds evaluates `workers` interior points at once
    (plain bisection for workers=1).
    """
    irc_nodes = list(irc_nodes)
    workers = max(1, int(workers))
    if len(irc_nodes) < 3:
        raise ValueError("The IRC needs at least 3 points to scan.")
    stride = max(1, int(stride))
    n_hessians = 0

    if ts_index is None:
        _ensure_energy(irc_nodes, engine)
        ts_index = int(np.argmax([float(n._cached_energy) for n in irc_nodes]))
    ts_node = irc_nodes[ts_index]
    _ensure_energy_gradient([ts_node], engine)
    masses = node_masses(ts_node)
    freq_units = "cm^-1" if masses is not None else "arb"

    _emit(on_event, "ts_hessian")
    ts_hessian = compute_cartesian_hessian(ts_node, engine)
    n_hessians += 1
    ts_modes = projected_frequencies(ts_hessian, ts_node.coords, masses, gradient=None)
    ts_imag_mw = ts_modes.modes_mw[:, 0]

    split = split_irc_branches(irc_nodes, ts_index)
    result = VRTScan(
        ts_index=ts_index, ts_modes=ts_modes, branches={}, n_hessians=0, freq_units=freq_units,
    )

    for name in branches:
        nodes = split[name]
        branch = BranchScan(name=name, nodes=nodes, s=_arc_length(nodes))
        result.branches[name] = branch
        if len(nodes) < 3:
            continue
        # Orient the TS imaginary mode along this branch.
        step = _mass_weighted(nodes[1]) - _mass_weighted(nodes[0])
        ts_tangent = ts_imag_mw if float(ts_imag_mw @ step) >= 0 else -ts_imag_mw

        indices = list(range(0, len(nodes) - 1, stride))
        _ensure_energy_gradient([nodes[k] for k in indices], engine)
        _emit(on_event, "branch_start", branch=name, total=len(indices))

        todo = [k for k in indices if k != 0]
        n_total = len(indices)
        offset = n_total - len(todo)
        hessians = dict(zip(todo, compute_hessians(
            [nodes[k] for k in todo], engine, workers,
            on_done=lambda i: _emit(on_event, "point_done", branch=name, index=i + offset, total=n_total),
        )))
        n_hessians += len(todo)

        prev_mode = None
        for k in indices:
            node = nodes[k]
            if k == 0:
                hessian = ts_hessian
                modes = projected_frequencies(
                    hessian, node.coords, masses, gradient=None, tangent_mw=ts_tangent,
                )
            else:
                hessian = hessians[k]
                modes = projected_frequencies(
                    hessian, node.coords, masses, gradient=node._cached_gradient,
                    tangent_mw=_chain_tangent_mw(nodes, k), grad_floor=grad_floor,
                )
            soft = modes.modes_mw[:, 0]
            overlap = None if prev_mode is None else float(abs(soft @ prev_mode))
            prev_mode = soft
            branch.hessians.append(np.asarray(hessian, dtype=np.float32))
            branch.points.append(ProjectedPoint(
                s=float(branch.s[k]),
                energy=float(node._cached_energy),
                grad_norm=modes.grad_norm,
                lowest_freqs=[float(f) for f in modes.freqs[:5]],
                lowest_eigval=float(modes.eigvals[0]),
                soft_overlap=overlap,
                tangent_source=modes.tangent_source,
                chain_index=k,
                soft_mode_cart=modes.lowest_mode_cart,
                node=node,
                second_mode_cart=modes.modes_cart[1] if len(modes.modes_cart) > 1 else None,
            ))

        freqs = [p.lowest_freq for p in branch.points]
        if max_vrt_depth is not None:
            from qcconst.constants import HARTREE_TO_KCAL_PER_MOL

            e_ts = float(ts_node._cached_energy)
            too_deep = [
                (e_ts - p.energy) * float(HARTREE_TO_KCAL_PER_MOL) > float(max_vrt_depth)
                for p in branch.points
            ]
            branch.late_dips = [
                p.s for p, deep, f in zip(branch.points, too_deep, freqs) if deep and f < -vrt_threshold
            ]
            freqs = [np.inf if deep else f for f, deep in zip(freqs, too_deep)]
        first, transients, reforms = _detect_vrt(freqs, vrt_threshold, persist)
        branch.double_ridge = [
            p.s for p, f in zip(branch.points, freqs)
            if np.isfinite(f) and len(p.lowest_freqs) > 1 and p.lowest_freqs[1] < -vrt_threshold
        ]
        branch.transient_dips = [branch.points[i].s for i in transients]
        branch.valley_reforms = reforms
        if first is None:
            continue

        # Bracket: last sampled point at or before `first` with a real
        # lowest frequency, and the point right after it.
        lo = first - 1
        while lo > 0 and branch.points[lo].lowest_eigval < 0:
            lo -= 1
        lo = max(lo, 0)
        hi = lo + 1
        a, b = branch.points[lo], branch.points[hi]
        _emit(on_event, "bisecting", branch=name, total=n_bisect)
        n_interior = min(workers, 7)
        for it in range(int(n_bisect)):
            taus = [(j + 1) / (n_interior + 1) for j in range(n_interior)]
            probe_nodes = [_interpolate_node(a.node, b.node, t) for t in taus]
            _ensure_energy_gradient(probe_nodes, engine)
            probe_hessians = compute_hessians(probe_nodes, engine, workers)
            n_hessians += len(probe_nodes)
            tangent = _mass_weighted(b.node) - _mass_weighted(a.node)
            probes = []
            for t, probe_node, hessian in zip(taus, probe_nodes, probe_hessians):
                modes = projected_frequencies(
                    hessian, probe_node.coords, masses, gradient=probe_node._cached_gradient,
                    tangent_mw=tangent, grad_floor=grad_floor,
                )
                probes.append(ProjectedPoint(
                    s=(1 - t) * a.s + t * b.s,
                    energy=float(probe_node._cached_energy),
                    grad_norm=modes.grad_norm,
                    lowest_freqs=[float(f) for f in modes.freqs[:5]],
                    lowest_eigval=float(modes.eigvals[0]),
                    soft_overlap=float(abs(modes.modes_mw[:, 0] @ _to_mw(b.soft_mode_cart, masses))),
                    tangent_source=modes.tangent_source,
                    chain_index=None,
                    soft_mode_cart=modes.lowest_mode_cart,
                    node=probe_node,
                ))
            branch.bisection_points.extend(probes)
            branch.bisection_hessians.extend(np.asarray(h, dtype=np.float32) for h in probe_hessians)
            # New bracket: last real probe before the first imaginary one.
            sequence = [a] + probes + [b]
            j = next(i for i, p_ in enumerate(sequence) if p_.lowest_eigval < 0)
            a, b = sequence[j - 1], sequence[j]
            _emit(on_event, "bisect_done", branch=name, index=it + 1, total=n_bisect)

        # Linear root of the eigenvalue (smooth through zero, unlike the
        # signed frequency) inside the final bracket.
        la, lb = a.lowest_eigval, b.lowest_eigval
        tau = 0.5 if la == lb else float(np.clip(la / (la - lb), 0.0, 1.0))
        vrt_node = _interpolate_node(a.node, b.node, tau)
        _ensure_energy_gradient([vrt_node], engine)
        branch.vrt = VRT(
            s=(1 - tau) * a.s + tau * b.s,
            node=vrt_node,
            ridge_mode_cart=b.soft_mode_cart,
            freq_before=a.lowest_freq,
            freq_after=b.lowest_freq,
            bracket_chain_indices=(branch.points[lo].chain_index, branch.points[hi].chain_index),
        )
        _emit(on_event, "vrt_found", branch=name, s=branch.vrt.s)

    result.n_hessians = n_hessians
    return result


def _to_mw(mode_cart: np.ndarray, masses: Optional[np.ndarray]) -> np.ndarray:
    v = np.asarray(mode_cart, dtype=float).reshape(-1)
    if masses is not None:
        v = v * np.repeat(np.sqrt(masses), 3)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------
# Species identity
# --------------------------------------------------------------------------


def labeled_bond_set(node: Node) -> Optional[frozenset]:
    """Atom-indexed bond set of a molecular node (from the same OpenBabel
    perception the node graphs use); None for toy nodes."""
    if not _is_molecular(node):
        return None
    graph = getattr(node, "graph", None)
    if graph is None:
        from mepd.qcdata_structure_helpers import structure_to_molecule

        graph = structure_to_molecule(node.structure)
    return frozenset(tuple(sorted((int(i), int(j)))) for i, j in graph.edges())


def _heavy_signature(node: Node, bonds: frozenset) -> tuple:
    """Atom-indexed heavy-atom bonds plus the number of H on each heavy
    atom: hydrogens are interchangeable (which of two CH2 hydrogens moved
    does not make a different product), heavy atoms are not (the two
    atom-distinct adducts of a degenerate bifurcation stay distinct)."""
    symbols = list(node.symbols)
    heavy = frozenset(b for b in bonds if symbols[b[0]] != "H" and symbols[b[1]] != "H")
    h_count: dict = {}
    for i, j in bonds:
        if symbols[i] == "H" and symbols[j] != "H":
            h_count[j] = h_count.get(j, 0) + 1
        elif symbols[j] == "H" and symbols[i] != "H":
            h_count[i] = h_count.get(i, 0) + 1
    return heavy, tuple(sorted(h_count.items()))


def same_species(a: Node, b: Node, *, xy_tol: float = 0.05) -> bool:
    """Same atom-indexed heavy-atom connectivity and H count per heavy atom
    (molecules), or within `xy_tol` of each other (toy potentials)."""
    bonds_a, bonds_b = labeled_bond_set(a), labeled_bond_set(b)
    if bonds_a is None or bonds_b is None:
        diff = np.asarray(a.coords, dtype=float) - np.asarray(b.coords, dtype=float)
        return float(np.linalg.norm(diff)) < xy_tol
    return _heavy_signature(a, bonds_a) == _heavy_signature(b, bonds_b)


def products_isomorphic(a: Node, b: Node) -> Optional[bool]:
    """Whether two atom-distinct products are the same molecule up to atom
    relabelling (a degenerate bifurcation). None for toy nodes."""
    if not _is_molecular(a) or not _is_molecular(b):
        return None
    from mepd.nodes.nodehelpers import _is_connectivity_identical
    from mepd.qcdata_structure_helpers import structure_to_molecule

    a2, b2 = a.copy(), b.copy()
    if getattr(a2, "graph", None) is None:
        a2.graph = structure_to_molecule(a2.structure)
    if getattr(b2, "graph", None) is None:
        b2.graph = structure_to_molecule(b2.structure)
    try:
        return bool(_is_connectivity_identical(a2, b2, verbose=False, collect_comparison=False))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Products and TS2
# --------------------------------------------------------------------------


def _optimize_one(engine: Engine, node: Node, keywords: Optional[dict]) -> Optional[Node]:
    try:
        try:
            trajectory = engine.compute_geometry_optimization(node, keywords=keywords)
        except TypeError:
            trajectory = engine.compute_geometry_optimization(node)
        return trajectory[-1] if trajectory else None
    except Exception as exc:
        logger.info("Optimization failed: %s", exc)
        return None


def _optimize_nodes(
    engine: Engine, nodes: List[Node], keywords: Optional[dict], workers: int = 1,
) -> List[Optional[Node]]:
    """Optimize each node; None where an optimization failed. With
    workers > 1, single optimizations run concurrently; otherwise the batch
    call is used when the engine has one, falling back to one-by-one if it
    raises (the same pattern as `run_hessian_sample`)."""
    if not nodes:
        return []
    if int(workers) > 1 and len(nodes) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            return list(pool.map(lambda n: _optimize_one(engine, n, keywords), nodes))
    batch = getattr(engine, "compute_geometry_optimizations", None)
    if callable(batch):
        try:
            try:
                trajectories = batch(nodes, keywords=keywords)
            except TypeError:
                trajectories = batch(nodes)
            if len(trajectories) == len(nodes):
                return [t[-1] if t else None for t in trajectories]
        except Exception as exc:
            logger.info("Batch optimization failed (%s); optimizing one by one.", exc)
    return [_optimize_one(engine, node, keywords) for node in nodes]


def push_along_mode(node: Node, mode_cart: np.ndarray, amplitude: float) -> List[Node]:
    """Two copies of `node` displaced +/- along `mode_cart`, scaled so the
    most-displaced atom moves `amplitude` bohr (for toy potentials: the
    whole vector has norm `amplitude`)."""
    mode = np.asarray(mode_cart, dtype=float).reshape(np.asarray(node.coords).shape)
    if mode.ndim == 2:
        scale = float(np.max(np.linalg.norm(mode, axis=1)))
    else:
        scale = float(np.linalg.norm(mode))
    step = mode / scale * float(amplitude)
    coords = np.asarray(node.coords, dtype=float)
    return [node.update_coords(coords + step), node.update_coords(coords - step)]


@dataclass
class BranchProducts:
    branch: str
    endpoint_optimized: Optional[Node] = None
    endpoint_n_imaginary: Optional[int] = None
    p1: Optional[Node] = None
    p2: Optional[Node] = None
    ts2: Optional[Node] = None
    ts2_source: Optional[str] = None       # "irc_endpoint" | "neb" | None
    ts2_verified: Optional[bool] = None
    ts2_checks: dict = field(default_factory=dict)
    degenerate: Optional[bool] = None      # P1 and P2 isomorphic
    evidence: Optional[str] = None         # how P2 was found: "vrt_push" | "ts2_endpoint"
    checks: Optional[dict] = None          # basin test / exact VRI / trajectories (vri_checks.check_branch)
    push_attempts: int = 0
    push_minima: List[Node] = field(default_factory=list)
    push_outcomes: List[dict] = field(default_factory=list)  # every optimized push vs P1
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def energy(n):
            return None if n is None or n._cached_energy is None else float(n._cached_energy)
        return {
            "branch": self.branch,
            "endpoint_n_imaginary": self.endpoint_n_imaginary,
            "p1_energy": energy(self.p1),
            "p2_energy": energy(self.p2),
            "ts2_energy": energy(self.ts2),
            "ts2_source": self.ts2_source,
            "ts2_verified": self.ts2_verified,
            "ts2_checks": self.ts2_checks,
            "degenerate": self.degenerate,
            "evidence": self.evidence,
            "checks": self.checks,
            "push_attempts": self.push_attempts,
            "n_push_minima": len(self.push_minima),
            "push_outcomes": self.push_outcomes,
            "stereo_second_product": any(o.get("same_bonds") and o.get("stereo_differs") for o in self.push_outcomes),
            "notes": self.notes,
        }


def _stereo_smiles(node: Node) -> Optional[str]:
    try:
        import qcinf

        return qcinf.structure_to_smiles(node.structure)
    except Exception:
        return None


def _push_outcomes(pushed: List[Optional[Node]], p1: Node) -> List[dict]:
    """Where each push went relative to P1: same atom-indexed bonds or not,
    and -- for same-bond outcomes -- whether it is a different stereoisomer
    (stereo SMILES) or just a different conformer (RMSD). A ridge that
    splits into two stereoisomers is a real bifurcation that bond-set
    identity alone does not see."""
    from mepd.discovery.vri_candidates import BOHR_TO_ANGSTROM, aligned_rmsd

    ref_smiles = _stereo_smiles(p1)
    out = []
    for node in pushed:
        if node is None:
            out.append({"failed": True})
            continue
        bonds_same = same_species(node, p1)
        smi = _stereo_smiles(node) if bonds_same else None
        out.append({
            "same_bonds": bool(bonds_same),
            "stereo_differs": bool(bonds_same and smi and ref_smiles and smi != ref_smiles),
            "rmsd_to_p1": float(aligned_rmsd(np.asarray(node.coords), np.asarray(p1.coords)) * BOHR_TO_ANGSTROM)
            if _is_molecular(node) else float(np.linalg.norm(np.asarray(node.coords) - np.asarray(p1.coords))),
            "rel_energy_to_p1_kcal": None if node._cached_energy is None or p1._cached_energy is None
            else (float(node._cached_energy) - float(p1._cached_energy)) * 627.5095,
        })
    return out


def _unique(nodes: List[Node]) -> List[Node]:
    out: List[Node] = []
    for node in nodes:
        if not any(same_species(node, other) for other in out):
            out.append(node)
    return out


def find_bifurcation_products(
    scan: VRTScan,
    branch_name: str,
    engine: Engine,
    *,
    opt_keywords: Optional[dict] = None,
    reference_nodes: Sequence[Node] = (),
    push_amplitude: float = 0.3,
    n_push_points: int = 3,
    imaginary_cutoff: float = 50.0,
    workers: int = 1,
    on_event: OnEvent = None,
) -> BranchProducts:
    """Find P1, P2 (and TS2 in the symmetric case) for a branch with a VRT.

    `reference_nodes` are species that do not count as a second product
    (typically the other IRC endpoint, i.e. the reactant side).
    `imaginary_cutoff` is in the scan's frequency units.
    """
    branch = scan.branches[branch_name]
    out = BranchProducts(branch=branch_name)
    if branch.vrt is None:
        out.notes.append("no VRT on this branch")
        return out

    _emit(on_event, "optimizing_endpoint", branch=branch_name)
    endpoint = _optimize_nodes(engine, [branch.nodes[-1]], opt_keywords)[0]
    if endpoint is None:
        out.notes.append("IRC endpoint optimization failed")
        return out
    _ensure_energy_gradient([endpoint], engine)
    out.endpoint_optimized = endpoint
    end_modes = stationary_point_modes(endpoint, engine)
    scan.n_hessians += 1
    out.endpoint_n_imaginary = end_modes.n_imaginary(imaginary_cutoff)

    if out.endpoint_n_imaginary == 1:
        # Symmetric ridge: the IRC stopped on TS2.
        out.ts2 = endpoint
        out.ts2_source = "irc_endpoint"
        seeds = push_along_mode(endpoint, end_modes.lowest_mode_cart, push_amplitude)
        out.push_attempts = len(seeds)
        _emit(on_event, "pushing", branch=branch_name, total=len(seeds))
        pushed = _optimize_nodes(engine, seeds, opt_keywords, workers)
        minima = [m for m in pushed if m is not None]
        minima = _confirmed_minima(minima, engine, scan, imaginary_cutoff, workers)
        minima = [m for m in _unique(minima) if not any(same_species(m, r) for r in reference_nodes)]
        out.push_minima = minima
        if len(minima) >= 2:
            out.p1, out.p2 = minima[0], minima[1]
            out.evidence = "ts2_endpoint"
            out.ts2_checks = {
                "n_imaginary": 1,
                "below_ts1": _below(endpoint, branch.nodes[0]),
                "connects_p1_p2": True,
            }
            out.ts2_verified = bool(out.ts2_checks["below_ts1"])
        elif len(minima) == 1:
            out.p1 = minima[0]
            out.notes.append("both pushes off the TS2 endpoint reached the same minimum")
        else:
            out.notes.append("pushes off the TS2 endpoint found no minimum")
    else:
        if out.endpoint_n_imaginary > 1:
            out.notes.append(
                f"IRC endpoint optimized to a structure with {out.endpoint_n_imaginary} imaginary modes"
            )
        out.p1 = endpoint
        seeds: List[Node] = []
        for node, mode in _push_sites(branch, n_push_points):
            seeds.extend(push_along_mode(node, mode, push_amplitude))
        out.push_attempts = len(seeds)
        _emit(on_event, "pushing", branch=branch_name, total=len(seeds))
        pushed = _optimize_nodes(engine, seeds, opt_keywords, workers)
        _ensure_energy_gradient([m for m in pushed if m is not None], engine)
        out.push_outcomes = _push_outcomes(pushed, endpoint)
        minima = [m for m in pushed if m is not None]
        minima = [
            m for m in _unique(minima)
            if not same_species(m, endpoint) and not any(same_species(m, r) for r in reference_nodes)
        ]
        minima = _confirmed_minima(minima, engine, scan, imaginary_cutoff, workers)
        out.push_minima = minima
        if minima:
            _ensure_energy_gradient(minima, engine)
            out.p2 = min(minima, key=lambda n: float(n._cached_energy))
            out.evidence = "vrt_push"
        else:
            out.notes.append("no second product found by pushing along the ridge mode")

    if out.p1 is not None and out.p2 is not None:
        out.degenerate = products_isomorphic(out.p1, out.p2)
    return out


def _below(a: Node, b: Node) -> Optional[bool]:
    if a._cached_energy is None or b._cached_energy is None:
        return None
    return float(a._cached_energy) < float(b._cached_energy)


def _confirmed_minima(
    nodes: List[Node], engine: Engine, scan: VRTScan, cutoff: float, workers: int = 1,
) -> List[Node]:
    kept = []
    for node, hessian in zip(nodes, compute_hessians(nodes, engine, workers)):
        scan.n_hessians += 1
        modes = projected_frequencies(hessian, node.coords, node_masses(node), gradient=None)
        if modes.n_imaginary(cutoff) == 0:
            kept.append(node)
    return kept


def _push_sites(branch: BranchScan, n_push_points: int):
    """(node, ridge mode) pairs to push from: the VRT itself, then sampled
    points past it that still have an imaginary soft mode, evenly spread."""
    sites = [(branch.vrt.node, branch.vrt.ridge_mode_cart)]
    first_after = branch.vrt.bracket_chain_indices[1]
    candidates = [
        p for p in branch.points
        if p.chain_index is not None and p.chain_index >= first_after and p.lowest_eigval < 0
    ]
    n_extra = max(0, int(n_push_points) - 1)
    if candidates and n_extra:
        picks = np.unique(np.linspace(0, len(candidates) - 1, n_extra + 1).round().astype(int))[1:]
        if len(picks) == 0:
            picks = [len(candidates) - 1]
        for i in picks:
            sites.append((candidates[i].node, candidates[i].soft_mode_cart))
    # Where the next-softest perpendicular mode is imaginary too, the ridge may
    # open into more than two valleys: push along that mode as well.
    for p in branch.points:
        if (p.chain_index is not None and p.chain_index >= first_after and p.second_mode_cart is not None
                and len(p.lowest_freqs) > 1 and p.lowest_freqs[1] < 0):
            sites.append((p.node, p.second_mode_cart))
            break
    return sites


def verify_ts2(
    ts2: Node,
    ts2_irc_nodes: Optional[Sequence[Node]],
    p1: Node,
    p2: Node,
    ts1: Node,
    engine: Engine,
    *,
    imaginary_cutoff: float = 50.0,
) -> dict:
    """Checks that a TS2 found between P1 and P2 is the ridge saddle: one
    imaginary mode, below TS1, IRC endpoints are P1 and P2 (either order)."""
    _ensure_energy_gradient([ts2, ts1], engine)
    modes = stationary_point_modes(ts2, engine)
    checks = {
        "n_imaginary": modes.n_imaginary(imaginary_cutoff),
        "below_ts1": _below(ts2, ts1),
        "connects_p1_p2": None,
        # A TS2 search from P1/P2 can land back on TS1 itself (when P2 is really
        # TS1's other side); that is not a second saddle.
        "distinct_from_ts1": _distinct_saddles(ts1, ts2),
    }
    if ts2_irc_nodes:
        a, b = ts2_irc_nodes[0], ts2_irc_nodes[-1]
        checks["connects_p1_p2"] = bool(
            (same_species(a, p1) and same_species(b, p2))
            or (same_species(a, p2) and same_species(b, p1))
        )
    checks["verified"] = bool(
        checks["n_imaginary"] == 1 and checks["below_ts1"] and checks["connects_p1_p2"]
        and checks["distinct_from_ts1"]
    )
    return checks


def _distinct_saddles(a: Node, b: Node, *, rmsd_min: float = 0.05, kcal_min: float = 0.5) -> bool:
    """Different stationary points: aligned RMSD above `rmsd_min` (Angstrom)
    or energies further apart than `kcal_min`."""
    if _is_molecular(a) and _is_molecular(b):
        from mepd.discovery.vri_candidates import BOHR_TO_ANGSTROM, aligned_rmsd

        rmsd = aligned_rmsd(np.asarray(a.coords), np.asarray(b.coords)) * BOHR_TO_ANGSTROM
    else:
        rmsd = float(np.linalg.norm(np.asarray(a.coords) - np.asarray(b.coords)))
    de = None
    if a._cached_energy is not None and b._cached_energy is not None:
        de = abs(float(a._cached_energy) - float(b._cached_energy)) * 627.5095
    return bool(rmsd > rmsd_min or (de is not None and de > kcal_min))


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

VERDICT_PRIORITY = [
    "bifurcation",
    "second_product_untested",
    "second_product_no_split",
    "vrt_no_second_product",
    "transient_softening",
    "no_vrt",
]


def basin_split(checks: Optional[dict]) -> bool:
    """Whether sideways pushes off the IRC drained into at least two
    different products (P1, P2 or another product; not back to the reactant
    side, not failures)."""
    counts = ((checks or {}).get("basin") or {}).get("counts") or {}
    products = [k for k, n in counts.items() if n > 0 and k.startswith(("P", "other"))]
    return len(products) >= 2


def branch_verdict(branch: BranchScan, products: Optional[BranchProducts]) -> str:
    """The basin test decides: a branch with a second product is a
    bifurcation only if steepest descent from sideways pushes off the IRC
    ends in at least two different products. (TS2 checks alone over-predict: a low saddle
    between P1 and P2 does not mean trajectories from TS1 reach P2.)"""
    if products is not None and products.p2 is not None:
        if not (products.checks or {}).get("basin"):
            return "second_product_untested"
        return "bifurcation" if basin_split(products.checks) else "second_product_no_split"
    if branch.vrt is None:
        return "transient_softening" if branch.transient_dips else "no_vrt"
    return "vrt_no_second_product"


def overall_verdict(verdicts: Sequence[str]) -> str:
    if not verdicts:
        return "no_vrt"
    return min(verdicts, key=VERDICT_PRIORITY.index)


def save_hessians(scan: "VRTScan", output_dir, prefix: str = "hessians") -> list:
    """Write each branch's Hessians to `<output_dir>/<prefix>_<branch>.npz`
    (compressed, float32): for every scanned IRC point and bisection point
    its arc length s, chain index (-1 for bisection points), energy,
    Cartesian gradient, coordinates (bohr) and Cartesian Hessian
    (Eh/bohr^2). Returns the written paths."""
    from pathlib import Path

    written = []
    for name, branch in scan.branches.items():
        pts = list(branch.points) + list(branch.bisection_points)
        hs = list(branch.hessians) + list(branch.bisection_hessians)
        if not pts or len(hs) != len(pts):
            continue
        fp = Path(output_dir) / f"{prefix}_{name}.npz"
        np.savez_compressed(
            fp,
            s=np.array([p.s for p in pts]),
            chain_index=np.array([-1 if p.chain_index is None else p.chain_index for p in pts]),
            bisection=np.array([p.chain_index is None for p in pts]),
            energy=np.array([p.energy for p in pts]),
            gradient=np.array([np.asarray(p.node._cached_gradient, dtype=np.float32).reshape(-1) for p in pts]),
            coords=np.array([np.asarray(p.node.coords, dtype=np.float64).reshape(-1) for p in pts]),
            symbols=np.array(list(getattr(pts[0].node, "symbols", []) or [])),
            hessian=np.array(hs, dtype=np.float32),
        )
        written.append(fp)
    return written
