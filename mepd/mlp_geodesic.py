"""MLP-GI: a path's geodesic length on the potential energy surface,
minimized directly (Table 1 and the algorithm of 10.1021/acs.jctc.5c01221;
ported from neb-dynamics' mlp_geodesic package, in numpy).

Each segment k of the path (nodes R_k, R_k+1) gets a parabola through the
energies at R_k, its Cartesian midpoint and R_k+1; its length s_k is the
arc length of that parabola, integrated in closed form. The loss is

    J = sum_k s_k + beta * sum_k (s_k / mean(s) - 1)^2

and its gradient (from the forces at the nodes and midpoints, projected off
the path tangent) moves the interior nodes. Two FIRE stages run: a
relaxation, then one with a climbing image at the highest node and, every
`refinement_step_interval` steps, new nodes inserted where a segment's
parabola puts a maximum the nodes miss. A stage stops early once the path
length and both barriers have each varied by less than their tolerance over
`fire_conv_window` steps.

Units: coordinates in Angstrom, energies in eV, forces in eV/Angstrom (the
tolerances in OptimizerConfig are in eV). `evaluate(coords [n, N, 3])`
returns (energies [n], forces [n, N, 3]) and is the only contact with a
potential.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger("mepd.mlp_geodesic")

EPS = np.finfo(np.float64).eps ** 0.25     # numerical-stability floor (the paper's eps^(1/4))


@dataclass
class OptimizerConfig:
    fire_stage1_iter: int = 200
    fire_stage2_iter: int = 500
    fire_grad_tol: float = 1e-2                   # eV/Angstrom
    variance_penalty_weight: float = 0.0433641    # beta: 1 kcal/mol in eV
    fire_conv_window: int = 20
    fire_conv_geolen_tol: float = 0.0108410       # 0.25 kcal/mol in eV
    fire_conv_erelpeak_tol: float = 0.0108410     # 0.25 kcal/mol in eV
    refinement_step_interval: int = 10
    refinement_dynamic_threshold_fraction: float = 0.1
    tangent_project: bool = True
    climb: bool = True
    alpha_climb: float = 0.5


@dataclass
class PathData:
    nodes: np.ndarray                    # [n, N, 3]
    energies: np.ndarray                 # [n]
    forces: np.ndarray                   # [n, N, 3]
    midpoint_energies: np.ndarray | None = None   # [n - 1]
    midpoint_forces: np.ndarray | None = None


# ------------------------------------------------------------------ geometry

def _tangent(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Unit tangent(s) at `cur`: the normalized sum of the unit vectors from
    the previous node and to the next. Arrays [..., N, 3]."""
    fwd, bwd = nxt - cur, cur - prev
    fwd = fwd / (np.linalg.norm(fwd, axis=(-2, -1), keepdims=True) + EPS)
    bwd = bwd / (np.linalg.norm(bwd, axis=(-2, -1), keepdims=True) + EPS)
    t = fwd + bwd
    return t / (np.linalg.norm(t, axis=(-2, -1), keepdims=True) + EPS)


def align_geom(ref: np.ndarray, geom: np.ndarray) -> np.ndarray:
    """`geom` rotated and translated onto `ref` (Kabsch)."""
    rc, gc = ref.mean(axis=0), geom.mean(axis=0)
    r, g = ref - rc, geom - gc
    try:
        u, _, vh = np.linalg.svd(g.T @ r)
    except np.linalg.LinAlgError:
        return g + rc
    rot = u @ vh
    if np.linalg.det(rot) < 0:
        vh[-1, :] *= -1
        rot = u @ vh
    return g @ rot + rc


def align_path(nodes: np.ndarray, product: np.ndarray) -> np.ndarray:
    """Each node aligned onto the one before it (the first stays put); the
    product is aligned from its original geometry each time, so alignment
    errors never accumulate in it."""
    if len(nodes) < 2:
        return nodes.copy()
    out = nodes.copy()
    for i in range(len(nodes) - 2):
        out[i + 1] = align_geom(out[i], out[i + 1])
    out[-1] = align_geom(out[-2], product)
    return out


# ------------------------------------------------------------------ the geodesic length

def segment_lengths(e: np.ndarray, e_mid: np.ndarray) -> np.ndarray:
    """s_k for each segment: the arc length of the parabola through
    (0, E_k), (1/2, E_mid,k), (1, E_k+1), in closed form."""
    n_seg = len(e) - 1
    if n_seg <= 0:
        return np.empty(0)
    ek, ek1 = e[:-1], e[1:]
    a = 2 * (ek + ek1 - 2 * e_mid)
    b = -3 * ek - ek1 + 4 * e_mid
    u0, u1 = b, 2 * a + b

    def g(u):
        root = np.sqrt(u ** 2 + EPS)
        arg = u + root
        with np.errstate(divide="ignore", invalid="ignore"):
            safe = np.where(arg < EPS, -EPS / (2 * u), arg)
            return u * root + EPS * np.log(safe)

    out = np.zeros(n_seg)
    taylor = np.abs(a) < EPS
    out[taylor] = np.sqrt(b[taylor] ** 2 + EPS)
    an = ~taylor
    if an.any():
        out[an] = (g(u1[an]) - g(u0[an])) / (4 * a[an])
    return out


def loss_gradient(pd: PathData, lengths: np.ndarray, beta: float, tangent_project: bool, climb: bool,
                  alpha_climb: float) -> np.ndarray:
    """dJ/dR at every node [n, N, 3] (endpoints included; the caller zeroes them)."""
    nodes = pd.nodes
    n_seg = len(nodes) - 1
    if n_seg <= 0 or pd.midpoint_energies is None:
        return np.zeros_like(nodes)
    ek, ek1, em = pd.energies[:-1], pd.energies[1:], pd.midpoint_energies
    a = 2 * (ek + ek1 - 2 * em)
    b = -3 * ek - ek1 + 4 * em
    dsda, dsdb = np.zeros_like(a), np.zeros_like(a)
    taylor = np.abs(a) < EPS
    an = ~taylor
    if an.any():
        aa, bb, ll = a[an], b[an], lengths[an]
        r0, r1 = np.sqrt(bb ** 2 + EPS), np.sqrt((2 * aa + bb) ** 2 + EPS)
        dsda[an], dsdb[an] = (r1 - ll) / aa, (r1 - r0) / (2 * aa)
    if taylor.any():
        bb = b[taylor]
        dsdb[taylor] = bb / np.sqrt(bb ** 2 + EPS)
    dl_dek, dl_dem, dl_dek1 = 2 * dsda - 3 * dsdb, -4 * dsda + 4 * dsdb, 2 * dsda - dsdb

    fk, fk1, fm = pd.forces[:-1], pd.forces[1:], pd.midpoint_forces
    g_rk = dl_dek[:, None, None] * (-fk) + dl_dem[:, None, None] * (-0.5 * fm)
    g_rk1 = dl_dek1[:, None, None] * (-fk1) + dl_dem[:, None, None] * (-0.5 * fm)

    grad_path, grad_var = np.zeros_like(nodes), np.zeros_like(nodes)
    grad_path[:-1] += g_rk
    grad_path[1:] += g_rk1
    if beta > 0 and len(lengths) > 1:
        mean = lengths.mean()
        factor = beta * (2.0 * (lengths - (lengths ** 2).sum() / lengths.sum()) / mean ** 2)
        grad_var[:-1] += factor[:, None, None] * g_rk
        grad_var[1:] += factor[:, None, None] * g_rk1

    if tangent_project and len(nodes) > 2:
        t = _tangent(nodes[:-2], nodes[1:-1], nodes[2:])
        inner = grad_path[1:-1]
        grad_path[1:-1] = inner - np.sum(inner * t, axis=(-2, -1), keepdims=True) * t

    grad = grad_path + grad_var
    if climb and len(nodes) > 2:
        k = int(np.argmax(pd.energies))
        if 0 < k < len(nodes) - 1:
            t = _tangent(nodes[k - 1], nodes[k], nodes[k + 1])
            if np.linalg.norm(t) > EPS:
                grad_u_par = np.sum(-pd.forces[k] * t)
                grad[k] = grad[k] - np.sum(grad[k] * t) * t - alpha_climb * grad_u_par * t
    return grad


# ------------------------------------------------------------------ refinement

def _parabola_extremum(ek, emid, ek1, rk, rk1):
    a, b, _ = np.polyfit([0.0, 0.5, 1.0], [ek, emid, ek1], 2)
    if abs(a) < EPS:
        return None
    x = -b / (2 * a)
    if not (EPS < x < 1.0 - EPS):
        return None
    return ("min" if a > 0 else "max"), (1.0 - x) * rk + x * rk1


def refine(opt: "GeodesicOptimizer", pd: PathData) -> tuple[PathData, bool]:
    """Insert a node at each segment's parabola maximum that is either well
    above the segment's sampled energies (a barrier the nodes miss) or well
    below them (the parabola is a poor fit). Returns the (possibly new)
    path data and whether nodes were inserted."""
    nodes = opt.aligned(pd.nodes)
    cur = opt.evaluate_path(nodes, midpoints=True)
    proposals = []
    for k in range(len(nodes) - 1):
        found = _parabola_extremum(cur.energies[k], cur.midpoint_energies[k], cur.energies[k + 1],
                                   nodes[k], nodes[k + 1])
        if found:
            proposals.append((k, *found))
    if not proposals:
        return cur, False
    e_cand, _ = opt.evaluate(np.stack([p[2] for p in proposals]))
    lengths = segment_lengths(cur.energies, cur.midpoint_energies)
    inserts = []
    for (k, kind, coords), e in zip(proposals, e_cand):
        if kind != "max":
            continue
        sampled = (cur.energies[k], cur.energies[k + 1], cur.midpoint_energies[k])
        hi, lo = max(sampled), min(sampled)
        threshold = opt.config.refinement_dynamic_threshold_fraction * lengths[k]
        if e > hi + threshold or e < max(hi - threshold, lo):
            inserts.append((k, coords))
    if not inserts:
        return cur, False
    out = list(nodes)
    for k, coords in sorted(inserts, key=lambda x: x[0], reverse=True):
        out.insert(k + 1, coords)
    log.info("MLP-GI refinement: %d node(s) inserted", len(inserts))
    new = opt.aligned(np.stack(out))
    return opt.evaluate_path(new, midpoints=True), True


# ------------------------------------------------------------------ the optimizer

class _StageConverged(Exception):
    pass


class _PathChanged(Exception):
    pass


class GeodesicOptimizer:
    """Optimizes the path `frames` [n, N, 3] (Angstrom) between its fixed
    ends on the surface `evaluate`. `on_step(path_data, caption)` is called
    after every step (e.g. to show the path live)."""

    def __init__(self, frames: np.ndarray, symbols: list[str],
                 evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
                 config: OptimizerConfig | None = None,
                 on_step: Callable[[PathData, str], None] | None = None,
                 on_status: Callable[[str], None] | None = None, align: bool = True):
        frames = np.asarray(frames, dtype=float)
        self.symbols = list(symbols)
        self.config = config or OptimizerConfig()
        self._evaluate = evaluate
        self.on_step = on_step
        self.on_status = on_status
        self.align = align                     # off with frozen atoms: a rotation would move them
        self.product = frames[-1].copy()       # the product as given: the alignment reference
        self.nodes = frames.copy()
        self.evaluations = 0
        self.last: PathData | None = None
        self._lengths = np.empty(0)
        self.steps = 0

    def aligned(self, nodes: np.ndarray) -> np.ndarray:
        return align_path(nodes, self.product) if self.align else nodes.copy()

    def _status(self, message: str) -> None:
        log.debug(message)
        if self.on_status:
            self.on_status(message)

    def evaluate(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(coords) == 0:
            return np.empty(0), np.empty((0, *coords.shape[1:]))
        self.evaluations += len(coords)
        e, f = self._evaluate(np.asarray(coords, dtype=float))
        return np.asarray(e, dtype=float).reshape(-1), np.asarray(f, dtype=float).reshape(coords.shape)

    def evaluate_path(self, nodes: np.ndarray, midpoints: bool = False) -> PathData:
        if midpoints and len(nodes) > 1:
            # One batch: nodes and midpoints together (engines parallelize over it).
            mids = 0.5 * (nodes[:-1] + nodes[1:])
            e, f = self.evaluate(np.concatenate([nodes, mids]))
            n = len(nodes)
            return PathData(nodes, e[:n], f[:n], e[n:], f[n:])
        e, f = self.evaluate(nodes)
        return PathData(nodes, e, f)

    def _loss(self, step: int, refine_now: bool, climb: bool) -> tuple[float, np.ndarray]:
        """Loss and its gradient on the interior nodes; raises _PathChanged
        when refinement inserted nodes (FIRE then restarts on the new path)."""
        cfg = self.config
        pd = self.evaluate_path(self.nodes, midpoints=True)
        interval = cfg.refinement_step_interval
        if refine_now and ((interval == 0 and step == 1) or (interval > 0 and step % interval == 0)):
            pd, changed = refine(self, pd)
            if changed:
                self.nodes = pd.nodes.copy()
                self.last = pd
                raise _PathChanged
        self.last = pd
        lengths = segment_lengths(pd.energies, pd.midpoint_energies)
        if len(lengths) == 0:
            return 0.0, np.zeros((0, *self.nodes.shape[1:]))
        beta = cfg.variance_penalty_weight
        loss = lengths.sum() + beta * ((lengths / lengths.mean() - 1.0) ** 2).sum()
        grad = loss_gradient(pd, lengths, beta, cfg.tangent_project, cfg.climb and climb, cfg.alpha_climb)
        self._lengths = lengths
        return float(loss), grad[1:-1]

    def _run_stage(self, name: str, max_iters: int, refine_on: bool, climb: bool) -> None:
        from ase import Atoms
        from ase.calculators.calculator import Calculator, all_changes
        from ase.optimize import FIRE

        cfg = self.config
        opt = self
        window = max(1, int(cfg.fire_conv_window))
        hist_len, hist_fwd, hist_back = deque(maxlen=window), deque(maxlen=window), deque(maxlen=window)
        done, calc, atoms = 0, None, None
        self._status(f"MLP-GI {name}: up to {max_iters} steps")

        class _Loss(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self, start):
                super().__init__()
                self.step = start
                self.changed = False

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                self.step += 1
                opt.nodes[1:-1] = self.atoms.get_positions().reshape(opt.nodes[1:-1].shape)
                try:
                    loss, grad = opt._loss(self.step, refine_on, climb)
                except _PathChanged:
                    self.changed = True
                    raise
                self.results["energy"] = loss
                self.results["forces"] = -grad.reshape(-1, 3)

        while done < max_iters:
            n_int = len(self.nodes) - 2
            if n_int <= 0:
                return
            atoms = Atoms(symbols=self.symbols * n_int, positions=self.nodes[1:-1].reshape(-1, 3))
            calc = _Loss(done)
            atoms.calc = calc
            fire = FIRE(atoms, logfile=None)

            def observe():
                pd = opt.last
                if pd is None:
                    return
                e = pd.energies
                fwd, back = e.max() - e[0], e.max() - e[-1]
                length = float(opt._lengths.sum()) if len(opt._lengths) else 0.0
                opt.steps += 1
                if opt.on_step:
                    opt.on_step(pd, f"MLP-GI {name} · step {calc.step} · path length {length:.3f} eV")
                hist_len.append(length)
                hist_fwd.append(fwd)
                hist_back.append(back)
                if len(hist_len) == window:
                    if (max(hist_len) - min(hist_len) < cfg.fire_conv_geolen_tol
                            and max(hist_fwd) - min(hist_fwd) < cfg.fire_conv_erelpeak_tol
                            and max(hist_back) - min(hist_back) < cfg.fire_conv_erelpeak_tol):
                        raise _StageConverged

            fire.attach(observe, interval=1)
            try:
                fire.run(fmax=cfg.fire_grad_tol, steps=max_iters - done)
                done = calc.step
                self._status(f"MLP-GI {name}: stopped (gradient below {cfg.fire_grad_tol:g} or step limit)")
                break
            except _PathChanged:
                done = calc.step
                self._status(f"MLP-GI {name}: {len(self.nodes)} nodes after refinement, continuing")
                continue
            except _StageConverged:
                done = calc.step
                self._status(f"MLP-GI {name}: converged (path length and barriers steady)")
                break
        if calc is not None and not calc.changed:
            self.nodes[1:-1] = atoms.get_positions().reshape(self.nodes[1:-1].shape)

    def optimize(self) -> PathData:
        """Both stages; returns the final path with its energies and forces."""
        cfg = self.config
        self.nodes = self.aligned(self.nodes)
        self._run_stage("stage 1 (relaxation)", int(cfg.fire_stage1_iter), refine_on=False, climb=False)
        self.nodes = self.aligned(self.nodes)
        interval = int(cfg.refinement_step_interval)
        iters = int(cfg.fire_stage2_iter)
        if interval > 0:
            # Ends half an interval after a refinement, so the last inserted
            # nodes get relaxed before the path is returned.
            iters = (iters // interval) * interval + interval // 2
        self._run_stage("stage 2 (climbing, refinement)", iters, refine_on=True, climb=True)
        self.last = self.evaluate_path(self.nodes)
        return self.last
