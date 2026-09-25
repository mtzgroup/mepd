"""Checks that go beyond the static VRI search:

- `refine_vri`: converge from a VRT to an exact valley-ridge inflection point,
  H(x) v = 0 with v perpendicular to the gradient (Gauss-Newton with the
  Moore-Penrose pseudoinverse, as in Schmidt & Quapp, TCA 132, 1305 (2013)).
- `basin_test`: steepest descent from points just to either side of the
  IRC past TS1. On a bifurcating surface, neighbouring starting points end in
  different products; otherwise they all return to P1.

Quasiclassical trajectories live in `mepd.discovery.qct`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from mepd.discovery.vri import (
    _is_molecular,
    compute_cartesian_hessian,
    node_masses,
    projected_frequencies,
    rigid_body_basis,
    same_species,
)

logger = logging.getLogger(__name__)
H2K = 627.509474
B2A = 0.529177210903


# --------------------------------------------------------------------------
# Exact VRI
# --------------------------------------------------------------------------


@dataclass
class VRIResult:
    converged: bool
    node: object
    v: np.ndarray                      # Cartesian zero-curvature direction (unit)
    iterations: int
    residuals: List[float] = field(default_factory=list)  # |F| per iteration
    grad_norm: float = 0.0
    zero_eigval: float = 0.0            # eigenvalue of the projected Hessian closest to 0
    v_dot_g: float = 0.0                # |v . g| / |g|
    n_hessians: int = 0

    def to_dict(self) -> dict:
        return {
            "converged": self.converged, "iterations": self.iterations,
            "residuals": self.residuals, "grad_norm": self.grad_norm,
            "zero_eigval": self.zero_eigval, "v_dot_g": self.v_dot_g,
            "n_hessians": self.n_hessians,
        }


def refine_vri(
    start,
    v0: np.ndarray,
    engine,
    *,
    max_iter: int = 20,
    tol: float = 1e-5,
    fd_step: float = 5e-3,
    max_step: float = 0.1,
    on_iter=None,
) -> VRIResult:
    """Gauss-Newton on F(x, v) = [H v; g.v; (v.v - 1)/2; R^T v] = 0.

    Coordinates are Cartesian (bohr); for molecules the rigid-body directions
    are handled by adding a large shift on them in H (so they are never the
    zero mode) and requiring v to be orthogonal to them (R^T v = 0). The
    Jacobian's x-block is d(Hv)/dx = D_v H, the directional derivative of H
    along v, from two extra Hessians at x +- `fd_step` v. The system is
    underdetermined (VRIs form a manifold), so the step is the minimum-norm
    pseudoinverse step: it converges to the VRI nearest the start. Steps are
    capped at `max_step` (bohr, largest component).
    """
    node = start.copy()
    x = np.asarray(node.coords, dtype=float).reshape(-1)
    shape = np.asarray(node.coords).shape
    n = x.size
    v = np.asarray(v0, dtype=float).reshape(-1)
    v = v / np.linalg.norm(v)
    molecular = _is_molecular(node)
    residuals: List[float] = []
    n_hess = 0

    def hessian_at(xx):
        nonlocal n_hess
        n_hess += 1
        return compute_cartesian_hessian(node.update_coords(xx.reshape(shape)), engine)

    def rigid(xx):
        return rigid_body_basis(xx.reshape(shape), np.ones(n // 3)) if molecular else np.zeros((n, 0))

    shift = 10.0  # Eh/bohr^2 on rigid-body directions: far from zero
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        cur = node.update_coords(x.reshape(shape))
        g = np.asarray(engine.compute_gradients([cur])[0], dtype=float).reshape(-1)
        R = rigid(x)
        H = hessian_at(x) + shift * (R @ R.T)
        F = np.concatenate([H @ v, [g @ v, 0.5 * (v @ v - 1.0)], R.T @ v])
        # scale so the residual is comparable across terms
        res = float(np.linalg.norm(F[:n]) + abs(F[n]) + abs(F[n + 1]) + np.linalg.norm(F[n + 2:]))
        residuals.append(res)
        if on_iter is not None:
            on_iter(it, res)
        if res < tol:
            converged = True
            break
        Hp = hessian_at(x + fd_step * v) + shift * (R @ R.T)
        Hm = hessian_at(x - fd_step * v) + shift * (R @ R.T)
        DvH = 0.5 * ((Hp - Hm) / (2.0 * fd_step) + ((Hp - Hm) / (2.0 * fd_step)).T)
        J = np.zeros((F.size, 2 * n))
        J[:n, :n] = DvH
        J[:n, n:] = H
        J[n, :n] = H @ v - shift * (R @ (R.T @ v))  # d(g.v)/dx = (H_true v)
        J[n, n:] = g
        J[n + 1, n:] = v
        J[n + 2:, n:] = R.T
        dz = -np.linalg.pinv(J, rcond=1e-8) @ F
        dx, dv = dz[:n], dz[n:]
        if molecular:  # no rigid-body motion of the geometry
            dx = dx - R @ (R.T @ dx)
        biggest = np.max(np.abs(dx)) if dx.size else 0.0
        if biggest > max_step:
            scale = max_step / biggest
            dx, dv = dx * scale, dv * scale
        x = x + dx
        v = v + dv
        v = v / np.linalg.norm(v)

    final = node.update_coords(x.reshape(shape))
    g = np.asarray(engine.compute_gradients([final])[0], dtype=float).reshape(-1)
    e = float(engine.compute_energies([final])[0])
    final._cached_gradient = g.reshape(shape)
    final._cached_energy = e
    H = hessian_at(x)
    masses = node_masses(final)
    modes = projected_frequencies(H, final.coords, masses, gradient=None)
    k = int(np.argmin(np.abs(modes.eigvals)))
    return VRIResult(
        converged=converged, node=final, v=v.reshape(shape), iterations=it, residuals=residuals,
        grad_norm=float(np.linalg.norm(g)), zero_eigval=float(modes.eigvals[k]),
        v_dot_g=float(abs(v @ g) / max(np.linalg.norm(g), 1e-12)), n_hessians=n_hess,
    )


# --------------------------------------------------------------------------
# Basin test
# --------------------------------------------------------------------------


def steepest_descent_to_minimum(
    node, engine, *, step: float = 0.04, max_steps: int = 1500, gtol: float = 2e-4,
    opt_keywords: Optional[dict] = None, record: Optional[list] = None, record_every: int = 10,
):
    """Mass-weighted steepest descent with a fixed arc-length step (bohr
    amu^1/2), then a local polish once the gradient is small. Unlike a
    quasi-Newton optimizer it cannot hop between basins, so where it ends
    tells which basin the start was in. With `record`, appends
    (coords, energy, gradient) every `record_every` steps and at the end
    (energy/gradient None where not computed)."""
    cur = node.copy()
    if type(cur).__name__ == "StructureNode":
        cur.has_molecular_graph = False
        cur.graph = None
    masses = node_masses(cur)
    sqrt_m = np.repeat(np.sqrt(masses), 3) if masses is not None else np.ones(np.asarray(cur.coords).size)
    shape = np.asarray(cur.coords).shape
    e_prev = None
    for k in range(int(max_steps)):
        g = np.asarray(engine.compute_gradients([cur])[0], dtype=float).reshape(-1)
        e = float(engine.compute_energies([cur])[0])
        if record is not None and k % max(1, int(record_every)) == 0:
            record.append((np.array(cur.coords, dtype=float), e, g.copy()))
        if np.linalg.norm(g) < gtol or (e_prev is not None and e > e_prev + 1e-7):
            break
        gm = g / sqrt_m
        dq = -step * gm / np.linalg.norm(gm)
        cur = cur.update_coords((np.asarray(cur.coords).reshape(-1) + dq / sqrt_m).reshape(shape))
        e_prev = e
    try:
        try:
            traj = engine.compute_geometry_optimization(cur, keywords=opt_keywords)
        except TypeError:
            traj = engine.compute_geometry_optimization(cur)
        if traj:
            cur = traj[-1]
    except Exception as exc:
        logger.info("polish failed: %s", exc)
    from mepd.nodes.node import StructureNode

    if record is not None:
        e_end = getattr(cur, "_cached_energy", None)
        g_end = getattr(cur, "_cached_gradient", None)
        record.append((np.array(cur.coords, dtype=float), None if e_end is None else float(e_end),
                       None if g_end is None else np.asarray(g_end, dtype=float).reshape(-1)))
    if isinstance(cur, StructureNode) and not cur.has_molecular_graph:
        cur = StructureNode(structure=cur.structure)
    return cur


class ProductRegistry:
    """Names every product the checks see: P1 and P2 as given, the
    reactant side as R, and anything else P3, P4, ... in order of first
    appearance (one entry per species, atom-indexed heavy-atom bonds +
    H counts, see `same_species`). Keeps one structure per product."""

    def __init__(self, p1, p2=None, reactants: Sequence = ()):
        # Without a known P2, the first further product found becomes P2.
        self.entries = [("P1", p1)] + ([("P2", p2)] if p2 is not None else [])
        self.reactants = list(reactants)
        self.counts: dict = {}

    def label(self, node) -> str:
        """Exact (atom-indexed) matches first; then extras that are the same
        molecule as an earlier extra share its label, and one that is the
        same molecule as P1 or P2 with different atoms bonded is P1' / P2'.
        P1 and P2 themselves stay distinct even when isomorphic (degenerate
        bifurcations)."""
        from mepd.discovery.vri import products_isomorphic

        if node is None:
            return "failed"
        for name, ref in self.entries:
            if same_species(node, ref):
                return name
        if any(same_species(node, r) for r in self.reactants):
            return "R"
        for name, ref in self.entries:
            if name in ("P1", "P2"):
                continue
            if products_isomorphic(node, ref):
                return name
        for name, ref in self.entries[:2]:
            if products_isomorphic(node, ref):
                prime = f"{name}'"
                if prime not in [n for n, _ in self.entries]:
                    self.entries.append((prime, node))
                return prime
        n_extra = sum(1 for n, _ in self.entries if not n.endswith("'")) + 1
        name = f"P{n_extra}"
        self.entries.append((name, node))
        return name

    def describe(self, e_ref: Optional[float] = None) -> List[dict]:
        """Per product: label, SMILES, energy vs `e_ref` (kcal/mol), and
        whether it is the same molecule as P1 or P2 with different atoms
        bonded (a symmetry-equivalent product) or genuinely different."""
        from mepd.discovery.vri import _stereo_smiles, products_isomorphic

        out = []
        for name, node in self.entries:
            eq = name[:-1] if name.endswith("'") else None
            e = getattr(node, "_cached_energy", None)
            out.append({"label": name, "smiles": _stereo_smiles(node),
                        "e_rel_kcal": None if e is None or e_ref is None else (float(e) - e_ref) * H2K,
                        "equivalent_to": eq})
        return out


@dataclass
class BasinResult:
    counts: dict
    starts: List[dict]  # per start: s, side, outcome
    paths: List[list] = field(default_factory=list, repr=False)  # per start: [(coords, energy, gradient), ...]

    def to_dict(self) -> dict:
        return {"counts": self.counts, "starts": self.starts}


def basin_test(
    irc_branch: Sequence,
    s_values: Sequence[float],
    ridge_modes: Sequence[np.ndarray],
    p1,
    p2,
    engine,
    *,
    displacements: Sequence[float] = (0.05, 0.1, 0.2, 0.3),
    workers: int = 1,
    opt_keywords: Optional[dict] = None,
    reference: Sequence = (),
    registry: Optional[ProductRegistry] = None,
) -> BasinResult:
    """For each IRC point (with its softest perpendicular mode), displace
    sideways by each of `displacements` (bohr, largest single-atom move) in
    both directions and run steepest descent. The IRC usually runs a little
    off the ridge crest, so several sizes are needed to see whether the
    other side of the ridge drains to P2. `ridge_modes` gives per point one
    mode or a list (softest first; the second is pushed along too where it is
    also imaginary). Outcome per start: "P1", "P2", "R" (back to a
    `reference` structure, the reactant side), "P3", "P4", ... (further
    products, named by the shared `registry`) or "failed". Symmetry-
    equivalent products can appear as P3 -- e.g. the third fluorine of a
    CF3 group migrating."""
    from mepd.discovery.vri import push_along_mode

    jobs = []
    for node, s, modes in zip(irc_branch, s_values, ridge_modes):
        modes = modes if isinstance(modes, (list, tuple)) else [modes]
        for m_index, mode in enumerate(modes):
            if mode is None:
                continue
            for d in displacements:
                plus, minus = push_along_mode(node, mode, d)
                jobs += [(s, d, f"+{m_index + 1}", plus), (s, d, f"-{m_index + 1}", minus)]

    def run(job):
        s, d, side, start = job
        path: list = []
        try:
            end = steepest_descent_to_minimum(start, engine, opt_keywords=opt_keywords, record=path)
        except Exception as exc:
            logger.info("basin descent failed: %s", exc)
            end = None
        return {"s": float(s), "displacement": float(d), "side": side}, end, path

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = list(pool.map(run, jobs))
    registry = registry or ProductRegistry(p1, p2, reference)
    starts, paths = [], []
    for info, end, path in results:
        starts.append({**info, "outcome": registry.label(end)})
        paths.append(path)
    counts: dict = {}
    for st in starts:
        counts[st["outcome"]] = counts.get(st["outcome"], 0) + 1
    return BasinResult(counts=counts, starts=starts, paths=paths)


def save_paths(fp, kind: str, paths: Sequence[list], labels: Sequence[str], info: Sequence[dict]) -> None:
    """Store check paths for the viewer in `fp` (npz), under `kind`
    ("basin" or "trajectory"), keeping the other kind if the file exists.
    Per path: frames of (coords bohr, energy Eh or None, gradient or None),
    its outcome label and a JSON-able info dict. Arrays: <kind>_coords
    (F, 3N), <kind>_energies (F,), <kind>_gradients (F, 3N) (NaN where
    missing), <kind>_offsets (P+1,), <kind>_labels, <kind>_info (JSON)."""
    import json
    from pathlib import Path

    fp = Path(fp)
    keep = {}
    if fp.exists():
        with np.load(fp) as z:
            keep = {k: z[k] for k in z.files if not k.startswith(kind + "_")}
    frames = [f for p in paths for f in p]
    if not frames:
        return
    n3 = np.asarray(frames[0][0]).size
    coords = np.array([np.asarray(c, dtype=float).reshape(-1) for c, _, _ in frames], dtype=np.float32)
    energies = np.array([np.nan if e is None else e for _, e, _ in frames], dtype=float)
    grads = np.full((len(frames), n3), np.nan, dtype=np.float32)
    for i, (_, _, g) in enumerate(frames):
        if g is not None:
            grads[i] = np.asarray(g, dtype=float).reshape(-1)
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in paths])]).astype(int)
    np.savez_compressed(fp, **keep, **{
        f"{kind}_coords": coords, f"{kind}_energies": energies, f"{kind}_gradients": grads,
        f"{kind}_offsets": offsets, f"{kind}_labels": np.array(list(labels), dtype=str),
        f"{kind}_info": np.array([json.dumps(i) for i in info], dtype=str)})


# --------------------------------------------------------------------------
# All checks for one VRI output branch
# --------------------------------------------------------------------------


def _load_nodes(fp, charge, multiplicity):
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs

    return list(Chain.from_xyz(fp, ChainInputs(), charge=charge, spinmult=multiplicity).nodes)


def check_branch(
    cand_dir,
    branch: str,
    engine,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    workers: int = 1,
    n_traj: int = 150,
    traj_fs: float = 400.0,
    do_refine: bool = True,
    do_basin: bool = True,
    n_basin_points: int = 5,
    opt_keywords: Optional[dict] = None,
    log=print,
) -> Optional[dict]:
    """Exact VRI, basin test and trajectories for one branch with a P1.
    P2 is the second product the VRI search found, if any; otherwise the
    first product the basin test reaches that is neither P1 nor the reactant
    becomes P2 (written to p2_<branch>.xyz, `out["p2_source"] = "basin"`).
    Writes vri_<branch>.xyz (the refined VRI) and returns the results."""
    import json
    from pathlib import Path

    from mepd.discovery import qct, vri
    from mepd.inputs import ChainInputs

    d = Path(cand_dir)
    summary = json.loads((d / "summary.json").read_text())
    scan = json.loads((d / "projected_freqs.json").read_text())
    if not (d / f"p1_{branch}.xyz").exists():
        return None
    p1 = _load_nodes(d / f"p1_{branch}.xyz", charge, multiplicity)[-1]
    p2_fp = d / f"p2_{branch}.xyz"
    p2 = _load_nodes(p2_fp, charge, multiplicity)[-1] if p2_fp.exists() else None
    irc = _load_nodes(d / "irc.xyz", charge, multiplicity)
    ts_index = int(scan["ts_index"])
    nodes = vri.split_irc_branches(irc, ts_index)[branch]
    s_arc = vri._arc_length(nodes)
    ts1 = nodes[0]
    e_ts1 = float(summary["ts1_energy"])
    masses = node_masses(ts1)
    out: dict = {"branch": branch, "p2_source": "search" if p2 is not None else None}

    def soft_mode(node):
        vri._ensure_energy_gradient([node], engine)
        H = compute_cartesian_hessian(node, engine)
        m = projected_frequencies(H, node.coords, masses, gradient=node._cached_gradient)
        return m.lowest_mode_cart, float(m.freqs[0])

    def soft_modes(node):
        """Softest perpendicular mode, plus the next one when it is imaginary too."""
        vri._ensure_energy_gradient([node], engine)
        H = compute_cartesian_hessian(node, engine)
        m = projected_frequencies(H, node.coords, masses, gradient=node._cached_gradient)
        second = m.modes_cart[1] if len(m.modes_cart) > 1 and m.freqs[1] < 0 else None
        return [m.lowest_mode_cart, second], [float(f) for f in m.freqs[:2]]

    reactant = vri.split_irc_branches(irc, ts_index)["reverse" if branch == "forward" else "forward"][-1]
    registry = ProductRegistry(p1, p2, [reactant])

    # 1. exact VRI
    if do_refine:
        log(f"  [{branch}] converging to the exact VRI...")
        if (d / f"vrt_{branch}.xyz").exists() and (d / f"ridge_mode_{branch}.xyz").exists():
            start = _load_nodes(d / f"vrt_{branch}.xyz", charge, multiplicity)[-1]
            rm = _load_nodes(d / f"ridge_mode_{branch}.xyz", charge, multiplicity)
            v0 = np.asarray(rm[5].coords) - np.asarray(rm[0].coords)
            origin = "VRT"
        else:  # off-IRC case: start at the IRC point with the softest perpendicular mode
            pts = (scan.get("branches") or {}).get(branch, {}).get("points", [])
            k = min(pts, key=lambda p: p["lowest_freqs"][0])["chain_index"] if pts else len(nodes) // 3
            start = nodes[k]
            v0, _ = soft_mode(start)
            origin = f"softest IRC point (s = {s_arc[k]:.2f})"
        try:
            res = refine_vri(start, v0, engine)
            vri._ensure_energy_gradient([res.node], engine)
            d_irc = min(np.linalg.norm(np.asarray(res.node.coords) - np.asarray(n.coords)) for n in nodes) * B2A
            out["vri"] = {**res.to_dict(), "start": origin,
                          "e_rel_ts1_kcal": (float(res.node._cached_energy) - e_ts1) * H2K,
                          "distance_to_irc_A": float(d_irc)}
            from mepd.chain import Chain

            Chain.model_validate({"nodes": [res.node], "parameters": ChainInputs()}).write_to_disk(d / f"vri_{branch}.xyz")
            log(f"    {'converged' if res.converged else 'NOT converged'} in {res.iterations} iterations; "
                f"E {out['vri']['e_rel_ts1_kcal']:+.1f} kcal/mol vs TS1, {d_irc:.2f} Å from the IRC, "
                f"|g| {res.grad_norm:.3f}, zero eigenvalue {res.zero_eigval:+.1e}")
        except Exception as exc:
            out["vri"] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"    failed: {exc}")

    # 2. basin test around the ridge
    if do_basin:
        log(f"  [{branch}] basin test (steepest descent from sideways pushes)...")
        vrt_s = ((summary["branches"][branch].get("vrt") or {}).get("s"))
        hi = min(s_arc[-1] * 0.8, (vrt_s + 3.0) if vrt_s else s_arc[-1] * 0.5)
        lo = max(0.2, (vrt_s - 1.0) if vrt_s else 0.2)
        picks = [int(np.argmin(np.abs(s_arc - t))) for t in np.linspace(lo, hi, n_basin_points)]
        picks = sorted(set(picks))
        pts = [nodes[k] for k in picks]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            found = list(pool.map(soft_modes, pts))
        modes = [m for m, _ in found]
        n_double = sum(1 for m in modes if m[1] is not None)
        res_b = basin_test(pts, [s_arc[k] for k in picks], modes, p1, p2, engine,
                           displacements=(0.1, 0.2, 0.3), workers=workers, opt_keywords=opt_keywords,
                           reference=[reactant], registry=registry)
        out["basin"] = res_b.to_dict()
        out["basin"]["points_with_two_imaginary_modes"] = n_double
        if p2 is None:
            found = dict(registry.entries).get("P2")
            if found is not None:
                p2 = found
                out["p2_source"] = "basin"
                vri._ensure_energy_gradient([p2], engine)
                from mepd.chain import Chain

                Chain.model_validate({"nodes": [p2], "parameters": ChainInputs()}).write_to_disk(p2_fp)
                log(f"    second product found by the basin test: {vri._stereo_smiles(p2)}")
        save_paths(d / f"paths_{branch}.npz", "basin", res_b.paths,
                   [st["outcome"] for st in res_b.starts], res_b.starts)
        log(f"    outcomes: {res_b.counts}" + (f" (second soft mode pushed at {n_double} point(s))" if n_double else ""))

    # 3. trajectories from TS1 into this branch
    if n_traj > 0:
        log(f"  [{branch}] {n_traj} quasiclassical trajectories from TS1...")
        ts_modes = vri.stationary_point_modes(ts1, engine)
        direction = vri._mass_weighted(nodes[1]) - vri._mass_weighted(nodes[0])
        other_end = vri.split_irc_branches(irc, ts_index)["reverse" if branch == "forward" else "forward"][-1]
        refs = {"P1": p1, "R": other_end, **({"P2": p2} if p2 is not None else {})}
        classify = qct.pattern_classifier(list(ts1.symbols), refs, ts1.coords)
        result = qct.run_qct(ts1, ts_modes, engine, direction, n_trajectories=n_traj, max_fs=traj_fs,
                             workers=workers, seed=0, opt_keywords=opt_keywords, commit_to=classify)
        counts = {"P1": 0, "P2": 0, "recrossed": 0, "failed": 0}
        for t in result.trajectories:
            if t.product is None:
                t.label = "failed"
            elif t.recrossed or same_species(t.product, other_end):
                t.label = "recrossed"
            else:
                t.label = registry.label(t.product)
                if t.label == "R":
                    t.label = "recrossed"
            counts[t.label] = counts.get(t.label, 0) + 1
        save_paths(d / f"paths_{branch}.npz", "trajectory",
                   [list(zip([f.coords for f in t.frames], t.potential_energies, t.frame_gradients)) for t in result.trajectories],
                   [t.label or "failed" for t in result.trajectories],
                   [{"index": t.index, "commit_fs": t.commit_fs, "recrossed": t.recrossed} for t in result.trajectories])
        drift = [abs(t.energy_drift) * H2K for t in result.trajectories if t.energy_drift is not None]
        commit_times = [t.commit_fs for t in result.trajectories if t.commit_fs is not None]
        out["trajectories"] = {"counts": counts, "n": n_traj, "fs": traj_fs,
                               "max_energy_drift_kcal": float(max(drift)) if drift else None,
                               "committed": len(commit_times),
                               "median_commit_fs": float(np.median(commit_times)) if commit_times else None}
        n12 = counts["P1"] + counts["P2"]
        ratio = f"P1:P2 = {counts['P1']}:{counts['P2']}" + (f" ({100 * counts['P2'] / n12:.0f}% P2)" if n12 else "")
        extra = {k: v for k, v in counts.items() if k.startswith("P") and k not in ("P1", "P2")}
        log(f"    {len(commit_times)} of {n_traj} trajectories committed to a product"
            + (f" (median {np.median(commit_times):.0f} fs)" if commit_times else ""))
        log(f"    {ratio}" + (f"; further products {extra}" if extra else "")
            + f"; recrossed {counts['recrossed']}, failed {counts['failed']}")

    # Every product seen, one structure each
    for _, node in registry.entries:
        if getattr(node, "_cached_energy", None) is None:
            try:
                vri._ensure_energy_gradient([node], engine)
            except Exception:
                pass
    out["products"] = registry.describe(e_ref=e_ts1)
    from mepd.chain import Chain

    Chain.model_validate({"nodes": [n for _, n in registry.entries], "parameters": ChainInputs()}).write_to_disk(
        d / f"products_{branch}.xyz")
    if len(registry.entries) > 2:
        for info in out["products"][2:]:
            log(f"    {info['label']}: {info['smiles']}"
                + (f", same molecule as {info['equivalent_to']} (symmetry-equivalent)" if info["equivalent_to"] else ", a different product")
                + (f", {info['e_rel_kcal']:+.1f} kcal/mol vs TS1" if info["e_rel_kcal"] is not None else ""))
    return out
