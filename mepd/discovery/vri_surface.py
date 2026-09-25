"""A 2D potential energy surface around a bifurcation, reconstructed from
computed points, for looking at a VRI result the way toy PES pictures are
drawn (TS1 at the top, a ridge down to TS2, P1 and P2 in two valleys).

Coordinates -- the layout of the toy bifurcation surfaces:

    x = progress through TS1: displacement along TS1's reaction mode (its
        imaginary eigenvector), u . (r - r_TS1), in Angstrom;
    y = which product: mean length of the bonds only P2 has minus mean
        length of the bonds only P1 has (Angstrom).

Reactant at x < 0, products at x > 0, TS1 at the origin as a pass; P1 at
y > 0, P2 at y < 0, the ridge and TS2 in between. (Plotting the two
product-bond lengths against each other, as in Chuang, Tantillo & Hsu,
separates P1 from P2 too, but TS1's reaction mode can be nearly
perpendicular to both -- e.g. when another group migrates at TS1 -- and TS1
then cannot appear as a pass.)

Energy on the grid is a relaxed scan (the construction Chuang, Tantillo &
Hsu use): at every grid node the structure is optimized with the engine
while flat-bottomless harmonic restraints hold (x, y) at the node, starting
from the nearest computed structure. Every surface point is therefore a
real relaxed energy with a real geometry. An interpolation of the computed
points alone (energies, gradients, Hessians; `interpolate`) is kept as a
cross-check, but with data only along a few 1D paths it extrapolates badly
into the regions between them.

Overlays: the IRC through TS1, the IRC through TS2 (P1 <-> P2), and the
stationary points TS1, VRT, TS2, P1, P2 -- computed independently of the
grid, so they check it: they should sit on its passes and in its basins.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

H2K = 627.509474
B2A = 0.529177210903


# --------------------------------------------------------------------------
# Collective coordinates
# --------------------------------------------------------------------------


def product_bond_sets(p1, p2):
    """(A, B): heavy-atom-or-not bonds only P1 has, only P2 has (atom-indexed)."""
    from mepd.discovery.vri import labeled_bond_set

    b1, b2 = labeled_bond_set(p1), labeled_bond_set(p2)
    return sorted(b1 - b2), sorted(b2 - b1)


def _mean_bond(X: np.ndarray, bonds: Sequence):
    val, grad = 0.0, np.zeros(X.size)
    for i, j in bonds:
        v = X[i] - X[j]
        d = np.linalg.norm(v)
        val += d * B2A / len(bonds)
        u = v / d * B2A / len(bonds)
        grad[3 * i:3 * i + 3] += u
        grad[3 * j:3 * j + 3] -= u
    return val, grad


class Axes:
    """x = u . (r - r_TS1) (Angstrom), y = mean(B bonds) - mean(A bonds)."""

    def __init__(self, x_ts1, mode, bonds_a, bonds_b):
        from mepd.discovery.vri import rigid_body_basis

        self.x_ts1 = np.asarray(x_ts1, dtype=float).reshape(-1)
        u = np.asarray(mode, dtype=float).reshape(-1)
        # Remove rigid translation/rotation from the direction, so x does not
        # register rigid motion as progress (it is still not invariant to
        # finite rotations: structures must be aligned to TS1, see align_to).
        R = rigid_body_basis(self.x_ts1.reshape(-1, 3), np.ones(self.x_ts1.size // 3))
        u = u - R @ (R.T @ u)
        self.u = u / np.linalg.norm(u)
        self.bonds_a, self.bonds_b = list(bonds_a), list(bonds_b)

    def __call__(self, coords_bohr):
        x = np.asarray(coords_bohr, dtype=float).reshape(-1)
        X = x.reshape(-1, 3)
        a, ga = _mean_bond(X, self.bonds_a)
        b, gb = _mean_bond(X, self.bonds_b)
        q = np.array([float(self.u @ (x - self.x_ts1)) * B2A, b - a])
        B = np.vstack([self.u * B2A, gb - ga])
        return q, B


def align_to(X: np.ndarray, ref: np.ndarray):
    """Kabsch-align X onto ref (both (N, 3), same atom order); returns the
    aligned coordinates and the rotation R such that aligned = (X - cX) R + c_ref.
    A gradient transforms as g R."""
    X = np.asarray(X, dtype=float).reshape(-1, 3)
    ref = np.asarray(ref, dtype=float).reshape(-1, 3)
    cX, cR = X.mean(0), ref.mean(0)
    U, _, Vt = np.linalg.svd((X - cX).T @ (ref - cR))
    d = np.sign(np.linalg.det(U @ Vt))
    Rot = U @ np.diag([1.0, 1.0, d]) @ Vt
    return (X - cX) @ Rot + cR, Rot


def q_and_B(coords_bohr: np.ndarray, axes: "Axes"):
    return axes(coords_bohr)


def project_gradient(B: np.ndarray, g_cart: np.ndarray) -> np.ndarray:
    """dE/dq (kcal/mol/Angstrom) from a Cartesian gradient (Eh/bohr): the
    least-squares solution of B^T g_q = g_x."""
    g_q, *_ = np.linalg.lstsq(B.T, np.asarray(g_cart, dtype=float).reshape(-1), rcond=None)
    return g_q * H2K


def relaxed_curvature(B: np.ndarray, H_cart: np.ndarray, coords_bohr: np.ndarray) -> Optional[np.ndarray]:
    """Relaxed Hessian in q (kcal/mol/Angstrom^2): (B H^+ B^T)^-1, with H^+
    the pseudo-inverse of the Cartesian Hessian on the space orthogonal to
    rigid translations/rotations. None if it is ill-conditioned."""
    from mepd.discovery.vri import rigid_body_basis

    X = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
    R = rigid_body_basis(X, np.ones(len(X)))
    P = np.eye(X.size) - R @ R.T
    Hp = P @ (0.5 * (H_cart + H_cart.T)) @ P
    w, V = np.linalg.eigh(Hp)
    keep = np.abs(w) > 1e-6
    keep[np.argsort(np.abs(w))[: R.shape[1]]] = False
    Hinv = (V[:, keep] / w[keep]) @ V[:, keep].T
    C = B @ Hinv @ B.T  # compliance, Angstrom^2 / Eh
    if np.linalg.cond(C) > 1e10:
        return None
    return np.linalg.inv(C) * H2K


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------


def _read_chain(fp: Path, charge: int, mult: int):
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs

    if not fp.exists():
        return []
    return list(Chain.from_xyz(fp, ChainInputs(), charge=charge, spinmult=mult).nodes)


def _plain(node):
    """Node copy without OpenBabel graph rebuilding on coordinate updates."""
    n = node.copy()
    if type(n).__name__ == "StructureNode":
        n.has_molecular_graph = False
        n.graph = None
    return n


def build_branch_surface(
    cand_dir: Path,
    branch: str,
    engine,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    workers: int = 1,
    n_hessians: int = 0,
    grid: int = 13,
    power: float = 4.0,
) -> Optional[dict]:
    """Surface data for one branch that has a P1 and a P2. Returns None when
    the branch has no second product."""
    import mepd.chainhelpers as ch
    from mepd.discovery import vri
    from mepd.inputs import ChainInputs

    cand_dir = Path(cand_dir)
    summary = json.loads((cand_dir / "summary.json").read_text())
    scan = json.loads((cand_dir / "projected_freqs.json").read_text())
    b = (summary.get("branches") or {}).get(branch) or {}
    pr = b.get("products") or {}
    p1_fp, p2_fp = cand_dir / f"p1_{branch}.xyz", cand_dir / f"p2_{branch}.xyz"
    if not (p1_fp.exists() and p2_fp.exists()):
        return None
    load1 = lambda fp: _read_chain(fp, charge, multiplicity)[-1]
    p1, p2 = load1(p1_fp), load1(p2_fp)
    bonds_a, bonds_b = product_bond_sets(p1, p2)
    if not bonds_a or not bonds_b:
        return None

    irc = _read_chain(cand_dir / "irc.xyz", charge, multiplicity)
    ts_index = int(scan["ts_index"])
    ts1 = irc[ts_index] if irc else load1(cand_dir / "ts1.xyz")
    e_ts1 = float(summary["ts1_energy"])
    ts2 = load1(cand_dir / f"ts2_{branch}.xyz") if (cand_dir / f"ts2_{branch}.xyz").exists() else None
    vrt = load1(cand_dir / f"vrt_{branch}.xyz") if (cand_dir / f"vrt_{branch}.xyz").exists() else None

    # TS2's IRC (label recorded in the checks; inferred for older outputs)
    checks = pr.get("ts2_checks") or {}
    label = _legacy_ts2_label(pr, branch)
    ts2_irc = _read_chain(cand_dir / f"{label}_irc.xyz", charge, multiplicity) if label else []

    # Fill paths: geodesic TS1 -> TS2 and TS1 -> P2 (cover the ridge and the P2 side)
    def geodesic(a, c, n=14):
        try:
            return list(ch.run_geodesic(chain=[a.copy(), c.copy()], chain_inputs=ChainInputs(), nimages=n).nodes)[1:-1]
        except Exception as exc:
            logger.info("geodesic fill failed: %s", exc)
            return []

    # TS1's reaction mode (imaginary eigenvector), oriented so products are at x > 0
    from mepd.discovery.vri import compute_cartesian_hessian, node_masses, projected_frequencies

    H_ts1 = compute_cartesian_hessian(_plain(ts1), engine)
    mode = np.asarray(projected_frequencies(H_ts1, ts1.coords, node_masses(ts1), gradient=None).lowest_mode_cart).reshape(-1)
    if float(mode @ (np.asarray(p1.coords).reshape(-1) - np.asarray(ts1.coords).reshape(-1))) < 0:
        mode = -mode
    axes = Axes(ts1.coords, mode, bonds_a, bonds_b)

    fill_ts2 = geodesic(ts1, ts2) if ts2 is not None else []
    fill_p2 = geodesic(ts1, p2)

    # Collect points, each with a role for display
    points: List[tuple] = []  # (node, role)
    step = max(1, len(irc) // 80) if irc else 1
    points += [(n, "IRC through TS1") for n in irc[::step]]
    points += [(n, "IRC through TS2") for n in ts2_irc[:: max(1, len(ts2_irc) // 60)]]
    points += [(n, "TS1 -> TS2 path") for n in fill_ts2]
    points += [(n, "TS1 -> P2 path") for n in fill_p2]
    marks = {"TS1": ts1, "VRT": vrt, "TS2": ts2, "P1": p1, "P2": p2}
    for name, node in marks.items():
        if node is not None:
            points.append((node, name))

    nodes = [_plain(n) for n, _ in points]
    need = [i for i, n in enumerate(nodes) if n._cached_gradient is None or n._cached_energy is None]
    if need:
        vri._ensure_energy_gradient([nodes[i] for i in need], engine)

    # Hessians: all stationary/marked points + an even spread of the rest
    marked = [i for i, (_, role) in enumerate(points) if role in marks]
    others = [i for i in range(len(points)) if i not in marked]
    extra = [others[k] for k in np.linspace(0, len(others) - 1, max(0, n_hessians - len(marked))).round().astype(int)] if others else []
    hess_idx = sorted(set(marked) | set(extra))
    hessians = dict(zip(hess_idx, vri.compute_hessians([nodes[i] for i in hess_idx], engine, workers)))

    data = []
    for i, (node, (orig, role)) in enumerate(zip(nodes, points)):
        q, B = q_and_B(node.coords, axes)
        g_q = project_gradient(B, node._cached_gradient)
        K = relaxed_curvature(B, hessians[i], node.coords) if i in hessians else None  # for `interpolate` cross-checks
        data.append({
            "q": q.tolist(), "e": (float(node._cached_energy) - e_ts1) * H2K,
            "g": g_q.tolist(), "K": None if K is None else K.tolist(), "role": role,
            "xyz": orig.structure.to_xyz(),
        })

    # Grid over the region of the stationary points (+ padding), relaxed scan
    key = np.array([d["q"] for d in data if d["role"] in marks])
    irc_q = np.array([d["q"] for d in data if d["role"] == "IRC through TS1"])
    lo, hi = key.min(0), key.max(0)
    # show some of the reactant side (x < 0) so TS1 reads as a pass
    if len(irc_q):
        lo[0] = min(lo[0], max(irc_q[:, 0].min(), -0.6 * (hi[0] - lo[0]) - 0.3))
    pad = 0.12 * (hi - lo) + 0.08
    xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid)
    ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid)
    # Bonds that change anywhere among reactant, P1 and P2 but are not on the
    # axes: holding them at the computed paths' values keeps each grid node on
    # this reaction instead of letting it relax into a product through them.
    from mepd.discovery.vri import labeled_bond_set

    other = "reverse" if branch == "forward" else "forward"
    reactant = None
    if irc:
        reactant = irc[0] if branch == "forward" else irc[-1]
    changing = set()
    if reactant is not None:
        br = labeled_bond_set(reactant)
        b1, b2 = labeled_bond_set(p1), labeled_bond_set(p2)
        changing = (br ^ b1) | (br ^ b2) | (b1 ^ b2)
    held = sorted(changing - set(bonds_a) - set(bonds_b))
    scan_grid = relaxed_grid(nodes, axes, engine, xs, ys, e_ref=e_ts1, workers=workers, held_bonds=held)
    E = scan_grid["E"]

    curves = {
        "IRC through TS1": [d["q"] for d in data if d["role"] == "IRC through TS1"],
        "IRC through TS2": [d["q"] for d in data if d["role"] == "IRC through TS2"],
    }
    return {
        "branch": branch,
        "axes": {
            "x": "progress through TS1: displacement along its reaction mode (Å); reactant ←  → products",
            "y": "which product (Å): mean " + ", ".join(f"{p1.symbols[i]}{i}–{p1.symbols[j]}{j}" for i, j in bonds_b)
                 + " (P2's bonds) minus mean " + ", ".join(f"{p1.symbols[i]}{i}–{p1.symbols[j]}{j}" for i, j in bonds_a)
                 + " (P1's bonds); P1 up, P2 down",
        },
        "x": xs.tolist(), "y": ys.tolist(),
        "E": [[None if not np.isfinite(v) else float(v) for v in row] for row in E],
        "ok": scan_grid["ok"].tolist(),
        "grid_xyz": scan_grid["xyz"],
        "grid_reached": scan_grid["reached"],
        "points": [{k: d[k] for k in ("q", "e", "role", "xyz")} | {"has_hessian": d["K"] is not None} for d in data],
        "curves": curves,
        "method": "partially relaxed 2D scan: both axes restrained to the grid node, the other reactive "
                  "bonds held at values blended from the nearest computed structures, everything else optimized",
        "held_bonds": [f"{p1.symbols[i]}{i}–{p1.symbols[j]}{j}" for i, j in held],
        "n_points": len(data), "n_hessians": sum(1 for d in data if d["K"] is not None),
    }


def relaxed_grid(
    seeds: list,
    axes: "Axes",
    engine,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    e_ref: float,
    k_restraint: float = 3000.0,
    maxiter: int = 400,
    workers: int = 1,
    held_bonds: Sequence = (),
    n_blend: int = 4,
    max_move: float = 0.4,
) -> dict:
    """Restrained optimizations on the (x, y) grid.

    `seeds`: nodes with cached coordinates to start from (each grid node
    starts from the seed nearest in (x, y)). `k_restraint` in kcal/mol/Å^2.
    Returns energies (kcal/mol vs `e_ref`, restraint excluded), the reached
    (x, y), xyz text of each relaxed structure, and a success mask."""
    from concurrent.futures import ThreadPoolExecutor

    from scipy.optimize import minimize

    seed_q = np.array([q_and_B(n.coords, axes)[0] for n in seeds])
    k_h = k_restraint / H2K  # Eh per Angstrom^2
    held = [tuple(b) for b in held_bonds]

    def lengths(x):
        X = np.asarray(x, dtype=float).reshape(-1, 3)
        return np.array([np.linalg.norm(X[i] - X[j]) * B2A for i, j in held])

    seed_held = np.array([lengths(n.coords) for n in seeds]) if held else None


    def relax(target):
        d2 = np.sum((seed_q - target) ** 2, axis=1)
        start = seeds[int(np.argmin(d2))]
        base = _plain(start)
        shape = np.asarray(base.coords).shape
        near = np.argsort(d2)[:n_blend]
        w = 1.0 / (d2[near] + 1e-4)
        if held:
            held_target = (w[:, None] * seed_held[near]).sum(0) / w.sum()


        def fun(x):
            node = base.update_coords(x.reshape(shape))
            g = np.asarray(engine.compute_gradients([node])[0], dtype=float).reshape(-1)
            e = float(engine.compute_energies([node])[0])
            q, B = q_and_B(x, axes)
            dq = q - target
            e_tot, g_tot = e + k_h * float(dq @ dq), g + 2.0 * k_h * (B.T @ dq)
            if held:
                X = x.reshape(-1, 3)
                for (i, j), t in zip(held, held_target):
                    v = X[i] - X[j]
                    d = np.linalg.norm(v) * B2A
                    e_tot += k_h * (d - t) ** 2
                    f = 2.0 * k_h * (d - t) * B2A * v / np.linalg.norm(v)
                    g_tot[3 * i:3 * i + 3] += f
                    g_tot[3 * j:3 * j + 3] -= f
            return e_tot, g_tot

        try:
            x0 = np.asarray(base.coords, float).reshape(-1)
            # Keep it local: stiff restraints make L-BFGS take huge first
            # steps, and a jump can land in another basin that satisfies the
            # same few restraints (seen at TS1: 22 kcal/mol too low).
            box = max_move / B2A
            res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=list(zip(x0 - box, x0 + box)),
                           options={"maxiter": maxiter, "gtol": 2e-4})
            x = res.x
            node = base.update_coords(x.reshape(shape))
            e = float(engine.compute_energies([node])[0])
            q, _ = q_and_B(x, axes)
            return (e - e_ref) * H2K, q.tolist(), node.structure.to_xyz(), bool(np.all(np.abs(q - target) < 0.05))
        except Exception as exc:
            logger.info("relaxed grid node failed: %s", exc)
            return None, None, None, False

    targets = [np.array([x, y]) for y in ys for x in xs]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = list(pool.map(relax, targets))
    ny, nx = len(ys), len(xs)
    E = np.full((ny, nx), np.nan)
    ok = np.zeros((ny, nx), dtype=bool)
    reached, xyz = [], []
    for idx, (e, q, xy, good) in enumerate(results):
        r, c = divmod(idx, nx)
        if e is not None:
            E[r, c] = e
            ok[r, c] = good
        reached.append(q)
        xyz.append(xy)
    return {"E": E, "ok": ok, "reached": reached, "xyz": xyz}


def interpolate(data: list, xs: np.ndarray, ys: np.ndarray, *, power: float = 4.0, eps: float = 1e-6,
                trust: Optional[float] = None) -> np.ndarray:
    """Modified Shepard interpolation on a grid (kcal/mol vs TS1). With
    `trust`, each point's gradient/curvature terms are damped beyond that
    distance (Gaussian), so a far-away point contributes its energy, not an
    extrapolated quadratic."""
    Q = np.array([d["q"] for d in data])
    Ei = np.array([d["e"] for d in data])
    G = np.array([d["g"] for d in data])
    Ks = [None if d["K"] is None else np.array(d["K"]) for d in data]
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    num = np.zeros(len(pts))
    den = np.zeros(len(pts))
    for i in range(len(data)):
        dq = pts - Q[i]
        r2 = np.sum(dq * dq, axis=1) + eps
        w = 1.0 / r2 ** (power / 2)
        corr = dq @ G[i]
        if Ks[i] is not None:
            corr = corr + 0.5 * np.einsum("ni,ij,nj->n", dq, Ks[i], dq)
        if trust:
            corr = corr * np.exp(-r2 / trust ** 2)
        T = Ei[i] + corr
        num += w * T
        den += w
    return (num / den).reshape(len(ys), len(xs))


def build_surfaces(vri_dir: Path, engine, *, charge: int = 0, multiplicity: int = 1, workers: int = 1,
                   grid: int = 13, on_log=print) -> list:
    """Surfaces for every branch with a P1 and P2, in every TS1 candidate of
    a VRI output directory; writes surface_<branch>.json next to the data."""
    vri_dir = Path(vri_dir)
    top = json.loads((vri_dir / "summary.json").read_text())
    written = []
    for cand in top.get("ts1_candidates") or [{"label": "input", "dir": str(vri_dir)}]:
        d = candidate_dir(vri_dir, cand)
        if not (d / "summary.json").exists():
            continue
        for branch in (json.loads((d / "summary.json").read_text()).get("branches") or {}):
            on_log(f"[{cand['label']}] {branch} branch: building surface...")
            try:
                surf = build_branch_surface(d, branch, engine, charge=charge, multiplicity=multiplicity,
                                            workers=workers, grid=grid)
            except Exception as exc:
                on_log(f"  failed: {type(exc).__name__}: {exc}")
                continue
            if surf is None:
                on_log("  no second product on this branch; skipped")
                continue
            fp = d / f"surface_{branch}.json"
            fp.write_text(json.dumps(surf))
            written.append(fp)
            n_ok = sum(sum(r) for r in surf["ok"])
            on_log(f"  relaxed grid {len(surf['x'])}x{len(surf['y'])} ({n_ok} nodes reached their target); "
                   f"{surf['n_points']} computed points overlaid -> {fp}")
    return written


# --------------------------------------------------------------------------
# Instant map from saved data (no engine calls)
# --------------------------------------------------------------------------


def _legacy_ts2_label(pr: dict, branch: str):
    """TS2 file label: recorded in newer outputs; older outputs written while
    product enumeration existed stored an enumeration-found TS2 under
    ts2_<branch>_enum<k> (read-only compatibility)."""
    label = (pr.get("ts2_checks") or {}).get("label")
    if label:
        return label
    if pr.get("ts2_energy") is None:
        return None
    attempts = pr.get("enumeration") or []
    if pr.get("evidence") == "enumeration" and attempts:
        return f"ts2_{branch}_enum{len(attempts) - 1}"
    return f"ts2_{branch}"


def _xyz_frames(fp: Path):
    """(symbols, [coords in bohr]) of a multi-frame xyz file, parsed directly
    (no molecular graphs: building one per frame is what makes loading slow)."""
    lines = fp.read_text().splitlines()
    frames, i, symbols = [], 0, None
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        rows = [r.split() for r in lines[i + 2:i + 2 + n]]
        symbols = [r[0] for r in rows]
        frames.append(np.array([[float(v) for v in r[1:4]] for r in rows]) / B2A)
        i += n + 2
    return symbols, frames


def _sidecar(fp: Path, suffix: str, n: int):
    side = fp.with_suffix(suffix)
    if not side.exists():
        return None
    try:
        arr = np.loadtxt(side, dtype=float)
    except Exception:
        return None
    arr = np.atleast_1d(arr) if suffix == ".energies" else np.atleast_2d(arr)
    if suffix == ".gradients":
        if arr.shape[0] != n:  # stored as one flattened row per frame, or per-atom rows
            arr = arr.reshape(n, -1)
    return arr if len(arr) == n else None


def _bond_set_from_xyz(symbols, coords_bohr, charge: int, multiplicity: int):
    from qcdata import Structure

    from mepd.discovery.vri import labeled_bond_set
    from mepd.nodes.node import StructureNode

    node = StructureNode(structure=Structure(symbols=symbols, geometry=coords_bohr, charge=charge,
                                             multiplicity=multiplicity))
    return labeled_bond_set(node)


def candidate_dir(vri_output, entry: dict) -> Path:
    """Directory of one TS1 candidate listed in a VRI summary: the output
    itself for the input TS, `<output>/<label>` for the others. Resolved
    from the directory given rather than the absolute path recorded at run
    time, so a copied or moved output reads and writes its own files."""
    vri_output = Path(vri_output)
    d = vri_output if entry.get("label", "input") == "input" else vri_output / entry["label"]
    return d if d.exists() or "dir" not in entry else Path(entry["dir"])


def quick_surface(
    cand_dir: Path,
    branch: str,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    grid: int = 90,
    power: float = 4.0,
    trust: float = 0.3,
) -> Optional[dict]:
    """The branch's map from data already on disk, in well under a second:
    modified Shepard interpolation (`interpolate`) of energies and projected
    gradients at every IRC / TS2-IRC frame and at P1, P2, TS2 and the VRT,
    plus relaxed curvatures from `hessians_<branch>.npz` when the search
    saved them. Same axes as `build_branch_surface`, except that TS1's
    reaction mode is taken as the IRC tangent at TS1 (the IRC leaves TS1
    along it). Grid cells farther than `trust` (Angstrom, in map
    coordinates) from every data point are left empty rather than
    extrapolated."""
    cand_dir = Path(cand_dir)
    summary = json.loads((cand_dir / "summary.json").read_text())
    scan = json.loads((cand_dir / "projected_freqs.json").read_text())
    pr = ((summary.get("branches") or {}).get(branch) or {}).get("products") or {}
    p1_fp, p2_fp, irc_fp = (cand_dir / f"{k}.xyz" for k in (f"p1_{branch}", f"p2_{branch}", "irc"))
    if not (p1_fp.exists() and p2_fp.exists() and irc_fp.exists()):
        return None
    e_ts1 = float(summary["ts1_energy"])
    symbols, irc = _xyz_frames(irc_fp)
    ts_index = int(scan["ts_index"])
    p1 = _xyz_frames(p1_fp)[1][-1]
    p2 = _xyz_frames(p2_fp)[1][-1]
    b1 = _bond_set_from_xyz(symbols, p1, charge, multiplicity)
    b2 = _bond_set_from_xyz(symbols, p2, charge, multiplicity)
    bonds_a, bonds_b = sorted(b1 - b2), sorted(b2 - b1)
    if not bonds_a or not bonds_b:
        return None

    # One frame of reference for everything: each structure is aligned onto TS1
    # before x is computed (x is a projection, so it depends on orientation).
    ref = irc[ts_index]
    irc = [align_to(X, ref)[0] for X in irc]
    p1 = align_to(p1, ref)[0]
    x_ts1 = irc[ts_index].reshape(-1)
    lo, hi = max(0, ts_index - 2), min(len(irc) - 1, ts_index + 2)
    mode = (irc[hi] - irc[lo]).reshape(-1)
    if float(mode @ (p1.reshape(-1) - x_ts1)) < 0:
        mode = -mode
    axes = Axes(x_ts1, mode, bonds_a, bonds_b)

    data, overlays = [], []

    def add_sequence(fp: Path, role: str, hessians: Optional[dict] = None):
        if not fp.exists():
            return
        _, frames = _xyz_frames(fp)
        energies = _sidecar(fp, ".energies", len(frames))
        grads = _sidecar(fp, ".gradients", len(frames))
        if energies is None:
            return
        for k, X in enumerate(frames):
            X, Rot = align_to(X, ref)
            if grads is not None:
                grads = grads.copy()
                grads[k] = (np.asarray(grads[k]).reshape(-1, 3) @ Rot).reshape(-1)
            q, B = axes(X)
            tag = {"frame": k} if role == "IRC through TS1" else {"xyz": _to_xyz(symbols, X)}
            overlays.append({"q": q.tolist(), "e": (float(energies[k]) - e_ts1) * H2K, "role": role, **tag})
            g = project_gradient(B, grads[k]) if grads is not None else np.zeros(2)
            K = None
            if hessians is not None and k in hessians:
                Bd = np.kron(np.eye(len(X)), Rot)  # rotate the Hessian into TS1's frame too
                K = relaxed_curvature(B, Bd.T @ hessians[k] @ Bd, X)
            data.append({"q": q.tolist(), "e": overlays[-1]["e"], "g": g.tolist(),
                         "K": None if K is None else K.tolist()})

    # Saved Hessians of this branch, keyed by global IRC frame index
    hess_by_frame = None
    npz = cand_dir / f"hessians_{branch}.npz"
    if npz.exists():
        z = np.load(npz)
        n = len(irc)
        hess_by_frame = {}
        for ci, is_bis, H in zip(z["chain_index"], z["bisection"], z["hessian"]):
            if is_bis:
                continue
            frame = ts_index + int(ci) if branch == "forward" else ts_index - int(ci)
            if 0 <= frame < n:
                hess_by_frame[frame] = np.asarray(H, dtype=float)
    add_sequence(irc_fp, "IRC through TS1", hess_by_frame)

    checks = pr.get("ts2_checks") or {}
    label = _legacy_ts2_label(pr, branch)
    if label:
        add_sequence(cand_dir / f"{label}_irc.xyz", "IRC through TS2")
        if checks.get("method") == "neb":  # a geodesic guess is unrelaxed: its energies would be inflated
            add_sequence(cand_dir / f"{label}_path.xyz", "P1 → P2 path")
    # The ridge check's TS1 -> TS2 path: saved by newer runs; otherwise its
    # geometries are regenerated (deterministic geodesic interpolation, no QC)
    # and paired with the energies the check recorded.
    ridge = checks.get("ridge") or {}
    ridge_fp = cand_dir / f"{label}_ridge.xyz" if label else None
    if ridge_fp is not None and ridge_fp.exists():
        add_sequence(ridge_fp, "TS1 → TS2 path")
    elif ridge.get("profile_kcal") and (cand_dir / f"ts2_{branch}.xyz").exists():
        try:
            prof = ridge["profile_kcal"]
            ts2_X = _xyz_frames(cand_dir / f"ts2_{branch}.xyz")[1][-1]
            path = [align_to(X, ref)[0] for X in _geodesic_frames(symbols, irc[ts_index], ts2_X, len(prof), charge, multiplicity)]
            for X, e in zip(path, prof):
                q, _ = axes(X)
                overlays.append({"q": q.tolist(), "e": float(e), "role": "TS1 → TS2 path", "xyz": _to_xyz(symbols, X)})
                data.append({"q": q.tolist(), "e": float(e), "g": [0.0, 0.0], "K": None})
        except Exception as exc:
            logger.info("ridge path regeneration failed: %s", exc)

    marks = []
    irc_e = _sidecar(irc_fp, ".energies", len(irc))
    r_frame = 0 if branch == "forward" else len(irc) - 1
    if irc_e is not None:
        marks.append({"q": axes(irc[r_frame])[0].tolist(), "e": (float(irc_e[r_frame]) - e_ts1) * H2K,
                      "role": "R", "mark": True, "xyz": _to_xyz(symbols, irc[r_frame])})
    for role, fp in (("TS1", None), ("VRT", cand_dir / f"vrt_{branch}.xyz"), ("VRI", cand_dir / f"vri_{branch}.xyz"), ("TS2", cand_dir / f"ts2_{branch}.xyz"),
                     ("P1", p1_fp), ("P2", p2_fp)):
        if role == "TS1":
            X, e = irc[ts_index], 0.0
            xyz = None
        else:
            if not fp.exists():
                continue
            _, frames = _xyz_frames(fp)
            energies = _sidecar(fp, ".energies", len(frames))
            if energies is None:
                continue
            X, Rot = align_to(frames[-1], ref)
            e = (float(energies[-1]) - e_ts1) * H2K
            grads = _sidecar(fp, ".gradients", len(frames))
            g = (np.asarray(grads[-1]).reshape(-1, 3) @ Rot).reshape(-1) if grads is not None else None
            q, B = axes(X)
            data.append({"q": q.tolist(), "e": e, "g": (project_gradient(B, g) if g is not None else np.zeros(2)).tolist(), "K": None})
        marks.append({"q": axes(X)[0].tolist(), "e": e, "role": role, "mark": True,
                      "xyz": _to_xyz(symbols, X)})

    check_paths = _check_paths(cand_dir / f"paths_{branch}.npz", ref, axes, e_ts1, data)

    # The ridge-mode animation at the VRT: map positions only (no energies were
    # computed for these displaced frames, so they are not interpolation data).
    ridge_fp = cand_dir / f"ridge_mode_{branch}.xyz"
    if ridge_fp.exists():
        for X in _xyz_frames(ridge_fp)[1]:
            X = align_to(X, ref)[0]
            overlays.append({"q": axes(X)[0].tolist(), "e": None, "role": "ridge mode", "xyz": _to_xyz(symbols, X)})

    if not data:
        return None
    Q = np.array([d["q"] for d in data])
    # Bounds: the paths above plus where the basin descents go (they end in the
    # products or back at R); trajectories are left to clip, since hot ones can
    # wander far from the region of interest.
    allq = np.array([o["q"] for o in overlays] + [m["q"] for m in marks]
                    + [q for cp in check_paths if cp["kind"] == "basin" for q in cp["q"]])
    lo_q, hi_q = allq.min(0), allq.max(0)
    pad = 0.06 * (hi_q - lo_q) + 0.08
    xs = np.linspace(lo_q[0] - pad[0], hi_q[0] + pad[0], grid)
    ys = np.linspace(lo_q[1] - pad[1], hi_q[1] + pad[1], grid)
    E = interpolate(data, xs, ys, power=power, trust=trust)
    # distance to the nearest data point decides what is shown
    X, Y = np.meshgrid(xs, ys)
    d2 = ((X.ravel()[:, None] - Q[None, :, 0]) ** 2 + (Y.ravel()[:, None] - Q[None, :, 1]) ** 2).min(1).reshape(E.shape)
    E = np.where(d2 <= trust ** 2, E, np.nan)
    fmt_bonds = lambda bonds: ", ".join(f"{symbols[i]}{i}–{symbols[j]}{j}" for i, j in bonds)
    return {
        "branch": branch,
        "axes": {"x": "progress through TS1 (Å)", "y": f"mean {fmt_bonds(bonds_b)} minus mean {fmt_bonds(bonds_a)} (Å)",
                 "bonds_p1": [f"{symbols[i]}{i}–{symbols[j]}{j}" for i, j in bonds_a],
                 "bonds_p2": [f"{symbols[i]}{i}–{symbols[j]}{j}" for i, j in bonds_b]},
        "x": xs.tolist(), "y": ys.tolist(),
        "E": [[None if not np.isfinite(v) else float(v) for v in row] for row in E],
        "points": overlays + marks,
        "symbols": list(symbols),
        "check_paths": check_paths,
        "method": ("interpolated from saved data (energies, gradients"
                   + (", Hessians" if hess_by_frame else "")
                   + (", basin-test descents" if any(cp["kind"] == "basin" for cp in check_paths) else "") + "); blank where no data is within "
                   + f"{trust} Å"),
        "n_points": len(data), "n_hessians": sum(1 for d in data if d["K"] is not None),
        "instant": True,
    }


def _check_paths(fp: Path, ref: np.ndarray, axes: "Axes", e_ts1: float, data: list) -> list:
    """Basin-test descents and trajectories saved by the checks
    (`vri_checks.save_paths`), in map coordinates. Basin-descent frames with
    an energy and gradient are also added to `data`, so the map covers where
    the descents went; trajectory frames are not (they are vibrationally
    hot, far above the relaxed surface at the same map position). Per path: kind, outcome label, info, and per frame q, energy
    (kcal/mol vs TS1, None if not computed) and coordinates (Angstrom,
    aligned to TS1, flattened)."""
    if not fp.exists():
        return []
    out = []
    try:
        z = np.load(fp)
    except Exception as exc:
        logger.info("could not read %s: %s", fp, exc)
        return []
    with z:
        for kind in ("basin", "trajectory"):
            if f"{kind}_coords" not in z.files:
                continue
            coords, energies, grads = z[f"{kind}_coords"], z[f"{kind}_energies"], z[f"{kind}_gradients"]
            offsets, labels, info = z[f"{kind}_offsets"], z[f"{kind}_labels"], z[f"{kind}_info"]
            for p in range(len(offsets) - 1):
                a, b = int(offsets[p]), int(offsets[p + 1])
                if b <= a:
                    continue
                qs, es, xs = [], [], []
                for k in range(a, b):
                    X, Rot = align_to(np.asarray(coords[k], dtype=float).reshape(-1, 3), ref)
                    q, B = axes(X)
                    e = None if not np.isfinite(energies[k]) else (float(energies[k]) - e_ts1) * H2K
                    qs.append(q.tolist())
                    es.append(e)
                    xs.append(np.round(X.reshape(-1) * B2A, 3).tolist())
                    g = np.asarray(grads[k], dtype=float)
                    if kind == "basin" and e is not None and np.all(np.isfinite(g)):
                        g = (g.reshape(-1, 3) @ Rot).reshape(-1)
                        data.append({"q": q.tolist(), "e": e, "g": project_gradient(B, g).tolist(), "K": None})
                out.append({"kind": kind, "label": str(labels[p]), "info": json.loads(str(info[p])),
                            "q": qs, "e": es, "X": xs})
    return out


def _geodesic_frames(symbols, a_bohr, b_bohr, n_images: int, charge: int, multiplicity: int) -> list:
    """The same geodesic interpolation `ridge_connected` uses (default
    ChainInputs, `n_images` frames including both ends), as bohr arrays."""
    from qcdata import Structure

    import mepd.chainhelpers as ch
    from mepd.inputs import ChainInputs
    from mepd.nodes.node import StructureNode

    mk = lambda X: StructureNode(structure=Structure(symbols=symbols, geometry=X, charge=charge, multiplicity=multiplicity),
                                 has_molecular_graph=False)
    chain = ch.run_geodesic(chain=[mk(a_bohr), mk(b_bohr)], chain_inputs=ChainInputs(), nimages=n_images)
    return [np.asarray(n.coords) for n in chain.nodes]


def _to_xyz(symbols, coords_bohr) -> str:
    X = np.asarray(coords_bohr).reshape(-1, 3) * B2A
    return f"{len(symbols)}\n\n" + "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, X)) + "\n"
