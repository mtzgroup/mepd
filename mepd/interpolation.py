"""Initial guess paths between two structures.

`chain_inputs.interpolation` picks how the first path of a search (and every
sub-path a recursive split creates) is built:

- ``geodesic`` (default): geodesic interpolation in redundant internal
  coordinates (`mepd.chainhelpers.run_geodesic`, tuned by ``[gi_inputs]``).
- ``linear``: straight lines in Cartesian space, after rigidly aligning the
  end structure onto the start.
- ``lst``: linear synchronous transit (Halgren & Lipscomb, Chem. Phys. Lett.
  49, 225 (1977)). Every interatomic distance is interpolated linearly, and
  each image independently takes the geometry that best matches its target
  distances (weights 1/d^4, so short distances such as bonds dominate),
  starting from the linear image.
- ``idpp``: image-dependent pair potential (Smidstrup, Pedersen, Stokbro &
  Jónsson, J. Chem. Phys. 140, 214106 (2014)). The same target distances
  as LST, but all images are relaxed together as an NEB on that surface
  (projected forces plus springs along the path), which keeps them evenly
  spaced and the path smooth.

Atoms in ``chain_inputs.frozen_atom_indices`` follow the linear path in every
method (and alignment is skipped, so they don't move).
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

METHODS = ("geodesic", "linear", "lst", "idpp")


def interpolation_method(chain_inputs) -> str:
    method = str(getattr(chain_inputs, "interpolation", "geodesic") or "geodesic").strip().lower()
    if method not in METHODS:
        raise ValueError(f"unknown interpolation {method!r}; choose one of {', '.join(METHODS)}")
    return method


# ------------------------------------------------------------ pair surface

def _pair_terms(x: np.ndarray, target: np.ndarray, weight: np.ndarray):
    """S = sum_{i<j} w_ij (d_ij - D_ij)^2 and dS/dx for one geometry x (N x 3)."""
    diff = x[:, None, :] - x[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(d, 1.0)
    r = d - target
    np.fill_diagonal(r, 0.0)
    value = 0.5 * float(np.sum(weight * r * r))
    # Two atoms on the same point (e.g. the linear midpoint of a 180-degree
    # rotation) have no direction to push apart along: floor the distance.
    coef = 2.0 * weight * r / np.maximum(d, 1e-3)
    np.fill_diagonal(coef, 0.0)
    grad = coef.sum(axis=1)[:, None] * x - coef @ x
    return value, grad


def _targets(xa: np.ndarray, xb: np.ndarray, t: float):
    da = np.linalg.norm(xa[:, None, :] - xa[None, :, :], axis=-1)
    db = np.linalg.norm(xb[:, None, :] - xb[None, :, :], axis=-1)
    target = (1.0 - t) * da + t * db
    # Floor target distances (atoms that coincide in an endpoint would get an
    # infinite weight): 0.3 bohr is far below any real interatomic distance.
    safe = np.maximum(target, 0.3)
    np.fill_diagonal(safe, 1.0)
    weight = 1.0 / safe ** 4
    np.fill_diagonal(weight, 0.0)
    return target, weight


def linear_path(xa, xb, nimages: int) -> np.ndarray:
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    _check_endpoints(xa, xb, nimages)
    ts = np.linspace(0.0, 1.0, int(nimages))
    return np.array([(1.0 - t) * xa + t * xb for t in ts])


def _start_path(xa, xb, nimages: int, frozen) -> np.ndarray:
    """The linear path, with a tiny fixed nudge on interior images whose
    atoms collide exactly (a symmetric start cannot break the tie itself)."""
    path = linear_path(xa, xb, nimages)
    mask = _free_mask(len(xa), frozen)
    rng = np.random.default_rng(0)
    for k in range(1, len(path) - 1):
        d = np.linalg.norm(path[k][:, None] - path[k][None], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.05:
            path[k][mask] += rng.normal(scale=0.02, size=path[k][mask].shape)
    return path


def lst_path(xa, xb, nimages: int, frozen: Optional[np.ndarray] = None, maxiter: int = 400,
             substeps: int = 1) -> np.ndarray:
    """Linear synchronous transit: each interior image fitted to its
    linearly interpolated interatomic distances. The fits march from the
    start to the end in `substeps` small steps per image, each starting from
    the last, so the path stays in one basin instead of jumping between
    alternative fits of the same distances."""
    from scipy.optimize import minimize

    from mepd.rigid_alignment import kabsch_align

    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    _check_endpoints(xa, xb, nimages)
    n = int(nimages)
    path = _start_path(xa, xb, n, frozen)
    mask = _free_mask(len(xa), frozen)
    if n <= 2:
        return path
    fine = np.linspace(0.0, 1.0, (n - 1) * max(1, int(substeps)) + 1)
    x = xa.copy()
    t_prev = 0.0
    for t in fine[1:-1]:
        target, weight = _targets(xa, xb, t)
        x_lin = (1.0 - t) * xa + t * xb
        x0 = x + (t - t_prev) * (xb - xa)

        def f(v, target=target, weight=weight, x_lin=x_lin):
            y = v.reshape(-1, 3)
            value, grad = _pair_terms(y, target, weight)
            # A weak pull toward the linear point fixes the rigid-body freedom.
            value += 1e-6 * float(np.sum((y - x_lin) ** 2))
            grad = grad + 2e-6 * (y - x_lin)
            grad[~mask] = 0.0
            return value, grad.ravel()

        res = minimize(f, x0.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        x = res.x.reshape(-1, 3)
        if mask.all():
            x = kabsch_align(x, x_lin)      # distances fix the shape, not the orientation
        else:
            x[~mask] = x_lin[~mask]
        t_prev = t
        k = t * (n - 1)
        if abs(k - round(k)) < 1e-9:
            path[int(round(k))] = x
    return path


def idpp_path(xa, xb, nimages: int, frozen: Optional[np.ndarray] = None, *, spring: float = 1.0,
              fmax: float = 1e-3, max_steps: int = 1000, start: str = "linear") -> np.ndarray:
    """Image-dependent pair potential: all images relaxed together as an
    NEB on the LST target-distance surface (FIRE steps).

    Rigid motion is kept out of the optimization (the pair potential can't
    see it, so images would drift and rotate): the path is relaxed with the
    end rigidly aligned onto the start, then the images are carried back by
    a smoothly interpolated rotation and translation, so the last image is
    exactly `xb` again."""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    _check_endpoints(xa, xb, nimages)
    mask = _free_mask(len(xa), frozen)
    if mask.all() and len(xa) > 1:
        rot, xb_aligned = _rigid_fit(xb, xa)
        path = _idpp_relax(xa, xb_aligned, nimages, frozen, spring=spring, fmax=fmax, max_steps=max_steps, start=start)
        return _undo_rigid_fit(path, xa, xb, rot)
    return _idpp_relax(xa, xb, nimages, frozen, spring=spring, fmax=fmax, max_steps=max_steps, start=start)


def _rigid_fit(mobile: np.ndarray, target: np.ndarray):
    """(R, mobile aligned onto target) with aligned = (mobile - c_m) @ R + c_t."""
    cm, ct = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - cm).T @ (target - ct))
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return rot, (mobile - cm) @ rot + ct


def _undo_rigid_fit(path: np.ndarray, xa: np.ndarray, xb: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Bring a path computed with the end aligned (end = (xb - c_b) @ rot + c_a)
    back to the real end: image k gets the fraction t_k of the inverse rigid
    motion (rotation about its centroid by t_k of the angle, and t_k of the
    translation)."""
    from scipy.spatial.transform import Rotation

    back = Rotation.from_matrix(rot.T)          # aligned frame -> real end's orientation
    rotvec = back.as_rotvec()
    ca, cb = xa.mean(axis=0), xb.mean(axis=0)
    out = path.copy()
    n = len(path)
    for k in range(n):
        t = k / (n - 1)
        partial = Rotation.from_rotvec(t * rotvec).as_matrix()
        c = path[k].mean(axis=0)
        out[k] = (path[k] - c) @ partial.T + c + t * (cb - ca)
    out[0], out[-1] = xa, xb
    return out


def _idpp_relax(xa, xb, nimages, frozen, *, spring, fmax, max_steps, start):
    path = lst_path(xa, xb, nimages, frozen) if start == "lst" else _start_path(xa, xb, nimages, frozen)
    n = len(path)
    if n <= 2:
        return path
    ts = np.linspace(0.0, 1.0, n)
    surfaces = [_targets(xa, xb, t) for t in ts]
    mask = _free_mask(len(xa), frozen)
    velocity = np.zeros_like(path)
    dt, dt_max, alpha, n_pos = 0.1, 1.0, 0.1, 0
    checkpoint = path.copy()
    for step_no in range(int(max_steps)):
        # Also stop once the path has effectively stopped changing: large
        # systems get there long before every force is below fmax.
        if step_no and step_no % 25 == 0:
            if np.sqrt(np.mean((path - checkpoint) ** 2)) < 1e-4:
                break
            checkpoint = path.copy()
        forces = np.zeros_like(path)
        for k in range(1, n - 1):
            _, grad = _pair_terms(path[k], *surfaces[k])
            fwd, back = path[k + 1] - path[k], path[k] - path[k - 1]
            tau = fwd / (np.linalg.norm(fwd) or 1.0) + back / (np.linalg.norm(back) or 1.0)
            tau /= np.linalg.norm(tau) or 1.0
            f_true = -grad
            f_perp = f_true - np.sum(f_true * tau) * tau
            f_spring = spring * (np.linalg.norm(fwd) - np.linalg.norm(back)) * tau
            force = f_perp + f_spring
            if mask.all():
                # The pair potential can't see rigid motion, so nothing would
                # stop images drifting and rotating relative to each other.
                force = _without_rigid_motion(force, path[k])
            forces[k] = force
        forces[:, ~mask] = 0.0
        if np.max(np.linalg.norm(forces, axis=-1)) < fmax:
            break
        # FIRE (Bitzek et al. 2006)
        power = float(np.sum(forces * velocity))
        if power > 0:
            fnorm = np.linalg.norm(forces) or 1.0
            velocity = (1 - alpha) * velocity + alpha * forces * (np.linalg.norm(velocity) / fnorm)
            n_pos += 1
            if n_pos > 5:
                dt, alpha = min(dt * 1.1, dt_max), alpha * 0.99
        else:
            velocity[:] = 0.0
            dt, alpha, n_pos = dt * 0.5, 0.1, 0
        velocity += dt * forces
        step = dt * velocity
        big = np.linalg.norm(step, axis=-1).max()
        if big > 0.2:          # bohr: never move an atom far in one step
            step *= 0.2 / big
        path[1:-1] += step[1:-1]
    idpp_path.last_steps = step_no + 1      # for diagnostics
    return path


def _without_rigid_motion(force: np.ndarray, x: np.ndarray) -> np.ndarray:
    """`force` with the components along the image's rigid translations and
    rotations removed."""
    c = x - x.mean(axis=0)
    modes = []
    for axis in np.eye(3):
        modes.append(np.tile(axis, (len(x), 1)))          # translation
        modes.append(np.cross(axis, c))                   # rotation about the centroid
    basis, r = np.linalg.qr(np.array([m.ravel() for m in modes]).T)
    basis = basis[:, np.abs(r.diagonal()) > 1e-8]         # linear/1-atom systems have fewer modes
    f = force.ravel()
    return (f - basis @ (basis.T @ f)).reshape(force.shape)


def _check_endpoints(xa: np.ndarray, xb: np.ndarray, nimages: int) -> None:
    if int(nimages) < 2:
        raise ValueError(f"an interpolated path needs at least 2 images (got {nimages})")
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(xb))):
        raise ValueError("endpoint coordinates contain NaN or infinity")


def _free_mask(natoms: int, frozen) -> np.ndarray:
    mask = np.ones(natoms, dtype=bool)
    if frozen is not None and len(frozen):
        mask[np.asarray(frozen, dtype=int)] = False
    return mask


def _frozen_indices(chain_inputs, natoms: Optional[int] = None) -> np.ndarray:
    raw = getattr(chain_inputs, "frozen_atom_indices", None)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return np.array([], dtype=int)
    try:
        if isinstance(raw, str):
            idx = np.array([int(x) for x in raw.replace(",", " ").split()], dtype=int)
        else:
            idx = np.asarray(raw, dtype=int).ravel()
    except (TypeError, ValueError):
        raise ValueError(f"frozen_atom_indices must be atom indices like \"0 3 4\", got {raw!r}") from None
    if natoms is not None and len(idx) and (idx.min() < 0 or idx.max() >= natoms):
        raise ValueError(f"frozen_atom_indices {idx.tolist()} out of range for {natoms} atoms (0-{natoms - 1})")
    return idx


# ------------------------------------------------------------ chains

def initial_chain(seed_chain, chain_inputs, gi_inputs, *, nimages: Optional[int] = None,
                  align: Optional[bool] = None, **geodesic_kwargs):
    """The first path between `seed_chain`'s two ends, built with
    `chain_inputs.interpolation` (see the module docstring). `align`
    overrides `gi_inputs.align` (False keeps both ends' coordinates exactly)."""
    from mepd import chainhelpers as ch
    from mepd.chain import Chain
    from mepd.rigid_alignment import kabsch_align

    nimages = int(gi_inputs.nimages if nimages is None else nimages)
    method = interpolation_method(chain_inputs)
    natoms = len(getattr(getattr(seed_chain[0], "structure", None), "symbols", None) or []) or None
    frozen = _frozen_indices(chain_inputs, natoms)
    if int(nimages) < 2:
        raise ValueError(f"an interpolated path needs at least 2 images (got {nimages})")
    if method == "geodesic":
        kwargs = {**(gi_inputs.extra_kwds or {}), **geodesic_kwargs}
        if len(frozen):
            kwargs.setdefault("ignore_atoms", frozen)       # hold frozen atoms, as sub-paths do
        return ch.run_geodesic(
            chain=seed_chain, chain_inputs=copy.deepcopy(chain_inputs), nimages=nimages,
            friction=gi_inputs.friction, nudge=gi_inputs.nudge, random_seed=gi_inputs.random_seed,
            align=gi_inputs.align if align is None else align, **kwargs,
        )
    start, end = seed_chain[0], seed_chain[-1]
    symbols = getattr(getattr(start, "structure", None), "symbols", None)
    if symbols is None:
        # Not a molecule (e.g. a 2-D model surface): only a straight line makes sense.
        if method != "linear":
            raise ValueError(f"{method} interpolation needs molecular structures")
        coords = linear_path(start.coords, end.coords, nimages)
    else:
        xa = np.asarray(start.coords, float)
        xb = ch._coords_reordered_to_reference_symbols(
            reference_symbols=list(start.symbols), symbols=list(end.symbols), coords=end.coords)
        _check_endpoints(xa, np.asarray(xb, float), nimages)
        do_align = getattr(gi_inputs, "align", True) if align is None else align
        if do_align and not len(frozen):
            xb = kabsch_align(xb, xa)
        if method == "linear":
            coords = linear_path(xa, xb, nimages)
        elif method == "lst":
            coords = lst_path(xa, xb, nimages, frozen)
        else:
            coords = idpp_path(xa, xb, nimages, frozen)
    # Fresh nodes throughout (endpoints included): a new path starts with no
    # cached energies or gradients, as it does from geodesic interpolation.
    nodes = [start.update_coords(c) for c in coords]
    return Chain.model_validate({"nodes": nodes, "parameters": copy.deepcopy(chain_inputs)})
