"""A live map of the paths a `mepd channels` run is relaxing: every image of
every path (one per reactant/product conformer pair and atom mapping,
recursive sub-paths included) placed in two coordinates, over an
approximate energy surface fitted to all of them -- to see at a glance how
similar or different the generated paths are.

Coordinates (`mode`):

  bonds     the pair's own reaction coordinates, the More O'Ferrall-Jencks
            way: x = how far the bonds that break have stretched, y = how
            far the bonds that form have closed, each 0 at the pair's
            reactant and 1 at its product (mean over those bonds). Every
            path gets its own bond sets -- different atom mappings break
            and form different bonds -- but the normalization puts them all
            in one square: a concerted path runs along the diagonal, a
            stepwise one along the edges (break first, or form first).
            A reaction that only forms bonds (e.g. a Diels-Alder) or only
            breaks them plots its bonds in two groups against each other
            instead: synchronous along the diagonal, asynchronous bowed
            toward an edge.
  distance  mapping-independent: x, y = how far a structure is from the
            lowest reactant / product conformer (RMS difference of sorted
            interatomic distances per element pair, in Angstrom), the same
            for every path whatever its atom numbering.

Energies are kcal/mol relative to the lowest reactant-side image found (the
lowest conformer, as barriers are measured). The surface is a Shepard
interpolation of the computed images only (no gradients in the live data),
so it is approximate between paths and left blank far from all of them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

H2K = 627.509474


def _frame(xyz: str):
    rows = [r.split() for r in xyz.strip().splitlines()[2:] if r.strip()]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def _bonds(symbols, X, scale: float = 1.25) -> set:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    r = np.array([pt.GetRcovalent(s) for s in symbols])
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    n = len(symbols)
    return {(i, j) for i in range(n) for j in range(i + 1, n) if d[i, j] < scale * (r[i] + r[j])}


def _dist(X, i, j) -> float:
    return float(np.linalg.norm(X[i] - X[j]))


def _fingerprint(symbols, X) -> np.ndarray:
    """Sorted interatomic distances, grouped by element pair: the same for
    any atom numbering (and rotation/translation)."""
    order = sorted(range(len(symbols)), key=lambda k: symbols[k])
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    groups: dict = {}
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            groups.setdefault(tuple(sorted((symbols[i], symbols[j]))), []).append(d[i, j])
    return np.concatenate([np.sort(groups[k]) for k in sorted(groups)])


def _paths(job_dir: Path) -> list[dict]:
    """Every path in the run's live streams: {stream, monitor, frames (xyz),
    energies (Hartree, absolute when known), active, finished}."""
    out = []
    for fp in sorted((job_dir / "live").glob("*.json")):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        monitors = data.get("monitors") or {}
        if not monitors and data.get("geometry"):
            monitors = {"main": data}
        for mid, m in monitors.items():
            frames = ((m.get("geometry") or {}).get("frames")) or []
            plot = m.get("plot") or {}
            y = plot.get("y") or []
            if len(frames) < 2 or len(y) != len(frames):
                continue
            ref = plot.get("energy_ref_hartree")
            energies = [None if v is None else (v / H2K + (ref if ref is not None else 0.0)) for v in y]
            out.append({"stream": fp.stem, "monitor": mid, "frames": frames, "energies": energies,
                        "absolute": ref is not None, "active": bool(m.get("active")),
                        "finished": bool(data.get("finished")), "caption": plot.get("caption") or ""})
    return out


def _surface(Q: np.ndarray, E: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """A smooth surface through the images: a thin-plate-spline radial basis
    fit with a little smoothing (images of one path sit close together and
    would otherwise make it ripple), falling back to inverse-distance
    weighting if the fit is singular."""
    from scipy.interpolate import RBFInterpolator

    Xg, Yg = np.meshgrid(xs, ys)
    grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    span = np.maximum(Q.max(axis=0) - Q.min(axis=0), 1e-6)
    try:
        f = RBFInterpolator(Q / span, E, kernel="thin_plate_spline", smoothing=1e-3 * len(E))
        return f(grid_pts / span).reshape(Xg.shape)
    except Exception:
        d = np.linalg.norm((grid_pts[:, None] - Q[None]) / span, axis=-1) + 1e-9
        w = 1.0 / d ** 2
        return ((w @ E) / w.sum(axis=1)).reshape(Xg.shape)


def channels_map(job_dir: Path, mode: str = "bonds", grid: int = 64, running: bool = True) -> dict:
    """The map for the web UI: {mode, axes, x, y, E (grid, None = blank),
    paths: [{id, pair, monitor, points: [[x, y, kcal]], active}],
    reference, warnings}."""
    paths = _paths(Path(job_dir))
    warnings = []
    if not paths:
        return {"mode": mode, "paths": [], "x": [], "y": [], "E": [], "warnings": [], "axes": {}}
    if not all(p["absolute"] for p in paths):
        warnings.append("Some paths carry energies relative to their own first image only (an older run): "
                        "their heights are not on the common scale.")

    # Energy reference: the lowest first image over all paths (reactant side).
    starts = [p["energies"][0] for p in paths if p["energies"][0] is not None]
    e_ref = min(starts) if starts else 0.0

    # The pair's own endpoints (its root path, branch-0, spans the whole pair).
    roots = {}
    for p in paths:
        if p["stream"] not in roots or p["monitor"] in ("branch-0", "main"):
            roots[p["stream"]] = p
    placed = []
    if mode == "distance":
        lowest_r = min(paths, key=lambda p: p["energies"][0] if p["energies"][0] is not None else 1e9)
        lowest_p = min(paths, key=lambda p: p["energies"][-1] if p["energies"][-1] is not None else 1e9)
        fr = _fingerprint(*_frame(lowest_r["frames"][0]))
        fp_ = _fingerprint(*_frame(lowest_p["frames"][-1]))
        scale = np.sqrt(len(fr)) or 1.0
        for p in paths:
            pts = []
            for xyz, e in zip(p["frames"], p["energies"]):
                f = _fingerprint(*_frame(xyz))
                if len(f) != len(fr):
                    continue
                pts.append([float(np.linalg.norm(f - fr) / scale), float(np.linalg.norm(f - fp_) / scale),
                            None if e is None else (e - e_ref) * H2K])
            placed.append((p, pts))
        axes = {"x": "distance from the lowest reactant conformer (Å, RMS of sorted interatomic distances)",
                "y": "distance from the lowest product conformer (Å)", "x_short": "← reactant", "y_short": "← product"}
    else:
        skipped = 0
        kinds: list[str] = []
        for p in paths:
            root = roots[p["stream"]]
            sym, R = _frame(root["frames"][0])
            _, P = _frame(root["frames"][-1])
            br = sorted(_bonds(sym, R) - _bonds(sym, P))
            fo = sorted(_bonds(sym, P) - _bonds(sym, R))
            if not br and not fo:
                skipped += 1
                continue
            kind = "break-form"
            if not br or not fo:   # only one kind of change: its bonds in two groups, against each other
                changed = br or fo
                if len(changed) >= 2:
                    kind = "formed" if fo else "broken"
                    half = (len(changed) + 1) // 2
                    br, fo = changed[:half], changed[half:]
            kinds.append(kind)
            pts = []
            for xyz, e in zip(p["frames"], p["energies"]):
                _, X = _frame(xyz)

                def progress(bonds, start, end):
                    vals = [(_dist(X, i, j) - _dist(start, i, j)) / ((_dist(end, i, j) - _dist(start, i, j)) or 1e-9)
                            for i, j in bonds]
                    return float(np.mean(vals)) if vals else None

                bx, fy = progress(br, R, P), progress(fo, R, P)
                pts.append([0.0 if bx is None else bx, 0.0 if fy is None else fy,
                            None if e is None else (e - e_ref) * H2K])
            name = lambda bonds: [f"{sym[i]}{i + 1}–{sym[j]}{j + 1}" for i, j in bonds]  # noqa: E731
            p["bonds"] = {"x": name(br), "y": name(fo), "kind": kind}
            placed.append((p, pts))
        if skipped:
            warnings.append(f"{skipped} path(s) change no bonds (conformer changes) and are not on this map; "
                            "the distance map shows them.")
        main = max(set(kinds), key=kinds.count) if kinds else "break-form"
        if main == "break-form":
            axes = {"x": "bonds broken: how far they have stretched (0 = reactant, 1 = product)",
                    "y": "bonds formed: how far they have closed (0 = reactant, 1 = product)",
                    "x_short": "breaking →", "y_short": "forming →"}
        else:
            verb = "formed" if main == "formed" else "broken"
            axes = {"x": f"first bond(s) {verb}: progress (0 = reactant, 1 = product)",
                    "y": f"other bond(s) {verb}: progress (0 = reactant, 1 = product)",
                    "x_short": "first →", "y_short": "second →"}
        if len(set(kinds)) > 1:
            warnings.append("Paths here change bonds in different ways (some break and form, some only form or only "
                            "break), so the axes do not mean the same for all of them: see each path's bonds.")

    data = [(x, y, e) for _, pts in placed for x, y, e in pts if e is not None]
    xs = ys = np.array([])
    E = []
    if len(data) >= 3:
        Q = np.array([[x, y] for x, y, _ in data])
        Ev = np.array([e for _, _, e in data])
        lo, hi = Q.min(axis=0), Q.max(axis=0)
        pad = 0.08 * np.maximum(hi - lo, 1e-3)
        xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid)
        ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid)
        Z = _surface(Q, Ev, xs, ys)
        # Blank where no computed image is near (a fraction of the map's size).
        trust = 0.12 * float(np.hypot(*(hi - lo + 2 * pad)))
        Xg, Yg = np.meshgrid(xs, ys)
        near = np.min(np.hypot(Xg.ravel()[:, None] - Q[:, 0][None], Yg.ravel()[:, None] - Q[:, 1][None]), axis=1)
        Z = np.where(near.reshape(Z.shape) <= trust, Z, np.nan)
        E = [[None if not np.isfinite(v) else round(float(v), 3) for v in row] for row in Z]
    return {
        "mode": mode, "axes": axes, "x": xs.round(4).tolist(), "y": ys.round(4).tolist(), "E": E,
        "reference": "lowest reactant-side image of all paths",
        "paths": [{"id": f"{p['stream']}/{p['monitor']}", "pair": p["stream"], "monitor": p["monitor"],
                   "active": running and p["active"] and not p["finished"], "bonds": p.get("bonds"),
                   "points": [[round(x, 4), round(y, 4), None if e is None else round(e, 3)] for x, y, e in pts]}
                  for p, pts in placed if pts],
        "warnings": warnings,
    }
