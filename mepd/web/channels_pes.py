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
  irc       post hoc, relative to one computed IRC (`ts` = its TS label):
            x = where a structure projects onto that IRC (arc length from
            the TS, Angstrom; reactant side < 0), y = how far it is from
            the IRC -- with the same numbering-independent distance, so
            every chain, whatever its atom mapping, is measured against the
            true path. The IRC itself runs along y = 0 through the TS.

Energies are kcal/mol relative to the lowest reactant-side image found (the
lowest conformer, as barriers are measured). The surface is fitted to the
computed images -- their energies and, when the live data has them, their
gradients projected onto the two map coordinates (dE/dq from the Cartesian
gradient through the coordinates' Jacobian) -- so it follows the slopes the
paths actually feel; it is still approximate between paths, and left blank
far from all of them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

H2K = 627.509474
BOHR = 0.529177210903   # Angstrom


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


def _xyz_text(symbols, X) -> str:
    return f"{len(symbols)}\n\n" + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(symbols, X))


def _dist(X, i, j) -> float:
    return float(np.linalg.norm(X[i] - X[j]))


def _fingerprint(symbols, X, pairs: bool = False):
    """Sorted interatomic distances, grouped by element pair: the same for
    any atom numbering (and rotation/translation). With `pairs`, also the
    (i, j) behind each entry (for derivatives)."""
    order = sorted(range(len(symbols)), key=lambda k: symbols[k])
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    groups: dict = {}
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            groups.setdefault(tuple(sorted((symbols[i], symbols[j]))), []).append((d[i, j], i, j))
    entries = [e for k in sorted(groups) for e in sorted(groups[k])]
    f = np.array([e[0] for e in entries])
    return (f, [(e[1], e[2]) for e in entries]) if pairs else f


def _ddist(X, i, j) -> np.ndarray:
    """d|Xi - Xj| / dX (per Angstrom), as a (n, 3) array."""
    g = np.zeros_like(X)
    u = X[i] - X[j]
    u = u / (np.linalg.norm(u) or 1.0)
    g[i], g[j] = u, -u
    return g


def _projected_gradient(B: np.ndarray, grad_hartree_bohr) -> Optional[np.ndarray]:
    """dE/dq (kcal/mol per unit q) from a Cartesian gradient: the least-
    squares solution of B^T g_q = g_x, with B = dq/dX per Angstrom."""
    if grad_hartree_bohr is None:
        return None
    g = np.asarray(grad_hartree_bohr, dtype=float).reshape(-1) / BOHR   # Hartree / Angstrom
    if g.size != B.shape[1]:
        return None
    g_q, *_ = np.linalg.lstsq(B.T, g, rcond=None)
    return g_q * H2K


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
            grads = ((m.get("geometry") or {}).get("gradients")) or None
            if grads is not None and len(grads) != len(frames):
                grads = None
            plot = m.get("plot") or {}
            y = plot.get("y") or []
            if len(frames) < 2 or len(y) != len(frames):
                continue
            ref = plot.get("energy_ref_hartree")
            energies = [None if v is None else (v / H2K + (ref if ref is not None else 0.0)) for v in y]
            out.append({"stream": fp.stem, "monitor": mid, "frames": frames, "energies": energies,
                        "gradients": grads,
                        "absolute": ref is not None, "active": bool(m.get("active")),
                        "finished": bool(data.get("finished")), "caption": plot.get("caption") or ""})
    return out


def _surface(Q: np.ndarray, E: np.ndarray, xs: np.ndarray, ys: np.ndarray, slopes=None, step=None) -> np.ndarray:
    """A smooth surface through the images: a thin-plate-spline radial basis
    fit with a little smoothing (images of one path sit close together and
    would otherwise make it ripple), falling back to inverse-distance
    weighting if the fit is singular. With `slopes` (dE/dq per image, None
    where unknown), each image also contributes four close neighbours at
    +-`step` along x and y whose energies follow its gradient -- the fit
    then honours the slopes, not just the heights."""
    from scipy.interpolate import RBFInterpolator

    if slopes is not None and step is not None:
        extra_q, extra_e = [], []
        for q, e, g in zip(Q, E, slopes):
            if g is None or not np.all(np.isfinite(g)):
                continue
            for axis in (0, 1):
                for sign in (1.0, -1.0):
                    dq = np.zeros(2)
                    dq[axis] = sign * step[axis]
                    extra_q.append(q + dq)
                    extra_e.append(e + float(np.dot(g, dq)))
        if extra_q:
            Q = np.vstack([Q, np.array(extra_q)])
            E = np.concatenate([E, np.array(extra_e)])
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


def _aligned_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """RMSD (Angstrom) of two geometries in the same atom order after the
    best rigid overlay."""
    A, B = A - A.mean(axis=0), B - B.mean(axis=0)
    u, _, vt = np.linalg.svd(A.T @ B)
    rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(np.mean(np.sum((A @ rot - B) ** 2, axis=1))))


def _fp_distance(f, pairs, X, ref, scale):
    """Fingerprint distance to `ref` and its derivative d/dX (flat, per A)."""
    diff = f - ref
    D = float(np.linalg.norm(diff))
    grad = np.zeros_like(X)
    if D > 1e-9:
        for c, (i, j) in zip(diff, pairs):
            if c:
                grad += (c / (scale * D)) * _ddist(X, i, j)
    return D / scale, grad.reshape(-1)


def _read_multiframe(fp: Path) -> list[str]:
    lines, out, i = fp.read_text().splitlines(), [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        out.append("\n".join(lines[i:i + n + 2]) + "\n")
        i += n + 2
    return out


def _sidecar(fp: Path, suffix: str, n: int, width: Optional[int] = None):
    try:
        vals = np.loadtxt(fp.with_suffix(suffix))
    except Exception:
        return None
    vals = np.atleast_1d(vals) if width is None else np.atleast_2d(vals)
    if width is not None and vals.size == n * width:
        vals = vals.reshape(n, width)
    return vals if len(vals) == n else None


class _IrcFrame:
    """Coordinates relative to one IRC: (s, distance) of any structure,
    and their derivatives, from fingerprint distances to the IRC frames."""

    def __init__(self, frames_xyz: list[str], energies, reactant_fp):
        self.fps = [_fingerprint(*_frame(x)) for x in frames_xyz]
        self.scale = np.sqrt(len(self.fps[0])) or 1.0
        # Reactant side first: the end nearer the lowest reactant conformer.
        if reactant_fp is not None and len(reactant_fp) == len(self.fps[0]) and \
                np.linalg.norm(self.fps[-1] - reactant_fp) < np.linalg.norm(self.fps[0] - reactant_fp):
            self.fps.reverse()
            frames_xyz = frames_xyz[::-1]
            energies = None if energies is None else energies[::-1]
        self.frames, self.energies = frames_xyz, energies
        steps = [np.linalg.norm(b - a) / self.scale for a, b in zip(self.fps, self.fps[1:])]
        s = np.concatenate([[0.0], np.cumsum(steps)])
        ts = int(np.argmax(energies)) if energies is not None else len(s) // 2
        self.s = s - s[ts]
        self.ts_index = ts

    def place(self, xyz: str):
        """(x, y, dq/dX as a 2 x 3N array) or None (different molecule)."""
        sym, X = _frame(xyz)
        f, pairs = _fingerprint(sym, X, pairs=True)
        if len(f) != len(self.fps[0]):
            return None
        d = np.array([np.linalg.norm(f - F) / self.scale for F in self.fps])
        k = int(np.argmin(d))
        n = len(self.fps)
        j = k + 1 if k == 0 else k - 1 if k == n - 1 else (k - 1 if d[k - 1] < d[k + 1] else k + 1)
        L = abs(self.s[j] - self.s[k]) or 1e-9
        dk, gk = _fp_distance(f, pairs, X, self.fps[k], self.scale)
        dj, gj = _fp_distance(f, pairs, X, self.fps[j], self.scale)
        a = (dk * dk - dj * dj + L * L) / (2 * L)          # along the segment k -> j, from k
        da = (dk * gk - dj * gj) / L
        if a < 0 or a > L:                                  # beyond the segment: stay at frame k
            a, da = 0.0, np.zeros_like(gk)
        sign = 1.0 if self.s[j] > self.s[k] else -1.0
        x = float(self.s[k] + sign * a)
        y2 = dk * dk - a * a
        y = float(np.sqrt(max(y2, 0.0)))
        dy = (dk * gk - a * da) / y if y > 1e-6 else np.zeros_like(gk)
        return x, y, np.array([sign * da, dy])


def _optimized_ts(output_dir: Optional[Path]) -> list[dict]:
    """Every TS the run optimized (output/ts/ts_<pair>_leaf_<k>.xyz, energy
    from its sidecar) and what its IRC made of it: a direct channel, a step
    of a multi-step channel, or neither (off-target / unconnected)."""
    if output_dir is None or not (Path(output_dir) / "ts").is_dir():
        return []
    out_dir = Path(output_dir)
    kind_of: dict[str, tuple[str, str]] = {}
    for members in out_dir.rglob("members.txt"):
        rel = members.relative_to(out_dir).parts
        kind = "direct" if rel[0] == "channels" else "multi-step" if rel[0].startswith("alternate") else "other"
        for line in members.read_text().splitlines():
            label = line.strip()
            if label.startswith("ts_") and label not in kind_of:
                kind_of[label] = (kind, rel[-2] if len(rel) > 1 else "")
    found = []
    for fp in sorted((out_dir / "ts").glob("ts_pair_*.xyz")):
        if fp.stem.endswith("_irc"):
            continue
        stream = fp.stem[len("ts_"):].split("_leaf_")[0]
        try:
            xyz = "\n".join(fp.read_text().splitlines()[:int(fp.read_text().split()[0]) + 2]) + "\n"
            energy = float(fp.with_suffix(".energies").read_text().split()[0])
        except Exception:
            continue
        kind, group = kind_of.get(fp.stem, ("other", ""))
        irc = fp.with_name(f"{fp.stem}_irc.xyz")
        found.append({"label": fp.stem, "stream": stream, "xyz": xyz, "energy": energy, "kind": kind, "group": group,
                      "irc": str(irc) if irc.exists() else None})
    return found


def channels_map(job_dir: Path, mode: str = "bonds", grid: int = 64, running: bool = True,
                 output_dir: Optional[Path] = None, ts: Optional[str] = None) -> dict:
    """The map for the web UI: {mode, axes, x, y, E (grid, None = blank),
    paths: [{id, pair, monitor, points: [[x, y, kcal]], active}],
    ts: [{label, pair, q, e, kind, closest}], reference, warnings}."""
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
    all_ts = _optimized_ts(output_dir)
    irc_choices = [{"label": t["label"], "pair": t["stream"], "kind": t["kind"], "group": t["group"],
                    "e": round((t["energy"] - e_ref) * H2K, 2)} for t in all_ts if t["irc"]]
    irc_line, chosen_ts = None, None
    if mode == "irc":
        pick = next((t for t in all_ts if t["label"] == ts and t["irc"]), None) or min(
            (t for t in all_ts if t["irc"]), key=lambda t: (t["kind"] != "direct", t["energy"]), default=None)
        if pick is None:
            return {"mode": mode, "paths": [], "x": [], "y": [], "E": [], "axes": {}, "ts": [], "irc_choices": [],
                    "warnings": ["No IRC has been computed yet: this view needs a finished TS optimization and its IRC."]}
        irc_fp = Path(pick["irc"])
        irc_frames = _read_multiframe(irc_fp)
        irc_e = _sidecar(irc_fp, ".energies", len(irc_frames))
        lowest_r = min(paths, key=lambda p: p["energies"][0] if p["energies"][0] is not None else 1e9)
        frame = _IrcFrame(irc_frames, irc_e, _fingerprint(*_frame(lowest_r["frames"][0])))

        def place_ts(stream, X, sym):
            got = frame.place(_xyz_text(sym, X))
            return None if got is None else [got[0], got[1]]

        for p in paths:
            pts = []
            for k, (xyz, e) in enumerate(zip(p["frames"], p["energies"])):
                got = frame.place(xyz)
                if got is None:
                    continue
                x, y, B = got
                g = _projected_gradient(B, p["gradients"][k]) if p["gradients"] else None
                pts.append([x, y, None if e is None else (e - e_ref) * H2K, g])
            placed.append((p, pts))
        # The IRC itself: y = 0, real energies and gradients (a data line of its own).
        irc_g = _sidecar(irc_fp, ".gradients", len(irc_frames), width=3 * len(_frame(irc_frames[0])[0]))
        if irc_g is not None and frame.frames is not irc_frames:
            irc_g = irc_g[::-1]
        irc_line = []
        for k, xyz in enumerate(frame.frames):
            got = frame.place(xyz)
            e = None if frame.energies is None else (float(frame.energies[k]) - e_ref) * H2K
            g = _projected_gradient(got[2], irc_g[k]) if (got is not None and irc_g is not None) else None
            irc_line.append([round(float(frame.s[k]), 4), 0.0, None if e is None else round(e, 3), g])
        axes = {"x": f"position along the IRC of {pick['label'].replace('ts_', '').replace('_', ' ')} (Å from its TS; "
                     "reactant ←, → product)",
                "y": "distance from that IRC (Å, numbering-independent)", "x_short": "along IRC", "y_short": "off IRC"}
        chosen_ts = pick["label"]
    elif mode == "distance":
        lowest_r = min(paths, key=lambda p: p["energies"][0] if p["energies"][0] is not None else 1e9)
        lowest_p = min(paths, key=lambda p: p["energies"][-1] if p["energies"][-1] is not None else 1e9)
        fr = _fingerprint(*_frame(lowest_r["frames"][0]))
        fp_ = _fingerprint(*_frame(lowest_p["frames"][-1]))
        scale = np.sqrt(len(fr)) or 1.0

        def place_ts(stream, X, sym):
            f = _fingerprint(sym, X)
            return None if len(f) != len(fr) else [float(np.linalg.norm(f - fr) / scale),
                                                   float(np.linalg.norm(f - fp_) / scale)]
        for p in paths:
            pts = []
            for k, (xyz, e) in enumerate(zip(p["frames"], p["energies"])):
                sym, X = _frame(xyz)
                f, pairs = _fingerprint(sym, X, pairs=True)
                if len(f) != len(fr):
                    continue
                q, rows = [], []
                for ref in (fr, fp_):
                    diff = f - ref
                    D = float(np.linalg.norm(diff))
                    q.append(D / scale)
                    row = np.zeros_like(X)
                    if D > 1e-9:
                        for c, (i, j) in zip(diff, pairs):
                            if c:
                                row += (c / (scale * D)) * _ddist(X, i, j)
                    rows.append(row.reshape(-1))
                g = _projected_gradient(np.array(rows), p["gradients"][k]) if p["gradients"] else None
                pts.append([q[0], q[1], None if e is None else (e - e_ref) * H2K, g])
            placed.append((p, pts))
        axes = {"x": "distance from the lowest reactant conformer (Å, RMS of sorted interatomic distances)",
                "y": "distance from the lowest product conformer (Å)", "x_short": "← reactant", "y_short": "← product"}
    else:
        skipped = 0
        kinds: list[str] = []
        pair_bonds: dict = {}

        def place_ts(stream, X, sym):
            if stream not in pair_bonds:
                return None
            br_, fo_, R_, P_ = pair_bonds[stream]
            prog = lambda bonds: float(np.mean([(_dist(X, i, j) - _dist(R_, i, j)) / ((_dist(P_, i, j) - _dist(R_, i, j)) or 1e-9)  # noqa: E731
                                                for i, j in bonds])) if bonds else 0.0
            return [prog(br_), prog(fo_)]

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
            pair_bonds.setdefault(p["stream"], (br, fo, R, P))
            pts = []
            for k, (xyz, e) in enumerate(zip(p["frames"], p["energies"])):
                _, X = _frame(xyz)

                def progress(bonds, start, end):
                    """(mean progress of the bonds, its derivative d/dX per Angstrom)."""
                    if not bonds:
                        return 0.0, np.zeros(X.size)
                    vals, grad = [], np.zeros_like(X)
                    for i, j in bonds:
                        span = (_dist(end, i, j) - _dist(start, i, j)) or 1e-9
                        vals.append((_dist(X, i, j) - _dist(start, i, j)) / span)
                        grad += _ddist(X, i, j) / span
                    return float(np.mean(vals)), (grad / len(bonds)).reshape(-1)

                (bx, dbx), (fy, dfy) = progress(br, R, P), progress(fo, R, P)
                g = _projected_gradient(np.array([dbx, dfy]), p["gradients"][k]) if p["gradients"] else None
                pts.append([bx, fy, None if e is None else (e - e_ref) * H2K, g])
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

    # Optimized TSs, placed with their own pair's coordinates; how close did its chains get?
    ts_marks = []
    for ts in all_ts:
        sym, X = _frame(ts["xyz"])
        q = place_ts(ts["stream"], X, sym)
        if q is None:
            continue
        closest = None
        for p, pts in placed:
            if p["stream"] != ts["stream"]:
                continue
            for k, xyz in enumerate(p["frames"]):
                _, F = _frame(xyz)
                if F.shape != X.shape:
                    continue
                r = _aligned_rmsd(F, X)
                if closest is None or r < closest["rmsd"]:
                    closest = {"rmsd": round(r, 3), "image": k, "monitor": p["monitor"],
                               "map_distance": round(float(np.hypot(pts[k][0] - q[0], pts[k][1] - q[1])), 4)}
        ts_marks.append({"label": ts["label"], "pair": ts["stream"], "q": [round(q[0], 4), round(q[1], 4)],
                         "e": round((ts["energy"] - e_ref) * H2K, 3), "kind": ts["kind"], "group": ts["group"],
                         "closest": closest, "xyz": ts["xyz"]})

    data = [(x, y, e, g) for _, pts in placed for x, y, e, g in pts if e is not None]
    if irc_line:
        data += [(x, y, e, g) for x, y, e, g in irc_line if e is not None]
    # A TS is a real stationary point: its energy, with zero slope, anchors the fit at the saddle.
    data += [(m["q"][0], m["q"][1], m["e"], np.zeros(2)) for m in ts_marks]
    xs = ys = np.array([])
    E = []
    with_slopes = sum(1 for d in data if d[3] is not None)
    if data and with_slopes < len(data):
        warnings.append(f"{len(data) - with_slopes} of {len(data)} images have no gradient in the live data (an "
                        "older run, or an engine that does not report them): the surface uses only their energies.")
    if len(data) >= 3:
        Q = np.array([[x, y] for x, y, _, _ in data])
        Ev = np.array([e for _, _, e, _ in data])
        lo, hi = Q.min(axis=0), Q.max(axis=0)
        pad = 0.08 * np.maximum(hi - lo, 1e-3)
        xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid)
        ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid)
        Z = _surface(Q, Ev, xs, ys, slopes=[d[3] for d in data], step=0.02 * np.maximum(hi - lo, 1e-3))
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
                   "points": [[round(x, 4), round(y, 4), None if e is None else round(e, 3)] for x, y, e, _ in pts]}
                  for p, pts in placed if pts],
        "ts": ts_marks,
        "irc_choices": irc_choices,
        "irc": {"ts": chosen_ts, "points": [[round(x, 4), round(y, 4), e] for x, y, e, _ in irc_line]} if irc_line else None,
        "warnings": warnings,
    }
