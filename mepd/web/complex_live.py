"""A `complex` job, live: what its programs have written so far, as frames
to play in order while it runs (there is no replay afterwards).

  dock   per molecule added (work/step_k): the newcomer beside what is
         there while aISS searches, then its optimized pose.
  qcg    the solute, then each solvent molecule as CREST adds it
         (grow/qcg_grow.xyz), then the cluster's optimization
         (qcg_tmp/tmp_grow/xtbopt.log, while it exists).
  nci    the docked start (as dock), CREST's first optimization
         (crestopt.log), then its metadynamics runs (MDFILES/crest_k.trj)
         as they grow; while it re-optimizes what it found, the last frame
         holds.

Only complete frames of files still being written are read. Frames are
taken at fixed strides, so a growing file only ever appends frames (the
page keeps playing where it was).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

MAX_PER_SEGMENT = 120


def _frames(fp: Path, stride: int = 1) -> list[tuple[list, np.ndarray, str]]:
    """(symbols, coords [N, 3] in Angstrom, comment) of each complete frame,
    atom counts free to change from frame to frame."""
    try:
        lines = fp.read_text().splitlines()
    except OSError:
        return []
    out, k = [], 0
    while k < len(lines):
        head = lines[k].split()
        if not head:
            k += 1
            continue
        try:
            n = int(head[0])
        except ValueError:
            break
        rows = lines[k + 2:k + 2 + n]
        if len(rows) < n:
            break                                  # being written
        try:
            syms = [r.split()[0] for r in rows]
            xyz = np.array([[float(v) for v in r.split()[1:4]] for r in rows])
        except (ValueError, IndexError):
            break
        out.append((syms, xyz, lines[k + 1]))
        k += 2 + n
    return out[::stride]


def _frame(symbols, xyz, caption: str, anchor: int = 0, tween: bool = False) -> dict:
    """One frame for the page, centred on its first `anchor` atoms (the
    part already there), else on all."""
    xyz = np.asarray(xyz, dtype=float)
    centre = xyz[:anchor].mean(axis=0) if 0 < anchor <= len(xyz) else xyz.mean(axis=0)
    return {"symbols": list(symbols), "xyz": np.round(xyz - centre, 2).reshape(-1).tolist(), "caption": caption,
            "tween": tween}


def _beside(host, guest):
    """The newcomer just off to the side of what is there (+x)."""
    h, g = np.asarray(host), np.asarray(guest) - np.asarray(guest).mean(axis=0)
    g = g + [h[:, 0].max() - g[:, 0].min() + 3.5, h[:, 1].mean(), h[:, 2].mean()]
    return np.vstack([h, g])


def _dock(base: Path, total: int) -> list[dict]:
    out = []
    steps = sorted(base.glob("step_*"), key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0)
    for d in steps:
        k = int(d.name.split("_")[1])
        host, guest = _frames(d / "host.xyz"), _frames(d / "guest.xyz")
        if not host or not guest:
            continue
        hs, hx, _ = host[0]
        gs, gx, _ = guest[0]
        what = f"Molecule {k + 1} of {total}" if total else f"Molecule {k + 1}"
        n = len(hs)
        # (aISS's screened poses are left out: unoptimized, their close
        # contacts would be drawn as bonds.)
        out.append(_frame(hs + gs, _beside(hx, gx), f"{what}: finding where it binds…", anchor=n))
        poses = _frames(d / "optimized_structures.xyz")
        if poses and len(poses[0][0]) == n + len(gs):
            def energy(c):
                m = re.search(r"-?\d+\.\d+", c)
                return float(m.group()) if m else 0.0
            syms, xyz, _ = min(poses, key=lambda p: energy(p[2]))
            out.append(_frame(syms, xyz, f"{what}: settled", anchor=n, tween=True))
    return out


def _qcg(work: Path, total: int) -> list[dict]:
    out = []
    solute, solvent = _frames(work / "solute.xyz"), _frames(work / "solvent.xyz")
    if not solute or not solvent:
        return out
    n0, n1 = len(solute[0][0]), len(solvent[0][0])
    out.append(_frame(solute[0][0], solute[0][1], "The solute", anchor=n0))
    for syms, xyz, _ in _frames(work / "grow" / "qcg_grow.xyz"):
        k = (len(syms) - n0) // n1
        out.append(_frame(syms, xyz, f"Solvent molecule {k}" + (f" of {total - 1}" if total else "") + " added",
                          anchor=n0))
    opt = _frames(work / "qcg_tmp" / "tmp_grow" / "xtbopt.log")
    stride = max(1, len(opt) // MAX_PER_SEGMENT + 1)
    for syms, xyz, _ in opt[::stride]:
        out.append(_frame(syms, xyz, "Optimizing the cluster", anchor=n0, tween=True))
    return out


def _nci(work: Path, total: int) -> list[dict]:
    out = _dock(work / "dock", total)
    first = _frames(work / "crestopt.log", stride=2)
    for syms, xyz, _ in first[:MAX_PER_SEGMENT]:
        out.append(_frame(syms, xyz, "Optimizing the starting complex"))
    runs = sorted((work / "TRIALMD").glob("*.trj")) + sorted(
        (work / "MDFILES").glob("crest_*.trj"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    for fp in runs:
        for syms, xyz, _ in _frames(fp, stride=4)[:MAX_PER_SEGMENT]:
            out.append(_frame(syms, xyz, "Exploring arrangements (CREST dynamics)"))
    return out


def complex_live(out: Path, method: str, total: int = 0) -> dict:
    """{"frames": [...], "stage": what is happening now} for a running
    complex job of `method` with `total` molecules."""
    work = Path(out) / "work"
    frames = {"dock": lambda: _dock(work, total), "qcg": lambda: _qcg(work, total),
              "nci": lambda: _nci(work, total)}.get(method, lambda: [])()
    stage = frames[-1]["caption"] if frames else "Building the complex…"
    if method == "nci" and (work / "crest_ensemble.xyz").exists():
        stage = "Optimizing the arrangements found…"
    if (Path(out) / "complexes.xyz").exists():
        stage = "Built: minimizing at the workspace level next"
    return {"frames": frames, "stage": stage, "method": method}
