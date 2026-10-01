"""Build a complex -- several molecules together, not bonded -- from its
molecules. Every method returns geometries only (best first): their
energies are a screening level's, so callers minimize them at their own
level of theory before comparing anything.

Methods:
  side    side by side along x, closest contacts `gap` apart. Instant.
  packed  random orientations inside the smallest sphere they fit in
          (the nanoreactor's packer; Packmol when it is installed). Instant.
  dock    xtb's aISS docking: interaction sites screened on a grid, then
          a genetic search and GFN2 optimizations of the best poses. Several
          molecules are added one at a time. Seconds to a minute.
  nci     CREST's conformer search in NCI mode, started from the docked
          complex: an ensemble of low-lying arrangements. Minutes.
  qcg     CREST's quantum cluster growth: N copies of a solvent grown around
          one solute, one at a time, and the cluster optimized. A minute or so.

References: aISS, C. Plett, S. Grimme, Angew. Chem. Int. Ed. 62, e202214477
(2023), doi:10.1002/anie.202214477; CREST NCI mode, P. Pracht, F. Bohle,
S. Grimme, Phys. Chem. Chem. Phys. 22, 7169 (2020), doi:10.1039/C9CP06869D;
QCG, S. Spicher, C. Plett, P. Pracht, A. Hansen, S. Grimme, J. Chem. Theory
Comput. 18, 3174 (2022), doi:10.1021/acs.jctc.2c00239; Packmol, L. Martinez,
R. Andrade, E. G. Birgin, J. M. Martinez, J. Comput. Chem. 30, 2157 (2009),
doi:10.1002/jcc.21224.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

METHODS = ("side", "packed", "dock", "nci", "qcg")
INSTANT = ("side", "packed")

_Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}


def _coords(s) -> np.ndarray:
    return np.asarray(s.geometry, dtype=float).reshape(-1, 3) / ANGSTROM_TO_BOHR


def _structure(symbols, coords_angstrom, charge: int):
    from qcdata import Structure

    electrons = sum(_Z.get(x, 0) for x in symbols) - int(charge)
    return Structure(symbols=list(symbols), geometry=np.asarray(coords_angstrom) * ANGSTROM_TO_BOHR,
                     charge=int(charge), multiplicity=1 if electrons % 2 == 0 else 2)


def side_by_side(structures: Sequence, gap: float = 2.6):
    """One structure holding every molecule, side by side along x, each
    centred, the closest contact between neighbours `gap` Angstrom."""
    symbols, blocks, x_end = [], [], None
    for s in structures:
        xyz = _coords(s)
        xyz = xyz - xyz.mean(axis=0)
        if x_end is not None:
            xyz[:, 0] += x_end - xyz[:, 0].min() + gap
        x_end = xyz[:, 0].max()
        symbols += list(s.symbols)
        blocks.append(xyz)
    coords = np.vstack(blocks)
    return _structure(symbols, coords - coords.mean(axis=0), sum(int(s.charge) for s in structures))


def packed(structures: Sequence, *, seed: int = 0, min_distance: float = 2.0):
    """Random orientations in the smallest sphere they fit in (grown 8% at
    a time from the largest molecule's size): Packmol when installed, else
    the nanoreactor's packer."""
    from mepd.discovery.nanoreactor import pack_reactor

    charge = sum(int(s.charge) for s in structures)
    extent = max(float(np.max(np.linalg.norm(_coords(s) - _coords(s).mean(axis=0), axis=1))) for s in structures)
    radius = extent + min_distance
    exe = shutil.which("packmol")
    for _ in range(60):
        try:
            if exe:
                return _structure(*_packmol(structures, radius, seed, min_distance, exe), charge)
            symbols, coords, _, _ = pack_reactor(structures, radius, seed=seed, min_distance=min_distance, tries=400)
            return _structure(symbols, coords, charge)
        except RuntimeError:
            radius *= 1.08
    raise RuntimeError("could not pack these molecules")


def _packmol(structures, radius: float, seed: int, tolerance: float, exe: str):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lines = [f"tolerance {tolerance}", "filetype xyz", "output packed.xyz", f"seed {seed + 1}"]
        for k, s in enumerate(structures):
            _write_xyz(tmp / f"m{k}.xyz", s.symbols, _coords(s))
            lines += [f"structure m{k}.xyz", "  number 1", f"  inside sphere 0. 0. 0. {radius:.3f}", "end structure"]
        (tmp / "pack.inp").write_text("\n".join(lines) + "\n")
        with (tmp / "pack.inp").open() as inp:
            done = subprocess.run([exe], stdin=inp, cwd=tmp, capture_output=True, text=True, timeout=600)
        if done.returncode != 0 or not (tmp / "packed.xyz").exists() or "Success!" not in done.stdout:
            raise RuntimeError("packmol could not pack them")
        symbols, frames, _ = _read_xyz(tmp / "packed.xyz")
    coords = frames[0]
    return symbols, coords - coords.mean(axis=0)


# ---------------------------------------------------------------- programs
def xtb_dock_executable() -> Optional[str]:
    """An xtb whose `dock` works: $MEPD_XTB_DOCK, the xtb shipped with g-xTB
    (the xtb 6.7.1 release build segfaults in aISS's final optimizations),
    else xtb on PATH."""
    for exe in (os.getenv("MEPD_XTB_DOCK"), str(Path.home() / ".local/opt/gxtb-2.0.1/bin/xtb"), shutil.which("xtb")):
        if exe and Path(exe).exists():
            return exe
    return None


def missing_programs(method: str) -> list[str]:
    if method == "dock" or method == "nci":
        need = ["xtb"] if xtb_dock_executable() is None else []
        return need + (["crest"] if method == "nci" and shutil.which("crest") is None else [])
    if method == "qcg":
        return ["crest"] if shutil.which("crest") is None else []
    return []


def _env(cwd: Path):
    """Single-threaded (parallelize across jobs), and the xtb whose `dock`
    works first on PATH: CREST's QCG docks with whatever `xtb` it finds."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": os.environ.get("OMP_STACKSIZE", "1G")}
    exe = xtb_dock_executable()
    if exe and exe != shutil.which("xtb"):
        bin_dir = Path(cwd) / ".bin"
        bin_dir.mkdir(exist_ok=True)
        link = bin_dir / "xtb"
        if not link.exists():
            link.symlink_to(exe)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def _run(argv, cwd: Path, what: str, timeout: float):
    # Intel-built xtb/CREST want a large stack (else segfaults in big systems).
    cmd = "ulimit -s unlimited 2>/dev/null; exec " + " ".join(_quote(a) for a in argv)
    done = subprocess.run(["bash", "-c", cmd], cwd=cwd, env=_env(cwd), capture_output=True, text=True, timeout=timeout)
    (cwd / f"{what}.log").write_text(done.stdout[-200000:] + done.stderr[-20000:])
    if done.returncode != 0:
        tail = (done.stdout + done.stderr).strip().splitlines()[-6:]
        raise RuntimeError(f"{what} failed (exit {done.returncode}): " + " | ".join(tail))
    return done


def _quote(a) -> str:
    import shlex

    return shlex.quote(str(a))


def _charge_flags(s, second: bool = False) -> list[str]:
    tag = "2" if second else ""
    out = ["--chrg" + tag, str(int(s.charge))]
    if int(s.multiplicity) > 1:
        out += ["--uhf" + tag, str(int(s.multiplicity) - 1)]
    return out


# ------------------------------------------------------------------- dock
def dock(structures: Sequence, workdir: Path, *, keep: int = 3, timeout: float = 3600):
    """aISS: the first molecule, then each next one docked onto what is
    there (the best pose carried on). Returns the last step's poses, best
    first."""
    exe = xtb_dock_executable()
    if exe is None:
        raise RuntimeError("docking needs xtb (6.6 or newer)")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(structures)), key=lambda k: -len(structures[k].symbols))   # largest first
    host = structures[order[0]]
    poses = [host]
    for step, k in enumerate(order[1:], start=1):
        guest = structures[k]
        d = workdir / f"step_{step}"
        d.mkdir(exist_ok=True)
        _write_xyz(d / "host.xyz", host.symbols, _coords(host))
        _write_xyz(d / "guest.xyz", guest.symbols, _coords(guest))
        try:
            _run([exe, "dock", "host.xyz", "guest.xyz", "--nfinal", str(max(keep, 3)),
                  *_charge_flags(host), *_charge_flags(guest, second=True)], d, "dock", timeout)
        except RuntimeError:
            # xtb's final GFN2 optimizations can crash (e.g. two ions pushed
            # together, or the optimizer itself): the screened poses are
            # still good starting points -- they get minimized anyway.
            pass
        symbols, frames, comments = _poses(d)
        if not len(frames):
            log = (d / "dock.log").read_text().strip().splitlines()[-3:] if (d / "dock.log").exists() else []
            raise RuntimeError(f"docking step {step} found no pose: " + " | ".join(log))
        energy = [_first_float(c) for c in comments]
        ranked = sorted(range(len(frames)), key=lambda i: energy[i] if energy[i] is not None else 0.0)
        charge = int(host.charge) + int(guest.charge)
        poses = [_structure(symbols, frames[i], charge) for i in ranked]
        host = poses[0]
    return poses[:keep]


def _poses(d: Path):
    """aISS's poses, best available: optimized, else (the final
    optimizations failed) the screened ones."""
    for name in ("optimized_structures.xyz", "final_structures.xyz", "best_after_gen.xyz", "best.xyz"):
        fp = d / name
        if fp.exists() and fp.stat().st_size:
            got = _read_xyz(fp)
            if len(got[1]):
                return got
    return [], np.zeros((0, 0, 3)), []


# -------------------------------------------------------------------- crest
def nci(structures: Sequence, workdir: Path, *, keep: int = 5, timeout: float = 6 * 3600):
    """CREST NCI-mode conformers of the complex, from the docked one (else
    packed when docking is not possible), best first."""
    if shutil.which("crest") is None:
        raise RuntimeError("the NCI ensemble needs CREST")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        start = dock(structures, workdir / "dock", keep=1)[0]
    except RuntimeError:
        start = packed(structures)
    _write_xyz(workdir / "start.xyz", start.symbols, _coords(start))
    argv = ["crest", "start.xyz", "--nci", "--T", "1", *_charge_flags(start)]
    try:
        try:
            _run(argv, workdir, "crest", timeout)
        except RuntimeError:
            # The start changed its bonds when CREST first optimized it (e.g.
            # an ion meeting a molecule it reacts with): search anyway, and let
            # the caller drop the arrangements that reacted.
            if "topology" not in (workdir / "crest.log").read_text():
                raise
            _run(argv + ["--noreftopo"], workdir, "crest", timeout)
    except RuntimeError as exc:
        if "Initial geometry optimization failed" in (workdir / "crest.log").read_text():
            hint = (f" With a total charge of {int(start.charge):+d}, several ions together rarely form a bound "
                    "complex in the gas phase: try fewer ions, or add counter-ions.") if abs(int(start.charge)) > 1 else ""
            raise RuntimeError("CREST could not optimize the starting complex at GFN2-xTB." + hint) from exc
        raise
    return _ensemble(workdir / "crest_conformers.xyz", int(start.charge), keep)


def qcg(solute, solvent, n: int, workdir: Path, *, keep: int = 5, timeout: float = 6 * 3600):
    """CREST QCG: `n` copies of `solvent` grown around `solute` (the grown,
    optimized cluster; CREST's own QCG ensemble step is not run)."""
    if shutil.which("crest") is None:
        raise RuntimeError("the solvation shell needs CREST")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    _write_xyz(workdir / "solute.xyz", solute.symbols, _coords(solute))
    _write_xyz(workdir / "solvent.xyz", solvent.symbols, _coords(solvent))
    _run(["crest", "solute.xyz", "--qcg", "solvent.xyz", "--nsolv", str(int(n)), "--T", "1",
          *_charge_flags(solute)], workdir, "crest", timeout)
    charge = int(solute.charge) + n * int(solvent.charge)
    for name in ("ensemble/final_ensemble.xyz", "grow/cluster_optimized.xyz", "grow/cluster.xyz"):
        if (workdir / name).exists():
            return _ensemble(workdir / name, charge, keep)
    raise RuntimeError("CREST QCG wrote no cluster")


def _ensemble(fp: Path, charge: int, keep: int):
    symbols, frames, comments = _read_xyz(fp)
    energy = [_first_float(c) for c in comments]
    ranked = sorted(range(len(frames)), key=lambda i: energy[i] if energy[i] is not None else 0.0)
    return [_structure(symbols, frames[i], charge) for i in ranked[:keep]]


# ---------------------------------------------------------------- one call
def build(structures: Sequence, method: str, workdir: Optional[Path] = None, *, keep: int = 3, seed: int = 0):
    """Geometries of the complex of `structures` (one per molecule, repeated
    for 2 A) by `method`, best first."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r} (one of {', '.join(METHODS)})")
    if len(structures) < 2:
        raise ValueError("a complex needs at least two molecules")
    if method == "side":
        return [side_by_side(structures)]
    if method == "packed":
        return [packed(structures, seed=seed)]
    if workdir is None:
        raise ValueError(f"{method} needs a working directory")
    if method == "dock":
        return dock(structures, workdir, keep=keep)
    if method == "nci":
        return nci(structures, workdir, keep=keep)
    # qcg: one solute, the rest copies of one solvent
    kinds: dict = {}
    for s in structures:
        kinds.setdefault((tuple(s.symbols), int(s.charge)), []).append(s)
    groups = sorted(kinds.values(), key=len)
    if len(groups) != 2 or len(groups[0]) != 1 or len(groups[1]) < 2:
        raise ValueError("a solvation shell needs one solute and several copies of one solvent")
    return qcg(groups[0][0], groups[1][0], len(groups[1]), workdir, keep=keep)


# -------------------------------------------------------------------- files
def _write_xyz(fp: Path, symbols, coords, comment: str = "") -> None:
    rows = "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(symbols, coords))
    Path(fp).write_text(f"{len(symbols)}\n{comment}\n{rows}")


def _read_xyz(fp: Path):
    from mepd.discovery.nanoreactor import read_xyz_frames

    return read_xyz_frames(fp)


def _first_float(text: str) -> Optional[float]:
    for tok in text.replace("=", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return None
