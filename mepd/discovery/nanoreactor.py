"""Nanoreactor: discover elementary reactions in a compressed, hot molecular
dynamics box, then extract each reaction with only the atoms it needs.

A reactor holds many molecules (reactants, solvent, possible shuttles). A
spherical wall that periodically contracts (a piston) pushes them together
at high temperature, so reactions happen within picoseconds. The trajectory
is then read as a sequence of bond-graph changes:

  * bonds are perceived per frame with hysteresis (a bond forms below
    `form` x the sum of covalent radii and breaks above `break_`), and a
    bond state that lives shorter than `min_lifetime_fs` is vibrational
    noise, not chemistry;
  * bond changes that are close in time (`merge_window_fs`) and touch the
    same molecules are one reaction event;
  * an event's subsystem is the closure of the molecules that contain its
    changed atoms, before and after (every molecule whose bonds change,
    including a water that shuttles a proton, and nothing else); spectators
    are dropped;
  * the reactant and product frames of that subsystem are cut out of the
    reactor (same atoms, same order: atom-mapped endpoints), optimized at
    the refinement level of theory, and each molecule is also optimized on
    its own.

Species are molecules (canonical SMILES, charge, multiplicity). A reaction
is a multiset of reactant species -> a multiset of product species; a
species on both sides is a shuttle/catalyst (keto + H2O -> enol + H2O is a
different reaction from keto -> enol, so two species can be joined by
several reactions). Energies of different subsystems have different atoms
and are never compared; every reported energy is a difference within one
reaction (products - reactants, TS - reactant complex).

Based on the ab initio nanoreactor (L.-P. Wang, A. Titov, R. McGibbon,
F. Liu, V. S. Pande, T. J. Martinez, Nat. Chem. 6, 1044-1048 (2014),
doi:10.1038/nchem.2099), reimplemented in mepd: the piston is xtb's
spherical wall potential switched between two radii, the MD is xtb's
(Bannwarth, Ehlert, Grimme, J. Chem. Theory Comput. 15, 1652 (2019),
doi:10.1021/acs.jctc.8b01176), and the event detection, subsystem
extraction and refinement are mepd's own. No code from the original is used.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional, Sequence

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR, HARTREE_TO_KCAL_PER_MOL

OnEvent = Optional[Callable[[str, dict], None]]

REFERENCES = {
    "nanoreactor": {
        "method": "piston-compressed high-temperature MD to discover reactions, then extraction and refinement of "
                  "each reaction event (mepd's reimplementation of the ab initio nanoreactor; no code from it is used)",
        "cite": ["L.-P. Wang, A. Titov, R. McGibbon, F. Liu, V. S. Pande, T. J. Martinez, Nat. Chem. 6, 1044-1048 "
                 "(2014), doi:10.1038/nchem.2099"],
    },
    "md": {
        "method": "molecular dynamics and spherical wall potential of the xtb program (GFN-xTB)",
        "cite": ["C. Bannwarth, S. Ehlert, S. Grimme, J. Chem. Theory Comput. 15, 1652-1671 (2019), "
                 "doi:10.1021/acs.jctc.8b01176",
                 "C. Bannwarth et al., WIREs Comput. Mol. Sci. 11, e1493 (2021), doi:10.1002/wcms.1493"],
    },
    "lewis_filter": {
        "method": "species identity: bond orders and formal charges from connectivity, RDKit DetermineBondOrders "
                  "(xyz2mol)",
        "cite": ["Y. Kim, W. Y. Kim, Bull. Korean Chem. Soc. 36, 1769-1777 (2015), doi:10.1002/bkcs.10334"],
    },
    "events": {
        "method": "mepd: bond-graph changes with hysteresis and a minimum lifetime, grouped into events by time "
                  "and shared molecules; the subsystem is the closure of the molecules whose bonds change, split "
                  "into groups that share atoms; covalent radii of Cordero et al.",
        "cite": ["B. Cordero et al., Dalton Trans. 2832-2838 (2008), doi:10.1039/b801115j"],
    },
}

# Covalent radii (Angstrom; Cordero et al., Dalton Trans. 2008, 2832).
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Fe": 1.32, "Cu": 1.32, "Zn": 1.22, "Se": 1.20, "Br": 1.20, "I": 1.39,
}

MD_METHODS = ("gfn2", "gfn1", "gxtb", "level")   # "level": any mepd engine (run_engine_md)


def engine_gxtb_executable(engine) -> Optional[str]:
    """The g-xTB executable a profile's g-xTB engine uses, if it is one."""
    exe = getattr(engine, "executable", None)
    if exe and type(engine).__name__ == "GXTBCalculator":
        found = shutil.which(str(exe)) or (str(exe) if Path(str(exe)).exists() else None)
        return found
    return None


def pick_md_method(engine=None) -> tuple[str, Optional[str]]:
    """'auto': the fastest MD this machine can run -- xtb's GFN2 MD, else
    g-xTB's MD (its executable from the environment or the profile's g-xTB
    engine), else the profile's own engine. Returns (method, executable)."""
    if not missing_programs("gfn2"):
        return "gfn2", None
    exe = engine_gxtb_executable(engine) if engine is not None else None
    try:
        _xtb_command("gxtb", exe, download=False)   # what is installed (not a download just to pick)
        return "gxtb", exe
    except RuntimeError:
        return "level", None


@dataclass
class ReactorSettings:
    temperature: float = 1500.0      # K
    time_ps: float = 10.0            # total simulated time
    step_fs: float = 0.5
    dump_fs: float = 2.0             # frame spacing of the trajectory
    radius: Optional[float] = None   # Angstrom, the wide wall; None = from the number of atoms
    compress: float = 0.7            # narrow radius / wide radius
    period_ps: float = 1.0           # one piston cycle
    duty: float = 0.75               # fraction of the cycle at the wide radius
    method: str = "gfn2"             # xtb Hamiltonian of the discovery MD
    wall_force: float = 20.0         # kcal/mol/Angstrom: the wall's push on an atom outside it
    ramp_fs: float = 100.0           # the piston closes over this long, in steps (no shock)
    electronic_temperature: float = 3000.0  # K, Fermi smearing: bonds break without SCC failures
    seed: int = 0

    def validate(self) -> None:
        if self.method not in MD_METHODS:
            raise ValueError(f"method must be one of {', '.join(MD_METHODS)}, got {self.method!r}.")
        for name in ("temperature", "time_ps", "step_fs", "dump_fs", "period_ps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0 < self.compress <= 1:
            raise ValueError("compress must be in (0, 1].")
        if not 0 < self.duty <= 1:
            raise ValueError("duty must be in (0, 1].")
        if self.dump_fs < self.step_fs:
            raise ValueError("dump_fs must be at least step_fs.")


@dataclass
class DetectSettings:
    form: float = 1.15               # x (r_i + r_j): a bond forms below this
    break_: float = 1.45             # x (r_i + r_j): a bond breaks above this
    min_lifetime_fs: float = 20.0    # shorter-lived bond states are noise
    merge_window_fs: float = 100.0   # bond changes this close, on shared molecules, are one event
    lead_fs: float = 20.0            # reactant frame: this long before the first change
    lag_fs: float = 20.0             # product frame: this long after the last change


# ---------------------------------------------------------------------------
# Building the reactor
# ---------------------------------------------------------------------------

def auto_radius(n_atoms: int, volume_per_atom: float = 25.0) -> float:
    """Wide-wall radius (Angstrom): about 2.5x the volume per atom of a
    liquid, so the molecules move freely until the piston compresses them."""
    return max(5.0, (3.0 * n_atoms * volume_per_atom / (4.0 * math.pi)) ** (1.0 / 3.0))


def _random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def pack_reactor(molecules: Sequence, radius: Optional[float] = None, *, seed: int = 0,
                 min_distance: float = 2.0, tries: int = 2000):
    """Place molecules (qcio/qcdata Structures, one per copy) at random
    orientations inside a sphere without close contacts. Returns (symbols,
    coords in Angstrom centered at the origin, radius, owner index per atom)."""
    if radius:
        return _pack(molecules, float(radius), seed, min_distance, tries)
    # No radius given: start from the atom-count estimate and grow it until
    # the molecules fit (it knows nothing of their shapes: two copies of an
    # elongated molecule need more room than their atom count says).
    radius = auto_radius(sum(len(m.symbols) for m in molecules))
    for _ in range(30):
        try:
            return _pack(molecules, radius, seed, min_distance, tries)
        except RuntimeError:
            radius *= 1.1
    return _pack(molecules, radius, seed, min_distance, tries)


def _pack(molecules: Sequence, radius: float, seed: int, min_distance: float, tries: int):
    rng = np.random.default_rng(seed)
    symbols, coords, owner = [], [], []
    # Largest first: they are the hardest to fit.
    order = sorted(range(len(molecules)), key=lambda k: -len(molecules[k].symbols))
    placed = np.zeros((0, 3))
    for k in order:
        m = molecules[k]
        xyz = np.asarray(m.geometry, dtype=float).reshape(-1, 3) / ANGSTROM_TO_BOHR
        xyz = xyz - xyz.mean(axis=0)
        extent = float(np.max(np.linalg.norm(xyz, axis=1))) if len(xyz) else 0.0
        room = radius - extent - 0.5
        for _ in range(tries):
            if room <= 0:
                break
            center = rng.uniform(-room, room, size=3)
            if np.linalg.norm(center) > room:
                continue
            trial = xyz @ _random_rotation(rng).T + center
            if len(placed) == 0 or np.min(np.linalg.norm(placed[:, None] - trial[None], axis=2)) >= min_distance:
                break
        else:
            raise RuntimeError(f"Could not fit all molecules into a {radius:.1f} Angstrom sphere; "
                               f"give a larger radius (or none, to size it automatically).")
        if room <= 0:
            raise RuntimeError(f"A molecule is larger than the {radius:.1f} Angstrom reactor; give a radius above "
                               f"{extent + 0.5:.1f} Angstrom (or none, to size it automatically).")
        placed = np.vstack([placed, trial])
        symbols.extend(m.symbols)
        coords.append(trial)
        owner.extend([k] * len(m.symbols))
    coords = np.vstack(coords)
    return symbols, coords - coords.mean(axis=0), radius, owner


# ---------------------------------------------------------------------------
# Discovery MD (xtb)
# ---------------------------------------------------------------------------

def _xtb_command(method: str, executable: Optional[str] = None, download: bool = True) -> list[str]:
    """xtb's command for `method` (gfn1/gfn2, or gxtb: the g-xTB build,
    fetched on first use: mepd.programs)."""
    from mepd import programs

    if method == "gxtb":
        exe = executable or programs.gxtb_executable(download)
        if not exe:
            raise RuntimeError("g-xTB MD needs the g-xTB xtb build: set GXTB_EXECUTABLE.")
        return [exe] + (["--gxtb"] if Path(exe).name != "gxtb" else [])
    exe = executable or programs.xtb_executable(download)
    if not exe:
        raise RuntimeError("The nanoreactor MD needs xtb (conda install -c conda-forge xtb), or g-xTB.")
    return [exe, "--gfn", method[-1]]


def _xtb_argv(cmd: list[str], input_file: str, *args: str) -> list[str]:
    """xtb's argv with the input file first: the g-xTB build reads the
    argument after --gxtb as its own, so `--gxtb file.xyz` loses the file
    ("No input file given")."""
    return [cmd[0], input_file, *cmd[1:], *args]


def missing_programs(method: str = "gfn2") -> list[str]:
    """What `method` needs that is neither installed nor fetched on first
    use (g-xTB is downloaded when needed, where a release build exists)."""
    if method == "level":   # a mepd engine: nothing extra to install
        return []
    try:
        _xtb_command(method, download=False)
        return []
    except RuntimeError:
        pass
    import platform

    from mepd import programs

    if not os.getenv("MEPD_NO_DOWNLOAD") and (platform.system(), platform.machine()) in programs._GXTB_ASSETS:
        return []
    return ["xtb"]


def _write_xyz(fp: Path, symbols, coords, comment: str = "") -> None:
    lines = [str(len(symbols)), comment]
    lines += [f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(symbols, coords)]
    fp.write_text("\n".join(lines) + "\n")


def read_xyz_frames(fp: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """All frames of a multi-frame xyz: (symbols, coords [F, N, 3] in Angstrom, comments)."""
    lines = Path(fp).read_text().splitlines()
    frames, comments, symbols, k = [], [], None, 0
    while k < len(lines):
        if not lines[k].strip():
            k += 1
            continue
        n = int(lines[k].split()[0])
        comments.append(lines[k + 1])
        block = [ln.split() for ln in lines[k + 2:k + 2 + n]]
        if symbols is None:
            symbols = [b[0] for b in block]
        frames.append([[float(v) for v in b[1:4]] for b in block])
        k += 2 + n
    return symbols or [], np.asarray(frames, dtype=float).reshape(len(frames), len(symbols or []), 3), comments


def piston_schedule(s: ReactorSettings, ramp_steps: int = 5) -> list[tuple[float, float]]:
    """(duration in ps, wall radius in Angstrom) segments of the run: each
    cycle is wide for `duty` of the period, then the wall closes to the
    narrow radius in `ramp_steps` steps over `ramp_fs` and stays there."""
    wide, narrow = float(s.radius), float(s.radius) * s.compress
    t_wide, t_narrow = s.period_ps * s.duty, s.period_ps * (1.0 - s.duty)
    ramp = min(s.ramp_fs / 1000.0, t_narrow) if s.compress < 1 else 0.0
    cycle = [(t_wide, wide)]
    if ramp > 0:
        cycle += [(ramp / ramp_steps, wide + (narrow - wide) * (k + 1) / ramp_steps) for k in range(ramp_steps)]
    cycle.append((t_narrow - ramp, narrow))
    out, t = [], 0.0
    while t < s.time_ps - 1e-9:
        for dur, r in cycle:
            dur = min(dur, s.time_ps - t)
            if dur > 1e-9:
                out.append((dur, r))
                t += dur
    return out


def run_reactor_md(symbols, coords_angstrom, *, charge: int, multiplicity: int, settings: ReactorSettings,
                   workdir: Path, executable: Optional[str] = None, on_event: OnEvent = None,
                   timeout_s: float = 24 * 3600) -> Path:
    """Run the piston MD with xtb in `workdir`; returns trajectory.xyz
    (every frame, Angstrom). Each piston segment is one xtb run restarted
    from the previous one's positions and velocities, with the wall at that
    segment's radius. A finished segment is not rerun (resume)."""
    settings.validate()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = _xtb_command(settings.method, executable)
    schedule = piston_schedule(settings)
    # For viewers: the wall radius at any time.
    (workdir / "schedule.json").write_text(json.dumps({"dump_fs": settings.dump_fs, "time_ps": settings.time_ps,
                                                       "segments": [[d, r] for d, r in schedule]}))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": os.environ.get("OMP_STACKSIZE", "1G")}
    uhf = max(0, int(multiplicity) - 1)
    common = ["--chrg", str(int(charge)), "--etemp", f"{settings.electronic_temperature:.1f}"]
    if uhf:
        common += ["--uhf", str(uhf)]
    # xtb's logfermi wall, kT ln(1 + exp(beta (r - R))), pushes an atom
    # outside it with a force of about kT*beta. At xtb's default (300 K,
    # beta 6/bohr) a hot molecule walks through it and the piston compresses
    # nothing; a steep wall kicks the atoms caught outside when it closes
    # and blows the reactor apart. So: beta = 1/bohr (a soft edge) and the
    # wall "temperature" that gives `wall_force`.
    beta = 1.0
    wall_t = settings.wall_force / HARTREE_TO_KCAL_PER_MOL / ANGSTROM_TO_BOHR / (beta * 3.166811563e-6)
    def wall(r):
        return ("$wall\n   potential=logfermi\n"
                f"   temp={wall_t:.1f}\n   beta={beta:.3f}\n"
                f"   sphere: {r * ANGSTROM_TO_BOHR:.6f}, all\n$end\n")

    def run(argv, what):
        done = subprocess.run(argv, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout_s)
        with (workdir / "md.log").open("a") as log:
            log.write(done.stdout[-20000:])
        if done.returncode != 0:
            raise RuntimeError(f"xtb {what} failed (exit {done.returncode}): {(done.stderr or done.stdout)[-800:]}")
        return done

    # A packed reactor is far from a minimum: relaxing it in the MD would
    # release hundreds of kcal/mol as heat (atoms get ejected). Relax it
    # first, inside the wide wall.
    if not (workdir / "reactor.xyz").exists():
        _write_xyz(workdir / "packed.xyz", symbols, coords_angstrom, "nanoreactor packed")
        (workdir / "opt.inp").write_text(wall(settings.radius))
        _emit(on_event, "md_relax", natoms=len(symbols))
        run(_xtb_argv(cmd, "packed.xyz", "--opt", "crude", "--input", "opt.inp", *common), "relaxation of the packed reactor")
        shutil.move(workdir / "xtbopt.xyz", workdir / "reactor.xyz")
        (workdir / "xtbrestart").unlink(missing_ok=True)
    t_done = 0.0
    for k, (dur, r) in enumerate(schedule):
        seg = workdir / f"segment_{k:03d}.xyz"
        restart_fp = workdir / f"segment_{k:03d}.mdrestart"
        if seg.exists() and restart_fp.exists():
            shutil.copy(restart_fp, workdir / "mdrestart")
            t_done += dur
            continue
        (workdir / "md.inp").write_text(
            "$md\n"
            f"   temp={settings.temperature:.2f}\n   time={dur:.6f}\n   dump={settings.dump_fs:.4f}\n"
            f"   step={settings.step_fs:.4f}\n   hmass=1\n   shake=0\n   nvt=true\n"
            f"   restart={'true' if k else 'false'}\n$end\n" + wall(r))
        done = run(_xtb_argv(cmd, "reactor.xyz", "--md", "--input", "md.inp", *common), f"MD (segment {k})")
        if not (workdir / "xtb.trj").exists():
            raise RuntimeError(f"xtb MD wrote no trajectory in segment {k}: {done.stdout[-800:]}")
        if "did not converge" in done.stdout:
            _emit(on_event, "warning", message=f"SCC did not converge somewhere in MD segment {k}")
        if "MD is unstable" in done.stdout:
            # xtb stopped the segment early (atoms too fast): the next segment
            # restarts from wherever it stopped, so the trajectory has a gap.
            _emit(on_event, "warning", message=f"xtb stopped MD segment {k} early (\"MD is unstable\"): "
                                               "the temperature is too high for this time step")
        shutil.move(workdir / "xtb.trj", seg)
        shutil.copy(workdir / "mdrestart", restart_fp)
        t_done += dur
        if on_event is not None:
            on_event("md_segment", {"segment": k + 1, "segments": len(schedule), "time_ps": t_done,
                                    "radius": r, "total_ps": settings.time_ps})
    traj = workdir / "trajectory.xyz"
    with traj.open("w") as out:
        for k in range(len(schedule)):
            out.write((workdir / f"segment_{k:03d}.xyz").read_text())
    return traj


# ---------------------------------------------------------------------------
# Discovery MD on any mepd engine (MLIPs, ASE calculators, g-xTB, ...)
# ---------------------------------------------------------------------------

# Unit conversions (atomic units for forces/energies, bohr and fs for motion).
_ACC = 0.93766                 # (Eh/bohr)/amu -> bohr/fs^2
_KIN = 1.066551                # amu bohr^2/fs^2 -> Eh
_KB = 3.166811563e-6           # Eh/K


def _masses(symbols) -> np.ndarray:
    from rdkit import Chem

    table = Chem.GetPeriodicTable()
    return np.array([table.GetAtomicWeight(s) for s in symbols])


def _wall(pos_bohr: np.ndarray, radius_bohr: float, force: float, beta: float = 1.0):
    """xtb's logfermi wall, kT ln(1 + exp(beta (r - R))), with kT*beta =
    `force` (Eh/bohr): (energy, gradient) per atom, around the origin."""
    r = np.linalg.norm(pos_bohr, axis=1)
    x = np.clip(beta * (r - radius_bohr), -50, 50)
    kt = force / beta
    energy = float(np.sum(kt * np.logaddexp(0.0, x)))
    sig = 1.0 / (1.0 + np.exp(-x))
    grad = (kt * beta * sig / np.maximum(r, 1e-9))[:, None] * pos_bohr
    return energy, grad


class _EngineForces:
    """Energy and gradient of the reactor from a mepd engine, plus the wall."""

    def __init__(self, engine, symbols, charge: int, multiplicity: int, wall_force_au: float):
        self.engine, self.symbols = engine, list(symbols)
        self.charge, self.multiplicity = int(charge), int(multiplicity)
        self.wall_force = wall_force_au
        self.calls = 0

    def __call__(self, pos_bohr: np.ndarray, radius_bohr: float):
        from qcdata import Structure

        from mepd.nodes.node import StructureNode

        node = StructureNode(structure=Structure(symbols=self.symbols, geometry=pos_bohr, charge=self.charge,
                                                 multiplicity=self.multiplicity))
        grad = np.asarray(self.engine.compute_gradients([node]))[0].reshape(-1, 3)
        energy = float(node.energy) if node._cached_energy is not None else float("nan")
        e_wall, g_wall = _wall(pos_bohr, radius_bohr, self.wall_force)
        self.calls += 1
        return energy + e_wall, grad + g_wall


def _relax_in_wall(forces: _EngineForces, pos: np.ndarray, radius_bohr: float, masses: np.ndarray,
                   steps: int = 300, fmax: float = 5e-3, trajectory: Optional[Path] = None,
                   symbols: Sequence = (), every: int = 3) -> np.ndarray:
    """FIRE minimization inside the wall: takes the strain out of a packed
    reactor (else it turns into heat and blows atoms out). `trajectory`:
    every few steps appended there (multi-frame xyz, like xtb's xtbopt.log),
    for live viewers."""
    import contextlib

    v = np.zeros_like(pos)
    dt, alpha, n_pos = 0.5, 0.1, 0
    with (Path(trajectory).open("w") if trajectory is not None else contextlib.nullcontext()) as trj:
        for k in range(steps):
            energy, g = forces(pos, radius_bohr)
            if trj is not None and k % every == 0:
                xyz = pos / ANGSTROM_TO_BOHR
                trj.write(f"{len(symbols)}\n energy: {energy:.10f}\n"
                          + "".join(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n" for sym, (x, y, z) in zip(symbols, xyz)))
                trj.flush()
            f = -g
            if np.max(np.linalg.norm(f, axis=1)) < fmax:
                break
            p = float(np.sum(f * v))
            if p > 0:
                fn = np.linalg.norm(f) or 1.0
                v = (1 - alpha) * v + alpha * np.linalg.norm(v) * f / fn
                n_pos += 1
                if n_pos > 5:
                    dt, alpha = min(dt * 1.1, 2.0), alpha * 0.99
            else:
                v[:] = 0.0
                dt, alpha, n_pos = dt * 0.5, 0.1, 0
            v = v + dt * _ACC * f / masses[:, None]
            step = dt * v
            norm = np.linalg.norm(step, axis=1, keepdims=True)
            pos = pos + np.where(norm > 0.2, step * 0.2 / np.maximum(norm, 1e-12), step)   # at most 0.2 bohr per atom
    return pos


def run_engine_md(symbols, coords_angstrom, *, charge: int, multiplicity: int, settings: ReactorSettings,
                  engine, workdir: Path, on_event: OnEvent = None, friction_per_fs: float = 0.01) -> Path:
    """The piston MD with gradients from any mepd engine (the level of
    theory of a profile: an MLIP, an ASE calculator, g-xTB, ...). Langevin
    dynamics (BAOAB), the same logfermi wall as the xtb path, and the same
    files: segment_k.xyz per piston segment (with a restart), the running
    segment in xtb.trj (so the live view sees it), trajectory.xyz at the
    end. A finished segment is not rerun (resume)."""
    settings.validate()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    schedule = piston_schedule(settings)
    (workdir / "schedule.json").write_text(json.dumps({"dump_fs": settings.dump_fs, "time_ps": settings.time_ps,
                                                       "segments": [[d, r] for d, r in schedule],
                                                       "engine": type(engine).__name__}))
    masses = _masses(symbols)
    wall_au = settings.wall_force / HARTREE_TO_KCAL_PER_MOL / ANGSTROM_TO_BOHR
    forces = _EngineForces(engine, symbols, charge, multiplicity, wall_au)
    kt = _KB * settings.temperature / _KIN               # amu bohr^2/fs^2
    rng = np.random.default_rng(settings.seed)
    dt = settings.step_fs
    every = max(1, int(round(settings.dump_fs / dt)))
    c1 = math.exp(-friction_per_fs * dt)
    c2 = np.sqrt((1 - c1 * c1) * kt / masses)[:, None]

    if (workdir / "reactor.xyz").exists():
        _, x0, _ = read_xyz_frames(workdir / "reactor.xyz")
        pos = x0[0] * ANGSTROM_TO_BOHR
    else:
        _write_xyz(workdir / "packed.xyz", symbols, coords_angstrom, "nanoreactor packed")
        _emit(on_event, "md_relax", natoms=len(symbols))
        pos = _relax_in_wall(forces, np.asarray(coords_angstrom) * ANGSTROM_TO_BOHR,
                             settings.radius * ANGSTROM_TO_BOHR, masses, trajectory=workdir / "relax.xyz",
                             symbols=symbols)
        _write_xyz(workdir / "reactor.xyz", symbols, pos / ANGSTROM_TO_BOHR, "nanoreactor relaxed")
    vel = rng.normal(size=pos.shape) * np.sqrt(kt / masses)[:, None]
    vel -= (masses[:, None] * vel).sum(axis=0) / masses.sum()      # no drift of the whole reactor

    t_done, grad = 0.0, None
    for k, (dur, r) in enumerate(schedule):
        seg, restart = workdir / f"segment_{k:03d}.xyz", workdir / f"segment_{k:03d}.restart.npz"
        if seg.exists() and restart.exists():
            state = np.load(restart)
            pos, vel, grad = state["pos"], state["vel"], None
            t_done += dur
            continue
        radius_bohr = r * ANGSTROM_TO_BOHR
        n_steps = max(1, int(round(dur * 1000.0 / dt)))
        running = workdir / "xtb.trj"
        running.unlink(missing_ok=True)
        with running.open("w") as trj:
            energy, grad = forces(pos, radius_bohr)
            for step in range(1, n_steps + 1):
                # BAOAB: half kick, half drift, thermostat, half drift, half kick
                vel -= 0.5 * dt * _ACC * grad / masses[:, None]
                pos = pos + 0.5 * dt * vel
                vel = c1 * vel + c2 * rng.normal(size=vel.shape)
                pos = pos + 0.5 * dt * vel
                energy, grad = forces(pos, radius_bohr)
                vel -= 0.5 * dt * _ACC * grad / masses[:, None]
                if step % every == 0:
                    xyz = pos / ANGSTROM_TO_BOHR
                    trj.write(f"{len(symbols)}\n energy: {energy:.10f} \n")
                    trj.write("".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(symbols, xyz)))
                    trj.flush()
                if not np.all(np.isfinite(pos)):
                    raise RuntimeError(f"the MD blew up in segment {k} (step {step}): non-finite positions")
        np.savez(restart, pos=pos, vel=vel)
        shutil.move(running, seg)
        t_done += dur
        temp = float(np.sum(masses[:, None] * vel ** 2) * _KIN / (3 * len(symbols) * _KB))
        _emit(on_event, "md_segment", segment=k + 1, segments=len(schedule), time_ps=t_done, radius=r,
              total_ps=settings.time_ps, temperature=temp)
    traj = workdir / "trajectory.xyz"
    with traj.open("w") as out:
        for k in range(len(schedule)):
            out.write((workdir / f"segment_{k:03d}.xyz").read_text())
    return traj


# ---------------------------------------------------------------------------
# Bonds along the trajectory, and reaction events
# ---------------------------------------------------------------------------

Pair = tuple[int, int]


def _radius_sum(symbols) -> np.ndarray:
    r = np.array([COVALENT_RADII.get(s, 1.5) for s in symbols])
    return r[:, None] + r[None, :]


def perceive_bonds(symbols, coords, factor: float = 1.25) -> set[Pair]:
    d = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    i, j = np.nonzero(np.triu(d < factor * _radius_sum(symbols), 1))
    return {(int(a), int(b)) for a, b in zip(i, j)}


def _denoise(states: list[bool], min_len: int) -> list[bool]:
    """Remove bond states that last fewer than `min_len` frames by merging
    them into their neighbours (shortest first). The first run stays (the
    start is taken as it is); a short run at the end is not yet a new state."""
    runs = [[s, len(list(g))] for s, g in itertools.groupby(states)]
    while True:
        short = [k for k in range(1, len(runs)) if runs[k][1] < min_len]
        if not short:
            break
        k = min(short, key=lambda k: runs[k][1])
        if k == len(runs) - 1:
            runs[k - 1][1] += runs[k][1]
            runs.pop()
        else:
            runs[k - 1][1] += runs[k][1] + runs[k + 1][1]
            del runs[k:k + 2]
    return [s for s, n in runs for _ in range(n)]


@dataclass
class BondHistory:
    """Stable bonds along the trajectory: the first frame's bonds plus every
    change after denoising."""
    n_frames: int
    initial: set
    changes: list  # (frame, i, j, formed)

    def bonds_at(self, frame: int) -> set[Pair]:
        bonds = set(self.initial)
        for f, i, j, formed in self.changes:
            if f > frame:
                break
            (bonds.add if formed else bonds.discard)((i, j))
        return bonds


def bond_history(symbols, frames: np.ndarray, s: DetectSettings, dt_fs: float) -> BondHistory:
    rsum = _radius_sum(symbols)
    n = len(symbols)
    iu = np.triu_indices(n, 1)
    # Only pairs that ever come within bonding range can be bonded: find
    # them first, so memory grows with those, not with all N^2 pairs.
    dmin = np.full(len(iu[0]), np.inf)
    for xyz in frames:
        np.minimum(dmin, np.linalg.norm(xyz[:, None] - xyz[None], axis=2)[iu], out=dmin)
    near = np.nonzero(dmin < np.maximum(s.form, 1.25) * rsum[iu])[0]
    iu = (iu[0][near], iu[1][near])
    form, brk = s.form * rsum[iu], s.break_ * rsum[iu]

    def dist(xyz):
        return np.linalg.norm(xyz[iu[0]] - xyz[iu[1]], axis=1)

    # A hydrogen always belongs somewhere: one caught between two partners
    # (beyond breaking range of the old, not yet in forming range of the
    # new) stays on the nearer one while it is within breaking range,
    # instead of counting as a free atom for a few frames.
    hydrogens = [a for a, sym in enumerate(symbols) if sym == "H"]
    pairs_of = {a: np.nonzero((iu[0] == a) | (iu[1] == a))[0] for a in hydrogens}
    state = dist(frames[0]) < 1.25 * rsum[iu]
    raw = np.zeros((len(frames), len(state)), dtype=bool)
    for f, xyz in enumerate(frames):
        d = dist(xyz)
        state = np.where(state, d <= brk, d < form)
        for a, idx in pairs_of.items():
            if len(idx) and not state[idx].any():
                ratio = d[idx] / rsum[iu][idx]
                k = int(np.argmin(ratio))
                if ratio[k] <= s.break_:
                    state[idx[k]] = True
        raw[f] = state
    min_len = max(1, int(round(s.min_lifetime_fs / dt_fs)))
    changes = []
    initial = set()
    for p in np.nonzero(raw.any(axis=0))[0]:
        col = raw[:, p]
        i, j = int(iu[0][p]), int(iu[1][p])
        if col.all():
            initial.add((i, j))
            continue
        clean = _denoise(col.tolist(), min_len)
        if clean[0]:
            initial.add((i, j))
        for f in range(1, len(clean)):
            if clean[f] != clean[f - 1]:
                changes.append((f, i, j, bool(clean[f])))
    changes.sort()
    return BondHistory(len(frames), initial, changes)


def components(n_atoms: int, bonds: set[Pair], atoms: Optional[set] = None) -> list[tuple[int, ...]]:
    """Connected components (molecules) of the bond graph, as sorted atom tuples."""
    atoms = set(range(n_atoms)) if atoms is None else set(atoms)
    parent = {a: a for a in atoms}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in bonds:
        if i in parent and j in parent:
            parent[find(i)] = find(j)
    groups: dict[int, list] = {}
    for a in atoms:
        groups.setdefault(find(a), []).append(a)
    return sorted(tuple(sorted(g)) for g in groups.values())


@dataclass
class Event:
    start: int                 # first frame with a changed bond
    end: int                   # last frame with a changed bond
    atoms: tuple               # reactor atoms of the subsystem
    reactant_frame: int
    product_frame: int
    reactants: list            # atom tuples (molecules) before
    products: list             # atom tuples after
    changes: list              # (frame, i, j, formed)


def detect_events(n_atoms: int, hist: BondHistory, s: DetectSettings, dt_fs: float) -> list[Event]:
    """Group bond changes into reaction events and cut out their subsystems."""
    ch = hist.changes
    if not ch:
        return []
    window = max(1, int(round(s.merge_window_fs / dt_fs)))
    # Per change: the molecules its two atoms belong to just before and after.
    touched = []
    for f, i, j, _ in ch:
        before, after = hist.bonds_at(f - 1), hist.bonds_at(f)
        mols = set()
        for bonds in (before, after):
            for comp in components(n_atoms, bonds):
                if i in comp or j in comp:
                    mols.update(comp)
        touched.append(mols)
    parent = list(range(len(ch)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(ch)):
        for b in range(a + 1, len(ch)):
            if ch[b][0] - ch[a][0] > window:
                break
            if touched[a] & touched[b]:
                parent[find(a)] = find(b)
    clusters: dict[int, list[int]] = {}
    for a in range(len(ch)):
        clusters.setdefault(find(a), []).append(a)
    lead, lag = max(1, int(round(s.lead_fs / dt_fs))), max(1, int(round(s.lag_fs / dt_fs)))
    events = []
    for members in sorted(clusters.values(), key=lambda m: ch[m[0]][0]):
        start, end = ch[members[0]][0], max(ch[m][0] for m in members)
        before, after = hist.bonds_at(start - 1), hist.bonds_at(end)
        atoms = set().union(*(touched[m] for m in members))
        while True:   # closure: whole molecules on both sides
            grown = set(atoms)
            for bonds in (before, after):
                for comp in components(n_atoms, bonds):
                    if grown.intersection(comp):
                        grown.update(comp)
            if grown == atoms:
                break
            atoms = grown
        mols_b = components(n_atoms, {p for p in before if p[0] in atoms}, atoms)
        mols_a = components(n_atoms, {p for p in after if p[0] in atoms}, atoms)
        # Split into independent reactions: molecules before and after are
        # linked when they share atoms. A group whose molecules come out
        # as they went in (same atoms, same bonds) only collided; a shuttle
        # that hands over one H and takes another is linked to the rest.
        for grp_b, grp_a in _atom_sharing_groups(mols_b, mols_a):
            if sorted(grp_b) == sorted(grp_a) and all(
                    {p for p in before if p[0] in m} == {p for p in after if p[0] in m} for m in grp_b):
                continue
            g_atoms = set().union(*grp_b)
            # Both atoms inside: a contact between two groups (a bond that formed
            # and broke between molecules that react separately) is neither's.
            members_g = [ch[m] for m in members if ch[m][1] in g_atoms and ch[m][2] in g_atoms]
            if not members_g:
                continue
            events.append(Event(members_g[0][0], max(c[0] for c in members_g), tuple(sorted(g_atoms)),
                                max(0, members_g[0][0] - lead), min(hist.n_frames - 1, max(c[0] for c in members_g) + lag),
                                sorted(grp_b), sorted(grp_a), members_g))
    events.sort(key=lambda e: (e.start, e.atoms))
    # Endpoint frames must not reach into a neighbouring event on the same atoms.
    for k, ev in enumerate(events):
        for other in events[:k]:
            if set(other.atoms) & set(ev.atoms) and other.end < ev.start:
                ev.reactant_frame = max(ev.reactant_frame, other.end)
        for other in events[k + 1:]:
            if set(other.atoms) & set(ev.atoms) and other.start > ev.end:
                ev.product_frame = min(ev.product_frame, other.start - 1)
    return events


def _atom_sharing_groups(before: list, after: list) -> list[tuple[list, list]]:
    """Molecules before and after, grouped so that two molecules sharing an
    atom are in one group: [(molecules before, molecules after), ...]."""
    owner = {}
    for k, m in enumerate(before):
        for a in m:
            owner[a] = k
    parent = list(range(len(before)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in after:
        ks = {owner[a] for a in m}
        first = ks.pop()
        for k in ks:
            parent[find(k)] = find(first)
    groups: dict[int, tuple[list, list]] = {}
    for k, m in enumerate(before):
        groups.setdefault(find(k), ([], []))[0].append(m)
    for m in after:
        groups[find(owner[m[0]])][1].append(m)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Species identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Label:
    smiles: str
    charge: int
    multiplicity: int
    formula: str

    @property
    def key(self) -> str:
        return f"{self.smiles}|{self.charge}|{self.multiplicity}"


def _round(x: float) -> int:
    """Half away from zero (Python's round(0.5) is 0)."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def _formula(symbols) -> str:
    c = Counter(symbols)
    order = (["C", "H"] if "C" in c else []) + sorted(k for k in c if k not in ("C", "H") or "C" not in c)
    return "".join(f"{k}{c[k] if c[k] > 1 else ''}" for k in order)


class Labeler:
    """SMILES, charge and spin of molecules cut from the reactor. A
    molecule's charge is not in the geometry. With partial charges of that
    frame (`hint`, atom -> charge), each molecule's charge is their rounded
    sum (as in the ab initio nanoreactor); the options for each molecule are
    charges that have a Lewis structure, and the set of molecules (summing
    to their known total) closest to the hint, then with the fewest charged
    molecules, then the fewest unpaired electrons, wins. Without a hint,
    neutral molecules (radicals if need be) are preferred: homolysis, not
    ion pairs."""

    def __init__(self, symbols: Sequence[str], max_charge: int = 2):
        self.symbols = list(symbols)
        self.max_charge = max_charge
        self._cache: dict = {}

    def candidates(self, atoms: tuple, bonds: set[Pair]) -> list[tuple[int, int, str]]:
        """(charge, unpaired electrons, smiles) options for one molecule."""
        local = {a: k for k, a in enumerate(atoms)}
        edges = tuple(sorted((local[i], local[j]) for i, j in bonds if i in local and j in local))
        syms = tuple(self.symbols[a] for a in atoms)
        key = (syms, edges)
        if key in self._cache:
            return self._cache[key]
        from mepd.discovery.network_expansion import METAL_CHARGES, _lewis_mol, _smiles, lewis_smiles

        out = []
        electrons = sum(_Z.get(s, 0) for s in syms)
        for q in sorted(range(-self.max_charge, self.max_charge + 1), key=abs):
            if electrons - q < 0:
                continue
            mult = 1 if (electrons - q) % 2 == 0 else 2
            if len(syms) == 1:   # a lone atom or ion: no bonds to assign
                sym = syms[0]
                smi = f"[{sym}{'+' * q if 0 < q < 2 else ''}{'-' * -q if -2 < q < 0 else ''}" \
                      f"{f'+{q}' if q >= 2 else ''}{f'{q}' if q <= -2 else ''}]"
                out.append((q, (electrons - q) % 2, smi))
                continue
            if any(s in METAL_CHARGES for s in syms):
                smi = lewis_smiles(syms, edges, q, mult, allow_radicals=True, allow_zwitterions=True)
                if smi:
                    out.append((q, mult - 1, smi))
                continue
            mol = _lewis_mol(syms, edges, q, mult, allow_radicals=True, allow_zwitterions=True)
            if mol is None and mult == 2:
                mol = _radical_mol(syms, edges, q)
            if mol is None and mult == 1:   # e.g. a ring opened by one broken bond: a diradical
                mol = _radical_mol(syms, edges, q, centres=2)
            if mol is not None:
                out.append((q, sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()), _smiles(mol)))
        if not out and not any(x in METAL_CHARGES for x in syms):
            mol = _valence_mol(syms, edges)   # open shell (a broken ring, a carbene, ...): radicals where valence is left
            if mol is not None:
                out.append((0, sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()), _smiles(mol)))
        if not out:   # no structure at all: a readable name that still tells different bondings apart
            import hashlib

            tag = hashlib.sha1(repr(edges).encode()).hexdigest()[:4]
            out.append((0, electrons % 2, f"{_formula(syms)} (unusual bonding #{tag})"))
        self._cache[key] = out
        return out

    def assign(self, mols: Sequence[tuple], bonds: set[Pair], total_charge: Optional[int],
               hint: Optional[Sequence[float]] = None) -> list[Label]:
        """Labels for molecules that together carry `total_charge` (None = free)."""
        cands = [self.candidates(m, bonds) for m in mols]
        target = [_round(sum(hint[a] for a in m)) if hint is not None else 0 for m in mols]
        best, best_score = None, None
        for combo in itertools.islice(itertools.product(*cands), 20000):
            q = sum(c[0] for c in combo)
            if total_charge is not None and q != total_charge:
                continue
            score = (sum(abs(c[0] - t) for c, t in zip(combo, target)), sum(1 for c in combo if c[0]),
                     sum(c[1] for c in combo))
            if best_score is None or score < best_score:
                best, best_score = combo, score
        if best is None:   # no assignment sums to the total: take each molecule's best on its own
            best = [min(c, key=lambda o: (abs(o[0] - t), abs(o[0]), o[1])) for c, t in zip(cands, target)]
        labels = []
        for m, (q, unpaired, smi) in zip(mols, best):
            syms = [self.symbols[a] for a in m]
            electrons = sum(_Z.get(s, 0) for s in syms) - q
            labels.append(Label(smi, int(q), 1 if electrons % 2 == 0 else 2, _formula(syms)))
        return labels


_VALENCE = {"H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1, "Si": 4, "P": 3, "S": 2, "Cl": 1, "Se": 2, "Br": 1, "I": 1}


def _valence_mol(syms, edges):
    """A neutral structure for any graph whose atoms are not over-bonded:
    each atom gets its usual valence, multiple bonds go where neighbours
    both have valence left (maximum matching, repeated for triple bonds),
    and what is left is radical electrons. The last resort when no
    closed-shell or one-radical structure exists."""
    import networkx as nx
    from rdkit import Chem

    n = len(syms)
    if any(s not in _VALENCE for s in syms):
        return None
    deg = [0] * n
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    free = [_VALENCE[s] - d for s, d in zip(syms, deg)]
    if any(f < 0 for f in free):
        return None
    order = {tuple(sorted(e)): 1 for e in edges}
    for _ in range(2):   # double, then triple bonds
        g = nx.Graph([(i, j) for i, j in order if free[i] > 0 and free[j] > 0 and order[(i, j)] < 3])
        if g.number_of_edges() == 0:
            break
        for i, j in nx.max_weight_matching(g, maxcardinality=True):
            key = tuple(sorted((i, j)))
            order[key] += 1
            free[i] -= 1
            free[j] -= 1
    rw = Chem.RWMol()
    for s, f in zip(syms, free):
        a = Chem.Atom(s)
        a.SetNoImplicit(True)
        a.SetNumRadicalElectrons(f)
        rw.AddAtom(a)
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for (i, j), k in order.items():
        rw.AddBond(int(i), int(j), kinds[k])
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return rw.GetMol()


def _radical_mol(syms, edges, q: int, centres: int = 1):
    """A structure with `centres` radical centres (1: a doublet, 2: a
    singlet diradical): xyz2mol assigns closed shells only, so take the
    closed-shell ion `centres` electrons away (charge q -/+ centres) and turn
    its charged atoms into the radical centres (OH- -> OH., CH3+ -> CH3.,
    a dianion with two carbanions -> a diradical). None if no such
    structure has normal valences."""
    import itertools

    from rdkit import Chem

    from mepd.discovery.network_expansion import _lewis_mol

    for ion, sign in ((q - centres, -1), (q + centres, 1)):
        mol = _lewis_mol(syms, edges, ion, 1, allow_radicals=True, allow_zwitterions=True)
        if mol is None:
            continue
        charged = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() * sign > 0]
        for picks in itertools.islice(itertools.combinations(charged, centres), 50):
            rw = Chem.RWMol(mol)
            for k in picks:
                a = rw.GetAtomWithIdx(k)
                a.SetFormalCharge(a.GetFormalCharge() - sign)
                a.SetNumRadicalElectrons(a.GetNumRadicalElectrons() + 1)
            if sum(a.GetFormalCharge() for a in rw.GetAtoms()) != q:
                continue
            try:
                Chem.SanitizeMol(rw)
            except Exception:
                continue
            return rw.GetMol()
    return None


def frame_partial_charges(symbols, coords_angstrom, *, charge: int, multiplicity: int, method: str,
                          electronic_temperature: float, workdir: Path,
                          executable: Optional[str] = None) -> Optional[np.ndarray]:
    """Atomic partial charges of one reactor frame (xtb single point), or
    None if xtb fails."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    _write_xyz(workdir / "frame.xyz", symbols, coords_angstrom)
    (workdir / "charges").unlink(missing_ok=True)
    argv = _xtb_argv(_xtb_command(method, executable), "frame.xyz", "--sp", "--chrg", str(int(charge)),
                     "--etemp", f"{electronic_temperature:.1f}")
    if multiplicity > 1:
        argv += ["--uhf", str(int(multiplicity) - 1)]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    try:
        subprocess.run(argv, cwd=workdir, env=env, capture_output=True, text=True, timeout=600)
        q = np.loadtxt(workdir / "charges")
    except Exception:
        return None
    return q if q.shape == (len(symbols),) else None


_Z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12,
      "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Fe": 26, "Cu": 29, "Zn": 30,
      "Se": 34, "Br": 35, "I": 53}


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

@dataclass
class SpeciesRecord:
    id: int
    smiles: str
    charge: int
    multiplicity: int
    formula: str
    first_fs: float
    count: int = 0                  # molecules of it seen at reactant/product frames
    initial: int = 0                # copies in the starting reactor
    instances: list = field(default_factory=list)   # (frame, atoms) to cut geometries from
    energy: Optional[float] = None  # Hartree, refinement level (lowest instance)
    file: Optional[str] = None      # optimized geometry
    md_file: Optional[str] = None   # as cut from the trajectory (first instance)
    note: str = ""


@dataclass
class ReactionRecord:
    id: int
    reactants: list                 # species ids (with repeats), in the orientation first seen
    products: list
    shuttles: list                  # species on both sides
    label: str
    first_fs: float
    count: int = 0                  # times seen in this direction
    reverse_count: int = 0          # ... and the other way
    instances: list = field(default_factory=list)   # dicts: frames, atoms, direction
    delta_e_kcal: Optional[float] = None            # sum(isolated products) - sum(isolated reactants)
    complex: dict = field(default_factory=dict)     # optimized subsystem endpoints
    ts: dict = field(default_factory=dict)


def _side_key(ids: Sequence[int]) -> tuple:
    return tuple(sorted(ids))


def build_network(symbols, frames: np.ndarray, hist: BondHistory, events: list[Event], labeler: Labeler,
                  total_charge: int, dt_fs: float, max_instances: int = 3,
                  charges_at: Optional[Callable[[int], Optional[np.ndarray]]] = None) -> tuple[list, list, list]:
    """Species and reactions (merged over events) from the detected events.
    `charges_at(frame)` gives that frame's atomic partial charges (or None),
    which decide each molecule's charge. Returns (species, reactions,
    per-event records)."""
    charges_at = charges_at or (lambda frame: None)
    species: dict[str, SpeciesRecord] = {}

    def sid(label: Label, t_fs: float, frame: int, atoms: tuple) -> int:
        rec = species.get(label.key)
        if rec is None:
            rec = species[label.key] = SpeciesRecord(len(species), label.smiles, label.charge, label.multiplicity,
                                                     label.formula, t_fs)
        rec.count += 1
        if len(rec.instances) < max_instances:
            rec.instances.append((int(frame), tuple(atoms)))
        return rec.id

    # The starting reactor's molecules first.
    b0 = hist.bonds_at(0)
    mols0 = components(len(symbols), b0)
    for m, lab in zip(mols0, labeler.assign(mols0, b0, total_charge, charges_at(0))):
        species_id = sid(lab, 0.0, 0, m)
        next(r for r in species.values() if r.id == species_id).initial += 1
    reactions: dict[tuple, ReactionRecord] = {}
    ev_records = []
    for n, ev in enumerate(events):
        bb, ba = hist.bonds_at(ev.start - 1), hist.bonds_at(ev.end)
        hint_b, hint_a = charges_at(ev.reactant_frame), charges_at(ev.product_frame)
        q_sub = _round(sum(hint_b[a] for a in ev.atoms)) if hint_b is not None else None
        lab_b = labeler.assign(ev.reactants, bb, q_sub, hint_b)
        q = sum(l.charge for l in lab_b)
        lab_a = labeler.assign(ev.products, ba, q, hint_a)
        t_fs = ev.start * dt_fs
        r_ids = [sid(l, t_fs, ev.reactant_frame, m) for l, m in zip(lab_b, ev.reactants)]
        p_ids = [sid(l, t_fs, ev.product_frame, m) for l, m in zip(lab_a, ev.products)]
        rec_ev = {"event": n, "start_fs": t_fs, "end_fs": ev.end * dt_fs, "atoms": list(ev.atoms),
                  "reactant_frame": ev.reactant_frame, "product_frame": ev.product_frame,
                  "reactants": r_ids, "products": p_ids, "charge": q,
                  "bond_changes": [{"frame": f, "atoms": [i, j], "formed": formed} for f, i, j, formed in ev.changes]}
        ev_records.append(rec_ev)
        names_now = {r.id: r.smiles for r in species.values()}
        rec_ev["label"] = reaction_label(r_ids, p_ids, names_now)
        rec_ev["shuttles"] = sorted((Counter(r_ids) & Counter(p_ids)).elements())
        if _side_key(r_ids) == _side_key(p_ids):
            rec_ev["reaction"] = None    # a degenerate exchange (e.g. H between two waters)
            continue
        fwd, rev = (_side_key(r_ids), _side_key(p_ids)), (_side_key(p_ids), _side_key(r_ids))
        direction = "forward"
        rxn = reactions.get(fwd)
        if rxn is None and rev in reactions:
            rxn, direction = reactions[rev], "reverse"
        if rxn is None:
            both = Counter(r_ids) & Counter(p_ids)
            rxn = reactions[fwd] = ReactionRecord(len(reactions), list(fwd[0]), list(fwd[1]),
                                                  sorted(both.elements()), "", t_fs)
        if direction == "forward":
            rxn.count += 1
        else:
            rxn.reverse_count += 1
        rec_ev["reaction"], rec_ev["direction"] = rxn.id, direction
        # Its identity, as the refined reactions carry it: the live analysis
        # renumbers events as the trajectory grows (an event can absorb what
        # follows it), so a reaction finds its events by this, not by number.
        rec_ev["key"] = reaction_key(rxn, {r.id: r for r in species.values()})
        if len(rxn.instances) < max_instances:
            rxn.instances.append({"event": n, "direction": direction, "atoms": list(ev.atoms), "charge": q,
                                  "reactant_frame": ev.reactant_frame, "product_frame": ev.product_frame,
                                  "start_fs": t_fs})
    sp = sorted(species.values(), key=lambda r: r.id)
    names = {r.id: r.smiles for r in sp}
    rx = sorted(reactions.values(), key=lambda r: r.id)
    for r in rx:
        r.label = reaction_label(r.reactants, r.products, names)
    return sp, rx, ev_records


def reaction_label(reactants, products, names: dict) -> str:
    """'A + B -> C + B': the species that change first, shuttles last."""
    both = Counter(reactants) & Counter(products)

    def side(ids):
        c = Counter(ids) - both
        parts = [f"{n} {names[i]}" if n > 1 else names[i] for i, n in sorted(c.items())]
        parts += [f"{n} {names[i]}" if n > 1 else names[i] for i, n in sorted(both.items())]
        return " + ".join(parts)
    return f"{side(reactants)} -> {side(products)}"


# ---------------------------------------------------------------------------
# Refinement at the level of theory of the workspace
# ---------------------------------------------------------------------------

def _structure(symbols, coords_angstrom, charge: int, multiplicity: int):
    from qcdata import Structure

    return Structure(symbols=list(symbols), geometry=np.asarray(coords_angstrom) * ANGSTROM_TO_BOHR,
                     charge=int(charge), multiplicity=int(multiplicity))


def _cut(symbols, frames, frame: int, atoms: Sequence[int]) -> tuple[list, np.ndarray]:
    return [symbols[a] for a in atoms], np.asarray(frames[frame][list(atoms)])


def _complex_multiplicity(symbols, charge: int) -> int:
    return 1 if (sum(_Z.get(s, 0) for s in symbols) - charge) % 2 == 0 else 2


def _optimize_one(engine, structure, maxiter: int):
    from mepd.discovery.network_expansion import _optimize
    from mepd.nodes.node import StructureNode

    out = _optimize(engine, [StructureNode(structure=structure)], maxiter, None)[0]
    if isinstance(out, Exception):
        raise out
    if out._cached_energy is None:
        engine.compute_energies([out])
    return out


def _same_bonds(symbols, coords_a, bonds_expected: set[Pair]) -> bool:
    return perceive_bonds(symbols, coords_a) == bonds_expected


def species_key(rec) -> str:
    """A species' identity across analyses (live and final): SMILES, charge, spin."""
    return f"{rec.smiles}|{rec.charge}|{rec.multiplicity}"


def reaction_key(rxn, by_id: dict) -> str:
    """A reaction's identity across analyses, whichever way it is written."""
    def side(ids):
        return "+".join(sorted(species_key(by_id[i]) for i in ids))
    a, b = side(rxn.reactants), side(rxn.products)
    return min(f"{a}>{b}", f"{b}>{a}")


def _species_cuts(symbols, frames, hist: BondHistory, rec) -> list:
    """(symbols, coords, bonds it must keep) per instance of a species."""
    out = []
    for frame, atoms in rec.instances:
        syms, xyz = _cut(symbols, frames, frame, atoms)
        local = {a: k for k, a in enumerate(atoms)}
        out.append((syms, xyz, {(local[i], local[j]) for i, j in hist.bonds_at(frame) if i in local and j in local}))
    return out


def refine_species_one(rec, cuts: list, engine, sp_dir: Path, maxiter: int, name: Optional[str] = None) -> None:
    """Optimize a species from each of its instances; keep the lowest whose
    bonds stay as they were (else note why)."""
    best = None
    for syms, xyz, expect in cuts:
        try:
            node = _optimize_one(engine, _structure(syms, xyz, rec.charge, rec.multiplicity), maxiter)
        except Exception as exc:
            rec.note = f"optimization failed: {type(exc).__name__}: {exc}"[:300]
            continue
        opt_xyz = np.asarray(node.coords) / ANGSTROM_TO_BOHR
        if len(syms) > 1 and not _same_bonds(syms, opt_xyz, expect):
            rec.note = "bonds changed on optimization (not a minimum at this level)"
            continue
        if best is None or float(node.energy) < float(best[0].energy):
            best = (node, syms, opt_xyz)
    if best is not None:
        rec.energy = float(best[0].energy)
        rec.note = ""
        fp = Path(sp_dir) / f"species_{name if name is not None else rec.id}.xyz"
        _write_xyz(fp, best[1], best[2], f"{rec.smiles} charge={rec.charge} mult={rec.multiplicity} "
                                         f"energy={rec.energy:.10f}")
        rec.file = str(fp)


def _reaction_cuts(symbols, frames, hist: BondHistory, rxn) -> list:
    """Per instance, oriented as the reaction is written: the subsystem at its
    reactant and product frames, its charge and spin, and their bonds."""
    out = []
    for k, inst in enumerate(rxn.instances):
        atoms = inst["atoms"]
        fr, fp_ = ((inst["reactant_frame"], inst["product_frame"]) if inst["direction"] == "forward"
                   else (inst["product_frame"], inst["reactant_frame"]))
        syms, xr = _cut(symbols, frames, fr, atoms)
        _, xp = _cut(symbols, frames, fp_, atoms)
        q = int(inst["charge"])
        local = {a: m for m, a in enumerate(atoms)}
        bonds = {n: {(local[i], local[j]) for i, j in hist.bonds_at(f) if i in local and j in local}
                 for n, f in (("reactant", fr), ("product", fp_))}
        out.append({"k": k, "syms": syms, "xr": xr, "xp": xp, "q": q, "mult": _complex_multiplicity(syms, q),
                    "fr": fr, "fp": fp_, "bonds": bonds})
    return out


def refine_reaction_one(rxn, cuts: list, engine, d: Path, maxiter: int) -> None:
    """Optimize a reaction's subsystem ends from each instance; the first
    whose ends keep their bonds becomes `complex`. If none does, the first
    whose optimized ends still differ in their bonds becomes it anyway,
    `complex["relaxed"]` saying what they are (e.g. two H radicals of the
    product recombined into H2: the sampled reaction, relaxed, is a
    dehydrogenation) -- the ends of a reaction that did happen, to search
    a TS between. Only when the ends optimize into the same molecules
    (products falling back to reactants, a barrierless reactant) is there
    no reaction: then the reason why."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for c in cuts:
        k, syms, q, mult = c["k"], c["syms"], c["q"], c["mult"]
        inst = rxn.instances[k]
        _write_xyz(d / f"instance_{k}_reactant_frame.xyz", syms, c["xr"], f"frame {c['fr']}")
        _write_xyz(d / f"instance_{k}_product_frame.xyz", syms, c["xp"], f"frame {c['fp']}")
        ends = {}
        for name, xyz in (("reactant", c["xr"]), ("product", c["xp"])):
            expect = c["bonds"][name]
            try:
                node = _optimize_one(engine, _structure(syms, xyz, q, mult), maxiter)
            except Exception as exc:
                ends[name] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                continue
            oxyz = np.asarray(node.coords) / ANGSTROM_TO_BOHR
            fp = d / f"instance_{k}_{name}_complex.xyz"
            _write_xyz(fp, syms, oxyz, f"energy={float(node.energy):.10f} charge={q} mult={mult}")
            got = perceive_bonds(syms, oxyz)
            ends[name] = {"file": str(fp), "energy": float(node.energy), "bonds_kept": got == expect}
            if got != expect:
                ends[name]["verdict"] = _verdict(name, got, expect,
                                                 c["bonds"]["product" if name == "reactant" else "reactant"], len(syms))
        inst["complex"] = ends
        r, p = ends.get("reactant", {}), ends.get("product", {})
        if "energy" in r and "energy" in p:
            inst["complex_delta_e_kcal"] = (p["energy"] - r["energy"]) * HARTREE_TO_KCAL_PER_MOL
        if r.get("bonds_kept") and p.get("bonds_kept") and not rxn.complex:
            rxn.complex = {"instance": k, "charge": q, "multiplicity": mult, "reactant": r["file"],
                           "product": p["file"], "reactant_energy": r["energy"], "product_energy": p["energy"],
                           "delta_e_kcal": inst["complex_delta_e_kcal"]}
    if not rxn.complex:
        rxn.complex = relaxed_complex(rxn.instances) or {}
    if not rxn.complex:
        verdicts = [v for inst in rxn.instances for v in
                    ((inst.get("complex") or {}).get(side, {}).get("verdict") for side in ("reactant", "product")) if v]
        v = verdicts[0] if verdicts else {"code": "failed", "text": "the optimizations of its ends failed",
                                           "ts": False}
        rxn.complex = {"error": v["text"], "reason": v["code"], "ts_makes_sense": v["ts"],
                       "tried": len(rxn.instances)}


def relaxed_complex(instances: list) -> Optional[dict]:
    """A reaction's ends when none kept its bonds on optimization: the first
    instance whose two optimized ends (files and energies in its
    `complex`) still differ in their bonds, as `complex` -- with
    `relaxed` = {reactants, products: SMILES of what they are, label, why}.
    None if every instance's ends optimized into the same molecules (or
    failed). Works on a finished run's saved instances too."""
    import re

    for k, inst in enumerate(instances):
        ends = inst.get("complex") or {}
        r, p = ends.get("reactant") or {}, ends.get("product") or {}
        if not (r.get("file") and p.get("file") and "energy" in r and "energy" in p):
            continue
        try:
            syms, rx, rc = read_xyz_frames(Path(r["file"]))
            _, px, _ = read_xyz_frames(Path(p["file"]))
        except (OSError, ValueError, IndexError):
            continue
        m = re.search(r"charge=(-?\d+) mult=(\d+)", rc[0] if rc else "")
        q, mult = (int(m.group(1)), int(m.group(2))) if m else (0, 1)
        r_got, p_got = perceive_bonds(syms, rx[0]), perceive_bonds(syms, px[0])
        if r_got == p_got:
            continue
        labeler = Labeler(syms, max_charge=max(1, abs(q)))
        sides = [[lab.smiles for lab in labeler.assign(components(len(syms), got), got, q)] for got in (r_got, p_got)]
        if Counter(sides[0]) == Counter(sides[1]):
            continue                      # the same molecules, renumbered: no reaction

        def side_text(smiles):
            return " + ".join(f"{n} {x}" if n > 1 else x for x, n in Counter(smiles).items())
        why = [v["text"] for v in (ends.get(s2, {}).get("verdict") for s2 in ("reactant", "product")) if v]
        return {"instance": k, "charge": q, "multiplicity": mult, "reactant": r["file"], "product": p["file"],
                "reactant_energy": r["energy"], "product_energy": p["energy"],
                "delta_e_kcal": (p["energy"] - r["energy"]) * HARTREE_TO_KCAL_PER_MOL,
                "relaxed": {"reactants": sides[0], "products": sides[1],
                            "label": f"{side_text(sides[0])} -> {side_text(sides[1])}", "why": why}}
    return None


def _set_delta_e(rxn, by_id: dict) -> None:
    es = [by_id[i].energy for i in rxn.reactants], [by_id[i].energy for i in rxn.products]
    if all(e is not None for e in es[0] + es[1]):
        rxn.delta_e_kcal = (sum(es[1]) - sum(es[0])) * HARTREE_TO_KCAL_PER_MOL


def refine(symbols, frames: np.ndarray, hist: BondHistory, species: list[SpeciesRecord],
           reactions: list[ReactionRecord], engine, output: Path, *, maxiter: int = 300,
           on_event: OnEvent = None, cache: Optional["LiveRefiner"] = None) -> None:
    """Optimize every species on its own (lowest instance kept) and every
    reaction's subsystem endpoints; fills energies, delta_e_kcal and
    `complex`. A structure whose bonds change while it is optimized is
    reported, not used (the molecule fell apart or reacted: not a minimum
    of that species at this level). What `cache` (a LiveRefiner) already
    refined while the MD ran is reused, by identity."""
    from mepd.chain import Chain  # noqa: F401  (engines import chains lazily)

    sp_dir, rx_dir = output / "species", output / "reactions"
    sp_dir.mkdir(parents=True, exist_ok=True)
    rx_dir.mkdir(parents=True, exist_ok=True)
    done_sp, done_rx = cache.results() if cache is not None else ({}, {})
    for n, rec in enumerate(species):
        prev = done_sp.get(species_key(rec))
        if prev is not None:
            rec.energy, rec.file, rec.note = prev.energy, prev.file, prev.note
            continue
        _emit(on_event, "refine_species", index=n, total=len(species), smiles=rec.smiles)
        refine_species_one(rec, _species_cuts(symbols, frames, hist, rec), engine, sp_dir, maxiter)
    by_id = {r.id: r for r in species}
    for n, rxn in enumerate(reactions):
        _set_delta_e(rxn, by_id)
        prev = done_rx.get(reaction_key(rxn, by_id))
        if prev is not None and prev.complex:
            rxn.complex = dict(prev.complex)
            if prev.complex.get("reactant") and \
                    _side_key(prev.reactant_keys) != _side_key([species_key(by_id[i]) for i in rxn.reactants]):
                # refined the other way round: swap the ends
                c = rxn.complex
                c["reactant"], c["product"] = c["product"], c["reactant"]
                c["reactant_energy"], c["product_energy"] = c["product_energy"], c["reactant_energy"]
                c["delta_e_kcal"] = -c["delta_e_kcal"]
            continue
        _emit(on_event, "refine_reaction", index=n, total=len(reactions), label=rxn.label)
        refine_reaction_one(rxn, _reaction_cuts(symbols, frames, hist, rxn), engine, rx_dir / f"reaction_{rxn.id}",
                            maxiter)
        if (rxn.complex or {}).get("relaxed"):      # relabelled: its energy change is its relaxed ends'
            rxn.delta_e_kcal = rxn.complex.get("delta_e_kcal")


class PreviousRefinement:
    """What an earlier analysis of this run refined (its network.json), as a
    cache for refine() when the run is extended: species and reactions are
    matched by identity, so only new ones are optimized. Their files are
    first copied to names of their own (output/previous/): the new analysis
    numbers species and reactions afresh and writes by number, which would
    overwrite them. A reaction's TS result is kept too (`ts(rxn, by_id)`)."""

    def __init__(self, output: Path):
        import hashlib

        self.species, self.reactions, self._ts = {}, {}, {}
        fp = Path(output) / "network.json"
        try:
            data = json.loads(fp.read_text())
        except (OSError, ValueError):
            return
        keep = Path(output) / "previous"

        def stash(path: Optional[str], tag: str) -> Optional[str]:
            if not path or not Path(path).exists():
                return path
            keep.mkdir(parents=True, exist_ok=True)
            dst = keep / f"{hashlib.sha1(tag.encode()).hexdigest()[:16]}_{Path(path).name}"
            if not dst.exists():
                shutil.copy(path, dst)
            return str(dst)

        sp = {d["id"]: d for d in data.get("species") or []}
        keys = {i: f"{d['smiles']}|{d['charge']}|{d['multiplicity']}" for i, d in sp.items()}
        for i, d in sp.items():
            if d.get("energy") is None and not d.get("note"):
                continue                           # never refined
            self.species[keys[i]] = SimpleNamespace(energy=d.get("energy"), file=stash(d.get("file"), keys[i]),
                                                     note=d.get("note") or "")
        for r in data.get("reactions") or []:
            if not all(i in keys for i in r["reactants"] + r["products"]):
                continue
            a = "+".join(sorted(keys[i] for i in r["reactants"]))
            b = "+".join(sorted(keys[i] for i in r["products"]))
            key = min(f"{a}>{b}", f"{b}>{a}")
            c = dict(r.get("complex") or {})
            for side in ("reactant", "product"):
                c[side] = stash(c.get(side), f"{key}|{side}")
            if c:
                self.reactions[key] = SimpleNamespace(complex=c, reactant_keys=[keys[i] for i in r["reactants"]])
            if (r.get("ts") or {}).get("barrier_kcal") is not None:
                self._ts[key] = (r["ts"], tuple(sorted(keys[i] for i in r["reactants"])))

    def results(self) -> tuple[dict, dict]:
        return dict(self.species), dict(self.reactions)

    def ts(self, rxn, by_id: dict) -> Optional[dict]:
        """The TS found for this reaction before, if written the same way
        round (its barrier is from that side); else None (search again)."""
        hit = self._ts.get(reaction_key(rxn, by_id))
        if hit is None or hit[1] != tuple(sorted(species_key(by_id[i]) for i in rxn.reactants)):
            return None
        return dict(hit[0])


class _Caches:
    """Several refine() caches as one (the earlier analysis, the live refiner)."""

    def __init__(self, *caches):
        self.caches = [c for c in caches if c is not None]

    def results(self) -> tuple[dict, dict]:
        sp, rx = {}, {}
        for c in self.caches:
            a, b = c.results()
            sp.update(a)
            rx.update(b)
        return sp, rx


class LiveRefiner:
    """Refines reactions while the MD still runs: each reaction once its
    event has settled, with its species, in a pool of `workers` threads (an
    engine that is not known to be thread-safe is used one call at a time
    by them; the MD itself is a separate process or holds its own engine).
    `on_change()` is called after each piece, to write the live network."""

    THREAD_SAFE = ("GXTBCalculator", "QCComputeEngine")

    def __init__(self, engine, output: Path, *, maxiter: int = 300, workers: int = 2,
                 on_change: Optional[Callable[[], None]] = None):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self.engine, self.output, self.maxiter = engine, Path(output), maxiter
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="refine")
        self.lock = threading.Lock()
        self.engine_lock = None if type(engine).__name__ in self.THREAD_SAFE else threading.Lock()
        self.on_change = on_change or (lambda: None)
        self.species: dict = {}        # key -> refined SpeciesRecord
        self.reactions: dict = {}      # key -> refined ReactionRecord (with reactant_keys, product_keys)
        self.submitted: set = set()
        self.futures: list = []
        self.n = 0

    def _engine_call(self, fn, *args):
        if self.engine_lock is None:
            return fn(*args)
        with self.engine_lock:
            return fn(*args)

    def submit(self, symbols, frames, hist: BondHistory, species: list, reactions: list, settled: set) -> int:
        """Queue every settled reaction (and its species) not yet queued."""
        import copy

        by_id = {r.id: r for r in species}
        queued = 0
        wanted_sp = [r for r in species if r.initial]
        for rxn in reactions:
            if rxn.id not in settled:
                continue
            rk = reaction_key(rxn, by_id)
            if rk in self.submitted:
                continue
            self.submitted.add(rk)
            wanted_sp += [by_id[i] for i in rxn.reactants + rxn.products]
            r2 = copy.deepcopy(rxn)
            r2.reactant_keys = [species_key(by_id[i]) for i in rxn.reactants]
            r2.product_keys = [species_key(by_id[i]) for i in rxn.products]
            cuts = _reaction_cuts(symbols, frames, hist, rxn)
            self.n += 1
            d = self.output / "reactions" / f"live_{self.n}"
            self.futures.append(self.pool.submit(self._run_reaction, rk, r2, cuts, d))
            queued += 1
        for rec in wanted_sp:
            sk = species_key(rec)
            if sk in self.submitted:
                continue
            self.submitted.add(sk)
            s2 = copy.deepcopy(rec)
            cuts = _species_cuts(symbols, frames, hist, rec)
            self.n += 1
            self.futures.append(self.pool.submit(self._run_species, sk, s2, cuts, f"live_{self.n}"))
        return queued

    def _run_species(self, key, rec, cuts, name):
        (self.output / "species").mkdir(parents=True, exist_ok=True)
        self._engine_call(refine_species_one, rec, cuts, self.engine, self.output / "species", self.maxiter, name)
        with self.lock:
            self.species[key] = rec
        self.on_change()

    def _run_reaction(self, key, rxn, cuts, d):
        self._engine_call(refine_reaction_one, rxn, cuts, self.engine, d, self.maxiter)
        with self.lock:
            self.reactions[key] = rxn
        self.on_change()

    def results(self) -> tuple[dict, dict]:
        with self.lock:
            return dict(self.species), dict(self.reactions)

    def network(self) -> dict:
        """The refined species and reactions so far, as network.json writes them
        (ids local to this file; adopted by key)."""
        sp, rx = self.results()
        keys = list(sp)
        ids = {k: n for n, k in enumerate(keys)}
        species = []
        for k in keys:
            d = asdict(sp[k])
            d.update(id=ids[k], key=k)
            species.append(d)
        reactions = []
        for n, (k, r) in enumerate(rx.items()):
            if not all(x in ids for x in r.reactant_keys + r.product_keys):
                continue   # a species still being optimized
            d = asdict(r)
            d.update(id=n, key=k, reactants=[ids[x] for x in r.reactant_keys],
                     products=[ids[x] for x in r.product_keys],
                     shuttles=sorted((Counter(ids[x] for x in r.reactant_keys)
                                      & Counter(ids[x] for x in r.product_keys)).elements()))
            es = [sp[x].energy for x in r.reactant_keys], [sp[x].energy for x in r.product_keys]
            if all(e is not None for e in es[0] + es[1]):
                d["delta_e_kcal"] = (sum(es[1]) - sum(es[0])) * HARTREE_TO_KCAL_PER_MOL
            for x in ("reactant_keys", "product_keys"):
                d.pop(x, None)
            reactions.append(d)
        return {"kind": "nanoreactor", "live": True, "species": species, "reactions": reactions,
                "pending": max(0, len(self.submitted) - len(sp) - len(rx))}

    def close(self, wait: bool = True) -> None:
        self.pool.shutdown(wait=wait)


# Why a reaction has no TS endpoints: what one of its optimized ends turned into.
VERDICTS = {
    "barrierless": ("barrierless: the reactants turn into the products as soon as they are optimized", False),
    "reverts": ("the products fall back to the reactants when optimized: not a stable product at this level", False),
    "recombines": ("barrierless: the fragments of the {side}s bond together when optimized (typically radicals)", False),
    "falls_apart": ("the {side}s fall apart when optimized: not a stable complex at this level", False),
    "rearranges": ("the {side}s rearrange into something else when optimized", True),
    "failed": ("the optimizations of its ends failed", True),
}


def _verdict(side: str, got: set, expect: set, other: set, n_atoms: int) -> dict:
    """What an optimized end became instead of what it was."""
    if got == other:
        code = "barrierless" if side == "reactant" else "reverts"
    else:
        n_got, n_exp = len(components(n_atoms, got)), len(components(n_atoms, expect))
        code = "recombines" if n_got < n_exp else "falls_apart" if n_got > n_exp else "rearranges"
    text, ts = VERDICTS[code]
    return {"code": code, "side": side, "text": text.format(side=side), "ts": ts}


def _emit(on_event: OnEvent, event: str, **payload) -> None:
    if on_event is not None:
        on_event(event, payload)


# ---------------------------------------------------------------------------
# One call
# ---------------------------------------------------------------------------

@dataclass
class NanoreactorResult:
    symbols: list
    species: list
    reactions: list
    events: list
    settings: dict
    trajectory: str
    dt_fs: float
    n_frames: int


def analyze_trajectory(traj: Path, *, total_charge: int, detect: DetectSettings, dt_fs: float,
                       max_instances: int = 3, charges: Optional[dict] = None):
    """Events, species and reactions of a reactor trajectory. `charges`
    ({"method", "electronic_temperature", "multiplicity", "workdir"}) turns
    on partial charges (one xtb single point per frame used) for the charge
    of each molecule; without it molecules are neutral where they can be."""
    symbols, frames, _ = read_xyz_frames(traj)
    hist = bond_history(symbols, frames, detect, dt_fs)
    events = detect_events(len(symbols), hist, detect, dt_fs)
    labeler = Labeler(symbols, max_charge=max(1, abs(int(total_charge))))
    cache: dict[int, Optional[np.ndarray]] = {}

    def charges_at(frame: int):
        if charges is None:
            return None
        if frame not in cache:
            cache[frame] = frame_partial_charges(
                symbols, frames[frame], charge=total_charge, multiplicity=int(charges.get("multiplicity", 1)),
                method=charges.get("method", "gfn2"),
                electronic_temperature=float(charges.get("electronic_temperature", 3000.0)),
                workdir=Path(charges["workdir"]), executable=charges.get("executable"))
        return cache[frame]

    species, reactions, ev_records = build_network(symbols, frames, hist, events, labeler, total_charge, dt_fs,
                                                   max_instances=max_instances, charges_at=charges_at)
    return symbols, frames, hist, species, reactions, ev_records


def read_segments(md_dir: Path) -> tuple[list, np.ndarray]:
    """The trajectory so far: every finished MD segment, in order."""
    frames, symbols = [], []
    for fp in sorted(Path(md_dir).glob("segment_*.xyz")):
        syms, xyz, _ = read_xyz_frames(fp)
        symbols = syms or symbols
        if len(xyz):
            frames.append(xyz)
    return symbols, (np.concatenate(frames) if frames else np.zeros((0, len(symbols), 3)))


def live_events(md_dir: Path, *, total_charge: int, detect: DetectSettings, dt_fs: float,
                charges_at: Optional[Callable[[int, np.ndarray], Optional[np.ndarray]]] = None,
                context: Optional[dict] = None) -> dict:
    """Events in the trajectory so far. Events that may still be unfolding
    at the end of the trajectory are listed as tentative ("analyzing")
    until they are over. `charges_at(frame, coords)` gives partial charges
    for the molecules' charges (else neutral where they can be). If
    `context` is a dict, it is filled with what a LiveRefiner needs
    (symbols, frames, hist, species, reactions, settled reaction ids)."""
    symbols, frames = read_segments(md_dir)
    return events_in_frames(symbols, frames, total_charge=total_charge, detect=detect, dt_fs=dt_fs,
                            charges_at=charges_at, context=context)


def events_in_frames(symbols, frames: np.ndarray, *, total_charge: int, detect: DetectSettings, dt_fs: float,
                     charges_at: Optional[Callable[[int, np.ndarray], Optional[np.ndarray]]] = None,
                     context: Optional[dict] = None) -> dict:
    """live_events on frames already in memory (Angstrom, `dt_fs` apart):
    the interactive reactor's events come from here too."""
    out = {"n_frames": int(len(frames)), "time_ps": len(frames) * dt_fs / 1000.0, "events": [], "final": False}
    if len(frames) < 2:
        return out
    hist = bond_history(symbols, frames, detect, dt_fs)
    events = detect_events(len(symbols), hist, detect, dt_fs)
    settle = int(round((detect.merge_window_fs + detect.lag_fs + detect.min_lifetime_fs) / dt_fs))
    edge = len(frames) - 1 - settle
    hint = (lambda f: charges_at(f, frames[f])) if charges_at is not None else None
    species, reactions, records = build_network(
        symbols, frames, hist, events, Labeler(symbols, max_charge=max(1, abs(int(total_charge)))), total_charge,
        dt_fs, max_instances=1, charges_at=hint)
    settled = set()
    for ev, rec in zip(events, records):
        if ev.end >= edge:
            rec["tentative"] = True
        elif rec.get("reaction") is not None:
            settled.add(rec["reaction"])
    # A reaction counts as settled only if none of its sightings is still unfolding.
    settled -= {rec["reaction"] for rec in records if rec.get("tentative") and rec.get("reaction") is not None}
    out["events"] = records
    if context is not None:
        context.update(symbols=symbols, frames=frames, hist=hist, species=species, reactions=reactions,
                       settled=settled)
    return out


def write_md_species(symbols, frames: np.ndarray, species: list[SpeciesRecord], output: Path) -> None:
    """Each species' first geometry as cut from the trajectory (not a minimum)."""
    d = Path(output) / "species"
    d.mkdir(parents=True, exist_ok=True)
    for rec in species:
        if rec.instances:
            frame, atoms = rec.instances[0]
            syms, xyz = _cut(symbols, frames, frame, atoms)
            fp = d / f"species_{rec.id}_md.xyz"
            _write_xyz(fp, syms, xyz, f"{rec.smiles} charge={rec.charge} mult={rec.multiplicity} frame={frame}")
            rec.md_file = str(fp)


def write_network(result: NanoreactorResult, output: Path) -> Path:
    data = {
        "kind": "nanoreactor",
        "settings": result.settings,
        "trajectory": result.trajectory,
        "frame_fs": result.dt_fs,
        "n_frames": result.n_frames,
        "species": [asdict(s) for s in result.species],
        "reactions": [asdict(r) for r in result.reactions],
        "events": result.events,
        "methods": REFERENCES,
        "note": "Energies of different reactions have different atoms and are not comparable; every energy "
                "here is a difference within one reaction (Hartree for species, kcal/mol for differences).",
    }
    fp = Path(output) / "network.json"
    fp.write_text(json.dumps(data, indent=2, default=str))
    return fp
