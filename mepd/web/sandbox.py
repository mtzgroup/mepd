"""Interactive reactor (Explore › select species › Interactive reactor): a
live MD the user steers -- drag an atom and it is pulled toward the
pointer, scroll to grow or shrink the wall, set the temperature.

The MD runs in its own process (one per sandbox): Langevin dynamics
(BAOAB) on GFN2-xTB from tblite (in-process, the wavefunction carried from
step to step: ~15 ms a step for 40 atoms), inside the nanoreactor's soft
logfermi wall. Pulls are harmonic springs from an atom to a target point,
their force capped so a fast drag cannot blow the system up. The page
sends commands; the process streams frames back (~25 per second). A
sandbox nobody has looked at for a while stops by itself.

Each holds a CPU while it runs. In the public demo `limits` keeps that
bounded: a few at once server-wide (a new one is refused, never someone
else's stopped), one per visitor, and each for a limited time.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import queue
import threading
import time
from typing import Optional

import numpy as np

BOHR = 0.529177210903            # Angstrom
KCAL = 627.509474                # kcal/mol per Hartree
IDLE_STOP_S = 900.0              # nobody has looked at it for this long: stop (15 min)
TRAJ_FS = 2.0                    # the event trajectory: one frame this often (as the nanoreactor's dump)
EVENT_EVERY_S = 1.5              # look for reaction events this often
EVENT_WINDOW = 15000             # ...in at most this many frames (30 ps); earlier settled events are kept
FRAME_EVERY_S = 0.04             # stream a frame at most this often
MAX_PULL = 0.08                  # Hartree/bohr cap on a pull's force (~100 kcal/mol/A)

_Z = {"H": 1, "He": 2, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12, "Si": 14, "P": 15,
      "S": 16, "Cl": 17, "K": 19, "Br": 35, "I": 53}


def _run(symbols, coords_A, charge, multiplicity, radius_A, temperature, cmd_q, out_q, traj_q):
    """The MD process: steps until told to stop, applying commands between steps."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from tblite.interface import Calculator

    from mepd.discovery.nanoreactor import _masses, _wall

    n = len(symbols)
    Z = np.array([_Z[s] for s in symbols])
    pos = np.asarray(coords_A, dtype=float) / BOHR
    masses = _masses(symbols)
    calc = Calculator("GFN2-xTB", Z, pos, charge=float(charge), uhf=int(multiplicity) - 1)
    calc.set("verbosity", 0)
    calc.set("temperature", 3000.0 * 3.166811563e-6)   # Fermi smearing: bonds break without SCF failures
    state = {"radius": float(radius_A), "T": float(temperature), "pulls": {}, "paused": False, "dt": 0.5,
             "friction": 0.02, "wall": 20.0 / KCAL * BOHR}   # 20 kcal/mol/A, in Eh/bohr
    res = [None]

    def forces(x):
        calc.update(x)
        try:
            r = calc.singlepoint(res[0]) if res[0] is not None else calc.singlepoint()
        except Exception:
            r = calc.singlepoint()          # without the previous wavefunction
        res[0] = r
        e, g = float(r.get("energy")), np.array(r.get("gradient"))
        ew, gw = _wall(x, state["radius"] / BOHR, state["wall"])
        e, g = e + ew, g + gw
        for atom, (target, k) in state["pulls"].items():
            d = x[atom] - target
            f = k * d
            norm = np.linalg.norm(f)
            if norm > MAX_PULL:
                f *= MAX_PULL / norm
            g[atom] += f
        return e, g

    def send(kind, **payload):
        try:
            out_q.put_nowait({"kind": kind, **payload})
        except queue.Full:
            pass

    # Take the strain out of the packing before any dynamics (else it turns into heat).
    try:
        for _ in range(60):
            _, g = forces(pos)
            step = -g / masses[:, None] * 20.0
            norm = np.linalg.norm(step, axis=1, keepdims=True)
            pos = pos + np.where(norm > 0.1, step * 0.1 / np.maximum(norm, 1e-12), step)
    except Exception as exc:
        send("error", message=f"GFN2-xTB failed on the packed reactor: {type(exc).__name__}: {exc}")
        return
    kB = 3.166811563e-6 / 1.066551       # amu bohr^2/fs^2 per K
    acc = 0.93766                         # (Eh/bohr)/amu -> bohr/fs^2
    rng = np.random.default_rng(0)
    vel = rng.normal(size=pos.shape) * np.sqrt(kB * state["T"] / masses)[:, None]
    energy, grad = forces(pos)
    e0 = energy
    t_fs, steps, last_frame, failures = 0.0, 0, 0.0, 0
    traj, next_traj = [], 0.0          # evenly spaced frames for event detection (sent losslessly)
    started = time.time()
    while True:
        while True:                      # commands, between steps
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                break
            op = cmd.get("op")
            if op == "stop":
                return
            if op == "pull":
                k = float(cmd.get("k", 20.0)) / KCAL * BOHR * BOHR      # kcal/mol/A^2 -> Eh/bohr^2
                state["pulls"][int(cmd["atom"])] = (np.asarray(cmd["target"], dtype=float) / BOHR, k)
            elif op == "release":
                state["pulls"].pop(int(cmd["atom"]), None)
            elif op == "release_all":
                state["pulls"].clear()
            elif op == "radius":
                state["radius"] = float(min(max(cmd["value"], 3.0), 60.0))
            elif op == "temperature":
                state["T"] = float(min(max(cmd["value"], 0.0), 5000.0))
            elif op == "pause":
                state["paused"] = bool(cmd.get("value", True))
        if state["paused"]:
            time.sleep(0.03)
            if time.time() - last_frame > 0.2:
                last_frame = time.time()
                send("frame", pos=np.round(pos * BOHR, 3).tolist(), t_fs=t_fs, radius=state["radius"],
                     T=state["T"], e_rel=(energy - e0) * KCAL, paused=True, rate=0.0,
                     pulls={a: (t * BOHR).tolist() for a, (t, _) in state["pulls"].items()})
            continue
        dt, gamma = state["dt"], state["friction"]
        c1 = math.exp(-gamma * dt)
        c2 = np.sqrt((1 - c1 * c1) * kB * state["T"] / masses)[:, None]
        old = (pos.copy(), vel.copy(), grad.copy())
        try:
            vel -= 0.5 * dt * acc * grad / masses[:, None]
            pos = pos + 0.5 * dt * vel
            vel = c1 * vel + c2 * rng.normal(size=vel.shape)
            pos = pos + 0.5 * dt * vel
            energy, grad = forces(pos)
            vel -= 0.5 * dt * acc * grad / masses[:, None]
            if not np.all(np.isfinite(pos)):
                raise FloatingPointError("non-finite positions")
            failures = 0
        except Exception as exc:          # an SCF that failed: step back, cool down, go on
            pos, vel, grad = old
            vel *= 0.5
            res[0] = None
            failures += 1
            if failures > 20:
                send("error", message=f"the dynamics keeps failing: {type(exc).__name__}: {exc}")
                return
            continue
        t_fs += dt
        steps += 1
        if t_fs >= next_traj - 1e-9:
            traj.append(np.round(pos * BOHR, 3).tolist())
            next_traj += TRAJ_FS
        now = time.time()
        if now - last_frame >= FRAME_EVERY_S:
            last_frame = now
            if traj:
                traj_q.put(traj)
                traj = []
            ekin = 0.5 * float((masses[:, None] * vel * vel).sum()) * 1.066551
            send("frame", pos=np.round(pos * BOHR, 3).tolist(), t_fs=round(t_fs, 1), radius=state["radius"],
                 T=state["T"], T_inst=round(2 * ekin / (3 * n * 3.166811563e-6), 0),
                 e_rel=round((energy - e0) * KCAL, 1), paused=False, rate=round(t_fs / max(now - started, 1e-6), 1),
                 pulls={a: (t * BOHR).tolist() for a, (t, _) in state["pulls"].items()})


class Sandbox:
    """One interactive reactor: its MD process, and the latest frame it sent."""

    def __init__(self, sid: str, symbols, coords_A, charge: int, multiplicity: int, radius: float,
                 temperature: float, names: str = "", owner: str = "", sources: Optional[list] = None,
                 idle_s: float = IDLE_STOP_S, max_s: Optional[float] = None):
        ctx = mp.get_context("spawn")
        self.id, self.symbols, self.charge, self.multiplicity, self.names = sid, list(symbols), int(charge), \
            int(multiplicity), names
        self.cmd_q, self.out_q, self.traj_q = ctx.Queue(), ctx.Queue(maxsize=8), ctx.Queue()
        self.proc = ctx.Process(target=_run, args=(self.symbols, np.asarray(coords_A).tolist(), charge, multiplicity,
                                                   radius, temperature, self.cmd_q, self.out_q, self.traj_q),
                                daemon=True)
        self.proc.start()
        self.latest: Optional[dict] = None
        self.seq = 0
        self.error: Optional[str] = None
        self.seen = time.time()
        self.cond = threading.Condition()
        self.radius = radius
        self.owner = owner                 # the workspace it was started from (listed there)
        self.sources = list(sources or [])  # the structures it was started from
        self.started = time.time()
        self.idle_s, self.max_s = float(idle_s), max_s   # stop when unwatched this long / after this long
        self.temperature = float(temperature)
        self.traj: list = []               # evenly spaced frames (TRAJ_FS apart), Angstrom
        self.events: list = []             # reaction events, as the nanoreactor's live view lists them
        self.events_rev = 0
        self._settled_before: list = []    # events earlier than the analysis window, kept as they were
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._analyze, daemon=True).start()

    def _read(self) -> None:
        while self.proc.is_alive() or not self.out_q.empty():
            while True:
                try:
                    self.traj.extend(self.traj_q.get_nowait())
                except queue.Empty:
                    break
            try:
                msg = self.out_q.get(timeout=0.5)
            except queue.Empty:
                now = time.time()
                if now - self.seen > self.idle_s or (self.max_s and now - self.started > self.max_s):
                    self.stop()
                continue
            with self.cond:
                if msg["kind"] == "error":
                    self.error = msg["message"]
                else:
                    self.seq += 1
                    self.latest = {**msg, "seq": self.seq}
                self.cond.notify_all()
        with self.cond:
            self.cond.notify_all()

    def _analyze(self) -> None:
        """Reaction events in the trajectory so far, by the nanoreactor's own
        detection (events_in_frames), every EVENT_EVERY_S."""
        from mepd.discovery.nanoreactor import DetectSettings, events_in_frames

        done = 0
        while self.proc.is_alive():
            time.sleep(EVENT_EVERY_S)
            n = len(self.traj)
            if n == done or n < 2:
                continue
            done = n
            start = max(0, n - EVENT_WINDOW)
            frames = np.asarray(self.traj[start:n], dtype=float)
            try:
                res = events_in_frames(self.symbols, frames, total_charge=self.charge, detect=DetectSettings(),
                                       dt_fs=TRAJ_FS)
            except Exception:
                continue
            shift = start * TRAJ_FS
            current = []
            for ev in res["events"]:
                ev = dict(ev)
                for key in ("start_fs", "end_fs"):
                    if ev.get(key) is not None:
                        ev[key] = round(ev[key] + shift, 1)
                for key in ("reactant_frame", "product_frame"):
                    if ev.get(key) is not None:
                        ev[key] += start
                current.append(ev)
            if start > 0:     # the window moved on: what it left behind stays as it was last seen
                kept = {}
                for e in self._settled_before + self.events:
                    if e.get("start_fs", 0) < shift and not e.get("tentative"):
                        kept.setdefault((e.get("start_fs"), e.get("label")), e)
                self._settled_before = list(kept.values())
                current = [e for e in current if (e.get("start_fs"), e.get("label")) not in kept]
            events = self._settled_before + current
            if events != self.events:
                self.events = events
                self.events_rev += 1

    def save(self, folder) -> tuple:
        """Stop the dynamics and write its trajectory (every frame,
        TRAJ_FS apart) and its first frame (with charge and spin) into
        `folder`, for a nanoreactor job to analyze. Returns the two paths."""
        from pathlib import Path

        self.stop()
        while True:                        # what was still on its way
            try:
                self.traj.extend(self.traj_q.get(timeout=0.2))
            except queue.Empty:
                break
        if len(self.traj) < 10:
            raise RuntimeError("there is no trajectory to analyze yet")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        def frame(xyz, comment):
            return f"{len(self.symbols)}\n{comment}\n" + "".join(
                f"{s} {x:.4f} {y:.4f} {z:.4f}\n" for s, (x, y, z) in zip(self.symbols, xyz))
        traj = folder / "trajectory.xyz"
        with traj.open("w") as fh:
            for k, xyz in enumerate(self.traj):
                fh.write(frame(xyz, f"t={k * TRAJ_FS:.1f} fs"))
        start = folder / "first_frame.xyz"
        start.write_text(frame(self.traj[0], f"qcdata_charge={self.charge} qcdata_multiplicity={self.multiplicity}"))
        return traj, start

    def events_view(self) -> dict:
        self.seen = time.time()
        return {"events": self.events, "rev": self.events_rev, "n_frames": len(self.traj), "frame_fs": TRAJ_FS}

    def frame(self, since: int = 0, wait: float = 0.5) -> dict:
        """The newest frame after `since` (waits up to `wait` s for one)."""
        self.seen = time.time()
        with self.cond:
            if self.seq <= since and self.error is None and self.proc.is_alive():
                self.cond.wait(timeout=wait)
            out = dict(self.latest or {"seq": 0})
        out.update(alive=self.proc.is_alive(), error=self.error, events_rev=self.events_rev)
        return out

    def command(self, cmd: dict) -> None:
        self.seen = time.time()
        if not self.proc.is_alive():
            raise RuntimeError(self.error or "this interactive reactor has stopped")
        self.cmd_q.put(cmd)

    def stop(self) -> None:
        try:
            self.cmd_q.put({"op": "stop"})
        except Exception:
            pass
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.kill()

    def describe(self) -> dict:
        return {"id": self.id, "symbols": self.symbols, "charge": self.charge, "multiplicity": self.multiplicity,
                "names": self.names, "alive": self.proc.is_alive(), "temperature": self.temperature,
                "started": self.started}


_SANDBOXES: dict[str, Sandbox] = {}
_LOCK = threading.Lock()
MAX_SANDBOXES = 2


def _make_room(owner: str, limits: Optional[dict]) -> dict:
    """Under _LOCK: stop what a new sandbox of `owner` replaces, and return
    the Sandbox timing arguments. Without `limits` the oldest sandboxes go
    beyond MAX_SANDBOXES; with them (the demo: {"total", "per_owner",
    "idle_s", "max_s"}) only the owner's own oldest go, and a full server
    refuses (RuntimeError)."""
    for sid, box in list(_SANDBOXES.items()):
        if not box.proc.is_alive():
            _SANDBOXES.pop(sid)
    if not limits:
        while len(_SANDBOXES) >= MAX_SANDBOXES:
            _SANDBOXES.pop(next(iter(_SANDBOXES))).stop()
        return {}
    mine = [sid for sid, box in _SANDBOXES.items() if box.owner == owner]
    while mine and len(mine) >= int(limits.get("per_owner", 1)):
        _SANDBOXES.pop(mine.pop(0)).stop()
    if len(_SANDBOXES) >= int(limits.get("total", MAX_SANDBOXES)):
        raise RuntimeError("every interactive reactor of the demo is in use: try again in a few minutes")
    return {"idle_s": float(limits.get("idle_s", IDLE_STOP_S)), "max_s": limits.get("max_s")}


def start(sid: str, structures: list, *, temperature: float = 800.0, radius: Optional[float] = None,
          names: str = "", owner: str = "", sources: Optional[list] = None, limits: Optional[dict] = None) -> Sandbox:
    """A new sandbox from one structure per molecule (repeated for copies),
    packed like the nanoreactor's. Room is made by _make_room."""
    from mepd.discovery.nanoreactor import pack_reactor

    symbols, coords, radius, _ = pack_reactor(structures, radius, seed=int(time.time()) % 1000)
    charge = sum(int(s.charge) for s in structures)
    electrons = sum(_Z.get(x, 0) for x in symbols) - charge
    mult = 1 if electrons % 2 == 0 else 2
    missing = sorted({x for x in symbols if x not in _Z})
    if missing:
        raise ValueError(f"the interactive reactor does not know {', '.join(missing)}")
    with _LOCK:
        timing = _make_room(owner, limits)
        box = Sandbox(sid, symbols, coords, charge, mult, float(radius) * 1.15, temperature, names, owner, sources,
                      **timing)
        _SANDBOXES[sid] = box
    return box


def start_from(sid: str, structure, *, temperature: float = 800.0, radius: Optional[float] = None,
               names: str = "", owner: str = "", sources: Optional[list] = None,
               limits: Optional[dict] = None) -> Sandbox:
    """A new sandbox starting from one geometry as it is (a complex's
    arrangement), centred; the wall a little beyond its farthest atom."""
    symbols = list(structure.symbols)
    missing = sorted({x for x in symbols if x not in _Z})
    if missing:
        raise ValueError(f"the interactive reactor does not know {', '.join(missing)}")
    xyz = np.asarray(structure.geometry, dtype=float).reshape(-1, 3) * BOHR
    xyz = xyz - xyz.mean(axis=0)
    radius = float(radius or (np.linalg.norm(xyz, axis=1).max() + 2.0))
    with _LOCK:
        timing = _make_room(owner, limits)
        box = Sandbox(sid, symbols, xyz, int(structure.charge), int(structure.multiplicity), radius, temperature, names,
                      owner, sources, **timing)
        _SANDBOXES[sid] = box
    return box


def running(owner: str) -> list[dict]:
    """The sandboxes started from this workspace that still run, newest first."""
    with _LOCK:
        boxes = [b for b in _SANDBOXES.values() if b.owner == owner and b.proc.is_alive()]
    return [b.describe() for b in sorted(boxes, key=lambda b: -b.started)]


def get(sid: str) -> Sandbox:
    box = _SANDBOXES.get(sid)
    if box is None:
        raise KeyError(sid)
    return box


def forget(sid: str) -> None:
    """Drop a sandbox from the registry (its process already stopped)."""
    with _LOCK:
        _SANDBOXES.pop(sid, None)


def stop(sid: str) -> None:
    with _LOCK:
        box = _SANDBOXES.pop(sid, None)
    if box is not None:
        box.stop()
