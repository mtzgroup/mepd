"""The nanoreactor inside a QM/MM environment: hot piston MD of the QM
region with QM/MM forces, the solvent (or protein) around it present all the
time.

* Forces: the run's QM/MM engine (mepd.engines.qmmm, wrapped in
  FrozenAtomsEngine): QM region at the profile's level, environment at the
  low level. Frozen atoms never move (zero gradient, no thermostat noise).
* Two temperatures (Langevin, BAOAB): the QM atoms are kept hot
  (`temperature`), the moving environment near `environment_temperature`
  with a stronger friction, so it takes up the heat the QM region gives off
  instead of boiling. Bonds breaking in the environment would be nonsense at
  the force-field level: such frames are never products.
* The piston: the nanoreactor's logfermi wall, switched between a wide and a
  narrow radius, acts on the QM atoms only, around where the QM region
  started (the environment is pushed only through its contacts).
* Events: the nanoreactor's bond-change detection on the QM atoms. Each
  product is the whole system right after an event (one per distinct QM bond
  graph), unless its frame changed a bond in the MM region or stretched a
  QM/MM cut bond.

Compare mepd.discovery.nanoreactor_generator, which runs the same MD on the
capped QM region in vacuum and puts the products into the environment
afterwards: much faster, but the environment plays no part in what reacts.

Based on the ab initio nanoreactor (L.-P. Wang et al., Nat. Chem. 6, 1044
(2014), doi:10.1038/nchem.2099); QM/MM MD and the integrator are mepd's.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR, HARTREE_TO_KCAL_PER_MOL


def run_qmmm_reactor_md(structure, engine, region, *, settings, workdir: Path,
                        environment_temperature: float = 300.0, friction_qm: float = 0.01,
                        friction_env: float = 0.05, on_event=None) -> Path:
    """Piston MD of a QM/MM system; returns trajectory.xyz (whole system,
    every `settings.dump_fs`). Resumes finished segments."""
    from mepd.discovery.nanoreactor import (_ACC, _KB, _KIN, _emit, _masses, _wall, piston_schedule,
                                            read_xyz_frames)
    from mepd.nodes.node import StructureNode

    settings.validate()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    symbols = [str(s) for s in structure.symbols]
    n = len(symbols)
    qm = np.asarray(region.qm_atoms, dtype=int)
    frozen = np.zeros(n, dtype=bool)
    frozen[np.asarray(region.frozen_atoms, dtype=int)] = True
    is_qm = np.zeros(n, dtype=bool)
    is_qm[qm] = True
    schedule = piston_schedule(settings)
    pos0 = np.asarray(structure.geometry, dtype=float)
    centre = pos0[qm].mean(axis=0)
    (workdir / "schedule.json").write_text(json.dumps({
        "dump_fs": settings.dump_fs, "time_ps": settings.time_ps, "segments": [[d, r] for d, r in schedule],
        "engine": "QM/MM", "qm_atoms": region.qm_atoms, "frozen_atoms": list(region.frozen_atoms),
        "centre_bohr": centre.tolist()}))
    masses = _masses(symbols)
    wall_au = settings.wall_force / HARTREE_TO_KCAL_PER_MOL / ANGSTROM_TO_BOHR
    dt = settings.step_fs
    every = max(1, int(round(settings.dump_fs / dt)))
    temps = np.where(is_qm, settings.temperature, environment_temperature)
    kt = _KB * temps / _KIN
    fric = np.where(is_qm, friction_qm, friction_env)
    c1 = np.exp(-fric * dt)[:, None]
    c2 = np.sqrt((1 - c1[:, 0] ** 2) * kt / masses)[:, None]
    c2[frozen] = 0.0
    rng = np.random.default_rng(settings.seed)
    calls = {"n": 0}

    def forces(pos, radius_bohr):
        node = StructureNode(structure=structure.model_copy(update={"geometry": pos}), has_molecular_graph=False)
        grad = np.array(engine.compute_gradients([node])[0], dtype=float).reshape(-1, 3)
        e_wall, g_wall = _wall(pos[qm] - centre, radius_bohr, wall_au)
        grad[qm] += g_wall
        calls["n"] += 1
        return float(node.energy) + e_wall, grad

    from mepd.discovery.nanoreactor import _write_xyz

    for name in ("packed.xyz", "reactor.xyz"):      # what the Reactor view reads before the first segment
        if not (workdir / name).exists():
            _write_xyz(workdir / name, symbols, pos0 / ANGSTROM_TO_BOHR, "QM/MM nanoreactor start")
    pos = pos0.copy()
    vel = rng.normal(size=pos.shape) * np.sqrt(kt / masses)[:, None]
    vel[frozen] = 0.0
    t_done = 0.0
    for k, (dur, r) in enumerate(schedule):
        seg, restart = workdir / f"segment_{k:03d}.xyz", workdir / f"segment_{k:03d}.restart.npz"
        if seg.exists() and restart.exists():
            state = np.load(restart)
            pos, vel = state["pos"], state["vel"]
            t_done += dur
            continue
        radius_bohr = r * ANGSTROM_TO_BOHR
        n_steps = max(1, int(round(dur * 1000.0 / dt)))
        running = workdir / "xtb.trj"      # the running segment, named like the xtb path's (live views)
        with running.open("w") as trj:
            energy, grad = forces(pos, radius_bohr)
            for step in range(1, n_steps + 1):
                vel -= 0.5 * dt * _ACC * grad / masses[:, None]
                pos = pos + 0.5 * dt * vel
                vel = c1 * vel + c2 * rng.normal(size=vel.shape)
                vel[frozen] = 0.0
                pos = pos + 0.5 * dt * vel
                energy, grad = forces(pos, radius_bohr)
                vel -= 0.5 * dt * _ACC * grad / masses[:, None]
                vel[frozen] = 0.0
                if step % every == 0:
                    xyz = pos / ANGSTROM_TO_BOHR
                    trj.write(f"{n}\n energy: {energy:.10f} \n"
                              + "".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(symbols, xyz)))
                    trj.flush()
                if not np.all(np.isfinite(pos)):
                    raise RuntimeError(f"the QM/MM MD blew up in segment {k} (step {step})")
        np.savez(restart, pos=pos, vel=vel)
        running.replace(seg)
        t_done += dur

        def temp(mask):
            return float(np.sum(masses[mask, None] * vel[mask] ** 2) * _KIN / (3 * max(1, mask.sum()) * _KB))

        _emit(on_event, "md_segment", segment=k + 1, segments=len(schedule), time_ps=t_done, radius=r,
              total_ps=settings.time_ps, temperature=temp(is_qm), environment_temperature=temp(~is_qm & ~frozen))
        try:
            from mepd.progress import update_status

            update_status(f"QM/MM nanoreactor: {t_done:.2f}/{settings.time_ps:g} ps, QM {temp(is_qm):.0f} K, "
                          f"environment {temp(~is_qm & ~frozen):.0f} K")
        except Exception:
            pass
    traj = workdir / "trajectory.xyz"
    with traj.open("w") as out:
        for k in range(len(schedule)):
            out.write((workdir / f"segment_{k:03d}.xyz").read_text())
    _ = read_xyz_frames   # same format as the vacuum reactor's
    return traj


def qmmm_reactor_products(structure, engine, region, *, max_products: int = 20, temperature: float = 2500.0,
                          time_ps: float = 5.0, compress: float = 0.7, period_ps: float = 1.0,
                          environment_temperature: float = 300.0, seed: int = 0, workdir: str = "",
                          on_event=None) -> tuple[list, dict]:
    """Whole-system products (the system's atom order) of QM/MM piston MD,
    and counts {"events", "products", "mm_changed", "boundary"}."""
    import tempfile

    from mepd.discovery.nanoreactor import (DetectSettings, ReactorSettings, auto_radius, bond_history,
                                            detect_events, read_xyz_frames)
    from mepd.qmmm import diagnose

    pos = np.asarray(structure.geometry, dtype=float) / ANGSTROM_TO_BOHR
    qm = list(region.qm_atoms)
    extent = float(np.max(np.linalg.norm(pos[qm] - pos[qm].mean(axis=0), axis=1)))
    settings = ReactorSettings(temperature=float(temperature), time_ps=float(time_ps), compress=float(compress),
                               period_ps=float(period_ps), method="level", seed=int(seed),
                               radius=max(auto_radius(len(qm)), extent + 1.5))
    detect = DetectSettings()
    from mepd.discovery.nanoreactor_generator import run_dir, segment_hook, write_live_events

    run = run_dir(workdir)
    with tempfile.TemporaryDirectory(prefix="mepd_qmmm_reactor_") as tmp:
        d = run / "md" if run is not None else Path(tmp)
        traj = run_qmmm_reactor_md(structure, engine, region, settings=settings, workdir=d,
                                   environment_temperature=environment_temperature,
                                   on_event=segment_hook(run, int(region.qm_charge), settings.dump_fs, on_event))
        symbols, frames, _ = read_xyz_frames(traj)
    write_live_events(run, total_charge=int(region.qm_charge), dt_fs=settings.dump_fs)
    sub = [symbols[i] for i in qm]
    hist = bond_history(sub, frames[:, qm], detect, settings.dump_fs)
    events = detect_events(len(qm), hist, detect, settings.dump_fs)
    seen = {frozenset(hist.initial)}
    bonded0 = {a for p in hist.initial for a in p}
    counts = {"events": len(events), "products": 0, "mm_changed": 0, "boundary": 0}
    out = []
    for ev in sorted(events, key=lambda e: (e.product_frame, e.start)):
        bonds = frozenset(hist.bonds_at(ev.product_frame))
        if bonds in seen:
            continue
        seen.add(bonds)
        if bonded0 - {a for p in bonds for a in p}:
            counts["torn"] = counts.get("torn", 0) + 1
            continue           # torn into atoms
        s = structure.model_copy(update={"geometry": frames[ev.product_frame] * ANGSTROM_TO_BOHR})
        rep = diagnose(region, [structure, s])
        if rep["frames"][1]["mm_bond_changes"]:
            counts["mm_changed"] += 1
            continue
        if any(x > 1.3 for x in rep["frames"][1]["boundary_stretch"]):
            counts["boundary"] += 1
            continue
        out.append(s)
        if len(out) >= int(max_products):
            break
    counts["products"] = len(out)
    from mepd.discovery.nanoreactor_generator import report

    report(len(events), len(seen) - 1, counts.get("torn", 0), len(out), settings.temperature)
    counts.setdefault("torn", 0)
    return out, counts
