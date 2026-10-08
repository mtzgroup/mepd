"""The QM/MM free-energy surface of the QM region: its environment averaged
out, so that any path method (NEB, FSM, GSM, MLP-GI) relaxes a reaction
path in the potential of mean force instead of on one frozen snapshot of
the solvent.

For a QM/MM system x = (s, y) -- s the solute (the QM atoms and the MM
atoms their cut bonds reach), y the environment --

    A(s) = -kT ln  integral exp(-E(s, y) / kT) dy
    dA/ds = < dE/ds >_y            (the mean force, s held fixed)

(the QM/MM minimum free-energy path: H. Hu, Z. Lu, W. Yang, J. Chem. Theory
Comput. 3, 390 (2007), doi:10.1021/ct600240y). The average is over an MD of
the environment at `temperature` with the solute held (and the region's
frozen outer shell: the droplet's or protein's boundary): xtb's dynamics at
the region's low level (GFN-FF, or GFN1/GFN2), the held atoms kept rigid by
stiff restraints on all their distances (xtb ignores exact fixing in MD),
each frame then turned back onto them and them put back exactly. In
the subtractive (mechanical) embedding the QM energy of a fixed solute does
not depend on the environment, so these dynamics sample the QM/MM ensemble
exactly and each frame costs only low-level gradients: one QM gradient per
geometry in all.

AMBER and TIP3P environments (OpenMM) are sampled by OpenMM Langevin
dynamics with the solute's atoms fixed exactly (mass 0). With electrostatic
embedding the QM energy depends on the environment's charges: the dynamics
see the QM region as fixed point charges (the QM program's atomic charges at
that geometry in its starting environment, link-atom charges on the atom
they cap; Hu, Lu and Yang use ESP charges), and the mean force is the
average of full QM/MM gradients, one QM calculation in each frame's field.

The environment's coordinates on the nodes are only a starting point: the
path minimizer sees them frozen (gradient zero; mepd.inputs freezes them),
and each sampling starts from the last frames sampled near that solute
geometry (sequential sampling: H. Hu, Z. Lu, J. M. Parks, S. K. Burger,
W. Yang, J. Chem. Phys. 128, 034105 (2008), doi:10.1063/1.2816557).

Energies: the free energy itself is not a per-geometry average. Each node
gets E_QM(s) + <E_int(s, y)> (the QM energy plus the mean solute-environment
interaction at the low level; no entropy) for the path methods' tangents and
climbing image; `free_energy_profile` integrates the mean forces along a
finished path (thermodynamic integration) for the barrier.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.engines.modified import ModifiedEngine
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache

KB_HARTREE = 3.166811563e-6        # Hartree per K
HARTREE_KCAL = 627.509474

DEFAULTS = {"temperature": 300.0, "equilibrate_ps": 0.5, "sample_ps": 1.0, "frames": 10, "restraint": 1.0,
            "timestep_fs": 1.0}


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, c_mobile, c_target): mobile @ R moved onto target (both centred)."""
    cm, ct = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - cm).T @ (target - ct))
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return r, cm, ct


def onto_solute(frame: np.ndarray, solute: np.ndarray, idx: list[int]) -> np.ndarray:
    """`frame` (whole system) turned so its solute lies on `solute`, then
    the solute put back exactly."""
    r, cm, ct = _kabsch(frame[idx], solute)
    out = (frame - cm) @ r + ct
    out[idx] = solute
    return out


def _bare(node: StructureNode, coords_bohr: np.ndarray) -> StructureNode:
    """`node`'s structure at other coordinates, without a bond graph or cache."""
    return StructureNode(structure=node.structure.model_copy(update={"geometry": np.asarray(coords_bohr, dtype=float)}),
                         has_molecular_graph=False)


def _read_xyz_frames(fp: Path, natoms: int) -> list[np.ndarray]:
    lines = fp.read_text().splitlines()
    out, k = [], 0
    while k + natoms + 2 <= len(lines) and lines[k].strip():
        rows = lines[k + 2:k + 2 + natoms]
        out.append(np.array([[float(v) for v in r.split()[1:4]] for r in rows]))
        k += natoms + 2
    return out


@dataclass
class MeanForceEngine(ModifiedEngine):
    """`base`: the QM/MM engine (mepd.engines.qmmm.QMMMEngine, subtractive
    xtb low level). Settings as in DEFAULTS (temperature K, picoseconds,
    frames kept per geometry, restraint force constant Eh/bohr^2)."""

    base: Engine
    temperature: float = DEFAULTS["temperature"]
    equilibrate_ps: float = DEFAULTS["equilibrate_ps"]
    sample_ps: float = DEFAULTS["sample_ps"]
    frames: int = DEFAULTS["frames"]
    restraint: float = DEFAULTS["restraint"]
    timestep_fs: float = DEFAULTS["timestep_fs"]
    n_parallel: int = 4
    _warm: list = field(default_factory=list, repr=False)       # (solute coords, last frame) seen so far
    _lock: object = field(default_factory=threading.Lock, repr=False)
    last_samples: dict = field(default_factory=dict, repr=False)
    _estimates: dict = field(default_factory=dict, repr=False)   # solute geometry -> its own energy estimate
    _samplings: dict = field(default_factory=dict, repr=False)   # solute geometry -> its frames, for the live view
    _shown: dict = field(default_factory=dict, repr=False)       # live stream -> the solute geometry it shows
    max_live_frames: int = 12

    def __post_init__(self):
        from mepd.engines.qmmm import QMMMEngine, _OpenMMLow, _XTBLow

        if not isinstance(self.base, QMMMEngine):
            raise ValueError("the mean-force (free-energy) environment needs a QM/MM system of mepd's own "
                             "(not TeraChem's QM/MM)")
        low = self.base._low
        if isinstance(low, _XTBLow):
            self.sampler = "xtb"
        elif isinstance(low, _OpenMMLow):
            self.sampler = "openmm"
        else:
            raise ValueError(f"no environment dynamics for a {type(low).__name__} environment")
        self.electrostatic = bool(getattr(low, "electrostatic", False))
        self.region = self.base.region
        r = self.region
        self.solute = sorted(set(r.qm_atoms) | {int(m) for _, m in r.links})
        self.environment = [i for i in range(r.natoms) if i not in set(self.solute)]
        # Held during the environment's dynamics: the solute, and the region's
        # frozen outer shell (the droplet's or protein's boundary).
        self.held = sorted(set(self.solute) | {int(i) for i in r.frozen_atoms})
        self._inherit()

    def __repr__(self) -> str:
        return (f"MeanForceEngine({self.base!r}, {self.temperature:g} K, {self.sample_ps:g} ps x {self.frames} "
                "frames per geometry)")

    # ------------------------------------------------------------ sampling
    def _start(self, x: np.ndarray) -> np.ndarray:
        """Where the environment's dynamics start for solute geometry x: the
        last frame sampled at the nearest solute geometry so far, else x."""
        s = x[self.solute]
        with self._lock:
            warm = list(self._warm)
        if not warm:
            return x.copy()
        best = min(warm, key=lambda w: float(np.sum((w[0] - s) ** 2)))
        return onto_solute(best[1], x[self.held], self.held)

    def _remember(self, x: np.ndarray, frame: np.ndarray) -> None:
        with self._lock:
            self._warm.append((x[self.solute].copy(), frame.copy()))
            del self._warm[:-200]

    def _md(self, start_bohr: np.ndarray, charge: int) -> list[np.ndarray]:
        """Environment frames (bohr) from xtb dynamics with the solute held."""
        low = self.base._low
        r = self.region
        total = float(self.equilibrate_ps) + float(self.sample_ps)
        dump_fs = max(float(self.timestep_fs), 1000.0 * float(self.sample_ps) / max(1, int(self.frames)))
        atoms = ",".join(str(i + 1) for i in self.held)
        with tempfile.TemporaryDirectory(prefix="mepd_meanforce_") as tmp:
            tmp = Path(tmp)
            angstrom = start_bohr / ANGSTROM_TO_BOHR
            (tmp / "system.xyz").write_text(f"{r.natoms}\n\n" + "".join(
                f"{s} {a:.8f} {b:.8f} {c:.8f}\n" for s, (a, b, c) in zip(r.symbols, angstrom)))
            (tmp / "md.inp").write_text(
                f"$constrain\n   force constant={float(self.restraint)}\n   atoms: {atoms}\n$end\n"
                f"$md\n   temp={float(self.temperature)}\n   time={total}\n   dump={dump_fs}\n"
                f"   step={float(self.timestep_fs)}\n   hmass=4\n   shake=0\n   nvt=true\n$end\n")
            args = [low.engine.executable, "system.xyz", "--md", "--input", "md.inp",
                    "--chrg", str(int(charge)), "--uhf", str(int(r.qm_multiplicity) - 1)]
            if low.engine.method == "gfnff":
                topo = low.engine.topology(list(r.symbols), int(charge), low.ref_real)
                if topo is not None:
                    shutil.copyfile(topo, tmp / "gfnff_topo")
                args.append("--gfnff")
            else:
                args.append("--" + low.engine.method)
            env = {**__import__("os").environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"}
            proc = subprocess.run(args, cwd=tmp, capture_output=True, text=True, env=env, timeout=36000)
            trj = tmp / "xtb.trj"
            if proc.returncode != 0 or not trj.exists():
                from mepd.errors import ElectronicStructureError

                raise ElectronicStructureError(msg=f"xtb dynamics of the environment failed: "
                                                   f"{(proc.stdout + proc.stderr)[-600:]}")
            frames = _read_xyz_frames(trj, r.natoms)
        n_equil = int(round(1000.0 * float(self.equilibrate_ps) / dump_fs))
        kept = frames[n_equil:] or frames[-1:]
        return [f * ANGSTROM_TO_BOHR for f in kept]

    def sample(self, node: StructureNode) -> dict:
        """Mean force and energy at this node's solute geometry."""
        if self.sampler == "openmm":
            return self._sample_openmm(node)
        x = np.asarray(node.coords, dtype=float)
        r = self.region
        charge = int(node.structure.charge)
        frames = [onto_solute(f, x[self.held], self.held) for f in self._md(self._start(x), charge)]
        self._remember(x, frames[-1])
        low = self.base._low.terms(frames, charge)                      # E_low(real) - E_low(model), per frame
        env_only = self._environment_energies(frames, charge)
        model = StructureNode(structure=r.model_structure(node.structure), has_molecular_graph=False)
        g_qm = np.asarray(self.base.base.compute_gradients([model])[0], dtype=float)
        e_qm = float(model.energy)
        grads = np.array([lw["gradient"] for lw in low])
        mean = np.zeros_like(x)
        mean[self.solute] = r.model_gradient_to_full(g_qm)[self.solute] + grads[:, self.solute].mean(axis=0)
        e_int = np.array([lw["energy"] - e for lw, e in zip(low, env_only)])
        stderr = grads[:, self.solute].std(axis=0) / np.sqrt(max(1, len(frames) - 1))
        self._keep_for_live(node, frames, e_int, {"force_stderr": float(np.abs(stderr).max())})
        return {"energy": e_qm + float(e_int.mean()), "gradient": mean, "frames": len(frames),
                "force_stderr": float(np.abs(stderr).max()),
                "interaction_kcal": float(e_int.mean() * HARTREE_KCAL),
                "interaction_stderr_kcal": float(e_int.std() / np.sqrt(max(1, len(e_int) - 1)) * HARTREE_KCAL),
                "last_frame": frames[-1]}

    # ------------------------------------------------------------ OpenMM environments (AMBER, TIP3P)
    def _solute_charges(self, node: StructureNode) -> tuple[np.ndarray, dict]:
        """The QM region's atomic charges (the QM program's, Mulliken) at this
        geometry in its starting environment: what the environment's dynamics
        see in place of the QM density (QM/MM-MFEP's ESP-charge sampling,
        with the program's own charges). A link atom's charge goes to the QM
        atom it caps. Also returns that QM/MM evaluation."""
        r = self.region
        res = self.base.evaluate([node])[0]
        q_model = res.get("qm_charges")
        if q_model is None:
            raise ValueError(f"{type(self.base.base).__name__} does not report atomic charges: the environment's "
                             "dynamics need the QM region's charges")
        q_model = np.asarray(q_model, dtype=float).reshape(-1)
        nq = len(r.qm_atoms)
        q = q_model[:nq].copy()
        pos = {a: k for k, a in enumerate(r.qm_atoms)}
        for k, (qa, _) in enumerate(r.links):
            if nq + k < len(q_model):
                q[pos[qa]] += q_model[nq + k]
        return q, res

    def _openmm_frames(self, start_bohr: np.ndarray, qm_charges: Optional[np.ndarray], seed: int) -> list[np.ndarray]:
        """Environment frames (bohr) from OpenMM Langevin dynamics with the
        solute's atoms fixed exactly (mass 0); with electrostatic embedding
        the QM atoms carry `qm_charges`."""
        import openmm
        import openmm.unit as u

        from mepd.engines.qmmm import BOHR_NM

        low, r = self.base._low, self.region
        with self._lock:
            if getattr(self, "_system_xml", None) is None:
                self._system_xml = openmm.XmlSerializer.serialize(low.context.getSystem())
            xml = self._system_xml
        system = openmm.XmlSerializer.deserialize(xml)
        for i in self.held:
            system.setParticleMass(i, 0.0)
        if qm_charges is not None:
            nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
            for i, qi in zip(r.qm_atoms, qm_charges):
                _, sig, eps = nb.getParticleParameters(i)
                nb.setParticleParameters(i, float(qi), sig, eps)
        dt = float(self.timestep_fs) / 1000.0
        integrator = openmm.LangevinMiddleIntegrator(float(self.temperature) * u.kelvin, 1.0 / u.picosecond,
                                                     dt * u.picoseconds)
        integrator.setRandomNumberSeed(int(seed))
        try:
            platform = openmm.Platform.getPlatformByName("CPU")
            context = openmm.Context(system, integrator, platform, {"Threads": "1"})
        except Exception:
            context = openmm.Context(system, integrator)
        context.setPositions(np.asarray(start_bohr, dtype=float) * BOHR_NM * u.nanometer)
        openmm.LocalEnergyMinimizer.minimize(context, 10.0, 200)       # settle a solute that moved since
        context.setVelocitiesToTemperature(float(self.temperature) * u.kelvin, int(seed))
        n_equil = int(round(float(self.equilibrate_ps) / dt))
        n_total = max(1, int(round(float(self.sample_ps) / dt)))
        every = max(1, n_total // max(1, int(self.frames)))
        if n_equil:
            integrator.step(n_equil)
        frames = []
        for _ in range(max(1, int(self.frames))):
            integrator.step(every)
            pos = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
            frames.append(np.asarray(pos, dtype=float) / BOHR_NM)
        del context, integrator
        return frames

    def _sample_openmm(self, node: StructureNode) -> dict:
        r = self.region
        x = np.asarray(node.coords, dtype=float)
        start = self._start(x)
        start_node = _bare(node, start)
        q, first = self._solute_charges(start_node) if self.electrostatic else (None, None)
        seed = (abs(hash(x[self.solute].round(5).tobytes())) + len(self._warm)) % (2 ** 31)
        frames = [onto_solute(f, x[self.held], self.held) for f in self._openmm_frames(start, q, seed)]
        self._remember(x, frames[-1])
        frame_nodes = [_bare(node, f) for f in frames]
        if self.electrostatic:
            # The QM energy now depends on the environment: a QM/MM gradient in
            # every frame's field, averaged.
            res = self.base.evaluate(frame_nodes)
            grads = np.array([rr["gradient"] for rr in res])
            e_frames = np.array([rr["qm"] for rr in res])
            energy = float(e_frames.mean())                       # the QM energy in the field, averaged
            e_int = e_frames - float(first["qm"]) if first is not None else e_frames - e_frames.mean()
        else:
            # Mechanical: the QM energy of a fixed solute is the same in every
            # frame; only the force field's forces on it change.
            model = StructureNode(structure=r.model_structure(node.structure), has_molecular_graph=False)
            g_qm = r.model_gradient_to_full(np.asarray(self.base.base.compute_gradients([model])[0], dtype=float))
            low = self.base._low.terms(frames, int(node.structure.charge))
            grads = np.array([g_qm + lw["gradient"] for lw in low])
            energy = float(model.energy)
            e_int = np.array([lw["energy"] for lw in low]) - float(np.mean([lw["energy"] for lw in low]))
        mean = np.zeros_like(x)
        mean[self.solute] = grads[:, self.solute].mean(axis=0)
        stderr = grads[:, self.solute].std(axis=0) / np.sqrt(max(1, len(frames) - 1))
        self._keep_for_live(node, frames, e_frames if self.electrostatic else e_int,
                            {"force_stderr": float(np.abs(stderr).max())})
        return {"energy": energy, "gradient": mean, "frames": len(frames),
                "force_stderr": float(np.abs(stderr).max()),
                "interaction_kcal": float(e_int.mean() * HARTREE_KCAL),
                "interaction_stderr_kcal": float(e_int.std() / np.sqrt(max(1, len(e_int) - 1)) * HARTREE_KCAL),
                "solute_charges": None if q is None else [round(float(v), 4) for v in q],
                "last_frame": frames[-1]}

    def _environment_energies(self, frames: list[np.ndarray], charge: int) -> list[float]:
        """The environment alone (no solute), so the energy keeps only the
        solute-environment interaction. With cut bonds there is no clean
        environment alone: 0 (the energy is then the whole low level)."""
        r = self.region
        if r.links or not self.environment:
            return [0.0] * len(frames)
        low = self.base._low
        symbols = [r.symbols[i] for i in self.environment]
        env_charge = int(charge) - int(r.qm_charge)
        ref = r.reference_structure()
        ref_xyz = None
        if ref is not None and low.engine.method == "gfnff":
            g = np.asarray(ref.geometry, dtype=float)[self.environment] / ANGSTROM_TO_BOHR
            ref_xyz = f"{len(symbols)}\n\n" + "".join(f"{s} {a:.8f} {b:.8f} {c:.8f}\n" for s, (a, b, c) in zip(symbols, g))
        res = low.engine.energy_gradients([(symbols, f[self.environment], env_charge, ref_xyz, 0) for f in frames])
        return [float(e) for e, _ in res]

    # ------------------------------------------------------------ engine
    def _run(self, chain: Union[Chain, List]) -> list[StructureNode]:
        nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [n for n in nodes if n._cached_energy is None or n._cached_gradient is None]
        if todo:
            width = max(1, min(len(todo), int(self.n_parallel or 1)))
            if width == 1:
                res = [self.sample(n) for n in todo]
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=width) as pool:
                    res = list(pool.map(self.sample, todo))
            for n, r in zip(todo, res):
                self.last_samples[id(n)] = {k: v for k, v in r.items() if k not in ("gradient", "last_frame")}
            update_node_cache(node_list=todo, results=[
                FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                    {"energy": r["energy"], "gradient": r["gradient"]})}) for r in res])
            with self._lock:
                for n, r in zip(todo, res):
                    self._estimates[self._key(n)] = r["energy"]
        if isinstance(chain, Chain) and len(nodes) >= 3:
            self._chain_energies(nodes)
        return nodes

    def _keep_for_live(self, node: StructureNode, frames: list, energies, info: dict) -> None:
        """The frames this geometry's mean force was averaged over (at most
        max_live_frames), for the live view; the newest one is also shown as
        the latest sampling. Nothing is kept when no viewer is attached."""
        from mepd.progress import live_viewer_attached, write_sampling

        if not live_viewer_attached() or not frames:
            return
        r = self.region
        # About 45 bytes per atom per frame: keep each image's stream near 2 MB
        # at most (a protein shows fewer frames).
        cap = max(2, min(int(self.max_live_frames), int(2_000_000 / (45 * max(1, r.natoms)))))
        step = max(1, int(np.ceil(len(frames) / cap)))
        keep = list(range(0, len(frames), step))
        ang = 1.0 / ANGSTROM_TO_BOHR
        xyz = [f"{r.natoms}\nsampled frame {k + 1} of {len(frames)}\n" + "".join(
            f"{s} {a * ang:.4f} {b * ang:.4f} {c * ang:.4f}\n" for s, (a, b, c) in zip(r.symbols, frames[k]))
            for k in keep]
        e = np.asarray(energies, dtype=float) * HARTREE_KCAL
        e_rel = [float(e[k] - e.mean()) for k in keep] if len(e) == len(frames) else [None] * len(keep)
        what = ("QM energy in each frame's field" if self.electrostatic else "force-field energy"
                if self.sampler == "openmm" else "solute-environment interaction")
        caption = (f"{len(frames)} frames over {self.sample_ps:g} ps at {self.temperature:g} K (QM region held) · "
                   f"mean-force standard error {info['force_stderr']:.1e} Eh/bohr · plotted: {what}, vs its mean")
        entry = {"frames": xyz, "energies": e_rel, "caption": caption}
        with self._lock:
            self._samplings[self._key(node)] = entry
            del_keys = list(self._samplings)[:-400]
            for k in del_keys:
                self._samplings.pop(k, None)
        write_sampling("sampling_latest", xyz, e_rel, label="solvent sampling (latest)", caption=caption)

    def _show_chain_samplings(self, nodes: list[StructureNode]) -> None:
        """Each image of a path handed over whole gets its own live stream:
        the sampling its current mean force came from."""
        from mepd.progress import write_sampling

        for i, n in enumerate(nodes):
            key = self._key(n)
            stream = f"sampling_image_{i:02d}"
            entry = self._samplings.get(key)
            if entry is None or self._shown.get(stream) == key:
                continue
            self._shown[stream] = key
            write_sampling(stream, entry["frames"], entry["energies"], label=f"image {i + 1} · solvent",
                           caption=entry["caption"])

    def _key(self, node: StructureNode) -> bytes:
        return np.round(np.asarray(node.coords, dtype=float)[self.solute], 6).tobytes()

    def _chain_energies(self, nodes: list[StructureNode]) -> None:
        """A path handed over whole: its energies become the mean forces
        integrated along it (trapezoid), from the first node's estimate, so a
        path method's tangents and climbing image follow the free energy
        rather than the per-geometry estimate."""
        first = self._estimates.get(self._key(nodes[0]), nodes[0]._cached_energy)
        acc = 0.0
        prev = None
        for n in nodes:
            x = np.asarray(n.coords, dtype=float)[self.solute]
            g = np.asarray(n._cached_gradient, dtype=float)[self.solute]
            if prev is not None:
                acc += 0.5 * float(np.sum((g + prev[1]) * (x - prev[0])))
            n._cached_energy = float(first) + acc
            prev = (x, g)
        self._show_chain_samplings(nodes)

    def compute_hessian(self, *args, **kwargs):
        raise NotImplementedError("Hessians, TS optimizations and IRCs on the mean-force (free-energy) surface are not "
                                  "supported: use the path's climbing image and free_energy_profile, or the cage "
                                  "environment for a TS optimization")

    compute_transition_state = compute_hessian
    compute_sd_irc = compute_hessian

    def prepare_node_for_comparison(self, node: StructureNode) -> StructureNode:
        return self.base.prepare_node_for_comparison(node)

    def decompose(self, structures) -> list[dict]:
        return self.base.decompose(structures)


def free_energy_profile(chain, solute: Optional[list[int]] = None, engine: Optional["MeanForceEngine"] = None,
                        panels: Union[int, str] = "auto", dense: bool = False, tolerance_kcal: float = 0.5,
                        max_panels: int = 8):
    """A(s_i) - A(s_0) in kcal/mol along a path whose nodes carry mean forces
    (gradients of the free energy), by thermodynamic integration over the
    solute's displacement between neighbouring nodes.

    With `engine`, the mean force is also sampled inside each segment and
    Simpson's rule is used over `panels` panels per segment (2*panels
    sub-intervals; exact for a cubic). "auto" doubles a segment's panels
    until its integral changes by less than `tolerance_kcal` (at most
    `max_panels`): a bond breaking within one segment (the Menshutkin
    reaction's ion pair forming) needs several. Without `engine`, the
    trapezoid rule, which is crude when images are far apart (on HCN -> HNC
    with 8 images it put the barrier 11 kcal/mol low; one Simpson panel was
    within 0.3). `dense=True` also returns (fraction along the path, A) at
    every point sampled."""
    nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
    xs = [np.asarray(n.coords, dtype=float) for n in nodes]
    gs = [np.asarray(n.gradient, dtype=float) for n in nodes]
    if solute is None:
        solute = engine.solute if engine is not None else list(range(len(xs[0])))
    total = max(1, len(nodes) - 1)
    a, fine = [0.0], [(0.0, 0.0)]
    if engine is None or len(nodes) < 2:
        for k in range(1, len(nodes)):
            dx = xs[k][solute] - xs[k - 1][solute]
            a.append(a[-1] + 0.5 * float(np.sum((gs[k][solute] + gs[k - 1][solute]) * dx)))
            fine.append((k / total, a[-1]))
        profile = np.asarray(a) * HARTREE_KCAL
        return (profile, [(t, v * HARTREE_KCAL) for t, v in fine]) if dense else profile

    slopes: dict = {}          # (segment, fraction as a reduced pair) -> dA/dt along the segment

    def need(m: int, segments) -> None:
        todo = []
        for k in segments:
            for jj in range(1, 2 * m):
                key = (k, *_reduced(jj, 2 * m))
                if key not in slopes and key not in [t[0] for t in todo]:
                    todo.append((key, xs[k - 1] + (xs[k] - xs[k - 1]) * jj / (2 * m)))
        if todo:
            pts = [nodes[0].update_coords(x) for _, x in todo]
            for (key, _), g in zip(todo, engine.compute_gradients(pts)):
                k = key[0]
                slopes[key] = float(np.sum(np.asarray(g, dtype=float)[solute] * (xs[k][solute] - xs[k - 1][solute])))

    def slope(k: int, jj: int, n: int) -> float:
        if jj == 0:
            return float(np.sum(gs[k - 1][solute] * (xs[k][solute] - xs[k - 1][solute])))
        if jj == n:
            return float(np.sum(gs[k][solute] * (xs[k][solute] - xs[k - 1][solute])))
        return slopes[(k, *_reduced(jj, n))]

    def integral(k: int, m: int) -> tuple[float, list]:
        h, acc, pts = 1.0 / (2 * m), 0.0, []
        for p in range(m):
            f0, f1, f2 = slope(k, 2 * p, 2 * m), slope(k, 2 * p + 1, 2 * m), slope(k, 2 * p + 2, 2 * m)
            pts.append(((2 * p + 1) * h, acc + h * (5 * f0 + 8 * f1 - f2) / 12))
            acc += h * (f0 + 4 * f1 + f2) / 3
            pts.append(((2 * p + 2) * h, acc))
        return acc, pts

    fixed = panels != "auto"
    m_seg = {k: (max(1, int(panels)) if fixed else 1) for k in range(1, len(nodes))}
    need(max(m_seg.values()), list(m_seg))
    if not fixed:
        tol = float(tolerance_kcal) / HARTREE_KCAL
        open_segments = set(m_seg)
        while open_segments:
            for k in open_segments:
                m_seg[k] *= 2
            need_m = {}
            for k in open_segments:
                need_m.setdefault(m_seg[k], []).append(k)
            for m, segments in need_m.items():
                need(m, segments)
            done = set()
            for k in open_segments:
                coarse, _ = integral(k, m_seg[k] // 2)
                finer, _ = integral(k, m_seg[k])
                if abs(finer - coarse) < tol or m_seg[k] >= int(max_panels):
                    done.add(k)
            open_segments -= done
    for k in range(1, len(nodes)):
        value, pts = integral(k, m_seg[k])
        fine.extend(((k - 1 + t) / total, a[-1] + v) for t, v in pts)
        a.append(a[-1] + value)
    profile = np.asarray(a) * HARTREE_KCAL
    if dense:
        return profile, [(t, v * HARTREE_KCAL) for t, v in fine]
    return profile


def _reduced(j: int, n: int) -> tuple[int, int]:
    from math import gcd

    g = gcd(j, n)
    return j // g, n // g


def mean_force_engine(engine) -> Optional[MeanForceEngine]:
    """The MeanForceEngine inside `engine` (through its wrappers), or None."""
    seen = 0
    while engine is not None and seen < 10:
        if isinstance(engine, MeanForceEngine):
            return engine
        engine = engine.__dict__.get("base") if hasattr(engine, "__dict__") else None
        seen += 1
    return None


def apply_free_energy_profile(chain, engine) -> Optional[np.ndarray]:
    """On the mean-force surface: replace a finished path's node energies by
    its free-energy profile (Simpson thermodynamic integration, midpoints
    sampled), so its barrier and reaction energy are free energies. The
    first node keeps the free energy it already got from an earlier path
    that ended there (recursive splits stay on one scale), else its own
    estimate. Returns the profile (kcal/mol), or None off that surface."""
    mf = mean_force_engine(engine)
    if mf is None:
        return None
    nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
    engine.compute_gradients(nodes)
    profile = free_energy_profile(nodes, engine=mf)
    key = lambda n: np.round(np.asarray(n.coords, dtype=float)[mf.solute], 6).tobytes()
    with mf._lock:
        anchor = mf.__dict__.setdefault("_free_energy", {}).get(key(nodes[0]), float(nodes[0].energy))
        for n, a in zip(nodes, profile):
            n._cached_energy = anchor + float(a) / HARTREE_KCAL
            mf._free_energy.setdefault(key(n), n._cached_energy)
    return profile
