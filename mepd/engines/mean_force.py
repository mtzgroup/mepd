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
the environment at `temperature` with the solute held: xtb's dynamics at
the region's low level (GFN-FF, or GFN1/GFN2), the solute kept rigid by
stiff restraints on all its distances (xtb ignores exact fixing in MD), each
frame then turned back onto the solute and the solute put back exactly. In
the subtractive (mechanical) embedding the QM energy of a fixed solute does
not depend on the environment, so these dynamics sample the QM/MM ensemble
exactly and each frame costs only low-level gradients: one QM gradient per
geometry in all.

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

    def __post_init__(self):
        from mepd.engines.qmmm import QMMMEngine, _XTBLow

        if not isinstance(self.base, QMMMEngine) or not isinstance(self.base._low, _XTBLow):
            raise ValueError("the mean-force (free-energy) environment needs a QM/MM system with an xTB-family "
                             "environment (GFN-FF, GFN1 or GFN2): AMBER/TIP3P environments are not supported yet")
        if self.base.region.embedding == "electrostatic":
            raise ValueError("the mean-force environment needs mechanical embedding (the QM energy of a fixed "
                             "solute must not depend on the environment)")
        self.region = self.base.region
        r = self.region
        self.solute = sorted(set(r.qm_atoms) | {int(m) for _, m in r.links})
        self.environment = [i for i in range(r.natoms) if i not in set(self.solute)]
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
        return onto_solute(best[1], s, self.solute)

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
        atoms = ",".join(str(i + 1) for i in self.solute)
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
        x = np.asarray(node.coords, dtype=float)
        r = self.region
        charge = int(node.structure.charge)
        frames = [onto_solute(f, x[self.solute], self.solute) for f in self._md(self._start(x), charge)]
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
        return {"energy": e_qm + float(e_int.mean()), "gradient": mean, "frames": len(frames),
                "force_stderr": float(np.abs(stderr).max()),
                "interaction_kcal": float(e_int.mean() * HARTREE_KCAL),
                "interaction_stderr_kcal": float(e_int.std() / np.sqrt(max(1, len(e_int) - 1)) * HARTREE_KCAL),
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
        return nodes

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


def free_energy_profile(chain, solute: Optional[list[int]] = None, engine: Optional["MeanForceEngine"] = None
                        ) -> np.ndarray:
    """A(s_i) - A(s_0) in kcal/mol along a path whose nodes carry mean forces
    (gradients of the free energy), by thermodynamic integration over the
    solute's displacement between neighbouring nodes. With `engine`, the mean
    force is also sampled halfway along each segment and Simpson's rule is
    used (exact for a cubic along the segment); without, the trapezoid rule,
    which is crude when images are far apart (on HCN -> HNC with 8 images it
    put the barrier 11 kcal/mol low; Simpson's was within 0.3)."""
    nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
    xs = [np.asarray(n.coords, dtype=float) for n in nodes]
    gs = [np.asarray(n.gradient, dtype=float) for n in nodes]
    if solute is None:
        solute = engine.solute if engine is not None else list(range(len(xs[0])))
    mids = None
    if engine is not None and len(nodes) > 1:
        mid_nodes = [nodes[k - 1].update_coords(0.5 * (xs[k - 1] + xs[k])) for k in range(1, len(nodes))]
        mids = [np.asarray(g, dtype=float) for g in engine.compute_gradients(mid_nodes)]
    a = [0.0]
    for k in range(1, len(nodes)):
        dx = xs[k][solute] - xs[k - 1][solute]
        if mids is None:
            step = 0.5 * float(np.sum((gs[k][solute] + gs[k - 1][solute]) * dx))
        else:
            step = float(np.sum((gs[k - 1][solute] + 4 * mids[k - 1][solute] + gs[k][solute]) * dx)) / 6.0
        a.append(a[-1] + step)
    return np.asarray(a) * HARTREE_KCAL


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
