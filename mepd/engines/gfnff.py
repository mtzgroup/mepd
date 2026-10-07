"""GFN-FF (the xtb program's generic force field) as an mepd engine.

GFN-FF needs no parameters from the user and covers almost the whole
periodic table, so it can describe any QM/MM environment (solvent shells,
proteins, metal sites) as given. It is the default low level of
`mepd.engines.qmmm.QMMMEngine`.

GFN-FF assigns its bonded terms from a *topology* worked out from a
geometry. Left to itself, xtb would work it out again from every new
geometry, so a bond breaking along a path would switch force-field terms
mid-path (a jump in energy). A `topology_reference` fixes it: every geometry
of a system is computed with the topology of its reference geometry (cached
on disk), so all structures of a network sit on one smooth surface.

GFN-FF: S. Spicher, S. Grimme, Angew. Chem. Int. Ed. 59, 15665 (2020),
doi:10.1002/anie.202004239.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from qcconst.constants import ANGSTROM_TO_BOHR

from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.errors import ElectronicStructureError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.nodehelpers import update_node_cache

_TOPO_LOCKS: dict[str, threading.Lock] = {}
_TOPO_GUARD = threading.Lock()


def _xyz(symbols, coords_bohr) -> str:
    x = np.asarray(coords_bohr, dtype=float).reshape(-1, 3) / ANGSTROM_TO_BOHR
    return f"{len(symbols)}\n\n" + "".join(f"{s} {a:.10f} {b:.10f} {c:.10f}\n" for s, (a, b, c) in zip(symbols, x))


def parse_engrad(text: str) -> tuple[float, np.ndarray]:
    """Energy (Eh) and gradient (Eh/bohr) from xtb's ORCA-style .engrad."""
    vals = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    n = int(vals[0])
    energy = float(vals[1])
    grad = np.array([float(v) for v in vals[2:2 + 3 * n]]).reshape(n, 3)
    return energy, grad


def _raise_stack_limit() -> None:
    """xtb's GFN-FF setup overflows the default 8 MB stack on large systems
    (a crash in gfnff neighbor setup at ~2700 atoms)."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    want = 512 * 1024 * 1024
    if soft != resource.RLIM_INFINITY and soft < want:
        resource.setrlimit(resource.RLIMIT_STACK, (want if hard == resource.RLIM_INFINITY else min(want, hard), hard))


def memory_gb(natoms: int) -> float:
    """Rough peak memory of one GFN-FF run (measured: 7.6 GB at 2685
    atoms; it grows with the square of the atom count)."""
    return 0.15 + 1.06e-6 * natoms ** 2


def available_gb() -> Optional[float]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def cache_dir() -> Path:
    base = os.getenv("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "mepd" / "gfnff_topo"


@dataclass
class GFNFFEngine(Engine):
    """`topology_reference`: xyz text whose GFN-FF topology every geometry
    with the same atoms uses (see the module doc); None = xtb's own (a
    fresh topology for each geometry)."""

    executable: Optional[str] = None
    method: str = "gfnff"          # or "gfn2" / "gfn1" (xtb's tight-binding methods; no topology)
    n_parallel: int = 4
    topology_reference: Optional[str] = None
    timeout: float = 3600.0
    _topologies: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.executable is None:
            from mepd.programs import xtb_executable

            self.executable = xtb_executable()
        if not self.executable:
            raise ElectronicStructureError(msg="GFN-FF needs the xtb program (set XTB_EXECUTABLE).")
        self.method = str(self.method).lower()
        if self.method not in ("gfnff", "gfn2", "gfn1"):
            raise ValueError(f"xtb method must be gfnff, gfn2 or gfn1, got {self.method!r}")

    # ------------------------------------------------------------ program
    def _run(self, workdir: Path, symbols, coords_bohr, charge: int, grad: bool = True,
             uhf: int = 0, point_charges=None) -> tuple[float, np.ndarray]:
        (workdir / "in.xyz").write_text(_xyz(symbols, coords_bohr))
        method = ["--gfnff"] if self.method == "gfnff" else ["--gfn", self.method[-1], "--uhf", str(int(uhf))]
        argv = [self.executable, "in.xyz", *method, "--chrg", str(int(charge))]
        if point_charges is not None and len(point_charges):
            # xtb's electrostatic embedding: charge, position (bohr) and the
            # element whose chemical hardness damps its interaction at short
            # range; the gradient on the charges goes to `pcgrad`.
            (workdir / "pcharge").write_text(f"{len(point_charges)}\n" + "".join(
                f"{q:.8f} {x:.10f} {y:.10f} {z:.10f} {int(el)}\n" for q, (x, y, z), el in point_charges))
            (workdir / "xcontrol").write_text("$embedding\n  input=pcharge\n  gradient=pcgrad\n$end\n")
            argv += ["--input", "xcontrol"]
        if grad:
            argv.append("--grad")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_STACKSIZE": "256M"}
        proc = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, env=env, timeout=self.timeout,
                              preexec_fn=_raise_stack_limit)
        engrad = workdir / "in.engrad"
        if proc.returncode != 0 or (grad and not engrad.exists()):
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
            raise ElectronicStructureError(msg=f"GFN-FF (xtb) failed:\n{tail}")
        if grad:
            return parse_engrad(engrad.read_text())
        for line in proc.stdout.splitlines():
            if "TOTAL ENERGY" in line:
                return float(line.split()[3]), np.zeros((len(symbols), 3))
        raise ElectronicStructureError(msg="GFN-FF (xtb): no energy in the output")

    def topology(self, symbols, charge: int, reference_xyz: Optional[str] = None) -> Optional[Path]:
        """The cached gfnff_topo for these atoms (built from the reference)."""
        ref = reference_xyz if reference_xyz is not None else self.topology_reference
        if ref is None or self.method != "gfnff":
            return None
        from qcdata import Structure

        s = Structure.from_xyz(ref)
        if [str(a) for a in s.symbols] != [str(a) for a in symbols]:
            return None
        key = hashlib.sha1(f"{charge}|{ref}|{self.executable}".encode()).hexdigest()[:16]
        if key in self._topologies:
            return self._topologies[key]
        with _TOPO_GUARD:
            lock = _TOPO_LOCKS.setdefault(key, threading.Lock())
        with lock:
            target = cache_dir() / key / "gfnff_topo"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
                    self._run(Path(tmp), list(s.symbols), np.asarray(s.geometry), charge, grad=True)
                    shutil.move(str(Path(tmp) / "gfnff_topo"), target)
            self._topologies[key] = target
        return target

    def energy_gradient(self, symbols, coords_bohr, charge: int = 0,
                        reference_xyz: Optional[str] = None, uhf: int = 0) -> tuple[float, np.ndarray]:
        topo = self.topology(symbols, charge, reference_xyz)
        with tempfile.TemporaryDirectory(prefix="mepd_xtb_") as tmp:
            if topo is not None:
                shutil.copyfile(topo, Path(tmp) / "gfnff_topo")
            return self._run(Path(tmp), symbols, coords_bohr, charge, uhf=uhf)

    def energy_gradients(self, jobs: list[tuple]) -> list[tuple[float, np.ndarray]]:
        """Several (symbols, coords_bohr, charge, reference_xyz[, uhf]) at once."""
        width = max(1, min(len(jobs), int(self.n_parallel or 1)))
        if self.method == "gfnff" and jobs:
            # Large systems: as many at once as memory allows (each run
            # needs memory_gb(N); half of what is free is used).
            free = available_gb()
            if free is not None:
                width = max(1, min(width, int(0.5 * free / memory_gb(max(len(j[0]) for j in jobs)))))
        if width == 1:
            return [self.energy_gradient(*j) for j in jobs]
        with ThreadPoolExecutor(max_workers=width) as pool:
            return list(pool.map(lambda j: self.energy_gradient(*j), jobs))

    # ------------------------------------------------------------ engine
    def _compute(self, chain: Union[Chain, List]) -> list:
        nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [n for n in nodes if n._cached_energy is None or n._cached_gradient is None]
        if todo:
            res = self.energy_gradients([(list(n.symbols), np.asarray(n.coords), int(n.structure.charge), None,
                                          int(n.structure.multiplicity) - 1) for n in todo])
            update_node_cache(node_list=todo, results=[
                FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                    {"energy": e, "gradient": g})}) for e, g in res])
        return nodes

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        return np.array([n.gradient for n in self._compute(chain)])

    def compute_energies(self, chain: Union[Chain, List]) -> NDArray:
        return np.array([n.energy for n in self._compute(chain)])

    def _ase(self, node):
        from mepd.engines.ase import ASEEngine
        from mepd.engines.gxtb import _GXTBASEResultsCalculator

        return ASEEngine(calculator=_GXTBASEResultsCalculator(self, charge=int(node.structure.charge),
                                                              multiplicity=int(node.structure.multiplicity)))

    def compute_geometry_optimization(self, node, keywords: dict | None = None):
        return self._ase(node).compute_geometry_optimization(node, keywords=dict(keywords or {}))

    def compute_transition_state(self, node, keywords: dict | None = None):
        return self._ase(node).compute_transition_state(node=node, keywords=keywords)

    def compute_irc_chain(self, ts_node, keywords: dict | None = None):
        return self._ase(ts_node).compute_irc_chain(ts_node=ts_node, keywords=keywords)


_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "P": 15, "S": 16, "Cl": 17, "K": 19, "Br": 35, "I": 53}


@dataclass
class XTBEngine(GFNFFEngine):
    """GFN2-xTB (or GFN1) through the xtb program, as a QM level
    (engine_name = "xtb"). It takes point charges, so it can be the QM level
    of electrostatically embedded QM/MM (e.g. in TIP3P water): seconds or
    less per gradient where DFT takes many.

    GFN2-xTB: C. Bannwarth, S. Ehlert, S. Grimme, J. Chem. Theory Comput. 15,
    1652 (2019), doi:10.1021/acs.jctc.8b01176."""

    method: str = "gfn2"
    supports_point_charges: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.method == "gfnff":
            raise ValueError("engine_name = \"xtb\" is GFN2- or GFN1-xTB (method = \"gfn2\" / \"gfn1\")")
        self._local = threading.local()

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_local", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._local = threading.local()

    @property
    def last_charges(self):
        """Atomic charges of this thread's last calculation (what the
        environment sees of the QM region between QM steps)."""
        return getattr(self._local, "charges", None)

    def energy_gradient(self, symbols, coords_bohr, charge: int = 0, multiplicity: int = 1,
                        point_charges=None, elements=None) -> tuple[float, np.ndarray, Optional[np.ndarray]]:
        """(energy Eh, gradient Eh/bohr, gradient on the point charges or
        None). `point_charges`: [(q, (x, y, z) in bohr), ...]; `elements`:
        the element of each (its hardness damps the interaction; H if not
        given)."""
        pcs = list(point_charges or [])
        els = list(elements) if elements is not None else ["H"] * len(pcs)
        pc = [(float(q), tuple(float(v) for v in xyz), _Z.get(str(el), 1)) for (q, xyz), el in zip(pcs, els)]
        with tempfile.TemporaryDirectory(prefix="mepd_xtb_") as tmp:
            e, g = self._run(Path(tmp), symbols, coords_bohr, charge, uhf=int(multiplicity) - 1,
                             point_charges=pc or None)
            fp = Path(tmp) / "charges"
            self._local.charges = np.loadtxt(fp).reshape(-1) if fp.exists() else None
            pcg = np.loadtxt(Path(tmp) / "pcgrad").reshape(-1, 3) if pc else None
        return e, g, pcg

    def energy_gradients(self, jobs: list[tuple]) -> list[tuple[float, np.ndarray]]:
        def one(j):
            symbols, coords, charge, _, uhf = (list(j) + [None, 0])[:5]
            return self.energy_gradient(symbols, coords, charge, int(uhf) + 1)[:2]

        width = max(1, min(len(jobs), int(self.n_parallel or 1)))
        if width == 1:
            return [one(j) for j in jobs]
        with ThreadPoolExecutor(max_workers=width) as pool:
            return list(pool.map(one, jobs))

