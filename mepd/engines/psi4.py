"""Psi4 as an mepd engine, run in Psi4's own Python environment.

Psi4 is free (LGPL) but is distributed through conda-forge, not PyPI, so it
cannot live in mepd's environment. mepd talks to a worker process started
with Psi4's Python (`python`: $MEPD_PSI4_PYTHON, ~/.local/opt/psi4/bin/python,
or the Python next to a `psi4` on PATH) over JSON lines: one start-up per
engine, not per gradient.

Besides plain energies and gradients, it computes them in a field of point
charges (`energy_gradient(..., point_charges=...)`), with the gradient on
the charges, which is what electrostatic QM/MM embedding needs (see
mepd.engines.qmmm). Point-charge positions are in bohr.

Install Psi4 once, e.g.
    micromamba create -p ~/.local/opt/psi4 -c conda-forge psi4 python=3.12

Psi4: D. G. A. Smith et al., J. Chem. Phys. 152, 184108 (2020),
doi:10.1063/5.0006002.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.errors import ElectronicStructureError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.nodehelpers import update_node_cache

_DRIVER = r'''
import json, sys, os, tempfile
import numpy as np
import psi4
cfg = json.loads(sys.argv[1])
scratch = tempfile.mkdtemp(prefix="mepd_psi4_")
os.chdir(scratch)
psi4.core.set_output_file(os.path.join(scratch, "psi4.out"), False)
psi4.set_memory(cfg["memory"])
psi4.set_num_threads(int(cfg["threads"]))
for line in sys.stdin:
    req = json.loads(line)
    try:
        psi4.core.clean()
        psi4.core.clean_options()
        opts = {"basis": cfg["basis"], "scf_type": "df", **cfg.get("options", {})}
        if req["multiplicity"] != 1:
            opts.setdefault("reference", "uks" if cfg.get("dft", True) else "uhf")
        psi4.set_options(opts)
        geom = "\n".join(f"{s} {x:.12f} {y:.12f} {z:.12f}" for s, (x, y, z) in zip(req["symbols"], req["coords"]))
        mol = psi4.geometry(f"{req['charge']} {req['multiplicity']}\n{geom}\nunits bohr\nno_reorient\nno_com\nsymmetry c1")
        kw = {"molecule": mol, "return_wfn": True}
        if req.get("point_charges"):
            kw["external_potentials"] = req["point_charges"]
        g, wfn = psi4.gradient(cfg["method"], **kw)
        out = {"energy": wfn.energy(), "gradient": np.asarray(g).tolist()}
        try:
            psi4.oeprop(wfn, "MULLIKEN_CHARGES", title="mepd")
            out["charges"] = np.asarray(wfn.atomic_point_charges()).tolist()
        except Exception:
            pass
        if req.get("point_charges"):
            out["pc_gradient"] = np.asarray(wfn.external_pot().gradient_on_charges()).tolist()
    except Exception as exc:
        out = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
'''


def psi4_python() -> Optional[str]:
    """The Python interpreter that has Psi4."""
    for cand in (os.getenv("MEPD_PSI4_PYTHON"), str(Path.home() / ".local/opt/psi4/bin/python")):
        if cand and Path(cand).exists():
            return cand
    exe = shutil.which("psi4")
    if exe:
        py = Path(exe).resolve().parent / "python"
        if py.exists():
            return str(py)
    return None


INSTALL = ("Psi4 is not installed: install it into its own environment, e.g. `micromamba create -p "
           "~/.local/opt/psi4 -c conda-forge psi4 python=3.12` (or set MEPD_PSI4_PYTHON to a Python that has psi4)")


@dataclass
class Psi4Engine(Engine):
    """`method`/`basis`: any Psi4 method and basis (e.g. "b3lyp", "def2-svp");
    `options`: extra Psi4 options; `threads`: Psi4 threads; `memory`."""

    method: str = "b3lyp"
    basis: str = "def2-svp"
    threads: int = 4
    memory: str = "4 GB"
    options: dict = field(default_factory=dict)
    python: Optional[str] = None
    timeout: float = 3600.0
    supports_point_charges: bool = True

    def __post_init__(self):
        self.python = self.python or psi4_python()
        if not self.python:
            raise ElectronicStructureError(msg=INSTALL)
        self._proc = None
        self._lock = threading.Lock()

    def __getstate__(self):     # deep copies / pickles get their own worker
        d = dict(self.__dict__)
        d["_proc"], d["_lock"] = None, None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._lock = threading.Lock()

    def _worker(self):
        if self._proc is None or self._proc.poll() is not None:
            cfg = {"method": self.method, "basis": self.basis, "threads": self.threads, "memory": self.memory,
                   "options": self.options, "dft": self.method.lower() not in ("hf", "scf", "mp2", "ccsd")}
            env = {**os.environ, "OMP_NUM_THREADS": str(self.threads), "PYTHONNOUSERSITE": "1"}
            env.pop("PYTHONPATH", None)
            self._proc = subprocess.Popen([self.python, "-c", _DRIVER, json.dumps(cfg)], stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        return self._proc

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def energy_gradient(self, symbols, coords_bohr, charge: int = 0, multiplicity: int = 1,
                        point_charges=None) -> tuple[float, np.ndarray, Optional[np.ndarray]]:
        """(energy Eh, gradient Eh/bohr, gradient on the point charges or
        None). `point_charges`: [(q, (x, y, z) in bohr), ...]."""
        req = {"symbols": [str(s) for s in symbols],
               "coords": np.asarray(coords_bohr, dtype=float).reshape(-1, 3).tolist(),
               "charge": int(charge), "multiplicity": int(multiplicity),
               "point_charges": [[float(q), [float(v) for v in xyz]] for q, xyz in (point_charges or [])]}
        with self._lock:
            proc = self._worker()
            try:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
            except BrokenPipeError:
                line = ""
            if not line:
                err = proc.stderr.read()[-2000:] if proc.stderr else ""
                self._proc = None
                raise ElectronicStructureError(msg=f"the Psi4 worker stopped:\n{err}")
        out = json.loads(line)
        if "error" in out:
            raise ElectronicStructureError(msg=f"Psi4 failed: {out['error']}")
        pcg = np.asarray(out["pc_gradient"]) if "pc_gradient" in out else None
        # Atomic charges of this calculation (Mulliken): what the environment
        # sees of the QM region between QM steps (microiterations).
        self.last_charges = np.asarray(out["charges"]) if "charges" in out else None
        return float(out["energy"]), np.asarray(out["gradient"]), pcg

    # ------------------------------------------------------------ engine
    def _compute(self, chain: Union[Chain, List]) -> list:
        nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [n for n in nodes if n._cached_energy is None or n._cached_gradient is None]
        results = []
        for n in todo:
            e, g, _ = self.energy_gradient(list(n.symbols), n.coords, int(n.structure.charge),
                                           int(n.structure.multiplicity))
            results.append(FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                {"energy": e, "gradient": g})}))
        if todo:
            update_node_cache(node_list=todo, results=results)
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
