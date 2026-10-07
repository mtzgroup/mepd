"""TeraChem's own QM/MM (electrostatic embedding, AMBER prmtop), ported
from neb-dynamics' QMMMEngine.

Each energy/gradient is one TeraChem run with the files
tc.in / ref.prmtop / ref.rst7 / qmindices.dat, through qccompute (a local
TeraChem) or ChemCloud. Geometry optimizations, TS searches and IRCs run
through ASE on these gradients (with the frozen atoms held by
mepd.engines.frozen.FrozenAtomsEngine), not through TeraChem's minimizer.

Set it up with `[qmmm] mm = "terachem"` (plus `prmtop`, and either `tcin`
(a tc.in template) or `program_kwds.model` for method/basis), or convert an
existing TeraChem input with `mepd qmmm from-tc tc.in`.

Untested against a live TeraChem in this repository (none is available
here); the input/output handling is checked against TeraChem output text.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from qcconst.constants import ANGSTROM_TO_BOHR

from mepd.chain import Chain
from mepd.engines.modified import ModifiedEngine
from mepd.errors import ElectronicStructureError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.nodehelpers import update_node_cache
from mepd.qmmm import QMMMRegion, parse_indices

logger = logging.getLogger(__name__)


# ------------------------------------------------------------- file formats
def structure_to_rst7(coords_bohr: np.ndarray, title: str = "mepd") -> str:
    """AMBER restart/inpcrd text (Å, 6 numbers per line)."""
    x = np.asarray(coords_bohr, dtype=float).reshape(-1) / ANGSTROM_TO_BOHR
    lines = [title, f"{len(x) // 3:6d}"]
    for k in range(0, len(x), 6):
        lines.append("".join(f"{v:12.7f}" for v in x[k:k + 6]))
    return "\n".join(lines) + "\n"


def rst7_coords(text: str) -> np.ndarray:
    """Coordinates (Å) from rst7 text (velocities and box ignored)."""
    lines = text.splitlines()
    natom = int(lines[1].split()[0])
    vals: list[float] = []
    for line in lines[2:]:
        for k in range(0, len(line.rstrip()), 12):
            chunk = line[k:k + 12].strip()
            if chunk:
                vals.append(float(chunk))
        if len(vals) >= 3 * natom:
            break
    return np.array(vals[:3 * natom]).reshape(natom, 3)


def prmtop_symbols(text: str) -> list[str]:
    from mepd.qmmm import _ELEMENT_Z

    by_z = {z: s for s, z in _ELEMENT_Z.items()}
    out, on = [], False
    for line in text.splitlines():
        if line.startswith("%FLAG"):
            on = line.split()[1] == "ATOMIC_NUMBER"
            continue
        if on and not line.startswith("%FORMAT"):
            out.extend(by_z[int(t)] for t in line.split())
    return out


def parse_qmmm_gradients(text: str) -> tuple[np.ndarray, np.ndarray]:
    """QM and MM gradient blocks from TeraChem QM/MM stdout."""
    qm, mm, mode = [], [], None
    for line in text.splitlines():
        clean = line.strip()
        if "dE/dX" in clean and "dE/dY" in clean:
            mode = "QM"
            continue
        if "MM / Point charge part" in clean:
            mode = "MM"
            continue
        if "Net gradient" in clean or (mode == "MM" and "---" in clean):
            mode = None
            continue
        parts = clean.split()
        if mode and len(parts) == 3:
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            (qm if mode == "QM" else mm).append(vals)
    return np.array(qm), np.array(mm)


def parse_final_energy(text: str) -> float:
    for line in text.splitlines():
        if "FINAL ENERGY:" in line:
            return float(line.split("FINAL ENERGY:")[1].split()[0])
    raise ElectronicStructureError(msg="TeraChem output has no FINAL ENERGY line")


def n_link_atoms(text: str) -> int:
    for line in text.splitlines():
        m = re.search(r"(\d+)\s+link atoms?", line, re.I)
        if m:
            return int(m.group(1))
    return 0


def parse_tcin(text: str) -> dict:
    """method/basis/charge/spinmult/run/prmtop/coordinates/qmindices, other
    keywords, and frozen atoms (0-based) from a TeraChem input's $constraints."""
    parsed = {"keywords": {}, "frozen_atom_indices": []}
    in_c = False
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if not line or line.startswith("#"):
            continue
        if low.startswith("$constraints"):
            in_c = True
            continue
        if in_c and low.startswith("$"):
            in_c = False
            continue
        if in_c:
            tok = line.split()
            if tok and tok[0].lower() == "atom" and len(tok) > 1:
                parsed["frozen_atom_indices"].extend(i - 1 for i in parse_indices(tok[1]))
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        key, val = parts[0].lower(), parts[1].strip()
        if key in ("method", "basis", "prmtop", "coordinates", "qmindices", "run", "min_coordinates"):
            parsed[key] = val
        elif key in ("charge", "spinmult"):
            parsed[key] = int(val)
        elif key != "scrdir":
            parsed["keywords"][parts[0]] = val
    parsed["frozen_atom_indices"] = sorted(set(parsed["frozen_atom_indices"]))
    return parsed


def with_keyword(text: str, key: str, value) -> str:
    pat = re.compile(rf"(?im)^\s*{re.escape(key)}\s+.*$")
    line = f"{key} {value}"
    return pat.sub(line, text, count=1) if pat.search(text) else text.rstrip() + f"\n{line}\n"


def without_constraints(text: str) -> str:
    out, skip = [], False
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("$constraints"):
            skip = True
            continue
        if skip and low.startswith("$end"):
            skip = False
            continue
        if not skip:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


# ------------------------------------------------------------------ engine
@dataclass
class TeraChemQMMMEngine(ModifiedEngine):
    tcin: str = ""                    # tc.in template (gradient run)
    prmtop: str = ""                  # prmtop text
    qm_atoms: list[int] = field(default_factory=list)
    compute_program: str = "chemcloud"   # or "qccompute" (a local TeraChem)
    chemcloud_queue: Optional[str] = None
    base: object = None

    def __post_init__(self):
        text = without_constraints(self.tcin)
        for key, val in (("run", "gradient"), ("coordinates", "ref.rst7"), ("prmtop", "ref.prmtop"),
                         ("qmindices", "qmindices.dat")):
            text = with_keyword(text, key, val)
        self.tcin = text
        self.qmindices = "\n".join(str(i) for i in self.qm_atoms) + "\n"

    @classmethod
    def from_region(cls, region: QMMMRegion, *, program_args=None, compute_program: str = "chemcloud",
                    chemcloud_queue: Optional[str] = None) -> "TeraChemQMMMEngine":
        if not region.prmtop:
            raise ValueError("qmmm.mm = 'terachem' needs `prmtop`")
        prmtop = Path(region.prmtop).read_text() if Path(region.prmtop).exists() else region.prmtop
        if region.tcin:
            tcin = Path(region.tcin).read_text() if Path(region.tcin).exists() else region.tcin
        else:
            model = dict(getattr(program_args, "model", None) or {})
            tcin = f"method {model.get('method', 'b3lyp')}\nbasis {model.get('basis', '6-31gs')}\n"
            for k, v in dict(getattr(program_args, "keywords", None) or {}).items():
                tcin += f"{k} {v}\n"
        tcin = with_keyword(with_keyword(tcin, "charge", region.qm_charge), "spinmult", region.qm_multiplicity)
        return cls(tcin=tcin, prmtop=prmtop, qm_atoms=list(region.qm_atoms), compute_program=compute_program,
                   chemcloud_queue=chemcloud_queue)

    def _inherit(self) -> None:
        return None

    def _input(self, coords_bohr):
        from qcdata.models.inputs import FileInput

        return FileInput(program="terachem", files={"tc.in": self.tcin, "ref.rst7": structure_to_rst7(coords_bohr),
                                                     "qmindices.dat": self.qmindices, "ref.prmtop": self.prmtop},
                         cmdline_args=["tc.in"])

    def _submit(self, inputs: list):
        if self.compute_program == "qccompute":
            from qccompute import compute

            return [compute(inp) for inp in inputs]
        from chemcloud import compute

        delay = 2.0
        for attempt in range(3):
            try:
                out = compute("terachem", inputs if len(inputs) > 1 else inputs[0], queue=self.chemcloud_queue)
                out = out.get() if hasattr(out, "get") else out
                return out if isinstance(out, list) else [out]
            except Exception as exc:
                if attempt == 2:
                    raise
                logger.warning("TeraChem QM/MM ChemCloud call failed (%s); retrying in %.0fs", exc, delay)
                time.sleep(delay)
                delay *= 2

    def energy_gradient_from_stdout(self, stdout: str, natoms: int) -> tuple[float, np.ndarray]:
        qm, mm = parse_qmmm_gradients(stdout)
        if len(qm) == 0:
            raise ElectronicStructureError(msg="TeraChem QM/MM: no QM gradient in the output")
        nlink = n_link_atoms(stdout)
        if nlink:
            qm = qm[:-nlink]
        grad = np.zeros((natoms, 3))
        grad[self.qm_atoms] = qm
        mm_idx = [i for i in range(natoms) if i not in set(self.qm_atoms)]
        if mm_idx:
            grad[mm_idx] = mm
        return parse_final_energy(stdout), grad

    def _run(self, chain: Union[Chain, List]):
        nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [n for n in nodes if n._cached_energy is None or n._cached_gradient is None]
        if todo:
            try:
                outs = self._submit([self._input(n.coords) for n in todo])
            except Exception as exc:
                raise ElectronicStructureError(msg=f"TeraChem QM/MM submission failed: {exc}") from exc
            results = []
            for n, out in zip(todo, outs):
                e, g = self.energy_gradient_from_stdout(getattr(out, "stdout", "") or "", len(n.symbols))
                results.append(FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                    {"energy": e, "gradient": g})}))
            update_node_cache(node_list=todo, results=results)
        return nodes

    def _ase(self, node):
        from mepd.engines.ase import ASEEngine
        from mepd.engines.gxtb import _GXTBASEResultsCalculator

        return ASEEngine(calculator=_GXTBASEResultsCalculator(self, charge=int(node.structure.charge),
                                                              multiplicity=int(node.structure.multiplicity)))
