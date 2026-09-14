from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mepd.chain import Chain
from mepd.elementarystep import ElemStepResults, check_if_elem_step
from mepd.engines.engine import Engine
from mepd.errors import ElectronicStructureError
from mepd.inputs import RunInputs
from mepd.pathminimizers.pathminimizer import PathMinimizer
from mepd.progress import get_progress_printer

# Same magic numbers the C++ ASE driver (GSM/ase.cpp) uses to convert between
# its internal eV/Angstrom-ish bookkeeping and Hartree -- match them exactly
# (rather than a more precise CODATA constant) so our writer/its reader round-trip.
_HARTREE_TO_EV = 27.2114
_KCAL_PER_HARTREE = 627.5

IS_ELEM_STEP = ElemStepResults(
    is_elem_step=True,
    is_concave=True,
    splitting_criterion=None,
    minimization_results=None,
    number_grad_calls=0,
)

# Standalone driver script invoked by the compiled `gsm` binary (built with
# -DGSM_ENABLE_ASE=1) as `./grad.py <endstr> <ncpu> <charge>` once per node
# gradient/energy request. It has no template placeholders: the engine and a
# reference node (for charge/multiplicity/symbols) are unpickled from
# `engine.pkl`, written alongside it in the run's working directory.
_GRAD_PY_SOURCE = '''#!/usr/bin/env python3
import pickle
import sys

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

HARTREE_TO_EV = 27.2114


def main():
    endstr = sys.argv[1]
    with open("scratch/structure" + endstr) as fh:
        lines = fh.read().splitlines()
    natoms = int(lines[0].strip())
    coords = []
    for row in lines[2:2 + natoms]:
        fields = row.split()
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
    coords_bohr = np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR

    with open("engine.pkl", "rb") as fh:
        payload = pickle.load(fh)
    node = payload["template_node"].update_coords(coords_bohr)
    engine = payload["engine"]

    energy = float(engine.compute_energies([node])[0])
    gradient = np.asarray(engine.compute_gradients([node])[0], dtype=float)
    # Hartree/Bohr -> Hartree/Angstrom, then to the eV-scaled units GSM's
    # reader divides back out (see GSM/ase.cpp: get_energy_grad).
    gradient_scaled = gradient * ANGSTROM_TO_BOHR * HARTREE_TO_EV

    with open("scratch/GRAD" + endstr, "w") as fh:
        fh.write(str(-energy * HARTREE_TO_EV))
        fh.write("\\n")
        for row in gradient_scaled:
            fh.write("{} {} {}\\n".format(row[0], row[1], row[2]))

    with open("grad_calls.count", "a") as fh:
        fh.write("1\\n")


if __name__ == "__main__":
    main()
'''


def _inpfileq_text(parameters: SimpleNamespace) -> str:
    p = parameters
    return f"""
# FSM/GSM/SSM inpfileq

------------- QCHEM Scratch Info ------------------------
$QCSCRATCH/    # path for scratch dir. end with "/"
GSM_go1q       # name of run
---------------------------------------------------------

------------ String Info --------------------------------
SM_TYPE                 GSM    # SSM, FSM or GSM
RESTART                 0      # read restart.xyz
MAX_OPT_ITERS           {int(getattr(p, "max_opt_iters", 80))}     # maximum iterations
STEP_OPT_ITERS          {int(getattr(p, "step_opt_iters", 30))}     # for FSM/SSM
CONV_TOL                {float(getattr(p, "conv_tol", 0.0005))} # perp grad
ADD_NODE_TOL            {float(getattr(p, "add_node_tol", 0.1))}    # for GSM
SCALING                 {float(getattr(p, "scaling", 1.0))}    # for opt steps
SSM_DQMAX               {float(getattr(p, "ssm_dqmax", 0.8))}    # add step size
GROWTH_DIRECTION        0      # normal/react/prod: 0/1/2
INT_THRESH              {float(getattr(p, "int_thresh", 2.0))}    # intermediate detection
MIN_SPACING             {float(getattr(p, "min_spacing", 5.0))}    # node spacing SSM
BOND_FRAGMENTS          {int(getattr(p, "bond_fragments", 1))}      # make IC's for fragments
INITIAL_OPT             {int(getattr(p, "initial_opt", 0))}      # opt steps first node
FINAL_OPT               {int(getattr(p, "final_opt", 150))}    # opt steps last SSM node
PRODUCT_LIMIT           {float(getattr(p, "product_limit", 100.0))}  # kcal/mol
TS_FINAL_TYPE           {int(getattr(p, "ts_final_type", 1))}      # any/delta bond: 0/1
NNODES                  {int(getattr(p, "nnodes", 9))}      # including endpoints
---------------------------------------------------------
"""


@dataclass
class GSM(PathMinimizer):
    """Double-ended path minimizer backed by the Zimmerman lab's compiled
    `molecularGSM` C++ growing string method (github.com/ZimmermanGroup/molecularGSM),
    not the `pyGSM` Python port.

    The upstream binary only talks to a handful of hardcoded QM packages by
    reading/writing files in a scratch directory. Built with
    `-DGSM_ENABLE_ASE=1`, it instead shells out to a `./grad.py <node-id> <nproc>
    <charge>` script once per node energy/gradient request. We generate that
    script per run to unpickle *this* `engine` and call its
    `compute_energies`/`compute_gradients` directly, so GSM is driven by the
    exact same engine (e.g. gxtb) as every other path minimizer in mepd.
    """

    initial_chain: Chain
    engine: Engine
    parameters: SimpleNamespace = None
    chain_trajectory: list = field(default_factory=list)

    def __post_init__(self):
        if self.parameters is None:
            ri = RunInputs(path_min_method="GSM")
            self.parameters = ri.path_min_inputs
        self.grad_calls_made = 0
        self.geom_grad_calls_made = 0

    def _log(self, *parts, level: str = "info", verbose: int = 1):
        if getattr(self.parameters, "verbosity", 1) < verbose:
            return
        message = " ".join(str(p) for p in parts)
        printer = get_progress_printer()
        if level == "warning":
            printer.print_warning(message)
        elif level == "error":
            printer.print_error(message)
        elif level == "success":
            printer.print_convergence(message)
        else:
            printer.update_status(message)

    def _resolve_executable(self) -> str:
        executable = getattr(self.parameters, "executable", None)
        return str(executable or os.getenv("GSM_EXECUTABLE") or "gsm")

    def optimize_chain(self) -> ElemStepResults:
        chain = self.initial_chain.copy()
        reactant = chain.nodes[0]
        product = chain.nodes[-1]

        # Cost of evaluating the two endpoints, mirrors FreezingNEB.optimize_chain.
        self.engine.compute_energies([reactant, product])
        self.grad_calls_made += 2
        e_reactant = float(reactant.energy)

        workdir = Path(tempfile.mkdtemp(prefix="gsm-"))
        try:
            self._write_inputs(workdir, reactant, product)
            counter_fp = workdir / "grad_calls.count"
            counter_fp.write_text("")

            self._log("Running molecularGSM (DE-GSM)...")
            self._run_gsm(workdir)

            n_calls = counter_fp.read_text().count("\n")
            self.grad_calls_made += n_calls
            self._log(f"molecularGSM made {n_calls} gradient/energy calls", verbose=2)

            final_chain = self._parse_stringfile(
                workdir, reactant, e_reactant, chain.parameters
            )
            self.chain_trajectory = [self.initial_chain.copy(), final_chain]
            # Every other PathMinimizer sets this after a successful optimize_chain()
            # -- msmep.run_minimize_chain reads it unconditionally right afterward.
            self.optimized = final_chain

            if getattr(self.parameters, "do_elem_step_checks", True):
                short_chain = Chain.model_validate(
                    {
                        "nodes": [final_chain[0], final_chain.get_ts_node(), final_chain[-1]],
                        "parameters": final_chain.parameters,
                    }
                )
                elem_step_results = check_if_elem_step(
                    short_chain,
                    engine=self.engine,
                    validate_minima_with_hessian=bool(
                        getattr(self.parameters, "validate_minima_with_hessian", False)
                    ),
                    hessian_minimum_frequency_cutoff=float(
                        getattr(self.parameters, "hessian_minimum_frequency_cutoff", 0.0)
                    ),
                    hessian_minima_rescue_displacement=float(
                        getattr(self.parameters, "hessian_minima_rescue_displacement", 0.1)
                    ),
                    disregard_stereochem=bool(
                        getattr(self.parameters, "disregard_stereochem", False)
                    ),
                )
                self.geom_grad_calls_made += elem_step_results.number_grad_calls
            else:
                elem_step_results = IS_ELEM_STEP
        finally:
            if not getattr(self.parameters, "keep_workdirs", False):
                shutil.rmtree(workdir, ignore_errors=True)
            else:
                self._log(f"Kept GSM workdir at {workdir}", level="warning")

        return elem_step_results

    def _write_inputs(self, workdir: Path, reactant, product) -> None:
        scratch = workdir / "scratch"
        scratch.mkdir()
        (scratch / "initial0000.xyz").write_text(
            reactant.structure.to_xyz() + product.structure.to_xyz()
        )
        (workdir / "inpfileq").write_text(_inpfileq_text(self.parameters))

        with open(workdir / "engine.pkl", "wb") as fh:
            pickle.dump({"engine": self.engine, "template_node": reactant}, fh)

        grad_py = workdir / "grad.py"
        grad_py.write_text(_GRAD_PY_SOURCE)
        grad_py.chmod(0o755)

    def _run_gsm(self, workdir: Path) -> None:
        executable = self._resolve_executable()
        # `./grad.py` is invoked by GSM via a bare `system("./grad.py ...")`
        # relying on its shebang + $PATH; make sure the interpreter that has
        # `mepd` importable (this process's own) is the one found first.
        env = os.environ.copy()
        venv_bin = str(Path(sys.executable).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        timeout = getattr(self.parameters, "timeout", None)
        try:
            completed = subprocess.run(
                [executable, "0", "1"],
                cwd=workdir,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ElectronicStructureError(
                msg=(
                    f"GSM executable `{executable}` was not found. Build "
                    "molecularGSM with -DGSM_ENABLE_ASE=1 and set GSM_EXECUTABLE "
                    "or path_min_inputs.executable to the resulting `gsm` binary."
                )
            ) from exc

        if completed.returncode != 0 or not (workdir / "stringfile.xyz0000").exists():
            raise ElectronicStructureError(
                msg=f"GSM calculation failed with exit code {completed.returncode}.",
                obj=completed.stdout + completed.stderr,
            )

    def _parse_stringfile(
        self, workdir: Path, reactant, e_reactant: float, chain_parameters
    ) -> Chain:
        text = (workdir / "stringfile.xyz0000").read_text()
        lines = text.splitlines()
        natoms_expected = len(reactant.symbols)

        nodes = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            natoms = int(stripped)
            assert natoms == natoms_expected, (
                f"GSM stringfile frame has {natoms} atoms, expected {natoms_expected}"
            )
            v_kcal = float(lines[i + 1].strip())
            coords = []
            for row in lines[i + 2 : i + 2 + natoms]:
                fields = row.split()
                coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
            from qcconst.constants import ANGSTROM_TO_BOHR

            coords_bohr = np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR
            node = reactant.update_coords(coords_bohr)
            node._cached_energy = e_reactant + v_kcal / _KCAL_PER_HARTREE
            nodes.append(node)
            i += 2 + natoms

        return Chain.model_validate({"nodes": nodes, "parameters": chain_parameters})
