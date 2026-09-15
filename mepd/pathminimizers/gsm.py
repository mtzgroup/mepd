from __future__ import annotations

import os
import pickle
import shutil
import socket
import subprocess
import sys
import tempfile
import time
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

# `./grad.py <endstr> <ncpu> <charge>` is what the compiled `gsm` binary
# (built with -DGSM_ENABLE_ASE=1) actually shells out to, once per node
# gradient/energy request -- hundreds of times per run. Naively unpickling
# the engine and importing mepd fresh in that script, every single call,
# was measured at ~1.9s/call (dominated by `import mepd...`'s transitive
# pydantic/qcdata/rdkit/ASE cost -- the real gxtb call itself is ~50-90ms),
# which is why a GSM run's wall-clock time didn't track its (much lower)
# gradient-call count vs. NEB. So `grad.py` is instead a near-zero-import
# client (stdlib `socket`+`sys` only) that forwards the node id to a
# long-lived _ENGINE_SERVER_SOURCE process (started once per
# `optimize_chain()` call, see `_start_engine_server`) over a Unix domain
# socket. The server pays the mepd import + engine unpickle cost exactly
# once, then answers every subsequent request for the life of the run.
_GRAD_PY_SOURCE = '''#!/usr/bin/env python3
import socket
import sys

SOCKET_PATH = "engine.sock"


def main():
    endstr = sys.argv[1]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)
    try:
        sock.sendall(endstr.encode())
        response = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response += chunk
    finally:
        sock.close()

    if not response.startswith(b"OK"):
        sys.stderr.write(response.decode(errors="replace"))
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

# Long-lived server `grad.py` clients above talk to. Started once per
# `optimize_chain()` call (see `_start_engine_server`), killed once GSM's
# subprocess exits. Unlike `grad.py`, this one does pay the real import/
# unpickle cost -- but only once for however many hundreds of node
# requests the run makes, instead of once per request.
_ENGINE_SERVER_SOURCE = '''#!/usr/bin/env python3
import os
import pickle
import socket
import sys
import traceback

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

HARTREE_TO_EV = 27.2114
KCAL_PER_HARTREE = 627.5
SOCKET_PATH = "engine.sock"
READY_MARKER = SOCKET_PATH + ".ready"
LIVE_PATH = "live_string.xyz0000"

# GSM's own optimization loop runs entirely inside the compiled binary, so
# per-node energy/gradient requests reaching this server are the only
# visibility mepd has into its progress. known_nodes remembers the most
# recently seen structure for every node index GSM has asked about (indices
# come from the ".NN" suffix of endstr, matching GSM's own node numbering),
# and gets rewritten to LIVE_PATH after every request in the exact block
# format _parse_stringfile already reads (see GSM._parse_string_blocks) --
# so the parent process can poll it and render a live, progressively
# updating chain profile the same way it already does for NEB
# (mepd.progress.print_chain_step).
known_nodes = {}


def _write_live_snapshot():
    lines = []
    for idx in sorted(known_nodes):
        symbols, coords, energy, e_reactant = known_nodes[idx]
        v_kcal = (energy - e_reactant) * KCAL_PER_HARTREE
        lines.append(" {}".format(len(symbols)))
        lines.append(" {:.10f}".format(v_kcal))
        for sym, (x, y, z) in zip(symbols, coords):
            lines.append("  {} {:.10f} {:.10f} {:.10f}".format(sym, x, y, z))
    tmp_path = LIVE_PATH + ".tmp"
    with open(tmp_path, "w") as fh:
        fh.write("\\n".join(lines) + "\\n")
    os.replace(tmp_path, LIVE_PATH)  # atomic, so a concurrent poll never sees a partial write


def handle(endstr, engine, template_node, e_reactant):
    with open("scratch/structure" + endstr) as fh:
        lines = fh.read().splitlines()
    natoms = int(lines[0].strip())
    coords = []
    for row in lines[2:2 + natoms]:
        fields = row.split()
        coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
    coords_bohr = np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR
    node = template_node.update_coords(coords_bohr)

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

    try:
        node_index = int(endstr.rsplit(".", 1)[-1])
    except ValueError:
        node_index = None
    if node_index is not None:
        known_nodes[node_index] = (list(template_node.symbols), coords, energy, e_reactant)
        _write_live_snapshot()


def main():
    with open("engine.pkl", "rb") as fh:
        payload = pickle.load(fh)
    engine = payload["engine"]
    template_node = payload["template_node"]
    e_reactant = payload["e_reactant"]

    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(8)
    # Written only after bind()+listen() succeed, so the parent process
    # polling for this file never races a socket that isn't accept()-able yet.
    with open(READY_MARKER, "w") as fh:
        fh.write("1")

    try:
        while True:
            conn, _ = server.accept()
            try:
                data = conn.recv(4096).decode().strip()
                if data == "__SHUTDOWN__":
                    conn.sendall(b"OK")
                    break
                try:
                    handle(data, engine, template_node, e_reactant)
                    conn.sendall(b"OK")
                except Exception:
                    conn.sendall(("ERROR\\n" + traceback.format_exc()).encode())
            finally:
                conn.close()
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)


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
        server_proc = None
        try:
            self._write_inputs(workdir, reactant, product, e_reactant)
            counter_fp = workdir / "grad_calls.count"
            counter_fp.write_text("")

            server_proc = self._start_engine_server(workdir)
            self._log("Running molecularGSM (DE-GSM)...")
            self._run_gsm(workdir, reactant, e_reactant, chain.parameters)

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
                # Pass the FULL converged chain, not a 3-node
                # [reactant, TS-guess, product] reduction -- check_if_elem_step's
                # minima-based concavity check (_get_ind_minima) scans the chain
                # it's given for local dips, and a 3-node chain (one interior
                # point, by construction the *highest*-energy node) can never
                # represent an "up-down-up" profile, so a real intermediate
                # visible in the full GSM path would be structurally invisible
                # to it. This matches how the full NEB class does it
                # (mepd/neb.py's check_if_elem_step calls all pass `final_chain`
                # directly) rather than FreezingNEB's 3-node reduction.
                elem_step_results = check_if_elem_step(
                    final_chain,
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
            self._stop_engine_server(server_proc, workdir)
            if not getattr(self.parameters, "keep_workdirs", False):
                shutil.rmtree(workdir, ignore_errors=True)
            else:
                self._log(f"Kept GSM workdir at {workdir}", level="warning")

        return elem_step_results

    def _write_inputs(
        self, workdir: Path, reactant, product, e_reactant: float
    ) -> None:
        scratch = workdir / "scratch"
        scratch.mkdir()
        (scratch / "initial0000.xyz").write_text(
            reactant.structure.to_xyz() + product.structure.to_xyz()
        )
        (workdir / "inpfileq").write_text(_inpfileq_text(self.parameters))

        with open(workdir / "engine.pkl", "wb") as fh:
            pickle.dump(
                {
                    "engine": self.engine,
                    "template_node": reactant,
                    "e_reactant": e_reactant,
                },
                fh,
            )

        grad_py = workdir / "grad.py"
        grad_py.write_text(_GRAD_PY_SOURCE)
        grad_py.chmod(0o755)

        engine_server_py = workdir / "engine_server.py"
        engine_server_py.write_text(_ENGINE_SERVER_SOURCE)
        engine_server_py.chmod(0o755)

    def _start_engine_server(self, workdir: Path) -> subprocess.Popen:
        """Launch the long-lived engine server `grad.py` clients talk to
        (see the module-level comment on `_ENGINE_SERVER_SOURCE`), and block
        until its socket is actually ready to accept connections.
        """
        ready_marker = workdir / "engine.sock.ready"
        proc = subprocess.Popen(
            [sys.executable, "engine_server.py"],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        timeout = 30.0
        poll_interval = 0.01
        waited = 0.0
        while not ready_marker.exists():
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise ElectronicStructureError(
                    msg="GSM engine server exited before it was ready.",
                    obj=out,
                )
            if waited >= timeout:
                proc.kill()
                raise ElectronicStructureError(
                    msg=f"GSM engine server did not become ready within {timeout}s."
                )
            time.sleep(poll_interval)
            waited += poll_interval

        return proc

    def _stop_engine_server(
        self, proc: subprocess.Popen | None, workdir: Path
    ) -> None:
        if proc is None:
            return
        if proc.poll() is not None:
            return
        socket_path = workdir / "engine.sock"
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(str(socket_path))
            sock.sendall(b"__SHUTDOWN__")
            sock.recv(64)
            sock.close()
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()
            proc.wait(timeout=5.0)

    def _run_gsm(
        self, workdir: Path, reactant, e_reactant: float, chain_parameters
    ) -> None:
        executable = self._resolve_executable()
        # `./grad.py` is invoked by GSM via a bare `system("./grad.py ...")`
        # relying on its shebang + $PATH; grad.py itself only needs stdlib
        # (see _GRAD_PY_SOURCE), so any python3 on PATH works, but keep this
        # for robustness in minimal environments.
        env = os.environ.copy()
        venv_bin = str(Path(sys.executable).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        timeout = getattr(self.parameters, "timeout", None)
        live_path = workdir / "live_string.xyz0000"
        poll_interval = 0.3

        try:
            proc = subprocess.Popen(
                [executable, "0", "1"],
                cwd=workdir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise ElectronicStructureError(
                msg=(
                    f"GSM executable `{executable}` was not found. Build "
                    "molecularGSM with -DGSM_ENABLE_ASE=1 and set GSM_EXECUTABLE "
                    "or path_min_inputs.executable to the resulting `gsm` binary."
                )
            ) from exc

        # GSM's own optimization loop is entirely inside the compiled binary,
        # so polling the live-snapshot file the engine server rewrites after
        # every node request (see _ENGINE_SERVER_SOURCE) is the only way to
        # show progress -- there's nothing to await synchronously otherwise.
        start_time = time.time()
        last_mtime = None
        while proc.poll() is None:
            if timeout is not None and (time.time() - start_time) > timeout:
                proc.kill()
                proc.wait()
                raise ElectronicStructureError(
                    msg=f"GSM calculation timed out after {timeout}s."
                )
            try:
                mtime = live_path.stat().st_mtime
            except OSError:
                mtime = None
            if mtime is not None and mtime != last_mtime:
                last_mtime = mtime
                self._print_live_chain(live_path, reactant, e_reactant, chain_parameters)
            time.sleep(poll_interval)

        # One last refresh in case a final write raced the process exit.
        self._print_live_chain(live_path, reactant, e_reactant, chain_parameters)

        stdout = proc.stdout.read() if proc.stdout else ""
        if proc.returncode != 0 or not (workdir / "stringfile.xyz0000").exists():
            raise ElectronicStructureError(
                msg=f"GSM calculation failed with exit code {proc.returncode}.",
                obj=stdout,
            )

    def _print_live_chain(
        self, live_path: Path, reactant, e_reactant: float, chain_parameters
    ) -> None:
        """Render whatever of the string the engine server has reported so
        far as a live, in-place-updating ASCII profile -- the GSM analogue of
        the live chain view NEB already gets via mepd.progress.print_chain_step
        after every optimizer step. Best-effort: a snapshot file that's
        missing, empty, or (despite the atomic rename in
        _ENGINE_SERVER_SOURCE) caught mid-write just skips this refresh
        rather than raising out of a background poll loop.
        """
        try:
            text = live_path.read_text()
        except OSError:
            return
        if not text.strip():
            return
        try:
            blocks = self._parse_string_blocks(text, len(reactant.symbols))
        except Exception:
            return
        if not blocks:
            return

        from mepd.progress import print_chain_step
        from qcconst.constants import ANGSTROM_TO_BOHR

        nodes = []
        for v_kcal, coords in blocks:
            coords_bohr = np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR
            node = reactant.update_coords(coords_bohr)
            node._cached_energy = e_reactant + v_kcal / _KCAL_PER_HARTREE
            nodes.append(node)
        live_chain = Chain.model_validate(
            {"nodes": nodes, "parameters": chain_parameters}
        )
        nnodes_target = int(getattr(self.parameters, "nnodes", 9))
        print_chain_step(
            live_chain,
            caption=f"GSM live string ({len(nodes)}/{nnodes_target} nodes seen)",
        )

    @staticmethod
    def _parse_string_blocks(
        text: str, natoms_expected: int
    ) -> list[tuple[float, list]]:
        """Parse the block format shared by GSM's own `stringfile.xyz<run>`
        output and our `live_string.xyz<run>` snapshot (see
        _ENGINE_SERVER_SOURCE): repeated blocks of `<natoms>` /
        `<energy, kcal/mol>` / `<natoms> "SYMBOL x y z"` lines (Angstrom), no
        blank lines between blocks. Returns (energy_kcal, coords_angstrom)
        per block, in file order.
        """
        lines = text.splitlines()
        blocks: list[tuple[float, list]] = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            natoms = int(stripped)
            assert natoms == natoms_expected, (
                f"GSM string block has {natoms} atoms, expected {natoms_expected}"
            )
            v_kcal = float(lines[i + 1].strip())
            coords = []
            for row in lines[i + 2 : i + 2 + natoms]:
                fields = row.split()
                coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
            blocks.append((v_kcal, coords))
            i += 2 + natoms
        return blocks

    def _parse_stringfile(
        self, workdir: Path, reactant, e_reactant: float, chain_parameters
    ) -> Chain:
        from qcconst.constants import ANGSTROM_TO_BOHR

        text = (workdir / "stringfile.xyz0000").read_text()
        blocks = self._parse_string_blocks(text, len(reactant.symbols))

        nodes = []
        for v_kcal, coords in blocks:
            coords_bohr = np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR
            node = reactant.update_coords(coords_bohr)
            node._cached_energy = e_reactant + v_kcal / _KCAL_PER_HARTREE
            nodes.append(node)

        return Chain.model_validate({"nodes": nodes, "parameters": chain_parameters})
