from __future__ import annotations

import os
import pickle
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
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


def _inpfileq_text(
    parameters: SimpleNamespace,
    *,
    restart: int = 0,
    nnodes_override: int | None = None,
) -> str:
    p = parameters
    nnodes = (
        int(nnodes_override)
        if nnodes_override is not None
        else int(getattr(p, "nnodes", 9))
    )
    return f"""
# FSM/GSM/SSM inpfileq

------------- QCHEM Scratch Info ------------------------
$QCSCRATCH/    # path for scratch dir. end with "/"
GSM_go1q       # name of run
---------------------------------------------------------

------------ String Info --------------------------------
SM_TYPE                 GSM    # SSM, FSM or GSM
RESTART                 {int(restart)}      # read restart.xyz
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
NNODES                  {nnodes}      # including endpoints
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

    By default GSM grows its own path from scratch using its internal-
    coordinate scheme (`GSM/icoord.cpp`) -- it needs no starting guess for
    the interior nodes at all. Setting
    `path_min_inputs.seed_with_geodesic_interpolation = True` instead seeds
    the search with a geodesic-interpolated path (`mepd.chainhelpers.run_geodesic`,
    the same helper NEB/FreezingNEB use) between the same two endpoints, via
    the compiled binary's native `RESTART` mechanism -- see
    `_build_geodesic_seed`'s docstring for how that's wired up.
    """

    initial_chain: Chain
    engine: Engine
    parameters: SimpleNamespace = None
    chain_trajectory: list = field(default_factory=list)
    gi_inputs: SimpleNamespace = None

    def __post_init__(self):
        if self.parameters is None:
            ri = RunInputs(path_min_method="GSM")
            self.parameters = ri.path_min_inputs
        if self.gi_inputs is None:
            ri = RunInputs(path_min_method="GSM")
            self.gi_inputs = ri.gi_inputs
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

        seed_nodes = None
        if bool(getattr(self.parameters, "seed_with_geodesic_interpolation", False)):
            seed_nodes = self._build_geodesic_seed(chain, reactant, product)

        do_elem_step_checks = bool(getattr(self.parameters, "do_elem_step_checks", True))
        # GSM's DE-GSM mode has no native "stop because an intermediate is
        # present" hook to lean on (see _run_gsm's docstring), so this can
        # only ever be as trustworthy as the check that verifies it --
        # never arm it without do_elem_step_checks also on.
        allow_early_stop = do_elem_step_checks and bool(
            getattr(self.parameters, "early_stop_on_minima", True)
        )

        final_chain, history, early_stopped = self._execute_gsm_attempt(
            chain, reactant, product, e_reactant, seed_nodes, allow_early_stop
        )
        self.chain_trajectory = [self.initial_chain.copy(), *history, final_chain]
        # Every other PathMinimizer sets this after a successful optimize_chain()
        # -- msmep.run_minimize_chain reads it unconditionally right afterward.
        self.optimized = final_chain

        elem_step_results = (
            self._check_elem_step(final_chain) if do_elem_step_checks else IS_ELEM_STEP
        )

        if early_stopped and elem_step_results.is_elem_step:
            # The persistent-minimum signal that triggered the early stop
            # was a false alarm per our own (more thorough) check -- rather
            # than settle for a chain that was never let converge, resume:
            # a fresh GSM invocation seeded (via RESTART -- see
            # _build_geodesic_seed's docstring for how that mechanism
            # works) from exactly the point we stopped at, this time run to
            # full natural completion (early_allow_stop=False, so this
            # can't recurse).
            self._log(
                "Early stop looked like a real intermediate but "
                "check_if_elem_step disagreed; resuming to full "
                "convergence from the same point.",
                level="warning",
            )
            final_chain, resume_history, _ = self._execute_gsm_attempt(
                chain, reactant, product, e_reactant, final_chain.nodes, False
            )
            self.chain_trajectory.extend(resume_history)
            self.chain_trajectory.append(final_chain)
            self.optimized = final_chain
            elem_step_results = (
                self._check_elem_step(final_chain)
                if do_elem_step_checks
                else IS_ELEM_STEP
            )

        return elem_step_results

    def _execute_gsm_attempt(
        self, chain, reactant, product, e_reactant, seed_nodes, allow_early_stop
    ) -> tuple[Chain, list, bool]:
        """One full GSM invocation: write inputs, start the engine server,
        run the binary (optionally monitoring for a persistent minimum to
        stop early on), tear down. Returns `(final_chain, history,
        early_stopped)`. Used both for the normal single-attempt path and,
        via `optimize_chain`'s resume logic, a second time after an early
        stop check_if_elem_step didn't confirm.
        """
        workdir = Path(tempfile.mkdtemp(prefix="gsm-"))
        server_proc = None
        try:
            self._write_inputs(
                workdir, reactant, product, e_reactant, seed_nodes=seed_nodes
            )
            counter_fp = workdir / "grad_calls.count"
            counter_fp.write_text("")

            server_proc = self._start_engine_server(workdir)
            self._log("Running molecularGSM (DE-GSM)...")
            history, early_stopped = self._run_gsm(
                workdir,
                reactant,
                product,
                e_reactant,
                chain.parameters,
                allow_early_stop=allow_early_stop,
            )

            n_calls = counter_fp.read_text().count("\n")
            self.grad_calls_made += n_calls
            self._log(f"molecularGSM made {n_calls} gradient/energy calls", verbose=2)

            if early_stopped:
                # A killed process never writes stringfile.xyz0000 -- the
                # last live snapshot (already has pinned, correct
                # reactant/product endpoints; see _build_live_chain) is the
                # best final chain there is until/unless optimize_chain's
                # resume logic decides to try again.
                final_chain = history[-1]
            else:
                final_chain = self._parse_stringfile(
                    workdir, reactant, e_reactant, chain.parameters
                )
            return final_chain, history, early_stopped
        finally:
            self._stop_engine_server(server_proc, workdir)
            if not getattr(self.parameters, "keep_workdirs", False):
                shutil.rmtree(workdir, ignore_errors=True)
            else:
                self._log(f"Kept GSM workdir at {workdir}", level="warning")

    def _check_elem_step(self, final_chain: Chain) -> ElemStepResults:
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
        return elem_step_results

    def _build_geodesic_seed(self, chain: Chain, reactant, product) -> list:
        """Build a geodesic-interpolated path between `reactant` and `product`
        and evaluate real energies on every node via `self.engine`, so this
        path can seed molecularGSM's search via its native `RESTART`
        mechanism instead of letting it grow its own path from scratch.

        Why this works -- `RESTART` in the C++ source (all in GSM/gstring.cpp):
        when `inpfileq`'s `RESTART` tag (parsed around line 1232) is nonzero,
        `String_Method_Optimization` (~line 562-587) skips `starting_string`/
        `growth_iters` entirely -- the from-scratch internal-coordinate growth
        this class normally relies on -- and instead calls
        `restart_string("restart.xyz0000")` (~line 6299), which reads that
        file via `read_string` (~line 6222) in the exact same block format
        `_parse_string_blocks` reads (and `_write_string_blocks` writes):
        `<natoms>` / `<energy kcal/mol>` / `<natoms> "SYMBOL x y z"` lines,
        Angstrom, no blank lines between blocks -- the same format
        `print_string` writes as `stringfile.xyz0000` output. `restart_string`
        then sets `nn = nnR = nnmax = nrnodes` (the frame count actually
        found in the file) and marks every interior node `active`, so GSM
        treats the restart file as an already-fully-grown string and jumps
        straight into its optimization/TS-search phase. We use `RESTART=1`
        (not 2): the restart file's own endpoint frames *are*
        `reactant`/`product`, so there's nothing to preserve separately
        (unlike `RESTART=2`'s use case of overriding a restart file with
        different endpoints than `initial0000.xyz`).
        """
        import mepd.chainhelpers as ch

        seed_chain = Chain.model_validate(
            {"nodes": [reactant, product], "parameters": chain.parameters}
        )
        gi = self.gi_inputs
        interpolated = ch.run_geodesic(
            chain=seed_chain,
            chain_inputs=chain.parameters,
            nimages=gi.nimages,
            friction=gi.friction,
            nudge=gi.nudge,
            random_seed=gi.random_seed,
            align=gi.align,
            **(gi.extra_kwds or {}),
        )
        # run_geodesic rebuilds every node fresh (including the endpoints)
        # from interpolated coordinates, so it has no idea reactant/product
        # were already evaluated moments ago in optimize_chain -- swap the
        # already-cached originals back in so compute_energies below only
        # pays for the genuinely new interior nodes, not two redundant calls
        # at coordinates it already has the answer for.
        interpolated.nodes[0] = reactant
        interpolated.nodes[-1] = product

        n_before = sum(1 for n in interpolated.nodes if n._cached_energy is None)
        self.engine.compute_energies(interpolated.nodes)
        self.grad_calls_made += n_before
        self._log(
            f"Seeding molecularGSM with a {len(interpolated.nodes)}-node "
            "geodesic-interpolated path (RESTART mode)."
        )
        return list(interpolated.nodes)

    @staticmethod
    def _write_string_blocks(fp: Path, nodes: list, e_reference: float) -> None:
        """Inverse of `_parse_string_blocks`: write `nodes` (each needing a
        real `.energy` and `.coords`) to `fp` in the same block format GSM's
        own `stringfile.xyz<run>`/`restart.xyz<run>` files use.
        """
        from qcconst.constants import ANGSTROM_TO_BOHR

        lines = []
        for node in nodes:
            symbols = list(node.symbols)
            coords_angstrom = np.asarray(node.coords, dtype=float) / ANGSTROM_TO_BOHR
            v_kcal = (float(node.energy) - e_reference) * _KCAL_PER_HARTREE
            lines.append(f" {len(symbols)}")
            lines.append(f" {v_kcal:.10f}")
            for symbol, (x, y, z) in zip(symbols, coords_angstrom):
                lines.append(f"  {symbol} {x:.10f} {y:.10f} {z:.10f}")
        fp.write_text("\n".join(lines) + "\n")

    def _write_inputs(
        self,
        workdir: Path,
        reactant,
        product,
        e_reactant: float,
        seed_nodes: list | None = None,
    ) -> None:
        scratch = workdir / "scratch"
        scratch.mkdir()
        (scratch / "initial0000.xyz").write_text(
            reactant.structure.to_xyz() + product.structure.to_xyz()
        )

        if seed_nodes:
            # Written at the workdir *root* (not scratch/) -- matches the
            # `restart.xyz` + 4-digit-run-number convention the binary uses
            # for both this input file and its `stringfile.xyz0000` output.
            self._write_string_blocks(
                workdir / "restart.xyz0000", seed_nodes, e_reactant
            )
            inpfileq_text = _inpfileq_text(
                self.parameters, restart=1, nnodes_override=len(seed_nodes)
            )
        else:
            inpfileq_text = _inpfileq_text(self.parameters)
        (workdir / "inpfileq").write_text(inpfileq_text)

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

    @staticmethod
    def _start_stdout_drain(proc: subprocess.Popen) -> list[str]:
        """Continuously read `proc.stdout` in a background thread for the
        life of the process, returning the (still being appended to) list of
        captured lines.

        Necessary because a Popen with `stdout=PIPE` deadlocks once the OS
        pipe buffer (~64KB) fills if nothing ever reads it: the child blocks
        on its own `write()` call and never gets to exit, so a parent that
        only checks `proc.poll()` waits forever while the child waits
        forever too. Both the `gsm` binary itself (over the course of a full
        run) and `engine_server.py` (if e.g. a warnings-module message fires
        on enough individual node requests) can produce enough output to hit
        this -- confirmed as the cause of mepd appearing to hang on GSM runs
        that happened to produce more output before an early stop than a
        normal run produces before finishing.
        """
        chunks: list[str] = []

        def _drain() -> None:
            if proc.stdout is None:
                return
            try:
                for line in proc.stdout:
                    chunks.append(line)
            except (ValueError, OSError):
                pass

        threading.Thread(target=_drain, daemon=True).start()
        return chunks

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
        self._server_stdout_chunks = self._start_stdout_drain(proc)

        timeout = 30.0
        poll_interval = 0.01
        waited = 0.0
        while not ready_marker.exists():
            if proc.poll() is not None:
                raise ElectronicStructureError(
                    msg="GSM engine server exited before it was ready.",
                    obj="".join(self._server_stdout_chunks),
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
        self,
        workdir: Path,
        reactant,
        product,
        e_reactant: float,
        chain_parameters,
        allow_early_stop: bool = False,
    ) -> tuple[list[Chain], bool]:
        """Run the compiled binary, live-printing and accumulating a real
        `Chain` snapshot (see `_build_live_chain`) every time the engine
        server reports new node data. Returns `(history, early_stopped)`.

        If `allow_early_stop`, also monitors that same history for a
        persistent local minimum (`_track_persistent_minima`) once growth
        has finished, and terminates the binary the moment one has held
        steady for `early_stop_persistence_window` consecutive updates --
        GSM's own DE-GSM mode has no native equivalent to lean on here (its
        `INT_THRESH`/`find_peaks(type=4)` multi-peak detection only fires
        for SSM, gated behind `if (isSSM && !added)` in `opt_iters`,
        `GSM/gstring.cpp`). When that happens `history[-1]` is the best
        final chain there is -- there's no `stringfile.xyz0000` from a
        killed process -- and `early_stopped=True` tells the caller to
        verify it with `check_if_elem_step` before trusting it, since a
        killed run is never as refined as one that converged naturally.
        """
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

        # Drain stdout continuously (see _start_stdout_drain's docstring) --
        # GSM produces substantial output over a run, and without an active
        # reader the OS pipe buffer fills and the child deadlocks blocked on
        # its own write(), which then makes proc.poll() below never return.
        stdout_chunks = self._start_stdout_drain(proc)

        # GSM's own optimization loop is entirely inside the compiled binary,
        # so polling the live-snapshot file the engine server rewrites after
        # every node request (see _ENGINE_SERVER_SOURCE) is the only way to
        # show progress -- there's nothing to await synchronously otherwise.
        history: list[Chain] = []
        early_stopped = False
        nnodes_target = int(getattr(self.parameters, "nnodes", 9))
        persistence_window = int(
            getattr(self.parameters, "early_stop_persistence_window", 3)
        )
        persistence_rtol = float(
            getattr(self.parameters, "early_stop_minima_rtol", 0.02)
        )
        persistence_min_depth_kcal = float(
            getattr(self.parameters, "early_stop_minima_min_depth_kcal", 1.0)
        )
        persistence_state = {"index": None, "energy_kcal": None, "count": 0}

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
                live_chain = self._build_live_chain(
                    live_path, reactant, product, e_reactant, chain_parameters
                )
                if live_chain is not None:
                    history.append(live_chain)
                    self._print_live_chain(live_chain)
                    triggered = allow_early_stop and self._track_persistent_minima(
                        live_chain,
                        nnodes_target,
                        persistence_window,
                        persistence_rtol,
                        persistence_min_depth_kcal,
                        persistence_state,
                    )
                    if triggered:
                        self._log(
                            "Persistent local minimum detected in the live "
                            "GSM string; stopping early to verify it "
                            "against check_if_elem_step.",
                            level="warning",
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=5.0)
                        early_stopped = True
                        break
            time.sleep(poll_interval)

        if not early_stopped:
            # One last refresh in case a final write raced the process exit
            # -- only if the file actually changed since our last in-loop
            # capture (_build_live_chain always returns a fresh object, so
            # comparing the chains themselves can't detect "nothing new
            # happened here").
            try:
                mtime = live_path.stat().st_mtime
            except OSError:
                mtime = None
            if mtime is not None and mtime != last_mtime:
                live_chain = self._build_live_chain(
                    live_path, reactant, product, e_reactant, chain_parameters
                )
                if live_chain is not None:
                    history.append(live_chain)
                    self._print_live_chain(live_chain)

            if proc.returncode != 0 or not (workdir / "stringfile.xyz0000").exists():
                raise ElectronicStructureError(
                    msg=f"GSM calculation failed with exit code {proc.returncode}.",
                    obj="".join(stdout_chunks),
                )

        return history, early_stopped

    @staticmethod
    def _track_persistent_minima(
        live_chain: Chain,
        nnodes_target: int,
        window: int,
        rtol: float,
        min_depth_kcal: float,
        state: dict,
    ) -> bool:
        """Update `state` (mutated in place) with the current live chain's
        deepest interior local-minimum candidate (reusing
        `mepd.elementarystep._get_ind_minima`, the same scan NEB's own
        concavity check is built on) and report whether a minimum of about
        the same *depth* (energy within `rtol` of the profile's energy span)
        has now held steady across `window` consecutive polls in a row,
        counting only polls taken after growth has finished -- a partial
        string's minima aren't real signal yet, since GSM keeps inserting
        interior nodes while it's still growing.

        `nnodes_target` (`path_min_inputs.nnodes`) is a ceiling, not a
        guarantee: confirmed directly on the oxy-Cope rearrangement, where
        the converged string stopped growing at 10 nodes despite `nnodes=11`
        being requested -- an earlier version of this check required
        `len(chain) == nnodes_target` exactly, so growth was (wrongly)
        considered permanently incomplete for the entire rest of that run,
        and persistence could never even start accumulating. "Growth
        finished" is instead detected empirically: the chain's length
        hasn't changed since the last poll.

        Deliberately tracks by energy value, not node index: GSM
        periodically reparametrizes the string (`ic_reparam`/`ic_reparam_g`
        in GSM/gstring.cpp) to keep nodes evenly spaced, which can shift
        which index a given physical location along the path corresponds to
        between polls even when the underlying chemistry -- a real, stable
        dip -- hasn't changed. Requiring the *same index* to hold across
        consecutive polls (an earlier version of this check) meant
        reparametrization noise could defeat detection of a minimum that
        was, in substance, already persistent.
        """
        length_stable = state.get("length") == len(live_chain)
        state["length"] = len(live_chain)
        if not length_stable:
            state["index"] = None
            state["energy_kcal"] = None
            state["count"] = 0
            return False

        from mepd.elementarystep import _get_ind_minima

        minima = _get_ind_minima(live_chain)
        if len(minima) == 0:
            state["index"] = None
            state["energy_kcal"] = None
            state["count"] = 0
            return False

        energies_kcal = live_chain.energies_kcalmol
        # Depth = the smaller of the two drops down to this candidate from
        # its immediate neighbors -- _get_ind_minima only guarantees it's
        # lower than both neighbors by *some* amount, which includes pure
        # numerical noise (confirmed empirically: an early poll on the
        # oxy-Cope rearrangement triggered on a "minimum" ~0.01 kcal/mol
        # deep, essentially flat, near the reactant end). Require a real
        # barrier drop, mirroring the spirit of GSM's own INT_THRESH
        # (PEAK4_EDIFF in GSM/gstring.cpp) -- which would do exactly this
        # for us if it applied to DE-GSM, but per _run_gsm's docstring it
        # doesn't.
        depths = np.minimum(
            energies_kcal[minima - 1] - energies_kcal[minima],
            energies_kcal[minima + 1] - energies_kcal[minima],
        )
        deep_enough = minima[depths >= min_depth_kcal]
        if len(deep_enough) == 0:
            state["index"] = None
            state["energy_kcal"] = None
            state["count"] = 0
            return False

        idx = int(deep_enough[np.argmin(energies_kcal[deep_enough])])
        energy = float(energies_kcal[idx])
        span = float(np.ptp(energies_kcal)) or 1.0

        if state["energy_kcal"] is not None and abs(energy - state["energy_kcal"]) / span <= rtol:
            state["count"] += 1
        else:
            state["count"] = 1
        state["index"] = idx
        state["energy_kcal"] = energy
        return state["count"] >= max(1, window)

    def _print_live_chain(self, live_chain: Chain) -> None:
        """Render an already-built live chain (see `_build_live_chain`) as a
        live, in-place-updating ASCII profile -- the GSM analogue of the live
        chain view NEB already gets via mepd.progress.print_chain_step after
        every optimizer step.
        """
        from mepd.progress import print_chain_step

        nnodes_target = int(getattr(self.parameters, "nnodes", 9))
        print_chain_step(
            live_chain,
            caption=f"GSM live string ({len(live_chain)}/{nnodes_target} nodes seen)",
        )

    def _build_live_chain(
        self, live_path: Path, reactant, product, e_reactant: float, chain_parameters
    ) -> Chain | None:
        """Reconstruct a `Chain` from whatever of the string the engine
        server has reported so far. Returns `None` if there's nothing usable
        to build from yet (missing/empty/malformed snapshot -- transient
        states any poller hitting this mid-write can see).

        The reactant/product endpoints are always pinned to the real,
        already-known nodes rather than whatever currently holds the lowest/
        highest seen node index in the live snapshot -- during growth, GSM
        inserts new interior nodes over time, so an index's *meaning* can
        shift, and a partial snapshot's nominal "first"/"last" entries can
        both still be close to the reactant (not yet differentiated into the
        product) -- letting that leak into e.g. the live view's start/end
        SMILES gave the misleading impression the endpoints were identical.
        """
        try:
            text = live_path.read_text()
        except OSError:
            return None
        if not text.strip():
            return None
        try:
            blocks = self._parse_string_blocks(text, len(reactant.symbols))
        except Exception:
            return None
        if not blocks:
            return None

        from qcconst.constants import ANGSTROM_TO_BOHR

        nodes = []
        for v_kcal, coords in blocks:
            coords_bohr = np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR
            node = reactant.update_coords(coords_bohr)
            node._cached_energy = e_reactant + v_kcal / _KCAL_PER_HARTREE
            nodes.append(node)
        if len(nodes) >= 2:
            nodes[0] = reactant
            nodes[-1] = product
        return Chain.model_validate({"nodes": nodes, "parameters": chain_parameters})

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
