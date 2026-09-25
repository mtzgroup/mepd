from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree
from numpy.typing import NDArray

from mepd.chain import Chain
from qcconst.constants import ANGSTROM_TO_BOHR
from mepd.engines.engine import (
    Engine,
    FiniteDifferenceHessianOutput,
    FiniteDifferenceHessianResults,
    build_hessian_result_from_matrix,
)
from mepd.errors import (
    ElectronicStructureError,
    EnergiesNotComputedError,
    GeometryOptimizationNotConvergedError,
    GradientsNotComputedError,
)
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache
from mepd.qcdata_structure_helpers import ase_atoms_to_structure


_TOTAL_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+Eh")
_OPT_FAILED_RE = re.compile(r"FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN (\d+) ITERATIONS", re.IGNORECASE)
_OPT_CONVERGED_RE = re.compile(r"GEOMETRY OPTIMIZATION CONVERGED AFTER (\d+) ITERATIONS", re.IGNORECASE)


def _check_gxtb_optimization_converged(stdout: str, *, maxiter: int | None) -> None:
    """Raise if g-xTB's own output reports the optimization did not converge.

    g-xTB happily returns its last frame (via xtbopt.xyz/xtbopt.log) even when
    it simply ran out of optimization cycles -- there is no other signal in
    the returned trajectory that distinguishes "already at a minimum" from
    "gave up mid-optimization". Only stdout says which one actually happened.
    """
    failed = _OPT_FAILED_RE.search(stdout)
    if failed is None:
        return
    iterations = failed.group(1)
    budget_note = f" (iteration budget: {int(maxiter)})" if maxiter is not None else ""
    raise GeometryOptimizationNotConvergedError(
        msg=(
            f"g-xTB geometry optimization did not converge in {iterations} iterations"
            f"{budget_note}. Refusing to treat an unconverged geometry as a minimum."
        ),
        obj=stdout,
    )



_XTBOPT_ENERGY_RE = re.compile(r"energy:\s*(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")


def _read_xtbopt_progress(fp: Path, natoms: int) -> tuple[list[float], list[str]]:
    """Energies and xyz frames of every step xtb has written to its
    optimization log so far -- cheap enough to call while the optimizer runs
    (the file is appended to; a half-written last frame is skipped)."""
    try:
        lines = fp.read_text().splitlines()
    except OSError:
        return [], []
    energies: list[float] = []
    frames: list[str] = []
    i = 0
    while i + natoms + 1 < len(lines):
        head = lines[i].strip()
        if head != str(natoms):
            i += 1
            continue
        block = lines[i + 2 : i + 2 + natoms]
        if len(block) < natoms or any(len(row.split()) < 4 for row in block):
            break
        match = _XTBOPT_ENERGY_RE.search(lines[i + 1])
        if match:
            energies.append(float(match.group(1)))
            frames.append("\n".join([str(natoms), lines[i + 1].strip(), *block]) + "\n")
        i += natoms + 2
    return energies, frames


class _GXTBASEResultsCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        engine: "GXTBCalculator",
        *,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, list(properties), system_changes)
        if self.atoms is None:
            raise ElectronicStructureError(
                msg="ASE did not provide atoms for g-xTB calculation."
            )

        charge = int(self.atoms.info.get("charge", self.charge))
        multiplicity = int(self.atoms.info.get("spin", self.multiplicity))
        structure = ase_atoms_to_structure(
            atoms=self.atoms,
            charge=charge,
            multiplicity=multiplicity,
        )
        node = StructureNode(structure=structure)

        energy_hartree = float(self.engine.compute_energies([node])[0])
        gradient_hartree_bohr = np.asarray(
            self.engine.compute_gradients([node])[0], dtype=float
        )
        self.results["energy"] = energy_hartree * Hartree
        # ASE forces are -dE/dx in eV/Angstrom.
        self.results["forces"] = -gradient_hartree_bohr * Hartree * ANGSTROM_TO_BOHR


@dataclass
class GXTBCalculator(Engine):
    """Direct local g-xTB engine using the xtb executable with the g-xTB flag."""

    executable: str | Path | None = None
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    keep_workdirs: bool = False
    add_gxtb_flag: bool = True
    n_threads: int = 1
    # Independent calculations (a chain's images, a batch of geometry
    # optimizations) run as up to this many concurrent g-xTB processes.
    # 0 = as many as the machine has cores for at n_threads each.
    n_parallel: int = 0

    def __post_init__(self) -> None:
        if self.executable is None:
            self.executable = os.getenv("GXTB_EXECUTABLE") or "gxtb"
        self.executable = str(self.executable)
        self.extra_args = [str(arg) for arg in self.extra_args]
        self.env = {str(k): str(v) for k, v in dict(self.env or {}).items()}

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        try:
            grads = np.array([node.gradient for node in chain])
        except GradientsNotComputedError:
            node_list = self._run_calc(chain=chain)
            grads = np.array([node.gradient for node in node_list])

        return grads

    def compute_energies(self, chain: Union[Chain, List]) -> NDArray:
        try:
            enes = np.array([node.energy for node in chain])
        except EnergiesNotComputedError:
            node_list = self._run_calc(chain=chain)
            enes = np.array([node.energy for node in node_list])

        return enes

    def _run_calc(self, chain: Union[Chain, List]) -> list[StructureNode]:
        node_list = self._coerce_nodes(chain)
        inds_cached = [
            i
            for i, node in enumerate(node_list)
            if node._cached_energy is not None and node._cached_gradient is not None
        ]
        results: list[FakeQCIOOutput | None] = [None] * len(node_list)
        for i in inds_cached:
            results[i] = node_list[i]._cached_result

        todo = [i for i in range(len(node_list)) if i not in inds_cached]
        for i, res in zip(todo, self._map(self._compute_node, [node_list[i] for i in todo])):
            results[i] = res

        update_node_cache(node_list=node_list, results=results)
        return node_list

    @staticmethod
    def _coerce_nodes(chain: Union[Chain, List]) -> list[StructureNode]:
        if isinstance(chain, Chain):
            node_list = chain.nodes
        elif isinstance(chain, list):
            node_list = chain
        else:
            raise ValueError(f"Input needs to be a Chain or a List. You input a: {type(chain)}")

        if not node_list:
            return []
        if not isinstance(node_list[0], StructureNode):
            raise AssertionError(
                f"input nodes are incompatible with GXTBCalculator: {node_list[0]}"
            )
        return node_list

    def _compute_node(self, node: StructureNode) -> FakeQCIOOutput:
        with tempfile.TemporaryDirectory(prefix="gxtb-") as tmp:
            workdir = Path(tmp)
            xyz_path = workdir / "structure.xyz"
            xyz_path.write_text(node.structure.to_xyz())
            completed = self._run_gxtb(
                xyz_path=xyz_path,
                charge=int(node.structure.charge),
                multiplicity=int(node.structure.multiplicity),
                cwd=workdir,
                optimize=False,
            )
            try:
                energy = self._parse_energy(workdir=workdir, stdout=completed.stdout)
                gradient = self._parse_gradient(workdir / "gradient", natoms=len(node.symbols))
            except Exception as exc:
                raise ElectronicStructureError(
                    msg="Failed to parse g-xTB output.", obj=completed.stdout + completed.stderr
                ) from exc

            if self.keep_workdirs:
                persistent = Path.cwd() / "gxtb-workdirs"
                persistent.mkdir(exist_ok=True)
                shutil.copytree(workdir, persistent / workdir.name, dirs_exist_ok=True)

        res = FakeQCIOResults.model_validate({"energy": energy, "gradient": gradient})
        return FakeQCIOOutput.model_validate({"results": res})

    def _run_gxtb(
        self,
        *,
        xyz_path: Path,
        charge: int,
        multiplicity: int,
        cwd: Path,
        optimize: bool,
        watch: Callable[[], None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            self.executable,
            str(xyz_path.name),
            "--silent",
            "--chrg",
            str(charge),
        ]
        cmd.append("--opt" if optimize else "--grad")
        uhf = max(0, int(multiplicity) - 1)
        if uhf:
            cmd.extend(["--uhf", str(uhf)])
        if self.add_gxtb_flag and Path(self.executable).name != "gxtb":
            cmd.append("--gxtb")
        cmd.extend(self.extra_args)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(self.n_threads))
        env.update(self.env)
        try:
            if watch is None:
                completed = subprocess.run(
                    cmd,
                    cwd=cwd,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            else:
                # Same run, but call `watch()` about twice a second while it
                # goes (a live viewer reading the optimizer's step log).
                proc = subprocess.Popen(
                    cmd, cwd=cwd, env=env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                while True:
                    try:
                        stdout, stderr = proc.communicate(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        try:
                            watch()
                        except Exception:
                            pass
                completed = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except FileNotFoundError as exc:
            raise ElectronicStructureError(
                msg=(
                    f"g-xTB executable `{self.executable}` was not found. "
                    "Set `GXTB_EXECUTABLE` or pass `executable` to GXTBCalculator."
                )
            ) from exc
        if completed.returncode != 0:
            raise ElectronicStructureError(
                msg=f"g-xTB calculation failed with exit code {completed.returncode}.",
                obj=completed.stdout + completed.stderr,
            )
        return completed

    def compute_hessian(
        self,
        node: StructureNode,
        step_size: float | None = None,
    ) -> NDArray:
        with tempfile.TemporaryDirectory(prefix="gxtb-hess-") as tmp:
            workdir = Path(tmp)
            xyz_path = workdir / "structure.xyz"
            xyz_path.write_text(node.structure.to_xyz())
            completed = self._run_gxtb_hessian(
                xyz_path=xyz_path,
                charge=int(node.structure.charge),
                multiplicity=int(node.structure.multiplicity),
                cwd=workdir,
            )
            try:
                hessian = self._parse_hessian(workdir / "hessian", natoms=len(node.symbols))
            except Exception as exc:
                raise ElectronicStructureError(
                    msg="Failed to parse g-xTB Hessian output.",
                    obj=completed.stdout + completed.stderr,
                ) from exc

            if self.keep_workdirs:
                persistent = Path.cwd() / "gxtb-workdirs"
                persistent.mkdir(exist_ok=True)
                shutil.copytree(workdir, persistent / workdir.name, dirs_exist_ok=True)

        return hessian

    def _compute_hessian_result(self, node: StructureNode, **kwargs):
        """Prefer g-xTB's own projected vibrational analysis (parsed from its
        g98.out output -- real frequencies with translation/rotation already
        removed by g-xTB itself, plus real normal-mode displacement vectors)
        over reimplementing mass-weighting/projection ourselves. Falls back
        to the generic build_hessian_result_from_matrix (mass-weight the raw
        Hessian, discard the lowest-magnitude modes as trans/rot) only if
        g98.out isn't available or fails to parse.
        """
        natoms = len(node.symbols)
        with tempfile.TemporaryDirectory(prefix="gxtb-hess-") as tmp:
            workdir = Path(tmp)
            xyz_path = workdir / "structure.xyz"
            xyz_path.write_text(node.structure.to_xyz())
            completed = self._run_gxtb_hessian(
                xyz_path=xyz_path,
                charge=int(node.structure.charge),
                multiplicity=int(node.structure.multiplicity),
                cwd=workdir,
            )
            try:
                hessian = self._parse_hessian(workdir / "hessian", natoms=natoms)
            except Exception as exc:
                raise ElectronicStructureError(
                    msg="Failed to parse g-xTB Hessian output.",
                    obj=completed.stdout + completed.stderr,
                ) from exc

            g98_path = workdir / "g98.out"
            freqs = modes = None
            if g98_path.exists():
                try:
                    freqs, modes = self._parse_g98_frequencies_and_modes(
                        g98_path.read_text(), natoms=natoms
                    )
                except Exception:
                    freqs = modes = None

            if self.keep_workdirs:
                persistent = Path.cwd() / "gxtb-workdirs"
                persistent.mkdir(exist_ok=True)
                shutil.copytree(workdir, persistent / workdir.name, dirs_exist_ok=True)

        if freqs and modes and len(freqs) == len(modes):
            return FiniteDifferenceHessianOutput(
                input_data=SimpleNamespace(structure=node.structure),
                results=FiniteDifferenceHessianResults(
                    hessian=hessian,
                    normal_modes_cartesian=modes,
                    freqs_wavenumber=freqs,
                ),
                success=True,
            )

        # Fallback: g-xTB's own frequency analysis wasn't available/parseable.
        return build_hessian_result_from_matrix(node=node, hessian=hessian)

    def _run_gxtb_hessian(
        self,
        *,
        xyz_path: Path,
        charge: int,
        multiplicity: int,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            self.executable,
            str(xyz_path.name),
            "--silent",
            "--chrg",
            str(charge),
            "--hess",
        ]
        uhf = max(0, int(multiplicity) - 1)
        if uhf:
            cmd.extend(["--uhf", str(uhf)])
        if self.add_gxtb_flag and Path(self.executable).name != "gxtb":
            cmd.append("--gxtb")
        cmd.extend(self.extra_args)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(self.n_threads))
        env.update(self.env)
        try:
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ElectronicStructureError(
                msg=(
                    f"g-xTB executable `{self.executable}` was not found. "
                    "Set `GXTB_EXECUTABLE` or pass `executable` to GXTBCalculator."
                )
            ) from exc
        if completed.returncode != 0:
            raise ElectronicStructureError(
                msg=f"g-xTB Hessian calculation failed with exit code {completed.returncode}.",
                obj=completed.stdout + completed.stderr,
            )
        return completed

    def compute_geometry_optimization(
        self,
        node: StructureNode,
        keywords: dict[str, Any] | None = None,
    ) -> list[StructureNode]:
        kwds = dict(keywords or {})
        extra_args = list(self.extra_args)
        maxiter = kwds.pop("maxiter", kwds.pop("maxit", kwds.pop("steps", None)))
        if maxiter is not None:
            extra_args.extend(["--cycles", str(int(maxiter))])

        original_extra_args = self.extra_args
        self.extra_args = extra_args
        try:
            with tempfile.TemporaryDirectory(prefix="gxtb-opt-") as tmp:
                workdir = Path(tmp)
                xyz_path = workdir / "structure.xyz"
                xyz_path.write_text(node.structure.to_xyz())
                from mepd import progress as _progress

                sink = _progress.minimization_sink()
                watch = None
                if sink is not None:
                    natoms = len(node.symbols)

                    def watch(final: bool = False) -> None:
                        energies, frames = _read_xtbopt_progress(workdir / "xtbopt.log", natoms)
                        if energies:
                            sink(energies, frames, final=final)

                completed = self._run_gxtb(
                    xyz_path=xyz_path,
                    charge=int(node.structure.charge),
                    multiplicity=int(node.structure.multiplicity),
                    cwd=workdir,
                    optimize=True,
                    watch=watch,
                )
                if watch is not None:
                    watch(final=True)  # the steps written after the last poll
                _check_gxtb_optimization_converged(completed.stdout, maxiter=maxiter)
                try:
                    opt_nodes = self._parse_optimization_trajectory(
                        node=node,
                        fp=workdir / "xtbopt.log",
                    )
                    if not opt_nodes:
                        opt_nodes = [
                            self._parse_optimized_node(node=node, fp=workdir / "xtbopt.xyz")
                        ]
                except Exception as exc:
                    raise ElectronicStructureError(
                        msg="Failed to parse g-xTB optimization output.",
                        obj=completed.stdout + completed.stderr,
                    ) from exc

                final_result = self._compute_node(opt_nodes[-1])
                opt_nodes[-1]._cached_result = final_result
                opt_nodes[-1]._cached_energy = final_result.results.energy
                opt_nodes[-1]._cached_gradient = final_result.results.gradient

                if self.keep_workdirs:
                    persistent = Path.cwd() / "gxtb-workdirs"
                    persistent.mkdir(exist_ok=True)
                    shutil.copytree(workdir, persistent / workdir.name, dirs_exist_ok=True)
        finally:
            self.extra_args = original_extra_args

        return opt_nodes

    def compute_geometry_optimizations(
        self,
        nodes: list[StructureNode],
        keywords: dict[str, Any] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[StructureNode]]:
        """Batch geometry optimizations, one g-xTB subprocess per node, up to
        `n_parallel` at a time.

        `progress_callback`, if given, is called as `progress_callback(completed,
        total)` after each node finishes, so per-candidate progress is
        available live, unlike a true remote batch backend.
        """
        lock = threading.Lock()
        done = [0]

        def _optimize(node):
            traj = self.compute_geometry_optimization(node=node, keywords=keywords)
            with lock:
                done[0] += 1
                if progress_callback is not None:
                    progress_callback(done[0], len(nodes))
            return traj

        return self._map(_optimize, list(nodes))

    def _map(self, fn, items: list) -> list:
        """`[fn(x) for x in items]`, with up to `n_parallel` running at once."""
        width = int(self.n_parallel) or max(1, (os.cpu_count() or 1) // max(1, int(self.n_threads)))
        width = min(width, len(items))
        if width <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=width) as pool:
            return list(pool.map(fn, items))

    def _as_ase_engine_for_node(self, node: StructureNode):
        from mepd.engines.ase import ASEEngine

        calculator = _GXTBASEResultsCalculator(
            self,
            charge=int(node.structure.charge),
            multiplicity=int(node.structure.multiplicity),
        )
        return ASEEngine(calculator=calculator)

    def compute_transition_state(
        self,
        node: StructureNode,
        keywords: dict[str, Any] | None = None,
    ) -> StructureNode:
        """Optimize a TS guess through ASE/Sella using local g-xTB energies/gradients."""
        return self._as_ase_engine_for_node(node).compute_transition_state(
            node=node,
            keywords=keywords,
        )

    def compute_irc_chain(
        self,
        ts_node: StructureNode,
        keywords: dict[str, Any] | None = None,
    ) -> Chain:
        """Compute an IRC through ASE/Sella using local g-xTB energies/gradients."""
        return self._as_ase_engine_for_node(ts_node).compute_irc_chain(
            ts_node=ts_node,
            keywords=keywords,
        )

    @staticmethod
    def _parse_energy(*, workdir: Path, stdout: str) -> float:
        energy_fp = workdir / "energy"
        if energy_fp.exists():
            for line in energy_fp.read_text().splitlines():
                fields = line.split()
                if len(fields) >= 2 and fields[0].isdigit():
                    return float(fields[1])

        matches = _TOTAL_ENERGY_RE.findall(stdout)
        if matches:
            return float(matches[-1])
        raise ValueError("Could not find total energy in g-xTB output.")

    @staticmethod
    def _parse_gradient(fp: Path, natoms: int) -> NDArray:
        if not fp.exists():
            raise FileNotFoundError(f"g-xTB gradient file not found: {fp}")
        rows = []
        for raw_line in fp.read_text().splitlines():
            fields = raw_line.split()
            if len(fields) != 3:
                continue
            try:
                rows.append([float(value.replace("D", "E")) for value in fields])
            except ValueError:
                continue
        if len(rows) < natoms:
            raise ValueError(f"Expected at least {natoms} gradient rows, found {len(rows)}.")
        return np.asarray(rows[-natoms:], dtype=float)

    @staticmethod
    def _parse_hessian(fp: Path, natoms: int) -> NDArray:
        if not fp.exists():
            raise FileNotFoundError(f"g-xTB Hessian file not found: {fp}")
        ndof = int(natoms) * 3
        values: list[float] = []
        in_block = False
        for raw_line in fp.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith("$hessian"):
                in_block = True
                continue
            if lower.startswith("$end"):
                break
            if not in_block and stripped.startswith("$"):
                continue
            if not in_block:
                continue
            for field in stripped.split():
                try:
                    values.append(float(field.replace("D", "E")))
                except ValueError:
                    continue
        expected = ndof * ndof
        if len(values) < expected:
            raise ValueError(
                f"Expected {expected} Hessian values for {natoms} atoms, found {len(values)}."
            )
        hessian = np.asarray(values[:expected], dtype=float).reshape((ndof, ndof))
        return 0.5 * (hessian + hessian.T)

    @staticmethod
    def _parse_g98_frequencies_and_modes(
        text: str, natoms: int
    ) -> tuple[list[float], list[NDArray]]:
        """Parse g-xTB's Gaussian-98-format frequency output (g98.out).

        Unlike the raw `hessian` file, this already reports g-xTB's own
        *projected* vibrational frequencies (translation/rotation removed)
        plus real per-atom normal-mode displacement vectors -- modes are
        printed in blocks of up to 3, each block starting with a
        "Frequencies --" line followed by per-atom XYZ displacement triplets.
        """
        lines = text.splitlines()
        freqs: list[float] = []
        mode_rows: list[list[list[float]]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("Frequencies --"):
                block_freqs = [float(v) for v in line.split("--", 1)[1].split()]
                n_in_block = len(block_freqs)
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("Atom"):
                    j += 1
                j += 1  # skip the "Atom AN X Y Z ..." header itself
                block_rows = [[] for _ in range(n_in_block)]
                for a in range(natoms):
                    values = [float(v) for v in lines[j + a].split()[2:]]
                    for m in range(n_in_block):
                        block_rows[m].append(values[3 * m: 3 * m + 3])
                freqs.extend(block_freqs)
                mode_rows.extend(block_rows)
                i = j + natoms
            else:
                i += 1

        modes = [np.asarray(rows, dtype=float) for rows in mode_rows]
        return freqs, modes

    @staticmethod
    def _parse_optimized_node(node: StructureNode, fp: Path) -> StructureNode:
        nodes = GXTBCalculator._parse_optimization_trajectory(node=node, fp=fp)
        if not nodes:
            raise ValueError(f"No optimized geometry found in {fp}.")
        return nodes[-1]

    @staticmethod
    def _parse_optimization_trajectory(node: StructureNode, fp: Path) -> list[StructureNode]:
        if not fp.exists():
            raise FileNotFoundError(f"g-xTB optimization file not found: {fp}")
        lines = fp.read_text().splitlines()
        nodes: list[StructureNode] = []
        i = 0
        natoms_expected = len(node.symbols)
        while i < len(lines):
            try:
                natoms = int(lines[i].strip())
            except ValueError:
                i += 1
                continue
            if natoms != natoms_expected or i + natoms + 1 >= len(lines):
                i += 1
                continue
            comment = lines[i + 1]
            coords_angstrom = []
            symbols = []
            for raw_line in lines[i + 2 : i + 2 + natoms]:
                fields = raw_line.split()
                if len(fields) < 4:
                    coords_angstrom = []
                    break
                symbols.append(fields[0])
                coords_angstrom.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if len(coords_angstrom) == natoms:
                new_node = node.update_coords(np.asarray(coords_angstrom, dtype=float) * ANGSTROM_TO_BOHR)
                energy_match = re.search(r"energy:\s*(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", comment)
                if energy_match:
                    energy = float(energy_match.group(1))
                    result = FakeQCIOOutput.model_validate(
                        {"results": FakeQCIOResults.model_validate({"energy": energy, "gradient": np.zeros_like(new_node.coords)})}
                    )
                    new_node._cached_result = result
                    new_node._cached_energy = energy
                    new_node._cached_gradient = np.zeros_like(new_node.coords)
                nodes.append(new_node)
            i += natoms + 2
        return nodes
