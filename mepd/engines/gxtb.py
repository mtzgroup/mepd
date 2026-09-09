from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree
from numpy.typing import NDArray

from mepd.chain import Chain
from qcconst.constants import ANGSTROM_TO_BOHR
from mepd.engines.engine import Engine, build_hessian_result_from_matrix
from mepd.errors import (
    ElectronicStructureError,
    EnergiesNotComputedError,
    GradientsNotComputedError,
)
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache
from mepd.qcdata_structure_helpers import ase_atoms_to_structure


_TOTAL_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+Eh")


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

        for i, node in enumerate(node_list):
            if i in inds_cached:
                continue
            results[i] = self._compute_node(node)

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
        hessian = self.compute_hessian(node=node)
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
                completed = self._run_gxtb(
                    xyz_path=xyz_path,
                    charge=int(node.structure.charge),
                    multiplicity=int(node.structure.multiplicity),
                    cwd=workdir,
                    optimize=True,
                )
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
    ) -> list[list[StructureNode]]:
        return [
            self.compute_geometry_optimization(node=node, keywords=keywords)
            for node in nodes
        ]

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
