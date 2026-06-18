from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import Hartree
from numpy.typing import NDArray
from qcconst.constants import ANGSTROM_TO_BOHR
from mepd.qcio_structure_helpers import (
    structure_to_ase_atoms,
    ase_atoms_to_structure,
)

from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.errors import (
    EnergiesNotComputedError,
    GradientsNotComputedError,
    ElectronicStructureError,
)
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache

from ase.optimize.optimize import Optimizer
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.bfgs import BFGS
from ase.optimize.fire import FIRE
from ase.optimize.mdmin import MDMin
try:
    from sella import Sella as SellaOptimizer, IRC as SellaIRC
except Exception:
    SellaOptimizer = None
    SellaIRC = None

from mepd.dynamics.chainbiaser import ChainBiaser


from ase.io import Trajectory

from pathlib import Path
import tempfile

AVAIL_OPTS = {
    "LBFGS": LBFGS,
    "BFGS": BFGS,
    "FIRE": FIRE,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
}


@dataclass
class ASEEngine(Engine):
    """
    !!! Warning:
    ASE uses the following standard units:
        - energy (eV)
        - positions (Angstroms)

        Neb-dynamics uses Hartree and Bohr for coordinates.
        Appropriate conversions must me made.
    """

    calculator: Calculator
    ase_optimizer: Optimizer = None
    geometry_optimizer: str = "LBFGSLineSearch"
    transition_state_optimizer: str = "SELLA"
    biaser: ChainBiaser = None

    def __post_init__(self):
        if self.ase_optimizer is None:
            assert (
                self.geometry_optimizer is not None and self.geometry_optimizer in AVAIL_OPTS.keys()
            ), f"Must input either an ase optimizer or a string name for an\
             available optimizer: {AVAIL_OPTS.keys()}"

            self.ase_optimizer = AVAIL_OPTS[self.geometry_optimizer]

    def _extract_optimizer_run_kwargs(
        self,
        keywords: dict[str, Any] | None,
        *,
        default_steps: int = 500,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        kwds = dict(keywords or {})
        fmax = float(kwds.pop("fmax", 0.01))
        maxiter = kwds.pop("maxiter", kwds.pop("maxit", kwds.pop("steps", default_steps)))
        run_kwds = {"fmax": fmax, "steps": int(maxiter)}

        optimizer_kwds = kwds.pop("optimizer_kwds", kwds.pop("optimizer_kwargs", {}))
        if optimizer_kwds is None:
            optimizer_kwds = {}
        elif not isinstance(optimizer_kwds, dict):
            raise ValueError("`optimizer_kwds` must be a dictionary if provided.")
        return run_kwds, dict(optimizer_kwds)

    def _run_ase_optimization(
        self,
        node: StructureNode,
        optimizer_cls: type[Any],
        keywords: dict[str, Any] | None = None,
    ) -> list[StructureNode]:
        run_kwds, optimizer_kwds = self._extract_optimizer_run_kwargs(keywords)

        atoms = structure_to_ase_atoms(node.structure)
        atoms.calc = self.calculator
        tmp = tempfile.NamedTemporaryFile(suffix=".traj", mode="w+", delete=False)

        optimizer = optimizer_cls(
            atoms=atoms,
            logfile=None,
            trajectory=tmp.name,
            **optimizer_kwds,
        )  # ASE does geometry updates in-place.
        try:
            optimizer.run(**run_kwds)
        except Exception as exc:
            raise ElectronicStructureError(msg="Electronic structure failed.", obj=exc)

        charge = node.structure.charge
        multiplicity = node.structure.multiplicity

        aT = Trajectory(tmp.name)
        traj_list = []
        for i, _ in enumerate(aT):
            traj_list.append(
                ase_atoms_to_structure(
                    atoms=aT[i],
                    charge=charge,
                    multiplicity=multiplicity,
                )
            )

        energies = [obj.get_potential_energy() / Hartree for obj in aT]
        # ASE forces are -dE/dx in eV/Angstrom; convert to +dE/dx in Hartree/Bohr.
        gradients = [(-1 * obj.get_forces() / ANGSTROM_TO_BOHR) / Hartree for obj in aT]
        all_results = []
        for ene, grad in zip(energies, gradients):
            res = FakeQCIOResults.model_validate({"energy": ene, "gradient": grad})
            out = FakeQCIOOutput.model_validate({"results": res})
            all_results.append(out)
        Path(tmp.name).unlink()
        node_list = [StructureNode(structure=struct) for struct in traj_list]
        update_node_cache(node_list=node_list, results=all_results)
        return node_list

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        try:
            grads = np.array([node.gradient for node in chain])
        except GradientsNotComputedError:
            node_list = self._run_calc(chain=chain, calctype="gradient")
            grads = np.array([node.gradient for node in node_list])
        return grads

    def compute_energies(self, chain: Chain) -> NDArray:
        try:
            enes = np.array([node.energy for node in chain])
        except EnergiesNotComputedError:
            node_list = self._run_calc(chain=chain, calctype="energy")
            enes = np.array([node.energy for node in node_list])

        return enes

    def _run_calc(
        self, calctype: str, chain: Union[Chain, List]
    ) -> List[StructureNode]:
        if isinstance(chain, Chain):
            assert isinstance(
                chain.nodes[0], StructureNode
            ), "input Chain has nodes incompatible with QCOPEngine."
            node_list = chain.nodes
        elif isinstance(chain, list):
            assert isinstance(
                chain[0], StructureNode
            ), "input list has nodes incompatible with QCOPEngine."
            node_list = chain
        else:
            raise ValueError(
                f"Input needs to be a Chain or a List. You input a: {type(chain)}"
            )

        # now create program inputs for each geometry that is not frozen or already computes
        inds_frozen = [
            i for i, node in enumerate(node_list) if node._cached_energy is not None
        ]
        all_ase_atoms = [structure_to_ase_atoms(
            node.structure) for node in node_list]
        non_frozen_ase_atoms = [
            atoms for i, atoms in enumerate(all_ase_atoms) if i not in inds_frozen
        ]
        non_frozen_results = [
            self.compute_func(atoms=atoms) for atoms in non_frozen_ase_atoms
        ]

        # merge the results
        all_results = []
        for i, node in enumerate(node_list):
            if i in inds_frozen:
                all_results.append(node_list[i]._cached_result)
            else:
                all_results.append(non_frozen_results.pop(0))
        update_node_cache(node_list=node_list, results=all_results)

        # compute bias if relevant
        if self.biaser:
            for node in node_list:
                ene_bias = self.biaser.energy_node_bias(node=node)
                grad_bias = self.biaser.gradient_node_bias(node=node)
                node._cached_energy += ene_bias
                node._cached_gradient += grad_bias

        return node_list

    def compute_func(self, atoms: Atoms):
        try:
            ene_ev = self.calculator.get_potential_energy(atoms=atoms)  # eV
            ene = ene_ev / Hartree  # Hartree

            # ASE outputs the negative gradient
            grad_ev_ang = self.calculator.get_forces(atoms=atoms) * (
                -1
            )  # eV / Angstroms
            grad = (grad_ev_ang / ANGSTROM_TO_BOHR) / Hartree  # Hartree / Bohr
            res = FakeQCIOResults.model_validate(
                {"energy": ene, "gradient": grad})
            return FakeQCIOOutput.model_validate({"results": res})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in ASE calculation: {e}")
            raise ElectronicStructureError(msg="Electronic structure failed.")

    def _compute_gradient_from_atoms(self, atoms: Atoms) -> NDArray:
        """Return dE/dx in Hartree/Bohr for an ASE Atoms object."""
        grad_ev_ang = self.calculator.get_forces(atoms=atoms) * (-1.0)
        return (grad_ev_ang / ANGSTROM_TO_BOHR) / Hartree

    def compute_hessian(
        self,
        node: StructureNode,
        step_size: float | None = None,
    ) -> NDArray:
        """
        Compute Hessian via central differences of gradients.

        This is significantly faster than the default Engine fallback because
        it scales as O(3N) gradient evaluations rather than O((3N)^2) energy
        evaluations.
        """
        if not isinstance(node, StructureNode):
            return super().compute_hessian(node=node, step_size=step_size)

        h = float(step_size if step_size is not None else self.finite_difference_hessian_step_size)
        if h <= 0:
            raise ValueError("finite-difference Hessian step size must be positive.")

        coords_bohr = np.asarray(node.coords, dtype=float)
        refshape = coords_bohr.shape
        x0 = coords_bohr.reshape(-1)
        ndof = x0.size
        if ndof == 0:
            raise ValueError("Cannot compute Hessian for a node with zero coordinates.")

        atoms_template = structure_to_ase_atoms(node.structure)
        hessian = np.zeros((ndof, ndof), dtype=float)
        for i in range(ndof):
            disp = np.zeros(ndof, dtype=float)
            disp[i] = h

            atoms_plus = atoms_template.copy()
            atoms_plus.positions = (x0 + disp).reshape(refshape) / ANGSTROM_TO_BOHR
            grad_plus = self._compute_gradient_from_atoms(atoms_plus).reshape(-1)

            atoms_minus = atoms_template.copy()
            atoms_minus.positions = (x0 - disp).reshape(refshape) / ANGSTROM_TO_BOHR
            grad_minus = self._compute_gradient_from_atoms(atoms_minus).reshape(-1)

            hessian[:, i] = (grad_plus - grad_minus) / (2.0 * h)

        # Enforce symmetry to damp numerical finite-difference noise.
        return 0.5 * (hessian + hessian.T)

    def compute_geometry_optimization(self, node: StructureNode, keywords={}) -> list[StructureNode]:
        """
        Computes a geometry optimization using ASE calculation and optimizer
        """
        return self._run_ase_optimization(
            node=node,
            optimizer_cls=self.ase_optimizer,
            keywords=keywords,
        )

    def compute_transition_state(
        self,
        node: StructureNode,
        keywords: dict | None = None,
    ) -> StructureNode:
        if self.transition_state_optimizer.upper() != "SELLA":
            raise ElectronicStructureError(
                msg=(
                    "ASE transition-state optimization now only supports `SELLA`."
                )
            )
        if SellaOptimizer is None:
            raise ElectronicStructureError(
                msg=(
                    "ASE transition-state optimization requires the `sella` package, "
                    "but it is unavailable in this environment. Install `sella` to use "
                    "`ASEEngine.compute_transition_state`."
                )
            )

        ts_keywords = dict(keywords or {})
        ts_optimizer_kwds = dict(ts_keywords.pop("optimizer_kwds", ts_keywords.pop("optimizer_kwargs", {})) or {})
        ts_optimizer_kwds.setdefault("order", 1)
        ts_keywords["optimizer_kwds"] = ts_optimizer_kwds

        trajectory = self._run_ase_optimization(
            node=node,
            optimizer_cls=SellaOptimizer,
            keywords=ts_keywords,
        )
        if len(trajectory) == 0:
            raise ElectronicStructureError(
                msg="ASE transition-state optimization completed but no trajectory was produced."
            )
        return trajectory[-1]

    def compute_irc_chain(
        self,
        ts_node: StructureNode,
        keywords: dict | None = None,
    ) -> Chain:
        if SellaIRC is None:
            raise ElectronicStructureError(
                msg=(
                    "ASE IRC computation requires the `sella` package, but it is unavailable "
                    "in this environment. Install `sella` to use IRC with `ASEEngine`."
                )
            )

        ts_keywords = dict(keywords or {})
        fmax = float(ts_keywords.pop("fmax", 0.1))
        steps = int(ts_keywords.pop("maxiter", ts_keywords.pop("maxit", ts_keywords.pop("steps", 1000))))

        optimizer_kwds = ts_keywords.pop("optimizer_kwds", ts_keywords.pop("optimizer_kwargs", {}))
        if optimizer_kwds is None:
            optimizer_kwds = {}
        elif not isinstance(optimizer_kwds, dict):
            raise ValueError("`optimizer_kwds` must be a dictionary if provided.")
        optimizer_kwds = dict(optimizer_kwds)

        for key in ("dx", "eta", "gamma", "keep_going"):
            if key in ts_keywords and key not in optimizer_kwds:
                optimizer_kwds[key] = ts_keywords.pop(key)

        if ts_keywords:
            # Forward any remaining keyword customizations to the Sella IRC constructor.
            optimizer_kwds.update(ts_keywords)

        self.compute_energies([ts_node])

        def _run_direction(direction: str) -> list[StructureNode]:
            atoms = structure_to_ase_atoms(ts_node.structure)
            atoms.calc = self.calculator
            tmp = tempfile.NamedTemporaryFile(suffix=".traj", mode="w+", delete=False)
            optimizer = SellaIRC(
                atoms=atoms,
                logfile=None,
                trajectory=tmp.name,
                **optimizer_kwds,
            )
            try:
                optimizer.run(fmax=fmax, steps=steps, direction=direction)
            except Exception as exc:
                raise ElectronicStructureError(msg="ASE IRC computation failed.", obj=exc)

            charge = ts_node.structure.charge
            multiplicity = ts_node.structure.multiplicity
            aT = Trajectory(tmp.name)
            traj_nodes = [
                StructureNode(
                    structure=ase_atoms_to_structure(
                        atoms=aT[i],
                        charge=charge,
                        multiplicity=multiplicity,
                    )
                )
                for i, _ in enumerate(aT)
            ]
            energies = [obj.get_potential_energy() / Hartree for obj in aT]
            gradients = [(-1 * obj.get_forces() / ANGSTROM_TO_BOHR) / Hartree for obj in aT]
            all_results = []
            for ene, grad in zip(energies, gradients):
                res = FakeQCIOResults.model_validate({"energy": ene, "gradient": grad})
                out = FakeQCIOOutput.model_validate({"results": res})
                all_results.append(out)
            Path(tmp.name).unlink()
            update_node_cache(node_list=traj_nodes, results=all_results)
            return traj_nodes

        reverse_nodes = _run_direction("reverse")
        forward_nodes = _run_direction("forward")

        if len(reverse_nodes) == 0 and len(forward_nodes) == 0:
            raise ElectronicStructureError(
                msg="ASE IRC completed but no trajectory points were produced."
            )

        def _strip_initial_ts(nodes: list[StructureNode]) -> list[StructureNode]:
            if len(nodes) == 0:
                return nodes
            if np.allclose(np.asarray(nodes[0].coords), np.asarray(ts_node.coords), atol=1e-6):
                return nodes[1:]
            return nodes

        left_branch = list(reversed(_strip_initial_ts(reverse_nodes)))
        right_branch = _strip_initial_ts(forward_nodes)
        irc_nodes = left_branch + [ts_node.copy()] + right_branch
        return Chain.model_validate({"nodes": irc_nodes})
