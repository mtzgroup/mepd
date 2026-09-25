"""FAIR-Chem machine-learned potentials (UMA and other fairchem checkpoints)
as a level of theory.

Energies and gradients for a whole chain are evaluated in batched forward
passes on the GPU instead of one image at a time; geometry optimizations,
transition-state searches and IRCs go through ASE exactly as for
`ASEEngine`. The model is loaded lazily and shared between copies of the
engine, and never pickled, so GSM's engine server and the parallel MSMEP
runner each reuse or reload it instead of serialising GPU tensors.

Requires `fairchem-core` (and a Hugging Face login with access to the
gated `facebook/UMA` weights for the named UMA models).
"""
from __future__ import annotations

from typing import List, Union

import numpy as np
from ase.units import Hartree
from numpy.typing import NDArray
from qcconst.constants import ANGSTROM_TO_BOHR

from mepd.chain import Chain
from mepd.engines.ase import AVAIL_OPTS, ASEEngine
from mepd.errors import ElectronicStructureError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache
from mepd.qcdata_structure_helpers import structure_to_ase_atoms

_EV_PER_ANG_TO_HARTREE_PER_BOHR = 1.0 / (Hartree * ANGSTROM_TO_BOHR)


class FAIRChemEngine(ASEEngine):
    """A fairchem MLIP. `model` is a pretrained model name (e.g. "uma-s-1p2p1",
    "uma-m-1p1"); `checkpoint`, if given, is a local checkpoint file used
    instead. `task` selects the UMA head ("omol" for molecules, which reads
    charge and spin multiplicity from each structure). `device` defaults to
    CUDA when available. `batch_size` caps how many structures go through
    one forward pass."""

    def __init__(
        self,
        model: str = "uma-s-1p2p1",
        task: str = "omol",
        device: str | None = None,
        checkpoint: str | None = None,
        inference_settings: str = "default",
        batch_size: int = 32,
        geometry_optimizer: str = "LBFGSLineSearch",
        transition_state_optimizer: str = "SELLA",
        ase_optimizer=None,
    ):
        self.model = model
        self.task = task
        self.device = device
        self.checkpoint = checkpoint
        self.inference_settings = inference_settings
        self.batch_size = int(batch_size)
        self.geometry_optimizer = geometry_optimizer
        self.transition_state_optimizer = transition_state_optimizer
        self.ase_optimizer = ase_optimizer if ase_optimizer is not None else AVAIL_OPTS[geometry_optimizer]
        self._predictor = None

    def __repr__(self) -> str:
        source = f"checkpoint={self.checkpoint!r}" if self.checkpoint else f"model={self.model!r}"
        return f"FAIRChemEngine({source}, task={self.task!r}, device={self.device!r})"

    def __eq__(self, other) -> bool:
        return self is other

    __hash__ = object.__hash__

    # --- model loading -------------------------------------------------
    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = self._load_predictor()
        return self._predictor

    def _resolved_device(self) -> str:
        if self.device:
            return str(self.device)
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_predictor(self):
        try:
            from fairchem.core import pretrained_mlip
        except ImportError as exc:
            raise ImportError(
                "engine_name='fairchem' needs fairchem-core (pip install fairchem-core); "
                "the UMA models also need a Hugging Face login with access to facebook/UMA."
            ) from exc
        device = self._resolved_device()
        if self.checkpoint:
            return pretrained_mlip.load_predict_unit(
                self.checkpoint, inference_settings=self.inference_settings, device=device
            )
        try:
            return pretrained_mlip.get_predict_unit(
                self.model, inference_settings=self.inference_settings, device=device
            )
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            if "Gated" in type(exc).__name__ or any(code in text for code in ("401", "403", "restricted")):
                from mepd.engines.mlip import MODELS, UMA_ACCESS_HELP

                spec = MODELS.get(self.model)
                repo = spec.options.get("repo", "facebook/UMA") if spec else "facebook/UMA"
                raise PermissionError(
                    f"Can't download {self.model!r}: {text.splitlines()[0]}\n" + UMA_ACCESS_HELP.format(repo=repo)
                ) from exc
            raise

    @property
    def calculator(self):
        """A fresh ASE calculator on the shared model: ASE calculators keep
        per-structure results, so optimizations running concurrently must
        not share one."""
        from fairchem.core import FAIRChemCalculator

        return FAIRChemCalculator(self.predictor, task_name=self.task)

    @calculator.setter
    def calculator(self, value) -> None:  # ASEEngine's dataclass __init__ is never used here
        raise AttributeError("FAIRChemEngine builds its calculator from `model`/`checkpoint`.")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_predictor"] = None
        return state

    def __deepcopy__(self, memo):
        # The model is read-only at inference time: share it rather than load
        # another copy onto the GPU (not copy.copy, which goes through
        # __getstate__ and would drop it).
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        clone.__dict__.update(self.__dict__)
        return clone

    # --- batched energies and gradients ---------------------------------
    def _batch_predict(self, atoms_chunk) -> tuple[np.ndarray, np.ndarray]:
        """One forward pass over `atoms_chunk`: energies (eV, one per
        structure) and forces (eV/Angstrom, all atoms concatenated in order)."""
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

        calc = self.calculator
        for atoms in atoms_chunk:
            calc.predictor.validate_atoms_data(atoms, calc.task_name)
        pred = calc.predictor.predict(atomicdata_list_to_batch([calc.a2g(atoms) for atoms in atoms_chunk]))
        return (pred["energy"].detach().cpu().numpy().reshape(-1),
                pred["forces"].detach().cpu().numpy().reshape(-1, 3))

    def _predict(self, atoms_list) -> tuple[list[float], list[np.ndarray]]:
        """Energies (Hartree) and gradients (Hartree/Bohr) for `atoms_list`,
        in batches of `batch_size`."""
        energies, gradients = [], []
        size = max(1, self.batch_size)
        for start in range(0, len(atoms_list), size):
            chunk = atoms_list[start:start + size]
            e, f = self._batch_predict(chunk)
            offset = 0
            for i, atoms in enumerate(chunk):
                n = len(atoms)
                energies.append(float(e[i]) / Hartree)
                gradients.append(-f[offset:offset + n] * _EV_PER_ANG_TO_HARTREE_PER_BOHR)
                offset += n
        return energies, gradients

    def _run_calc(self, calctype: str, chain: Union[Chain, List]) -> List[StructureNode]:
        node_list = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [i for i, node in enumerate(node_list) if node._cached_energy is None]
        results = [node_list[i]._cached_result if i not in todo else None for i in range(len(node_list))]
        if todo:
            try:
                energies, gradients = self._predict([structure_to_ase_atoms(node_list[i].structure) for i in todo])
            except ImportError:
                raise
            except Exception as exc:
                raise ElectronicStructureError(msg=f"FAIR-Chem prediction failed: {exc}", obj=exc) from exc
            for i, e, g in zip(todo, energies, gradients):
                res = FakeQCIOResults.model_validate({"energy": e, "gradient": g})
                results[i] = FakeQCIOOutput.model_validate({"results": res})
        update_node_cache(node_list=node_list, results=results)
        return node_list

    def compute_func(self, atoms):
        energies, gradients = self._predict([atoms])
        res = FakeQCIOResults.model_validate({"energy": energies[0], "gradient": gradients[0]})
        return FakeQCIOOutput.model_validate({"results": res})

    def _compute_gradient_from_atoms(self, atoms) -> NDArray:
        return self._predict([atoms])[1][0]

    def compute_hessian(self, node: StructureNode, step_size: float | None = None) -> NDArray:
        """Central differences of the model's own gradients, with all 6N
        displaced structures evaluated in batched passes."""
        if not isinstance(node, StructureNode):
            return super().compute_hessian(node=node, step_size=step_size)
        h = float(step_size if step_size is not None else self.finite_difference_hessian_step_size)
        x0 = np.asarray(node.coords, dtype=float)
        shape, flat = x0.shape, x0.reshape(-1)
        base = structure_to_ase_atoms(node.structure)
        displaced = []
        for i in range(flat.size):
            for sign in (1.0, -1.0):
                x = flat.copy()
                x[i] += sign * h
                atoms = base.copy()
                atoms.positions = x.reshape(shape) / ANGSTROM_TO_BOHR
                displaced.append(atoms)
        _, grads = self._predict(displaced)
        hessian = np.array([(grads[2 * i] - grads[2 * i + 1]).reshape(-1) / (2.0 * h) for i in range(flat.size)])
        return 0.5 * (hessian + hessian.T)
