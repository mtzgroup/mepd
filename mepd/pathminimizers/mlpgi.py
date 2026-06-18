from mepd.mlp_geodesic.utils import OptimizerConfig
from mepd.mlp_geodesic.optimizer import GeodesicOptimizer
from mepd.mlp_geodesic.mlp_tools import resolve_fairchem_model_path
from mepd.qcio_structure_helpers import structure_to_ase_atoms
import logging
import os
import sys
from mepd.pathminimizers.pathminimizer import PathMinimizer
from mepd.engines.engine import Engine
from mepd.elementarystep import ElemStepResults, check_if_elem_step
import numpy as np
import warnings
from mepd.scripts.progress import update_status, print_persistent
warnings.filterwarnings('ignore')

from dataclasses import dataclass, field, fields
from mepd import Chain, RunInputs
from mepd.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROMS
from mepd.nodes.node import StructureNode

import torch
from ase.units import Hartree

_KCAL_MOL_TO_EV = 0.0433641
_HARTREE_TO_EV = float(Hartree)
_ENGINE_BACKEND_ALIASES = {"engine", "target-engine", "chemcloud", "qcop", "crest"}


def _parameter_dict(parameters: object | None) -> dict:
    if parameters is None:
        return {}
    if isinstance(parameters, dict):
        return dict(parameters)
    if hasattr(parameters, "__dict__"):
        return dict(parameters.__dict__)
    return {}


def _coerce_refinement_fraction(value):
    fraction = float(value)
    return fraction / 100.0 if fraction > 1.0 else fraction


def _kcal_mol_to_ev(value):
    return float(value) * _KCAL_MOL_TO_EV


def _resolve_mlpgi_backend(parameters: object | None, engine: Engine | None = None) -> tuple[str, str]:
    raw_payload = _parameter_dict(parameters)
    requested = str(raw_payload.get("backend") or os.environ.get("NEB_MLPGI_BACKEND", "auto")).strip()
    normalized = requested.lower()

    if normalized == "auto":
        program = str(getattr(engine, "program", "") or "").lower()
        compute_program = str(getattr(engine, "compute_program", "") or "").lower()
        if program == "crest" and compute_program in {"chemcloud", "qcop"}:
            return requested, "engine"
        return requested, "fairchem"

    if normalized in _ENGINE_BACKEND_ALIASES:
        return requested, "engine"
    return requested, normalized


def _resolve_optimizer_config_values(parameters: object | None) -> dict:
    # Defaults correspond to Table 1 in 10.1021/acs.jctc.5c01221.
    config_values = {
        "fire_stage1_iter": 200,
        "fire_stage2_iter": 500,
        "fire_grad_tol": 1e-2,
        "variance_penalty_weight": 0.0433641,  # 1 kcal/mol in eV
        "fire_conv_window": 20,
        "fire_conv_geolen_tol": 0.0108410,     # 0.25 kcal/mol in eV
        "fire_conv_erelpeak_tol": 0.0108410,   # 0.25 kcal/mol in eV
        "refinement_step_interval": 10,
        "refinement_dynamic_threshold_fraction": 0.1,
        "tangent_project": True,
        "climb": True,
        "alpha_climb": 0.5,
    }

    raw = _parameter_dict(parameters)
    config_keys = {f.name for f in fields(OptimizerConfig)}
    kcal_inputs = {"fire_conv_geolen_tol", "fire_conv_erelpeak_tol"}
    for key in config_keys - kcal_inputs:
        if raw.get(key) is not None:
            config_values[key] = raw[key]
    if raw.get("fire_conv_geolen_tol") is not None:
        config_values["fire_conv_geolen_tol"] = _kcal_mol_to_ev(raw["fire_conv_geolen_tol"])
    if raw.get("fire_conv_erelpeak_tol") is not None:
        config_values["fire_conv_erelpeak_tol"] = _kcal_mol_to_ev(raw["fire_conv_erelpeak_tol"])

    # Paper/CLI compatibility aliases in path_min_inputs.
    if raw.get("beta") is not None and raw.get("variance_penalty_weight") is None:
        config_values["variance_penalty_weight"] = _kcal_mol_to_ev(raw["beta"])
    if raw.get("tau_refine") is not None and raw.get("refinement_step_interval") is None:
        config_values["refinement_step_interval"] = int(raw["tau_refine"])
    if raw.get("cutoff") is not None and raw.get("refinement_dynamic_threshold_fraction") is None:
        config_values["refinement_dynamic_threshold_fraction"] = _coerce_refinement_fraction(raw["cutoff"])
    if raw.get("convergence_window") is not None and raw.get("fire_conv_window") is None:
        config_values["fire_conv_window"] = int(raw["convergence_window"])
    if raw.get("path_length_tolerance") is not None and raw.get("fire_conv_geolen_tol") is None:
        config_values["fire_conv_geolen_tol"] = _kcal_mol_to_ev(raw["path_length_tolerance"])
    if raw.get("barrier_height_tolerance") is not None and raw.get("fire_conv_erelpeak_tol") is None:
        config_values["fire_conv_erelpeak_tol"] = _kcal_mol_to_ev(raw["barrier_height_tolerance"])

    # If users provide percentages (e.g. 10), normalize to fraction (0.1).
    config_values["refinement_dynamic_threshold_fraction"] = _coerce_refinement_fraction(
        config_values["refinement_dynamic_threshold_fraction"]
    )
    return config_values


@dataclass
class MLPGI(PathMinimizer):
    initial_chain: Chain
    engine: Engine
    parameters: object = None

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    gradient_trajectory: list[np.array] = field(default_factory=list)
    geom_grad_calls_made: int = 0
    grad_calls_made: int = 0
    _verbose: bool = False

    def __post_init__(self):
        self.config = OptimizerConfig(**_resolve_optimizer_config_values(self.parameters))

        logger = logging.getLogger('geodesic')
        logger.propagate = False
        logger.setLevel(logging.WARNING)
        self._verbose = bool(_parameter_dict(self.parameters).get("v", False))

    def _status(self, message: str, persistent: bool = False):
        if self._verbose:
            print(message)
            sys.stdout.flush()
            return
        if persistent:
            print_persistent(message=message)
        else:
            update_status(message)


    def optimize_chain(self) -> ElemStepResults:
        chain = self.initial_chain.copy()
        self._status("MLPGI: computing initial chain energies")
        self.engine.compute_energies(chain)
        self.grad_calls_made += len(chain)  # assuming one grad call per node
        self.chain_trajectory.append(chain)
        # convert structure
        initial_frames = [structure_to_ase_atoms(node.structure) for node in chain]

        # Allow overrides from environment for portability without changing API.
        requested_backend, backend = _resolve_mlpgi_backend(self.parameters, self.engine)
        model_path = getattr(self.parameters, "model_path", None) or os.environ.get(
            "NEB_MLPGI_MODEL",
            "esen_sm_conserving_all.pt",
        )
        auto_download_model = getattr(
            self.parameters, "auto_download_model", None
        )
        if auto_download_model is None:
            auto_download_model = os.environ.get(
                "NEB_MLPGI_AUTO_DOWNLOAD", ""
            ).lower() in {"1", "true", "yes"}
        model_repo = getattr(self.parameters, "model_repo", None) or os.environ.get(
            "NEB_MLPGI_MODEL_REPO",
            "facebook/OMol25",
        )
        model_cache_dir = getattr(
            self.parameters, "model_cache_dir", None
        ) or os.environ.get("NEB_MLPGI_MODEL_CACHE_DIR")
        hf_token = getattr(self.parameters, "hf_token", None) or os.environ.get(
            "HF_TOKEN"
        )
        dtype_name = (
            getattr(self.parameters, "dtype", None)
            or os.environ.get("NEB_MLPGI_DTYPE", "float32")
        ).lower()
        dtype = torch.float64 if dtype_name == "float64" else torch.float32
        device_override = getattr(self.parameters, "device", None)
        if backend == "engine":
            device_default = "cpu"
        else:
            device_default = "cuda" if torch.cuda.is_available() else "cpu"
        device = device_override or os.environ.get("NEB_MLPGI_DEVICE", device_default)
        if device.startswith("cuda") and not torch.cuda.is_available():
            self._status("MLPGI: CUDA unavailable, using CPU")
            device = "cpu"

        batch_evaluator = None
        if backend == "engine":
            if not isinstance(chain[0], StructureNode):
                raise TypeError(
                    "MLPGI engine backend requires StructureNode images."
                )
            template_node = chain[0].copy()
            force_scale = _HARTREE_TO_EV / BOHR_TO_ANGSTROMS

            def _engine_batch_evaluator(
                nodes_for_eval: torch.Tensor,
                out_device: torch.device,
                out_dtype: torch.dtype,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if nodes_for_eval.size(0) == 0:
                    return (
                        torch.empty(0, device=out_device, dtype=out_dtype),
                        torch.empty((0, template_node.coords.size), device=out_device, dtype=out_dtype),
                    )
                coords_ang = nodes_for_eval.detach().to(device="cpu", dtype=torch.float64).numpy()
                eval_nodes = [
                    template_node.update_coords(coords * ANGSTROM_TO_BOHR)
                    for coords in coords_ang
                ]
                gradients_h_bohr = np.asarray(
                    self.engine.compute_gradients(eval_nodes), dtype=float
                )
                try:
                    energies_h = np.asarray([float(node.energy) for node in eval_nodes], dtype=float)
                except Exception:
                    energies_h = np.asarray(self.engine.compute_energies(eval_nodes), dtype=float)

                energies_ev = energies_h * _HARTREE_TO_EV
                forces_ev_ang = -gradients_h_bohr * force_scale
                return (
                    torch.as_tensor(energies_ev, device=out_device, dtype=out_dtype),
                    torch.as_tensor(
                        forces_ev_ang.reshape(len(eval_nodes), -1),
                        device=out_device,
                        dtype=out_dtype,
                    ),
                )

            batch_evaluator = _engine_batch_evaluator

        if backend == "fairchem":
            self._status("MLPGI: resolving fairchem checkpoint")
            model_path = resolve_fairchem_model_path(
                model_path=model_path,
                auto_download=bool(auto_download_model),
                model_repo=model_repo,
                cache_dir=model_cache_dir,
                hf_token=hf_token,
                status_callback=self._status,
            )
        self._status(
            f"MLPGI setup: backend={backend} (requested={requested_backend}) device={device} dtype={dtype_name}",
            persistent=True,
        )

        # 3. Initialize Optimizer
        opt = GeodesicOptimizer(
            frames=initial_frames,
            backend=backend,
            model_path=model_path,
            device=device,
            dtype=dtype,
            config=self.config,
            status_callback=self._status,
            batch_evaluator=batch_evaluator,
        )

        self._status("MLPGI: starting optimizer stages")
        main_coords, main_E = opt.optimize()

        new_chain = chain.copy()
        new_nodes = [chain[0].update_coords(c*ANGSTROM_TO_BOHR) for c in main_coords]
        new_chain.nodes = new_nodes
        self._status("MLPGI: evaluating optimized chain on target engine")
        self.engine.compute_energies(new_chain)
        self.grad_calls_made += len(new_chain)  # assuming one grad call per node

        self.chain_trajectory.append(new_chain)
        self.optimized = new_chain

        self._status("MLPGI: running elementary-step checks")
        elem_step_results = check_if_elem_step(
            inp_chain=new_chain,
            engine=self.engine,
            validate_minima_with_hessian=bool(
                self._params.get("validate_minima_with_hessian", False)
            ),
            hessian_minimum_frequency_cutoff=float(
                self._params.get("hessian_minimum_frequency_cutoff", 0.0)
            ),
        )
        self.geom_grad_calls_made += elem_step_results.number_grad_calls

        self.optimized = new_chain
        self._status("MLPGI: complete", persistent=True)
        return elem_step_results
