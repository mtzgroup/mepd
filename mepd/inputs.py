from __future__ import annotations
import shutil
from qcio import ProgramArgs
from types import SimpleNamespace
from dataclasses import dataclass, field
from dataclasses import is_dataclass, asdict

from mepd.optimizers.vpo import VelocityProjectedOptimizer
from mepd.optimizers.cg import ConjugateGradient
from mepd.optimizers.lbfgs import LBFGS
from mepd.optimizers.adam import AdamOptimizer
from mepd.optimizers.amg import AdaptiveMomentumGradient
from mepd.optimizers.fire import FIREOptimizer
from mepd.optimizers.sgd import SGDOptimizer
from mepd.optimizers.gd import DeterministicGradientDescentOptimizer
import tomli
import tomli_w
from pathlib import Path
import warnings

_MLPGI_DEFAULT_PATH_MIN_INPUTS = {
    # Generic MSMEP behavior flags
    "skip_identical_graphs": True,
    "do_elem_step_checks": True,
    "v": False,
    # Backend/model options
    "backend": "auto",
    "model_path": "esen_sm_conserving_all.pt",
    "auto_download_model": False,
    "model_repo": "facebook/OMol25",
    "model_cache_dir": None,
    "hf_token": None,
    "device": None,
    "dtype": "float32",
    # Optimizer settings (Table 1 in 10.1021/acs.jctc.5c01221)
    "fire_stage1_iter": 200,
    "fire_stage2_iter": 500,
    "fire_grad_tol": 1e-2,
    "variance_penalty_weight": 0.0433641,  # 1 kcal/mol in eV
    "fire_conv_window": 20,
    "fire_conv_geolen_tol": 0.25,          # kcal/mol (converted to eV internally)
    "fire_conv_erelpeak_tol": 0.25,        # kcal/mol (converted to eV internally)
    "refinement_step_interval": 10,
    "refinement_dynamic_threshold_fraction": 0.1,
    "tangent_project": True,
    "climb": True,
    "alpha_climb": 0.5,
}

_ASE_OMOL25_DEFAULT_MODEL_PATH = "/home/diptarka/fairchem/esen_sm_conserving_all.pt"


def _normalized_path_method(path_min_method: str) -> str:
    method = str(path_min_method or "").strip().upper().replace("_", "-")
    aliases = {
        "NEBDLF": "NEB-DLF",
        "DLFNEB": "NEB-DLF",
        "DLFIND": "NEB-DLF",
        "DL-FIND": "NEB-DLF",
        "GEOMETRIC": "GEOMETRIC-NEB",
        "GEOMETRICNEB": "GEOMETRIC-NEB",
    }
    return aliases.get(method, method)


def _resolve_ase_omol25_model_settings(
    path_min_inputs: object | None,
    program_kwds: object | None,
) -> tuple[str, str]:
    """Resolve FAIR-Chem model settings for ASE OMol25 from user-provided inputs."""
    model_path = None
    device = None

    for source in (program_kwds, path_min_inputs):
        payload = _serialize_input_value(source)
        if not isinstance(payload, dict):
            continue

        if model_path is None:
            model_path = payload.get("model_path")
        if device is None:
            device = payload.get("device")

        model_payload = payload.get("model")
        if isinstance(model_payload, dict):
            if model_path is None:
                model_path = (
                    model_payload.get("model_path")
                    or model_payload.get("path")
                )
            if device is None:
                device = model_payload.get("device")

    return (
        str(model_path or _ASE_OMOL25_DEFAULT_MODEL_PATH),
        str(device or "cuda"),
    )


def _serialize_input_value(value):
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, SimpleNamespace):
        return dict(value.__dict__)
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _toml_safe(value):
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            if item is None:
                continue
            safe[key] = _toml_safe(item)
        return safe
    if isinstance(value, (list, tuple)):
        return [_toml_safe(item) for item in value if item is not None]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class PathMinInputs:
    keywords: dict = field(default_factory=dict)


@dataclass
class NEBInputs:
    """
    Object containing inputs relating to NEB convergence.
    `tol`: tolerace for optimizations (Hartrees)

    `climb`: whether to use climbing image NEB

    `en_thre`: energy difference threshold. (default: tol/450)

    `rms_grad_thre`: RMS of perpendicular gradient threhsold (default: tol)

    `max_rms_grad_thre`: maximum(RMS) of perpedicular gradients threshold (default: tol*2.5)

    `ts_grad_thre`= infinity norm of TS node threshold (default: tol*2.5)

    `ts_spring_thre`= infinity norm of spring forces of triplet around TS node (default: tol * 1.5),

    `skip_identical_graphs`: whether to skip minimizations where endpoints have identical graphs

    `early_stop_force_thre`: threshold used for early elementary-step checks; both
        TS-guess |g_perp| and TS-triplet spring-force inf-norm must be below it \
        (default: 0.0 | i.e. no early stop check)

    `max_steps`: maximum number of NEB steps allowed (default: 1000)

    `v`: whether to be verbose (default: True)

    `preopt_with_xtb`: whether to preconverge a chain using XTB (default: False)

    `adaptive_resolution`: enable automatic insertion of images where the chain is under-resolved

    `adaptive_segment_ratio`: legacy setting retained for compatibility (no longer used to gate insertion)

    `adaptive_energy_ratio`: legacy setting retained for compatibility (no longer used to gate insertion)

    `adaptive_use_energy`: legacy setting retained for compatibility (no longer used to gate insertion)

    `adaptive_max_images`: hard cap on total images while adaptively refining the chain

    `adaptive_cooldown_steps`: minimum optimization steps between adaptive insertions

    `adaptive_plateau_window`: number of recent steps to assess whether convergence metrics have stalled,
        both for adaptive image insertion and plateau-based early exit

    `adaptive_plateau_rtol`: relative-improvement tolerance used by plateau detection
        (smaller means stricter plateau requirement)

    `plateau_exit_window`: deprecated; plateau-based early exit uses `adaptive_plateau_window`

    `plateau_exit_rtol`: deprecated; plateau-based early exit uses `adaptive_plateau_rtol`

    `validate_minima_with_hessian`: when a minima-based autosplit is proposed,
        compute Hessians for optimized split candidates and reject candidates with
        significant imaginary modes

    `hessian_minimum_frequency_cutoff`: minimum allowed frequency, in cm^-1 when
        frequencies are available, for Hessian-validated minima

    `hessian_minima_rescue_displacement`: displacement, in bohr, applied along the
        lowest-frequency mode when attempting to rescue a Hessian-rejected minimum

    `network_completion_max_followup_requests`: maximum number of follow-up recursive
        network-split requests queued after the initial request_0 run

    `recursive_split_max_depth`: maximum recursive split depth allowed before
        stopping further subdivision, even if a branch is still non-elementary

    `recursive_same_pair_split_limit`: maximum number of consecutive recursive
        splits allowed for the same endpoint pair along one branch
    """

    climb: bool = True
    en_thre: float = None
    rms_grad_thre: float = None
    max_rms_grad_thre: float = None
    skip_identical_graphs: bool = True
    disable_molecular_graphs: bool = False

    ts_grad_thre: float = None
    ts_spring_thre: float = None
    barrier_thre: float = .1  # kcal/mol

    early_stop_force_thre: float = 0.01

    use_geodesic_tangent: bool = False
    do_elem_step_checks: bool = True
    adaptive_resolution: bool = True
    adaptive_segment_ratio: float = 2.0
    adaptive_energy_ratio: float = 2.0
    adaptive_use_energy: bool = True
    adaptive_max_images: int = 50
    adaptive_cooldown_steps: int = 5
    adaptive_plateau_window: int = 3
    adaptive_plateau_rtol: float = 0.05
    plateau_exit_window: int = 50
    plateau_exit_rtol: float = 0.05
    validate_minima_with_hessian: bool = False
    hessian_minimum_frequency_cutoff: float = 0.0
    hessian_minima_rescue_displacement: float = 0.1
    network_completion_max_followup_requests: int = 1000
    recursive_split_max_depth: int = 200
    recursive_same_pair_split_limit: int = 5

    max_steps: float = 1000

    v: bool = True

    def __post_init__(self):

        if self.en_thre is None:
            self.en_thre = 1e-4

        if self.rms_grad_thre is None:
            self.rms_grad_thre = 0.005

        if self.ts_grad_thre is None:
            self.ts_grad_thre = 0.001

        if self.ts_spring_thre is None:
            self.ts_spring_thre = 0.01

        if self.max_rms_grad_thre is None:
            self.max_rms_grad_thre = 0.03

    def copy(self) -> NEBInputs:
        return NEBInputs(**self.__dict__)


@dataclass
class ChainInputs:
    """
    Object containing parameters relevant to chain.
    `k`: maximum spring constant.
    `delta_k`: parameter to use for calculating energy weighted spring constants
            see: https://pubs.acs.org/doi/full/10.1021/acs.jctc.1c00462

    `node_class`: type of node to use
    `do_parallel`: whether to compute gradients and energies in parallel
    `use_geodesic_interpolation`: whether to use GI in interpolations
    `do_chain_biasing`: whether to use chain biasing (Under Development, not ready for use)
    `cb`: Chain biaser object (Under Development, not ready for use)

    `node_freezing`: whether to freeze nodes in NEB convergence
    `fraction_freeze`: multiplier applied to convergence thresholds when deciding
                       whether an individual node can be frozen
    `node_conf_en_thre`: float = threshold for energy difference (kcal/mol) of geometries
                            for identifying identical conformers

    `tc_model_method`: 'method' parameter for electronic structure calculations
    `tc_model_basis`: 'method' parameter for electronic structure calculations
    `tc_kwds`: keyword arguments for electronic structure calculations
    """

    k: float = 0.05
    delta_k: float = 0.0

    do_parallel: bool = True
    use_geodesic_interpolation: bool = True

    node_freezing: bool = True
    fraction_freeze: float = 0.1

    node_rms_thre: float = 5.0  # Bohr
    node_ene_thre: float = 5.0  # kcal/mol
    frozen_atom_indices: str = ""

    def _post_init__(self):
        if len(self.frozen_atom_indices) > 0:
            self.frozen_atom_indices = [
                int(x) for x in self.frozen_atom_indices.split(" ")]

    def copy(self) -> ChainInputs:
        return ChainInputs(**self.__dict__)


@dataclass
class GIInputs:
    """
    Inputs for geodesic interpolation. See \
        [geodesic interpolation](https://pubs.aip.org/aip/jcp/article/150/16/164103/198363/Geodesic-interpolation-for-reaction-pathways) \
            for details.

    `nimages`: number of images to use (default: 15)

    `friction`: value for friction parameter. influences the penalty for \
        pairwise distances becoming too large. (default: 0.01)

    `nudge`: value for nudge parameter. (default: 0.1)

    `random_seed`: NumPy seed for deterministic geodesic interpolation nudges. (default: 0)

    `extra_kwds`: dictionary containing other keywords geodesic interpolation might use.

    !Protip: run multiple geodesic interpolations with high nudge values and select the path
    with the shortest length.
    """

    nimages: int = 7
    friction: float = 0.01
    nudge: float = 0.1
    random_seed: int = 0
    extra_kwds: dict = field(default_factory=dict)
    align: bool = True

    def copy(self) -> GIInputs:
        return GIInputs(**self.__dict__)


@dataclass
class NetworkInputs:
    n_max_conformers: int = 10  # maximum number of conformers to keep of each endpoint
    subsample_confs: bool = True

    conf_rmsd_cutoff: float = 0.5
    # minimum distance to be considered new conformer
    # given that the graphs are identical

    network_nodes_are_conformers: bool = False
    # whether each conformer should be a separate node in the network

    maximum_barrier_height: float = 1000  # kcal/mol
    # will only populate edges with a barrier lower than this input

    use_slurm: bool = False
    # whether to submit minimization jobs to slurm queue

    verbose: bool = True

    tolerate_kinks: bool = True
    # whether to include chains with a minimum apparently present in the
    # network construction

    CREST_temp: float = 298.15  # Kelvin
    CREST_ewin: float = 6.0  # kcal/mol
    # crest inputs for conformer generation. Incomplete list.
    collapse_node_rms_thre: float = 5.0  # Bohr
    collapse_node_ene_thre: float = 5.0  # kcal/mol


@dataclass
class RunInputs:
    engine_name: str = "chemcloud"
    program: str = "crest"
    chemcloud_queue: str = "cpu"
    write_qcio: bool = False
    print_stdout: bool = False
    qcop_local_parallel_workers: int = 12
    nanoreactor_inputs: dict = None

    path_min_method: str = 'NEB'
    path_min_inputs: dict = None

    chain_inputs: dict = None
    gi_inputs: dict = None
    network_inputs: dict = None

    program_kwds: ProgramArgs = None
    ase_engine_kwds: dict = None
    gxtb_engine_kwds: dict = None
    geometry_optimizer_kwds: dict = None
    optimizer_kwds: dict = None

    def __post_init__(self):
        disable_molecular_graphs = False
        default_kwds = {}
        path_method = _normalized_path_method(self.path_min_method)

        if path_method == "NEB":
            default_kwds = NEBInputs().__dict__

        elif path_method == "FNEB":
            default_kwds = {
                "max_min_iter": 100,
                "max_grow_iter": 20,
                "verbosity": 1,
                "skip_identical_graphs": True,
                "do_elem_step_checks": True,
                "grad_tol": 0.05,  # Hartree/Bohr,
                "barrier_thre": 5,  # kcal/mol,
                "tangent": 'geodesic',
                "tangent_alpha": 1.0,  # mixing coefficient for tangents,
                "use_xtb_grow": True,
                "distance_metric": "GEODESIC",
                "min_images": 10,
                "todd_way": True,
                "dist_err": 0.1,

            }
        elif path_method == "NEB-DLF":
            default_kwds = {
                "nstep": 200,
                "min_image": None,
                "min_nebk": 0.01,
                "max_nebk": None,
                "new_minimizer": "no",
                "skip_identical_graphs": True,
                "do_elem_step_checks": True,
                "early_stop_stage": False,
                "early_stop_loose_overrides": {},
                "collect_files": True,
                "dlfind_keywords": {},
                "v": False,
            }
        elif path_method == "GEOMETRIC-NEB":
            default_kwds = {
                "max_steps": 200,
                "rms_grad_thre": 0.02,
                "max_rms_grad_thre": 0.05,
                "skip_identical_graphs": True,
                "do_elem_step_checks": True,
                "batch_engine_calls": True,
                "align": True,
                "optep": False,
                "plain": 0,
                "nebk": 1.0,
                "guessk": 0.05,
                "guessw": 0.1,
                "climb": 0.5,
                "trust": 0.1,
                "tmax": 0.3,
                "tmin": 1.2e-3,
                "v": False,
            }
        #     default_kwds = FSMInputs()
        # elif self.path_min_method.upper() == "PYGSM":
        #     default_kwds = PYGSMInputs()

        if path_method == 'MLPGI':
            default_kwds = dict(_MLPGI_DEFAULT_PATH_MIN_INPUTS)

        if self.path_min_inputs is None:
            self.path_min_inputs = SimpleNamespace(**default_kwds)

        else:
            for key, val in self.path_min_inputs.items():
                default_kwds[key] = val

            self.path_min_inputs = SimpleNamespace(**default_kwds)
        disable_flag_raw = getattr(self.path_min_inputs, "disable_molecular_graphs", None)
        disable_molecular_graphs = bool(disable_flag_raw) if disable_flag_raw is not None else False
        if disable_flag_raw is not None:
            setattr(
                self.path_min_inputs,
                "disable_molecular_graphs",
                disable_molecular_graphs,
            )
        if disable_molecular_graphs and bool(
            getattr(self.path_min_inputs, "skip_identical_graphs", False)
        ):
            warnings.warn(
                "You set path_min_inputs.disable_molecular_graphs=true together with "
                "path_min_inputs.skip_identical_graphs=true. "
                "With molecular graphs disabled, skip_identical_graphs is ignored.",
                UserWarning,
                stacklevel=2,
            )
        from mepd.nodes.node import StructureNode
        StructureNode.set_global_disable_molecular_graphs(disable_molecular_graphs)

        if self.gi_inputs is None:
            self.gi_inputs = GIInputs()
        else:
            self.gi_inputs = GIInputs(**self.gi_inputs)

        if self.network_inputs is None:
            self.network_inputs = NetworkInputs()
        else:
            self.network_inputs = NetworkInputs(**self.network_inputs)

        if self.program_kwds is None:
            if self.engine_name == "gxtb":
                program_args = None
            elif self.program in {"xtb", "crest"}:
                if shutil.which("crest") is not None:
                    self.program = 'crest'
                    program_args = ProgramArgs(
                        model={"method": "gfn2",
                               "basis": "gfn2"},
                        keywords={"threads": 1})
                elif self.program == "xtb":
                    program_args = ProgramArgs(
                        model={"method": "GFN2xTB", "basis": "GFN2xTB"},
                        keywords={})
                else:
                    program_args = ProgramArgs(
                        model={"method": "gfn2", "basis": "gfn2"},
                        keywords={"threads": 1})

            elif "terachem" in self.program:
                program_args = ProgramArgs(
                    model={"method": "ub3lyp", "basis": "3-21g"},
                    keywords={})
            else:
                raise ValueError("Need to specify program arguments")

            if self.engine_name in ['qcop', 'chemcloud']:
                self.program_kwds = program_args
        elif self.program_kwds is not None and self.engine_name in ['qcop', 'chemcloud']:
            program_args = ProgramArgs(**self.program_kwds)
            self.program_kwds = program_args

        if self.nanoreactor_inputs is None:
            self.nanoreactor_inputs = {}
        else:
            self.nanoreactor_inputs = dict(self.nanoreactor_inputs)

        if self.ase_engine_kwds is None:
            self.ase_engine_kwds = {}
        else:
            self.ase_engine_kwds = dict(self.ase_engine_kwds)

        if self.gxtb_engine_kwds is None:
            self.gxtb_engine_kwds = {}
        else:
            self.gxtb_engine_kwds = dict(self.gxtb_engine_kwds)

        if self.geometry_optimizer_kwds is None:
            self.geometry_optimizer_kwds = {}
        else:
            self.geometry_optimizer_kwds = dict(self.geometry_optimizer_kwds)

        if self.chain_inputs is None:
            self.chain_inputs = ChainInputs()

        else:
            if "friction_optimal_gi" in self.chain_inputs:
                raise ValueError(
                    "chain_inputs.friction_optimal_gi has been removed. "
                    "Set gi_inputs.friction directly instead."
                )
            self.chain_inputs = ChainInputs(**self.chain_inputs)

        if self.optimizer_kwds is None:
            self.optimizer_kwds = {
                "name": "gd",
                "timestep": 1.0,
                "max_step_norm": 2.0,
            }
        elif "name" not in self.optimizer_kwds:
            self.optimizer_kwds["name"] = "cg"

        if self.engine_name == 'qcop' or self.engine_name == 'chemcloud':
            from mepd.engines.qcop import QCOPEngine
            eng = QCOPEngine(program_args=self.program_kwds,
                             program=self.program,
                             compute_program=self.engine_name,
                             chemcloud_queue=self.chemcloud_queue,
                             write_qcio=self.write_qcio,
                             print_stdout=self.print_stdout,
                             local_parallel_workers=max(
                                 1, int(self.qcop_local_parallel_workers)
                             ),
                             geometry_optimizer_kwds=self.geometry_optimizer_kwds,
                             )
        elif self.engine_name == 'ase':
            from mepd.engines.ase import ASEEngine
            ase_progs = ['omol25']
            assert self.program in ase_progs, f"{self.program} not yet supported with ASEEngine. Use one of {ase_progs} instead."
            if self.program == 'omol25':
                try:
                    from fairchem.core import pretrained_mlip, FAIRChemCalculator
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "ASE program 'omol25' requires 'fairchem-core'. "
                        "Install a compatible fairchem-core build (currently unavailable on Python 3.14) "
                        "or use a different engine/program."
                    ) from exc
                model_path, model_device = _resolve_ase_omol25_model_settings(
                    self.path_min_inputs,
                    self.program_kwds,
                )
                try:
                    predictor = pretrained_mlip.load_predict_unit(
                        model_path,
                        device=model_device,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to load OMol25 model for ASE engine "
                        f"(model_path='{model_path}', device='{model_device}')."
                    ) from exc
                calc = FAIRChemCalculator(predictor, task_name="omol")
            else:
                raise ValueError(f"Unsupported program: {self.program}")
            ase_kwds = dict(self.ase_engine_kwds or {})
            if "geometry_optimizer" in ase_kwds:
                ase_kwds["geometry_optimizer"] = str(
                    ase_kwds["geometry_optimizer"]
                )
            if "transition_state_optimizer" in ase_kwds:
                ase_kwds["transition_state_optimizer"] = str(
                    ase_kwds["transition_state_optimizer"]
                )
            eng = ASEEngine(calculator=calc, **ase_kwds)
        elif self.engine_name == 'gxtb':
            from mepd.engines.gxtb import GXTBCalculator
            eng = GXTBCalculator(**dict(self.gxtb_engine_kwds or {}))
        else:
            raise ValueError(f"Unsupported engine: {self.engine_name}")

        setattr(eng, "disable_molecular_graphs", disable_molecular_graphs)
        self.engine = eng
        optimizer_kwds = dict(self.optimizer_kwds)
        optimizer_name = optimizer_kwds.pop("name").lower()
        optimizer_map = {
            "cg": ConjugateGradient,
            "conjugate_gradient": ConjugateGradient,
            "vpo": VelocityProjectedOptimizer,
            "velocity_projected": VelocityProjectedOptimizer,
            "lbfgs": LBFGS,
            "adam": AdamOptimizer,
            "sgd": SGDOptimizer,
            "stochastic_gradient_descent": SGDOptimizer,
            "gd": DeterministicGradientDescentOptimizer,
            "gradient_descent": DeterministicGradientDescentOptimizer,
            "deterministic_gradient_descent": DeterministicGradientDescentOptimizer,
            "amg": AdaptiveMomentumGradient,
            "adaptive_momentum": AdaptiveMomentumGradient,
            "fire": FIREOptimizer,
        }
        if optimizer_name not in optimizer_map:
            available = ", ".join(sorted(set(optimizer_map.keys())))
            raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported values: {available}")
        self.optimizer = optimizer_map[optimizer_name](**optimizer_kwds)

    @classmethod
    def open(cls, fp):

        fp = Path(fp)
        with open(fp, 'rb') as f:
            data = tomli.load(f)

        # data_dict = json.loads(data)
        obj = cls(**data)
        if hasattr(obj.program_kwds, 'files') and obj.program_kwds.files is not None:
            file_keys = obj.program_kwds.files.keys()
            if "ca0" in file_keys and "cb0" in file_keys:
                obj.program_kwds.files['ca0'] = Path(
                    obj.program_kwds.files['ca0']).read_bytes()
                obj.program_kwds.files['cb0'] = Path(
                    obj.program_kwds.files['cb0']).read_bytes()

        return obj

    def save(self, fp):
        def _toml_safe(value):
            if value is None:
                return None
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                cleaned = {}
                for sub_key, sub_val in value.items():
                    normalized = _toml_safe(sub_val)
                    if normalized is not None:
                        cleaned[sub_key] = normalized
                return cleaned
            if isinstance(value, (list, tuple)):
                return [_toml_safe(item) for item in value]
            return value

        json_dict = self.__dict__.copy()
        del json_dict['engine']
        del json_dict['optimizer']
        json_dict.pop("network_inputs", None)
        deprecated_path_min_keys = {
            "plateau_exit_window",
            "plateau_exit_rtol",
        }
        for key, val in json_dict.items():
            if 'input' in key:
                json_dict[key] = _serialize_input_value(val)
                if key == "path_min_inputs" and isinstance(json_dict[key], dict):
                    for deprecated_key in deprecated_path_min_keys:
                        json_dict[key].pop(deprecated_key, None)
            elif 'program_kwds' in key:
                d = val.json()

                if d != None:
                    d = d.replace("null", "None")
                    json_dict[key] = eval(d)
                else:
                    d = ""
                    json_dict[key] = d

        json_dict = _toml_safe(json_dict)

        with open(fp, "w+") as f:
            # json.dump(json_dict, f)
            f.write(tomli_w.dumps(json_dict))
