from __future__ import annotations
import shutil
from mepd.program_args import ProgramArgs
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

    `disregard_stereochem`: whether connectivity comparisons ignore the
        stereochemical SMILES equality check after graph isomorphism succeeds

    `early_stop_force_thre`: threshold used for early elementary-step checks; both
        TS-guess |g_perp| and TS-triplet spring-force inf-norm must be below it \
        (default: 0.0 | i.e. no early stop check)

    `negative_steps_thre`: number of steps chain can oscillate until the step size is halved
        (default: 2). Synced onto `NEB.optimizer` at construction time if the optimizer
        has a same-named attribute (currently only `ConjugateGradient` does).

    `positive_steps_thre`: number of stable steps before increasing the step size
        (default: 15). Synced onto `NEB.optimizer` the same way as `negative_steps_thre`.

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

    `hessian_minima_rescue_displacement`: first displacement, in bohr, applied along
        the lowest-frequency mode when attempting to rescue a Hessian-rejected
        minimum; if that fails the rescue escalates to 0.3 and 0.5 bohr
        (elementarystep.RESCUE_ESCALATION_BOHR)

    `recursive_same_pair_split_limit`: during recursive autosplitting, if a branch
        repeats the exact same (start, end) endpoint pair this many times in a row
        with no new chemistry found (a genuinely unproductive loop -- a split that
        DOES discover a new molecule/conformer is never cut off by this), stop
        splitting that branch further rather than recursing forever. For floppy
        systems that legitimately need more attempts before finding a real
        intermediate, raise this (default: 5).
    """

    climb: bool = True
    en_thre: float = None
    rms_grad_thre: float = None
    max_rms_grad_thre: float = None
    skip_identical_graphs: bool = True
    disable_molecular_graphs: bool = False
    disregard_stereochem: bool = False

    ts_grad_thre: float = None
    ts_spring_thre: float = None
    barrier_thre: float = .1  # kcal/mol

    early_stop_force_thre: float = 0.01

    negative_steps_thre: int = 2
    positive_steps_thre: int = 15
    use_geodesic_tangent: bool = False
    do_elem_step_checks: bool = True
    adaptive_resolution: bool = False
    adaptive_segment_ratio: float = 2.0
    adaptive_energy_ratio: float = 2.0
    adaptive_use_energy: bool = True
    adaptive_max_images: int = 20
    adaptive_cooldown_steps: int = 10
    adaptive_plateau_window: int = 500
    adaptive_plateau_rtol: float = 0.01
    plateau_exit_window: int = 50
    plateau_exit_rtol: float = 0.05
    validate_minima_with_hessian: bool = False
    hessian_minimum_frequency_cutoff: float = 0.0
    hessian_minima_rescue_displacement: float = 0.1
    recursive_same_pair_split_limit: int = 5

    max_steps: float = 2000

    v: bool = False

    def __post_init__(self):

        if self.en_thre is None:
            self.en_thre = 1e-4

        if self.rms_grad_thre is None:
            self.rms_grad_thre = 0.005

        if self.ts_grad_thre is None:
            self.ts_grad_thre = 0.005

        if self.ts_spring_thre is None:
            self.ts_spring_thre = 0.005

        if self.max_rms_grad_thre is None:
            self.max_rms_grad_thre = 0.01

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

    `node_freezing`: whether to freeze nodes in NEB convergence
    `fraction_freeze`: multiplier applied to convergence thresholds when deciding
                       whether an individual node can be frozen
    `node_conf_en_thre`: float = threshold for energy difference (kcal/mol) of geometries
                            for identifying identical conformers

    `tc_model_method`: 'method' parameter for electronic structure calculations
    `tc_model_basis`: 'method' parameter for electronic structure calculations
    `tc_kwds`: keyword arguments for electronic structure calculations
    """

    k: float = 0.1
    delta_k: float = 0.09

    do_parallel: bool = True
    use_geodesic_interpolation: bool = True

    node_freezing: bool = True
    fraction_freeze: float = 0.1

    node_rms_thre: float = 5.0  # Bohr
    node_ene_thre: float = 5.0  # kcal/mol
    frozen_atom_indices: str = ""

    def __post_init__(self):
        if isinstance(self.frozen_atom_indices, str) and len(self.frozen_atom_indices) > 0:
            self.frozen_atom_indices = [
                int(x) for x in self.frozen_atom_indices.split()]
        for name in ("k", "delta_k"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"ChainInputs.{name} must be a number, got {value!r}.")
            if value < 0:
                raise ValueError(f"ChainInputs.{name} must be >= 0, got {value}.")

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

    nimages: int = 10
    friction: float = 0.001
    nudge: float = 0.1
    random_seed: int = 0
    extra_kwds: dict = field(default_factory=dict)
    align: bool = True

    def copy(self) -> GIInputs:
        return GIInputs(**self.__dict__)


@dataclass
class AtomMappingInputs:
    """
    Inputs for `--atom-mapping`'s multi-candidate atom-mapping selection
    (`mepd/atom_mapping_selection.py`).

    `n_candidates`: how many of SLAPMapper's equal-minimal-cost candidate
        mappings (plus their symmetry-orbit expansions, see
        `expand_mapping_by_symmetry`) to keep and consider (default: 200).
        These are ties, not ranked by quality among themselves.

    `metric`: how each candidate (including "don't reindex") is scored from
        its geodesic-interpolated path -- one of "geodesic-distance" (the
        geodesic optimizer's own path length, free), "path-rmsd"
        (cumulative per-frame RMSD along the path, free), or "gi-energy"
        (highest QM energy along the path, most expensive). Default:
        "geodesic-distance" -- free to evaluate even across many candidates,
        unlike "gi-energy" (the metric the old single-candidate veto used).

    `veto_margin`: a non-identity candidate must beat "don't reindex" by
        more than this (in the selected metric's own units) to be adopted;
        otherwise the original ordering is kept even if some candidate
        scored marginally better. Default: 0.0 (pure best-of-N, identity
        wins exact ties) -- unlike `metric`, there is no well-calibrated
        nonzero default here yet for any of the three metrics.

    `recheck_on_split`: experimental. Also re-run this same selection at
        every new (reactant, product) pair MSMEP's recursive splitting
        discovers, not just the original --start/--end pair (default:
        False). Independent of --atom-mapping.
    """

    n_candidates: int = 200
    metric: str = "geodesic-distance"
    veto_margin: float = 0.0
    recheck_on_split: bool = False

    def copy(self) -> AtomMappingInputs:
        return AtomMappingInputs(**self.__dict__)


@dataclass
class NetworkInputs:
    """
    Inputs for NetworkBuilder (network-completion): builds/dedupes a reaction
    network graph from a set of already-completed MSMEP results. Deliberately
    minimal -- only the fields NetworkBuilder's core dedup/graph-build path
    actually reads. The original upstream NetworkInputs also carried CREST/
    slurm/conformer-sampling settings for an HPC candidate-generation pipeline
    that isn't part of this port.

    `verbose`: whether to print progress while building the network

    `tolerate_kinks`: whether to include leaf chains with an apparent
        intermediate minimum in the network construction (if False, such
        chains are excluded rather than treated as single edges)

    `maximum_barrier_height`: only add edges with a barrier lower than this
        (kcal/mol)
    """

    verbose: bool = True
    tolerate_kinks: bool = True
    maximum_barrier_height: float = 1000.0


@dataclass
class RunInputs:
    engine_name: str = "gxtb"
    program: str = "xtb"
    chemcloud_queue: str = None
    write_qcio: bool = False
    print_stdout: bool = False
    nanoreactor_inputs: dict = None

    path_min_method: str = 'NEB'
    path_min_inputs: dict = None

    chain_inputs: dict = None
    gi_inputs: dict = None
    atom_mapping_inputs: dict = None

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
                "disregard_stereochem": False,
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
                "phi": 0.5,
                "drstep": 0.1,

            }
        elif path_method == "NEB-DLF":
            default_kwds = {
                "nstep": 200,
                "min_image": None,
                "min_nebk": 0.01,
                "max_nebk": None,
                "new_minimizer": "no",
                "skip_identical_graphs": True,
                "disregard_stereochem": False,
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
                "disregard_stereochem": False,
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
        elif path_method == "GSM":
            default_kwds = {
                "executable": None,  # falls back to $GSM_EXECUTABLE, then "gsm"
                "nnodes": 9,
                "max_opt_iters": 80,
                "step_opt_iters": 30,
                "conv_tol": 0.0005,
                "add_node_tol": 0.1,
                "ts_final_type": 1,  # 0 = no bond breaking, 1 = bond breaking
                "scaling": 1.0,
                "ssm_dqmax": 0.8,
                "int_thresh": 2.0,
                "min_spacing": 5.0,
                "bond_fragments": 1,
                "initial_opt": 0,
                "final_opt": 150,
                "product_limit": 100.0,
                "timeout": None,
                "keep_workdirs": False,
                "seed_with_geodesic_interpolation": True,
                # Off by default: when on, GSM is killed once a local minimum
                # in the live string has held steady for
                # `early_stop_persistence_window` consecutive updates (only
                # checked once growth has finished, and only when
                # do_elem_step_checks is also True -- see GSM._run_gsm). In
                # practice it cut runs short too often; the elementary-step
                # check on the fully converged string still splits real
                # multi-step paths.
                "early_stop_on_minima": False,
                "early_stop_persistence_window": 3,
                "early_stop_minima_rtol": 0.02,
                "early_stop_minima_min_depth_kcal": 1.0,
                "do_elem_step_checks": True,
                "skip_identical_graphs": True,
                "disregard_stereochem": False,
                "validate_minima_with_hessian": False,
                "hessian_minimum_frequency_cutoff": 0.0,
                "hessian_minima_rescue_displacement": 0.1,
                "verbosity": 1,
            }

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

        if self.atom_mapping_inputs is None:
            self.atom_mapping_inputs = AtomMappingInputs()
        else:
            self.atom_mapping_inputs = AtomMappingInputs(**self.atom_mapping_inputs)

        if self.program_kwds == "":
            # TOML has no native null; to_dict()/save() writes an absent
            # program_kwds as "" so it round-trips through TOML, but that
            # leaves it as the string "" (not None) after RunInputs.open().
            self.program_kwds = None

        if self.program_kwds is None:
            if self.engine_name in {"gxtb", "ase"}:
                # Neither engine uses the qccompute/chemcloud ProgramArgs/qcdata
                # input construct -- gxtb shells out directly, and ASEEngine
                # takes an already-constructed ase.Calculator.
                program_args = None
            elif self.program == "xtb":
                if shutil.which("crest") is not None:
                    self.program = 'crest'
                    program_args = ProgramArgs(
                        model={"method": "gfn2",
                               "basis": "gfn2"},
                        keywords={"threads": 1})
                else:
                    program_args = ProgramArgs(
                        model={"method": "GFN2xTB", "basis": "GFN2xTB"},
                        keywords={})

            elif "terachem" in self.program:
                program_args = ProgramArgs(
                    model={"method": "ub3lyp", "basis": "3-21g"},
                    keywords={})
            else:
                raise ValueError("Need to specify program arguments")

            if self.engine_name in {'chemcloud', 'qccompute'}:
                self.program_kwds = program_args
        elif self.program_kwds is not None and self.engine_name in {'chemcloud', 'qccompute'}:
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
            self.optimizer_kwds = {"name": "cg"}
        elif "name" not in self.optimizer_kwds:
            self.optimizer_kwds["name"] = "cg"

        if self.engine_name in {'chemcloud', 'qccompute'}:
            from mepd.engines.qccompute import QCComputeEngine
            eng = QCComputeEngine(program_args=self.program_kwds,
                             program=self.program,
                             compute_program=self.engine_name,
                             chemcloud_queue=self.chemcloud_queue,
                             write_qcio=self.write_qcio,
                             print_stdout=self.print_stdout,
                             geometry_optimizer_kwds=self.geometry_optimizer_kwds,
                             frozen_atom_indices=self.chain_inputs.frozen_atom_indices,
                             )
        elif self.engine_name == 'ase':
            try:
                from mepd.engines.ase import ASEEngine
            except ImportError as exc:
                raise ImportError(
                    "engine_name='ase' requires the 'ase' extra "
                    "(pip install mepd[ase])."
                ) from exc
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
            try:
                from mepd.engines.gxtb import GXTBCalculator
            except ImportError as exc:
                raise ImportError(
                    "engine_name='gxtb' requires the 'gxtb' extra "
                    "(pip install mepd[gxtb])."
                ) from exc
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

        obj = cls(**data)
        if hasattr(obj.program_kwds, 'files') and obj.program_kwds.files is not None:
            file_keys = obj.program_kwds.files.keys()
            if "ca0" in file_keys and "cb0" in file_keys:
                obj.program_kwds.files['ca0'] = Path(
                    obj.program_kwds.files['ca0']).read_bytes()
                obj.program_kwds.files['cb0'] = Path(
                    obj.program_kwds.files['cb0']).read_bytes()

        return obj

    def to_dict(self) -> dict:
        """Serialize the config actually in effect (engine_name, program,
        path_min_inputs/chain_inputs/gi_inputs/etc.) to a plain, TOML-safe
        dict -- excludes the constructed `engine`/`optimizer` objects
        themselves. Used by both `save()` (write to a TOML file) and the
        CLI's run-settings summary (print what's actually being used).
        """
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
        for key, val in json_dict.items():
            if 'input' in key:
                json_dict[key] = _serialize_input_value(val)
            elif 'program_kwds' in key:
                if val is None:
                    json_dict[key] = ""
                elif isinstance(val, dict):
                    # e.g. the gxtb engine, which never wraps program_kwds
                    # in a ProgramArgs object -- see RunInputs.__post_init__.
                    json_dict[key] = val
                else:
                    d = val.json()
                    d = d.replace("null", "None")
                    json_dict[key] = eval(d)

        return _toml_safe(json_dict)

    def save(self, fp):
        json_dict = self.to_dict()
        with open(fp, "w+") as f:
            f.write(tomli_w.dumps(json_dict))
