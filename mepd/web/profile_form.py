"""A form over a RunInputs TOML profile: a few big choices up front (path
method, engine and level of theory, initial path, optimizer, images) and an
Advanced section that shows only the settings those choices use.

The TOML stays the source of truth. `form(text)` describes the form for a
profile; `apply(text, path, value)` changes one setting and returns the new
TOML plus the new form. Only settings the form shows can be written (each is
typed and checked); everything else in the file is kept as it is. Switching
the path method, engine or program keeps what the user set: settings the
new choice doesn't read stay in the file (listed as unused) instead of being
thrown away. Defaults come from mepd itself, so an empty field means "mepd's
default", and clearing one removes the key so the default applies again.
"""

from __future__ import annotations

import copy
import functools
import math
import re
import tomllib
from typing import Any, Optional

import tomli_w

from mepd.web.workspace import WorkspaceError

# ------------------------------------------------------------------ choices

PATH_METHODS = [
    ("NEB", "NEB", "Nudged elastic band (mepd's own), optionally with a climbing image."),
    ("FNEB", "FSM (FNEB)", "Freezing string method: grows the path from both ends, then relaxes it."),
    ("GSM", "GSM", "Growing string method (molecularGSM binary)."),
    ("GEOMETRIC-NEB", "geomeTRIC NEB", "geomeTRIC's NEB implementation."),
    ("NEB-DLF", "DL-FIND NEB", "DL-FIND's NEB inside TeraChem (needs TeraChem via QCCompute or ChemCloud)."),
]
ENGINES = [
    ("gxtb", "g-xTB (local)", "The g-xTB executable on this machine."),
    ("qccompute", "QCCompute (local program)", "Any program QCCompute can run here: xtb, TeraChem, psi4, …"),
    ("chemcloud", "ChemCloud (remote)", "The same programs, run on ChemCloud."),
    ("mlip", "Machine-learned potential", "AIMNet2, Orb, ANI, MACE-OFF or UMA, chosen by name "
                                          "(each structure's charge and spin reach the model)."),
    ("fairchem", "FAIR-Chem model (e.g. UMA)", "A FAIR-Chem machine-learned potential such as UMA, with all of "
                                               "FAIR-Chem's settings."),
    ("ase", "ASE calculator", "An ASE calculator: MACE, tblite (xTB), Psi4, ORCA, … or your own, by import path."),
]
MLIP_DEFAULTS = {"device": "", "checkpoint": None, "family": "", "geometry_optimizer": "LBFGSLineSearch"}
MLIP_PACKAGES = {"aimnet2": "aimnet", "orb": "orb_models", "ani": "torchani", "mace": "mace", "fairchem": "fairchem"}
PROGRAMS = ["xtb", "terachem", "psi4", "pyscf", "orca", "crest"]
INTERPOLATIONS = [
    ("geodesic", "Geodesic", "Geodesic interpolation in internal coordinates: curved paths that keep bonds sensible."),
    ("idpp", "IDPP", "Image-dependent pair potential: interatomic distances interpolated, images relaxed together "
                     "and evenly spaced (Smidstrup 2014)."),
    ("lst", "LST", "Linear synchronous transit: each image matches linearly interpolated interatomic distances "
                   "(images can be unevenly spaced)."),
    ("linear", "Linear", "Straight lines in Cartesian space (after aligning the two structures). Atoms can pass close."),
]
OPTIMIZERS = [
    ("cg", "Conjugate gradient"), ("lbfgs", "L-BFGS"), ("fire", "FIRE"), ("vpo", "Velocity-projected"),
    ("amg", "Adaptive momentum"), ("adam", "Adam"), ("sgd", "SGD"), ("gd", "Gradient descent"),
]
# Constructor defaults of each chain optimizer (mepd/optimizers/*.py).
OPTIMIZER_DEFAULTS = {
    "cg": {"timestep": 0.5, "max_step_norm": 1.0, "step_up": 1.2, "step_down": 0.5, "min_timestep": None,
           "max_timestep": None, "adaptive_dt": True, "corr_decrease_thre": 0.75, "corr_increase_thre": 0.95,
           "negative_steps_thre": 2, "positive_steps_thre": 15},
    "vpo": {"timestep": 1.0, "activation_tol": 0.1, "zero_vel_count_thre": 5},
    "lbfgs": {"timestep": 1.0, "history_size": 10, "min_curvature": 1e-10},
    "adam": {"timestep": 0.05, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
    "sgd": {"timestep": 0.05, "momentum": 0.0, "dampening": 0.0, "nesterov": False, "max_step_norm": None},
    "gd": {"timestep": 0.05, "weight_decay": 0.0, "max_step_norm": None, "adaptive_dt": False, "step_up": 1.1,
           "step_down": 0.6, "min_timestep": None, "max_timestep": None, "corr_increase_thre": 0.8,
           "corr_decrease_thre": -0.1, "plateau_window": 3, "plateau_rtol": 0.05, "plateau_growth_lag": 1},
    "amg": {"timestep": 0.1, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "max_step_norm": 1.0,
            "min_timestep": 1e-4, "max_timestep": 1.0, "step_up": 1.1, "step_down": 0.6},
    "fire": {"timestep": 0.1, "dt_max": 1.0, "finc": 1.1, "fdec": 0.5, "alpha_start": 0.1, "falpha": 0.99,
             "n_min": 5, "max_step_norm": 1.0},
}
OPTIMIZER_ALIASES = {"conjugate_gradient": "cg", "velocity_projected": "vpo", "stochastic_gradient_descent": "sgd",
                     "gradient_descent": "gd", "deterministic_gradient_descent": "gd", "adaptive_momentum": "amg"}

GXTB_DEFAULTS = {"executable": None, "n_threads": 1, "n_parallel": 0, "hessian_acc": 0.001,
                 "add_gxtb_flag": True, "keep_workdirs": False}
FAIRCHEM_DEFAULTS = {"model": "uma-s-1p2p1", "task": "omol", "device": "", "checkpoint": None,
                     "inference_settings": "default", "batch_size": 32, "geometry_optimizer": "LBFGSLineSearch"}
GEOMOPT_DEFAULTS = {"coordsys": "cart", "maxit": 500, "convergence_set": "GAU_TIGHT"}
GI_DEFAULTS = {"nimages": 10, "friction": 0.001, "nudge": 0.1, "random_seed": 0, "align": True}
CHAIN_DEFAULTS = {"k": 0.1, "delta_k": 0.09, "do_parallel": True, "node_freezing": True, "fraction_freeze": 0.1,
                  "node_rms_thre": 5.0, "node_ene_thre": 5.0, "frozen_atom_indices": ""}
MAPPING_DEFAULTS = {"n_candidates": 200, "metric": "geodesic-distance", "veto_margin": 0.0, "recheck_on_split": False}
TERACHEM_MODEL = {"method": "ub3lyp", "basis": "3-21g"}

# Keys every path method reads.
SHARED_PATH_KEYS = ("skip_identical_graphs", "disregard_stereochem", "do_elem_step_checks", "disable_molecular_graphs",
                    "recursive_split_max_depth")
# Set per calculation by the web forms / `mepd run` flags, so a profile value would be ignored.
CLI_OWNED_PATH_KEYS = {"validate_minima_with_hessian", "hessian_minimum_frequency_cutoff",
                       "hessian_minima_rescue_displacement", "recursive_same_pair_split_limit"}
# Read by a method's code although not in its default dict (aliases, optional keys).
EXTRA_PATH_KEYS = {
    "GEOMETRIC-NEB": {"ncimg", "epsilon", "images", "prefix", "maxg", "avgg", "neb_maxcyc"},
    "NEB-DLF": {"ts_method", "constraints_text", "input_files", "staged_elem_check", "early_stop_two_stage",
                "loose_path_min_inputs", "early_stop_loose_path_min_inputs", "dlf_keywords"},
}
# Unused by mepd (legacy or never read): not shown.
HIDDEN_PATH_KEYS = {"adaptive_segment_ratio", "adaptive_energy_ratio", "adaptive_use_energy", "plateau_exit_window",
                    "plateau_exit_rtol", "tangent_alpha", "v"}
# The settings worth seeing first, per method (the rest follow).
KEY_PATH_FIELDS = {
    "NEB": ["climb", "max_steps", "rms_grad_thre", "ts_grad_thre", "ts_converged_stop", "adaptive_resolution"],
    "FNEB": ["max_grow_iter", "max_min_iter", "grad_tol", "min_images", "tangent", "distance_metric"],
    "GSM": ["nnodes", "seed_with_geodesic_interpolation", "max_opt_iters", "conv_tol", "ts_final_type",
            "early_stop_on_minima", "executable"],
    "GEOMETRIC-NEB": ["max_steps", "rms_grad_thre", "max_rms_grad_thre", "climb", "nebk"],
    "NEB-DLF": ["nstep", "min_nebk", "max_nebk", "min_image", "new_minimizer"],
}
# Types of settings whose mepd default is None (so the default says nothing).
TYPE_HINTS = {"min_image": "int", "max_nebk": "float", "timeout": "float", "min_timestep": "float",
              "max_timestep": "float", "max_step_norm": "float", "executable": "text", "checkpoint": "text",
              "chemcloud_queue": "text", "model_path": "text", "calculator": "text"}
# Smallest sensible value per setting (else: >= 0 unless the default is negative).
MINIMUMS = {"nimages": 3, "max_steps": 1, "nnodes": 3, "n_candidates": 1, "n_threads": 1, "batch_size": 1, "maxit": 1,
            "max_opt_iters": 1, "step_opt_iters": 1, "max_grow_iter": 1, "max_min_iter": 1, "min_images": 2,
            "nstep": 1, "adaptive_max_images": 2, "history_size": 1, "n_min": 1}

OPTIONS = {
    "path_min_inputs.tangent": [("geodesic", "geodesic"), ("linear", "linear")],
    "path_min_inputs.distance_metric": [("GEODESIC", "geodesic"), ("RMSD", "RMSD"), ("LINEAR", "linear")],
    "path_min_inputs.new_minimizer": [("no", "no"), ("yes", "yes")],
    "path_min_inputs.ts_final_type": [(1, "1: bond breaking"), (0, "0: no bond breaking")],
    "geometry_optimizer_kwds.coordsys": [(c, c) for c in ("cart", "tric", "dlc", "hdlc", "prim")],
    "geometry_optimizer_kwds.convergence_set": [(c, c) for c in ("GAU_LOOSE", "GAU", "GAU_TIGHT", "GAU_VERYTIGHT",
                                                                  "NWCHEM_LOOSE", "TURBOMOLE", "INTERFRAG_TIGHT")],
    "ase_engine_kwds.geometry_optimizer": [(o, o) for o in ("LBFGSLineSearch", "LBFGS", "BFGS", "FIRE", "MDMin")],
    "fairchem_engine_kwds.geometry_optimizer": [(o, o) for o in ("LBFGSLineSearch", "LBFGS", "BFGS", "FIRE", "MDMin")],
    "fairchem_engine_kwds.task": [(t, t) for t in ("omol", "omat", "oc20", "odac", "omc")],
    "fairchem_engine_kwds.inference_settings": [("default", "default"), ("turbo", "turbo")],
    "fairchem_engine_kwds.device": [("", "automatic (GPU if available)"), ("cuda", "GPU (cuda)"), ("cpu", "CPU")],
    "mlip_engine_kwds.device": [("", "automatic (GPU if available)"), ("cuda", "GPU (cuda)"), ("cpu", "CPU")],
    "mlip_engine_kwds.family": [("", "from the model name"), *((f, f) for f in ("aimnet2", "orb", "mace", "ani", "fairchem"))],
    "mlip_engine_kwds.geometry_optimizer": [(o, o) for o in ("LBFGSLineSearch", "LBFGS", "BFGS", "FIRE", "MDMin")],
    "program_kwds.device": [("cuda", "GPU (cuda)"), ("cpu", "CPU")],
    "atom_mapping_inputs.metric": [(m, m) for m in ("geodesic-distance", "path-rmsd", "gi-energy", "endpoint-rmsd")],
}

HELP = {
    "climb": "Climbing image: the highest image climbs to the saddle point.",
    "max_steps": "Most optimization steps before stopping.",
    "rms_grad_thre": "Converged when the RMS perpendicular gradient falls below this (Eh/bohr).",
    "max_rms_grad_thre": "…and the largest image's RMS gradient falls below this.",
    "ts_grad_thre": "Gradient threshold for the TS guess (climbing image).",
    "ts_spring_thre": "Spring-force threshold at the TS guess.",
    "ts_converged_stop": "Stop as soon as the TS region has converged.",
    "en_thre": "Energy change threshold between steps (Eh).",
    "barrier_thre": "Barriers below this (kcal/mol) don't count as a separate step.",
    "early_stop_force_thre": "Force below which the run may stop early to check for intermediate minima.",
    "adaptive_resolution": "Add images where the path needs them (up to 'adaptive max images').",
    "use_geodesic_tangent": "Use the geodesic path's tangent instead of the image-difference tangent.",
    "do_elem_step_checks": "Check whether a converged path is one elementary step (else split it).",
    "skip_identical_graphs": "Skip path searches between structures with the same bond graph.",
    "disregard_stereochem": "Ignore stereochemistry when comparing structures.",
    "disable_molecular_graphs": "Don't build bond graphs (no graph-based checks).",
    "nnodes": "Nodes along the string. Ignored when seeded from the initial path (the string then has 'Images' nodes).",
    "seed_with_geodesic_interpolation": "Start GSM from the initial path (built with the chosen Initial path method) "
                                        "instead of growing the string from the ends.",
    "max_opt_iters": "GSM optimization cycles.",
    "conv_tol": "GSM convergence tolerance.",
    "ts_final_type": "How GSM treats the final TS: 1 if a bond breaks, 0 otherwise.",
    "early_stop_on_minima": "Stop GSM early when the string develops an intermediate minimum.",
    "executable": "Path to the program; empty uses $GSM_EXECUTABLE / $GXTB_EXECUTABLE or the one on PATH.",
    "grad_tol": "FSM gradient tolerance (Eh/bohr).",
    "max_grow_iter": "Growth iterations (FSM grows the path from both ends).",
    "max_min_iter": "Minimization iterations per growth step.",
    "min_images": "Fewest images FSM keeps.",
    "tangent": "Tangent used while growing.",
    "distance_metric": "How distances between images are measured.",
    "nebk": "Spring constant.",
    "nstep": "DL-FIND NEB steps.",
    "min_nebk": "Smallest spring constant.",
    "max_nebk": "Largest spring constant (empty: same as the smallest).",
    "min_image": "Images (empty: the chain length).",
    "n_threads": "Threads per g-xTB call. Keep 1 and parallelize across calls instead.",
    "n_parallel": "g-xTB calls at once (0: all cores).",
    "hessian_acc": "SCC accuracy for numerical Hessians (smaller = tighter).",
    "coordsys": "Coordinate system for geomeTRIC optimizations.",
    "maxit": "geomeTRIC's maximum iterations.",
    "convergence_set": "geomeTRIC convergence criteria.",
    "friction": "Geodesic smoothing friction.",
    "nudge": "Geodesic nudge.",
    "align": "Align the two structures (and images) before interpolating.",
    "random_seed": "Random seed for the interpolation.",
    "k": "Spring constant between images.",
    "delta_k": "Spring-constant spread (energy-weighted springs).",
    "node_freezing": "Freeze converged images to save engine calls.",
    "fraction_freeze": "Fraction of the path that may be frozen.",
    "do_parallel": "Compute images in parallel.",
    "frozen_atom_indices": "Atoms held fixed (space-separated indices, from 0).",
    "n_candidates": "Atom-mapping candidates scored per mechanism.",
    "metric": "How candidate atom mappings are scored.",
    "veto_margin": "A remapping must beat 'don't reindex' by more than this.",
    "model": "Pretrained FAIR-Chem model name (e.g. uma-s-1p2p1, uma-m-1p1).",
    "task": "Which UMA head: omol for molecules (reads charge and multiplicity).",
    "checkpoint": "A local checkpoint file used instead of the named model.",
    "inference_settings": "FAIR-Chem inference settings (turbo trades some accuracy for speed).",
    "batch_size": "Structures per forward pass.",
    "calculator": "Import path of an ASE calculator class or factory, e.g. my_package.calculator:MyCalculator.",
    "calculator_kwds": "Arguments for the calculator, written as TOML key = value pairs separated by commas, "
                       'e.g. model = "medium", device = "cpu".',
    "family": "Only for a local checkpoint: which kind of model it is.",
    "geometry_optimizer": "ASE optimizer for minimizations (TS optimizations use Sella).",
}

_WORDS = {"thre": "threshold", "tol": "tolerance", "rtol": "relative tolerance", "ts": "TS", "rms": "RMS",
          "gi": "GI", "nebk": "NEB k", "opt": "optimization", "iters": "iterations", "n": "number of",
          "acc": "accuracy", "dt": "Δt", "ene": "energy", "neb": "NEB", "grad": "gradient"}
LABELS = {"nimages": "Images", "nnodes": "Nodes", "k": "Spring constant k", "delta_k": "Spring spread Δk",
          "n_threads": "Threads per call", "n_parallel": "Calls at once", "maxit": "Max iterations",
          "n_candidates": "Candidates", "n_min": "Steps before speeding up", "nstep": "Steps"}


def _label(key: str) -> str:
    if key in LABELS:
        return LABELS[key]
    text = " ".join(_WORDS.get(w, w) for w in key.split("_") if w)
    return text[:1].upper() + text[1:]


# ------------------------------------------------------------------ helpers

def _parse(text: str) -> dict:
    try:
        return tomllib.loads((text or "").lstrip("﻿"))
    except tomllib.TOMLDecodeError as exc:
        raise WorkspaceError(f"not valid TOML: {exc}") from None


def _dump(data: dict) -> str:
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if v is not None}
        return x
    return tomli_w.dumps(clean(data))


def _has_comments(text: str) -> bool:
    """A `#` outside quoted strings anywhere (whole-line or inline)."""
    no_strings = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:[^"\\\n]|\\.)*"|\'[^\'\n]*\'', "", text or "")
    return "#" in no_strings


def _get(data: dict, path: str, default=None):
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _table(data: dict, key: str) -> dict:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _set(data: dict, path: str, value) -> None:
    """Set a validated setting (None removes it)."""
    parts = path.split(".")
    cur = data
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    if value is None:
        cur.pop(parts[-1], None)
    else:
        cur[parts[-1]] = value


def normalized_method(value: Optional[str]) -> str:
    from mepd.inputs import _normalized_path_method

    return _normalized_path_method(value or "NEB")


@functools.lru_cache(maxsize=16)
def method_defaults(method: str) -> dict:
    """Every path_min_inputs default mepd uses for this method."""
    from mepd.inputs import RunInputs
    from mepd.nodes.node import StructureNode

    # Building RunInputs sets a process-wide molecular-graph switch; the web
    # server must not have it changed under it.
    saved = StructureNode._global_disable_molecular_graphs
    try:
        return dict(vars(RunInputs(path_min_method=method).path_min_inputs))
    finally:
        StructureNode.set_global_disable_molecular_graphs(saved)


def _known_method(method: str) -> bool:
    return method in {k for k, _, _ in PATH_METHODS}


def _used_path_keys(method: str) -> set:
    return set(method_defaults(method)) | set(SHARED_PATH_KEYS) | EXTRA_PATH_KEYS.get(method, set()) | CLI_OWNED_PATH_KEYS


def _optimizer_name(data: dict) -> str:
    name = str(_get(data, "optimizer_kwds.name", "cg") or "cg").lower()
    return OPTIMIZER_ALIASES.get(name, name)


def _program_default_model(program: str) -> Optional[dict]:
    return dict(TERACHEM_MODEL) if "terachem" in (program or "").lower() else None


# ------------------------------------------------------------------ fields

def _field(data: dict, path: str, default, *, key: bool = False, label: Optional[str] = None,
           help: Optional[str] = None, kind: Optional[str] = None, placeholder: Optional[str] = None) -> dict:
    name = path.rsplit(".", 1)[-1]
    options = OPTIONS.get(path)
    if kind is None:
        if options:
            kind = "select"
        elif isinstance(default, bool):
            kind = "bool"
        elif isinstance(default, int):
            kind = "int"
        elif isinstance(default, float):
            kind = "float"
        else:
            kind = TYPE_HINTS.get(name, "text")
    value = _get(data, path)
    if kind == "inline" and isinstance(value, dict):
        value = _inline(value)
    return {"path": path, "label": label or _label(name), "type": kind, "value": value,
            "default": default, "options": [{"value": v, "label": l} for v, l in options] if options else None,
            "help": help if help is not None else HELP.get(name, ""), "key": key, "placeholder": placeholder}


def _inline(table: dict) -> str:
    return ", ".join(tomli_w.dumps({k: v}).strip() for k, v in table.items()) if table else ""


# ------------------------------------------------------------------ ASE / MLIP choices

CUSTOM = "custom"


def _calc_spec(data: dict):
    from mepd.engines.ase_calculators import lookup

    name = _get(data, "ase_engine_kwds.calculator")
    return lookup(name) if isinstance(name, str) and name.strip() else None


def _calc_key(name: str) -> Optional[str]:
    from mepd.engines.ase_calculators import CALCULATORS, lookup

    spec = lookup(name)
    return next((k for k, s in CALCULATORS.items() if s is spec), None) if spec else None


def _ase_calculator_choice(data: dict):
    """The Calculator choice, whether it's an import path (custom), and what's
    wrong with it (None when it can run here)."""
    import importlib.util

    from mepd.engines.ase_calculators import CALCULATORS, installed, resolve_path

    raw = _get(data, "ase_engine_kwds.calculator")
    name = raw.strip() if isinstance(raw, str) else ""
    key = _calc_key(name) if name else None
    options = [("", "Choose a calculator…", "Pick one of these, or Other for any ASE calculator by import path.")]
    for k, spec in CALCULATORS.items():
        ok = installed(spec)
        how = f" Install: {spec.install}." if spec.install and not ok else ""
        options.append((k, f"{k}: {spec.summary}{'' if ok else ' (not installed here)'}",
                        f"{spec.summary}. Charge and spin: {spec.charge_and_spin}.{how}"))
    options.append((CUSTOM, "Other ASE calculator (import path)",
                    "Any ASE calculator class or factory, as package.module:ClassName."))
    custom = raw is not None and key is None
    value = key or (CUSTOM if custom else "")
    problem = None
    if raw is None:
        problem = "The ASE engine needs a calculator: choose one."
    elif custom and not name:
        problem = "Write the calculator's import path (package.module:ClassName)."
    elif custom:
        try:
            path = resolve_path(name)
            package = path.split(":", 1)[0].split(".", 1)[0]
            if importlib.util.find_spec(package) is None:
                problem = f"{package} (from {name}) is not importable in this environment."
        except (ValueError, ImportError) as exc:
            problem = str(exc)
    elif not installed(CALCULATORS[key]):
        spec = CALCULATORS[key]
        problem = f"{key} is not installed on this machine: {spec.install}."
    return _choice("calculator", "Calculator", value, options, "The ASE calculator giving energies and forces."), \
        custom, problem


def _mlip_model_choice(data: dict):
    import importlib.util

    from mepd.engines.mlip import MODELS

    model = str(_get(data, "mlip_engine_kwds.model") or "")
    options = [("", "Choose a model…", "Run `mepd models` for what each covers.")] if not model else []
    for name, spec in MODELS.items():
        ok = importlib.util.find_spec(MLIP_PACKAGES.get(spec.family, spec.family)) is not None
        flags = [s for s, on in (("Hugging Face access needed", spec.gated), ("not installed here", not ok)) if on]
        options.append((name, f"{name}{' (' + '; '.join(flags) + ')' if flags else ''}",
                        f"{spec.summary}. Elements: {spec.elements or 'see mepd models'}. "
                        f"Charge and spin: {spec.charge_and_spin}.{'' if ok else ' Install: ' + spec.install + '.'}"))
    problem = None
    spec = MODELS.get(model)
    if _get(data, "mlip_engine_kwds.calculator"):
        problem = None
    elif not model:
        problem = "The machine-learned potential needs a model: choose one."
    elif spec and importlib.util.find_spec(MLIP_PACKAGES.get(spec.family, spec.family)) is None:
        problem = f"{model} is not installed on this machine: {spec.install}."
    return _choice("mlip_model", "Model", model, options, "The machine-learned potential."), problem


def _choice(key: str, label: str, value, options: list, help: str = "", disabled: Optional[dict] = None) -> dict:
    disabled = disabled or {}
    if value not in {v for v, _, _ in options}:       # show a value the form doesn't know rather than hide it
        options = options + [(value, f"{value} (not recognized)", "")]
    return {"key": key, "label": label, "value": value, "help": help,
            "options": [{"value": v, "label": l, "help": h, "disabled": disabled.get(v)} for v, l, h in options]}


def _interp(data: dict) -> str:
    return str(_get(data, "chain_inputs.interpolation", "geodesic") or "geodesic").strip().lower()


# ------------------------------------------------------------------ form

def form(text: str) -> dict:
    data = _parse(text)
    issues, notes = [], []
    path_issues = []   # the subset only a path search is affected by
    for section in ("path_min_inputs", "chain_inputs", "gi_inputs", "optimizer_kwds", "atom_mapping_inputs",
                    "gxtb_engine_kwds", "fairchem_engine_kwds", "ase_engine_kwds", "mlip_engine_kwds",
                    "geometry_optimizer_kwds"):
        if section in data and not isinstance(data[section], dict):
            issues.append(f"[{section}] is not a table ({type(data[section]).__name__}); fix it on the TOML tab.")
    method = normalized_method(data.get("path_min_method"))
    if not _known_method(method):
        issues.append(f"Unknown path method {data.get('path_min_method')!r}: pick one here.")
        path_issues.append(issues[-1])
    engine = str(data.get("engine_name") or "gxtb").lower()
    program = str(data.get("program") or "xtb")
    interp = _interp(data)
    if interp not in {k for k, _, _ in INTERPOLATIONS}:
        issues.append(f"Unknown initial path {interp!r}: pick one here.")
    if _get(data, "chain_inputs.use_geodesic_interpolation") is False and "interpolation" not in _table(data, "chain_inputs"):
        issues.append("chain_inputs.use_geodesic_interpolation = false has been removed and makes this profile fail "
                      "to load: pick the Initial path here (Linear keeps its old meaning).")
    opt = _optimizer_name(data)

    tc_ok = engine in ("qccompute", "chemcloud") and "terachem" in program.lower()
    tc_reason = None if tc_ok else "needs TeraChem through QCCompute or ChemCloud"
    if method == "NEB-DLF" and not tc_ok:
        issues.append(f"DL-FIND NEB {tc_reason}: pick another path method or switch the engine to TeraChem.")
        path_issues.append(issues[-1])

    basic = [
        _choice("path_method", "Path method", method, PATH_METHODS, "How the minimum-energy path is found.",
                disabled={"NEB-DLF": tc_reason} if tc_reason and method != "NEB-DLF" else None),
        _choice("engine", "Engine", engine, ENGINES, "Where energies and gradients come from."),
    ]
    level_fields = []
    if engine in ("qccompute", "chemcloud"):
        basic.append(_choice("program", "Program", program, [(p, p, "") for p in PROGRAMS],
                             "The quantum chemistry program."))
        if "terachem" in program.lower():
            ph = ("ub3lyp", "3-21g")
        elif program == "xtb":
            ph = ("mepd default: GFN2-xTB (CREST's gfn2 when crest is installed)", "mepd default")
        else:
            ph = ("required, e.g. b3lyp", "required, e.g. def2-svp")
        level_fields = [
            _field(data, "program_kwds.model.method", None, key=True, label="Method", kind="text",
                   help="e.g. ub3lyp, wb97x-d3, GFN2xTB", placeholder=ph[0]),
            _field(data, "program_kwds.model.basis", None, key=True, label="Basis", kind="text",
                   help="e.g. 6-31gs, def2-svp", placeholder=ph[1]),
        ]
        if program != "xtb" and "terachem" not in program.lower() and not _get(data, "program_kwds.model.method"):
            issues.append(f"{program} needs a method and basis (Level of theory).")
    if engine == "ase":
        choice, custom, problem = _ase_calculator_choice(data)
        basic.append(choice)
        if custom:
            level_fields = [_field(data, "ase_engine_kwds.calculator", None, key=True, label="Import path",
                                   kind="text", placeholder="package.module:ClassName")]
        if problem and not (program == "omol25" and not _table(data, "ase_engine_kwds").get("calculator")):
            issues.append(problem)
    elif engine == "mlip":
        choice, problem = _mlip_model_choice(data)
        basic.append(choice)
        if problem:
            issues.append(problem)
    basic.append(_choice("interpolation", "Initial path", interp, INTERPOLATIONS,
                         "How the first path between the two structures is built, and every sub-path a recursive "
                         "split creates." + (" FSM grows its own path from the two ends, so here this only sets "
                                             "how the ends are aligned." if method == "FNEB" else "")))
    uses_optimizer = method in ("NEB", "FNEB")
    if uses_optimizer:
        basic.append(_choice("optimizer", "Chain optimizer", opt, [(k, l, "") for k, l in OPTIMIZERS],
                             "How the path's images are moved each step."))
    images = _field(data, "gi_inputs.nimages", GI_DEFAULTS["nimages"], key=True, label="Images",
                    help="Images along the path (GSM: its nodes, when seeded from the initial path).")

    groups = []
    defaults = method_defaults(method) if _known_method(method) else {}
    keys = [k for k in KEY_PATH_FIELDS.get(method, []) if k in defaults]
    rest = [k for k in defaults if k not in keys and k not in HIDDEN_PATH_KEYS and k not in CLI_OWNED_PATH_KEYS
            and not isinstance(defaults[k], (dict, list))]
    title = dict((k, l) for k, l, _ in PATH_METHODS).get(method, method)
    groups.append({"id": "path", "title": f"{title} settings",
                   "note": "Hessian checks of minima and the same-pair split limit are set per calculation, in the calculation forms.",
                   "fields": [_field(data, f"path_min_inputs.{k}", defaults[k], key=True) for k in keys]
                   + [_field(data, f"path_min_inputs.{k}", defaults[k]) for k in rest]})
    if engine == "gxtb":
        groups.append({"id": "engine", "title": "g-xTB",
                       "fields": [_field(data, f"gxtb_engine_kwds.{k}", v, key=k in ("n_threads", "n_parallel"))
                                  for k, v in GXTB_DEFAULTS.items()]})
    elif engine in ("qccompute", "chemcloud"):
        fields = [_field(data, f"geometry_optimizer_kwds.{k}", v) for k, v in GEOMOPT_DEFAULTS.items()]
        if engine == "chemcloud":
            fields.insert(0, _field(data, "chemcloud_queue", None, label="ChemCloud queue", kind="text",
                                    help="Queue to submit to (empty: the default queue)."))
        groups.append({"id": "engine", "title": f"{'ChemCloud' if engine == 'chemcloud' else 'QCCompute'} and geomeTRIC",
                       "note": "Program keywords (e.g. TeraChem settings) go in [program_kwds.keywords] on the TOML tab.",
                       "fields": fields})
    elif engine == "fairchem":
        groups.append({"id": "engine", "title": "FAIR-Chem model", "fields": [
            _field(data, f"fairchem_engine_kwds.{k}", v, key=k in ("model", "task", "device"))
            for k, v in FAIRCHEM_DEFAULTS.items()]})
    elif engine == "mlip":
        groups.append({"id": "engine", "title": "Machine-learned potential",
                       "note": "Run `mepd models` for what each model covers. A local model file: set Checkpoint "
                               "and Family.",
                       "fields": [_field(data, f"mlip_engine_kwds.{k}", v, key=k == "device")
                                  for k, v in MLIP_DEFAULTS.items()]})
    elif engine == "ase":
        fields = [_field(data, "ase_engine_kwds.calculator_kwds", None, key=True, label="Calculator arguments",
                         kind="inline", placeholder=_calc_spec(data).kwds_hint if _calc_spec(data) else "none"),
                  _field(data, "ase_engine_kwds.geometry_optimizer", "LBFGSLineSearch", label="Geometry optimizer")]
        if not _table(data, "ase_engine_kwds").get("calculator"):
            if program == "omol25":
                notes.append('This profile uses the older OMol25 route (program = "omol25"); the FAIR-Chem engine '
                             "runs the same model with more options.")
                fields += [_field(data, "program_kwds.model_path", None, label="Model file", kind="text",
                                  help="The FAIR-Chem checkpoint (empty: mepd's default model)."),
                           _field(data, "program_kwds.device", "cuda", label="Device")]
        groups.append({"id": "engine", "title": "ASE calculator", "fields": fields})
    gi = [_field(data, f"gi_inputs.{k}", v) for k, v in GI_DEFAULTS.items() if k != "nimages"]
    if interp != "geodesic":
        gi = [f for f in gi if f["path"] == "gi_inputs.align"]
    groups.append({"id": "interp", "title": "Initial path", "fields": gi,
                   "note": None if interp == "geodesic" else "Only alignment applies to this method."})
    if uses_optimizer:
        groups.append({"id": "optimizer", "title": f"{dict(OPTIMIZERS).get(opt, opt)} optimizer",
                       "fields": [_field(data, f"optimizer_kwds.{k}", v) for k, v in OPTIMIZER_DEFAULTS.get(opt, {}).items()]})
    groups.append({"id": "chain", "title": "Chain", "fields": [_field(data, f"chain_inputs.{k}", v) for k, v in CHAIN_DEFAULTS.items()]})
    groups.append({"id": "mapping", "title": "Atom mapping",
                   "fields": [_field(data, f"atom_mapping_inputs.{k}", v) for k, v in MAPPING_DEFAULTS.items()]})

    unused = sorted(k for k in _table(data, "path_min_inputs")
                    if _known_method(method) and k not in _used_path_keys(method))
    idle = [s for s, owner in (("gxtb_engine_kwds", {"gxtb"}), ("fairchem_engine_kwds", {"fairchem"}),
                               ("mlip_engine_kwds", {"mlip"}),
                               ("ase_engine_kwds", {"ase"}), ("program_kwds", {"qccompute", "chemcloud", "ase"}))
            if _table(data, s) and engine not in owner]
    if idle:
        notes.append(f"Kept but not used by this engine: {', '.join('[' + s + ']' for s in idle)} "
                     "(switching back uses them again).")
    return {"basic": basic, "level": level_fields, "images": images, "groups": groups, "issues": issues,
            "path_issues": path_issues,
            "notes": notes, "unused": unused, "has_comments": _has_comments(text)}


# ------------------------------------------------------------------ apply

def _fields_by_path(f: dict) -> dict:
    out = {f["images"]["path"]: f["images"]}
    out.update({x["path"]: x for x in f["level"]})
    for g in f["groups"]:
        out.update({x["path"]: x for x in g["fields"]})
    return out


def _coerce(field: dict, value):
    """Type and check a submitted value for `field`; None clears it."""
    label, kind, default = field["label"], field["type"], field["default"]
    if value is None or (isinstance(value, str) and not value.strip() and kind != "select"):
        return None
    name = field["path"].rsplit(".", 1)[-1]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
        raise WorkspaceError(f"{label}: expected on/off, got {value!r}")
    if kind == "inline":
        if isinstance(value, dict):
            return value
        try:
            return tomllib.loads(f"t = {{{value}}}")["t"]
        except tomllib.TOMLDecodeError as exc:
            raise WorkspaceError(f'{label}: write key = value pairs separated by commas, '
                                 f'e.g. model = "medium", device = "cpu" ({exc})') from None
    if kind == "select":
        for opt in field["options"] or []:
            if str(opt["value"]) == str(value):
                return None if opt["value"] == "" else opt["value"]
        raise WorkspaceError(f"{label}: {value!r} is not one of the choices")
    if kind in ("int", "float"):
        if isinstance(value, bool):
            raise WorkspaceError(f"{label}: expected a number")
        try:
            number = float(str(value).strip())
        except ValueError:
            raise WorkspaceError(f"{label}: {value!r} is not a number") from None
        if not math.isfinite(number):
            raise WorkspaceError(f"{label}: must be a finite number")
        if kind == "int":
            if number != int(number):
                raise WorkspaceError(f"{label}: must be a whole number")
            number = int(number)
            if abs(number) > 10 ** 9:
                raise WorkspaceError(f"{label}: {number} is too large")
        lowest = MINIMUMS.get(name)
        if lowest is None and not (isinstance(default, (int, float)) and not isinstance(default, bool) and default < 0):
            lowest = 0
        if lowest is not None and number < lowest:
            raise WorkspaceError(f"{label}: must be at least {lowest}")
        return number
    return str(value).strip()


def apply(text: str, path: str, value) -> dict:
    """Change one setting and return {"text": new TOML, "form": new form}."""
    new = copy.deepcopy(_parse(text))
    if path == "path_method":
        _apply_method(new, value)
    elif path == "engine":
        engine = str(value or "").strip().lower() if isinstance(value, str) else ""
        if engine not in {k for k, _, _ in ENGINES}:
            raise WorkspaceError(f"unknown engine {value!r}")
        new["engine_name"] = engine
        # Nothing else is removed: engines that don't read program /
        # program_kwds ignore them, so switching back keeps the level.
        if engine in ("qccompute", "chemcloud") and str(new.get("program") or "") in ("", "omol25"):
            new["program"] = "xtb"
        if engine == "mlip" and not _get(new, "mlip_engine_kwds.model") and not _get(new, "mlip_engine_kwds.calculator"):
            _set(new, "mlip_engine_kwds.model", "aimnet2-rxn")   # open, and trained for reactions
    elif path == "program":
        program = value.strip() if isinstance(value, str) else ""
        if not program:
            raise WorkspaceError("program: choose a program")
        old = str(new.get("program") or "")
        new["program"] = program
        kw = new.get("program_kwds") if isinstance(new.get("program_kwds"), dict) else {}
        if not kw.get("model") or kw.get("model") == _program_default_model(old):
            # No level, or just the old program's default: use the new program's.
            fresh = _program_default_model(program)
            if fresh:
                kw["model"] = fresh
            else:
                kw.pop("model", None)
        if kw:
            new["program_kwds"] = kw
        else:
            new.pop("program_kwds", None)
    elif path == "calculator":
        from mepd.engines.ase_calculators import CALCULATORS

        name = value.strip() if isinstance(value, str) else ""
        if name == CUSTOM:
            current = _get(new, "ase_engine_kwds.calculator")
            if not isinstance(current, str) or _calc_key(current):
                _set(new, "ase_engine_kwds.calculator", "")    # an empty import path, to be written
        elif name in CALCULATORS:
            if name != _calc_key(str(_get(new, "ase_engine_kwds.calculator") or "")):
                _set(new, "ase_engine_kwds.calculator", name)
                _set(new, "ase_engine_kwds.calculator_kwds", None)   # the old calculator's arguments
        else:
            raise WorkspaceError(f"unknown calculator {value!r}: pick one, or Other for an import path")
    elif path == "mlip_model":
        from mepd.engines.mlip import MODELS

        if value not in MODELS:
            raise WorkspaceError(f"unknown model {value!r} (run `mepd models` for the list)")
        _set(new, "mlip_engine_kwds.model", value)
    elif path == "interpolation":
        method = value.strip().lower() if isinstance(value, str) else ""
        if method not in {k for k, _, _ in INTERPOLATIONS}:
            raise WorkspaceError(f"unknown initial path {value!r}")
        _set(new, "chain_inputs.interpolation", method)
        _set(new, "chain_inputs.use_geodesic_interpolation", None)   # the removed switch
    elif path == "optimizer":
        name = OPTIMIZER_ALIASES.get(str(value).lower(), str(value).lower())
        if name not in OPTIMIZER_DEFAULTS:
            raise WorkspaceError(f"unknown optimizer {value!r}")
        if name != _optimizer_name(new):
            new["optimizer_kwds"] = {"name": name}   # each optimizer takes different settings
    elif path == "remove_unused":
        method = normalized_method(new.get("path_min_method"))
        if _known_method(method):
            used = _used_path_keys(method)
            new["path_min_inputs"] = {k: v for k, v in _table(new, "path_min_inputs").items() if k in used}
    else:
        field = _fields_by_path(form(_dump(new))).get(path)
        if field is None:
            raise WorkspaceError(f"{path!r} is not a setting on this form (edit it on the TOML tab)")
        _set(new, path, _coerce(field, value))
    out = _dump(new)
    return {"text": out, "form": form(out)}


def _apply_method(new: dict, value) -> None:
    method = normalized_method(value) if isinstance(value, str) and value.strip() else ""
    if not _known_method(method):
        raise WorkspaceError(f"unknown path method {value!r}")
    engine = str(new.get("engine_name") or "gxtb").lower()
    if method == "NEB-DLF" and not (engine in ("qccompute", "chemcloud")
                                    and "terachem" in str(new.get("program") or "").lower()):
        raise WorkspaceError("DL-FIND NEB needs TeraChem through QCCompute or ChemCloud")
    old_method = normalized_method(new.get("path_min_method"))
    old_defaults = method_defaults(old_method) if _known_method(old_method) else {}
    new_defaults = method_defaults(method)
    new_used = _used_path_keys(method)
    kept = {}
    for k, v in _table(new, "path_min_inputs").items():
        at_old_default = k in old_defaults and v == old_defaults[k]
        if at_old_default and k in new_defaults and new_defaults[k] != v:
            continue    # the old method's default: let the new method's default apply
        if at_old_default and k not in new_used:
            continue    # an untouched setting only the old method reads
        kept[k] = v     # the user's value (settings the new method ignores show as unused)
    new["path_min_method"] = method
    new["path_min_inputs"] = kept
