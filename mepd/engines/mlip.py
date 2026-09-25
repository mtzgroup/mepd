"""Machine-learned interatomic potentials as a level of theory.

Pick a model by name -- the only thing to change to switch models:

    engine_name = "mlip"
    [mlip_engine_kwds]
    model = "aimnet2-rxn"        # see `mepd models` for every name
    device = "cuda"              # optional; defaults to CUDA when available

A local model file of a known family:

    [mlip_engine_kwds]
    model = "my-model"
    family = "aimnet2"           # or "orb", "fairchem", "mace", "ani"
    checkpoint = "/path/to/model.pt"

Anything else with an ASE calculator:

    [mlip_engine_kwds]
    calculator = "my_package.module:MyCalculator"   # a class or factory
    calculator_kwds = { some_option = 1 }

Every structure's charge and spin multiplicity reach the model through
`atoms.info["charge"]` / `atoms.info["spin"]`. The model loads on first use,
is shared by copies of the engine and is never pickled (GSM's engine server
reloads it), and each optimization gets its own calculator wrapper around
the shared model.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

from mepd.engines.ase import AVAIL_OPTS, ASEEngine

UMA_ACCESS_HELP = (
    "The FAIR-Chem models (UMA, eSEN) are gated on Hugging Face. To use one:\n"
    "  1. Open https://huggingface.co/{repo} while logged in and request access\n"
    "     (accept the license; approval can take a while).\n"
    "  2. Log in on this machine: `hf auth login`.\n"
    "Or switch to a model that needs no access, e.g. mlip_engine_kwds.model = \"aimnet2-rxn\"\n"
    "or \"orb-v3-conservative-omol\" (run `mepd models` for the list), or point\n"
    "`checkpoint` at a local model file."
)


@dataclass(frozen=True)
class MLIPSpec:
    family: str
    summary: str
    install: str
    gated: bool = False
    charge_and_spin: str = "charge and multiplicity"
    elements: str = ""
    options: dict = field(default_factory=dict)


def _dev(device):
    if device:
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# family -> (load(name, checkpoint, device, options) -> model, wrap(model) -> fresh ASE calculator)
def _aimnet2_load(name, checkpoint, device, options):
    from aimnet.calculators import AIMNet2Calculator

    return AIMNet2Calculator(checkpoint or options.get("registry_name", name), device=_dev(device))


def _aimnet2_wrap(model):
    from aimnet.calculators.aimnet2ase import AIMNet2ASE

    return AIMNet2ASE(model)


def _orb_load(name, checkpoint, device, options):
    import orb_models.forcefield.pretrained as pretrained

    loader = getattr(pretrained, options.get("loader", name.replace("-", "_")))
    kwds = {"device": _dev(device), "precision": options.get("precision", "float32-high")}
    if checkpoint:
        kwds["weights_path"] = checkpoint
    return loader(**kwds)  # (model, atoms_adapter)


def _orb_wrap(model):
    from orb_models.forcefield.inference.calculator import ORBCalculator

    regressor, adapter = model
    return ORBCalculator(regressor, atoms_adapter=adapter, device=next(regressor.parameters()).device)


def _ani_load(name, checkpoint, device, options):
    import torchani

    model = torchani.models.ANI2x(periodic_table_index=True)
    return model.to(_dev(device))


def _ani_wrap(model):
    calc = model.ase()
    calculate = calc.calculate

    def calculate_with_energy(*args, **kwargs):
        # TorchANI stores only free_energy when forces are requested, so ASE
        # trajectories lose the energy; for a potential they are the same.
        calculate(*args, **kwargs)
        if "energy" not in calc.results and "free_energy" in calc.results:
            calc.results["energy"] = calc.results["free_energy"]

    calc.calculate = calculate_with_energy
    return calc


def _mace_load(name, checkpoint, device, options):
    if checkpoint:
        return ("file", checkpoint, _dev(device))
    return ("off", options.get("size", "medium"), _dev(device))


def _mace_wrap(model):
    kind, what, device = model
    if kind == "file":
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=what, device=device, default_dtype="float64")
    from mace.calculators import mace_off

    return mace_off(model=what, device=device, default_dtype="float64")


FAMILIES: dict[str, tuple[Callable, Callable]] = {
    "aimnet2": (_aimnet2_load, _aimnet2_wrap),
    "orb": (_orb_load, _orb_wrap),
    "ani": (_ani_load, _ani_wrap),
    "mace": (_mace_load, _mace_wrap),
}

_AIMNET_ELEMENTS = "H B C N O F Si P S Cl As Se Br I"
_ORB_INSTALL = "pip install orb-models  (or the `orb` extra; not in the same environment as fairchem-core)"
MODELS: dict[str, MLIPSpec] = {
    "aimnet2-rxn": MLIPSpec("aimnet2", "AIMNet2 trained for reactions (wB97M-D3 level)",
                            'pip install "aimnet[ase]"', elements="H C N O",
                            charge_and_spin="closed-shell, any charge"),
    "aimnet2": MLIPSpec("aimnet2", "AIMNet2, general organic/main-group molecules (wB97M-D3)",
                        'pip install "aimnet[ase]"', elements=_AIMNET_ELEMENTS,
                        charge_and_spin="closed-shell, any charge"),
    "aimnet2-nse": MLIPSpec("aimnet2", "AIMNet2 with explicit spin: radicals and open-shell species",
                            'pip install "aimnet[ase]"', elements=_AIMNET_ELEMENTS),
    "aimnet2-2025": MLIPSpec("aimnet2", "AIMNet2 2025 release (B97-3c)", 'pip install "aimnet[ase]"',
                             elements=_AIMNET_ELEMENTS, charge_and_spin="closed-shell, any charge"),
    "orb-v3-conservative-omol": MLIPSpec("orb", "Orb v3 trained on OMol25 (the UMA training data), conservative forces",
                                         _ORB_INSTALL, elements="most of the periodic table"),
    "orb-v3-direct-omol": MLIPSpec("orb", "Orb v3 on OMol25, direct (non-conservative) forces -- faster, less smooth",
                                   _ORB_INSTALL, elements="most of the periodic table"),
    "ani-2x": MLIPSpec("ani", "ANI-2x (wB97X/6-31G*)", "pip install torchani",
                       elements="H C N O F S Cl", charge_and_spin="neutral closed-shell only"),
    "mace-off": MLIPSpec("mace", "MACE-OFF23 organic force field (size: small/medium/large via options)",
                         "pip install mace-torch  (or the `mace` extra; not in the same environment as fairchem-core)",
                         elements="H C N O F P S Cl Br I", charge_and_spin="neutral closed-shell only"),
}
for _name, _repo in (("uma-s-1p2p1", "facebook/UMA"), ("uma-s-1p2", "facebook/UMA"),
                     ("uma-s-1p1", "facebook/UMA"), ("uma-m-1p1", "facebook/UMA"),
                     ("esen-sm-conserving-all-omol", "facebook/OMol25"),
                     ("esen-md-direct-all-omol", "facebook/OMol25")):
    MODELS[_name] = MLIPSpec("fairchem", f"FAIR-Chem {_name} (OMol25 head for molecules); batched on the GPU",
                             "pip install fairchem-core", gated=True,
                             elements="most of the periodic table", options={"repo": _repo})


def unknown_model_message(name: str) -> str:
    open_names = [n for n, s in MODELS.items() if not s.gated]
    gated = [n for n, s in MODELS.items() if s.gated]
    return (f"Unknown MLIP model {name!r}. Open models: {', '.join(open_names)}. "
            f"Gated (Hugging Face access needed): {', '.join(gated)}. "
            "For a local file set `family` and `checkpoint`; for any other ASE calculator set "
            "`calculator = \"package.module:Class\"`. `mepd models` describes each.")


def _import_attr(path: str):
    module_name, _, attr = str(path).replace(":", ".").rpartition(".")
    if not module_name:
        raise ValueError(f"calculator must be 'package.module:ClassOrFactory', got {path!r}.")
    return getattr(importlib.import_module(module_name), attr)


class MLIPEngine(ASEEngine):
    """A machine-learned potential run through ASE (see the module docstring
    for how to choose one)."""

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        checkpoint: str | None = None,
        family: str | None = None,
        calculator: str | None = None,
        calculator_kwds: dict | None = None,
        options: dict | None = None,
        geometry_optimizer: str = "LBFGSLineSearch",
        transition_state_optimizer: str = "SELLA",
        ase_optimizer=None,
    ):
        if calculator is None and model is None:
            raise ValueError("mlip_engine_kwds needs `model` (see `mepd models`) or `calculator`.")
        spec = MODELS.get(model) if model else None
        if calculator is None:
            family = family or (spec.family if spec else None)
            if family is None:
                raise ValueError(unknown_model_message(model))
            if family not in FAMILIES and family != "fairchem":
                raise ValueError(f"Unknown model family {family!r}; known: {', '.join([*FAMILIES, 'fairchem'])}.")
        self.model = model
        self.device = device
        self.checkpoint = checkpoint
        self.family = family
        self.calculator_path = calculator
        self.calculator_kwds = dict(calculator_kwds or {})
        self.options = {**(spec.options if spec else {}), **dict(options or {})}
        self.geometry_optimizer = geometry_optimizer
        self.transition_state_optimizer = transition_state_optimizer
        self.ase_optimizer = ase_optimizer if ase_optimizer is not None else AVAIL_OPTS[geometry_optimizer]
        self._model = None

    def __repr__(self) -> str:
        what = self.calculator_path or self.model
        return f"MLIPEngine({what!r}{', checkpoint=' + repr(self.checkpoint) if self.checkpoint else ''})"

    def __eq__(self, other) -> bool:
        return self is other

    __hash__ = object.__hash__

    def _load(self):
        spec = MODELS.get(self.model)
        try:
            if self.calculator_path:
                return _import_attr(self.calculator_path)
            load, _ = FAMILIES[self.family]
            return load(self.model, self.checkpoint, self.device, self.options)
        except ImportError as exc:
            hint = spec.install if spec else f"install the package providing the {self.family or self.calculator_path} model"
            raise ImportError(f"MLIP model {self.model or self.calculator_path!r} needs a package that is not installed: {hint}. ({exc})") from exc

    @property
    def calculator(self):
        if self._model is None:
            self._model = self._load()
        if self.calculator_path:
            return self._model(**self.calculator_kwds)
        return FAMILIES[self.family][1](self._model)

    @calculator.setter
    def calculator(self, value) -> None:
        raise AttributeError("MLIPEngine builds its calculator from `model` / `calculator`.")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __deepcopy__(self, memo):
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        clone.__dict__.update(self.__dict__)
        return clone


def build_mlip_engine(kwds: dict[str, Any]):
    """The engine for `mlip_engine_kwds`: FAIR-Chem models get the batched
    FAIRChemEngine, everything else MLIPEngine."""
    kwds = dict(kwds or {})
    model = kwds.get("model")
    spec = MODELS.get(model) if model else None
    if kwds.get("calculator") is None and (kwds.get("family") or (spec.family if spec else None)) == "fairchem":
        from mepd.engines.fairchem import FAIRChemEngine

        fair = {k: kwds[k] for k in ("device", "checkpoint", "task", "inference_settings", "batch_size",
                                     "geometry_optimizer", "transition_state_optimizer") if k in kwds}
        return FAIRChemEngine(model=model or "uma-s-1p2p1", **fair)
    return MLIPEngine(**kwds)
