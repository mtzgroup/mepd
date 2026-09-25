"""The ASE calculators `engine_name = "ase"` knows by name.

`ase_engine_kwds.calculator` takes either one of the short names below
(case-insensitive, e.g. "mace-off", "tblite", "emt") or the import path of
any other ASE calculator class or factory, "package.module:Name". The
calculator's own arguments go in `[ase_engine_kwds.calculator_kwds]`.

Machine-learned potentials with a model registry (AIMNet2, Orb, ANI, UMA,
...) are better run through `engine_name = "mlip"`, which passes each
structure's charge and spin to the model; see mepd/engines/mlip.py and
`mepd models`.
"""
from __future__ import annotations

import importlib
import importlib.util
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class CalculatorSpec:
    path: str          # "package.module:Name"
    summary: str
    install: str       # how to get the package ("" when it ships with ASE)
    charge_and_spin: str = "not read from the structure: set them in calculator_kwds if the calculator takes them"
    needs: str = ""    # what else must be present: a module, or "bin:<program>" for an executable
    kwds_hint: str = ""  # typical calculator_kwds, as TOML

    @property
    def package(self) -> str:
        return self.path.split(":", 1)[0].split(".", 1)[0]


_MACE = "pip install mace-torch  (not in the same environment as fairchem-core)"
_TBLITE = "pip install tblite  (or conda install -c conda-forge tblite-python)"
CALCULATORS: dict[str, CalculatorSpec] = {
    "mace-off": CalculatorSpec("mace.calculators:mace_off", "MACE-OFF23 organic force field (model = small/medium/large)",
                               _MACE, "neutral closed-shell only",
                               kwds_hint='model = "medium", device = "cpu", default_dtype = "float64"'),
    "mace-mp": CalculatorSpec("mace.calculators:mace_mp", "MACE-MP-0 foundation model, trained on materials (MPtrj)",
                              _MACE, "neutral only",
                              kwds_hint='model = "medium", device = "cpu", default_dtype = "float64"'),
    "tblite": CalculatorSpec("tblite.ase:TBLite", "GFN2-xTB / GFN1-xTB tight binding (method = \"GFN2-xTB\")", _TBLITE,
                             "set charge and multiplicity in calculator_kwds",
                             kwds_hint='method = "GFN2-xTB", charge = 0, multiplicity = 1'),
    "psi4": CalculatorSpec("ase.calculators.psi4:Psi4", "Psi4 DFT/wavefunction (method, basis in calculator_kwds)",
                           "conda install -c conda-forge psi4", "set charge and multiplicity in calculator_kwds", needs="psi4",
                           kwds_hint='method = "b3lyp", basis = "def2-svp", charge = 0, multiplicity = 1'),
    "orca": CalculatorSpec("ase.calculators.orca:ORCA", "ORCA (orcasimpleinput in calculator_kwds; needs the orca binary)",
                           "ships with ASE; install ORCA itself from orcaforum.kofo.mpg.de",
                           "set charge and mult in calculator_kwds", needs="bin:orca",
                           kwds_hint='orcasimpleinput = "B3LYP def2-SVP EnGrad", charge = 0, mult = 1'),
    "gaussian": CalculatorSpec("ase.calculators.gaussian:Gaussian", "Gaussian (method, basis in calculator_kwds; needs g16)",
                               "ships with ASE; needs a Gaussian install", "set charge and mult in calculator_kwds",
                               needs="bin:g16", kwds_hint='method = "b3lyp", basis = "6-31g*", charge = 0, mult = 1'),
    "nwchem": CalculatorSpec("ase.calculators.nwchem:NWChem", "NWChem (needs the nwchem binary)",
                             "ships with ASE; needs an NWChem install", "set charge and mult in calculator_kwds",
                             needs="bin:nwchem"),
    "dftb": CalculatorSpec("ase.calculators.dftb:Dftb", "DFTB+ (needs the dftb+ binary and Slater-Koster files)",
                           "ships with ASE; conda install -c conda-forge dftbplus", needs="bin:dftb+"),
    "emt": CalculatorSpec("ase.calculators.emt:EMT", "Effective medium theory: a toy potential, for testing only",
                          "", "ignored"),
    "lj": CalculatorSpec("ase.calculators.lj:LennardJones", "Lennard-Jones: a toy potential, for testing only",
                         "", "ignored"),
}
ALIASES = {"mace": "mace-off", "maceoff": "mace-off", "macemp": "mace-mp", "xtb": "tblite", "gfn2": "tblite",
           "gfn2-xtb": "tblite", "lennardjones": "lj", "lennard-jones": "lj", "dftb+": "dftb"}


def _key(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def lookup(name: str) -> CalculatorSpec | None:
    key = _key(name)
    key = ALIASES.get(key, ALIASES.get(key.replace("-", ""), key))
    if key in CALCULATORS:
        return CALCULATORS[key]
    dotted = str(name).strip().replace(":", ".")      # "a.b:C" and "a.b.C" name the same thing
    return next((s for s in CALCULATORS.values() if s.path.replace(":", ".") == dotted), None)


def installed(spec: CalculatorSpec) -> bool:
    if importlib.util.find_spec(spec.package) is None:
        return False
    if spec.needs.startswith("bin:"):
        return shutil.which(spec.needs[4:]) is not None
    return not spec.needs or importlib.util.find_spec(spec.needs) is not None


def _mlip_hint(name: str) -> str:
    try:
        from mepd.engines.mlip import FAMILIES, MODELS
    except ImportError:
        return ""
    key = _key(name)
    if key in MODELS or key in FAMILIES or key in ("uma", "fairchem", "esen", "ani", "ani2x"):
        model = key if key in MODELS else next(
            (n for n, s in MODELS.items() if s.family == key or n.startswith(key)), "aimnet2-rxn")
        return (f" {name!r} is a machine-learned potential: use engine_name = \"mlip\" with "
                f"[mlip_engine_kwds] model = \"{model}\" (run `mepd models` for the list).")
    return ""


def supported_message() -> str:
    names = ", ".join(f"{n}{'' if installed(s) else ' (not installed)'}" for n, s in CALCULATORS.items())
    return (f"Known names: {names}. Any other ASE calculator: its import path, \"package.module:ClassName\". "
            "Machine-learned potentials (AIMNet2, Orb, ANI, UMA): engine_name = \"mlip\", see `mepd models`.")


def resolve_path(name: str) -> str:
    """The import path for a short name or an import path; raises ValueError
    that lists what is supported."""
    spec = lookup(name)
    if spec is not None:
        return spec.path
    text = str(name).strip()
    if ":" in text or "." in text:
        return text
    raise ValueError(f"Unknown ASE calculator {name!r}.{_mlip_hint(name)} {supported_message()}")


def load_calculator(name: str):
    """The calculator class or factory for `ase_engine_kwds.calculator`, with
    errors that say what to install or write instead."""
    path = resolve_path(name)
    spec = lookup(name) or next((s for s in CALCULATORS.values() if s.path == path), None)
    module_name, _, attr = path.replace(":", ".").rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"ase_engine_kwds.calculator {name!r} is not \"package.module:ClassName\". {supported_message()}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        how = spec.install if spec and spec.install else f"install the package that provides {module_name}"
        raise ImportError(f"ASE calculator {name!r} needs a package that is not installed here: {how}. ({exc})") from exc
    try:
        return getattr(module, attr)
    except AttributeError:
        public = sorted(n for n in dir(module) if n[:1].isupper() or n.endswith("_mp") or n.endswith("_off"))
        raise ValueError(f"{module_name} has no {attr!r} (ase_engine_kwds.calculator = {name!r}). "
                         f"It has: {', '.join(public[:15]) or 'nothing that looks like a calculator'}.") from None
