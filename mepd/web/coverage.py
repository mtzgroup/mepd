"""Which elements a level of theory was parametrized (or trained) for, so the
Design tab can warn before a structure with other elements is computed:
results for elements outside a method's parametrization are unlikely to be
meaningful.

Returns None where the coverage depends on something mepd can't see (a
DFT basis set, a custom calculator): the UI then says to check it.
"""
from __future__ import annotations

from typing import Optional

_SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
            "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()


def _upto(z: int) -> frozenset:
    return frozenset(_SYMBOLS[:z])


# g-xTB: H-Lr (Froitzheim, Müller, Hansen, Grimme, ChemRxiv 2025); GFN2-xTB: H-Rn (Bannwarth, Ehlert,
# Grimme, JCTC 2019).
GXTB = _upto(103)
GFN2 = _upto(86)
_ASE_KNOWN = {"mace-off": frozenset("H C N O F P S Cl Br I".split()), "tblite": GFN2,
              "emt": frozenset("H C N O Al Ni Cu Pd Ag Pt Au".split())}


def coverage(profile_text: Optional[str]) -> tuple[str, Optional[frozenset]]:
    """(method name, elements it covers or None if unknown) for a profile."""
    import tomllib

    data = tomllib.loads(profile_text) if profile_text else {}
    engine = str(data.get("engine_name") or "gxtb").lower()
    if engine == "gxtb":
        return "g-xTB", GXTB
    if engine in ("qccompute", "chemcloud"):
        program = str(data.get("program") or "xtb").lower()
        method = str(((data.get("program_kwds") or {}).get("model") or {}).get("method") or "").lower()
        if program in ("xtb", "crest") and ("gfn1" not in method and "gfn0" not in method):
            return "GFN2-xTB", GFN2
        return f"{program} {method}".strip(), None
    if engine == "mlip":
        from mepd.engines.mlip import MODELS

        name = str((data.get("mlip_engine_kwds") or {}).get("model") or "")
        spec = MODELS.get(name)
        if spec and spec.elements and "most of" not in spec.elements:
            return name, frozenset(spec.elements.split())
        return name or "the MLIP", None
    if engine == "ase":
        from mepd.engines.ase_calculators import CALCULATORS, lookup

        calc = str((data.get("ase_engine_kwds") or {}).get("calculator") or "")
        spec = lookup(calc) if calc else None
        key = next((k for k, v in CALCULATORS.items() if v is spec), None)
        return calc or "the ASE calculator", _ASE_KNOWN.get(key)
    return engine, None


def element_warnings(profile_text: Optional[str], symbols) -> list[str]:
    """Warnings for elements in `symbols` that the profile's method does not cover."""
    name, covered = coverage(profile_text)
    present = sorted(set(symbols), key=lambda s: _SYMBOLS.index(s) if s in _SYMBOLS else 999)
    if covered is None:
        heavy = [s for s in present if s not in ("H", "C", "N", "O")]
        return [f"Check that {name} is parametrized for {', '.join(heavy)} (mepd can't tell for this level of theory)."] \
            if heavy else []
    outside = [s for s in present if s not in covered]
    if not outside:
        return []
    return [f"{name} is not parametrized for {', '.join(outside)}: energies, geometries and barriers involving "
            f"{'it' if len(outside) == 1 else 'them'} are unlikely to be meaningful. Pick a level of theory that "
            "covers them (Profiles) before trusting results."]
