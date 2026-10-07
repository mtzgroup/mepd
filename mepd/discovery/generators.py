"""Swappable product generators for reaction network expansion.

A generator proposes the products of one species as bond changes on its
molecular graph -- (bonds broken, bonds formed), in the species' own atom
order -- and network_expansion does the rest the same way for all of them:
Lewis/charge/spin check, a 3D guess for each product, optimization, the
live view, and path searches between the species found.

    bond-rules    mepd's reimplementation of the break/form enumeration of
                  ZStruct (Zimmerman 2013) and YARP (Zhao & Savoie 2021);
                  see network_expansion.enumerate_bond_changes and REFERENCES
    crest-msreact CREST's msreact fragment generator (fragments and isomers
                  from biased GFN2-xTB optimizations); see crest_msreact
    nanoreactor   hot piston MD of the structure: its state after each
                  reaction event; see nanoreactor_generator

Anything else plugs in by import path ("package.module:function", returning
product Structures in the same atom order); see network_expansion.

To add a generator: write `propose(symbols, coords_angstrom, edges, *, ...)`
returning (proposals, stats) like `_bond_rules` (kind "bonds"), or
`propose(structure, *, max_products, options)` returning product Structures
in the species' atom order (kind "structures"), and register a `Generator`
in GENERATORS.
"""
from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from mepd.discovery.nanoreactor_generator import OPTIONS as _REACTOR_OPTIONS


@dataclass(frozen=True)
class Generator:
    name: str
    label: str
    summary: str
    propose: Callable
    package: str = ""          # module that must be importable ("" = built in)
    install: str = ""
    references: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)   # generator-specific options and their defaults
    kind: str = "bonds"        # "bonds": (broken, formed) proposals; "structures": product geometries
    programs: tuple = ()       # executables that must be on PATH

    def available(self) -> bool:
        return (not self.package or importlib.util.find_spec(self.package) is not None) \
            and all(shutil.which(p) for p in self.programs)

    def check(self) -> None:
        if self.package and importlib.util.find_spec(self.package) is None:
            raise ImportError(f"The {self.label} generator needs the {self.package!r} package: {self.install}")
        missing = [p for p in self.programs if shutil.which(p) is None]
        if missing:
            raise RuntimeError(f"The {self.label} generator needs {', '.join(missing)} on PATH: {self.install}")


def _bond_rules(symbols, coords, edges, *, charge, multiplicity, n_break, n_form, form_distance, max_products,
                allow_radicals, allow_zwitterions, source, options):
    from mepd.discovery.network_expansion import enumerate_bond_changes

    if options:
        raise ValueError(f"bond-rules takes no generator options (got {', '.join(options)}).")
    return enumerate_bond_changes(
        symbols, coords, edges, charge=charge, multiplicity=multiplicity, n_break=n_break, n_form=n_form,
        form_distance=form_distance, max_products=max_products, allow_radicals=allow_radicals,
        allow_zwitterions=allow_zwitterions, source=source)


def _crest_msreact(structure, *, max_products, options):
    from mepd.discovery.crest_msreact import msreact_products

    known = {"mode", "nbonds", "nshifts", "nshifts2"}
    unknown = set(options) - known
    if unknown:
        raise ValueError(f"crest-msreact options are {', '.join(sorted(known))} (got {', '.join(sorted(unknown))}).")
    return msreact_products(structure, max_products=max_products, **options)


def _nanoreactor(structure, *, max_products, options):
    from mepd.discovery.nanoreactor_generator import OPTIONS, reactor_products

    if "embedded" in options or "environment_temperature" in options:
        raise ValueError("nanoreactor embedded=true (MD with QM/MM forces) runs only on a QM/MM system "
                         "(a profile with a [qmmm] table).")
    unknown = set(options) - set(OPTIONS)
    if unknown:
        raise ValueError(f"nanoreactor options are {', '.join(sorted(OPTIONS))} (got {', '.join(sorted(unknown))}).")
    return reactor_products(structure, max_products=max_products, **options)


GENERATORS: dict[str, Generator] = {
    "bond-rules": Generator(
        "bond-rules", "Bond rules (mepd)",
        "Every combination of up to n breaks and m formations between nearby atoms, kept if coordination "
        "and a Lewis structure allow it.", _bond_rules),
    "crest-msreact": Generator(
        "crest-msreact", "CREST msreact (fragments and isomers)",
        "CREST's mass-spectrometry fragment generator: repulsive potentials on bonds, GFN2-xTB optimizations, "
        "the distinct fragments and isomers kept. Likely fragments (precursors, read backwards) and nearby "
        "isomers, without enumerating bond changes.", _crest_msreact, kind="structures", programs=("crest", "xtb"),
        install="conda install -c conda-forge crest xtb (or the release binaries of crest-lab/crest and grimme-lab/xtb)",
        options={"mode": "all", "nbonds": 3, "nshifts": 0, "nshifts2": 0},
        references={"crest-msreact": {
            "method": "CREST msreact: automated fragment generation from biased GFN2-xTB optimizations",
            "cite": ["P. Pracht, S. Grimme, C. Bannwarth, F. Bohle, S. Ehlert, G. Feldmann, J. Gorges, M. Müller, "
                     "T. Neudecker, C. Plett, S. Spicher, P. Steinbach, P. A. Wesołowski, F. Zeller, J. Chem. Phys. "
                     "160, 114110 (2024), doi:10.1063/5.0197592"]}}),
    "nanoreactor": Generator(
        "nanoreactor", "Nanoreactor MD (hot, squeezed)",
        "Hot molecular dynamics of the structure inside a periodically contracting wall; the structure right "
        "after each reaction event is a product. Finds what the structure does on its own when pushed hard.",
        _nanoreactor, kind="structures",
        install="conda install -c conda-forge xtb (or set GXTB_EXECUTABLE)",
        options={k: v for k, v in _REACTOR_OPTIONS.items() if k != "workdir"},
        references={"nanoreactor": {
            "method": "piston-compressed high-temperature MD (mepd's reimplementation of the ab initio nanoreactor; "
                      "no code from it is used), with xtb's MD and wall",
            "cite": ["L.-P. Wang, A. Titov, R. McGibbon, F. Liu, V. S. Pande, T. J. Martinez, Nat. Chem. 6, 1044-1048 "
                     "(2014), doi:10.1038/nchem.2099"]}}),
}


def get_generator(name: str) -> Generator:
    gen = GENERATORS.get(str(name).strip().lower())
    if gen is None:
        raise ValueError(f"Unknown generator {name!r}. Built in: "
                         + ", ".join(f"{g.name}{'' if g.available() else ' (not installed)'}" for g in GENERATORS.values())
                         + "; or your own as 'package.module:function'.")
    return gen


def is_named(name: Optional[str]) -> bool:
    """A registered generator (as opposed to an import path)."""
    return str(name or "").strip().lower() in GENERATORS


def propose(name: str, symbols: Sequence[str], coords_angstrom, edges, *, options: Optional[dict] = None, **kw):
    gen = get_generator(name)
    gen.check()
    return gen.propose(list(symbols), coords_angstrom, set(edges), options=dict(options or {}), **kw)
