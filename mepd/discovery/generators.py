"""Swappable product generators for reaction network expansion.

A generator proposes the products of one species as bond changes on its
molecular graph -- (bonds broken, bonds formed), in the species' own atom
order -- and network_expansion does the rest the same way for all of them:
Lewis/charge/spin check, a 3D guess for each product, optimization, the
live view, and path searches between the species found.

    bond-rules  mepd's reimplementation of the break/form enumeration of
                ZStruct (Zimmerman 2013) and YARP (Zhao & Savoie 2021);
                see network_expansion.enumerate_bond_changes and REFERENCES

Anything else plugs in by import path ("package.module:function", returning
product Structures in the same atom order); see network_expansion.

To add a generator: write `propose(symbols, coords_angstrom, edges, *, ...)`
returning (proposals, stats) like `_bond_rules`, and register a `Generator`
in GENERATORS.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


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

    def available(self) -> bool:
        return not self.package or importlib.util.find_spec(self.package) is not None

    def check(self) -> None:
        if not self.available():
            raise ImportError(f"The {self.label} generator needs the {self.package!r} package: {self.install}")


def _bond_rules(symbols, coords, edges, *, charge, multiplicity, n_break, n_form, form_distance, max_products,
                allow_radicals, allow_zwitterions, source, options):
    from mepd.discovery.network_expansion import enumerate_bond_changes

    if options:
        raise ValueError(f"bond-rules takes no generator options (got {', '.join(options)}).")
    return enumerate_bond_changes(
        symbols, coords, edges, charge=charge, multiplicity=multiplicity, n_break=n_break, n_form=n_form,
        form_distance=form_distance, max_products=max_products, allow_radicals=allow_radicals,
        allow_zwitterions=allow_zwitterions, source=source)


GENERATORS: dict[str, Generator] = {
    "bond-rules": Generator(
        "bond-rules", "Bond rules (mepd)",
        "Every combination of up to n breaks and m formations between nearby atoms, kept if coordination "
        "and a Lewis structure allow it.", _bond_rules),
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
