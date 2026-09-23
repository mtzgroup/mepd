"""Structure parsing/description helpers for the web workspace.

Everything here is cheap (no QM): SMILES embedding, xyz parsing, SMILES
perception from a geometry, and 2D depictions for library cards and graph
nodes.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from qcdata import Structure

_XYZ_FIRST_LINE = re.compile(r"^\s*\d+\s*$")


def looks_like_xyz(text: str) -> bool:
    lines = text.strip().splitlines()
    return bool(lines) and bool(_XYZ_FIRST_LINE.match(lines[0]))


def structures_from_xyz_text(
    text: str, charge: Optional[int] = None, multiplicity: Optional[int] = None
) -> list[Structure]:
    """Every frame of a (possibly multi-frame) xyz text. A charge/multiplicity
    given here overrides whatever the xyz comment line carries."""
    text = text.strip() + "\n"
    try:
        frames = Structure.from_xyz_multi(text)
    except Exception:
        frames = [Structure.from_xyz(text)]
    out = []
    for s in frames:
        updates = {}
        if charge is not None:
            updates["charge"] = charge
        if multiplicity is not None:
            updates["multiplicity"] = multiplicity
        out.append(s.model_copy(update=updates) if updates else s)
    return out


def structure_from_smiles(
    smiles: str, charge: Optional[int] = None, multiplicity: Optional[int] = None
) -> Structure:
    # Same embedding path (rdkit, then openbabel for multi-fragment SMILES)
    # the CLI uses for a SMILES --start/--end, so what the user sees here is
    # what a job started with that SMILES would see.
    from mepd.cli import _load_structure_from_smiles_or_xyz

    return _load_structure_from_smiles_or_xyz(smiles.strip(), charge, multiplicity)


def perceive_smiles(structure: Structure) -> Optional[str]:
    import qcinf

    for backend in ("rdkit", "openbabel"):
        try:
            return qcinf.structure_to_smiles(structure, backend=backend)
        except Exception:
            continue
    return None


def canonical_key(smiles: str) -> str:
    """Order-independent identity of a (possibly multi-fragment) SMILES, so
    'C=C.O.N' and 'C=C.N.O' compare equal."""
    try:
        from rdkit import Chem

        frags = []
        for part in smiles.split("."):
            mol = Chem.MolFromSmiles(part)
            frags.append(Chem.MolToSmiles(mol) if mol is not None else part)
        return ".".join(sorted(frags))
    except Exception:
        return ".".join(sorted(smiles.split(".")))


def formula(structure: Structure) -> str:
    try:
        return structure.formula
    except Exception:
        counts: dict[str, int] = {}
        for sym in structure.symbols:
            counts[sym] = counts.get(sym, 0) + 1
        return "".join(f"{k}{v if v > 1 else ''}" for k, v in sorted(counts.items()))


@lru_cache(maxsize=512)
def depict_svg(smiles: str, width: int = 220, height: int = 160) -> Optional[str]:
    """2D depiction of a SMILES (transparent background, theme-neutral
    colours so it reads on light and dark cards)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2
        opts.padding = 0.12
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", "")
    except Exception:
        return None
