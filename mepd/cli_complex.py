"""`mepd complex`: a complex (several molecules together, not bonded)
built from its molecules -- packed, docked (xtb aISS), a CREST
NCI ensemble or a CREST QCG solvation shell (see mepd.complexes).

Writes <output>/complexes.xyz (best first; the comment line holds the
screening energy, which is not at your level of theory) and summary.json.
Minimize them at one level (`mepd optimize`) before comparing energies.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import typer


def complex_(
    molecules: List[str] = typer.Argument(..., help="SMILES or xyz files; 'X*3' for three copies."),
    method: str = typer.Option("dock", "--method", help="side, packed, dock (xtb aISS), nci (CREST NCI ensemble) "
                                                        "or qcg (CREST solvation shell: one solute, N solvents)."),
    keep: int = typer.Option(3, "--keep", min=1, help="Geometries kept (dock and nci; best first)."),
    seed: int = typer.Option(0, "--seed", help="Packing seed (packed)."),
    output: Path = typer.Option(Path("mepd_complex_output"), "--output", "-o"),
):
    from mepd import complexes
    from mepd.cli_common import _load_structure_from_smiles_or_xyz
    from mepd.discovery.cli_nanoreactor import _parse_molecule

    if method not in complexes.METHODS:
        raise typer.BadParameter(f"--method must be one of {', '.join(complexes.METHODS)}")
    missing = complexes.missing_programs(method)
    if missing:
        raise typer.BadParameter(f"--method {method} needs {', '.join(missing)} on PATH")
    structures = []
    for item in molecules:
        value, count = _parse_molecule(item)
        s = _load_structure_from_smiles_or_xyz(value, None, None)
        structures += [s] * count
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    typer.echo(f"Building a complex of {len(structures)} molecules ({method})...")
    try:
        found = complexes.build(structures, method, output / "work", keep=keep, seed=seed)
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Could not build the complex: {exc}")
        raise typer.Exit(code=1)
    frames = []
    for k, s in enumerate(found):
        frames.append(s.to_xyz().splitlines())
        frames[-1][1] += f" complex={k} method={method}"   # keeps qcdata's charge/multiplicity fields
    (output / "complexes.xyz").write_text("".join("\n".join(f) + "\n" for f in frames))
    (output / "summary.json").write_text(json.dumps({
        "method": method, "n_molecules": len(structures), "n_complexes": len(found),
        "seconds": round(time.time() - t0, 1),
        "note": "Geometries only: minimize them at your level of theory before comparing energies."}, indent=2))
    typer.echo(f"{len(found)} complex geometr{'y' if len(found) == 1 else 'ies'} in {output / 'complexes.xyz'}")
