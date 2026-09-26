"""`mepd conformers`: sample the conformers of one molecule with RDKit or CREST
(mepd.conformers), optionally minimizing each at the --inputs level of theory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer


def conformers(
    structure: str = typer.Argument(..., help="The molecule: a path to an xyz file, or a SMILES string."),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", help="RunInputs TOML (level of theory for --minimize). Built-in defaults if omitted."),
    charge: Optional[int] = typer.Option(None, "--charge", help="Override the molecular charge."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity", help="Override the spin multiplicity."),
    backend: str = typer.Option("rdkit", "--backend", help="'rdkit' (ETKDG + MMFF) or 'crest' (CREST metadynamics)."),
    n_conformers: int = typer.Option(20, "--n-conformers", help="Most distinct conformers to keep (0 = no cap)."),
    rmsd_cutoff: float = typer.Option(
        0.5, "--rmsd-cutoff", help="Conformers closer than this (bohr, symmetry-aware RMSD) count as one."),
    crest_method: str = typer.Option("--gfn2", "--crest-method", help="CREST's level: --gfn2, --gfnff or --gfn2//gfnff."),
    crest_ewin: float = typer.Option(6.0, "--crest-ewin", help="CREST energy window (kcal/mol)."),
    minimize: bool = typer.Option(
        True, "--minimize/--no-minimize",
        help="Minimize every conformer at the --inputs level (as `mepd optimize`), so their energies can be "
        "compared. Without it, energies are the backend's own (MMFF or CREST) and are not written as mepd energies."),
    validate_minima_with_hessian: bool = typer.Option(
        False, "--validate-minima-with-hessian/--no-validate-minima-with-hessian", "-H/-noH",
        help="With --minimize: Hessian-check every minimized conformer."),
    output: Path = typer.Option(Path("mepd_conformers_output"), "--output", "-o", help="Directory to write results into."),
) -> None:
    """Conformers of one molecule, from RDKit (ETKDG + MMFF) or CREST.

    Writes conformers.xyz (every kept conformer, most stable first by the
    backend's energy) and summary.json. With --minimize (default), also runs
    `mepd optimize` on them into optimized/ (opt_<i>.xyz + summary.json).
    CREST runs single-threaded (-T 1).
    """
    from mepd.chain import Chain
    from mepd.cli_common import _load_structure_from_smiles_or_xyz
    from mepd.conformers import ConformerInputs, CrestInputs, generate_conformers
    from mepd.inputs import ChainInputs
    from mepd.nodes.node import StructureNode

    if backend not in ("rdkit", "crest"):
        raise typer.BadParameter("--backend must be 'rdkit' or 'crest'.")
    seed = StructureNode(structure=_load_structure_from_smiles_or_xyz(structure, charge, multiplicity))
    output.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    settings = ConformerInputs(backend=backend, n_conformers=n_conformers or None, rmsd_cutoff=rmsd_cutoff,
                               crest=CrestInputs(method=crest_method, ewin_kcal=crest_ewin, threads=1))
    typer.echo(f"Sampling conformers with {backend}...")
    found = generate_conformers(seed, settings, stats)
    typer.echo(f"Kept {len(found)} conformer(s) ({stats})")
    Chain.model_validate({"nodes": [n.copy() for n in found], "parameters": ChainInputs()}).write_to_disk(
        output / "conformers.xyz")
    summary = {"structure": structure, "backend": backend, "stats": stats, "n_conformers": len(found),
               "minimized": bool(minimize), "conformers_file": str(output / "conformers.xyz")}
    (output / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if minimize:
        from mepd.cli import optimize

        typer.echo("Minimizing each conformer at the --inputs level...")
        optimize(structures=[output / "conformers.xyz"], inputs=inputs, charge=seed.structure.charge,
                 multiplicity=seed.structure.multiplicity, validate_minima_with_hessian=validate_minima_with_hessian,
                 hessian_minimum_frequency_cutoff=0.0, hessian_minima_rescue_displacement=0.1,
                 output=output / "optimized")
