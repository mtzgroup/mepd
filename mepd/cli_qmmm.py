"""`mepd qmmm`: set up and check QM/MM systems (see mepd/qmmm.py).

  build    a solute in a shell of explicit solvent; the solute is the QM region
  reaction a gas-phase reaction in solvent: the start solvated, the end (and a
           TS) put into that same solvent shell
  embed    another geometry of the solute (product, TS) into an existing system
  region   a QM/MM region on a structure you bring (xyz or PDB)
  from-tc  convert a TeraChem QM/MM input (tc.in + prmtop + rst7 + qmindices)
  inspect  per-frame checks (frozen drift, boundary bonds, MM bond changes,
           clashes) and, with --inputs, the QM / environment energy split

A region file is used by any mepd command through the profile:

    [qmmm]
    file = "region.json"

and every structure given to that command must be the region's system (same
atoms, same order).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

qmmm_app = typer.Typer(help="Set up and check QM/MM systems (a QM region embedded in an MM environment).",
                       no_args_is_help=True)


def _load_system(path: str, charge: Optional[int], multiplicity: Optional[int]):
    fp = Path(path)
    if fp.suffix.lower() == ".pdb":
        import numpy as np
        import openmm.app as app
        from qcconst.constants import ANGSTROM_TO_BOHR
        from qcdata import Structure

        pdb = app.PDBFile(str(fp))
        xyz = np.asarray(pdb.getPositions(asNumpy=True)._value) * 10.0
        symbols = [a.element.symbol for a in pdb.topology.atoms()]
        return Structure(symbols=symbols, geometry=xyz * ANGSTROM_TO_BOHR, charge=int(charge or 0),
                         multiplicity=int(multiplicity or 1))
    from mepd.cli_common import _load_structure_from_smiles_or_xyz

    return _load_structure_from_smiles_or_xyz(path, charge, multiplicity)


def _write(output: Path, system, region, extra: Optional[dict] = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    system.save(str(output / "system.xyz"))
    region.save(output / "region.json")
    profile = output / "qmmm_profile.toml"
    if not profile.exists():
        profile.write_text('# QM level: edit like any profile. The [qmmm] table embeds it.\nengine_name = "gxtb"\n'
                           'path_min_method = "FNEB"\n\n[qmmm]\nfile = "region.json"\n')
    summary = {"system": str(output / "system.xyz"), "region": str(output / "region.json"),
               "profile": str(profile), "natoms": region.natoms, "qm_atoms": len(region.qm_atoms),
               "links": [list(x) for x in region.links], "frozen_atoms": len(region.frozen_atoms),
               "active_mm_atoms": len(region.active_mm_atoms), "mm": region.mm, "problems": region.check(system),
               **(extra or {})}
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    typer.echo(region.summary())
    for p in summary["problems"]:
        typer.secho(f"  ! {p}", fg="yellow")
    typer.echo(f"Wrote {output}/system.xyz, region.json and qmmm_profile.toml.")


@qmmm_app.command("build")
def build(
    solute: str = typer.Argument(..., help="SMILES or xyz file of the solute (the QM region)."),
    solvent: str = typer.Option("water", "--solvent", help="Solvent of the shell (water, methanol, "
                                "acetonitrile, dmso, ...)."),
    shell: float = typer.Option(6.0, "--shell", help="Shell thickness beyond the solute, Å."),
    n_molecules: Optional[int] = typer.Option(None, "--n-molecules", help="Number of solvent molecules "
                                              "(default: the liquid's density)."),
    qm: Optional[str] = typer.Option(None, "--qm", help="QM atoms (default: every solute atom), e.g. '0-11 14'."),
    active_radius: float = typer.Option(5.0, "--active-radius", help="Environment within this many Å of the "
                                        "QM region moves; the rest is frozen."),
    mm: str = typer.Option("gfnff", "--mm", help="Low level: gfnff, gfn2, gfn1, or tip3p (water; electrostatic "
                           "embedding, needs a QM engine that takes point charges: Psi4)."),
    relax: bool = typer.Option(True, "--relax/--no-relax", help="Relax the shell with GFN-FF (solute fixed)."),
    charge: Optional[int] = typer.Option(None, "--charge"),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity"),
    seed: int = typer.Option(0, "--seed"),
    output: Path = typer.Option(Path("mepd_qmmm"), "--output", "-o"),
):
    """A solute in a droplet of explicit solvent, with the solute as the QM region."""
    from mepd.cli_common import _load_structure_from_smiles_or_xyz
    from mepd.qmmm_build import SOLVENTS, solvated_system

    if solvent not in SOLVENTS:
        raise typer.BadParameter(f"--solvent must be one of {', '.join(SOLVENTS)}")
    s = _load_structure_from_smiles_or_xyz(solute, charge, multiplicity)
    system, region = solvated_system(s, solvent, shell, qm_atoms=qm, active_radius=active_radius, mm=mm,
                                     relax=relax, seed=seed, n_molecules=n_molecules)
    _write(output, system, region, {"solute": solute, "solvent": solvent, "shell": shell})


def _embed_all(items, system, region, output: Path, names) -> list[dict]:
    from mepd.qmmm_build import embed

    out = []
    for item, name in zip(items, names):
        placed, rep = embed(item, system, region)
        placed.save(str(output / f"{name}.xyz"))
        out.append({"name": name, "file": str(output / f"{name}.xyz"), **rep})
        note = f"closest contact {rep['closest_contact']:.2f} Å" if rep["closest_contact"] is not None else ""
        typer.echo(f"{name}: placed ({note}; {rep['rmsd_from_solute']:.2f} Å from the solute it replaces)")
        for w in rep["warnings"]:
            typer.secho(f"  ! {w}", fg="yellow")
    return out


def _reorder(structure, order):
    import numpy as np

    return structure.model_copy(update={"symbols": [structure.symbols[i] for i in order],
                                        "geometry": np.asarray(structure.geometry)[order]})


def _end_onto_start(start, end) -> Optional[list]:
    """The end's atom order that matches the start atom for atom (index k of
    the result: which end atom is start atom k). Already matching: identity;
    otherwise SLAPMapper plus the best relabeling of symmetric atoms, as
    `mepd run` maps its endpoints. None if no mapping is found."""
    import numpy as np

    n = len(start.symbols)
    if len(end.symbols) != n or sorted(map(str, start.symbols)) != sorted(map(str, end.symbols)):
        return None
    from mepd.cli_common import _check_endpoint_atom_mapping
    from mepd.inputs import RunInputs

    try:
        mapped = _check_endpoint_atom_mapping(start, end, True, RunInputs())
    except Exception as exc:
        typer.echo(f"Atom mapping failed ({type(exc).__name__}: {exc})")
        mapped = end
    if list(map(str, mapped.symbols)) != list(map(str, start.symbols)):
        return None
    x_end, x_map = np.asarray(end.geometry), np.asarray(mapped.geometry)
    order = [int(np.argmin(np.linalg.norm(x_end - x_map[k], axis=1))) for k in range(n)]
    return order if sorted(order) == list(range(n)) else None


@qmmm_app.command("reaction")
def reaction(
    start: str = typer.Option(..., "--start", help="Reactant (xyz or SMILES): it is solvated."),
    end: str = typer.Option(..., "--end", help="Product (xyz): the same atoms in the same order."),
    ts: Optional[str] = typer.Option(None, "--ts", help="A TS (xyz, same atoms) to put in too."),
    solvent: str = typer.Option("water", "--solvent"),
    shell: float = typer.Option(6.0, "--shell", help="Shell thickness beyond the reactant, Å."),
    active_radius: float = typer.Option(5.0, "--active-radius"),
    mm: str = typer.Option("gfnff", "--mm", help="gfnff, gfn2, gfn1, or tip3p (water, electrostatic embedding)."),
    charge: Optional[int] = typer.Option(None, "--charge"),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity"),
    relax: bool = typer.Option(True, "--relax/--no-relax", help="Relax the shell with GFN-FF (reactant fixed)."),
    seed: int = typer.Option(0, "--seed"),
    output: Path = typer.Option(Path("mepd_qmmm_reaction"), "--output", "-o"),
):
    """A gas-phase reaction in explicit solvent: the reactant is solvated and
    the product (and TS) are put into that same solvent shell, so all of them
    are one QM/MM system. Minimize start and end (and optimize the TS) with
    the profile written here before a path search."""
    from mepd.cli_common import _load_structure_from_smiles_or_xyz
    from mepd.qmmm_build import SOLVENTS, solvated_system

    if solvent not in SOLVENTS:
        raise typer.BadParameter(f"--solvent must be one of {', '.join(SOLVENTS)}")
    a = _load_structure_from_smiles_or_xyz(start, charge, multiplicity)
    b = _load_structure_from_smiles_or_xyz(end, int(a.charge), int(a.multiplicity))
    order = _end_onto_start(a, b)
    if order is None:
        raise typer.BadParameter("--end could not be matched atom for atom to --start (different atoms, or no "
                                 "atom mapping found): give the end in the start's atom order")
    items = [_reorder(b, order)]
    others = [("product", end)]
    if ts:
        t = _load_structure_from_smiles_or_xyz(ts, int(a.charge), int(a.multiplicity))
        if list(t.symbols) == list(a.symbols):
            items.append(t)                  # already in the start's order
        elif list(t.symbols) == list(b.symbols):
            items.append(_reorder(t, order))  # in the end's order: the end's mapping
        else:
            raise typer.BadParameter("--ts must list the start's (or the end's) atoms in the same order")
        others.append(("ts", ts))
    system, region = solvated_system(a, solvent, shell, active_radius=active_radius, mm=mm, seed=seed, relax=relax)
    _write(output, system, region, {"start": start, "end": end, "ts": ts, "solvent": solvent, "shell": shell})
    placed = _embed_all(items, system, region, output, [n for n, _ in others])
    summary = json.loads((output / "summary.json").read_text())
    summary["embedded"] = placed
    (output / "summary.json").write_text(json.dumps(summary, indent=1))
    typer.echo(f"Next: mepd optimize {output}/system.xyz {output}/product.xyz -i {output}/qmmm_profile.toml"
               + (f"; mepd ts --guess {output}/ts.xyz -i {output}/qmmm_profile.toml --irc" if ts else ""))


@qmmm_app.command("embed")
def embed_cmd(
    structures: list[Path] = typer.Argument(..., exists=True, help="Geometries of the solute (xyz): the solute's "
                                            "atoms in its order."),
    system_dir: Optional[Path] = typer.Option(None, "--system", help="A folder from `mepd qmmm build`/`reaction` "
                                              "(system.xyz + region.json)."),
    region_file: Optional[Path] = typer.Option(None, "--region", help="region.json (instead of --system)."),
    into: Optional[Path] = typer.Option(None, "--into", help="The system structure to put them into (default: "
                                        "the system's, e.g. the minimized reactant in solvent)."),
    output: Path = typer.Option(Path("mepd_qmmm_embed"), "--output", "-o"),
):
    """Put other geometries of the solute (a product, a TS) into an existing
    QM/MM system, in place of the solute: aligned onto it and moved out of
    contact with the environment, which stays as it is."""
    from qcdata import Structure

    from mepd.qmmm import QMMMRegion

    if system_dir is None and region_file is None:
        raise typer.BadParameter("give --system DIR or --region region.json")
    region = QMMMRegion.open(region_file or system_dir / "region.json")
    base = into or (system_dir / "system.xyz" if system_dir else None)
    system = Structure.open(str(base)) if base else region.reference_structure()
    system = system.model_copy(update={"charge": region.charge, "multiplicity": region.qm_multiplicity})
    output.mkdir(parents=True, exist_ok=True)
    items = []
    for fp in structures:
        try:
            frames = Structure.open_multi(str(fp))
        except Exception:
            frames = [Structure.open(str(fp))]
        items += frames
    names = [f"embedded_{k}" for k in range(len(items))]
    placed = _embed_all(items, system, region, output, names)
    (output / "summary.json").write_text(json.dumps({"system": str(base), "embedded": placed}, indent=1))


@qmmm_app.command("region")
def region_cmd(
    system: str = typer.Argument(..., help="The whole system: xyz or PDB."),
    qm: str = typer.Option(..., "--qm", help="QM atoms (0-based), e.g. '0-11 14'."),
    qm_charge: Optional[int] = typer.Option(None, "--qm-charge", help="Charge of the QM region (default: --charge)."),
    multiplicity: int = typer.Option(1, "--multiplicity", help="Spin multiplicity of the QM region."),
    charge: Optional[int] = typer.Option(None, "--charge", help="Charge of the whole system."),
    active_radius: Optional[float] = typer.Option(6.0, "--active-radius", help="Environment within this many Å "
                                                  "of the QM region moves; the rest is frozen. 0 = none frozen."),
    frozen: str = typer.Option("", "--frozen", help="Extra frozen atoms."),
    mm: str = typer.Option("gfnff", "--mm", help="Low level: gfnff, gfn2, gfn1, tip3p (water), amber (OpenMM) or "
                           "terachem."),
    embedding: str = typer.Option("mechanical", "--embedding", help="mechanical, or electrostatic (tip3p/amber; "
                                  "the QM engine must take point charges: Psi4)."),
    prmtop: Optional[Path] = typer.Option(None, "--prmtop", help="AMBER topology (mm amber/terachem)."),
    output: Path = typer.Option(Path("mepd_qmmm"), "--output", "-o"),
):
    """A QM/MM region on your own structure."""
    from mepd.qmmm import QMMMRegion

    s = _load_system(system, charge, None)
    extra = {}
    if prmtop:
        extra["prmtop"] = str(prmtop.resolve())
    if mm == "amber" and not prmtop and Path(system).suffix.lower() == ".pdb":
        extra["pdb"] = str(Path(system).resolve())
    if embedding != "mechanical" or mm != "tip3p":
        extra["embedding"] = embedding if mm != "tip3p" else "electrostatic"
    region = QMMMRegion.build(s, qm, qm_charge=qm_charge if qm_charge is not None else int(s.charge),
                              qm_multiplicity=multiplicity, active_radius=active_radius or None,
                              frozen_atoms=frozen, mm=mm, **extra)
    _write(output, s, region)


@qmmm_app.command("from-tc")
def from_tc(
    tcin: Path = typer.Argument(..., exists=True, help="TeraChem QM/MM input (its prmtop, coordinates and "
                                "qmindices files next to it)."),
    mm: str = typer.Option("amber", "--mm", help="amber: run it now with any QM engine through OpenMM; "
                           "terachem: keep TeraChem's own QM/MM (needs TeraChem/ChemCloud)."),
    active_radius: Optional[float] = typer.Option(None, "--active-radius", help="Also freeze everything beyond "
                                                  "this many Å of the QM region (default: only $constraints)."),
    output: Path = typer.Option(Path("mepd_qmmm"), "--output", "-o"),
):
    """Convert a neb-dynamics / TeraChem QM/MM setup for mepd."""
    from mepd.qmmm_build import from_terachem

    system, region, qm = from_terachem(tcin, mm=mm, active_radius=active_radius)
    _write(output, system, region, {"tcin": str(tcin), "terachem_method": qm})
    if mm == "terachem":
        prof = output / "qmmm_profile.toml"
        prof.write_text(f'engine_name = "chemcloud"\nprogram = "terachem"\npath_min_method = "FNEB"\n\n'
                        f'[program_kwds.model]\nmethod = "{qm["method"]}"\nbasis = "{qm["basis"]}"\n\n'
                        '[qmmm]\nfile = "region.json"\n')
    typer.echo(f"TeraChem QM level was {qm['method']}/{qm['basis']}"
               + ("" if mm == "terachem" else "; the profile uses g-xTB for the QM region: change it there."))


@qmmm_app.command("inspect")
def inspect(
    region_file: Path = typer.Argument(..., exists=True, help="region.json"),
    frames: Path = typer.Argument(..., exists=True, help="xyz (one or many frames) of the region's system."),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="Profile with this [qmmm]: also split "
                                          "each frame's energy into QM and environment parts."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write the report (JSON) here."),
):
    """Check a QM/MM structure or path for nonsense."""
    from qcdata import Structure

    from mepd.qmmm import QMMMRegion, diagnose

    region = QMMMRegion.open(region_file)
    try:
        structures = Structure.open_multi(str(frames))
    except Exception:
        structures = [Structure.open(str(frames))]
    structures = [s.model_copy(update={"charge": region.charge, "multiplicity": region.qm_multiplicity})
                  for s in structures]
    report = diagnose(region, structures)
    if inputs is not None:
        from mepd.cli_common import _open_run_inputs

        ri = _open_run_inputs(inputs)
        parts = ri.engine.decompose(structures)
        e0 = parts[0]["energy"]
        for rec, p in zip(report["frames"], parts):
            rec.update({k: p[k] for k in ("energy", "qm", "environment", "low_real", "low_model", "max_force_qm")})
            rec["rel_kcal"] = (p["energy"] - e0) * 627.509474
    for w in report["warnings"]:
        typer.secho(f"! {w}", fg="yellow")
    w = report["worst"]
    typer.echo(f"{len(structures)} frame(s): frozen drift ≤ {w['frozen']:.2e} Å, boundary stretch ≤ "
               f"{w['boundary']:.2f}, closest QM–MM contact {w['contact'] or float('nan'):.2f} Å, "
               f"MM bond changes {w['mm_changes']}")
    if output:
        output.write_text(json.dumps(report, indent=1))
        typer.echo(f"Wrote {output}")
