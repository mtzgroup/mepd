"""`mepd qmmm`: set up and check QM/MM systems (see mepd/qmmm.py).

  build    a solute in a shell of explicit solvent; the solute is the QM region
  reaction a gas-phase reaction in solvent: the start solvated, the end (and a
           TS) put into that same solvent shell
  embed    another geometry of the solute (product, TS) into an existing system
  region   a QM/MM region on a structure you bring (xyz or PDB)
  protein-sites  dock a species (rigid, its own geometry) into a protein:
           the sites it binds, best first
  protein-build  the species at one of those sites as a QM/MM system (the
           protein and a water shell around it are the environment)
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
        # Electrostatic embedding needs a QM level that takes point charges.
        engine = "xtb" if region.embedding == "electrostatic" else "gxtb"
        profile.write_text(f'# QM level: edit like any profile. The [qmmm] table embeds it.\nengine_name = "{engine}"\n'
                           # NEB, not FNEB: FNEB regrows its string by geodesic interpolation,
                           # slow for hundreds of moving solvent atoms.
                           'path_min_method = "NEB"\n\n[qmmm]\nfile = "region.json"\n')
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


@qmmm_app.command("protein-sites")
def protein_sites(
    species: str = typer.Argument(..., help="The molecule or complex to place (xyz or SMILES): docked rigid, in "
                                  "this geometry (e.g. a reactant optimized with QM)."),
    protein: str = typer.Option(..., "--protein", help="The protein: a PDB file, or a PDB ID to download."),
    chains: Optional[str] = typer.Option(None, "--chains", help="Chains to keep, e.g. 'A,B,C' (default: all)."),
    ph: float = typer.Option(7.0, "--ph", help="pH for the protonation states of the protein's residues."),
    box: float = typer.Option(20.0, "--box", help="Edge of each docking box, Å."),
    spacing: float = typer.Option(12.0, "--spacing", help="Spacing of the box centres over the protein, Å."),
    exhaustiveness: int = typer.Option(8, "--exhaustiveness", help="Vina search effort per box."),
    max_sites: int = typer.Option(20, "--max-sites"),
    workers: int = typer.Option(4, "--workers", help="Docking processes at once."),
    charge: Optional[int] = typer.Option(None, "--charge", help="Charge of the species."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity"),
    output: Path = typer.Option(Path("mepd_protein_sites"), "--output", "-o"),
):
    """Where a species binds in a protein: the protein prepared (hydrogens at
    --ph, ligands and crystal water removed), the species docked rigid with
    AutoDock Vina in boxes covering it, the poses grouped into sites."""
    import numpy as np

    from mepd.cli_common import _load_structure_from_smiles_or_xyz
    from mepd.qmmm_protein import fetch_pdb, find_sites, prepare_protein, sites_to_json

    output.mkdir(parents=True, exist_ok=True)
    s = _load_structure_from_smiles_or_xyz(species, charge, multiplicity)
    src = Path(protein)
    if not src.exists():
        typer.echo(f"Downloading {protein} from the PDB...")
        src = fetch_pdb(protein, output)
    prep = prepare_protein(src, output / "protein.pdb", chains=chains.split(",") if chains else None, ph=ph)
    typer.echo(f"Protein: {prep['atoms']} atoms, {prep['residues']} residues, chains {','.join(prep['chains'])}, "
               f"charge {prep['charge']:+d} at pH {ph:g}"
               + (f"; removed {', '.join(prep['removed'])}" if prep["removed"] else "")
               + (f"; {prep['gaps_not_rebuilt']} gap(s) not rebuilt" if prep["gaps_not_rebuilt"] else ""))
    s.save(str(output / "species.xyz"))
    x = np.asarray(s.geometry) / 1.8897259886

    def progress(k, n):
        if k % max(1, n // 10) == 0 or k == n:
            typer.echo(f"  docked {k}/{n} boxes")

    typer.echo("Docking the species (rigid) over the protein...")
    sites = find_sites(output / "protein.pdb", [str(a) for a in s.symbols], x, output / "dock", box=box,
                       spacing=spacing, exhaustiveness=exhaustiveness, max_sites=max_sites, workers=workers,
                       progress=progress)
    (output / "sites.json").write_text(json.dumps({
        "protein": "protein.pdb", "species": "species.xyz", "charge": int(s.charge),
        "multiplicity": int(s.multiplicity), "prepared": prep, "sites": sites_to_json(sites)}, indent=1))
    for st in sites[:10]:
        typer.echo(f"  site {st.id}: {st.score:.1f} kcal/mol, found {st.hits}x, near "
                   + " ".join(st.residues[:6]) + (" ..." if len(st.residues) > 6 else ""))
    typer.echo(f"Wrote {output}/sites.json ({len(sites)} sites). Next: mepd qmmm protein-build {output} --site 0")


@qmmm_app.command("protein-build")
def protein_build(
    sites_dir: Path = typer.Argument(..., exists=True, help="A folder from `mepd qmmm protein-sites`."),
    site: int = typer.Option(0, "--site", help="Which site (its id in sites.json)."),
    qm_residues: str = typer.Option("", "--qm-residues", help="Residue side chains to add to the QM region, e.g. "
                                    "'A:ARG90 A:GLU78' (cut at CA-CB, with link atoms)."),
    water_shell: float = typer.Option(8.0, "--water-shell", help="TIP3P water within this many Å of the species."),
    active_radius: float = typer.Option(6.0, "--active-radius", help="Water and protein atoms within this many Å "
                                        "of the QM region move; the rest is frozen."),
    freeze_protein: bool = typer.Option(False, "--freeze-protein", help="Freeze every protein atom outside the QM "
                                        "region (only the water around the species moves)."),
    cutoff: float = typer.Option(12.0, "--cutoff", help="Force-field non-bonded cutoff, Å (0: none). The QM region "
                                 "feels every charge regardless."),
    end: Optional[str] = typer.Option(None, "--end", help="The reaction's other end (xyz: the species' atoms): put "
                                      "into the site in place of the species, as in `mepd qmmm reaction`."),
    ts: Optional[str] = typer.Option(None, "--ts", help="A TS between them (xyz, same atoms) to put in too."),
    output: Path = typer.Option(Path("mepd_protein_qmmm"), "--output", "-o"),
):
    """The species at a docked site as a QM/MM system: the protein (AMBER
    ff14SB) and a TIP3P water shell as the environment, whose charges act on
    the QM region (electrostatic embedding: use engine_name "xtb" or "psi4")."""
    import numpy as np
    from qcdata import Structure

    from mepd.qmmm_protein import build_site_system

    info = json.loads((sites_dir / "sites.json").read_text())
    rec = next((x for x in info["sites"] if x["id"] == site), None)
    if rec is None:
        raise typer.BadParameter(f"no site {site} in {sites_dir}/sites.json")
    species = Structure.open(str(sites_dir / info["species"]))
    rep = build_site_system(sites_dir / info["protein"], [str(a) for a in species.symbols], np.asarray(rec["coords"]),
                            output, ligand_charge=info["charge"], ligand_multiplicity=info["multiplicity"],
                            qm_residues=qm_residues.split(), water_shell=water_shell, active_radius=active_radius,
                            freeze_protein=freeze_protein, cutoff=cutoff or None,
                            name=f"site {site} of {Path(info['prepared']['pdb']).stem}")
    typer.echo(f"Site {site} ({rec['score']:.1f} kcal/mol): {rep['protein_atoms']} protein atoms, {rep['waters']} "
               f"waters, the species; system charge {rep['charge']:+d}.")
    _write(output, rep["structure"], rep["region_obj"], {"site": rec["id"], "score": rec["score"],
                                                        "residues": rec["residues"], "sites": str(sites_dir)})
    if end:
        # The reaction's other end (and TS) into the same site, in the
        # species' atom order: one QM/MM system, as for a reaction in solvent.
        b = Structure.open(end) if Path(end).exists() else None
        if b is None:
            raise typer.BadParameter("--end must be an xyz file")
        b = b.model_copy(update={"charge": species.charge, "multiplicity": species.multiplicity})
        order = _end_onto_start(species, b)
        if order is None:
            raise typer.BadParameter("--end could not be matched atom for atom to the species")
        items, names = [_reorder(b, order)], ["product"]
        if ts:
            t = Structure.open(ts).model_copy(update={"charge": species.charge, "multiplicity": species.multiplicity})
            if list(t.symbols) == list(species.symbols):
                items.append(t)
            elif list(t.symbols) == list(b.symbols):
                items.append(_reorder(t, order))
            else:
                raise typer.BadParameter("--ts must list the species' (or the end's) atoms in the same order")
            names.append("ts")
        placed = _embed_all(items, rep["structure"], rep["region_obj"], output, names)
        summary = json.loads((output / "summary.json").read_text())
        summary.update(embedded=placed, end=end, ts=ts)
        (output / "summary.json").write_text(json.dumps(summary, indent=1))


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
