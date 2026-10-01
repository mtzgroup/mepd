"""`mepd discovery nanoreactor`: discover reactions in a piston-compressed
hot MD box and extract each with only the molecules it needs (see
mepd.discovery.nanoreactor)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import typer

from mepd.discovery.cli import discovery_app


def _parse_molecule(item: str) -> tuple[str, int]:
    """'SMILES', 'SMILES*3', '3*SMILES' or an xyz path (optionally '*N')."""
    left, sep, right = item.rpartition("*")
    if sep and right.strip().isdigit():
        return left.strip(), int(right)
    left, sep, right = item.partition("*")
    if sep and left.strip().isdigit():
        return right.strip(), int(left)
    return item.strip(), 1


def _ts_for_reaction(rxn, species_by_id, run_inputs, output: Path, workers: int) -> dict:
    """Path search, TS optimization and IRC between a reaction's optimized
    subsystem endpoints. Keeps the lowest TS whose IRC connects the same
    reactant and product bonds; barriers are measured from the reactant
    complex (and, for reference, from the separated reactants)."""
    from qcconst.constants import ANGSTROM_TO_BOHR, HARTREE_TO_KCAL_PER_MOL

    from mepd.discovery.cli_expand import _ts_connector
    from mepd.discovery.nanoreactor import _structure, perceive_bonds, read_xyz_frames
    from mepd.nodes.node import StructureNode

    c = rxn.complex
    ends = []
    for key in ("reactant", "product"):
        syms, xyz, _ = read_xyz_frames(Path(c[key]))
        node = StructureNode(structure=_structure(syms, xyz[0], c["charge"], c["multiplicity"]))
        node._cached_energy = c[f"{key}_energy"]
        ends.append((node, perceive_bonds(syms, xyz[0])))
    d = output / "reactions" / f"reaction_{rxn.id}"
    found = _ts_connector(run_inputs, d, workers)([(0, 1)], [ends[0][0], ends[1][0]])
    best = None
    for step in found:
        syms = list(step["start"].symbols)
        got = [perceive_bonds(syms, n.coords / ANGSTROM_TO_BOHR) for n in (step["start"], step["end"])]
        want = [ends[0][1], ends[1][1]]
        if got not in (want, want[::-1]):
            continue
        if best is None or float(step["ts"].energy) < float(best["ts"].energy):
            best = step
    if best is None:
        return {"found": len(found), "error": "no TS whose IRC connects this reaction's reactants and products"}
    e_ts = float(best["ts"].energy)
    sep_r = [species_by_id[i].energy for i in rxn.reactants]
    sep_p = [species_by_id[i].energy for i in rxn.products]
    out = {"found": len(found), "label": best["label"], "files": best["files"], "energy": e_ts,
           "barrier_kcal": (e_ts - c["reactant_energy"]) * HARTREE_TO_KCAL_PER_MOL,
           "reverse_barrier_kcal": (e_ts - c["product_energy"]) * HARTREE_TO_KCAL_PER_MOL}
    if all(e is not None for e in sep_r + sep_p):
        out["barrier_from_separated_kcal"] = (e_ts - sum(sep_r)) * HARTREE_TO_KCAL_PER_MOL
        out["reverse_barrier_from_separated_kcal"] = (e_ts - sum(sep_p)) * HARTREE_TO_KCAL_PER_MOL
    return out


@discovery_app.command("nanoreactor")
def nanoreactor(
    molecules: List[str] = typer.Argument(
        ..., help="What goes in the reactor: SMILES or xyz files, each optionally with a count "
        "('O*8' or '8*O' = eight waters). One xyz file with every molecule is used as the reactor as it is."),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True,
        help="RunInputs TOML: the level of theory that refines species, reactions and TSs. Built-in defaults if "
        "omitted."),
    charge: Optional[int] = typer.Option(None, "--charge", help="Total charge (default: sum of the molecules')."),
    multiplicity: Optional[int] = typer.Option(
        None, "--multiplicity", help="Total spin multiplicity (default: 1 for an even electron count, else 2)."),
    temperature: float = typer.Option(2000.0, "--temperature", help="MD temperature (K)."),
    time_ps: float = typer.Option(20.0, "--time", help="Simulated time (ps)."),
    radius: Optional[float] = typer.Option(
        None, "--radius", help="Wide wall radius (Angstrom). Default: from the number of atoms."),
    compress: float = typer.Option(0.6, "--compress", help="Narrow radius as a fraction of the wide one."),
    period: float = typer.Option(1.0, "--period", help="Piston period (ps)."),
    duty: float = typer.Option(0.75, "--duty", help="Fraction of each period at the wide radius."),
    md_method: str = typer.Option(
        "auto", "--md-method",
        help="What drives the discovery MD: gfn2, gfn1 or gxtb (xtb's own MD: fast), 'level': any mepd engine, "
        "the level of theory of --md-inputs (default --inputs): MLIPs, ASE calculators, g-xTB, ...; or 'auto' "
        "(default): gfn2 if xtb is installed, else gxtb if the g-xTB program is found (also from the profile's "
        "g-xTB engine), else 'level'."),
    md_inputs: Optional[Path] = typer.Option(
        None, "--md-inputs", exists=True,
        help="--md-method level: RunInputs TOML for the MD (e.g. an MLIP), when it should differ from --inputs."),
    step_fs: float = typer.Option(0.5, "--step", help="MD time step (fs)."),
    dump_fs: float = typer.Option(2.0, "--dump", help="Trajectory frame spacing (fs)."),
    electronic_temperature: float = typer.Option(
        3000.0, "--electronic-temperature", help="Fermi smearing of the MD (K): lets bonds break cleanly."),
    wall_force: float = typer.Option(20.0, "--wall-force", help="Push of the wall on an atom outside it (kcal/mol/A)."),
    seed: int = typer.Option(0, "--seed", help="Random seed of the packing."),
    trajectory: Optional[Path] = typer.Option(
        None, "--trajectory", exists=True, dir_okay=False,
        help="Skip the MD: analyze this multi-frame xyz (from any MD code; frames --dump fs apart)."),
    min_lifetime: float = typer.Option(
        20.0, "--min-lifetime", help="Bond states shorter than this (fs) are vibrations, not chemistry."),
    merge_window: float = typer.Option(
        100.0, "--merge-window", help="Bond changes this close in time (fs) on shared molecules are one reaction."),
    partial_charges: bool = typer.Option(
        True, "--partial-charges/--no-partial-charges",
        help="Charge of each molecule from xtb partial charges of its frame (else neutral where possible)."),
    refine: bool = typer.Option(
        True, "--refine/--no-refine",
        help="Optimize each species and each reaction's subsystem ends at the --inputs level."),
    instances: int = typer.Option(3, "--instances", help="Occurrences kept (and refined) per reaction and species."),
    refine_live: bool = typer.Option(
        False, "--refine-live/--no-refine-live",
        help="Refine each reaction (its species and subsystem ends) as soon as its event has settled, while the MD "
        "goes on, in --live-workers parallel workers; written to live_network.json as it goes (the web app shows "
        "those reactions at once). The final pass reuses what is done."),
    live_workers: int = typer.Option(2, "--live-workers", help="--refine-live: parallel refinement workers."),
    connect: bool = typer.Option(
        False, "--connect/--no-connect",
        help="Then find each reaction's TS: path search between its optimized subsystem ends, TS optimization "
        "and IRC."),
    max_connect: int = typer.Option(20, "--max-connect", help="At most this many reactions get a TS search."),
    maxiter: int = typer.Option(300, "--maxiter", help="Geometry-optimization steps per structure."),
    workers: int = typer.Option(2, "--workers", help="Parallel TS searches for --connect."),
    output: Path = typer.Option(Path("mepd_nanoreactor_output"), "--output", "-o", help="Directory to write into."),
) -> None:
    """Nanoreactor: pack the molecules into a sphere, run hot MD while a
    piston wall periodically compresses them, and read the trajectory as
    reaction events. Each event keeps only the molecules whose bonds change
    (a water that shuttles a proton is part of it; spectators are not);
    those atoms are cut out at a frame before and after, optimized, and
    each molecule is also optimized alone. A species on both sides of a
    reaction is a shuttle, so keto -> enol and keto + H2O -> enol + H2O are
    two reactions between the same species. Energies are only compared
    within a reaction (the atoms of different reactions differ).

    Based on the ab initio nanoreactor (Wang, Titov, McGibbon, Liu, Pande,
    Martinez, Nat. Chem. 2014), reimplemented in mepd (no code from it is
    used); the MD and wall are xtb's (Bannwarth, Ehlert, Grimme, JCTC 2019).
    Full references in network.json.

    Writes md/ (trajectory.xyz), species/, reactions/ (per reaction: the
    cut frames and the optimized subsystem ends of each instance) and
    network.json (species, reactions, events)."""
    from qcconst.constants import ANGSTROM_TO_BOHR

    from mepd.cli_common import _echo_run_inputs_summary, _load_structure_from_smiles_or_xyz, _open_run_inputs
    from mepd.discovery import nanoreactor as nr

    md_method = md_method.strip().lower()
    # g-xTB's program may be known only to the profile (its engine's executable).
    md_executable = None
    if md_method in ("auto", "gxtb") and trajectory is None:
        profile_engine = _open_run_inputs(md_inputs or inputs).engine
        if md_method == "auto":
            md_method, md_executable = nr.pick_md_method(profile_engine)
            typer.echo(f"MD: {md_method} (automatic choice)")
        else:
            md_executable = nr.engine_gxtb_executable(profile_engine)
    elif md_method == "auto":
        md_method = "gfn2"
    settings = nr.ReactorSettings(temperature=temperature, time_ps=time_ps, step_fs=step_fs, dump_fs=dump_fs,
                                  radius=radius, compress=compress, period_ps=period, duty=duty, method=md_method,
                                  wall_force=wall_force, electronic_temperature=electronic_temperature, seed=seed)
    try:
        settings.radius = settings.radius or 1.0    # validated before the real radius is known
        settings.validate()
        settings.radius = radius
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    detect = nr.DetectSettings(min_lifetime_fs=min_lifetime, merge_window_fs=merge_window)
    output.mkdir(parents=True, exist_ok=True)

    import time

    live = {"at_ps": 0.0, "cost_s": 0.0, "wall": 0.0}

    def write_live_events(force: bool = False) -> None:
        """The events found so far, for the web page's live reactor view:
        at most every simulated ps, and never more than a fifth of the time."""
        now = time.time()
        if not force and now - live["wall"] < 5 * live["cost_s"]:
            return
        ctx: dict = {}
        try:
            data = nr.live_events(output / "md", total_charge=total_charge, detect=detect, dt_fs=dump_fs,
                                  charges_at=live.get("charges_at") if live.get("refiner") else None, context=ctx)
        except Exception as exc:   # the MD matters more than the preview
            typer.echo(f"(live events not updated: {type(exc).__name__}: {exc})")
            return
        tmp = output / "live_events.json.tmp"
        tmp.write_text(json.dumps(data, default=str))
        tmp.replace(output / "live_events.json")
        live["cost_s"], live["wall"] = time.time() - now, time.time()
        if live.get("refiner") is not None and ctx:
            n = live["refiner"].submit(ctx["symbols"], ctx["frames"], ctx["hist"], ctx["species"], ctx["reactions"],
                                       ctx["settled"])
            if n:
                typer.echo(f"{n} new reaction(s) queued for refinement")

    def say(event, payload):
        if event == "md_segment" and payload["time_ps"] - live["at_ps"] >= 1.0 - 1e-9:
            live["at_ps"] = payload["time_ps"]
            write_live_events()
        if event == "md_relax":
            typer.echo(f"Relaxing the packed reactor ({payload['natoms']} atoms)...")
        elif event == "md_segment":
            typer.echo(f"MD {payload['time_ps']:.2f} / {payload['total_ps']:.2f} ps (wall {payload['radius']:.1f} A"
                       + (f", {payload['temperature']:.0f} K" if payload.get("temperature") else "") + ")")
        elif event == "refine_species":
            typer.echo(f"Optimizing species {payload['index'] + 1}/{payload['total']}: {payload['smiles']}")
        elif event == "refine_reaction":
            typer.echo(f"Optimizing reaction {payload['index'] + 1}/{payload['total']}: {payload['label']}")
        elif event == "warning":
            typer.echo(f"WARNING: {payload['message']}")

    # --- the reactor --------------------------------------------------------
    parsed = [_parse_molecule(m) for m in molecules]
    structures = []
    for value, count in parsed:
        s = _load_structure_from_smiles_or_xyz(value, None, None)
        structures += [s] * count
    total_charge = sum(int(s.charge) for s in structures) if charge is None else int(charge)
    if trajectory is None:
        if len(structures) == 1 and len(parsed) == 1 and Path(parsed[0][0]).exists():
            s = structures[0]   # a whole reactor as one xyz
            symbols = list(s.symbols)
            coords = s.geometry.reshape(-1, 3) / ANGSTROM_TO_BOHR
            coords = coords - coords.mean(axis=0)
            import numpy as np

            settings.radius = radius or max(nr.auto_radius(len(symbols)),
                                            float(np.max(np.linalg.norm(coords, axis=1))) + 1.0)
        else:
            symbols, coords, settings.radius, _ = nr.pack_reactor(structures, radius, seed=seed)
    else:
        symbols = list(nr.read_xyz_frames(trajectory)[0])
    electrons = sum(nr._Z.get(s, 0) for s in symbols) - total_charge
    mult = multiplicity or (1 if electrons % 2 == 0 else 2)
    typer.echo(f"Reactor: {len(symbols)} atoms, charge {total_charge}, multiplicity {mult}"
               + (f", wall {settings.radius:.1f} -> {settings.radius * compress:.1f} A" if trajectory is None else ""))
    # Partial charges only label the molecules (which charge each carries):
    # an xtb single point does that whatever drove the MD, when xtb is there.
    q_method, q_exe = (md_method, md_executable) if md_method != "level" else ("gfn2", None)
    if md_method == "level" and nr.missing_programs("gfn2"):
        q_method, q_exe = "gxtb", nr.engine_gxtb_executable(_open_run_inputs(inputs).engine)
    try:
        nr._xtb_command(q_method, q_exe)
        q_ok = True
    except RuntimeError:
        q_ok = False
    charges = ({"method": q_method, "executable": q_exe, "electronic_temperature": electronic_temperature,
                "multiplicity": mult, "workdir": str(output / "partial_charges")} if partial_charges and q_ok
               else None)

    # --- refine as reactions appear (optional) -------------------------------
    import threading

    refiner = None
    net_lock = threading.Lock()

    def write_live_network() -> None:
        with net_lock:
            data = refiner.network()
            tmp = output / "live_network.json.tmp"
            tmp.write_text(json.dumps(data, default=str))
            tmp.replace(output / "live_network.json")

    q_cache: dict = {}

    def live_charges(frame, coords):
        if charges is None:
            return None
        if frame not in q_cache:
            q_cache[frame] = nr.frame_partial_charges(
                symbols, coords, charge=total_charge, multiplicity=mult, method=q_method,
                electronic_temperature=electronic_temperature, workdir=output / "partial_charges" / "live",
                executable=q_exe)
        return q_cache[frame]

    live["charges_at"] = live_charges
    if refine_live and refine and trajectory is None:
        refiner = nr.LiveRefiner(_open_run_inputs(inputs).engine, output, maxiter=maxiter, workers=live_workers,
                                 on_change=write_live_network)
        live["refiner"] = refiner
        typer.echo(f"Refining reactions as they appear ({live_workers} workers).")

    if trajectory is None:
        if md_method != "level":
            try:
                nr._xtb_command(md_method, md_executable)
            except RuntimeError as exc:
                raise typer.BadParameter(f"{exc} Or use --md-method level (the MD on your profile's own calculator).")
        try:
            if md_method == "level":
                md_run_inputs = _open_run_inputs(md_inputs or inputs)
                typer.echo(f"MD engine: {type(md_run_inputs.engine).__name__} "
                           f"({md_inputs or inputs or 'built-in defaults'})")
                traj = nr.run_engine_md(symbols, coords, charge=total_charge, multiplicity=mult, settings=settings,
                                        engine=md_run_inputs.engine, workdir=output / "md", on_event=say)
            else:
                traj = nr.run_reactor_md(symbols, coords, charge=total_charge, multiplicity=mult, settings=settings,
                                         workdir=output / "md", executable=md_executable, on_event=say)
        except Exception as exc:
            typer.echo(f"The reactor MD failed: {type(exc).__name__}: {exc}")
            raise typer.Exit(code=1)
    else:
        traj = trajectory

    # --- events -> species and reactions -------------------------------------
    if refiner is not None:
        write_live_events(force=True)   # the last settled reactions, too
    typer.echo("Finding reaction events in the trajectory...")
    symbols, frames, hist, species, reactions, events = nr.analyze_trajectory(
        traj, total_charge=total_charge, detect=detect, dt_fs=dump_fs, max_instances=instances, charges=charges)
    typer.echo(f"{len(events)} events, {len(species)} species, {len(reactions)} distinct reactions.")
    for r in reactions:
        typer.echo(f"  R{r.id}: {r.label}  (seen {r.count}x" + (f", reverse {r.reverse_count}x" if r.reverse_count
                                                                 else "") + ")")

    result = nr.NanoreactorResult(symbols, species, reactions, events,
                                  {**asdict(settings), **{f"detect_{k}": v for k, v in asdict(detect).items()},
                                   "charge": total_charge, "multiplicity": mult,
                                   "molecules": [f"{v}*{n}" for v, n in parsed],
                                   "inputs": str(inputs) if inputs else None},
                                  str(traj), dump_fs, len(frames))
    nr.write_md_species(symbols, frames, species, output)
    nr.write_network(result, output)
    if refine and (species or reactions):
        run_inputs = _open_run_inputs(inputs)
        _echo_run_inputs_summary(run_inputs)
        if refiner is not None:
            typer.echo("Waiting for the live refinement to finish...")
            refiner.close(wait=True)
        nr.refine(symbols, frames, hist, species, reactions, run_inputs.engine, output, maxiter=maxiter,
                  on_event=say, cache=refiner)
        nr.write_network(result, output)
        if connect:
            by_id = {s.id: s for s in species}
            todo = [r for r in reactions if r.complex.get("reactant")][:max_connect]
            for n, rxn in enumerate(todo, start=1):
                typer.echo(f"TS search {n}/{len(todo)}: {rxn.label}")
                try:
                    rxn.ts = _ts_for_reaction(rxn, by_id, run_inputs, output, workers)
                except Exception as exc:
                    rxn.ts = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                nr.write_network(result, output)
    for r in reactions:
        bits = []
        if r.delta_e_kcal is not None:
            bits.append(f"dE {r.delta_e_kcal:+.1f}")
        if r.ts.get("barrier_kcal") is not None:
            bits.append(f"barrier {r.ts['barrier_kcal']:.1f}")
        if bits:
            typer.echo(f"  R{r.id}: {r.label}  {', '.join(bits)} kcal/mol")
    typer.echo(f"Results in {output}/ (network.json)")
    (output / "summary.json").write_text(json.dumps({
        "kind": "nanoreactor", "n_species": len(species), "n_reactions": len(reactions), "n_events": len(events),
        "network": str(output / "network.json"), "methods": nr.REFERENCES}, indent=2))
