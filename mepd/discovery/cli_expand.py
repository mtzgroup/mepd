"""`mepd discovery expand`: reaction network expansion from a structure by
graph rules (see mepd.discovery.network_expansion)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import typer

from mepd.chain import Chain
from mepd.discovery.cli import _progress, _validation_kwargs, discovery_app


def _parse_options(values: List[str]) -> dict:
    out = {}
    for item in values or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise typer.BadParameter(f"--generator-option takes key=value, got {item!r}.")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def _ts_connector(run_inputs, output: Path, workers: int):
    """`connect` for flux steering: a recursive path search (MSMEP) for each
    (source, product) pair, then TS optimization + IRC of every leaf's TS
    guess. Returns one record per step whose IRC runs between two different
    minima: its two IRC ends and TS. Resumable: finished pair trees and
    TS/IRC files are reused from `output`."""
    from mepd.cli_channels import _load_ts_and_irc_from_disk
    from mepd.cli_common import _fork_map, _optimize_ts_and_irc, _run_msmep_pairs, _ts_guess_tasks_from_tree
    from mepd.inputs import ChainInputs
    from mepd.TreeNode import TreeNode

    pairs_dir, ts_dir = output / "pairs", output / "ts"

    def connect(pairs, nodes):
        pairs_dir.mkdir(parents=True, exist_ok=True)
        ts_dir.mkdir(parents=True, exist_ok=True)
        charge, mult = int(nodes[0].structure.charge), int(nodes[0].structure.multiplicity)
        _run_msmep_pairs([n.copy() for n in nodes], list(pairs), pairs_dir, run_inputs,
                         parallel=False, parallel_workers=None, workers=workers)
        tasks = []
        for i, j in pairs:
            tree_dir = pairs_dir / f"pair_{i}_{j}" / "tree"
            if not (tree_dir / "node_0.xyz").exists():
                typer.echo(f"No path found for reaction {i} -> {j}; it carries no flux.")
                continue
            try:
                tree = TreeNode.read_from_disk(tree_dir, chain_parameters=ChainInputs(), charge=charge,
                                               multiplicity=mult)
            except Exception as exc:
                typer.echo(f"Could not read the path search for {i} -> {j}: {type(exc).__name__}: {exc}")
                continue
            tasks.extend(_ts_guess_tasks_from_tree(tree, f"pair_{i}_{j}_"))

        def ts_and_irc(task):
            label, guess = task
            if _load_ts_and_irc_from_disk(ts_dir, label, charge, mult) is None:
                _optimize_ts_and_irc(guess, run_inputs, ts_dir, run_irc=True, label=label)
            return label

        _fork_map(ts_and_irc, tasks, workers)
        found = []
        for label, _ in tasks:
            res = _load_ts_and_irc_from_disk(ts_dir, label, charge, mult)
            if res is None or res.irc_chain is None or len(res.irc_chain) < 2:
                continue
            ts = res.ts_node
            if ts._cached_energy is None:
                run_inputs.engine.compute_energies([ts])
            ends = [res.irc_chain[0], res.irc_chain[-1]]
            missing = [n for n in ends if n._cached_energy is None]
            if missing:
                run_inputs.engine.compute_energies(missing)
            found.append({"start": ends[0], "end": ends[1], "ts": ts, "label": label,
                          "files": {"ts": str(ts_dir / f"{label}.xyz"), "irc": str(ts_dir / f"{label}_irc.xyz")}})
        return found

    return connect


@discovery_app.command("expand")
def expand(
    structure: str = typer.Argument(..., help="Seed structure: a path to an xyz file, or a SMILES string."),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True, help="RunInputs TOML (level of theory). Built-in defaults if omitted."),
    charge: Optional[int] = typer.Option(None, "--charge", help="Override the seed's molecular charge."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity", help="Override the seed's spin multiplicity."),
    rounds: int = typer.Option(
        1, "--rounds", help="Expansion depth: round 2 proposes products of round 1's new species, and so on."),
    steer: str = typer.Option(
        "auto", "--steer",
        help="Which species the next round expands. 'flux': find a verified TS (path search, TS opt, IRC) for "
        "every reaction, run microkinetics from the seed, and expand only species whose concentration flux "
        "reaches --flux-threshold, so the network stops growing on its own. 'window': every new species "
        "within --energy-window. 'auto' (default): flux when --rounds > 1, else window."),
    temperature: float = typer.Option(298.15, "--temperature", help="--steer flux: temperature (K) for the rates."),
    time_s: float = typer.Option(3600.0, "--time", help="--steer flux: simulated time (s), starting from pure seed."),
    flux_threshold: float = typer.Option(
        0.01, "--flux-threshold",
        help="--steer flux: expand a species once the material that flowed into it (fraction of the seed) "
        "reaches this."),
    n_break: int = typer.Option(2, "--n-break", help="Most bonds broken in one proposed step."),
    n_form: int = typer.Option(2, "--n-form", help="Most bonds formed in one proposed step."),
    form_distance: float = typer.Option(
        4.0, "--form-distance", help="Only form bonds between atoms at most this far apart (Angstrom) in the source."),
    max_products: int = typer.Option(
        50, "--max-products", help="Proposals kept per source species (fewest bond changes first)."),
    energy_window: float = typer.Option(
        60.0, "--energy-window",
        help="Only species within this many kcal/mol of the seed are expanded in the next round."),
    allow_radicals: bool = typer.Option(
        False, "--allow-radicals/--no-allow-radicals",
        help="Also propose products with unpaired electrons beyond the multiplicity (e.g. carbenes, diradicals)."),
    allow_zwitterions: bool = typer.Option(
        False, "--allow-zwitterions", help="Also propose products that need separated formal charges."),
    generator: str = typer.Option(
        "bond-rules", "--generator",
        help="Which tool proposes the products: 'bond-rules' (built in: the ZStruct/YARP break-and-form "
        "enumeration, reimplemented), or 'package.module:function' for your own: function(structure, "
        "**options) returning product Structures (or xyz paths) in the same atom order."),
    generator_option: List[str] = typer.Option(
        [], "--generator-option", help="key=value passed to a custom --generator (values parsed as JSON). Repeatable."),
    products: Optional[Path] = typer.Option(
        None, "--products", exists=True, dir_okay=False,
        help="Import products another tool proposed (autodE, Chemoton, ...): a multi-frame xyz, "
        "atom-mapped to the seed (same atoms, same order). Replaces the generator for the seed."),
    maxiter: int = typer.Option(500, "--maxiter", help="Maximum geometry-optimization steps per proposal."),
    max_species: int = typer.Option(200, "--max-species", help="Stop adding species beyond this many."),
    workers: int = typer.Option(
        min(4, max(1, (os.cpu_count() or 1) // 2)), "--workers",
        help="Processes building product guesses (and the web live view's animations) in parallel. "
        "Optimizations use the engine's own parallelism (e.g. g-xTB runs one process per core)."),
    validate_minima_with_hessian: bool = typer.Option(
        False, "--validate-minima-with-hessian/--no-validate-minima-with-hessian", "-H/-noH",
        help="Hessian-check every new species (rescue push along an unstable mode; dropped if still not a minimum)."),
    hessian_minimum_frequency_cutoff: float = typer.Option(
        0.0, "--hessian-minimum-frequency-cutoff", help="Minimum allowed frequency (cm^-1)."),
    hessian_minima_rescue_displacement: float = typer.Option(
        0.1, "--hessian-minima-rescue-displacement", help="First rescue push along the unstable mode (bohr)."),
    connect: bool = typer.Option(
        False, "--connect/--no-connect",
        help="Then run a recursive path search (MSMEP, as in `mepd network-splits`) for every proposed "
        "reaction that landed on a different species, and build the network from the paths found."),
    max_pairs: int = typer.Option(50, "--max-pairs", help="Cap on path searches for --connect."),
    parallel: bool = typer.Option(False, "--parallel", help="--connect: evaluate MSMEP branches in parallel."),
    parallel_workers: Optional[int] = typer.Option(None, "--parallel-workers", help="Workers for --parallel."),
    output: Path = typer.Option(Path("mepd_expand_output"), "--output", "-o", help="Directory to write results into."),
) -> None:
    """Reaction network expansion: propose products of the seed by breaking
    and forming bonds (no Hessian sampling), optimize them, and repeat from
    the new species; optionally connect each proposed reaction by a path
    search. With --rounds > 1 the network grows only where the kinetics
    say material flows (--steer flux).

    Based on published methods, reimplemented in mepd (no code from them is
    used): the break/form enumeration on the bond graph is that of ZStruct
    (Zimmerman, J. Comput. Chem. 2013) and YARP (Zhao & Savoie, Nat. Comput.
    Sci. 2021); the Lewis-structure filter is RDKit DetermineBondOrders, i.e.
    xyz2mol (Kim & Kim, Bull. Korean Chem. Soc. 2015); flux steering follows
    Bensberg & Reiher (Isr. J. Chem. 2023) and the rate-based enlargement of
    Susnow et al. (J. Phys. Chem. A 1997). Product guesses use a restrained
    relaxation of mepd's own. Full references in summary.json.

    Writes species.xyz (the seed first, then every species found),
    species/species_<k>.xyz, proposals.xyz (every product guess, before
    optimization), rejected.xyz (if Hessian-checked) and summary.json (the
    species, and one record per proposed reaction: source, target, bonds
    broken/formed, outcome). With --connect, also pairs/ and network.json.
    """
    from mepd.cli_common import (
        _completed_tree_dirs, _echo_run_inputs_summary, _load_structure_from_smiles_or_xyz, _open_run_inputs,
        _run_msmep_pairs,
    )
    from mepd.discovery.network_expansion import REFERENCES, expand_network
    from mepd.nodes.node import StructureNode

    for name, value in (("--rounds", rounds), ("--max-products", max_products), ("--maxiter", maxiter),
                        ("--max-species", max_species), ("--max-pairs", max_pairs)):
        if value <= 0:
            raise typer.BadParameter(f"{name} must be a positive integer.")
    if n_break < 0 or n_form < 0 or n_break + n_form == 0:
        raise typer.BadParameter("--n-break and --n-form must be >= 0 and not both 0.")

    from mepd.discovery.generators import get_generator, is_named

    if products is None and is_named(generator):
        try:
            get_generator(generator).check()
        except ImportError as exc:
            raise typer.BadParameter(str(exc), param_hint="--generator") from None
    elif products is None and ":" not in generator and "." not in generator:
        try:
            get_generator(generator)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--generator") from None

    steer = steer.strip().lower()
    if steer not in ("auto", "window", "flux"):
        raise typer.BadParameter("--steer must be auto, window or flux.")
    if steer == "auto":
        steer = "flux" if rounds > 1 else "window"
    if steer == "flux" and (temperature <= 0 or time_s <= 0 or flux_threshold <= 0):
        raise typer.BadParameter("--temperature, --time and --flux-threshold must be positive.")
    run_inputs = _open_run_inputs(inputs)
    _echo_run_inputs_summary(run_inputs)
    seed = StructureNode(structure=_load_structure_from_smiles_or_xyz(structure, charge, multiplicity))
    validation = _validation_kwargs(validate_minima_with_hessian, hessian_minimum_frequency_cutoff,
                                    hessian_minima_rescue_displacement)

    from rich.console import Console

    console = Console()
    with _progress() as progress:
        task = {"id": None}

        def on_event(event, payload):
            if event == "proposed":
                if task["id"] is not None:
                    progress.remove_task(task["id"])
                task["id"] = progress.add_task(f"Round {payload['round']}: optimizing {payload['total']} proposal(s)",
                                               total=payload["total"] or None)
            elif event == "candidate_done" and task["id"] is not None:
                progress.update(task["id"], completed=payload["index"])
            elif event == "species_found":
                progress.console.print(f"[green]✓ species {payload['index']}[/green] {payload['smiles']}  "
                                       f"ΔE={payload['rel_energy_kcal']:+.1f} kcal/mol", highlight=False)

        try:
            result = expand_network(
                seed, run_inputs.engine, rounds=rounds, energy_window_kcal=energy_window, generator=generator,
                generator_options=_parse_options(generator_option), products_file=str(products) if products else None,
                maxiter=maxiter, n_break=n_break, n_form=n_form, form_distance=form_distance,
                max_products=max_products, allow_radicals=allow_radicals, allow_zwitterions=allow_zwitterions,
                max_species=max_species, workers=workers, on_event=on_event, steer=steer,
                connect=_ts_connector(run_inputs, output, workers) if steer == "flux" else None,
                kinetics={"temperature": temperature, "time_s": time_s, "threshold": flux_threshold},
                validate_minima={"frequency_cutoff": validation["hessian_minimum_frequency_cutoff"],
                                 "rescue_displacement": validation["hessian_minima_rescue_displacement"]}
                if validation else None,
            )
        except Exception as exc:
            typer.echo(f"Network expansion failed: {type(exc).__name__}: {exc}")
            raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)
    write_qcio = bool(getattr(run_inputs, "write_qcio", False))

    def write(nodes, fp):
        if nodes:
            Chain.model_validate({"nodes": [n.copy() for n in nodes], "parameters": run_inputs.chain_inputs}
                                 ).write_to_disk(fp, write_qcio=write_qcio)
            return str(fp)
        return None

    species_nodes = [s.node for s in result.species]
    files = {"species": write(species_nodes, output / "species.xyz"),
             "proposals": write([StructureNode(structure=e.proposal.structure) for e in result.edges
                                 if e.proposal.structure is not None], output / "proposals.xyz"),
             "rejected": write(result.rejected, output / "rejected.xyz")}
    (output / "species").mkdir(exist_ok=True)
    species_files = [output / "species" / f"species_{k}.xyz" for k in range(len(species_nodes))]
    for node, fp in zip(species_nodes, species_files):
        write([node], fp)

    pairs = result.connections()
    summary = {
        "structure": structure, "inputs": str(inputs) if inputs else None,
        "settings": {"rounds": rounds, "generator": generator if products is None else f"file:{products}",
                     "n_break": n_break, "n_form": n_form, "form_distance_angstrom": form_distance,
                     "max_products": max_products, "energy_window_kcal": energy_window,
                     "allow_radicals": allow_radicals, "allow_zwitterions": allow_zwitterions,
                     "hessian_validation": bool(validation)},
        "seed_energy": float(result.species[0].node.energy) if result.species[0].node._cached_energy is not None else None,
        "species": [{"index": k, "smiles": s.smiles, "round": s.round, "energy": float(s.node.energy),
                     "rel_energy_kcal": s.rel_energy_kcal, "validation": s.validation, "file": str(species_files[k])}
                    for k, s in enumerate(result.species)],
        "reactions": [{"source": e.source, "target": e.target, "outcome": e.outcome, "proposed_smiles": e.proposal.smiles,
                       "broken": [list(b) for b in e.proposal.broken], "formed": [list(f) for f in e.proposal.formed],
                       "landed_as_proposed": e.intended, "error": e.error or None} for e in result.edges],
        # A named generator swaps in its own enumeration reference; imported
        # products and custom generators only share the animation.
        "methods": {**({k: v for k, v in REFERENCES.items() if k != "kinetics"}
                       if products is None and is_named(generator)
                       else {k: v for k, v in REFERENCES.items() if k == "animation"}),
                    **(get_generator(generator).references if products is None and is_named(generator) else {}),
                    **({"kinetics": REFERENCES["kinetics"]} if steer == "flux" else {})},
        "rounds": result.rounds,
        "steering": {"mode": steer, **({"temperature_K": temperature, "time_s": time_s, "flux_threshold": flux_threshold}
                                        if steer == "flux" else {"energy_window_kcal": energy_window})},
        "steps": [{"a": st.a, "b": st.b, "ts_energy": st.ts_energy, "barrier_kcal": list(bar), "label": st.label,
                   "files": st.files}
                  for st, bar in zip(result.steps, result.kinetics.barriers_kcal if result.kinetics else [])],
        "connections": [list(p) for p in pairs],
        "output_files": files,
    }
    counts = {}
    for e in result.edges:
        counts[e.outcome] = counts.get(e.outcome, 0) + 1
    summary["outcomes"] = counts

    if result.kinetics is not None:
        for rec, f, cmax, cend in zip(summary["species"], result.kinetics.flux, result.kinetics.max_concentration,
                                      result.kinetics.final_concentration):
            rec.update(flux=f, max_concentration=cmax, final_concentration=cend)
    network_note = ""
    if steer == "flux":
        connect = True   # the pair trees exist already: network.json is built from them below
    if connect and pairs:
        if len(pairs) > max_pairs:
            typer.echo(f"{len(pairs)} reactions to connect, capping at --max-pairs={max_pairs}.")
            pairs = pairs[:max_pairs]
        pairs_dir = output / "pairs"
        pairs_dir.mkdir(exist_ok=True)
        if steer != "flux":
            _run_msmep_pairs([s.node.copy() for s in result.species], pairs, pairs_dir, run_inputs,
                             parallel=parallel, parallel_workers=parallel_workers)
        tree_dirs = _completed_tree_dirs(pairs_dir)
        if tree_dirs:
            from mepd.inputs import NetworkInputs
            from mepd.NetworkBuilder import NetworkBuilder

            try:
                pot = NetworkBuilder(data_dir=output, network_inputs=NetworkInputs()).create_rxn_network_from_paths(tree_dirs)
                pot.write_to_disk(output / "network.json")
                files["network"] = str(output / "network.json")
                network_note = f", network.json: {pot.number_of_nodes} nodes / {pot.graph.number_of_edges()} edges"
            except Exception as exc:
                typer.echo(f"Network construction failed: {type(exc).__name__}: {exc}")
        else:
            typer.echo("No path search completed; no network.json written.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    console.print(f"[bold green]{len(result.species) - 1} new species[/bold green] from "
                  f"{len(result.edges)} proposed reactions ({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))}); "
                  f"{len(result.connections())} distinct reactions to connect{network_note}. Results in {output}/",
                  highlight=False)
