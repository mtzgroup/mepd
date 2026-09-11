"""Minimal command-line interface for mepd.

This is intentionally a thin CLI, scoped to what this package currently
supports: running a single- or multi-step nudged-elastic-band (NEB)
optimization between two endpoint structures (including recursive
autosplitting via MSMEP, serial or parallel), transition-state
optimization, and network-completion (building/completing a reaction
network graph from already-computed MSMEP/IRC results).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List, Optional

import typer
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import NetworkInputs, RunInputs

app = typer.Typer(help="mepd: minimum-energy-path discovery tools.")


def _format_run_inputs_value(value) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        return ", ".join(f"{k}={_format_run_inputs_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return str(list(value)) if value else "[]"
    if isinstance(value, bool):
        return "[green]True[/green]" if value else "[red]False[/red]"
    if value == "":
        return "[dim]--[/dim]"
    return str(value)


def _dataclass_field_values(obj) -> dict:
    """All active fields of a dataclass instance (engine/optimizer), with
    their currently-resolved values -- including ones left at their class
    default and never mentioned in the TOML. Private/internal fields (leading
    underscore) are excluded."""
    import dataclasses

    if obj is None or not dataclasses.is_dataclass(obj):
        return {}
    return {
        f.name: getattr(obj, f.name, None)
        for f in dataclasses.fields(obj)
        if not f.name.startswith("_")
    }


def _echo_run_inputs_summary(run_inputs: RunInputs) -> None:
    """Print the settings actually in effect for this run (after any
    --inputs TOML has been loaded and any CLI-flag overrides applied)."""
    try:
        config = run_inputs.to_dict()
    except Exception as exc:
        typer.echo(f"(could not render RunInputs summary: {exc})", err=True)
        return

    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold cyan]RunInputs[/bold cyan]")

    # The raw *_kwds dicts only reflect what the TOML happened to set --
    # replaced below with the fully resolved engine/optimizer objects, which
    # include every active field even when left at its default.
    for key in ("optimizer_kwds", "ase_engine_kwds", "gxtb_engine_kwds"):
        config.pop(key, None)

    top_level, sections = {}, {}
    for key, value in config.items():
        if isinstance(value, dict):
            sections[key] = value
        else:
            top_level[key] = value

    def _new_table(title: str | None = None) -> Table:
        table = Table(title=title, box=box.ROUNDED, show_header=True, title_style="bold")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value")
        return table

    table = _new_table()
    for key, value in top_level.items():
        table.add_row(key, _format_run_inputs_value(value))
    console.print(table)

    for key, value in sections.items():
        table = _new_table(title=key)
        if not value:
            table.add_row("[dim](empty)[/dim]", "")
        else:
            for sub_key, sub_value in value.items():
                table.add_row(sub_key, _format_run_inputs_value(sub_value))
        console.print(table)

    for label, obj in (("engine", run_inputs.engine), ("optimizer", run_inputs.optimizer)):
        fields = _dataclass_field_values(obj)
        table = _new_table(title=f"{label} ({type(obj).__name__})")
        if not fields:
            table.add_row("[dim](no fields)[/dim]", "")
        else:
            for sub_key, sub_value in fields.items():
                table.add_row(sub_key, _format_run_inputs_value(sub_value))
        console.print(table)


def _normalized_path_method(method: str) -> str:
    return str(method or "").strip().upper().replace("_", "-")


def _build_path_minimizer(initial_chain: Chain, run_inputs: RunInputs):
    """Dispatch to the path minimizer selected by `run_inputs.path_min_method`.

    Mirrors the (much larger) dispatch in the source neb-dynamics MSMEP class,
    trimmed to only the path minimizers this package actually ports in Phase 1.
    """
    from mepd.neb import NEB
    from mepd.pathminimizers.fneb import FreezingNEB
    from mepd.pathminimizers.nebdlf import DLFindNEB
    from mepd.pathminimizers.geometric_neb import GeometricNEB

    method = _normalized_path_method(run_inputs.path_min_method)
    if method == "NEB":
        return NEB(
            initial_chain=initial_chain,
            parameters=run_inputs.path_min_inputs,
            optimizer=run_inputs.optimizer,
            engine=run_inputs.engine,
        )
    if method == "FNEB":
        return FreezingNEB(
            initial_chain=initial_chain,
            engine=run_inputs.engine,
            parameters=run_inputs.path_min_inputs,
            optimizer=run_inputs.optimizer,
            gi_inputs=run_inputs.gi_inputs,
        )
    if method == "NEB-DLF":
        return DLFindNEB(
            initial_chain=initial_chain,
            engine=run_inputs.engine,
            parameters=run_inputs.path_min_inputs,
        )
    if method == "GEOMETRIC-NEB":
        return GeometricNEB(
            initial_chain=initial_chain,
            engine=run_inputs.engine,
            parameters=run_inputs.path_min_inputs,
        )
    raise typer.BadParameter(
        f"Unsupported path_min_method '{run_inputs.path_min_method}'. "
        "This build supports: NEB, FNEB, NEB-DLF, GEOMETRIC-NEB."
    )


def _load_endpoint(fp: Path, charge: Optional[int], multiplicity: Optional[int]) -> Structure:
    structure = Structure.open(str(fp))
    updates = {}
    if charge is not None:
        updates["charge"] = charge
    if multiplicity is not None:
        updates["multiplicity"] = multiplicity
    if updates:
        structure = structure.model_copy(update=updates)
    return structure


def _load_structure_from_smiles_or_xyz(
    value: str, charge: Optional[int], multiplicity: Optional[int]
) -> Structure:
    """Load a Structure from an xyz file path, or -- if `value` isn't an
    existing file -- embed it in 3D from a SMILES string."""
    path = Path(value)
    if path.exists():
        return _load_endpoint(path, charge, multiplicity)

    import qcinf

    kwargs = {}
    if charge is not None:
        kwargs["charge"] = charge
    if multiplicity is not None:
        kwargs["multiplicity"] = multiplicity
    try:
        return qcinf.smiles_to_structure(value, **kwargs)
    except Exception as exc:
        raise typer.BadParameter(
            f"'{value}' is neither an existing xyz file nor a valid SMILES string "
            f"({type(exc).__name__}: {exc})."
        )


def _geometry_optimizer_keywords(run_inputs: RunInputs, *, default_maxiter: int = 500) -> dict:
    keywords = {"coordsys": "cart", "maxit": int(default_maxiter)}
    keywords.update(dict(getattr(run_inputs, "geometry_optimizer_kwds", {}) or {}))
    return keywords


def _refuse_unconverged_endpoint(label: str, exc: Exception, run_inputs: RunInputs) -> None:
    """Hard-stop: an endpoint failed to actually converge during --minimize-ends.

    Unlike other minimization failures (engine doesn't support optimization,
    a transient crash), a reported non-convergence means we'd otherwise hand
    NEB an endpoint that isn't really a minimum -- silently proceeding would
    make the whole run meaningless. Refuse instead of guessing.
    """
    current_maxiter = _geometry_optimizer_keywords(run_inputs).get("maxit")
    typer.echo(
        f"{label.capitalize()} endpoint minimization did not converge: {exc}\n"
        "Refusing to proceed with an un-minimized endpoint.\n"
        "Options:\n"
        "  1. Provide an already-minimized structure for this endpoint instead of using --minimize-ends.\n"
        "  2. Increase the optimizer's iteration budget by setting maxit (or maxiter) under "
        f"[geometry_optimizer_kwds] in your RunInputs TOML (current: {current_maxiter})."
    )
    raise typer.Exit(code=1)


def _minimize_endpoints(start_node, end_node, run_inputs: RunInputs):
    """Optimize the start/end endpoint geometries before building the initial chain.

    Mirrors the source neb-dynamics CLI's `--minimize-ends` behavior: prefer a
    batched call if the engine supports it, fall back to per-node calls. Most
    failures (engine doesn't support optimization, a transient crash) keep the
    original input geometry with a warning rather than aborting the run -- but
    a reported non-convergence (GeometryOptimizationNotConvergedError) is a
    hard stop, since silently continuing would run NEB on a non-minimum.
    """
    from mepd.errors import GeometryOptimizationNotConvergedError

    typer.echo("Minimizing input endpoints...")
    keywords = _geometry_optimizer_keywords(run_inputs)
    endpoints = [start_node, end_node]
    labels = ("start", "end")

    batch_optimizer = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    if callable(batch_optimizer):
        try:
            try:
                trajectories = batch_optimizer(endpoints, keywords=keywords)
            except TypeError:
                trajectories = batch_optimizer(endpoints)
        except GeometryOptimizationNotConvergedError as exc:
            _refuse_unconverged_endpoint("an", exc, run_inputs)
        except Exception as exc:
            typer.echo(f"Endpoint batch minimization failed ({type(exc).__name__}: {exc}); keeping input geometries.")
        else:
            if not isinstance(trajectories, (list, tuple)) or len(trajectories) < 2:
                typer.echo("Endpoint batch minimization returned an unexpected result; keeping input geometries.")
            else:
                for i, label in enumerate(labels):
                    trajectory = trajectories[i]
                    if trajectory:
                        endpoints[i] = trajectory[-1]
                    else:
                        typer.echo(f"{label.capitalize()} endpoint optimization returned an empty trajectory; keeping input geometry.")
        return endpoints[0], endpoints[1]

    single_optimizer = getattr(run_inputs.engine, "compute_geometry_optimization", None)
    if not callable(single_optimizer):
        typer.echo(f"Engine {type(run_inputs.engine).__name__} does not support geometry optimization; keeping input geometries.")
        return start_node, end_node

    for i, label in enumerate(labels):
        typer.echo(f"Minimizing {label} endpoint...")
        try:
            trajectory = single_optimizer(endpoints[i], keywords=keywords)
            if trajectory:
                endpoints[i] = trajectory[-1]
            else:
                typer.echo(f"{label.capitalize()} endpoint optimization returned an empty trajectory; keeping input geometry.")
        except GeometryOptimizationNotConvergedError as exc:
            _refuse_unconverged_endpoint(label, exc, run_inputs)
        except Exception as exc:
            typer.echo(f"{label.capitalize()} endpoint minimization failed ({type(exc).__name__}: {exc}); keeping input geometry.")

    return endpoints[0], endpoints[1]


def _completed_tree_dirs(completion_dir: Path) -> list[Path]:
    if not completion_dir.is_dir():
        return []
    return sorted(
        p / "tree" for p in completion_dir.iterdir()
        if (p / "tree" / "adj_matrix.txt").exists()
    )


def _run_network_completion(
    *,
    output: Path,
    initial_tree: Path,
    start_node,
    end_node,
    run_inputs: RunInputs,
    mode: str,
    max_followups: int,
    parallel: bool,
    parallel_workers: Optional[int],
) -> None:
    """Auto-generate and run follow-up NEB/MSMEP requests connecting
    newly-discovered intermediates, then build the completed reaction network.

    Resumable with no separate manifest/state file: each follow-up pair's
    MSMEP output lives at a deterministic path
    (<output>/network_completion/pair_<i>_<j>/tree/), and a pair is skipped
    if that directory already holds a completed tree. Re-running the exact
    same command against the same --output therefore picks up wherever a
    prior run left off -- the directory tree on disk IS the resume state.
    """
    from mepd.msmep import MSMEP
    from mepd.NetworkBuilder import NetworkBuilder
    import mepd.chainhelpers as ch

    completion_dir = output / "network_completion"
    completion_dir.mkdir(parents=True, exist_ok=True)

    def _tree_dirs() -> list[Path]:
        return [initial_tree, *_completed_tree_dirs(completion_dir)]

    builder = NetworkBuilder(data_dir=output, network_inputs=NetworkInputs())
    structures, edges = builder._load_network_data(_tree_dirs())
    start_ind = int(builder._get_ind_td(ref_list=structures, td=start_node))
    end_ind = int(builder._get_ind_td(ref_list=structures, td=end_node))

    def _has_edge(i: int, j: int) -> bool:
        return f"{i}-{j}" in edges or f"{j}-{i}" in edges

    if mode == "linear":
        others = [k for k in range(len(structures)) if k not in (start_ind, end_ind)]
        candidates = [(start_ind, k) for k in others] + [(k, end_ind) for k in others]
    else:  # all-to-all
        candidates = [
            (i, j) for i in range(len(structures)) for j in range(i + 1, len(structures))
        ]
    candidates = [(i, j) for i, j in candidates if not _has_edge(i, j)]

    if len(candidates) > max_followups:
        typer.echo(
            f"{len(candidates)} candidate follow-up pairs found, capping at "
            f"--network-max-followups={max_followups}."
        )
        candidates = candidates[:max_followups]
    if not candidates:
        typer.echo("No new candidate pairs for --network-completion.")

    for i, j in candidates:
        pair_dir = completion_dir / f"pair_{i}_{j}"
        tree_dir = pair_dir / "tree"
        if (tree_dir / "adj_matrix.txt").exists():
            typer.echo(f"Skipping pair ({i}, {j}): already completed.")
            continue
        pair_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Running follow-up NEB/MSMEP for pair ({i}, {j})...")
        try:
            seed_chain = Chain.model_validate({
                "nodes": [structures[i], structures[j]],
                "parameters": copy.deepcopy(run_inputs.chain_inputs),
            })
            pair_chain = ch.run_geodesic(
                chain=seed_chain,
                chain_inputs=copy.deepcopy(run_inputs.chain_inputs),
                nimages=run_inputs.gi_inputs.nimages,
                friction=run_inputs.gi_inputs.friction,
                nudge=run_inputs.gi_inputs.nudge,
                random_seed=run_inputs.gi_inputs.random_seed,
                align=run_inputs.gi_inputs.align,
                **(run_inputs.gi_inputs.extra_kwds or {}),
            )
            msmep = MSMEP(inputs=run_inputs)
            if parallel:
                pair_history = msmep.run_parallel_recursive_minimize(
                    pair_chain, max_workers=parallel_workers
                )
            else:
                pair_history = msmep.run_recursive_minimize(pair_chain)
            pair_history.write_to_disk(tree_dir)
        except Exception as exc:
            typer.echo(f"Follow-up pair ({i}, {j}) failed ({type(exc).__name__}: {exc}); skipping.")
            continue

    builder = NetworkBuilder(data_dir=output, network_inputs=NetworkInputs())
    try:
        pot = builder.create_rxn_network_from_paths(_tree_dirs())
    except Exception as exc:
        typer.echo(f"Network construction failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    network_path = output / "network.json"
    pot.write_to_disk(network_path)
    typer.echo(
        f"Wrote completed network to {network_path} "
        f"({pot.number_of_nodes} nodes, {pot.graph.number_of_edges()} edges)"
    )


def _optimize_ts_and_irc(
    ts_guess_node,
    run_inputs: RunInputs,
    output: Path,
    *,
    run_irc: bool,
    label: str = "ts",
):
    """Optimize a TS-guess node with the engine and, if requested, follow up
    with an IRC -- writes <label>.xyz (and <label>_irc.xyz) into `output`.

    Shared by `ts` and `run --use-tsopt`. Never raises/exits itself: returns
    the optimized StructureNode, or None on failure/no support, so each
    caller decides whether that is fatal (a standalone `ts` invocation should
    exit non-zero; a TS opt launched automatically after `run` should just
    warn and let the NEB result stand).
    """
    from mepd.nodes.node import StructureNode

    compute_ts = getattr(run_inputs.engine, "compute_transition_state", None)
    if not callable(compute_ts):
        typer.echo(
            f"Engine {type(run_inputs.engine).__name__} does not support "
            "transition-state optimization."
        )
        return None

    typer.echo(f"Optimizing transition state ({label})...")
    try:
        result = compute_ts(node=ts_guess_node)
    except Exception as exc:
        typer.echo(f"Transition-state optimization failed ({label}): {type(exc).__name__}: {exc}")
        return None

    if not isinstance(result, StructureNode):
        typer.echo(
            f"Transition-state optimization did not converge to a usable structure "
            f"({label}; engine returned {type(result).__name__})."
        )
        return None

    ts_node = result
    output.mkdir(parents=True, exist_ok=True)
    ts_path = output / f"{label}.xyz"
    ts_path.write_text(ts_node.structure.to_xyz())
    typer.echo(f"Wrote optimized TS structure to {ts_path}")

    if not run_irc:
        return ts_node

    typer.echo(f"Computing IRC ({label})...")
    irc_fn = getattr(run_inputs.engine, "compute_irc_chain", None)
    try:
        if callable(irc_fn):
            irc_chain = irc_fn(ts_node)
        else:
            from mepd.irc import compute_irc_chain_with_geometric
            irc_chain = compute_irc_chain_with_geometric(run_inputs.engine, ts_node)
    except Exception as exc:
        typer.echo(f"IRC computation failed ({label}; {type(exc).__name__}: {exc}); TS structure was still written.")
        return ts_node

    irc_path = output / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
    irc_chain.write_to_disk(irc_path)
    typer.echo(f"Wrote IRC path to {irc_path}")
    return ts_node


@app.command("run")
def run(
    start: Path = typer.Option(..., "--start", exists=True, help="Path to the start-structure xyz file."),
    end: Path = typer.Option(..., "--end", exists=True, help="Path to the end-structure xyz file."),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True,
        help="Path to a RunInputs TOML file. Uses built-in defaults if omitted.",
    ),
    charge: Optional[int] = typer.Option(
        None, "--charge", help="Override the molecular charge on both endpoints."
    ),
    multiplicity: Optional[int] = typer.Option(
        None, "--multiplicity", help="Override the spin multiplicity on both endpoints."
    ),
    minimize_ends: bool = typer.Option(
        False, "--minimize-ends",
        help="Optimize the start/end endpoint geometries before running NEB.",
    ),
    recursive: bool = typer.Option(
        False, "--recursive",
        help="Recursively autosplit the path (MSMEP) instead of running a single NEB.",
    ),
    parallel: bool = typer.Option(
        False, "--parallel",
        help="Recursively autosplit the path (MSMEP), evaluating split branches in "
        "parallel. Implies recursive splitting; cannot be combined with --recursive.",
    ),
    parallel_workers: Optional[int] = typer.Option(
        None, "--parallel-workers",
        help="Maximum number of concurrent workers for --parallel. Defaults to "
        "min(4, cpu count).",
    ),
    network_completion: bool = typer.Option(
        False, "--network-completion",
        help="After the recursive MSMEP run, auto-generate and run follow-up "
        "NEB/MSMEP requests connecting newly-discovered intermediates, then "
        "build the completed reaction network (archaically: --network-splits). "
        "Implies --recursive if neither --recursive nor --parallel is given.",
    ),
    network_completion_mode: str = typer.Option(
        "linear", "--network-completion-mode",
        help="Follow-up pair strategy: 'linear' connects each discovered "
        "intermediate to the original start/end only; 'all-to-all' also "
        "connects every other non-adjacent pair of discovered structures.",
    ),
    network_max_followups: int = typer.Option(
        25, "--network-max-followups",
        help="Cap on the number of follow-up pairs --network-completion will run "
        "(guards against combinatorial blowup, especially with --network-completion-mode all-to-all).",
    ),
    validate_minima_with_hessian: bool = typer.Option(
        True, "--validate-minima-with-hessian/--no-validate-minima-with-hessian", "-H/-noH",
        help="When a minima-based autosplit is proposed during MSMEP, compute "
        "Hessians for optimized split candidates and reject candidates with "
        "significant imaginary modes. On by default -- this is a correctness "
        "check, not a convenience.",
    ),
    hessian_minimum_frequency_cutoff: float = typer.Option(
        0.0, "--hessian-minimum-frequency-cutoff",
        help="Minimum allowed frequency (cm^-1) for --validate-minima-with-hessian.",
    ),
    hessian_minima_rescue_displacement: float = typer.Option(
        0.1, "--hessian-minima-rescue-displacement",
        help="Displacement (bohr) applied along the lowest-frequency mode when "
        "rescuing a Hessian-rejected minimum, for --validate-minima-with-hessian.",
    ),
    use_tsopt: bool = typer.Option(
        False, "--use-tsopt",
        help="After the NEB/MSMEP run, automatically optimize a transition state "
        "from each result's TS-guess node (the highest-energy interior image). "
        "Failure is a warning, not a fatal error -- the NEB result still stands.",
    ),
    irc: bool = typer.Option(
        False, "--irc",
        help="Follow up each --use-tsopt transition state with an IRC. Requires --use-tsopt.",
    ),
    output: Path = typer.Option(
        Path("mepd_output"), "--output", "-o",
        help="Directory to write the optimized trajectory/energies into.",
    ),
) -> None:
    """Run a NEB optimization between two endpoint structures."""
    from mepd.nodes.node import StructureNode
    import mepd.chainhelpers as ch

    if recursive and parallel:
        raise typer.BadParameter(
            "--parallel cannot be combined with --recursive. Use one mode."
        )
    if network_completion_mode not in ("linear", "all-to-all"):
        raise typer.BadParameter(
            "--network-completion-mode must be 'linear' or 'all-to-all'."
        )
    if irc and not use_tsopt:
        raise typer.BadParameter("--irc requires --use-tsopt.")
    if network_completion and not recursive and not parallel:
        typer.echo("--network-completion requires recursive splitting; enabling --recursive.")
        recursive = True

    run_inputs = RunInputs.open(inputs) if inputs is not None else RunInputs()
    run_inputs.path_min_inputs.validate_minima_with_hessian = validate_minima_with_hessian
    run_inputs.path_min_inputs.hessian_minimum_frequency_cutoff = hessian_minimum_frequency_cutoff
    run_inputs.path_min_inputs.hessian_minima_rescue_displacement = hessian_minima_rescue_displacement
    _echo_run_inputs_summary(run_inputs)

    start_structure = _load_endpoint(start, charge, multiplicity)
    end_structure = _load_endpoint(end, charge, multiplicity)

    start_node = StructureNode(structure=start_structure)
    end_node = StructureNode(structure=end_structure)

    if minimize_ends:
        start_node, end_node = _minimize_endpoints(start_node, end_node, run_inputs)

    seed_chain = Chain.model_validate({
        "nodes": [start_node, end_node],
        "parameters": copy.deepcopy(run_inputs.chain_inputs),
    })

    typer.echo("Building initial path via geodesic interpolation...")
    initial_chain = ch.run_geodesic(
        chain=seed_chain,
        chain_inputs=copy.deepcopy(run_inputs.chain_inputs),
        nimages=run_inputs.gi_inputs.nimages,
        friction=run_inputs.gi_inputs.friction,
        nudge=run_inputs.gi_inputs.nudge,
        random_seed=run_inputs.gi_inputs.random_seed,
        align=run_inputs.gi_inputs.align,
        **(run_inputs.gi_inputs.extra_kwds or {}),
    )

    output.mkdir(parents=True, exist_ok=True)

    if recursive or parallel:
        from mepd.msmep import MSMEP
        from mepd.TreeNode import TreeNode

        tree_path = output / "tree"
        if network_completion and (tree_path / "adj_matrix.txt").exists():
            typer.echo(f"Skipping initial MSMEP run: {tree_path} already complete.")
            history = TreeNode.read_from_disk(
                tree_path,
                neb_parameters=run_inputs.path_min_inputs,
                chain_parameters=run_inputs.chain_inputs,
                gi_parameters=run_inputs.gi_inputs,
                optimizer=run_inputs.optimizer,
                engine=run_inputs.engine,
            )
        else:
            msmep = MSMEP(inputs=run_inputs)
            if parallel:
                typer.echo("Running parallel recursive autosplitting (MSMEP)...")
                history = msmep.run_parallel_recursive_minimize(
                    initial_chain, max_workers=parallel_workers
                )
            else:
                typer.echo("Running recursive autosplitting (MSMEP)...")
                history = msmep.run_recursive_minimize(initial_chain)

            history.write_to_disk(tree_path)
            typer.echo(f"Wrote split-tree history to {tree_path}")

        if network_completion:
            _run_network_completion(
                output=output,
                initial_tree=tree_path,
                start_node=start_node,
                end_node=end_node,
                run_inputs=run_inputs,
                mode=network_completion_mode,
                max_followups=network_max_followups,
                parallel=parallel,
                parallel_workers=parallel_workers,
            )

        try:
            final_chain = history.output_chain
        except ValueError as exc:
            typer.echo(f"Could not assemble a final output chain: {exc}")
            return
        out_path = output / "mep_output.xyz"
        final_chain.write_to_disk(out_path)
        typer.echo(f"Wrote assembled output path to {out_path}")

        if use_tsopt:
            for leaf in history.ordered_leaves:
                if not leaf.data or not leaf.data.chain_trajectory:
                    continue
                _optimize_ts_and_irc(
                    leaf.data.chain_trajectory[-1].get_ts_node(),
                    run_inputs,
                    output,
                    run_irc=irc,
                    label=f"ts_leaf_{leaf.index}",
                )
        return

    minimizer = _build_path_minimizer(initial_chain, run_inputs)

    try:
        minimizer.optimize_chain()
        typer.echo("NEB converged.")
    except Exception as exc:  # e.g. NoneConvergedException
        typer.echo(f"NEB did not fully converge: {exc}")

    final_chain = minimizer.chain_trajectory[-1] if minimizer.chain_trajectory else initial_chain
    run_inputs.engine.compute_energies(final_chain)
    run_inputs.engine.compute_gradients(final_chain)

    out_path = output / "mep_output.xyz"
    final_chain.write_to_disk(out_path)
    typer.echo(f"Wrote optimized path to {out_path}")

    if use_tsopt:
        _optimize_ts_and_irc(final_chain.get_ts_node(), run_inputs, output, run_irc=irc, label="ts")


@app.command("ts")
def ts(
    guess: Path = typer.Option(..., "--guess", exists=True, help="Path to the TS-guess xyz file."),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True,
        help="Path to a RunInputs TOML file. Uses built-in defaults if omitted.",
    ),
    charge: Optional[int] = typer.Option(
        None, "--charge", help="Override the molecular charge on the guess structure."
    ),
    multiplicity: Optional[int] = typer.Option(
        None, "--multiplicity", help="Override the spin multiplicity on the guess structure."
    ),
    irc: bool = typer.Option(
        False, "--irc", help="Follow up with an IRC from the optimized transition state."
    ),
    output: Path = typer.Option(
        Path("mepd_ts_output"), "--output", "-o",
        help="Directory to write the optimized TS structure (and IRC path) into.",
    ),
) -> None:
    """Optimize a transition-state guess structure."""
    from mepd.nodes.node import StructureNode

    run_inputs = RunInputs.open(inputs) if inputs is not None else RunInputs()
    _echo_run_inputs_summary(run_inputs)

    guess_structure = _load_endpoint(guess, charge, multiplicity)
    guess_node = StructureNode(structure=guess_structure)

    ts_node = _optimize_ts_and_irc(guess_node, run_inputs, output, run_irc=irc, label="ts")
    if ts_node is None:
        raise typer.Exit(code=1)


@app.command("hessian-sample")
def hessian_sample(
    structure: str = typer.Argument(
        ..., help="Seed structure: a path to an xyz file, or a SMILES string."
    ),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True,
        help="Path to a RunInputs TOML file. Uses built-in defaults if omitted.",
    ),
    charge: Optional[int] = typer.Option(
        None, "--charge", help="Override the molecular charge on the seed structure."
    ),
    multiplicity: Optional[int] = typer.Option(
        None, "--multiplicity", help="Override the spin multiplicity on the seed structure."
    ),
    dr: float = typer.Option(
        0.1, "--dr",
        help="Per-atom displacement factor; effective mode displacement is dr * n_atoms.",
    ),
    max_candidates: int = typer.Option(
        100, "--max-candidates",
        help="Hard cap on the number of displaced candidates generated/optimized. "
        "Sampling every normal mode in both directions is otherwise unbounded for "
        "large molecules -- this is the control that limits the exploration.",
    ),
    maxiter: int = typer.Option(
        500, "--maxiter",
        help="Maximum geometry-optimization steps for each displaced candidate.",
    ),
    output: Path = typer.Option(
        Path("mepd_hessian_sample_output"), "--output", "-o",
        help="Directory to write results into.",
    ),
) -> None:
    """Explore minima near a seed structure by displacing along Hessian normal modes."""
    from qcconst.constants import HARTREE_TO_KCAL_PER_MOL

    from mepd.hessian_sample import run_hessian_sample
    from mepd.nodes.node import StructureNode

    if dr <= 0:
        raise typer.BadParameter("--dr must be positive.")
    if max_candidates <= 0:
        raise typer.BadParameter("--max-candidates must be a positive integer.")
    if maxiter <= 0:
        raise typer.BadParameter("--maxiter must be a positive integer.")

    run_inputs = RunInputs.open(inputs) if inputs is not None else RunInputs()
    _echo_run_inputs_summary(run_inputs)

    seed_structure = _load_structure_from_smiles_or_xyz(structure, charge, multiplicity)
    seed_node = StructureNode(structure=seed_structure)

    typer.echo(
        f"Computing Hessian and sampling normal modes (dr={dr:g}, "
        f"max_candidates={max_candidates}, maxiter={maxiter})..."
    )
    try:
        result = run_hessian_sample(
            seed_node,
            run_inputs.engine,
            dr=dr,
            max_candidates=max_candidates,
            maxiter=maxiter,
            chain_inputs=run_inputs.chain_inputs,
        )
    except Exception as exc:
        typer.echo(f"Hessian sampling failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    if result.candidates_clipped:
        typer.echo(
            f"Reached --max-candidates ({max_candidates}); not all normal modes were sampled."
        )

    output.mkdir(parents=True, exist_ok=True)
    write_qcio = bool(getattr(run_inputs, "write_qcio", False))

    hessian_fp = output / "hessian.json"
    if hasattr(result.hessian_result, "save"):
        try:
            result.hessian_result.save(hessian_fp)
        except Exception:
            hessian_fp = None
    else:
        hessian_fp = None

    def _write_chain(nodes, filename: str):
        if not nodes:
            return None
        fp = output / filename
        chain_out = Chain.model_validate({
            "nodes": [n.copy() for n in nodes],
            "parameters": run_inputs.chain_inputs,
        })
        chain_out.write_to_disk(fp, write_qcio=write_qcio)
        return fp

    displaced_fp = _write_chain(result.displaced_nodes, "displaced.xyz")
    optimized_fp = _write_chain(result.optimized_nodes, "optimized.xyz")
    unique_fp = _write_chain(result.unique_minima, "unique.xyz")

    def _candidate_meta_dict(meta) -> dict:
        return {
            "mode_index": meta.mode_index,
            "direction": meta.direction,
            "frequency_wavenumber": meta.frequency_wavenumber,
            "dr": meta.dr,
            "effective_dr": meta.effective_dr,
        }

    summary_payload = {
        "structure": structure,
        "inputs": str(inputs) if inputs is not None else None,
        "dr": dr,
        "max_candidates": max_candidates,
        "maxiter": maxiter,
        "seed_energy": result.seed_energy,
        "normal_modes_total": len(result.frequencies_wavenumber),
        "frequencies_wavenumber": result.frequencies_wavenumber,
        "displaced_candidates": len(result.displaced_nodes),
        "candidates_clipped": result.candidates_clipped,
        "optimized_candidates": len(result.optimized_nodes),
        "failed_candidates": len(result.failed_candidates),
        "unique_minima": len(result.unique_minima),
        "optimization_submission_mode": result.optimization_submission_mode,
        "chain_inputs_thresholds": {
            "node_rms_thre": run_inputs.chain_inputs.node_rms_thre,
            "node_ene_thre": run_inputs.chain_inputs.node_ene_thre,
        },
        "unique_minima_energies_eh": [float(n.energy) for n in result.unique_minima],
        "unique_minima_rel_energies_kcal_mol": [
            (float(n.energy) - result.seed_energy) * float(HARTREE_TO_KCAL_PER_MOL)
            for n in result.unique_minima
        ],
        "displaced_metadata": [_candidate_meta_dict(m) for m in result.displaced_metadata],
        "optimized_metadata": [_candidate_meta_dict(m) for m in result.optimized_metadata],
        "failed_candidate_details": [
            {**_candidate_meta_dict(f["meta"]), "error": f["error"]}
            for f in result.failed_candidates
        ],
        "output_files": {
            "hessian": str(hessian_fp) if hessian_fp else None,
            "displaced": str(displaced_fp) if displaced_fp else None,
            "optimized": str(optimized_fp) if optimized_fp else None,
            "unique": str(unique_fp) if unique_fp else None,
        },
    }
    summary_fp = output / "summary.json"
    summary_fp.write_text(json.dumps(summary_payload, indent=2))

    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    result_table = Table(box=box.ROUNDED, show_header=False)
    result_table.add_column(style="bold cyan")
    result_table.add_column(style="white")
    result_table.add_row("Normal modes", str(len(result.frequencies_wavenumber)))
    result_table.add_row("Displaced candidates", str(len(result.displaced_nodes)))
    result_table.add_row("Optimized candidates", str(len(result.optimized_nodes)))
    result_table.add_row("Failed candidates", str(len(result.failed_candidates)))
    result_table.add_row("Unique minima", str(len(result.unique_minima)))
    result_table.add_row("Optimization mode", result.optimization_submission_mode)
    if hessian_fp:
        result_table.add_row("Hessian", str(hessian_fp))
    if displaced_fp:
        result_table.add_row("Displaced", str(displaced_fp))
    if optimized_fp:
        result_table.add_row("Optimized", str(optimized_fp))
    if unique_fp:
        result_table.add_row("Unique", str(unique_fp))
    result_table.add_row("Summary", str(summary_fp))
    Console().print(
        Panel(
            result_table,
            title="[bold green]Hessian Sample Complete[/bold green]",
            border_style="green",
        )
    )

    if not result.optimized_nodes:
        typer.echo("All displaced-candidate optimizations failed.")
        raise typer.Exit(code=1)


def _load_visualization_object(result_path: Path, charge: int, multiplicity: int):
    """Load whatever mepd result `result_path` points to, for `mepd
    visualize`: a chain xyz file, a network.json (a `Pot`), a split-tree
    directory (has adj_matrix.txt), or a bare NEB history directory (has
    traj_*.xyz but no adj_matrix.txt -- e.g. a manually saved
    `<name>_history/` folder)."""
    from mepd.inputs import ChainInputs
    from mepd.neb import NEB
    from mepd.pot import Pot
    from mepd.TreeNode import TreeNode

    if result_path.is_dir():
        if (result_path / "adj_matrix.txt").exists():
            return TreeNode.read_from_disk(
                result_path, chain_parameters=ChainInputs(), charge=charge, multiplicity=multiplicity
            )
        if list(result_path.glob("traj_*.xyz")):
            return NEB.read_from_disk(
                fp=result_path / "_unused.xyz",
                history_folder=result_path,
                chain_parameters=ChainInputs(),
                charge=charge,
                multiplicity=multiplicity,
            )
        raise typer.BadParameter(
            f"'{result_path}' is a directory but has neither adj_matrix.txt (a split-tree) "
            "nor traj_*.xyz files (a NEB history)."
        )

    if result_path.suffix == ".json":
        return Pot.read_from_disk(result_path)

    return Chain.from_xyz(result_path, ChainInputs(), charge=charge, spinmult=multiplicity)


@app.command("visualize")
def visualize(
    result_path: Path = typer.Argument(
        ..., exists=True,
        help="Path to a mepd result: a chain xyz file (mep_output.xyz, unique.xyz -- "
        "a matching <stem>.energies sidecar, if present, is used for the energy profile), "
        "a network.json, or a split-tree/NEB-history directory.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output HTML file path. Defaults to <result_path stem>_visualize.html "
        "next to the input.",
    ),
    charge: int = typer.Option(0, "--charge", help="Charge used when reading the xyz geometries."),
    multiplicity: int = typer.Option(
        1, "--multiplicity", help="Spin multiplicity used when reading the xyz geometries."
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Do not automatically open the HTML file in a browser."
    ),
    show_atom_indices: bool = typer.Option(
        False, "--show-atom-indices", help="Label atom indices in the structure viewer."
    ),
) -> None:
    """Render an interactive visualization: a frame scrubber (with a
    highlighted energy-profile point) for a chain, plus -- for a split-tree
    or network.json -- a diagram of tree nodes/network edges to click
    through, and a trajectory-step slider for whichever one is selected."""
    try:
        obj = _load_visualization_object(result_path, charge, multiplicity)
    except Exception as exc:
        typer.echo(f"Could not load '{result_path}': {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    from mepd import viz

    try:
        html = viz.render_visualization_html(
            obj, title=result_path.stem, show_atom_indices=show_atom_indices
        )
    except (ValueError, TypeError) as exc:
        typer.echo(f"Could not build visualization: {exc}")
        raise typer.Exit(code=1)

    out_fp = output if output is not None else result_path.parent / f"{result_path.stem}_visualize.html"
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(html, encoding="utf-8")
    typer.echo(f"Wrote visualization to {out_fp}")

    if not no_open:
        import webbrowser
        webbrowser.open(out_fp.resolve().as_uri())


@app.command("network-build")
def network_build(
    tree: List[Path] = typer.Option(
        ..., "--tree", exists=True, file_okay=False,
        help="Path to a completed MSMEP output tree directory (as written by "
        "`mepd run --recursive`/`--parallel` to <output>/tree). Repeatable to "
        "combine results from multiple runs into one network.",
    ),
    output: Path = typer.Option(
        Path("network.json"), "--output", "-o",
        help="Path to write the resulting network (Pot) JSON to.",
    ),
) -> None:
    """Build/dedupe a reaction network from one or more completed MSMEP trees."""
    from mepd.NetworkBuilder import NetworkBuilder

    builder = NetworkBuilder(data_dir=tree[0].parent)
    try:
        pot = builder.create_rxn_network_from_paths(list(tree))
    except Exception as exc:
        typer.echo(f"Network construction failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    pot.write_to_disk(output)
    typer.echo(
        f"Wrote network to {output} "
        f"({pot.number_of_nodes} nodes, {pot.graph.number_of_edges()} edges)"
    )


@app.command("irc-network")
def irc_network(
    directory: Path = typer.Option(
        ..., "--dir", exists=True, file_okay=False,
        help="Directory containing IRC xyz/.energies file pairs.",
    ),
    pattern: str = typer.Option(
        "*.xyz", "--pattern", help="Glob pattern used to find IRC xyz files.",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", help="Search the directory recursively.",
    ),
    charge: int = typer.Option(0, "--charge", help="Molecular charge of the IRC structures."),
    multiplicity: int = typer.Option(
        1, "--multiplicity", help="Spin multiplicity of the IRC structures.",
    ),
    output: Path = typer.Option(
        Path("network.json"), "--output", "-o",
        help="Path to write the resulting network (Pot) JSON to.",
    ),
) -> None:
    """Build a reaction network by scanning a directory of IRC xyz/energy pairs."""
    from mepd.irc_network import build_irc_network

    try:
        scan = build_irc_network(
            directory,
            pattern=pattern,
            recursive=recursive,
            charge=charge,
            multiplicity=multiplicity,
        )
    except Exception as exc:
        typer.echo(f"IRC network construction failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    scan.pot.write_to_disk(output)
    typer.echo(
        f"Wrote network to {output} "
        f"({scan.pot.number_of_nodes} nodes, {scan.pot.graph.number_of_edges()} edges, "
        f"{len(scan.xyz_files)} files used, {len(scan.skipped_xyz_files)} skipped)"
    )


@app.command("make-default-inputs")
@app.command("defaults")
def make_default_inputs(
    output: Path = typer.Option(
        Path("mepd_inputs.toml"), "--output", "-o",
        help="Path to write a starter RunInputs TOML file to.",
    ),
) -> None:
    """Write a starter RunInputs TOML file reflecting the package defaults."""
    run_inputs = RunInputs()
    run_inputs.save(output)
    typer.echo(f"Wrote default inputs to {output}")


if __name__ == "__main__":
    app()
