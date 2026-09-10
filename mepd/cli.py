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
from pathlib import Path
from typing import List, Optional

import typer
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import RunInputs

app = typer.Typer(help="mepd: minimum-energy-path discovery tools.")


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

    run_inputs = RunInputs.open(inputs) if inputs is not None else RunInputs()

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

        msmep = MSMEP(inputs=run_inputs)
        if parallel:
            typer.echo("Running parallel recursive autosplitting (MSMEP)...")
            history = msmep.run_parallel_recursive_minimize(
                initial_chain, max_workers=parallel_workers
            )
        else:
            typer.echo("Running recursive autosplitting (MSMEP)...")
            history = msmep.run_recursive_minimize(initial_chain)

        tree_path = output / "tree"
        history.write_to_disk(tree_path)
        typer.echo(f"Wrote split-tree history to {tree_path}")

        try:
            final_chain = history.output_chain
        except ValueError as exc:
            typer.echo(f"Could not assemble a final output chain: {exc}")
            return
        out_path = output / "mep_output.xyz"
        final_chain.write_to_disk(out_path)
        typer.echo(f"Wrote assembled output path to {out_path}")
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

    guess_structure = _load_endpoint(guess, charge, multiplicity)
    guess_node = StructureNode(structure=guess_structure)

    compute_ts = getattr(run_inputs.engine, "compute_transition_state", None)
    if not callable(compute_ts):
        typer.echo(
            f"Engine {type(run_inputs.engine).__name__} does not support "
            "transition-state optimization."
        )
        raise typer.Exit(code=1)

    typer.echo("Optimizing transition state...")
    try:
        result = compute_ts(node=guess_node)
    except Exception as exc:
        typer.echo(f"Transition-state optimization failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    if not isinstance(result, StructureNode):
        typer.echo(
            "Transition-state optimization did not converge to a usable structure "
            f"(engine returned {type(result).__name__})."
        )
        raise typer.Exit(code=1)

    ts_node = result
    output.mkdir(parents=True, exist_ok=True)
    ts_path = output / "ts.xyz"
    ts_path.write_text(ts_node.structure.to_xyz())
    typer.echo(f"Wrote optimized TS structure to {ts_path}")

    if not irc:
        return

    typer.echo("Computing IRC...")
    irc_fn = getattr(run_inputs.engine, "compute_irc_chain", None)
    try:
        if callable(irc_fn):
            irc_chain = irc_fn(ts_node)
        else:
            from mepd.irc import compute_irc_chain_with_geometric
            irc_chain = compute_irc_chain_with_geometric(run_inputs.engine, ts_node)
    except Exception as exc:
        typer.echo(f"IRC computation failed ({type(exc).__name__}: {exc}); TS structure was still written.")
        return

    irc_path = output / "irc.xyz"
    irc_chain.write_to_disk(irc_path)
    typer.echo(f"Wrote IRC path to {irc_path}")


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
