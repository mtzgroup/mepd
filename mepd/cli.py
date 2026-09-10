"""Minimal command-line interface for mepd.

This is intentionally a thin CLI, scoped to what this package currently
supports: running a single nudged-elastic-band (NEB) optimization between
two endpoint structures. Autosplitting/MSMEP, network-completion, and
transition-state-optimization CLI commands will be added as those features
are ported in later phases.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

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
    output: Path = typer.Option(
        Path("mepd_output"), "--output", "-o",
        help="Directory to write the optimized trajectory/energies into.",
    ),
) -> None:
    """Run a single NEB optimization between two endpoint structures."""
    from mepd.nodes.node import StructureNode
    import mepd.chainhelpers as ch

    run_inputs = RunInputs.open(inputs) if inputs is not None else RunInputs()

    start_structure = _load_endpoint(start, charge, multiplicity)
    end_structure = _load_endpoint(end, charge, multiplicity)

    start_node = StructureNode(structure=start_structure)
    end_node = StructureNode(structure=end_structure)

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

    minimizer = _build_path_minimizer(initial_chain, run_inputs)

    try:
        minimizer.optimize_chain()
        typer.echo("NEB converged.")
    except Exception as exc:  # e.g. NoneConvergedException
        typer.echo(f"NEB did not fully converge: {exc}")

    final_chain = minimizer.chain_trajectory[-1] if minimizer.chain_trajectory else initial_chain
    run_inputs.engine.compute_energies(final_chain)
    run_inputs.engine.compute_gradients(final_chain)

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "mep_output.xyz"
    final_chain.write_to_disk(out_path)
    typer.echo(f"Wrote optimized path to {out_path}")


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
