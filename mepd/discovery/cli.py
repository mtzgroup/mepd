"""CLI commands for mepd's structure-discovery/global-optimization features.

Registered as the `mepd discovery` sub-app (see `mepd.cli`). Kept separate
from the core CLI so this area can grow (more sampling/global-search
strategies) without bloating `mepd/cli.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from mepd.chain import Chain
from mepd.inputs import RunInputs

discovery_app = typer.Typer(help="Structure-discovery/global-optimization tools.")


def _progress() -> Progress:
    """A `rich.progress.Progress` shared by both commands below: a spinner +
    description for indeterminate stages (e.g. computing a Hessian), a bar +
    x/y count for determinate ones (e.g. optimizing N candidates), plus
    elapsed time throughout so a long-running call still visibly ticks."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


class _HessianSampleProgress:
    """Renders `run_hessian_sample`'s `on_event` callbacks onto a `Progress`:
    a Hessian-computation spinner, then a per-candidate optimization bar."""

    def __init__(self, progress: Progress, *, description_prefix: str = "") -> None:
        self._progress = progress
        self._prefix = description_prefix
        self._hessian_task: Optional[int] = None
        self._candidate_task: Optional[int] = None

    def __call__(self, event: str, payload: dict) -> None:
        p = self._progress
        if event == "hessian_computing":
            self._hessian_task = p.add_task(f"{self._prefix}Computing Hessian...", total=None)
        elif event == "hessian_computed":
            if self._hessian_task is not None:
                p.remove_task(self._hessian_task)
                self._hessian_task = None
        elif event == "optimizing_candidates":
            total = payload["total"]
            self._candidate_task = p.add_task(
                f"{self._prefix}Optimizing {total} candidate(s)", total=total or None,
            )
        elif event == "candidate_done":
            if self._candidate_task is not None:
                p.update(self._candidate_task, completed=payload["index"], total=payload["total"])
        elif event == "candidates_optimized":
            if self._candidate_task is not None:
                p.remove_task(self._candidate_task)
                self._candidate_task = None


class _HessianGlobalProgress(_HessianSampleProgress):
    """Adds a round-level bar (out of --max-rounds) on top of
    `_HessianSampleProgress`'s Hessian/candidate stages, re-labelled per
    round/source as the basin-hopping search visits each queued minimum."""

    def __init__(self, progress: Progress, *, max_rounds: int) -> None:
        super().__init__(progress)
        self._max_rounds = max_rounds
        self._round_task = progress.add_task(f"Round 0/{max_rounds}", total=max_rounds)
        self._round_index = 0
        self._n_sources = 0

    def __call__(self, event: str, payload: dict) -> None:
        p = self._progress
        if event == "round_start":
            self._round_index = payload["round"]
            self._n_sources = payload["n_sources"]
            p.update(
                self._round_task,
                description=(
                    f"Round {self._round_index + 1}/{self._max_rounds} "
                    f"({self._n_sources} source(s))"
                ),
            )
        elif event == "source_start":
            source_index = payload["source_index"]
            self._prefix = (
                f"  [round {self._round_index + 1}, source {source_index + 1}/{self._n_sources}] "
            )
        elif event == "round_done":
            p.update(
                self._round_task,
                completed=self._round_index + 1,
                description=(
                    f"Round {self._round_index + 1}/{self._max_rounds} done "
                    f"({payload['accepted']} accepted, {payload['candidates_optimized']} optimized)"
                ),
            )
        else:
            super().__call__(event, payload)


def _describe_node(node) -> str:
    """A short human-readable label for a discovered structure: a canonical
    SMILES when the geometry's connectivity can be perceived, else a Hill
    formula (e.g. "C6H6O") -- good enough to recognize a species at a
    glance in the live discovery stream without waiting for the run to
    finish and inspecting the xyz file."""
    try:
        import qcinf

        return qcinf.structure_to_smiles(node.structure)
    except Exception:
        pass
    from collections import Counter

    counts = Counter(node.symbols)
    order = ["C", "H"] + sorted(el for el in counts if el not in ("C", "H"))
    return "".join(f"{el}{counts[el]}" for el in order if counts.get(el))


class _LiveMinimaWriter:
    """Writes newly accepted minima to `output_fp` incrementally, as each one
    is found, instead of only once at the very end -- so `accepted_fp` is a
    valid, loadable chain (e.g. for `mepd run`) at any point while a
    long-running `hessian-global` search is still going, not just after it
    completes. Also prints a one-line description of each new find."""

    def __init__(self, *, console, output_fp: Path, chain_inputs, write_qcio: bool) -> None:
        self._console = console
        self._output_fp = output_fp
        self._chain_inputs = chain_inputs
        self._write_qcio = write_qcio
        self.nodes: List = []

    def __call__(self, event: str, payload: dict) -> None:
        if event != "minimum_accepted":
            return
        node = payload["node"]
        self.nodes.append(node.copy())
        label = _describe_node(node)
        self._console.print(
            f"[bold green]✓ New minimum #{len(self.nodes)}[/bold green] "
            f"(round {payload['round'] + 1}): {label}  "
            f"ΔE={payload['rel_energy_kcal']:+.2f} kcal/mol -> {self._output_fp}"
        )
        chain_out = Chain.model_validate({
            "nodes": [n.copy() for n in self.nodes],
            "parameters": self._chain_inputs,
        })
        chain_out.write_to_disk(self._output_fp, write_qcio=self._write_qcio)


@discovery_app.command("hessian-sample")
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

    from mepd.cli import _echo_run_inputs_summary, _load_structure_from_smiles_or_xyz
    from mepd.discovery.hessian_sample import run_hessian_sample
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
        with _progress() as progress:
            result = run_hessian_sample(
                seed_node,
                run_inputs.engine,
                dr=dr,
                max_candidates=max_candidates,
                maxiter=maxiter,
                chain_inputs=run_inputs.chain_inputs,
                on_event=_HessianSampleProgress(progress),
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


def _parse_dr_scan_values(raw_values: str) -> List[float]:
    values: List[float] = []
    for raw_item in str(raw_values).split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise typer.BadParameter(
                f"--dr-scan-values must be a comma-separated list of positive numbers; got {item!r}."
            ) from exc
        if value <= 0:
            raise typer.BadParameter("--dr-scan-values entries must be positive.")
        values.append(value)
    if not values:
        raise typer.BadParameter("--dr-scan-values must contain at least one positive value.")
    return values


@discovery_app.command("hessian-global")
def hessian_global(
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
        help="Hard cap on the number of displaced candidates generated/optimized per source, per round.",
    ),
    maxiter: int = typer.Option(
        500, "--maxiter",
        help="Maximum geometry-optimization steps for each displaced candidate.",
    ),
    temperature: float = typer.Option(
        298.15, "--temperature",
        help="Temperature (Kelvin) for the Metropolis/Boltzmann acceptance criterion. "
        "Higher accepts more uphill moves.",
    ),
    energy_tolerance_kcal: float = typer.Option(
        1.0e-4, "--energy-tolerance-kcal",
        help="Moves within this many kcal/mol of the seed's energy are treated as flat "
        "(always accepted) rather than run through the Boltzmann test.",
    ),
    max_rounds: int = typer.Option(
        100, "--max-rounds",
        help="Hard cap on the number of basin-hopping rounds. Each round Hessian-samples "
        "from every currently-queued accepted minimum -- this is the control that bounds "
        "an otherwise-unbounded global search.",
    ),
    random_seed: Optional[int] = typer.Option(
        None, "--random-seed",
        help="Seed for the acceptance-test random draws, for reproducible runs. "
        "Omit for non-deterministic acceptance.",
    ),
    full_dr_scan: bool = typer.Option(
        False, "--full-dr-scan/--no-full-dr-scan",
        help="Use the expensive Hessian-global scan: every round, displace by all "
        "--dr-scan-values (instead of just --dr) per source minimum.",
    ),
    dr_scan_values: str = typer.Option(
        "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0", "--dr-scan-values",
        help="Comma-separated displacement factors for --full-dr-scan. Each value is "
        "capped independently at --max-candidates.",
    ),
    output: Path = typer.Option(
        Path("mepd_hessian_global_output"), "--output", "-o",
        help="Directory to write results into.",
    ),
) -> None:
    """Basin-hopping-style global optimization: repeatedly Hessian-sample
    from every accepted minimum found so far, accepting new minima via a
    Metropolis/Boltzmann criterion, until the search exhausts itself or
    --max-rounds is reached."""
    from mepd.cli import _echo_run_inputs_summary, _load_structure_from_smiles_or_xyz
    from mepd.discovery.hessian_sample import run_hessian_global_optimization
    from mepd.nodes.node import StructureNode

    if dr <= 0:
        raise typer.BadParameter("--dr must be positive.")
    if max_candidates <= 0:
        raise typer.BadParameter("--max-candidates must be a positive integer.")
    if maxiter <= 0:
        raise typer.BadParameter("--maxiter must be a positive integer.")
    if temperature <= 0:
        raise typer.BadParameter("--temperature must be positive.")
    if max_rounds <= 0:
        raise typer.BadParameter("--max-rounds must be a positive integer.")
    dr_scan_values_list = _parse_dr_scan_values(dr_scan_values) if full_dr_scan else None

    run_inputs = RunInputs.open(inputs) if inputs is not None else RunInputs()
    _echo_run_inputs_summary(run_inputs)

    seed_structure = _load_structure_from_smiles_or_xyz(structure, charge, multiplicity)
    seed_node = StructureNode(structure=seed_structure)

    output.mkdir(parents=True, exist_ok=True)
    write_qcio = bool(getattr(run_inputs, "write_qcio", False))
    accepted_fp = output / "accepted_minima.xyz"

    dr_label = (
        f"dr_scan_values={','.join(f'{v:g}' for v in dr_scan_values_list)}"
        if full_dr_scan
        else f"dr={dr:g}"
    )
    typer.echo(
        f"Running basin-hopping global optimization ({dr_label}, max_candidates={max_candidates}, "
        f"temperature={temperature:g}K, max_rounds={max_rounds})..."
    )
    try:
        with _progress() as progress:
            renderer = _HessianGlobalProgress(progress, max_rounds=max_rounds)
            live_writer = _LiveMinimaWriter(
                console=progress.console, output_fp=accepted_fp,
                chain_inputs=run_inputs.chain_inputs, write_qcio=write_qcio,
            )

            def _on_event(event: str, payload: dict) -> None:
                renderer(event, payload)
                live_writer(event, payload)

            result = run_hessian_global_optimization(
                seed_node,
                run_inputs.engine,
                dr=dr,
                dr_values=dr_scan_values_list,
                max_candidates=max_candidates,
                maxiter=maxiter,
                temperature=temperature,
                energy_tolerance_kcal=energy_tolerance_kcal,
                max_rounds=max_rounds,
                random_seed=random_seed,
                chain_inputs=run_inputs.chain_inputs,
                on_event=_on_event,
            )
    except Exception as exc:
        typer.echo(f"Global optimization failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    if not result.accepted_minima:
        accepted_fp = None
    else:
        # Already written incrementally as each minimum was found (see
        # _LiveMinimaWriter); rewritten once more here as cheap insurance
        # that the final file reflects exactly `result.accepted_minima`.
        chain_out = Chain.model_validate({
            "nodes": [n.copy() for n in result.accepted_minima],
            "parameters": run_inputs.chain_inputs,
        })
        chain_out.write_to_disk(accepted_fp, write_qcio=write_qcio)

    from qcconst.constants import HARTREE_TO_KCAL_PER_MOL

    hartree_to_kcal = float(HARTREE_TO_KCAL_PER_MOL)
    summary_payload = {
        "structure": structure,
        "inputs": str(inputs) if inputs is not None else None,
        "dr": dr,
        "full_dr_scan": full_dr_scan,
        "dr_scan_values": dr_scan_values_list if full_dr_scan else [],
        "max_candidates": max_candidates,
        "maxiter": maxiter,
        "temperature": temperature,
        "energy_tolerance_kcal": energy_tolerance_kcal,
        "max_rounds": max_rounds,
        "random_seed": random_seed,
        "start_energy": result.start_energy,
        "rounds_run": result.rounds_run,
        "stopped_reason": result.stopped_reason,
        "accepted_minima": len(result.accepted_minima),
        "accepted_minima_energies_eh": [float(n.energy) for n in result.accepted_minima],
        "accepted_minima_rel_energies_kcal_mol": [
            (float(n.energy) - result.start_energy) * hartree_to_kcal
            for n in result.accepted_minima
        ],
        "round_summaries": result.round_summaries,
        "output_files": {
            "accepted_minima": str(accepted_fp) if accepted_fp else None,
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
    result_table.add_row("Rounds run", f"{result.rounds_run} ({result.stopped_reason})")
    result_table.add_row("Accepted minima", str(len(result.accepted_minima)))
    if accepted_fp:
        result_table.add_row("Accepted minima xyz", str(accepted_fp))
    result_table.add_row("Summary", str(summary_fp))
    Console().print(
        Panel(
            result_table,
            title="[bold green]Hessian Global Optimization Complete[/bold green]",
            border_style="green",
        )
    )

    if not result.accepted_minima:
        typer.echo("No new minima were accepted.")
