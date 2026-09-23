"""CLI commands for mepd's structure-discovery/global-optimization features.

Registered as the `mepd discovery` sub-app (see `mepd.cli`). Kept separate
from the core CLI so this area can grow (more sampling/global-search
strategies) without bloating `mepd/cli.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
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
            f"ΔE={payload['rel_energy_kcal']:+.2f} kcal/mol -> {self._output_fp}",
            highlight=False,
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
        help="Target per-atom RMS displacement (bohr), used when "
        "--amplitude-policy=fixed-cartesian. Effective mode displacement is "
        "dr * sqrt(n_atoms), which keeps this size-invariant (a fixed "
        "per-mode displacement in bohr would otherwise give systematically "
        "weaker per-atom kicks for larger molecules).",
    ),
    amplitude_policy: str = typer.Option(
        "fixed-cartesian", "--amplitude-policy",
        help="How far each mode is displaced. 'fixed-cartesian' (default): every "
        "mode gets the same Cartesian distance (--dr), unaware of how stiff or "
        "soft it is. 'energy': --target-energy-kcal is a target harmonic "
        "displacement energy instead -- each mode's amplitude is calibrated "
        "(from its own harmonic force constant) so displacing it costs roughly "
        "that much energy, regardless of stiffness.",
    ),
    target_energy_kcal: float = typer.Option(
        25.0, "--target-energy-kcal",
        help="Target harmonic displacement energy (kcal/mol), used when "
        "--amplitude-policy=energy.",
    ),
    imaginary_mode_amplitude: float = typer.Option(
        0.3, "--imaginary-mode-amplitude",
        help="Fixed displacement (bohr) for an imaginary (negative-frequency) "
        "mode -- the reaction coordinate at a TS seed -- under "
        "--amplitude-policy=energy, where harmonic calibration is undefined.",
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

    amplitude_policy_internal = amplitude_policy.replace("-", "_")
    if amplitude_policy_internal not in ("fixed_cartesian", "energy"):
        raise typer.BadParameter("--amplitude-policy must be 'fixed-cartesian' or 'energy'.")
    if dr <= 0:
        raise typer.BadParameter("--dr must be positive.")
    if target_energy_kcal <= 0:
        raise typer.BadParameter("--target-energy-kcal must be positive.")
    if max_candidates <= 0:
        raise typer.BadParameter("--max-candidates must be a positive integer.")
    if maxiter <= 0:
        raise typer.BadParameter("--maxiter must be a positive integer.")

    run_inputs = RunInputs.open(inputs) if inputs is not None else RunInputs()
    _echo_run_inputs_summary(run_inputs)

    seed_structure = _load_structure_from_smiles_or_xyz(structure, charge, multiplicity)
    seed_node = StructureNode(structure=seed_structure)

    amplitude_label = (
        f"dr={dr:g}" if amplitude_policy_internal == "fixed_cartesian"
        else f"target_energy_kcal={target_energy_kcal:g}"
    )
    typer.echo(
        f"Computing Hessian and sampling normal modes ({amplitude_label}, "
        f"amplitude_policy={amplitude_policy}, max_candidates={max_candidates}, "
        f"maxiter={maxiter})..."
    )
    try:
        with _progress() as progress:
            result = run_hessian_sample(
                seed_node,
                run_inputs.engine,
                dr=dr,
                amplitude_policy=amplitude_policy_internal,
                target_energy_kcal=target_energy_kcal,
                imaginary_mode_amplitude=imaginary_mode_amplitude,
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
        "amplitude_policy": amplitude_policy,
        "target_energy_kcal": target_energy_kcal,
        "imaginary_mode_amplitude": imaginary_mode_amplitude,
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
        help="Target per-atom RMS displacement (bohr). Effective mode "
        "displacement is dr * sqrt(n_atoms), which keeps this size-invariant "
        "(a fixed per-mode displacement in bohr would otherwise give "
        "systematically weaker per-atom kicks for larger molecules).",
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
        help="Moves within this many kcal/mol of the acceptance baseline are treated as "
        "flat (always accepted) rather than run through the Boltzmann test.",
    ),
    acceptance_baseline: str = typer.Option(
        "connected", "--acceptance-baseline",
        help="What each candidate's energy is compared against: 'connected' (default) "
        "compares to the specific minimum it was Hessian-sampled from -- standard "
        "Metropolis basin-hopping semantics, and the fix for a real bug in always "
        "comparing to the original seed ('seed'), which makes acceptance unconditional "
        "and inert once the search has moved past a high-energy seed (e.g. a TS guess). "
        "'running_best' compares to the lowest energy found so far.",
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
    if acceptance_baseline not in ("seed", "connected", "running_best"):
        raise typer.BadParameter(
            "--acceptance-baseline must be 'seed', 'connected', or 'running_best'."
        )
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
                acceptance_baseline=acceptance_baseline,
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
        "acceptance_baseline": acceptance_baseline,
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


class _VRIProgress:
    """Renders `scan_irc_for_vrt` / `find_bifurcation_products` events."""

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._task: Optional[int] = None

    def _replace(self, description: str, total: Optional[int]) -> None:
        if self._task is not None:
            self._progress.remove_task(self._task)
        self._task = self._progress.add_task(description, total=total)

    def __call__(self, event: str, payload: dict) -> None:
        p = self._progress
        if event == "ts_hessian":
            self._replace("TS1 Hessian...", None)
        elif event == "branch_start":
            self._replace(f"Projected Hessians along {payload['branch']} IRC", payload["total"] or None)
        elif event in ("point_done", "bisect_done"):
            if self._task is not None:
                p.update(self._task, completed=payload["index"], total=payload["total"])
        elif event == "bisecting":
            self._replace(f"Bisecting VRT on {payload['branch']} branch", payload["total"] or None)
        elif event == "optimizing_endpoint":
            self._replace(f"Optimizing {payload['branch']} IRC endpoint", None)
        elif event == "pushing":
            self._replace(
                f"Pushing along ridge mode ({payload['total']} optimizations, {payload['branch']})", None,
            )

    def finish(self) -> None:
        if self._task is not None:
            self._progress.remove_task(self._task)
            self._task = None


def _run_vri_irc(engine, ts_node, *, irc_step: float, irc_fmax: float) -> Chain:
    """IRC with tighter settings than mepd's default: projected frequencies
    are only meaningful on an accurate steepest-descent path. The step and
    fmax apply to the Sella backend (g-xTB/ASE engines); other engines use
    their own IRC defaults."""
    from mepd.engines.ase import ASEEngine
    from mepd.engines.gxtb import GXTBCalculator
    from mepd.irc import compute_irc_chain_with_geometric

    irc_fn = getattr(engine, "compute_irc_chain", None)
    if isinstance(engine, (ASEEngine, GXTBCalculator)):
        return irc_fn(ts_node, keywords={"dx": irc_step, "fmax": irc_fmax})
    if callable(irc_fn):
        return irc_fn(ts_node)
    return compute_irc_chain_with_geometric(engine, ts_node)


def _locate_ts2(p1, p2, run_inputs: RunInputs, output: Path, label: str):
    """TS2 between the two products: geodesic interpolation -> path
    minimizer -> TS optimization + IRC, reusing `mepd run`'s machinery."""
    import copy

    import mepd.chainhelpers as ch
    from mepd.cli import _build_path_minimizer, _optimize_ts_and_irc

    seed_chain = Chain.model_validate({
        "nodes": [p1.copy(), p2.copy()],
        "parameters": copy.deepcopy(run_inputs.chain_inputs),
    })
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
    except Exception as exc:
        typer.echo(f"P1 -> P2 path did not fully converge ({label}): {exc}")
    final_chain = minimizer.chain_trajectory[-1] if minimizer.chain_trajectory else initial_chain
    run_inputs.engine.compute_energies(final_chain)
    final_chain.write_to_disk(output / f"{label}_path.xyz")
    return _optimize_ts_and_irc(final_chain.get_ts_node(), run_inputs, output, run_irc=True, label=label)


def _ridge_mode_frames(node, mode, amplitude: float, n_frames: int = 21) -> list:
    from mepd.discovery.vri import push_along_mode

    frames = []
    for phase in np.sin(np.linspace(0.0, 2.0 * np.pi, n_frames)):
        if abs(phase) < 1e-12:
            frames.append(node.copy())
        else:
            plus, minus = push_along_mode(node, mode, amplitude * abs(phase))
            frames.append(plus if phase > 0 else minus)
    return frames


@discovery_app.command("vri")
def vri_search(
    ts: str = typer.Argument(..., help="Optimized TS1 structure (xyz file)."),
    irc: Optional[Path] = typer.Option(
        None, "--irc", exists=True,
        help="Existing IRC through TS1 (xyz + .energies/.gradients sidecars, as `mepd ts --irc` writes). "
        "Computed if omitted.",
    ),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True,
        help="Path to a RunInputs TOML file. Uses built-in defaults if omitted.",
    ),
    charge: Optional[int] = typer.Option(None, "--charge", help="Override the molecular charge."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity", help="Override the spin multiplicity."),
    irc_step: float = typer.Option(
        0.05, "--irc-step",
        help="IRC step (Sella dx, Å·amu^1/2). Smaller than mepd's usual IRC so the VRT is resolved.",
    ),
    irc_fmax: float = typer.Option(0.01, "--irc-fmax", help="IRC stopping force (Sella fmax, eV/Å)."),
    stride: int = typer.Option(1, "--stride", help="Compute a Hessian at every Nth IRC point."),
    n_bisect: int = typer.Option(4, "--n-bisect", help="Bisection steps to refine the VRT between IRC points."),
    vrt_threshold: float = typer.Option(
        20.0, "--vrt-threshold",
        help="A projected frequency must drop below -this (cm^-1) to count as imaginary.",
    ),
    persist: int = typer.Option(
        2, "--persist", help="Consecutive imaginary points required to call a VRT (rejects noise).",
    ),
    imaginary_cutoff: float = typer.Option(
        50.0, "--imaginary-cutoff",
        help="Frequencies below -this (cm^-1) count as imaginary at stationary points.",
    ),
    grad_floor: float = typer.Option(
        1e-4, "--grad-floor",
        help="Below this gradient norm (Eh/bohr) the path tangent comes from the chain, not the gradient.",
    ),
    push_amplitude: float = typer.Option(
        0.3, "--push-amplitude",
        help="Displacement along the ridge mode when searching for P2 (bohr, largest single-atom move).",
    ),
    n_push_points: int = typer.Option(
        3, "--n-push-points", help="Points past the VRT (including the VRT) to push from.",
    ),
    branches: str = typer.Option("both", "--branches", help="IRC branches to scan: both, forward or reverse."),
    skip_ts2: bool = typer.Option(
        False, "--skip-ts2/--find-ts2", help="Skip the NEB + TS optimization search for TS2.",
    ),
    output: Path = typer.Option(Path("mepd_vri_output"), "--output", "-o", help="Directory to write results into."),
) -> None:
    """Search for a valley-ridge inflection (post-TS bifurcation) along the IRC of a TS.

    Scans path-projected frequencies along the IRC for a valley-ridge
    transition, then looks for the second product and the TS2 between the
    products, and reports a verdict."""
    from qcconst.constants import HARTREE_TO_KCAL_PER_MOL

    from mepd.cli import _echo_run_inputs_summary, _geometry_optimizer_keywords, _load_structure_from_smiles_or_xyz
    from mepd.discovery import vri
    from mepd.inputs import ChainInputs
    from mepd.nodes.node import StructureNode

    branch_names = {"both": ("forward", "reverse"), "forward": ("forward",), "reverse": ("reverse",)}.get(branches)
    if branch_names is None:
        raise typer.BadParameter("--branches must be 'both', 'forward' or 'reverse'.")
    for name, value in (("--irc-step", irc_step), ("--irc-fmax", irc_fmax), ("--push-amplitude", push_amplitude)):
        if value <= 0:
            raise typer.BadParameter(f"{name} must be positive.")
    if stride <= 0 or n_push_points <= 0 or persist <= 0 or n_bisect < 0:
        raise typer.BadParameter("--stride, --n-push-points and --persist must be positive; --n-bisect >= 0.")

    run_inputs = RunInputs.open(inputs) if inputs is not None else RunInputs()
    _echo_run_inputs_summary(run_inputs)
    engine = run_inputs.engine
    output.mkdir(parents=True, exist_ok=True)
    write_qcio = bool(getattr(run_inputs, "write_qcio", False))
    hartree_to_kcal = float(HARTREE_TO_KCAL_PER_MOL)

    ts_structure = _load_structure_from_smiles_or_xyz(ts, charge, multiplicity)
    ts_node = StructureNode(structure=ts_structure)

    def _write(nodes, filename: str) -> Optional[str]:
        if not nodes:
            return None
        fp = output / filename
        Chain.model_validate({
            "nodes": [n.copy() for n in nodes], "parameters": run_inputs.chain_inputs,
        }).write_to_disk(fp, write_qcio=write_qcio)
        return str(fp)

    typer.echo("Checking TS1 curvature...")
    try:
        ts_modes = vri.stationary_point_modes(ts_node, engine)
    except Exception as exc:
        typer.echo(f"TS1 Hessian failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)
    ts_n_imag = ts_modes.n_imaginary(imaginary_cutoff)
    if ts_n_imag == 0:
        typer.echo("TS1 has no imaginary frequency; it is not a transition state. Optimize it first (`mepd ts`).")
        raise typer.Exit(code=1)
    if ts_n_imag > 1:
        typer.echo(f"Warning: TS1 has {ts_n_imag} imaginary frequencies; the IRC may not start cleanly.")

    output_files: dict = {}
    if irc is not None:
        irc_chain = Chain.from_xyz(
            irc, ChainInputs(), charge=ts_structure.charge, spinmult=ts_structure.multiplicity,
        )
    else:
        typer.echo(f"Computing IRC (step {irc_step:g}, fmax {irc_fmax:g})...")
        try:
            irc_chain = _run_vri_irc(engine, ts_node, irc_step=irc_step, irc_fmax=irc_fmax)
        except Exception as exc:
            typer.echo(f"IRC failed: {type(exc).__name__}: {exc}")
            raise typer.Exit(code=1)
    irc_nodes = list(irc_chain.nodes)

    typer.echo(f"Scanning projected frequencies along {len(irc_nodes)} IRC points (stride {stride})...")
    try:
        with _progress() as progress:
            reporter = _VRIProgress(progress)
            scan = vri.scan_irc_for_vrt(
                irc_nodes, engine, branches=branch_names, stride=stride, n_bisect=n_bisect,
                vrt_threshold=vrt_threshold, persist=persist, grad_floor=grad_floor, on_event=reporter,
            )
            reporter.finish()
    except Exception as exc:
        typer.echo(f"VRT scan failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)
    if irc is None:
        output_files["irc"] = _write(irc_nodes, "irc.xyz")
    freqs_fp = output / "projected_freqs.json"
    freqs_fp.write_text(json.dumps(scan.to_dict(), indent=2))
    output_files["projected_freqs"] = str(freqs_fp)

    ts1 = irc_nodes[scan.ts_index]
    opt_keywords = _geometry_optimizer_keywords(run_inputs)
    branch_results: dict = {}
    verdicts: List[str] = []
    for name in branch_names:
        branch = scan.branches[name]
        products = None
        if branch.vrt is not None:
            output_files[f"vrt_{name}"] = _write([branch.vrt.node], f"vrt_{name}.xyz")
            output_files[f"ridge_mode_{name}"] = _write(
                _ridge_mode_frames(branch.vrt.node, branch.vrt.ridge_mode_cart, push_amplitude),
                f"ridge_mode_{name}.xyz",
            )
            # The other IRC endpoint (reactant side) never counts as a second product.
            other_end = vri.split_irc_branches(irc_nodes, scan.ts_index)
            reference = [other_end["reverse" if name == "forward" else "forward"][-1]]
            typer.echo(f"VRT on the {name} branch at s = {branch.vrt.s:.3f}; searching for products...")
            try:
                with _progress() as progress:
                    reporter = _VRIProgress(progress)
                    products = vri.find_bifurcation_products(
                        scan, name, engine, opt_keywords=opt_keywords, reference_nodes=reference,
                        push_amplitude=push_amplitude, n_push_points=n_push_points,
                        imaginary_cutoff=imaginary_cutoff, on_event=reporter,
                    )
                    reporter.finish()
            except Exception as exc:
                typer.echo(f"Product search failed on the {name} branch: {type(exc).__name__}: {exc}")

            if products is not None:
                output_files[f"p1_{name}"] = _write([products.p1] if products.p1 else [], f"p1_{name}.xyz")
                output_files[f"p2_{name}"] = _write([products.p2] if products.p2 else [], f"p2_{name}.xyz")
                if products.p2 is not None and products.ts2 is None and not skip_ts2:
                    label = f"ts2_{name}"
                    typer.echo(f"Locating TS2 between P1 and P2 ({name} branch)...")
                    try:
                        found = _locate_ts2(products.p1, products.p2, run_inputs, output, label)
                    except Exception as exc:
                        found = None
                        typer.echo(f"TS2 search failed: {type(exc).__name__}: {exc}")
                    if found is not None:
                        products.ts2 = found.ts_node
                        products.ts2_source = "neb"
                        irc_ts2 = list(found.irc_chain.nodes) if found.irc_chain is not None else None
                        products.ts2_checks = vri.verify_ts2(
                            found.ts_node, irc_ts2, products.p1, products.p2, ts1, engine,
                            imaginary_cutoff=imaginary_cutoff,
                        )
                        products.ts2_verified = products.ts2_checks["verified"]
                        output_files[label] = str(output / f"{label}.xyz")
                elif products.ts2 is not None:
                    output_files[f"ts2_{name}"] = _write([products.ts2], f"ts2_{name}.xyz")

        verdict = vri.branch_verdict(branch, products)
        verdicts.append(verdict)
        branch_results[name] = {
            "verdict": verdict,
            "vrt": branch.vrt.to_dict() if branch.vrt else None,
            "valley_reforms": branch.valley_reforms,
            "transient_dips_s": branch.transient_dips,
            "products": products.to_dict() if products else None,
        }

    ts1_energy = float(ts1._cached_energy)

    def _rel(e):
        return None if e is None else (e - ts1_energy) * hartree_to_kcal

    for res in branch_results.values():
        prod = res["products"]
        if prod:
            for key in ("p1", "p2", "ts2"):
                prod[f"{key}_rel_ts1_kcal_mol"] = _rel(prod[f"{key}_energy"])
        if res["vrt"]:
            res["vrt"]["rel_ts1_kcal_mol"] = _rel(res["vrt"]["energy"])

    overall = vri.overall_verdict(verdicts)
    summary = {
        "verdict": overall,
        "ts": ts,
        "irc": str(irc) if irc is not None else None,
        "inputs": str(inputs) if inputs is not None else None,
        "ts1_energy": ts1_energy,
        "ts1_n_imaginary": ts_n_imag,
        "ts1_lowest_freqs_cm": [float(f) for f in ts_modes.freqs[:4]],
        "n_irc_points": len(irc_nodes),
        "n_hessians": scan.n_hessians + 1,
        "settings": {
            "irc_step": irc_step, "irc_fmax": irc_fmax, "stride": stride, "n_bisect": n_bisect,
            "vrt_threshold": vrt_threshold, "persist": persist, "imaginary_cutoff": imaginary_cutoff,
            "grad_floor": grad_floor, "push_amplitude": push_amplitude, "n_push_points": n_push_points,
            "branches": list(branch_names), "skip_ts2": skip_ts2,
        },
        "branches": branch_results,
        "output_files": {k: v for k, v in output_files.items() if v},
    }
    summary_fp = output / "summary.json"
    summary_fp.write_text(json.dumps(summary, indent=2))

    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column(style="bold cyan")
    table.add_column(style="white")
    table.add_row("Verdict", overall)
    for name, res in branch_results.items():
        vrt_desc = "none"
        if res["vrt"]:
            vrt_desc = f"s = {res['vrt']['s']:.3f}"
            if res["vrt"]["rel_ts1_kcal_mol"] is not None:
                vrt_desc += f", {res['vrt']['rel_ts1_kcal_mol']:+.1f} kcal/mol vs TS1"
        table.add_row(f"{name} branch", f"{res['verdict']} (VRT: {vrt_desc})")
        prod = res["products"]
        if prod and prod.get("ts2_rel_ts1_kcal_mol") is not None:
            table.add_row(
                f"  TS2 ({prod['ts2_source']})",
                f"{prod['ts2_rel_ts1_kcal_mol']:+.1f} kcal/mol vs TS1, verified={prod['ts2_verified']}",
            )
        if prod and prod.get("degenerate") is not None:
            table.add_row("  P1/P2 isomorphic", str(prod["degenerate"]))
    table.add_row("Hessians", str(summary["n_hessians"]))
    table.add_row("Summary", str(summary_fp))
    Console().print(Panel(table, title="[bold green]VRI Search Complete[/bold green]", border_style="green"))
