"""CLI commands for `mepd agentic`: LLM-guided evolutionary tuning of NEB
`input.toml` parameters (prototype). Registered as the `mepd agentic`
sub-app (see `mepd.cli`), mirroring `mepd/discovery/cli.py`'s pattern so
this stays self-contained and optional (the `agentic` extra) without
bloating `mepd/cli.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mepd.agentic import benchmark, loop, schema
from mepd.agentic.llm_client import get_backend
from mepd.inputs import RunInputs

agentic_app = typer.Typer(help="LLM-guided evolutionary parameter tuning (prototype).")


def _on_event(event: str, payload: dict) -> None:
    if event == "seed_evaluating":
        typer.echo("Evaluating seed parameters (iteration 0)...")
    elif event == "resumed":
        typer.echo(f"Resuming from {payload['completed_iterations']} completed iteration(s).")
    elif event == "iteration_start":
        typer.echo(f"Iteration {payload['iteration']}: asking LLM for a proposal...")
    elif event == "iteration_done":
        record = payload["record"]
        fitness = record.get("fitness")
        typer.echo(
            f"  iteration {record['iteration']} [{payload['state']}] "
            f"fitness={fitness if fitness is not None else '--'}"
        )
        for warning in payload.get("warnings") or []:
            typer.echo(f"    warning: {warning}")
    elif event == "stopped_patience":
        typer.echo(f"Stopping: no improvement for {payload['since_improvement']} iteration(s).")
    elif event == "stopped_max_iterations":
        typer.echo(f"Stopping: reached --iterations={payload['max_iterations']}.")


@agentic_app.command("tune")
def tune(
    seed: Path = typer.Option(
        ..., "--seed", exists=True,
        help="Seed RunInputs TOML file (e.g. examples/gxtb.toml).",
    ),
    output: Path = typer.Option(
        ..., "--output", "-o",
        help="Directory to write the tuning trajectory/state/results into.",
    ),
    iterations: int = typer.Option(
        10, "--iterations", "-n",
        help="Number of LLM-proposed candidate parameter sets to try "
        "(plus the seed itself as iteration 0).",
    ),
    patience: Optional[int] = typer.Option(
        None, "--patience",
        help="Stop early after this many consecutive non-improving "
        "iterations. Unlimited if omitted.",
    ),
    backend_name: str = typer.Option(
        "ollama", "--backend", help="LLM backend: 'ollama' or 'null' (no-op, for testing).",
    ),
    model: str = typer.Option(
        "qwen3.5:latest", "--model", help="Model name for the chosen backend.",
    ),
    ollama_base_url: Optional[str] = typer.Option(
        None, "--ollama-base-url",
        help="Ollama server base URL. Defaults to $OLLAMA_API_BASE or http://127.0.0.1:11434.",
    ),
    temperature: float = typer.Option(0.4, "--temperature", help="LLM sampling temperature."),
    llm_timeout: float = typer.Option(
        120.0, "--llm-timeout",
        help="Timeout (seconds) for each LLM request. Heavier "
        "reasoning-style local models can need well over 120s for a "
        "schema-constrained response -- raise this if OllamaBackend "
        "requests are timing out.",
    ),
    timeout_per_reaction: float = typer.Option(
        1200.0, "--timeout-per-reaction",
        help="Per-reaction wall-clock timeout (seconds) before a candidate "
        "is marked failed/timed-out.",
    ),
    parallel_reactions: bool = typer.Option(
        True, "--parallel-reactions/--sequential-reactions",
        help="Run the benchmark's reactions concurrently (one subprocess each).",
    ),
    parallel_msmep: bool = typer.Option(
        False, "--parallel-msmep/--sequential-msmep",
        help="Also evaluate each reaction's own MSMEP split branches in "
        "parallel (increases total concurrency).",
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume",
        help="Resume from a previous run's state.json in --output if present.",
    ),
) -> None:
    """Run the LLM-guided tuning loop: propose new NEB parameter values,
    score them against the fixed 3-reaction benchmark, keep whatever
    improves on the current best."""
    problems = benchmark.validate_benchmark_data()
    if problems:
        for problem in problems:
            typer.echo(f"Benchmark data problem: {problem}", err=True)
        raise typer.Exit(code=1)

    seed_run_inputs = RunInputs.open(seed)
    try:
        schema.get_schema(seed_run_inputs.path_min_method)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    backend_kwargs = (
        {
            "model": model, "base_url": ollama_base_url, "temperature": temperature,
            "timeout_s": llm_timeout,
        }
        if backend_name == "ollama" else {}
    )
    try:
        llm_backend = get_backend(backend_name, **backend_kwargs)
    except (ValueError, ImportError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    state = loop.run_tuning_loop(
        seed_run_inputs=seed_run_inputs,
        backend=llm_backend,
        output_dir=output,
        max_iterations=iterations,
        patience=patience,
        timeout_per_reaction=timeout_per_reaction,
        parallel_reactions=parallel_reactions,
        parallel_msmep=parallel_msmep,
        resume=resume,
        on_event=_on_event,
    )

    best_toml = output / "best_inputs.toml"
    RunInputs(**state.best_params).save(best_toml)
    typer.echo(
        f"Best fitness={state.best_fitness} at iteration {state.best_iteration}. "
        f"Wrote best parameters to {best_toml}"
    )


@agentic_app.command("show-schema")
def show_schema(
    path_min_method: str = typer.Option(
        "NEB", "--path-min-method",
        help="Path-minimization method whose tunable-knob schema to show.",
    ),
) -> None:
    """Print the curated, safety-bounded set of parameters `mepd agentic
    tune` is allowed to propose new values for."""
    try:
        param_schema = schema.get_schema(path_min_method)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    typer.echo(schema.render_schema_for_prompt(param_schema))


@agentic_app.command("check-benchmark")
def check_benchmark() -> None:
    """Verify the fixed 3-reaction benchmark's endpoint xyz files resolve
    correctly on this checkout."""
    problems = benchmark.validate_benchmark_data()
    if problems:
        for problem in problems:
            typer.echo(problem)
        raise typer.Exit(code=1)
    for reaction in benchmark.BENCHMARK_REACTIONS:
        typer.echo(f"{reaction.name}: {reaction.start} -> {reaction.end}")
    typer.echo("All benchmark reaction endpoints found.")
