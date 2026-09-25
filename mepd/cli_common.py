"""Shared infrastructure for `mepd/cli.py`'s commands: loading/echoing
`RunInputs`, endpoint loading (xyz/SMILES), endpoint minimization,
best-of-N atom-mapping selection at the top-level --start/--end pair,
forked-process parallelism (`_fork_map`), running a batch of MSMEP pair
searches to completion, TS+IRC optimization, and collecting TS-guess
tasks from a tree/network. Used by `run`, `ts`, `network-splits` (in
`mepd/cli.py`) and by `channels` (in `mepd/cli_channels.py`) -- split out
so those command modules don't have to import from each other.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import RunInputs
from mepd.nodes.nodehelpers import _connectivity_matches, _n_trans_small_ring_alkenes  # noqa: F401


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


def _load_endpoint(fp: Path, charge: Optional[int], multiplicity: Optional[int]) -> Structure:
    try:
        structure = Structure.open(str(fp))
    except Exception as exc:
        raise typer.BadParameter(f"Could not load '{fp}': {type(exc).__name__}: {exc}")
    updates = {}
    if charge is not None:
        updates["charge"] = charge
    if multiplicity is not None:
        updates["multiplicity"] = multiplicity
    if updates:
        structure = structure.model_copy(update=updates)
    return structure


def _open_run_inputs(inputs: Optional[Path]) -> RunInputs:
    if inputs is None:
        return RunInputs()
    try:
        return RunInputs.open(inputs)
    except Exception as exc:
        raise typer.BadParameter(f"Could not load '{inputs}': {type(exc).__name__}: {exc}")


def _load_structure_from_smiles_or_xyz(
    value: str, charge: Optional[int], multiplicity: Optional[int]
) -> Structure:
    """Load a Structure from an xyz file path, or -- if `value` isn't an
    existing file -- embed it in 3D from a SMILES string.

    Tries qcinf's default RDKit backend first, then falls back to openbabel:
    RDKit refuses multi-fragment SMILES (e.g. "C=C.O.O.O" for a solute plus
    explicit waters), which openbabel embeds fine.
    """
    path = Path(value)
    if path.exists():
        return _load_endpoint(path, charge, multiplicity)

    import qcinf

    # Only forward `multiplicity` to qcinf -- both of its smiles_to_structure
    # backends always compute their own `charge` (from the RDKit mol's formal
    # charge, or from Open Babel's partial charges) and pass it explicitly to
    # Structure(...), so also forwarding a user-supplied `charge` here collides
    # ("got multiple values for keyword argument 'charge'"). Apply a charge
    # override afterward instead, the same way `_load_endpoint` does for xyz
    # files.
    kwargs = {}
    if multiplicity is not None:
        kwargs["multiplicity"] = multiplicity

    errors = []
    structure = None
    for backend in ("rdkit", "openbabel"):
        try:
            structure = qcinf.smiles_to_structure(value, backend=backend, **kwargs)
            break
        except Exception as exc:
            errors.append(f"{backend}: {type(exc).__name__}: {exc}")

    if structure is None:
        hint = ""
        if path.suffix or "/" in value or "\\" in value:
            hint = (
                f" (looks like a file path -- check for a typo; '{value}' does "
                "not exist)"
            )
        raise typer.BadParameter(
            f"'{value}' is neither an existing xyz file nor a valid SMILES string{hint}.\n"
            + "\n".join(errors)
        )

    if charge is not None:
        structure = structure.model_copy(update={"charge": charge})
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


def _dump_atom_mapping_candidates(
    candidates: list,
    metric: str,
    start_structure: Structure,
    run_inputs: "RunInputs",
    debug_dump_dir: Path,
):
    """`--debug-dump` variant of candidate scoring: computes ALL THREE
    metrics (`mepd.atom_mapping_selection.METRICS`) for every candidate --
    not just the selected one -- and writes each candidate's
    geodesic-interpolated path (with energies) plus a `scores.txt` summary
    table under `debug_dump_dir`, so a real run's --debug-dump doubles as
    comparative data across metrics. Returns a `SelectionResult` using
    `metric` for the actual decision, same as `select_best_candidate`."""
    from mepd.atom_mapping_selection import METRICS, SelectionResult, score_candidate_all_metrics

    debug_dump_dir.mkdir(parents=True, exist_ok=True)
    all_scores: dict[str, dict[str, float]] = {}
    chains: dict[str, Chain] = {}
    for candidate in candidates:
        scores, chain = score_candidate_all_metrics(candidate, start_structure, run_inputs)
        all_scores[candidate.label] = scores
        chains[candidate.label] = chain
        chain.write_to_disk(debug_dump_dir / f"{candidate.label}.xyz")

    identity_label = candidates[0].label
    best = min(candidates, key=lambda c: all_scores[c.label][metric])
    veto_margin = run_inputs.atom_mapping_inputs.veto_margin
    if (
        best.label != identity_label
        and all_scores[identity_label][metric] - all_scores[best.label][metric] <= veto_margin
    ):
        best = candidates[0]

    lines = ["candidate\t" + "\t".join(METRICS)]
    for candidate in candidates:
        row = [candidate.label] + [f"{all_scores[candidate.label][m]:.6f}" for m in METRICS]
        marker = "  <- selected" if candidate.label == best.label else ""
        lines.append("\t".join(row) + marker)
    scores_path = debug_dump_dir / "scores.txt"
    scores_path.write_text("\n".join(lines) + "\n")
    typer.echo(f"  --debug-dump: wrote all-candidate, all-metric comparison to {scores_path}")

    return SelectionResult(
        winner=best,
        scores={label: scores[metric] for label, scores in all_scores.items()},
        chains=chains,
    )


def _check_endpoint_atom_mapping(
    start_structure: Structure,
    end_structure: Structure,
    atom_mapping: bool,
    run_inputs: "RunInputs",
    *,
    debug_dump: bool = False,
    output: Optional[Path] = None,
) -> Structure:
    """Sanity-check --start/--end's atom correspondence via SLAPMapper's
    Weisfeiler-Lehman-like/sequential-LAP atom-to-atom mapping (Koda,
    ChemRxiv 2025), warning if it disagrees with the identity mapping
    implied by the two Structures sharing the same atom indexing.

    Returns `end_structure` unchanged, unless `atom_mapping` is set and a
    disagreeing mapping was found -- in which case up to
    `run_inputs.atom_mapping_inputs.n_candidates` of SLAPMapper's candidate
    mappings are each scored (alongside "don't reindex") via
    `mepd.atom_mapping_selection.select_best_candidate`, and the
    best-scoring option's `end_structure` is returned.

    When `debug_dump` is set (and output is given), every candidate is
    additionally scored under all three selection metrics and dumped to
    `<output>/realign_debug/` -- see `_dump_atom_mapping_candidates`.
    """
    if len(start_structure.symbols) != len(end_structure.symbols):
        typer.echo(
            "Note: skipping the --start/--end atom-mapping sanity check "
            "(--start and --end have different atom counts)."
        )
        return end_structure

    from mepd.atom_mapping import HAS_SLAPMAPPER

    if not HAS_SLAPMAPPER:
        typer.echo(
            "Note: skipping the --start/--end atom-mapping sanity check "
            "('slapmapper' not installed; `pip install mepd[aam]` to enable it)."
        )
        return end_structure

    from mepd.atom_mapping import check_atom_mapping

    try:
        atom_map = check_atom_mapping(start_structure, end_structure)
    except Exception as exc:
        typer.echo(
            f"Atom-mapping check between --start and --end failed "
            f"({type(exc).__name__}: {exc}); skipping."
        )
        return end_structure

    if atom_map is None:
        typer.echo(
            "Note: SLAPMapper could not find any --start/--end atom mapping "
            "(e.g. mismatched element composition); skipping the reordering check."
        )
        return end_structure

    if atom_map.is_identity:
        typer.echo(
            "--start/--end atom-mapping check: SLAPMapper's suggested mapping "
            "agrees with the existing atom ordering; no reindexing needed."
        )
        return end_structure

    if not atom_mapping:
        return end_structure

    from mepd.atom_mapping import suggest_atom_mapping_candidates
    from mepd.atom_mapping_selection import build_candidates, select_best_candidate

    n_candidates = run_inputs.atom_mapping_inputs.n_candidates
    try:
        atom_maps = suggest_atom_mapping_candidates(
            start_structure, end_structure, max_candidates=n_candidates
        )
    except Exception as exc:
        typer.echo(
            f"Could not enumerate --atom-mapping candidate mappings "
            f"({type(exc).__name__}: {exc}); keeping --end's original atom ordering."
        )
        return end_structure

    candidates = build_candidates(start_structure, end_structure, atom_maps)
    if len(candidates) == 1:
        # Every candidate SLAPMapper returned was the identity mapping.
        return end_structure

    metric = run_inputs.atom_mapping_inputs.metric
    typer.echo(
        f"--atom-mapping: found {len(candidates) - 1} distinct candidate mapping(s) "
        f"(of up to {n_candidates} considered). Selecting among 'identity' (don't "
        f"reindex) and the candidates by --atom-mapping-metric={metric}..."
    )

    debug_dump_dir = Path(output) / "realign_debug" if debug_dump and output is not None else None
    try:
        if debug_dump_dir is not None:
            result = _dump_atom_mapping_candidates(
                candidates, metric, start_structure, run_inputs, debug_dump_dir
            )
        else:
            result = select_best_candidate(
                candidates, metric, start_structure, run_inputs,
                veto_margin=run_inputs.atom_mapping_inputs.veto_margin,
            )
    except Exception as exc:
        typer.echo(
            f"Atom-mapping candidate selection failed ({type(exc).__name__}: {exc}); "
            "keeping --end's original atom ordering."
        )
        return end_structure

    for candidate in candidates:
        marker = "  <- selected" if candidate.label == result.winner.label else ""
        typer.echo(f"  {candidate.label}: {metric}={result.scores[candidate.label]:.4f}{marker}")

    if result.winner.label == "identity":
        typer.echo("--atom-mapping: keeping --end's original atom ordering.")
    else:
        typer.echo(f"--atom-mapping: reindexing --end's atoms per '{result.winner.label}'.")

    return result.winner.end_structure


def _completed_tree_dirs(completion_dir: Path) -> list[Path]:
    """Pair trees that finished AND have a usable root. A pair whose very
    first NEB failed still writes a tree (adj_matrix.txt, with the root saved
    as node_0_failed.xyz) -- enough to mark it done for resuming, but
    network construction needs node_0.xyz, and one such tree used to sink
    the whole run's classification."""
    if not completion_dir.is_dir():
        return []
    trees = [
        p / "tree" for p in completion_dir.iterdir()
        if (p / "tree" / "adj_matrix.txt").exists()
    ]
    failed = [t for t in trees if not (t / "node_0.xyz").exists()]
    if failed:
        typer.echo(
            f"Skipping {len(failed)} pair tree(s) whose root NEB failed: "
            + ", ".join(t.parent.name for t in sorted(failed))
        )
    return sorted(t for t in trees if (t / "node_0.xyz").exists())


_FORK_JOB: dict = {}


def _fork_call(item):
    return _FORK_JOB["fn"](item)


def _fork_probe_child() -> None:
    """Runs in a throwaway forked child: does the one thing every real worker
    does first -- launch a subprocess -- so a parent with broken `atfork`
    handlers kills us here, in milliseconds, instead of mid-run.

    `cwd` is set deliberately: CPython only takes its `posix_spawn` shortcut
    when `cwd is None`, and it is the `fork()` path that runs the handlers, so
    a probe without a `cwd` would not exercise what the engines exercise."""
    import subprocess

    subprocess.run(["/usr/bin/true"], capture_output=True, cwd="/")


def _fork_probe_ok() -> bool:
    """Whether a forked child survives long enough to start a subprocess."""
    import multiprocessing

    try:
        child = multiprocessing.get_context("fork").Process(target=_fork_probe_child)
        child.start()
    except OSError:
        return False
    child.join(60)
    if child.is_alive():  # wedged on an inherited lock rather than killed
        child.kill()
        child.join()
        return False
    return child.exitcode == 0


def _unsafe_to_fork() -> Optional[str]:
    """Why forking children that will launch subprocesses is unsafe here, or
    None when it is fine.

    On macOS a `pthread_atfork` handler that grabs a lock -- Tcl's notifier is
    the one mepd hit, via `import tkinter`; CoreFoundation and the ObjC runtime
    have the same shape -- hands every forked child a lock owned by a thread
    that does not exist there. The child is then SIGKILLed the instant it forks
    again to run the engine, with no Python traceback and nothing but a
    `BrokenProcessPool` to show for it. `mepd/__init__.py` keeps Tk out of the
    interpreter to prevent the known case; the probe then *verifies* the
    result rather than inferring it from which modules happen to be imported,
    since the handlers are registered in C and are not all enumerable from
    Python. One fork and one `/usr/bin/true`, a few times per run.

    On any platform, a process that has initialised CUDA (a GPU model such as
    a FAIR-Chem MLIP is loaded) cannot fork: the children inherit a CUDA
    context they cannot use."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        return "CUDA is initialised in this process (a GPU model is loaded) and does not survive fork"
    if sys.platform != "darwin":
        return None
    if sys.modules.get("_tkinter") is not None:
        return "tkinter/Tcl is loaded, and its atfork handlers kill forked children on macOS"
    if not _fork_probe_ok():
        return "a probe child was killed as soon as it launched a subprocess (macOS atfork handlers)"
    return None


def _fork_map(fn, items: list, workers: int) -> list:
    """`[fn(x) for x in items]`, across `workers` forked processes when
    workers > 1. `fn` (and everything it closes over -- structures, the
    engine, RunInputs) reaches the children through fork rather than
    pickling; only each item and its return value are pickled.

    Falls back to running `items` serially -- rather than losing the run --
    both when forking is known to be unsafe up front and when a child dies
    without raising (SIGKILL, a segfaulting engine, the OOM killer), which
    `ProcessPoolExecutor` can only report as `BrokenProcessPool` for the whole
    pool. Every caller's `fn` writes its own files or returns its own value,
    so re-running is wasted time, never a wrong answer."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]

    unsafe = _unsafe_to_fork()
    if unsafe:
        typer.echo(
            f"Running {len(items)} item(s) serially instead of across "
            f"{min(workers, len(items))} worker process(es): {unsafe}."
        )
        return [fn(x) for x in items]

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    _FORK_JOB["fn"] = fn
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(items)),
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            return list(pool.map(_fork_call, items))
    except BrokenProcessPool:
        typer.echo(
            "A worker process died without reporting an error (killed by the "
            "OS, or a crash inside the engine); retrying the remaining work "
            "serially. Re-run with --workers 1 to skip the failed attempt."
        )
        return [fn(x) for x in items]
    finally:
        _FORK_JOB.pop("fn", None)


def _run_msmep_pairs(
    structures,
    candidates: list,
    pairs_dir: Path,
    run_inputs: RunInputs,
    *,
    parallel: bool,
    parallel_workers: Optional[int],
    workers: int = 1,
) -> None:
    """Runs recursive NEB/MSMEP autosplitting for each (i, j) structure-index
    pair in `candidates`, writing each result to <pairs_dir>/pair_<i>_<j>/tree/.

    Resumable with no separate manifest/state file: a pair is skipped if its
    directory already holds a completed tree, so re-running against the same
    `pairs_dir` picks up wherever a prior run left off -- the directory tree
    on disk IS the resume state. Shared by --network-completion (candidates
    seeded from newly-discovered intermediates) and `network-splits`
    (candidates seeded from a user-supplied list of minima).

    `workers` > 1 runs that many pairs at once, each in its own forked
    process; pairs share nothing but read-only inputs and write to their
    own pair_<i>_<j>/ directory.
    """
    from mepd.msmep import MSMEP
    import mepd.chainhelpers as ch

    todo = []
    for i, j in candidates:
        if (pairs_dir / f"pair_{i}_{j}" / "tree" / "adj_matrix.txt").exists():
            typer.echo(f"Skipping pair ({i}, {j}): already completed.")
        else:
            todo.append((i, j))

    def _one(pair) -> None:
        i, j = pair
        pair_dir = pairs_dir / f"pair_{i}_{j}"
        tree_dir = pair_dir / "tree"
        # `todo` was filtered against the disk before the first attempt, but
        # `_fork_map` may re-run this list serially after a worker died; pairs
        # that did finish in the meantime are on disk and stay there.
        if (tree_dir / "adj_matrix.txt").exists():
            typer.echo(f"Skipping pair ({i}, {j}): already completed.")
            return
        pair_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Running NEB/MSMEP for pair ({i}, {j})...")
        # MSMEP's "endpoints already attempted elsewhere" dedup (meant to stop
        # redundant re-splitting of the same species discovered mid-recursion
        # within ONE pair's own tree) stores its cache directly on this shared
        # `run_inputs.path_min_inputs` object -- reset it before each pair so
        # it can't leak across pairs and wrongly skip a later, genuinely
        # different pair that merely resembles an earlier one under
        # `is_identical`'s generous default thresholds (this matters most for
        # `mepd channels`, where many pairs deliberately share the same
        # reactant/product molecular graph across different conformers).
        setattr(run_inputs.path_min_inputs, "attempted_pairs_payload", [])
        from mepd import progress as _progress

        # One live-view stream per pair: pairs run concurrently in separate
        # worker processes and must not overwrite each other's live file.
        _progress.set_live_stream(f"pair_{i}_{j}")
        stream_status = "failed"
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
            stream_status = "done"
        except Exception as exc:
            typer.echo(f"Pair ({i}, {j}) failed ({type(exc).__name__}: {exc}); skipping.")
        finally:
            _progress.end_live_stream(stream_status)

    if workers > 1 and len(todo) > 1:
        typer.echo(f"Running {len(todo)} pair(s) across {min(workers, len(todo))} worker process(es).")
    _fork_map(_one, todo, workers)


@dataclass
class TsIrcResult:
    ts_node: object
    irc_chain: Optional[Chain] = None


def _optimize_ts_and_irc(
    ts_guess_node,
    run_inputs: RunInputs,
    output: Path,
    *,
    run_irc: bool,
    label: str = "ts",
) -> Optional["TsIrcResult"]:
    """Optimize a TS-guess node with the engine and, if requested, follow up
    with an IRC -- writes <label>.xyz (and <label>_irc.xyz) into `output`.

    Shared by `ts`, `run --use-tsopt`, and `channels`. Never raises/exits
    itself: returns a `TsIrcResult` (with `irc_chain=None` if `run_irc` was
    False or the IRC itself failed), or None on TS-optimization
    failure/no support, so each caller decides whether that is fatal (a
    standalone `ts` invocation should exit non-zero; a TS opt launched
    automatically after `run` should just warn and let the NEB result
    stand; `channels` needs the IRC chain itself to classify the result).
    """
    from mepd.nodes.node import StructureNode

    from mepd.inputs import ChainInputs
    from mepd.irc import compute_irc_chain_with_geometric

    engine = run_inputs.engine
    compute_ts = getattr(engine, "compute_transition_state", None)
    if not callable(compute_ts):
        typer.echo(f"Engine {type(engine).__name__} does not support transition-state optimization.")
        return None
    typer.echo(f"Optimizing transition state ({label})...")
    try:
        ts_node = compute_ts(node=ts_guess_node)
    except Exception as exc:
        typer.echo(f"Transition-state optimization failed ({label}): {type(exc).__name__}: {exc}")
        return None
    if not isinstance(ts_node, StructureNode):
        typer.echo(
            f"Transition-state optimization did not converge to a usable structure "
            f"({label}; engine returned {type(ts_node).__name__})."
        )
        return None

    output.mkdir(parents=True, exist_ok=True)
    ts_path = output / f"{label}.xyz"
    Chain.model_validate({"nodes": [ts_node], "parameters": ChainInputs()}).write_to_disk(ts_path)
    typer.echo(f"Wrote optimized TS structure to {ts_path}")
    if not run_irc:
        return TsIrcResult(ts_node=ts_node)

    typer.echo(f"Computing IRC ({label})...")
    try:
        irc_fn = getattr(engine, "compute_irc_chain", None)
        irc_chain = irc_fn(ts_node) if callable(irc_fn) else compute_irc_chain_with_geometric(engine, ts_node)
    except Exception as exc:
        typer.echo(f"IRC computation failed ({label}; {type(exc).__name__}: {exc}); TS structure was still written.")
        return TsIrcResult(ts_node=ts_node)

    irc_path = output / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
    irc_chain.write_to_disk(irc_path)
    typer.echo(f"Wrote IRC path to {irc_path}")
    return TsIrcResult(ts_node=ts_node, irc_chain=irc_chain)


def _ts_guess_tasks_from_tree(tree, label_prefix: str) -> list[tuple[str, object]]:
    """Every leaf's TS-guess node from an MSMEP split-tree, labeled
    `<label_prefix>leaf_<leaf.index>` -- matches the naming `mepd run
    --use-tsopt` already uses for a bare tree (`ts_leaf_<index>`), so the two
    commands' outputs land on the same files in a shared --output directory."""
    tasks = []
    for leaf in tree.ordered_leaves:
        if not leaf.data or not leaf.data.chain_trajectory:
            continue
        guess_node = leaf.data.chain_trajectory[-1].get_ts_node()
        tasks.append((f"{label_prefix}leaf_{leaf.index}", guess_node))
    return tasks


def _ts_guess_tasks_from_network(pot) -> list[tuple[str, object]]:
    """Every edge's TS-guess node(s) from a network.json (`Pot`), labeled
    `ts_edge_<source>_<target>` (or `..._<i>` if an edge holds more than one
    NEB chain)."""
    tasks = []
    for u, v, data in pot.graph.edges(data=True):
        nebs = data.get("list_of_nebs") or []
        for i, chain in enumerate(nebs):
            try:
                guess_node = chain.get_ts_node()
            except Exception:
                continue
            suffix = "" if len(nebs) == 1 else f"_{i}"
            tasks.append((f"ts_edge_{u}_{v}{suffix}", guess_node))
    return tasks


def _collect_ts_guess_tasks(
    input_path: Path, charge: Optional[int], multiplicity: Optional[int]
) -> list[tuple[str, object]]:
    """Scan `input_path` for TS-guess nodes to optimize, auto-detecting its
    shape (mirrors `_load_visualization_object`'s detection, minus the bare
    NEB-history-folder and single-chain-xyz cases, which have no leaf/edge
    structure to pull guesses from):

    - a directory with adj_matrix.txt: an MSMEP split-tree (`mepd run
      --recursive`/`--parallel` output) -- one guess per leaf.
    - a directory with conformers/ and/or pairs/: a `mepd channels` output
      -- one guess per leaf of each completed pair's tree.
    - a .json file: a network.json (`Pot`) -- one guess per edge.
    - anything else: a single TS-guess xyz file, labeled "ts" (the original,
      pre-rework behavior).
    """
    from mepd.inputs import ChainInputs
    from mepd.nodes.node import StructureNode
    from mepd.pot import Pot
    from mepd.TreeNode import TreeNode

    tree_charge = charge if charge is not None else 0
    tree_multiplicity = multiplicity if multiplicity is not None else 1

    if input_path.is_dir():
        if (input_path / "adj_matrix.txt").exists():
            tree = TreeNode.read_from_disk(
                input_path, chain_parameters=ChainInputs(),
                charge=tree_charge, multiplicity=tree_multiplicity,
            )
            return _ts_guess_tasks_from_tree(tree, "ts_")

        pairs_dir = input_path / "pairs"
        if pairs_dir.is_dir() or (input_path / "conformers").is_dir():
            tasks: list[tuple[str, object]] = []
            if pairs_dir.is_dir():
                for pair_dir in sorted(pairs_dir.iterdir()):
                    tree_dir = pair_dir / "tree"
                    if not (tree_dir / "adj_matrix.txt").exists():
                        continue
                    try:
                        pair_tree = TreeNode.read_from_disk(
                            tree_dir, chain_parameters=ChainInputs(),
                            charge=tree_charge, multiplicity=tree_multiplicity,
                        )
                    except Exception as exc:
                        typer.echo(
                            f"Skipping {pair_dir.name}: could not load tree "
                            f"({type(exc).__name__}: {exc})"
                        )
                        continue
                    tasks.extend(
                        _ts_guess_tasks_from_tree(pair_tree, f"ts_{pair_dir.name}_")
                    )
            return tasks

        raise typer.BadParameter(
            f"'{input_path}' is a directory but has neither adj_matrix.txt (a "
            "split-tree) nor conformers/ or pairs/ (a `mepd channels` output)."
        )

    if input_path.suffix == ".json":
        pot = Pot.read_from_disk(input_path)
        return _ts_guess_tasks_from_network(pot)

    guess_structure = _load_endpoint(input_path, charge, multiplicity)
    return [("ts", StructureNode(structure=guess_structure))]


