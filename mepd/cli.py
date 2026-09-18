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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typer
from qcdata import Structure

from mepd.atom_mapping_selection import METRICS as _ATOM_MAPPING_METRICS
from mepd.chain import Chain
from mepd.inputs import NetworkInputs, RunInputs


def _teardown_live_display(*_args, **_kwargs) -> None:
    """Tear down the progress printer's live display after every command.

    `ProgressPrinter` starts a rich `Live` lazily (`_render_live_monitors`) and
    only stops it when something else needs the terminal. Without this, a
    command returns with the Live still running: rich leaves the cursor hidden,
    and the stdout redirection it installed on start is only unwound later --
    against whatever stream happens to be current by then.
    """
    from mepd.progress import stop_status

    stop_status()


app = typer.Typer(
    help="mepd: minimum-energy-path discovery tools.",
    result_callback=_teardown_live_display,
)


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
    if method == "GSM":
        from mepd.pathminimizers.gsm import GSM

        return GSM(
            initial_chain=initial_chain,
            engine=run_inputs.engine,
            parameters=run_inputs.path_min_inputs,
            gi_inputs=run_inputs.gi_inputs,
        )
    raise typer.BadParameter(
        f"Unsupported path_min_method '{run_inputs.path_min_method}'. "
        "This build supports: NEB, FNEB, NEB-DLF, GEOMETRIC-NEB, GSM."
    )


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
    if not completion_dir.is_dir():
        return []
    return sorted(
        p / "tree" for p in completion_dir.iterdir()
        if (p / "tree" / "adj_matrix.txt").exists()
    )


def _run_msmep_pairs(
    structures,
    candidates: list,
    pairs_dir: Path,
    run_inputs: RunInputs,
    *,
    parallel: bool,
    parallel_workers: Optional[int],
) -> None:
    """Runs recursive NEB/MSMEP autosplitting for each (i, j) structure-index
    pair in `candidates`, writing each result to <pairs_dir>/pair_<i>_<j>/tree/.

    Resumable with no separate manifest/state file: a pair is skipped if its
    directory already holds a completed tree, so re-running against the same
    `pairs_dir` picks up wherever a prior run left off -- the directory tree
    on disk IS the resume state. Shared by --network-completion (candidates
    seeded from newly-discovered intermediates) and `network-splits`
    (candidates seeded from a user-supplied list of minima).
    """
    from mepd.msmep import MSMEP
    import mepd.chainhelpers as ch

    for i, j in candidates:
        pair_dir = pairs_dir / f"pair_{i}_{j}"
        tree_dir = pair_dir / "tree"
        if (tree_dir / "adj_matrix.txt").exists():
            typer.echo(f"Skipping pair ({i}, {j}): already completed.")
            continue
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
            typer.echo(f"Pair ({i}, {j}) failed ({type(exc).__name__}: {exc}); skipping.")
            continue


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

    Resumable: see `_run_msmep_pairs`. Re-running the exact same command
    against the same --output picks up wherever a prior run left off.
    """
    from mepd.NetworkBuilder import NetworkBuilder

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

    _run_msmep_pairs(
        structures, candidates, completion_dir, run_inputs,
        parallel=parallel, parallel_workers=parallel_workers,
    )

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
    from mepd.inputs import ChainInputs

    Chain.model_validate(
        {"nodes": [ts_node], "parameters": ChainInputs()}
    ).write_to_disk(ts_path)
    typer.echo(f"Wrote optimized TS structure to {ts_path}")

    if not run_irc:
        return TsIrcResult(ts_node=ts_node)

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
        return TsIrcResult(ts_node=ts_node)

    irc_path = output / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
    irc_chain.write_to_disk(irc_path)
    typer.echo(f"Wrote IRC path to {irc_path}")
    return TsIrcResult(ts_node=ts_node, irc_chain=irc_chain)


@app.command("run")
def run(
    start: str = typer.Option(..., "--start", help="Path to the start-structure xyz file, or a SMILES string."),
    end: str = typer.Option(..., "--end", help="Path to the end-structure xyz file, or a SMILES string."),
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
    minimize_ends: Optional[bool] = typer.Option(
        None, "--minimize-ends/--no-minimize-ends",
        help="Optimize the start/end endpoint geometries before running NEB. "
        "Defaults to on when either --start/--end is given as SMILES (an "
        "RDKit-embedded guess, not a real minimum on the target engine's "
        "surface) and off when both are already-provided xyz files "
        "(presumed already minimized). Pass explicitly to override either way.",
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
    same_pair_split_limit: int = typer.Option(
        5, "--same-pair-split-limit",
        help="If a branch repeats the exact same (start, end) endpoint pair this "
        "many times in a row with no new chemistry found, stop splitting that "
        "branch further. A split that DOES discover a new molecule/conformer is "
        "never cut off by this. Raise it for floppy systems that legitimately "
        "need more attempts before finding a real intermediate.",
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
    atom_mapping: bool = typer.Option(
        False, "--atom-mapping",
        help="Every run checks whether SLAPMapper's Weisfeiler-Lehman-like/"
        "sequential-LAP atom-to-atom mapping (Koda, ChemRxiv 2025) between "
        "--start and --end agrees with their shared input atom ordering, "
        "warning if not. Passing this flag additionally reindexes --end's "
        "atoms to match the suggested mapping instead of just warning. "
        "No-op when --start/--end are both SMILES (already mapped "
        "consistently before 3D embedding) or atom counts differ.",
    ),
    debug_dump: bool = typer.Option(
        False, "--debug-dump",
        help="When --atom-mapping triggers its candidate-mapping selection, "
        "also write every candidate's interpolated path (xyz + energies) and a "
        "scores.txt comparing all three selection metrics per candidate to "
        "<output>/realign_debug/, e.g. for inspecting them with `mepd visualize`.",
    ),
    atom_mapping_candidates: int = typer.Option(
        200, "--atom-mapping-candidates",
        help="--atom-mapping: how many of SLAPMapper's equal-minimal-cost "
        "candidate mappings (plus their symmetry-orbit expansions) to keep "
        "and consider (they're ties, not ranked by quality among themselves).",
    ),
    atom_mapping_metric: str = typer.Option(
        "geodesic-distance", "--atom-mapping-metric",
        help="--atom-mapping: how each candidate mapping (including 'don't "
        "reindex') is scored from its geodesic-interpolated path -- "
        "'geodesic-distance' (the geodesic optimizer's own path length; "
        "free), 'path-rmsd' (cumulative per-frame RMSD along the path; "
        "free), or 'gi-energy' (highest QM energy along the path; most "
        "expensive, one engine energy evaluation per candidate). Which is "
        "actually the best predictor of a correct mapping isn't settled -- "
        "--debug-dump records all three per candidate to help compare them.",
    ),
    atom_mapping_veto_margin: float = typer.Option(
        0.0, "--atom-mapping-veto-margin",
        help="--atom-mapping: a non-identity candidate must beat 'don't "
        "reindex' by more than this (in --atom-mapping-metric's own units) to "
        "be adopted; otherwise the original --end ordering is kept even if "
        "some candidate scored marginally better. Default 0.0 is a pure "
        "best-of-N with no calibrated stability margin yet.",
    ),
    atom_mapping_recheck_splits: bool = typer.Option(
        False, "--atom-mapping-recheck-splits",
        help="Experimental: also re-run the same best-of-N atom-mapping "
        "selection (--atom-mapping-metric etc.) at every new (reactant, "
        "product) pair MSMEP's recursive splitting discovers, not just the "
        "original --start/--end pair. Independent of --atom-mapping. Atom "
        "order is otherwise preserved throughout the recursion, so this only "
        "changes anything for a split whose own reactant/product pair "
        "happens to have SLAPMapper-detectable symmetry.",
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
    if same_pair_split_limit <= 0:
        raise typer.BadParameter("--same-pair-split-limit must be a positive integer.")
    if atom_mapping_metric not in _ATOM_MAPPING_METRICS:
        raise typer.BadParameter(
            f"--atom-mapping-metric must be one of {_ATOM_MAPPING_METRICS}."
        )
    if atom_mapping_candidates <= 0:
        raise typer.BadParameter("--atom-mapping-candidates must be a positive integer.")
    if network_completion and not recursive and not parallel:
        typer.echo("--network-completion requires recursive splitting; enabling --recursive.")
        recursive = True

    run_inputs = _open_run_inputs(inputs)
    run_inputs.path_min_inputs.validate_minima_with_hessian = validate_minima_with_hessian
    run_inputs.path_min_inputs.hessian_minimum_frequency_cutoff = hessian_minimum_frequency_cutoff
    run_inputs.path_min_inputs.hessian_minima_rescue_displacement = hessian_minima_rescue_displacement
    run_inputs.path_min_inputs.recursive_same_pair_split_limit = same_pair_split_limit
    run_inputs.atom_mapping_inputs.n_candidates = atom_mapping_candidates
    run_inputs.atom_mapping_inputs.metric = atom_mapping_metric
    run_inputs.atom_mapping_inputs.veto_margin = atom_mapping_veto_margin
    run_inputs.atom_mapping_inputs.recheck_on_split = atom_mapping_recheck_splits
    _echo_run_inputs_summary(run_inputs)

    start_is_smiles = not Path(start).exists()
    end_is_smiles = not Path(end).exists()
    both_smiles_pair = start_is_smiles and end_is_smiles

    if both_smiles_pair:
        typer.echo(
            "--start/--end are both SMILES strings; computing a SLAPMapper "
            "atom-to-atom mapping to build a consistently-indexed structure pair..."
        )
        from mepd.atom_mapping import map_smiles_pair

        try:
            start_structure, end_structure = map_smiles_pair(
                start, end,
                charge_start=charge, charge_end=charge,
                multiplicity_start=multiplicity or 1, multiplicity_end=multiplicity or 1,
            )
        except ValueError as exc:
            raise typer.BadParameter(
                f"--start ({start!r}) and --end ({end!r}) don't exist as files, and "
                f"couldn't be parsed as a SMILES pair either: {exc}"
            )
    else:
        start_structure = _load_structure_from_smiles_or_xyz(start, charge, multiplicity)
        end_structure = _load_structure_from_smiles_or_xyz(end, charge, multiplicity)

    start_node = StructureNode(structure=start_structure)
    end_node = StructureNode(structure=end_structure)

    effective_minimize_ends = minimize_ends
    if effective_minimize_ends is None:
        effective_minimize_ends = start_is_smiles or end_is_smiles
        if effective_minimize_ends:
            typer.echo(
                "An endpoint was given as SMILES (an RDKit-embedded guess, not a "
                "real minimum); minimizing endpoints by default. Pass "
                "--no-minimize-ends to skip."
            )
    if effective_minimize_ends:
        start_node, end_node = _minimize_endpoints(start_node, end_node, run_inputs)

    # The --atom-mapping sanity check compares geodesic-interpolation path
    # energies between the unpermuted and permuted --end -- run it AFTER
    # minimization (rather than on the raw SMILES embedding) so that
    # comparison reflects real minima, not embedding artifacts/strain that
    # can otherwise dominate the energy delta the veto decision is based on.
    if not both_smiles_pair:
        realigned_end_structure = _check_endpoint_atom_mapping(
            start_node.structure, end_node.structure, atom_mapping, run_inputs,
            debug_dump=debug_dump, output=output,
        )
        if realigned_end_structure is not end_node.structure:
            end_node = StructureNode(structure=realigned_end_structure)

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


@app.command("ts")
def ts(
    guess: Path = typer.Option(
        ..., "--guess", exists=True,
        help="Path to a TS-guess input, auto-detected by content: a single "
        "TS-guess xyz file (optimizes just that one guess), an MSMEP "
        "split-tree directory (has adj_matrix.txt, as written by `mepd run "
        "--recursive`/`--parallel` to <output>/tree), a `mepd channels` "
        "output directory (has conformers/ and/or pairs/), or a network.json "
        "-- for the latter three, every TS guess found is optimized.",
    ),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True,
        help="Path to a RunInputs TOML file. Uses built-in defaults if omitted.",
    ),
    charge: Optional[int] = typer.Option(
        None, "--charge", help="Override the molecular charge on the guess structure(s)."
    ),
    multiplicity: Optional[int] = typer.Option(
        None, "--multiplicity", help="Override the spin multiplicity on the guess structure(s)."
    ),
    irc: bool = typer.Option(
        False, "--irc", help="Follow up each optimized transition state with an IRC."
    ),
    output: Path = typer.Option(
        Path("mepd_ts_output"), "--output", "-o",
        help="Directory to write optimized TS structure(s) (and IRC path(s)) into. "
        "Resumable: a guess whose <label>.xyz already exists in --output is "
        "skipped, so re-running against the same --output picks up wherever "
        "a prior run left off.",
    ),
) -> None:
    """Optimize transition-state guess(es). `--guess` accepts a single
    TS-guess xyz file, or a whole result to scan for every not-yet-optimized
    TS guess it contains: an MSMEP split-tree directory, a `mepd channels`
    output directory, or a network.json."""
    run_inputs = _open_run_inputs(inputs)
    _echo_run_inputs_summary(run_inputs)

    try:
        tasks = _collect_ts_guess_tasks(guess, charge, multiplicity)
    except typer.BadParameter:
        raise
    except Exception as exc:
        typer.echo(f"Could not load '{guess}': {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    if not tasks:
        typer.echo(f"No TS guesses found in '{guess}'.")
        raise typer.Exit(code=1)

    output.mkdir(parents=True, exist_ok=True)

    n_done = n_skipped = n_failed = 0
    for label, guess_node in tasks:
        ts_path = output / f"{label}.xyz"
        if ts_path.exists():
            typer.echo(f"Skipping {label}: already optimized ({ts_path}).")
            n_skipped += 1
            continue
        result = _optimize_ts_and_irc(guess_node, run_inputs, output, run_irc=irc, label=label)
        if result is None:
            n_failed += 1
        else:
            n_done += 1

    typer.echo(
        f"Optimized {n_done} TS guess(es), skipped {n_skipped} already-optimized, "
        f"{n_failed} failed."
    )
    if n_done == 0 and n_failed > 0:
        raise typer.Exit(code=1)


def _load_channels_result(result_path: Path, charge: int, multiplicity: int):
    """Load a `mepd channels` output directory (has a conformers/ pool
    directory and/or a pairs/ directory of per-pair MSMEP trees) into a
    `ChannelsResult` for `mepd visualize`."""
    from mepd.inputs import ChainInputs
    from mepd.pot import Pot
    from mepd.TreeNode import TreeNode
    from mepd.viz import ChannelsResult

    def _load_conformer_pool(fp: Path) -> list:
        if not fp.exists():
            return []
        return list(
            Chain.from_xyz(fp, ChainInputs(), charge=charge, spinmult=multiplicity).nodes
        )

    reactant_conformers = _load_conformer_pool(result_path / "conformers" / "start.xyz")
    product_conformers = _load_conformer_pool(result_path / "conformers" / "end.xyz")

    pairs = []
    load_errors = []
    pairs_dir = result_path / "pairs"
    if pairs_dir.is_dir():
        for pair_dir in sorted(pairs_dir.iterdir()):
            tree_dir = pair_dir / "tree"
            if not (tree_dir / "adj_matrix.txt").exists():
                continue
            try:
                tree = TreeNode.read_from_disk(
                    tree_dir, chain_parameters=ChainInputs(), charge=charge, multiplicity=multiplicity
                )
                chain = tree.output_chain
            except Exception as exc:
                # adj_matrix.txt existing doesn't guarantee the tree has any
                # usable node data -- e.g. every node in it failed to compute
                # (missing/misconfigured engine) and only an empty tree got
                # written. Report it instead of just silently omitting the
                # pair, so "no MEP outputs shown" doesn't look like this
                # feature is broken when the real cause is upstream.
                load_errors.append(f"{pair_dir.name}: {type(exc).__name__}: {exc}")
                continue
            pairs.append((pair_dir.name, chain))

    if load_errors:
        typer.echo(
            f"Warning: {len(load_errors)} pair tree(s) under 'pairs/' have an "
            "adj_matrix.txt but no usable chain data (most likely every node "
            "in them failed to compute -- e.g. the run's engine wasn't "
            "available) and are not shown:"
        )
        for msg in load_errors:
            typer.echo(f"  - {msg}")

    network = None
    network_path = result_path / "network.json"
    if network_path.exists():
        try:
            network = Pot.read_from_disk(network_path)
        except Exception:
            network = None

    return ChannelsResult(
        reactant_conformers=reactant_conformers,
        product_conformers=product_conformers,
        pairs=pairs,
        network=network,
    )


def _load_channel_classification(result_path: Path) -> dict[str, str]:
    """If `result_path` (a `mepd ts`-shaped directory) is the `ts/`
    subdirectory of a `mepd channels` output, map every TS-guess label
    (e.g. "ts_pair_0_3_leaf_3") to the channel/alternate-route it was
    classified into (e.g. "Channel 0"), by reading the `members.txt`
    `_write_classified_group` already wrote under the sibling
    `channels/`/`alternate-routes/` folders. Returns an empty dict if
    `result_path` isn't part of a `mepd channels` output (a plain `mepd ts`
    output has no such siblings), so callers can fall back to their own
    generic labeling."""
    membership: dict[str, str] = {}
    root = result_path.parent
    for dirname, prefix in (("channels", "Channel"), ("alternate-routes", "Alternate route")):
        group_root = root / dirname
        if not group_root.is_dir():
            continue
        for group_dir in sorted(group_root.iterdir()):
            members_fp = group_dir / "members.txt"
            if not members_fp.is_file():
                continue
            index = group_dir.name.rsplit("_", 1)[-1]
            for line in members_fp.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("connects:"):
                    continue
                membership[line] = f"{prefix} {index}"
    return membership


def _load_channel_reference_reactant(result_path: Path, charge: int, multiplicity: int):
    """If `result_path` (a `mepd ts`-shaped directory) is the `ts/`
    subdirectory of a `mepd channels` output, return any one conformer of
    the original --start endpoint (from the sibling `conformers/start.xyz`
    pool) as a connectivity reference -- any conformer works, since
    connectivity matching ignores geometry entirely. Used to reorient each
    "Channel"-classified IRC so its reactant side is consistently
    `chain[0]` across every displayed channel, since displayed energies
    are relative to `chain[0]` (see `Chain.energies_kcalmol`) -- without
    this, two channels' IRCs could be shown with opposite orientations,
    making one display a forward barrier and the other a reverse barrier,
    not directly comparable. Returns `None` if there's no such sibling
    (e.g. a plain `mepd ts` output), so callers can skip reorientation."""
    from mepd.inputs import ChainInputs

    start_fp = result_path.parent / "conformers" / "start.xyz"
    if not start_fp.is_file():
        return None
    try:
        pool = Chain.from_xyz(start_fp, ChainInputs(), charge=charge or 0, spinmult=multiplicity or 1)
    except Exception:
        return None
    return pool[0] if len(pool) > 0 else None


def _load_ts_output_result(result_path: Path, charge: int, multiplicity: int):
    """Load a `mepd ts` (or `mepd run --use-tsopt`) output directory: one or
    more <label>.xyz TS structures, each optionally paired with an IRC path
    (<label>_irc.xyz, or irc.xyz for the bare "ts" label) -- see
    `_optimize_ts_and_irc`, which writes these -- into a `TsOutputResult`
    for `mepd visualize`. If this is a `mepd channels` output's `ts/`
    directory, also attaches its real channel/alternate-route
    classification (see `_load_channel_classification`) and reorients every
    "Channel"-classified IRC so its reactant side is consistently
    `chain[0]` (see `_load_channel_reference_reactant`)."""
    from mepd.inputs import ChainInputs
    from mepd.viz import TsOutputResult

    structures = []
    irc_paths = []
    load_errors = []

    group_labels = _load_channel_classification(result_path)
    reference_reactant = _load_channel_reference_reactant(result_path, charge, multiplicity)

    ts_files = sorted(
        p for p in result_path.glob("ts*.xyz") if not p.stem.endswith("_irc")
    )
    for ts_fp in ts_files:
        label = ts_fp.stem
        try:
            chain = Chain.from_xyz(ts_fp, ChainInputs(), charge=charge, spinmult=multiplicity)
            if len(chain) == 0:
                raise ValueError("empty xyz file")
        except Exception as exc:
            load_errors.append(f"{ts_fp.name}: {type(exc).__name__}: {exc}")
            continue
        structures.append((label, chain[0]))

        irc_fp = result_path / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
        if irc_fp.exists():
            try:
                irc_chain = Chain.from_xyz(irc_fp, ChainInputs(), charge=charge, spinmult=multiplicity)
                if (
                    reference_reactant is not None
                    and group_labels.get(label, "").startswith("Channel")
                    and len(irc_chain) > 1
                    and not _connectivity_matches(irc_chain[0], reference_reactant)
                    and _connectivity_matches(irc_chain[-1], reference_reactant)
                ):
                    irc_chain = irc_chain.copy()
                    irc_chain.nodes.reverse()
                irc_paths.append((label, irc_chain))
            except Exception as exc:
                load_errors.append(f"{irc_fp.name}: {type(exc).__name__}: {exc}")

    if load_errors:
        typer.echo(f"Warning: {len(load_errors)} file(s) could not be loaded and are not shown:")
        for msg in load_errors:
            typer.echo(f"  - {msg}")

    return TsOutputResult(structures=structures, irc_paths=irc_paths, group_labels=group_labels)


def _load_visualization_object(result_path: Path, charge: int, multiplicity: int):
    """Load whatever mepd result `result_path` points to, for `mepd
    visualize`: a chain xyz file, a network.json (a `Pot`), a split-tree
    directory (has adj_matrix.txt), a bare NEB history directory (has
    traj_*.xyz but no adj_matrix.txt -- e.g. a manually saved
    `<name>_history/` folder), a `mepd channels` output directory (has
    conformers/ and/or pairs/), or a `mepd ts`/`run --use-tsopt` output
    directory (has one or more ts*.xyz files)."""
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
        if (result_path / "conformers").is_dir() or (result_path / "pairs").is_dir():
            return _load_channels_result(result_path, charge, multiplicity)
        if list(result_path.glob("ts*.xyz")):
            return _load_ts_output_result(result_path, charge, multiplicity)
        raise typer.BadParameter(
            f"'{result_path}' is a directory but has neither adj_matrix.txt (a split-tree), "
            "traj_*.xyz files (a NEB history), conformers/ or pairs/ (a `mepd channels` "
            "output), nor ts*.xyz files (a `mepd ts` output)."
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
        "a network.json, a split-tree/NEB-history directory, a `mepd channels` "
        "output directory, or a `mepd ts`/`run --use-tsopt` output directory.",
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
    through, and a trajectory-step slider for whichever one is selected.
    For a `mepd channels` output directory, shows reactant conformers,
    product conformers, and completed MEP outputs as clickable groups. For
    a `mepd ts`/`run --use-tsopt` output directory, shows every optimized
    TS structure and its IRC path (if computed) as clickable groups."""
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
    paths: List[Path] = typer.Argument(
        ..., exists=True, file_okay=False,
        help="One or more input directories, auto-detected by content: MSMEP "
        "tree directories (contain adj_matrix.txt, as written by "
        "`mepd run --recursive`/`--parallel` to <output>/tree -- give several "
        "to combine into one network) OR a single directory of IRC "
        "xyz/.energies file pairs. The two kinds cannot be mixed in one run.",
    ),
    pattern: str = typer.Option(
        "*.xyz", "--pattern",
        help="Glob pattern for IRC xyz files. Only used when given an IRC directory.",
    ),
    recursive: bool = typer.Option(
        False, "--recursive",
        help="Search the IRC directory recursively. Only used when given an IRC directory.",
    ),
    charge: int = typer.Option(
        0, "--charge", help="Molecular charge of IRC structures. Only used when given an IRC directory.",
    ),
    multiplicity: int = typer.Option(
        1, "--multiplicity",
        help="Spin multiplicity of IRC structures. Only used when given an IRC directory.",
    ),
    output: Path = typer.Option(
        Path("network.json"), "--output", "-o",
        help="Path to write the resulting network (Pot) JSON to.",
    ),
) -> None:
    """Build/dedupe a reaction network from MSMEP tree directories and/or an
    IRC scan directory -- which kind each path is gets auto-detected."""
    tree_dirs = [p for p in paths if (p / "adj_matrix.txt").exists()]
    irc_dirs = [p for p in paths if p not in tree_dirs]

    if tree_dirs and irc_dirs:
        raise typer.BadParameter(
            "Cannot mix MSMEP tree directories (adj_matrix.txt) with an IRC scan "
            "directory in the same run. Build them separately."
        )
    if len(irc_dirs) > 1:
        raise typer.BadParameter("Only one IRC scan directory is supported per run.")

    if irc_dirs:
        from mepd.irc_network import build_irc_network

        try:
            scan = build_irc_network(
                irc_dirs[0], pattern=pattern, recursive=recursive,
                charge=charge, multiplicity=multiplicity,
            )
        except Exception as exc:
            typer.echo(f"IRC network construction failed: {type(exc).__name__}: {exc}")
            raise typer.Exit(code=1)

        pot = scan.pot
        pot.write_to_disk(output)
        typer.echo(
            f"Wrote network to {output} "
            f"({pot.number_of_nodes} nodes, {pot.graph.number_of_edges()} edges, "
            f"{len(scan.xyz_files)} files used, {len(scan.skipped_xyz_files)} skipped)"
        )
        return

    from mepd.NetworkBuilder import NetworkBuilder

    builder = NetworkBuilder(data_dir=tree_dirs[0].parent)
    try:
        pot = builder.create_rxn_network_from_paths(tree_dirs)
    except Exception as exc:
        typer.echo(f"Network construction failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    pot.write_to_disk(output)
    typer.echo(
        f"Wrote network to {output} "
        f"({pot.number_of_nodes} nodes, {pot.graph.number_of_edges()} edges)"
    )


@app.command("network-splits")
def network_splits(
    minima: List[Path] = typer.Argument(
        ..., exists=True, dir_okay=False,
        help="Two or more xyz files, each a single minimum geometry (not a "
        "multi-frame trajectory/chain). Every pair is connected by a "
        "recursive NEB/MSMEP run by default (--mode all-to-all); the "
        "results are combined into a completed reaction network.",
    ),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", exists=True,
        help="Path to a RunInputs TOML file. Uses built-in defaults if omitted.",
    ),
    charge: Optional[int] = typer.Option(
        None, "--charge", help="Override the molecular charge on every minimum."
    ),
    multiplicity: Optional[int] = typer.Option(
        None, "--multiplicity", help="Override the spin multiplicity on every minimum."
    ),
    mode: str = typer.Option(
        "all-to-all", "--mode",
        help="Pairing heuristic connecting the given minima. Currently only "
        "'all-to-all' (every pair) is supported; more heuristics (e.g. "
        "energy- or similarity-based pruning) may be added later.",
    ),
    max_pairs: int = typer.Option(
        100, "--max-pairs",
        help="Cap on the number of pairs to run (guards against the "
        "combinatorial blowup of all-to-all pairing for many minima).",
    ),
    parallel: bool = typer.Option(
        False, "--parallel",
        help="Run each pair's recursive autosplitting (MSMEP) with branches "
        "evaluated in parallel.",
    ),
    parallel_workers: Optional[int] = typer.Option(
        None, "--parallel-workers",
        help="Maximum number of concurrent workers for --parallel. Defaults to "
        "min(4, cpu count).",
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
    same_pair_split_limit: int = typer.Option(
        5, "--same-pair-split-limit",
        help="If a branch repeats the exact same (start, end) endpoint pair this "
        "many times in a row with no new chemistry found, stop splitting that "
        "branch further. A split that DOES discover a new molecule/conformer is "
        "never cut off by this.",
    ),
    output: Path = typer.Option(
        Path("mepd_network_splits_output"), "--output", "-o",
        help="Directory to write pair trees and the completed network into.",
    ),
) -> None:
    """Run the network-splits workflow directly from a list of minima,
    instead of discovering intermediates from a single start/end MSMEP run:
    every pair of the given minima is connected by a recursive NEB/MSMEP run
    (all-to-all by default), and the results are combined into a completed
    reaction network."""
    if len(minima) < 2:
        raise typer.BadParameter("Provide at least two minima.")
    if mode != "all-to-all":
        raise typer.BadParameter("--mode currently only supports 'all-to-all'.")
    if max_pairs <= 0:
        raise typer.BadParameter("--max-pairs must be a positive integer.")
    if same_pair_split_limit <= 0:
        raise typer.BadParameter("--same-pair-split-limit must be a positive integer.")

    from mepd.nodes.node import StructureNode
    from mepd.NetworkBuilder import NetworkBuilder

    run_inputs = _open_run_inputs(inputs)
    run_inputs.path_min_inputs.validate_minima_with_hessian = validate_minima_with_hessian
    run_inputs.path_min_inputs.hessian_minimum_frequency_cutoff = hessian_minimum_frequency_cutoff
    run_inputs.path_min_inputs.hessian_minima_rescue_displacement = hessian_minima_rescue_displacement
    run_inputs.path_min_inputs.recursive_same_pair_split_limit = same_pair_split_limit
    _echo_run_inputs_summary(run_inputs)

    structures = [
        StructureNode(structure=_load_endpoint(fp, charge, multiplicity))
        for fp in minima
    ]

    candidates = [
        (i, j) for i in range(len(structures)) for j in range(i + 1, len(structures))
    ]

    identical_pairs = [
        (i, j) for i, j in candidates
        if _connectivity_matches(structures[i], structures[j])
    ]
    if identical_pairs:
        for i, j in identical_pairs:
            typer.echo(
                f"Skipping pair ({minima[i].name}, {minima[j].name}): "
                "endpoints are connectivity-identical, nothing to connect."
            )
        candidates = [pair for pair in candidates if pair not in identical_pairs]
        if not candidates:
            raise typer.BadParameter(
                "Every candidate pair has connectivity-identical endpoints -- "
                "provide at least two structurally distinct minima."
            )

    if len(candidates) > max_pairs:
        typer.echo(
            f"{len(candidates)} candidate pairs found (all-to-all over "
            f"{len(structures)} minima), capping at --max-pairs={max_pairs}."
        )
        candidates = candidates[:max_pairs]

    output.mkdir(parents=True, exist_ok=True)
    pairs_dir = output / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    _run_msmep_pairs(
        structures, candidates, pairs_dir, run_inputs,
        parallel=parallel, parallel_workers=parallel_workers,
    )

    tree_dirs = _completed_tree_dirs(pairs_dir)
    if not tree_dirs:
        typer.echo("No pairs completed successfully; nothing to build a network from.")
        raise typer.Exit(code=1)

    builder = NetworkBuilder(data_dir=output, network_inputs=NetworkInputs())
    try:
        pot = builder.create_rxn_network_from_paths(tree_dirs)
    except Exception as exc:
        typer.echo(f"Network construction failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    network_path = output / "network.json"
    pot.write_to_disk(network_path)
    typer.echo(
        f"Wrote network to {network_path} "
        f"({pot.number_of_nodes} nodes, {pot.graph.number_of_edges()} edges)"
    )


def _minimize_conformer_pool(nodes: list, label: str, run_inputs: RunInputs) -> list:
    """Optimize every node in a conformer pool with the QM engine, dropping
    (with a warning) any conformer that fails to converge or produces an
    empty trajectory rather than failing the whole batch -- unlike
    `_minimize_endpoints` (which hard-stops on a single non-convergent
    endpoint), one bad conformer out of many shouldn't sink the run.
    """
    from mepd.errors import GeometryOptimizationNotConvergedError

    if not nodes:
        return nodes

    keywords = _geometry_optimizer_keywords(run_inputs)
    optimized: list = []

    batch_optimizer = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    if callable(batch_optimizer):
        try:
            try:
                trajectories = batch_optimizer(nodes, keywords=keywords)
            except TypeError:
                trajectories = batch_optimizer(nodes)
        except Exception as exc:
            typer.echo(
                f"{label} conformer batch minimization failed "
                f"({type(exc).__name__}: {exc}); keeping input geometries."
            )
            return nodes
        for i, trajectory in enumerate(trajectories):
            if trajectory:
                optimized.append(trajectory[-1])
            else:
                typer.echo(f"{label} conformer {i} optimization returned an empty trajectory; dropping it.")
        return optimized

    single_optimizer = getattr(run_inputs.engine, "compute_geometry_optimization", None)
    if not callable(single_optimizer):
        typer.echo(
            f"Engine {type(run_inputs.engine).__name__} does not support geometry "
            f"optimization; keeping input geometries for {label}."
        )
        return nodes

    for i, node in enumerate(nodes):
        try:
            trajectory = single_optimizer(node, keywords=keywords)
            if trajectory:
                optimized.append(trajectory[-1])
            else:
                typer.echo(f"{label} conformer {i} optimization returned an empty trajectory; dropping it.")
        except GeometryOptimizationNotConvergedError as exc:
            typer.echo(f"{label} conformer {i} did not converge ({exc}); dropping it.")
        except Exception as exc:
            typer.echo(f"{label} conformer {i} minimization failed ({type(exc).__name__}: {exc}); dropping it.")
    return optimized


def _connectivity_matches(a, b) -> bool:
    """Same molecule (bond connectivity + stereochemistry), ignoring
    conformation -- mirrors `NetworkBuilder._graph_equivalent`, used here to
    compare an IRC-recovered endpoint against the originally requested
    --start/--end structures without caring which conformer it landed on."""
    from mepd.nodes.nodehelpers import _is_connectivity_identical

    if getattr(a, "graph", None) is None or getattr(b, "graph", None) is None:
        return False
    return _is_connectivity_identical(a, b, verbose=False, collect_comparison=False)


def _register_route_class(known: list, node) -> int:
    """Greedily assign `node` to an existing connectivity class in `known`
    (a list of representative nodes so far), registering a new class if none
    match. Mirrors `NetworkBuilder._get_ind_td`'s registration pattern."""
    for i, rep in enumerate(known):
        if _connectivity_matches(node, rep):
            return i
    known.append(node)
    return len(known) - 1


def _cluster_by_ts_identity(candidates: list[tuple], run_inputs: RunInputs) -> list[list[tuple]]:
    """Greedily cluster `(ts_node, irc_chain, label)` candidates into classes
    whose TS structures are pairwise `is_identical` -- the same cutoffs
    `NetworkBuilder._equality_function`/`check_if_elem_step` already use for
    "are these the same structure"."""
    from mepd.nodes.nodehelpers import is_identical

    clusters: list[list[tuple]] = []
    for candidate in candidates:
        ts_node = candidate[0]
        for cluster in clusters:
            if is_identical(
                ts_node, cluster[0][0],
                fragment_rmsd_cutoff=run_inputs.chain_inputs.node_rms_thre,
                kcal_mol_cutoff=run_inputs.chain_inputs.node_ene_thre,
                verbose=False,
            ):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    return clusters


def _load_ts_and_irc_from_disk(
    ts_dir: Path, label: str, charge: int, multiplicity: int
) -> Optional["TsIrcResult"]:
    """Resume support for `channels`: if a prior run already wrote this
    label's TS (and IRC) to disk, reload them instead of skip-and-forget --
    unlike `ts`'s simpler "already done" skip, `channels` needs the actual
    IRC endpoints to classify this result, not just a done/not-done flag."""
    from mepd.inputs import ChainInputs

    ts_path = ts_dir / f"{label}.xyz"
    if not ts_path.exists():
        return None
    ts_chain = Chain.from_xyz(ts_path, ChainInputs(), charge=charge, spinmult=multiplicity)
    ts_node = ts_chain[0]

    irc_path = ts_dir / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz")
    irc_chain = None
    if irc_path.exists():
        irc_chain = Chain.from_xyz(irc_path, ChainInputs(), charge=charge, spinmult=multiplicity)
    return TsIrcResult(ts_node=ts_node, irc_chain=irc_chain)


def _write_classified_group(
    candidates: list[tuple],
    out_dir: Path,
    prefix: str,
    index: int,
    extra_info: Optional[str] = None,
) -> None:
    from mepd.inputs import ChainInputs

    group_dir = out_dir / f"{prefix}_{index}"
    group_dir.mkdir(parents=True, exist_ok=True)

    def _ts_energy(candidate):
        try:
            return candidate[0].energy
        except Exception:
            return float("inf")

    ts_node, irc_chain, _ = min(candidates, key=_ts_energy)
    Chain.model_validate(
        {"nodes": [ts_node], "parameters": ChainInputs()}
    ).write_to_disk(group_dir / "ts.xyz")
    irc_chain.write_to_disk(group_dir / "irc.xyz")

    lines = [extra_info] if extra_info else []
    lines.extend(label for _, _, label in candidates)
    (group_dir / "members.txt").write_text("\n".join(lines) + "\n")


def _discover_channels(
    output: Path,
    start_node,
    end_node,
    run_inputs: RunInputs,
    charge: Optional[int],
    multiplicity: Optional[int],
) -> None:
    """TS-opt + IRC every leaf across every completed pair tree in `output`,
    then classify by IRC endpoints: a leaf whose IRC reconnects the
    original --start/--end pair (regardless of which conformer/seed
    produced it) is a channel candidate; a leaf whose IRC connects some
    other pair of real minima is an alternate-route candidate; a leaf whose
    TS optimization or IRC failed to converge is dropped from both. Within
    each bucket, candidates are further deduplicated into distinct TS
    classes -- this is what turns e.g. 4 conformer-pair runs into "2
    channels" when 2 of those runs converged to essentially the same TS.
    """
    from mepd.NetworkBuilder import _stereochemical_smiles_key

    tree_charge = charge if charge is not None else 0
    tree_multiplicity = multiplicity if multiplicity is not None else 1

    tasks = _collect_ts_guess_tasks(output, charge, multiplicity)
    if not tasks:
        typer.echo("No TS guesses found across completed pairs; no channels to classify.")
        return

    ts_dir = output / "ts"
    ts_dir.mkdir(parents=True, exist_ok=True)

    channel_candidates: list[tuple] = []
    route_classes: list = []
    route_buckets: dict = {}
    n_failed = 0

    for label, guess_node in tasks:
        result = _load_ts_and_irc_from_disk(ts_dir, label, tree_charge, tree_multiplicity)
        if result is None:
            result = _optimize_ts_and_irc(guess_node, run_inputs, ts_dir, run_irc=True, label=label)
        if result is None or result.irc_chain is None or len(result.irc_chain) < 2:
            n_failed += 1
            continue

        irc_first, irc_last = result.irc_chain[0], result.irc_chain[-1]
        candidate = (result.ts_node, result.irc_chain, label)
        if (
            _connectivity_matches(irc_first, start_node) and _connectivity_matches(irc_last, end_node)
        ) or (
            _connectivity_matches(irc_last, start_node) and _connectivity_matches(irc_first, end_node)
        ):
            channel_candidates.append(candidate)
            continue

        i = _register_route_class(route_classes, irc_first)
        j = _register_route_class(route_classes, irc_last)
        if i == j:
            # IRC collapsed to the same minimum on both sides -- not a
            # genuine two-minima route.
            n_failed += 1
            continue
        route_buckets.setdefault(frozenset((i, j)), []).append(candidate)

    channels_dir = output / "channels"
    alt_dir = output / "alternate-routes"

    channel_clusters = _cluster_by_ts_identity(channel_candidates, run_inputs)
    for k, cluster in enumerate(channel_clusters):
        _write_classified_group(cluster, channels_dir, "channel", k)

    n_routes = 0
    for (i, j), bucket_candidates in route_buckets.items():
        smi_a = _stereochemical_smiles_key(route_classes[i]) or "?"
        smi_b = _stereochemical_smiles_key(route_classes[j]) or "?"
        for cluster in _cluster_by_ts_identity(bucket_candidates, run_inputs):
            _write_classified_group(
                cluster, alt_dir, "route", n_routes,
                extra_info=f"connects: {smi_a} <-> {smi_b}",
            )
            n_routes += 1

    typer.echo(
        f"{len(channel_clusters)} channel(s) found for the requested pair "
        f"({len(channel_candidates)} contributing run(s)), {n_routes} "
        f"alternate route(s), {n_failed} failed."
    )


@app.command("channels")
def channels(
    start: str = typer.Option(..., "--start", help="Path to the reactant-endpoint xyz file, or a SMILES string."),
    end: str = typer.Option(..., "--end", help="Path to the product-endpoint xyz file, or a SMILES string."),
    method: str = typer.Option(
        "conformers", "--method",
        help="Seed-generation strategy for perturbing the search to surface "
        "alternate TS channels between --start and --end. Currently only "
        "'conformers' (RDKit conformer sampling of each endpoint) is "
        "implemented; more methods may be added later.",
    ),
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
    atom_mapping: bool = typer.Option(
        False, "--atom-mapping",
        help="Every run checks whether SLAPMapper's Weisfeiler-Lehman-like/"
        "sequential-LAP atom-to-atom mapping (Koda, ChemRxiv 2025) between "
        "--start and --end agrees with their shared input atom ordering, "
        "warning if not. Passing this flag additionally reindexes --end's "
        "atoms (and therefore every product conformer generated from it) to "
        "match the suggested mapping instead of just warning. No-op when "
        "--start/--end are both SMILES or atom counts differ.",
    ),
    debug_dump: bool = typer.Option(
        False, "--debug-dump",
        help="Write extra diagnostic structures to disk for inspection (e.g. via "
        "`mepd visualize`): (1) when --start/--end is SMILES and --minimize-ends "
        "runs its pre-conformer-sampling minimization, the minimized endpoints go "
        "to <output>/smiles_minimization_debug/; (2) when --atom-mapping triggers "
        "its candidate-mapping selection, every candidate's interpolated path "
        "(xyz + energies) and a scores.txt comparing all three selection metrics "
        "per candidate go to <output>/realign_debug/.",
    ),
    atom_mapping_candidates: int = typer.Option(
        200, "--atom-mapping-candidates",
        help="--atom-mapping: how many of SLAPMapper's equal-minimal-cost "
        "candidate mappings (plus their symmetry-orbit expansions) to keep "
        "and consider (they're ties, not ranked by quality among themselves).",
    ),
    atom_mapping_metric: str = typer.Option(
        "geodesic-distance", "--atom-mapping-metric",
        help="--atom-mapping: how each candidate mapping (including 'don't "
        "reindex') is scored from its geodesic-interpolated path -- "
        "'geodesic-distance' (the geodesic optimizer's own path length; "
        "free), 'path-rmsd' (cumulative per-frame RMSD along the path; "
        "free), or 'gi-energy' (highest QM energy along the path; most "
        "expensive, one engine energy evaluation per candidate). Which is "
        "actually the best predictor of a correct mapping isn't settled -- "
        "--debug-dump records all three per candidate to help compare them.",
    ),
    atom_mapping_veto_margin: float = typer.Option(
        0.0, "--atom-mapping-veto-margin",
        help="--atom-mapping: a non-identity candidate must beat 'don't "
        "reindex' by more than this (in --atom-mapping-metric's own units) to "
        "be adopted; otherwise the original --end ordering is kept even if "
        "some candidate scored marginally better. Default 0.0 is a pure "
        "best-of-N with no calibrated stability margin yet.",
    ),
    atom_mapping_recheck_splits: bool = typer.Option(
        False, "--atom-mapping-recheck-splits",
        help="Experimental: also re-run the same best-of-N atom-mapping "
        "selection (--atom-mapping-metric etc.) at every new (reactant, "
        "product) pair MSMEP's recursive splitting discovers, not just the "
        "original --start/--end pair. Independent of --atom-mapping. Atom "
        "order is otherwise preserved throughout the recursion, so this only "
        "changes anything for a split whose own reactant/product pair "
        "happens to have SLAPMapper-detectable symmetry.",
    ),
    backend: str = typer.Option(
        "rdkit", "--backend",
        help="Conformer-generation backend. Currently only 'rdkit' is implemented; "
        "'crest' is planned as a follow-up.",
    ),
    n_conformers: int = typer.Option(
        10, "--n-conformers",
        help="Maximum number of distinct conformers to keep for EACH endpoint "
        "after generation and RMSD-based deduplication.",
    ),
    n_embed: int = typer.Option(
        50, "--n-embed",
        help="Number of raw conformer embeddings to attempt per endpoint before "
        "deduplication (rdkit backend only). Should comfortably exceed --n-conformers.",
    ),
    rmsd_cutoff: float = typer.Option(
        0.5, "--rmsd-cutoff",
        help="Minimum pairwise RMSD (bohr) for two conformers of the same endpoint "
        "to count as distinct.",
    ),
    random_seed: int = typer.Option(0, "--random-seed", help="Random seed for conformer embedding."),
    minimize_ends: bool = typer.Option(
        True, "--minimize-ends/--no-minimize-ends",
        help="Optimize endpoint geometries with the QM engine. On by default. "
        "This governs two separate minimizations: (1) BEFORE conformer sampling, "
        "if --start/--end was given as SMILES (an RDKit/openbabel-embedded guess, "
        "not a real minimum), that raw embedded structure is minimized at your "
        "input level of theory first, so conformer sampling starts from an actual "
        "minimum rather than an arbitrary embedding; and (2) every generated "
        "conformer is optimized before pairing, since RDKit/MMFF conformers are "
        "not QM minima either. A conformer that fails to converge in step (2) is "
        "dropped rather than failing the whole run; a SMILES endpoint that fails "
        "to converge in step (1) is a hard stop (see `mepd run --minimize-ends`).",
    ),
    max_pairs: int = typer.Option(
        100, "--max-pairs",
        help="Hard cap on the total number of reactant x product conformer pairs to "
        "run (default matches the default 10 x 10 endpoint caps; raise this if more "
        "compute is available).",
    ),
    parallel: bool = typer.Option(
        False, "--parallel",
        help="Run each pair's recursive autosplitting (MSMEP) with branches "
        "evaluated in parallel.",
    ),
    parallel_workers: Optional[int] = typer.Option(
        None, "--parallel-workers",
        help="Maximum number of concurrent workers for --parallel. Defaults to "
        "min(4, cpu count).",
    ),
    validate_minima_with_hessian: bool = typer.Option(
        True, "--validate-minima-with-hessian/--no-validate-minima-with-hessian", "-H/-noH",
        help="When a minima-based autosplit is proposed during each pair's MSMEP, "
        "compute Hessians for optimized split candidates and reject candidates "
        "with significant imaginary modes. On by default -- this is a "
        "correctness check, not a convenience.",
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
    output: Path = typer.Option(
        Path("mepd_channels_output"), "--output", "-o",
        help="Directory to write seed pools, per-seed trees, TS-opt+IRC "
        "results, classified channels/alternate-routes, and the completed "
        "byproduct network into.",
    ),
) -> None:
    """Multi-channel TS discovery for a fixed --start/--end pair: generate
    alternate seed guesses (--method), run a recursive NEB/MSMEP for every
    seed, optimize a TS and IRC for every resulting leaf, then classify by
    IRC endpoints -- results whose IRC reconnects the requested --start/--end
    pair land in <output>/channels/ (deduplicated into distinct TS classes,
    e.g. 4 conformer-pair runs converging to 2 real channels), results that
    reconnect some other pair of minima land in <output>/alternate-routes/,
    and everything completed is also combined into one byproduct network at
    <output>/network.json."""
    if method != "conformers":
        raise typer.BadParameter(f"Unknown --method '{method}'. Known: 'conformers'.")
    if backend not in ("rdkit", "crest"):
        raise typer.BadParameter(f"Unknown --backend '{backend}'. Known: 'rdkit', 'crest'.")
    if n_conformers <= 0:
        raise typer.BadParameter("--n-conformers must be a positive integer.")
    if n_embed <= 0:
        raise typer.BadParameter("--n-embed must be a positive integer.")
    if rmsd_cutoff <= 0:
        raise typer.BadParameter("--rmsd-cutoff must be a positive number.")
    if max_pairs <= 0:
        raise typer.BadParameter("--max-pairs must be a positive integer.")
    if atom_mapping_metric not in _ATOM_MAPPING_METRICS:
        raise typer.BadParameter(
            f"--atom-mapping-metric must be one of {_ATOM_MAPPING_METRICS}."
        )
    if atom_mapping_candidates <= 0:
        raise typer.BadParameter("--atom-mapping-candidates must be a positive integer.")

    from mepd.conformers import ConformerInputs
    from mepd.sampling import generate_seed_pairs
    from mepd.nodes.node import StructureNode
    from mepd.NetworkBuilder import NetworkBuilder

    run_inputs = _open_run_inputs(inputs)
    run_inputs.path_min_inputs.validate_minima_with_hessian = validate_minima_with_hessian
    run_inputs.path_min_inputs.hessian_minimum_frequency_cutoff = hessian_minimum_frequency_cutoff
    run_inputs.path_min_inputs.hessian_minima_rescue_displacement = hessian_minima_rescue_displacement
    run_inputs.atom_mapping_inputs.n_candidates = atom_mapping_candidates
    run_inputs.atom_mapping_inputs.metric = atom_mapping_metric
    run_inputs.atom_mapping_inputs.veto_margin = atom_mapping_veto_margin
    run_inputs.atom_mapping_inputs.recheck_on_split = atom_mapping_recheck_splits
    _echo_run_inputs_summary(run_inputs)

    start_is_smiles = not Path(start).exists()
    end_is_smiles = not Path(end).exists()

    start_structure = _load_structure_from_smiles_or_xyz(start, charge, multiplicity)
    end_structure = _load_structure_from_smiles_or_xyz(end, charge, multiplicity)
    start_node = StructureNode(structure=start_structure)
    end_node = StructureNode(structure=end_structure)

    if minimize_ends and (start_is_smiles or end_is_smiles):
        typer.echo(
            "An endpoint was given as SMILES (an RDKit/openbabel-embedded guess, "
            "not a real minimum); minimizing endpoints at the input level of "
            "theory before checking the --start/--end atom mapping and "
            "sampling conformers. Pass --no-minimize-ends to skip."
        )
        start_node, end_node = _minimize_endpoints(start_node, end_node, run_inputs)
        if debug_dump:
            from mepd.inputs import ChainInputs

            debug_dir = Path(output) / "smiles_minimization_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            Chain.model_validate(
                {"nodes": [start_node], "parameters": ChainInputs()}
            ).write_to_disk(debug_dir / "start_minimized.xyz")
            Chain.model_validate(
                {"nodes": [end_node], "parameters": ChainInputs()}
            ).write_to_disk(debug_dir / "end_minimized.xyz")
            typer.echo(f"  --debug-dump: wrote minimized SMILES endpoints to {debug_dir}")

    # The --atom-mapping sanity check compares geodesic-interpolation path
    # energies between the unpermuted and permuted --end -- run it AFTER
    # minimization (rather than on the raw SMILES embedding) so that
    # comparison reflects real minima, not embedding artifacts/strain that
    # can otherwise dominate the energy delta the veto decision is based on.
    realigned_end_structure = _check_endpoint_atom_mapping(
        start_node.structure, end_node.structure, atom_mapping, run_inputs,
        debug_dump=debug_dump, output=output,
    )
    if realigned_end_structure is not end_node.structure:
        end_node = StructureNode(structure=realigned_end_structure)

    conformer_inputs = ConformerInputs(
        backend=backend,
        n_conformers=n_conformers,
        n_embed=n_embed,
        rmsd_cutoff=rmsd_cutoff,
        random_seed=random_seed,
    )

    typer.echo(f"Generating seed pairs (--method {method})...")
    start_confs, end_confs = generate_seed_pairs(
        method, start_node, end_node, conformer_inputs=conformer_inputs,
    )
    typer.echo(
        f"  -> {len(start_confs)} start-endpoint seed(s), "
        f"{len(end_confs)} end-endpoint seed(s)."
    )

    if minimize_ends:
        typer.echo("Minimizing start-endpoint conformers...")
        start_confs = _minimize_conformer_pool(start_confs, "start", run_inputs)
        typer.echo("Minimizing end-endpoint conformers...")
        end_confs = _minimize_conformer_pool(end_confs, "end", run_inputs)

    if not start_confs or not end_confs:
        typer.echo("No usable conformers for one or both endpoints; nothing to run.")
        raise typer.Exit(code=1)

    structures = start_confs + end_confs
    n_start = len(start_confs)
    candidates = [
        (i, n_start + j) for i in range(n_start) for j in range(len(end_confs))
    ]

    if len(candidates) > max_pairs:
        typer.echo(
            f"{len(candidates)} candidate reactant x product conformer pairs found "
            f"({n_start} x {len(end_confs)}), capping at --max-pairs={max_pairs}."
        )
        candidates = candidates[:max_pairs]

    output.mkdir(parents=True, exist_ok=True)

    # Persisted so `mepd visualize <output>` can show the actual conformer
    # pools used for pairing (post-minimization, if --minimize-ends), not
    # just the completed pairs -- otherwise there'd be no way to see e.g. a
    # reactant conformer that never made it into any --max-pairs-capped pair.
    from mepd.inputs import ChainInputs

    conformers_dir = output / "conformers"
    conformers_dir.mkdir(parents=True, exist_ok=True)
    Chain.model_validate(
        {"nodes": start_confs, "parameters": ChainInputs()}
    ).write_to_disk(conformers_dir / "start.xyz")
    Chain.model_validate(
        {"nodes": end_confs, "parameters": ChainInputs()}
    ).write_to_disk(conformers_dir / "end.xyz")

    pairs_dir = output / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    _run_msmep_pairs(
        structures, candidates, pairs_dir, run_inputs,
        parallel=parallel, parallel_workers=parallel_workers,
    )

    tree_dirs = _completed_tree_dirs(pairs_dir)
    if not tree_dirs:
        typer.echo("No pairs completed successfully; nothing to build a network from.")
        raise typer.Exit(code=1)

    builder = NetworkBuilder(data_dir=output, network_inputs=NetworkInputs())
    try:
        pot = builder.create_rxn_network_from_paths(tree_dirs)
    except Exception as exc:
        typer.echo(f"Network construction failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    network_path = output / "network.json"
    pot.write_to_disk(network_path)
    typer.echo(
        f"Wrote network to {network_path} "
        f"({pot.number_of_nodes} nodes, {pot.graph.number_of_edges()} edges)"
    )

    _discover_channels(output, start_node, end_node, run_inputs, charge, multiplicity)


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


try:
    from mepd.discovery.cli import discovery_app  # noqa: E402
except ImportError:
    discovery_app = None

if discovery_app is not None:
    app.add_typer(discovery_app, name="discovery")
else:

    @app.command(
        "discovery",
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    def _discovery_unavailable(ctx: typer.Context) -> None:  # noqa: E402
        """Structure-discovery tools (unavailable: mepd.discovery failed to import)."""
        typer.echo(
            "mepd discovery is unavailable: the mepd.discovery submodule could not be "
            "imported. Install its dependencies (e.g. `pip install mepd[discovery]`) "
            "and try again."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
