"""Minimal command-line interface for mepd.

This is intentionally a thin CLI, scoped to what this package currently
supports: running a single- or multi-step nudged-elastic-band (NEB)
optimization between two endpoint structures (including recursive
autosplitting via MSMEP, serial or parallel), transition-state
optimization, and network-completion (building/completing a reaction
network graph from already-computed MSMEP/IRC results).

Shared run/endpoint/TS-guess/MSMEP-pair-search plumbing lives in
`mepd.cli_common`; `channels` (by far the largest single command) lives in
`mepd.cli_channels` and is registered onto `app` below.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List, Optional

import typer

from mepd.atom_mapping_selection import METRICS as _ATOM_MAPPING_METRICS
from mepd.chain import Chain
from mepd.cli_channels import channels
from mepd.cli_common import (
    _check_endpoint_atom_mapping,
    _collect_ts_guess_tasks,
    _completed_tree_dirs,
    _connectivity_matches,
    _echo_run_inputs_summary,
    _geometry_optimizer_keywords,
    _load_endpoint,
    _load_structure_from_smiles_or_xyz,
    _minimize_endpoints,
    _open_run_inputs,
    _optimize_ts_and_irc,
    _run_msmep_pairs,
)
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
        "reindex') is scored -- 'geodesic-distance' (the geodesic optimizer's "
        "own path length; needs a full interpolation per candidate) or "
        "'path-rmsd' (cumulative per-frame RMSD along the path; same cost) are "
        "the defaults' cost class; 'gi-energy' (highest QM energy along the "
        "path) adds one engine evaluation per candidate on top of that; "
        "'endpoint-rmsd' (Kabsch RMSD between the two fixed endpoints, no "
        "interpolation at all -- orders of magnitude cheaper, but knows "
        "nothing about what happens ALONG the path, so it's the weakest "
        "signal of the four; EXPERIMENTAL, see docs/channels_candidates.md's "
        "open-problem note on mapping cost before relying on it). Which "
        "actually best predicts a correct mapping isn't settled -- "
        "--debug-dump records all four per candidate to help compare them.",
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
    (e.g. "ts_pair_0_3_leaf_3") to the channel/alternate-channel/
    off-target-exit-channel it was
    classified into (e.g. "Channel 0"), by reading the `members.txt`
    `_write_classified_group` already wrote under the sibling
    `channels/`/`alternate-channels/`/`offtarget-exit-channels/` folders.
    Returns an empty dict if `result_path` isn't part of a `mepd channels`
    output (a plain `mepd ts` output has no such siblings), so callers can
    fall back to their own generic labeling.

    `alternate-channels/` nests one level deeper than the other two -- a
    multistep route holds a `step_<n>/` per leg -- so its members are
    labeled with both the route and which step of it this TS is."""
    membership: dict[str, str] = {}
    root = result_path.parent

    def _absorb(group_dir: Path, label: str) -> None:
        members_fp = group_dir / "members.txt"
        if not members_fp.is_file():
            return
        for line in members_fp.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("connects:"):
                continue
            membership[line] = label

    for dirname, prefix in (
        ("channels", "Channel"),
        ("offtarget-exit-channels", "Off-target exit channel"),
    ):
        group_root = root / dirname
        if not group_root.is_dir():
            continue
        for group_dir in sorted(group_root.iterdir()):
            _absorb(group_dir, f"{prefix} {group_dir.name.rsplit('_', 1)[-1]}")

    alt_root = root / "alternate-channels"
    if alt_root.is_dir():
        for route_dir in sorted(alt_root.iterdir()):
            if not route_dir.is_dir():
                continue
            route_index = route_dir.name.rsplit("_", 1)[-1]
            for step_dir in sorted(route_dir.iterdir()):
                if not step_dir.is_dir():
                    continue
                step_index = step_dir.name.rsplit("_", 1)[-1]
                _absorb(step_dir, f"Alternate channel {route_index} step {step_index}")
    return membership


def _irc_needs_reversal(irc_chain, reactant, product) -> bool:
    """Whether to flip `irc_chain` so every IRC of a `mepd channels` run is
    drawn from the same molecular graph (energies are shown relative to
    `chain[0]`, so mixed orientations would mix forward and reverse
    barriers).

    Decided by the first rule that tells the two ends apart:
      1. the end that is the requested reactant (connectivity AND
         stereochemistry) goes first;
      2. else the end with the reactant's connectivity, stereochemistry
         ignored (e.g. trans-3,4-dimethylcyclobutene in a run started from
         the cis isomer) goes first;
      3. else the end that is the requested product -- again exact, then
         connectivity only -- goes last.
    If no rule tells them apart, the IRC is left as written."""
    from mepd.nodes.nodehelpers import _is_connectivity_identical

    if irc_chain is None or len(irc_chain) < 2:
        return False
    first, last = irc_chain[0], irc_chain[-1]

    def _matches(node, ref, disregard_stereochem):
        if ref is None or getattr(node, "graph", None) is None or getattr(ref, "graph", None) is None:
            return False
        try:
            return _is_connectivity_identical(
                node, ref, verbose=False, collect_comparison=False,
                disregard_stereochem=disregard_stereochem,
            )
        except Exception:
            return False

    for ref, want_first in ((reactant, True), (product, False)):
        for loose in (False, True):
            a, b = _matches(first, ref, loose), _matches(last, ref, loose)
            if a != b:
                return b if want_first else a
    return False


def _load_channel_reference_reactant(
    result_path: Path, charge: int, multiplicity: int, side: str = "start",
):
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

    start_fp = result_path.parent / "conformers" / f"{side}.xyz"
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
    directory, also attaches its real channel / alternate-channel /
    off-target-exit-channel classification (see
    `_load_channel_classification`) and reorients every classified IRC that
    actually relaxes into the reactant on its far side so that side is
    consistently `chain[0]` (see `_load_channel_reference_reactant`) -- the
    first leg of an alternate channel, and an off-target exit off the
    reactant, want the same orientation a direct channel gets."""
    from mepd.inputs import ChainInputs
    from mepd.viz import TsOutputResult

    structures = []
    irc_paths = []
    load_errors = []

    group_labels = _load_channel_classification(result_path)
    reference_reactant = _load_channel_reference_reactant(result_path, charge, multiplicity)
    reference_product = _load_channel_reference_reactant(
        result_path, charge, multiplicity, side="end"
    )

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
                if _irc_needs_reversal(irc_chain, reference_reactant, reference_product):
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


# `channels` lives in mepd/cli_channels.py (by far the largest single
# command); registered here rather than decorated there so that module
# never needs to import from this one.
app.command("channels")(channels)


@app.command("optimize")
def optimize(
    structures: List[Path] = typer.Argument(
        ..., help="xyz file(s) to optimize. A multi-frame xyz contributes every frame."
    ),
    inputs: Optional[Path] = typer.Option(
        None, "--inputs", "-i", help="Path to a RunInputs TOML file (engine / level of theory)."
    ),
    charge: Optional[int] = typer.Option(None, "--charge", help="Override the molecular charge."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity", help="Override the spin multiplicity."),
    validate_minima_with_hessian: bool = typer.Option(
        False,
        "--validate-minima-with-hessian/--no-validate-minima-with-hessian",
        "-H/-noH",
        help="After optimizing, compute a Hessian and require no frequency below "
        "--hessian-minimum-frequency-cutoff. A structure that fails is displaced along its "
        "lowest mode (both directions) and reoptimized once; if it still fails it is reported "
        "as not a minimum.",
    ),
    hessian_minimum_frequency_cutoff: float = typer.Option(
        0.0, "--hessian-minimum-frequency-cutoff", help="Minimum allowed frequency (cm^-1)."
    ),
    hessian_minima_rescue_displacement: float = typer.Option(
        0.1, "--hessian-minima-rescue-displacement", help="Rescue displacement along the unstable mode (bohr)."
    ),
    output: Path = typer.Option(
        Path("mepd_optimize_output"), "--output", "-o", help="Directory to write results into."
    ),
) -> None:
    """Geometry-optimize structures at the level of theory in --inputs.

    Writes opt_<i>.xyz (+ .energies) for every structure that converged, in
    input order, and summary.json with one record per structure
    ({index, source, converged, energy, error, and with Hessian validation
    is_minimum, min_frequency, rescued, validation}). One structure failing
    does not stop the others; the exit code is 1 only if none converged.
    """
    from mepd.errors import GeometryOptimizationNotConvergedError
    from mepd.inputs import ChainInputs
    from mepd.nodes.node import StructureNode
    from mepd.qcdata_structure_helpers import read_multiple_structure_from_file

    run_inputs = RunInputs.open(inputs) if inputs else RunInputs()
    output.mkdir(parents=True, exist_ok=True)

    nodes, sources = [], []
    for fp in structures:
        frames = read_multiple_structure_from_file(fp, charge=charge or 0, spinmult=multiplicity or 1)
        for k, st in enumerate(frames):
            updates = {}
            if charge is not None:
                updates["charge"] = charge
            if multiplicity is not None:
                updates["multiplicity"] = multiplicity
            nodes.append(StructureNode(structure=st.model_copy(update=updates) if updates else st))
            sources.append(f"{fp.name}" + (f"[{k}]" if len(frames) > 1 else ""))

    keywords = _geometry_optimizer_keywords(run_inputs)
    engine = run_inputs.engine
    results: list = [None] * len(nodes)
    errors: list = [None] * len(nodes)

    def _single(i: int) -> None:
        try:
            try:
                traj = engine.compute_geometry_optimization(nodes[i], keywords=keywords)
            except TypeError:
                traj = engine.compute_geometry_optimization(nodes[i])
            if traj:
                results[i] = traj[-1]
            else:
                errors[i] = "optimizer returned an empty trajectory"
        except GeometryOptimizationNotConvergedError as exc:
            errors[i] = f"did not converge: {exc}"
        except Exception as exc:
            errors[i] = f"{type(exc).__name__}: {exc}"

    typer.echo(f"Optimizing {len(nodes)} structure(s) with {type(engine).__name__}...")
    batch = getattr(engine, "compute_geometry_optimizations", None)
    batched = False
    if callable(batch) and len(nodes) > 1:
        try:
            try:
                trajs = batch(nodes, keywords=keywords)
            except TypeError:
                trajs = batch(nodes)
            if isinstance(trajs, (list, tuple)) and len(trajs) == len(nodes):
                for i, traj in enumerate(trajs):
                    if traj:
                        results[i] = traj[-1]
                    else:
                        errors[i] = "optimizer returned an empty trajectory"
                batched = True
        except Exception as exc:
            # One bad structure fails a whole batch: redo them one by one so
            # the failure is attributed and the rest still get optimized.
            typer.echo(f"Batch optimization failed ({type(exc).__name__}: {exc}); retrying one at a time.")
    if not batched:
        for i in range(len(nodes)):
            _single(i)
            typer.echo(f"  [{i + 1}/{len(nodes)}] {sources[i]}: {'ok' if results[i] is not None else errors[i]}")

    validations: list = [None] * len(nodes)
    if validate_minima_with_hessian:
        from mepd.elementarystep import validate_minimum_with_rescue

        for i, node in enumerate(results):
            if node is None:
                continue
            results[i], validations[i] = validate_minimum_with_rescue(
                node, engine, frequency_cutoff=hessian_minimum_frequency_cutoff,
                rescue_displacement=hessian_minima_rescue_displacement, label=sources[i],
            )
            typer.echo(f"  Hessian check {sources[i]}: {validations[i]['validation']}")

    summary = []
    for i, node in enumerate(results):
        rec = {"index": i, "source": sources[i], "converged": node is not None, "energy": None, "error": errors[i]}
        if validations[i] is not None:
            rec.update(validations[i])
        if node is not None:
            chain = Chain.model_validate({"nodes": [node], "parameters": ChainInputs()})
            chain.write_to_disk(output / f"opt_{i}.xyz")
            energy = getattr(node, "_cached_energy", None)
            rec["energy"] = float(energy) if energy is not None else None
        summary.append(rec)
    (output / "summary.json").write_text(json.dumps({"structures": summary}, indent=1))
    n_ok = sum(r["converged"] for r in summary)
    typer.echo(f"Optimized {n_ok}/{len(summary)} structure(s); results in {output}")
    if n_ok == 0:
        raise typer.Exit(code=1)


_DEFAULT_INPUTS_PATH_METHODS = ("NEB", "FNEB", "NEB-DLF", "GEOMETRIC-NEB", "GSM")


@app.command("init")
def make_default_inputs(
    output: Path = typer.Option(
        Path("mepd_inputs.toml"), "--output", "-o",
        help="Path to write a starter RunInputs TOML file to.",
    ),
    method: str = typer.Option(
        "NEB", "--method", "-m",
        help="Path-minimization method the starter file's [path_min_inputs] "
        f"section should default to: {', '.join(_DEFAULT_INPUTS_PATH_METHODS)} "
        "(case-insensitive; underscores/spaces normalize to '-', so 'geometric_neb' "
        "and 'neb-dlf'/'dlfind' also work). Each method's own defaults -- e.g. GSM's "
        "executable/nnodes/conv_tol vs NEB's tol/climbing-image settings -- are used, "
        "matching what `mepd run --path-min-method` would fall back on.",
    ),
) -> None:
    """Write a starter RunInputs TOML file reflecting the package defaults for
    --method's path minimizer, ready to hand-edit."""
    from mepd.inputs import _normalized_path_method

    normalized = _normalized_path_method(method)
    if normalized not in _DEFAULT_INPUTS_PATH_METHODS:
        raise typer.BadParameter(
            f"Unknown --method '{method}'. Known: {', '.join(_DEFAULT_INPUTS_PATH_METHODS)}."
        )
    run_inputs = RunInputs(path_min_method=normalized)
    run_inputs.save(output)
    typer.echo(f"Wrote default inputs ({normalized} path minimizer) to {output}")


@app.command("web")
def web(
    workspace: Path = typer.Argument(
        Path("mepd_workspace"),
        help="Workspace directory (structure library, reaction graph, profiles, job outputs). Created if missing.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind. Use 0.0.0.0 to serve other machines."),
    port: int = typer.Option(8765, "--port", help="Port to serve on."),
    max_jobs: int = typer.Option(
        2, "--max-jobs", help="How many mepd jobs may run at once; the rest wait in the queue."
    ),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open a browser tab."),
    auth: Optional[bool] = typer.Option(
        None, "--auth/--no-auth",
        help="Require a login token. On by default whenever --host is not localhost; turn it on "
        "explicitly when a proxy on this machine (e.g. `tailscale serve`) exposes the port.",
    ),
    new_token: bool = typer.Option(False, "--new-token", help="Rotate the access token (logs every device out)."),
    demo: bool = typer.Option(
        False, "--demo",
        help="Public demo mode: WORKSPACE is the demo root; anyone with the shared password gets private "
        "workspaces there, with read-only admin profiles (WORKSPACE/profiles), no filesystem access and "
        "size/time limits. Run it in the container from deploy/demo/.",
    ),
    demo_password: Optional[str] = typer.Option(
        None, "--demo-password", envvar="MEPD_DEMO_PASSWORD",
        help="Shared demo password (default: generated once, stored in WORKSPACE/.demo_password).",
    ),
) -> None:
    """Serve the mepd web interface over a workspace directory."""
    try:
        import uvicorn

        from mepd.web.app import create_app
    except ImportError as exc:
        typer.echo(f"mepd web needs the `web` extra (pip install 'mepd[web]'): {exc}")
        raise typer.Exit(code=1)

    local_only = host in ("127.0.0.1", "localhost", "::1")
    if demo:
        _serve_demo(workspace, host, port, max_jobs, demo_password)
        return
    if auth is None:
        auth = not local_only
    if not auth and not local_only:
        typer.echo("Refusing to serve beyond localhost without --auth: anyone who can reach the port "
                   "could run jobs as you and read your files.")
        raise typer.Exit(code=1)
    token = None
    if auth:
        from mepd.web.auth import load_or_create_token, token_path

        token = load_or_create_token(rotate=new_token)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    typer.echo(f"mepd web: workspace {workspace.resolve()} -> {url}")
    if token:
        typer.echo(f"Login required. Token (also in {token_path()}): {token}")
        typer.echo(f"One-click login link: {url}/login?token={token}")
        ts_name = _tailscale_dns_name()
        if ts_name:
            typer.echo(f"Over Tailscale, after `tailscale serve --bg {port}`: https://{ts_name}/login?token={token}")
        url = f"{url}/login?token={token}"
    if not no_open:
        import threading
        import webbrowser

        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(workspace, max_concurrent=max_jobs, auth_token=token), host=host, port=port,
                log_level="warning", proxy_headers=True, forwarded_allow_ips="127.0.0.1")


def _serve_demo(root: Path, host: str, port: int, max_jobs: int, password: Optional[str]) -> None:
    import os
    import secrets
    import subprocess
    import sys

    import uvicorn

    from mepd.web.app import create_app
    from mepd.web.demo import DemoPolicy

    root = root.resolve()
    (root / "profiles").mkdir(parents=True, exist_ok=True)
    if not list((root / "profiles").glob("*.toml")):
        # Starter profiles the admin can edit on disk (visitors cannot).
        for name, method in (("default", "NEB"), ("gsm", "GSM")):
            subprocess.run([sys.executable, "-m", "mepd.cli", "init", "--method", method,
                            "--output", str(root / "profiles" / f"{name}.toml")], check=False, capture_output=True)
    pw_file = root / ".demo_password"
    if not password:
        if pw_file.exists():
            password = pw_file.read_text().strip()
        else:
            password = secrets.token_urlsafe(9)
            pw_file.write_text(password + "\n")
            os.chmod(pw_file, 0o600)
    policy = DemoPolicy()
    typer.echo(f"mepd web DEMO: root {root} -> http://{host}:{port}")
    typer.echo(f"  password: {password}")
    typer.echo(f"  profiles (read-only for visitors): {', '.join(p.stem for p in sorted((root / 'profiles').glob('*.toml')))}")
    typer.echo(f"  limits: {policy.max_atoms} atoms/structure, {policy.max_active_jobs} active jobs/visitor, "
               f"{policy.global_concurrency} running overall, {policy.job_timeout_s / 60:.0f} min/job")
    uvicorn.run(create_app(root, max_concurrent=max_jobs, demo=policy, demo_password=password),
                host=host, port=port, log_level="warning", proxy_headers=True, forwarded_allow_ips="*")


def _tailscale_dns_name() -> Optional[str]:
    """This machine's MagicDNS name, if Tailscale is running (for the hint)."""
    import shutil
    import subprocess

    if not shutil.which("tailscale"):
        return None
    try:
        out = subprocess.run(["tailscale", "status", "--self", "--json"], capture_output=True, text=True, timeout=5)
        return json.loads(out.stdout)["Self"]["DNSName"].rstrip(".") or None
    except Exception:
        return None


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
