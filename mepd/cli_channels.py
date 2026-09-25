"""`mepd channels`: multi-conformer, multi-atom-mapping transition-state
discovery between two reaction endpoints. Split out of `mepd/cli.py` (which
still defines every other command) because this is by far the largest and
most actively-developed single command -- conformer-pool generation and
minimization, per-pair mechanism/atom-mapping expansion, MSMEP path search,
TS/IRC optimization, and route/channel classification (single-step
`channels/`, multistep `alternate-channels/`, `offtarget-exit-channels/`).

`channels` (the command function) is imported and registered onto `app` by
`mepd/cli.py` rather than decorated here, so this module never needs to
import anything from `mepd.cli` -- `mepd.cli` depends on this module, not
the other way around. Shared run/endpoint/TS-guess/MSMEP-pair-search
plumbing used by other commands too lives in `mepd.cli_common`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from mepd.atom_mapping_selection import METRICS as _ATOM_MAPPING_METRICS
from mepd.chain import Chain
from mepd.cli_common import (
    TsIrcResult,
    _check_endpoint_atom_mapping,
    _collect_ts_guess_tasks,
    _completed_tree_dirs,
    _connectivity_matches,
    _echo_run_inputs_summary,
    _fork_map,
    _geometry_optimizer_keywords,
    _load_structure_from_smiles_or_xyz,
    _minimize_endpoints,
    _open_run_inputs,
    _optimize_ts_and_irc,
    _run_msmep_pairs,
)
from mepd.inputs import NetworkInputs, RunInputs


def _expand_pairs_by_mechanism(
    structures: list,
    candidates: list[tuple[int, int]],
    atom_mapping: bool,
    run_inputs: "RunInputs",
    *,
    pairs_per_mechanism: int = 0,
    workers: int = 1,
    output: Optional[Path] = None,
) -> tuple[list, list[tuple[int, int]], dict]:
    """--atom-mapping for conformer pairs: turn every (reactant, product)
    conformer pair into one path search PER MECHANISM.

    SLAPMapper's equal-minimal-cost mappings can describe different
    chemistry -- for a Claisen, the [3,3] shift and a [1,3] shift tie,
    because bond orders aren't scored -- so a pair is not reduced to one
    mapping. Within each mechanism, the symmetry-equivalent relabelings
    (which of a CH2's hydrogens goes where) are all scored against THIS
    pair's geometry with --atom-mapping-metric and the best is kept, since
    a bad assignment forces e.g. a CH2 rotation onto the path; across
    mechanisms, nothing is thrown away (`select_per_mechanism`).

    `pairs_per_mechanism` > 0 then keeps, for each mechanism, only the pairs
    it scores best on, instead of every pair x every mechanism.

    Each kept (pair, mechanism) gets its own product entry appended to
    `structures` (the conformer pools on disk stay as sampled), and the
    table of what each path search is -- reactant/product conformer,
    mechanism, score -- goes to <output>/pair_mechanisms.json.

    Returns `(structures, candidates, summary)`; a no-op when `atom_mapping`
    is off."""
    if not atom_mapping or not candidates:
        return structures, candidates, {}

    from mepd.atom_mapping_selection import select_per_mechanism
    from mepd.nodes.node import StructureNode

    metric = run_inputs.atom_mapping_inputs.metric
    budget = run_inputs.atom_mapping_inputs.n_candidates
    typer.echo(
        f"--atom-mapping: finding every mechanism for {len(candidates)} conformer "
        f"pair(s) (best symmetry variant per mechanism by {metric})..."
    )

    def _one(pair):
        i, j = pair
        try:
            choices = select_per_mechanism(
                structures[i].structure, structures[j].structure, metric, run_inputs,
                max_variants_per_mechanism=budget,
            )
        except Exception as exc:
            return pair, None, f"{type(exc).__name__}: {exc}"
        return pair, [(c.key, c.score, c.n_variants, c.winner.end_structure, c.winner.atom_map is None)
                      for c in choices], None

    rows = []  # one per (pair, mechanism)
    n_failed = 0
    for (i, j), choices, error in _fork_map(_one, list(candidates), workers):
        if not choices:
            if error:
                typer.echo(f"  pair ({i}, {j}): mechanism enumeration failed ({error}); keeping it as is.")
                n_failed += 1
            rows.append({"i": i, "j": j, "key": "unmapped", "score": None,
                         "n_variants": 0, "structure": None})
            continue
        for key, score, n_variants, end_structure, is_identity in choices:
            rows.append({"i": i, "j": j, "key": key, "score": score, "n_variants": n_variants,
                         "structure": None if is_identity else end_structure})

    keys = sorted({r["key"] for r in rows})
    if pairs_per_mechanism > 0:
        kept = []
        for key in keys:
            mine = [r for r in rows if r["key"] == key]
            mine.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0))
            kept += mine[:pairs_per_mechanism]
        rows = kept

    new_structures = list(structures)
    new_candidates = []
    for r in rows:
        if r["structure"] is None:
            r["pair_j"] = r["j"]
        else:
            new_structures.append(StructureNode(structure=r["structure"]))
            r["pair_j"] = len(new_structures) - 1
        new_candidates.append((r["i"], r["pair_j"]))

    counts = {key: sum(1 for r in rows if r["key"] == key) for key in keys}
    typer.echo(f"--atom-mapping: {len(keys)} mechanism(s) across the pairs:")
    for key in keys:
        typer.echo(f"    {counts[key]:>4} path search(es)  {key}")
    typer.echo(
        f"--atom-mapping: {len(candidates)} conformer pair(s) -> {len(new_candidates)} "
        f"path search(es)"
        + (f" (best {pairs_per_mechanism} pair(s) per mechanism)" if pairs_per_mechanism > 0 else "")
        + (f"; {n_failed} enumeration(s) failed." if n_failed else ".")
    )

    if output is not None:
        import json

        table = [
            {"pair": f"pair_{r['i']}_{r['pair_j']}", "start_conformer": r["i"],
             "end_structure": r["j"], "mechanism": r["key"], "score": r["score"],
             "n_symmetry_variants": r["n_variants"]}
            for r in rows
        ]
        (output / "pair_mechanisms.json").write_text(json.dumps(table, indent=2) + "\n")

    summary = {"n_mechanisms": len(keys), "path_searches_per_mechanism": counts}
    return new_structures, new_candidates, summary


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


def _register_route_class(known: list, node) -> int:
    """Greedily assign `node` to an existing connectivity class in `known`
    (a list of representative nodes so far), registering a new class if none
    match. Mirrors `NetworkBuilder._get_ind_td`'s registration pattern."""
    for i, rep in enumerate(known):
        if _connectivity_matches(node, rep):
            return i
    known.append(node)
    return len(known) - 1


_TS_FINGERPRINT_TOL_BOHR = 0.1


def _same_ts(a, b, rmsd_cutoff: float, kcal_mol_cutoff: float) -> bool:
    """Whether two optimized TSs are the same saddle point.

    Unlike `is_identical` (index-wise Kabsch fit, plus a perceived-graph
    check), this must see through atom relabeling: every conformer pair gets
    its own --atom-mapping reindexing, so the same TS reached from two pairs
    routinely comes back with equivalent atoms (e.g. a methylene's two H)
    swapped, and mirror-image pairs return its mirror image. So: energies
    within `kcal_mol_cutoff`, then snap-RMSD (permutation-aware) to b or its
    mirror image under `rmsd_cutoff`. snap-RMSD perceives connectivity from
    geometry, which is ambiguous at a saddle point's half-formed bonds; when
    it can't, fall back to the sorted interatomic-distance list, which is
    itself invariant to permutation, rotation and reflection."""
    import numpy as np

    from mepd._qcinf_compat import snap_rmsd
    from mepd.conformers import mirror_image

    try:
        if abs(a.energy - b.energy) * 627.5 >= kcal_mol_cutoff:
            return False
    except Exception:
        pass
    try:
        return min(
            snap_rmsd(a.structure, b.structure),
            snap_rmsd(a.structure, mirror_image(b.structure)),
        ) < rmsd_cutoff
    except Exception:
        pass
    if list(a.structure.symbols) != list(b.structure.symbols):
        return False

    def _fingerprint(node):
        g = np.asarray(node.structure.geometry, dtype=float)
        d = np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1)
        return np.sort(d[np.triu_indices(len(g), 1)])

    return float(np.abs(_fingerprint(a) - _fingerprint(b)).max()) < _TS_FINGERPRINT_TOL_BOHR


def _cluster_by_ts_identity(candidates: list[tuple], run_inputs: RunInputs) -> list[list[tuple]]:
    """Greedily cluster `(ts_node, irc_chain, label)` candidates into classes
    of the same TS (`_same_ts`, with the cutoffs `NetworkBuilder` uses for
    "are these the same structure": node_rms_thre bohr, node_ene_thre
    kcal/mol)."""
    clusters: list[list[tuple]] = []
    for candidate in candidates:
        ts_node = candidate[0]
        for cluster in clusters:
            if _same_ts(
                ts_node, cluster[0][0],
                rmsd_cutoff=run_inputs.chain_inputs.node_rms_thre,
                kcal_mol_cutoff=run_inputs.chain_inputs.node_ene_thre,
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


_MAX_ALTERNATE_CHANNEL_STEPS = 6
_MAX_ALTERNATE_CHANNELS = 200


def _species_label(node) -> str:
    """Best-effort SMILES for a discovered minimum, to name it in the
    human-readable `path.txt`/`members.txt` sidecars.

    Prefers whatever identifiers the structure already carries, and
    otherwise derives one from the perceived bond graph: a structure
    reloaded from a bare xyz -- which is exactly what resuming a run gives
    you -- has no identifiers at all, and a route written out as
    "? -> ? -> ?" tells you nothing about the chemistry it found."""
    from mepd.NetworkBuilder import _stereochemical_smiles_key

    try:
        key = _stereochemical_smiles_key(node)
        if key:
            return key
    except Exception:
        pass
    try:
        smiles = getattr(getattr(node, "graph", None), "smiles", "")
        if smiles:
            return str(smiles)
    except Exception:
        pass
    return "?"


def _cluster_ts_energy(cluster: list) -> float:
    """Energy of a TS class, for ranking distinct classes of the same step."""
    try:
        return cluster[0][0].energy
    except Exception:
        return float("inf")


def _write_multistep_group(
    out_dir: Path,
    prefix: str,
    index: int,
    species_path: list,
    step_clusters: list,
    species_labels: list,
) -> None:
    """Write one multistep route -- an *alternate channel* -- as a `path.txt`
    naming the species it walks through plus one `step_<n>/` folder per
    elementary step, each holding that leg's TS/IRC in exactly the layout a
    single-step group uses.

    `step_clusters[n]` is every distinct TS class found for leg n. The
    lowest-barrier one defines the route and goes in `step_<n>/` itself; a
    leg found with genuinely different TSs keeps the rest alongside it in
    `step_<n>/alternate_ts_<m>/` rather than dropping them -- they are real
    mechanisms for that step, just not the cheapest one."""
    group_dir = out_dir / f"{prefix}_{index}"
    group_dir.mkdir(parents=True, exist_ok=True)

    arrow = "\n  -> ".join(species_labels[c] for c in species_path)
    lines = [
        f"{len(step_clusters)} step(s), {len(species_path) - 2} intermediate(s)",
        f"  {arrow}",
    ]
    for n, clusters in enumerate(step_clusters):
        ranked = sorted(clusters, key=_cluster_ts_energy)
        extra_info = (
            f"connects: {species_labels[species_path[n]]} "
            f"<-> {species_labels[species_path[n + 1]]}"
        )
        _write_classified_group(ranked[0], group_dir, "step", n, extra_info=extra_info)
        if len(ranked) > 1:
            lines.append(
                f"  step_{n}: {len(ranked) - 1} further distinct TS class(es) "
                f"for this step in step_{n}/alternate_ts_*/"
            )
        for m, alternate in enumerate(ranked[1:]):
            _write_classified_group(
                alternate, group_dir / f"step_{n}", "alternate_ts", m,
                extra_info=extra_info,
            )
    (group_dir / "path.txt").write_text("\n".join(lines) + "\n")


def _discover_channels(
    output: Path,
    start_node,
    end_node,
    run_inputs: RunInputs,
    charge: Optional[int],
    multiplicity: Optional[int],
    workers: int = 1,
) -> None:
    """TS-opt + IRC every leaf across every completed pair tree in `output`,
    then classify by walking the reaction graph those IRCs span.

    Every converged leaf contributes one elementary step: an edge between
    the connectivity classes of the two minima its IRC relaxes into. Leaves
    whose TS-opt or IRC failed to converge, or whose IRC collapsed into the
    same minimum on both sides, are dropped. Seeding that graph's species
    list with the requested --start/--end pair means the edges then sort
    into three buckets:

    * `channels/` -- a single elementary step straight from --start to
      --end.
    * `alternate-channels/` -- a *multistep* route from --start to --end: a
      simple path through one or more discovered intermediates, every leg
      of it an IRC-verified step. One folder per route, `step_<n>/` per leg.
    * `offtarget-exit-channels/` -- a step off some discovered minimum for
      which no onward sequence of steps ever reaches --end. A TS to an
      intermediate we never found a way forward from is an exit off the
      reaction path, not a route to the requested product.

    Within a bucket, candidates are further deduplicated into distinct TS
    classes -- this is what turns e.g. 4 conformer-pair runs into "2
    channels" when 2 of those runs converged to essentially the same TS.
    """
    import itertools

    import networkx as nx

    tree_charge = charge if charge is not None else 0
    tree_multiplicity = multiplicity if multiplicity is not None else 1

    tasks = _collect_ts_guess_tasks(output, charge, multiplicity)
    if not tasks:
        typer.echo("No TS guesses found across completed pairs; no channels to classify.")
        return

    ts_dir = output / "ts"
    ts_dir.mkdir(parents=True, exist_ok=True)

    # Species classes, seeded so that --start is class 0 and --end is class 1;
    # every IRC endpoint is then registered against the same list, which is
    # what lets a multi-leg route be recognized as reaching the real product.
    species: list = []
    start_cls = _register_route_class(species, start_node)
    end_cls = _register_route_class(species, end_node)
    degenerate_pair = start_cls == end_cls
    if degenerate_pair:
        typer.echo(
            "WARNING: --start and --end have identical connectivity; no "
            "channel can be defined for this pair, so every discovered step "
            "is reported as an off-target exit channel."
        )

    edge_candidates: dict = {}
    n_failed = 0

    # TS-opt + IRC every leaf not already on disk, `workers` at a time. Each
    # leaf writes only its own <label>*.xyz, and the classification loop
    # below reads everything back from disk, so it is identical either way;
    # leaves whose optimization failed are remembered so they aren't retried.
    pending = [
        (label, node) for label, node in tasks
        if _load_ts_and_irc_from_disk(ts_dir, label, tree_charge, tree_multiplicity) is None
    ]
    failed_labels: set = set()
    if workers > 1 and len(pending) > 1:
        typer.echo(
            f"Optimizing {len(pending)} TS guess(es) across "
            f"{min(workers, len(pending))} worker process(es)."
        )

        def _opt(task):
            label, node = task
            result = _optimize_ts_and_irc(node, run_inputs, ts_dir, run_irc=True, label=label)
            ok = result is not None and result.irc_chain is not None and len(result.irc_chain) >= 2
            return label, ok

        failed_labels = {label for label, ok in _fork_map(_opt, pending, workers) if not ok}

    for label, guess_node in tasks:
        if label in failed_labels:
            n_failed += 1
            continue
        result = _load_ts_and_irc_from_disk(ts_dir, label, tree_charge, tree_multiplicity)
        if result is None:
            result = _optimize_ts_and_irc(guess_node, run_inputs, ts_dir, run_irc=True, label=label)
        if result is None or result.irc_chain is None or len(result.irc_chain) < 2:
            n_failed += 1
            continue

        irc_first, irc_last = result.irc_chain[0], result.irc_chain[-1]
        i = _register_route_class(species, irc_first)
        j = _register_route_class(species, irc_last)
        if i == j:
            # IRC collapsed to the same minimum on both sides -- not a
            # genuine two-minima step.
            n_failed += 1
            continue
        edge_candidates.setdefault(frozenset((i, j)), []).append(
            (result.ts_node, result.irc_chain, label)
        )

    # One graph edge per pair of species; each edge carries the distinct TS
    # classes found for that step.
    edge_steps: dict = {
        key: _cluster_by_ts_identity(cands, run_inputs)
        for key, cands in edge_candidates.items()
    }
    species_labels = [_species_label(node) for node in species]

    graph = nx.Graph()
    graph.add_nodes_from(range(len(species)))
    graph.add_edges_from(tuple(sorted(key)) for key in edge_steps)

    # Every simple --start -> --end path; a 1-edge path is a direct channel,
    # anything longer is an alternate (multistep) channel.
    routes: list = []
    if not degenerate_pair and graph.has_node(start_cls) and graph.has_node(end_cls):
        routes = list(
            itertools.islice(
                nx.all_simple_paths(
                    graph, start_cls, end_cls, cutoff=_MAX_ALTERNATE_CHANNEL_STEPS
                ),
                _MAX_ALTERNATE_CHANNELS,
            )
        )
    routes.sort(key=len)

    on_route_edges: set = set()
    multistep_routes: list = []
    for path in routes:
        keys = [frozenset((path[n], path[n + 1])) for n in range(len(path) - 1)]
        on_route_edges.update(keys)
        if len(keys) > 1:
            multistep_routes.append((path, keys))

    direct_key = frozenset((start_cls, end_cls))
    channel_clusters = [] if degenerate_pair else edge_steps.get(direct_key, [])
    for k, cluster in enumerate(channel_clusters):
        _write_classified_group(cluster, output / "channels", "channel", k)

    alt_dir = output / "alternate-channels"
    for k, (path, keys) in enumerate(multistep_routes):
        _write_multistep_group(
            alt_dir, "alternate_channel", k, path,
            [edge_steps[key] for key in keys], species_labels,
        )

    offtarget_dir = output / "offtarget-exit-channels"
    n_offtarget = 0
    for key in edge_steps:
        if key in on_route_edges:
            continue
        i, j = sorted(key)
        for cluster in edge_steps[key]:
            _write_classified_group(
                cluster, offtarget_dir, "offtarget_exit_channel", n_offtarget,
                extra_info=f"connects: {species_labels[i]} <-> {species_labels[j]}",
            )
            n_offtarget += 1

    n_contributing = sum(len(c) for c in channel_clusters)
    typer.echo(
        f"{len(channel_clusters)} channel(s) found for the requested pair "
        f"({n_contributing} contributing run(s)), "
        f"{len(multistep_routes)} alternate channel(s), "
        f"{n_offtarget} off-target exit channel(s), {n_failed} failed."
    )


def channels(
    start: str = typer.Option(..., "--start", help="Path to the reactant-endpoint xyz file, or a SMILES string."),
    end: str = typer.Option(..., "--end", help="Path to the product-endpoint xyz file, or a SMILES string."),
    method: str = typer.Option(
        "conformers", "--method",
        help="Seed-generation strategy for perturbing the search to surface "
        "alternate TS channels between --start and --end. Currently only "
        "'conformers' (conformer sampling of each endpoint, see --backend) is "
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
        True, "--atom-mapping/--no-atom-mapping",
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
    backend: str = typer.Option(
        "rdkit", "--backend",
        help="Conformer-generation backend: 'rdkit' (ETKDG embedding + MMFF "
        "relaxation; cheap) or 'crest' (CREST iterative metadynamics; needs the "
        "`crest` binary, much more expensive, see --crest-*). Either way the "
        "candidates are deduplicated by snap-RMSD (--rmsd-cutoff), capped at "
        "--n-conformers, and re-minimized with the engine (--minimize-ends).",
    ),
    crest_method: str = typer.Option(
        "--gfn2", "--crest-method",
        help="--backend crest: CREST sampling-level flag, e.g. '--gfn2', "
        "'--gfnff', or '--gfn2//gfnff'.",
    ),
    crest_threads: int = typer.Option(
        1, "--crest-threads", help="--backend crest: CREST's -T thread count.",
    ),
    crest_ewin: float = typer.Option(
        6.0, "--crest-ewin",
        help="--backend crest: CREST's --ewin energy window (kcal/mol) for "
        "retained conformers.",
    ),
    crest_timeout: float = typer.Option(
        3600.0, "--crest-timeout",
        help="--backend crest: wall-clock limit (s) for one CREST call.",
    ),
    crest_nci: bool = typer.Option(
        True, "--crest-nci/--no-crest-nci",
        help="--backend crest: run CREST in NCI mode (an ellipsoidal wall that "
        "keeps a complex together) for endpoints made of more than one molecule.",
    ),
    complex_energy_tol: float = typer.Option(
        0.5, "--complex-energy-tol",
        help="For endpoints made of more than one molecule: after minimization, "
        "drop a conformer when every molecule's own conformation matches an "
        "already-kept one (snap-RMSD < --rmsd-cutoff, molecule by molecule) and "
        "the energies agree within this many kcal/mol. 0 disables.",
    ),
    n_conformers: int = typer.Option(
        50, "--n-conformers",
        help="Maximum number of distinct conformers to keep for EACH endpoint, "
        "after generation and RMSD-based deduplication run to completion "
        "uncapped. If dedup still leaves more than this, the pool is capped by "
        "farthest-point selection (maximize each new pick's distance to the "
        "nearest already-kept conformer) rather than truncation, so a capped "
        "pool still spans the conformational landscape broadly instead of "
        "clustering around the low-energy end. 0 = no cap: keep every distinct "
        "conformer the backend's own parameters let through (CREST's "
        "--crest-ewin; RDKit's --n-embed and --rdkit-ewin) -- some molecules "
        "are combinatorially flexible enough that this is thousands of pairs.",
    ),
    n_embed: int = typer.Option(
        0, "--n-embed",
        help="Number of raw conformer embeddings to attempt per endpoint before "
        "deduplication (rdkit backend only). Should comfortably exceed --n-conformers. "
        "0 = pick from rotatable-bond count (50 / 200 / 300 for <=7 / 8-12 / >12).",
    ),
    rdkit_torsion_prefs: str = typer.Option(
        "both", "--rdkit-torsion-prefs",
        help="--backend rdkit: embed with ETKDG's experimental torsion preferences "
        "('etkdg'), without them ('none'), or both pooled ('both', default). The "
        "preferences alone never sample e.g. s-cis dienes.",
    ),
    rdkit_ewin: Optional[float] = typer.Option(
        None, "--rdkit-ewin",
        help="--backend rdkit: keep only embeddings within this many kcal/mol "
        "(MMFF94) of the lowest -- the counterpart of --crest-ewin. Default: no window.",
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
        0, "--max-pairs",
        help="Hard cap on the number of reactant x product conformer pairs considered "
        "(before --atom-mapping expands them per mechanism). 0 (default) = no cap: "
        "selection is left to --pairs-per-mechanism.",
    ),
    conformers_only: bool = typer.Option(
        False, "--conformers-only",
        help="Stop after building, minimizing, deduplicating and atom-mapping the "
        "conformer pools (writes conformers/ and stats.json) -- i.e. before any "
        "path search. For sizing a run: how many pairs would it take, and what "
        "did conformer generation cost?",
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--no-parallel",
        help="Run each pair's recursive autosplitting (MSMEP) with branches "
        "evaluated in parallel, whenever a pair's path actually splits at an "
        "intermediate. On by default: for a single interactive query this is "
        "free extra parallelism on top of --workers, which alone leaves most "
        "cores idle once there are only a handful of pairs left to search.",
    ),
    parallel_workers: Optional[int] = typer.Option(
        None, "--parallel-workers",
        help="Concurrent workers for --parallel, per pair. Default: auto -- "
        "coordinated with --workers so workers x parallel_workers fills the "
        "machine (cpu_count // workers) rather than MSMEP's own context-free "
        "min(4, cpu count) default.",
    ),
    pairs_per_mechanism: int = typer.Option(
        3, "--pairs-per-mechanism",
        help="--atom-mapping: every conformer pair is mapped once per mechanism "
        "SLAPMapper allows (e.g. [3,3] and [1,3] shifts for a Claisen); for each "
        "mechanism, only the K pairs its geodesic score (--atom-mapping-metric) "
        "likes best get a path search. 0 = every pair x every mechanism (slow, "
        "mostly redundant). See docs/channels_candidates.md.",
    ),
    workers: int = typer.Option(
        0, "--workers", "-j",
        help="Run this many conformer pairs' MSMEP searches at once, and "
        "afterwards this many TS optimizations + IRCs at once, each in its own "
        "process. Stacks with --parallel (branches within one pair): together "
        "they're what determines whether a single query actually uses more "
        "than one core. 0 (default) = auto: size to the number of independent "
        "searches at each stage, capped at the machine's core count, so a "
        "single interactive query uses the whole machine without needing to "
        "be told to.",
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
        "results, classified channels/alternate-channels/offtarget-exit-"
        "channels, and the completed byproduct network into.",
    ),
) -> None:
    """Multi-channel TS discovery for a fixed --start/--end pair: generate
    alternate seed guesses (--method), run a recursive NEB/MSMEP for every
    seed, optimize a TS and IRC for every resulting leaf, then classify by
    walking the reaction graph those IRCs span.

    A leaf whose IRC reconnects the requested --start/--end pair in one step
    is a channel, and lands in <output>/channels/ (deduplicated into
    distinct TS classes, e.g. 4 conformer-pair runs converging to 2 real
    channels). A *sequence* of steps that gets from --start to --end through
    discovered intermediates is an alternate channel, and lands in
    <output>/alternate-channels/ as one folder per route with a step_<n>/
    per leg. A step that leaves some minimum but from which no onward
    sequence of discovered steps ever reaches --end is an off-target exit
    channel, and lands in <output>/offtarget-exit-channels/. Everything
    completed is also combined into one byproduct network at
    <output>/network.json."""
    if method != "conformers":
        raise typer.BadParameter(f"Unknown --method '{method}'. Known: 'conformers'.")
    if backend not in ("rdkit", "crest"):
        raise typer.BadParameter(f"Unknown --backend '{backend}'. Known: 'rdkit', 'crest'.")
    if n_conformers < 0:
        raise typer.BadParameter("--n-conformers must be a non-negative integer (0 = no cap).")
    if n_embed < 0:
        raise typer.BadParameter("--n-embed must be a non-negative integer (0 = auto).")
    if rmsd_cutoff <= 0:
        raise typer.BadParameter("--rmsd-cutoff must be a positive number.")
    if max_pairs < 0:
        raise typer.BadParameter("--max-pairs must be a non-negative integer (0 = no cap).")
    if rdkit_ewin is not None and rdkit_ewin <= 0:
        raise typer.BadParameter("--rdkit-ewin must be positive.")
    if atom_mapping_metric not in _ATOM_MAPPING_METRICS:
        raise typer.BadParameter(
            f"--atom-mapping-metric must be one of {_ATOM_MAPPING_METRICS}."
        )
    if atom_mapping_candidates <= 0:
        raise typer.BadParameter("--atom-mapping-candidates must be a positive integer.")
    if crest_threads <= 0:
        raise typer.BadParameter("--crest-threads must be a positive integer.")
    if workers < 0:
        raise typer.BadParameter("--workers must be non-negative (0 = auto).")
    if pairs_per_mechanism < 0:
        raise typer.BadParameter("--pairs-per-mechanism must be non-negative (0 = all pairs).")
    if crest_timeout <= 0:
        raise typer.BadParameter("--crest-timeout must be positive.")
    if rdkit_torsion_prefs not in ("both", "etkdg", "none"):
        raise typer.BadParameter("--rdkit-torsion-prefs must be 'both', 'etkdg' or 'none'.")
    if complex_energy_tol < 0:
        raise typer.BadParameter("--complex-energy-tol must be non-negative (0 disables).")

    import os
    import time

    from mepd.conformers import (
        ConformerInputs, CrestInputs, _subselect_conformers,
        merge_degenerate_complex_conformers, merge_mirror_images,
    )
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
        n_conformers=n_conformers or None,
        n_embed=n_embed or None,
        rdkit_ewin_kcal=rdkit_ewin,
        rdkit_torsion_prefs=rdkit_torsion_prefs,
        rmsd_cutoff=rmsd_cutoff,
        random_seed=random_seed,
        crest=CrestInputs(
            method=crest_method,
            threads=crest_threads,
            ewin_kcal=crest_ewin,
            timeout_s=crest_timeout,
            nci_for_complexes=crest_nci,
        ),
    )

    # Per-stage yield and wall time, rewritten after every stage so a run
    # that dies (or a --conformers-only run) still leaves what it measured.
    run_started = time.perf_counter()
    stats: dict = {"backend": backend, "workers": workers, "conformers": {}}
    output.mkdir(parents=True, exist_ok=True)

    def _write_stats() -> None:
        stats["total_seconds"] = round(time.perf_counter() - run_started, 3)
        (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    typer.echo(f"Generating seed pairs (--method {method})...")
    start_confs, end_confs = generate_seed_pairs(
        method, start_node, end_node, conformer_inputs=conformer_inputs,
        stats=stats["conformers"],
    )
    typer.echo(
        f"  -> {len(start_confs)} start-endpoint seed(s), "
        f"{len(end_confs)} end-endpoint seed(s)."
    )

    pools = {"start": start_confs, "end": end_confs}
    for label in ("start", "end"):
        side = stats["conformers"].setdefault(label, {})
        if minimize_ends:
            typer.echo(f"Minimizing {label}-endpoint conformers...")
            t0 = time.perf_counter()
            pools[label] = _minimize_conformer_pool(pools[label], label, run_inputs)
            side["minimize_seconds"] = round(time.perf_counter() - t0, 3)
            side["n_minimized"] = len(pools[label])
            # Distinct starting geometries routinely relax into the same QM
            # minimum; every duplicate left in would multiply the pair count
            # with runs that repeat each other, so dedup again at the level
            # the paths will actually be computed at.
            pools[label] = _subselect_conformers(pools[label], n_max=None, rmsd_cutoff=rmsd_cutoff)
            dropped = side["n_minimized"] - len(pools[label])
            if dropped:
                typer.echo(
                    f"  {dropped} {label} conformer(s) minimized into an "
                    f"already-kept minimum (snap-RMSD < {rmsd_cutoff}); dropped."
                )
            # A loosely bound complex's conformers differ mostly in how far
            # apart the molecules sit; compare molecule by molecule instead.
            before = len(pools[label])
            pools[label] = merge_degenerate_complex_conformers(
                pools[label], rmsd_cutoff, complex_energy_tol,
            )
            side["n_complex_degenerate_merged"] = before - len(pools[label])
            if side["n_complex_degenerate_merged"]:
                typer.echo(
                    f"  {side['n_complex_degenerate_merged']} {label} conformer(s) of the "
                    f"complex match a kept one fragment by fragment within "
                    f"{complex_energy_tol} kcal/mol; dropped."
                )

    # Mirror-image conformers of an achiral molecule give mirror-image paths,
    # so pairs built from them repeat each other -- but they can only be
    # merged in ONE of the two pools. With {R, R*} x {P, P*}, (R, P) mirrors
    # (R*, P*) while (R, P*) mirrors (R*, P): two genuinely different
    # combinations, and merging both pools would keep only (R, P) and lose
    # the other. Merging one pool drops exactly the redundant pairs; pick
    # whichever leaves fewer.
    merged = {label: merge_mirror_images(pools[label], rmsd_cutoff) for label in pools}
    n_if = {
        "start": len(merged["start"]) * len(pools["end"]),
        "end": len(pools["start"]) * len(merged["end"]),
    }
    mirror_side = min(n_if, key=n_if.get)
    n_mirrors = len(pools[mirror_side]) - len(merged[mirror_side])
    if n_mirrors:
        typer.echo(
            f"  {n_mirrors} {mirror_side} conformer(s) are mirror images of another "
            f"(achiral molecule); merged, {len(pools['start']) * len(pools['end'])} "
            f"-> {n_if[mirror_side]} pairs."
        )
        pools[mirror_side] = merged[mirror_side]
    for label in pools:
        stats["conformers"][label]["n_mirror_images_merged"] = n_mirrors if label == mirror_side else 0
        stats["conformers"][label]["n_final"] = len(pools[label])
    start_confs, end_confs = pools["start"], pools["end"]
    _write_stats()

    if not start_confs or not end_confs:
        typer.echo("No usable conformers for one or both endpoints; nothing to run.")
        raise typer.Exit(code=1)

    # Persisted as soon as the pools are final -- before pairing and
    # mapping, which can take a long time for big pools, so an interrupted
    # run still keeps its (expensive) conformers -- and so `mepd visualize
    # <output>` can show every conformer, not just those in searched pairs.
    from mepd.inputs import ChainInputs

    conformers_dir = output / "conformers"
    conformers_dir.mkdir(parents=True, exist_ok=True)
    Chain.model_validate(
        {"nodes": start_confs, "parameters": ChainInputs()}
    ).write_to_disk(conformers_dir / "start.xyz")
    Chain.model_validate(
        {"nodes": end_confs, "parameters": ChainInputs()}
    ).write_to_disk(conformers_dir / "end.xyz")

    structures = start_confs + end_confs
    n_start = len(start_confs)
    candidates = [
        (i, n_start + j) for i in range(n_start) for j in range(len(end_confs))
    ]
    stats["n_pairs_possible"] = len(candidates)

    if max_pairs and len(candidates) > max_pairs:
        typer.echo(
            f"{len(candidates)} candidate reactant x product conformer pairs found "
            f"({n_start} x {len(end_confs)}), capping at --max-pairs={max_pairs}."
        )
        candidates = candidates[:max_pairs]
    stats["n_pairs"] = len(candidates)

    # workers=0 / parallel_workers=None ("auto"): resolved fresh at each
    # stage from how many genuinely independent things there are to run --
    # a fixed number picked once up front would be wrong at BOTH ends: too
    # small for mapping (hundreds of independent (pair, mechanism) scores,
    # cheap and embarrassingly parallel under --atom-mapping-metric
    # endpoint-rmsd) and too large for path search (--pairs-per-mechanism
    # usually leaves only a handful of pairs, so naively handing all of them
    # `cpu_count` workers wastes the rest of the machine that --parallel
    # could otherwise be using per pair).
    cpu_count = os.cpu_count() or 1
    workers_auto = workers == 0
    parallel_workers_auto = parallel_workers is None
    if workers_auto:
        workers = max(1, min(len(candidates), cpu_count))

    # The endpoint-level --atom-mapping check above ran on the two input
    # structures, before any conformer existed; which mechanism a pair can
    # follow, and how its equivalent atoms are best labeled, depend on that
    # pair's own geometry. Map every pair, one path search per mechanism.
    t0 = time.perf_counter()
    structures, candidates, mechanism_summary = _expand_pairs_by_mechanism(
        structures, candidates, atom_mapping, run_inputs,
        pairs_per_mechanism=pairs_per_mechanism, workers=workers, output=output,
    )
    stats["pair_atom_mapping_seconds"] = round(time.perf_counter() - t0, 3)
    stats.update(mechanism_summary)
    stats["n_path_searches"] = len(candidates)
    _write_stats()

    if workers_auto:
        workers = max(1, min(len(candidates), cpu_count))
    if parallel and parallel_workers_auto:
        parallel_workers = max(1, cpu_count // workers)
    stats["workers"] = workers
    stats["parallel_workers"] = parallel_workers if parallel else None
    # Engines that can run a chain's images concurrently (n_parallel = 0 is
    # "auto") get what is left of the machine per branch.
    if getattr(run_inputs.engine, "n_parallel", None) == 0:
        run_inputs.engine.n_parallel = max(
            1, cpu_count // (workers * (parallel_workers if parallel else 1))
        )
    stats["engine_parallel"] = getattr(run_inputs.engine, "n_parallel", None)

    if conformers_only:
        typer.echo(
            f"--conformers-only: {len(start_confs)} x {len(end_confs)} conformers, "
            f"{len(candidates)} pair(s); stopping before path search."
        )
        return

    pairs_dir = output / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    _run_msmep_pairs(
        structures, candidates, pairs_dir, run_inputs,
        parallel=parallel, parallel_workers=parallel_workers, workers=workers,
    )
    stats["msmep_seconds"] = round(time.perf_counter() - t0, 3)
    _write_stats()

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

    t0 = time.perf_counter()
    _discover_channels(
        output, start_node, end_node, run_inputs, charge, multiplicity, workers=workers,
    )
    stats["ts_discovery_seconds"] = round(time.perf_counter() - t0, 3)
    _write_stats()
