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
from mepd.atom_mapping_metrics import HELP as _MAPPING_HELP, OFFERED as _OFFERED_METRICS
from mepd.chain import Chain
from mepd.nodes.nodehelpers import _is_connectivity_identical
from mepd.cli_common import (
    TsIrcResult,
    _check_endpoint_atom_mapping,
    _collect_ts_guess_tasks,
    _completed_tree_dirs,
    _connectivity_matches,
    _echo_run_inputs_summary,
    _fork_map,
    _geometry_optimizer_keywords,
    _check_endpoint_options,
    _load_structure_from_smiles_or_xyz,
    _reaction_endpoints,
    _minimize_endpoints,
    _open_run_inputs,
    _optimize_ts_and_irc,
    _run_msmep_pairs,
)
from mepd.inputs import NetworkInputs, RunInputs


def _describe_mechanism(key: str) -> str:
    """'break C27-C29,C28-O20 | form C27-C28' -> 'break C27-C29 and C28-O20, form C27-C28';
    symmetry-equivalent bonds (the same name) are counted: 'form 2 x C10-C12'."""
    parts = []
    for side in key.split("|"):
        verb, _, bonds = side.strip().partition(" ")
        counts: dict[str, int] = {}
        for b in bonds.split(","):
            if b and b != "none":
                counts[b] = counts.get(b, 0) + 1
        names = [f"{k} x {b}" if k > 1 else b for b, k in counts.items()]
        if names:
            parts.append(f"{verb} " + (" and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]))
    return ", ".join(parts) or "no bond changes"


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

    from mepd.atom_mapping_selection import near_lowest, rmsd_window, select_per_mechanism
    from mepd.nodes.node import StructureNode

    metric = run_inputs.atom_mapping_inputs.metric
    budget = run_inputs.atom_mapping_inputs.n_candidates
    # Filtered GI path: every pair by endpoint RMSD first (microseconds each),
    # GI paths only for the pairs near each mechanism's lowest RMSD. Snap +
    # GI + xtb: every pair by snap's pick first, GI + xtb only for the pairs kept.
    filtered = metric == "rmsd-geodesic"
    first = "endpoint-rmsd" if filtered else "snap" if metric == "snap-gi-xtb" else metric
    typer.echo(
        f"Finding the mechanisms of the {len(candidates)} pair(s): every way the reactant's atoms can "
        f"become the product's, grouped by which bonds break and form (per mechanism, the atom "
        f"numbering that fits best by {first} is used)..."
    )

    def _one(pair, metric=first, only_keys=None):
        i, j = pair
        try:
            choices = select_per_mechanism(
                structures[i].structure, structures[j].structure, metric, run_inputs,
                max_variants_per_mechanism=budget, **({"only_keys": only_keys} if only_keys is not None else {}),
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
    if filtered:
        rows = _gi_near_lowest_rmsd(rows, keys, _one, workers, rmsd_window(run_inputs),
                                    max(1, pairs_per_mechanism), near_lowest)
    if pairs_per_mechanism > 0:
        kept = []
        for key in keys:
            mine = [r for r in rows if r["key"] == key]
            mine.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0))
            kept += mine[:pairs_per_mechanism]
        rows = kept
    if metric == "snap-gi-xtb":
        todo = [r for r in rows if r["key"] != "unmapped"]
        typer.echo(f"Choosing each kept pair's numbering among snap's picks by GI + xtb ({len(todo)} pair(s))...")
        for r, (_, choices, _) in zip(todo, _fork_map(lambda r: _one((r["i"], r["j"]), "snap-gi-xtb", {r["key"]}),
                                                     todo, workers)):
            for key, score, _, end_structure, is_identity in choices or []:
                if key == r["key"]:
                    r["structure"] = None if is_identity else end_structure

    # A rerun into the same folder (e.g. sampling more pairs per mechanism)
    # must give every (reactant, product, mechanism) its earlier label: a
    # finished pair is found -- and skipped -- by its folder name, and a label
    # re-used for a different pair would attach its results to the wrong one.
    # Earlier rows keep their product slots; new rows go after them.
    prior = {}
    if output is not None and (output / "pair_mechanisms.json").exists():
        import json as _json

        try:
            for row in _json.loads((output / "pair_mechanisms.json").read_text()):
                pj = int(str(row["pair"]).rsplit("_", 1)[1])
                prior[(int(row["start_conformer"]), int(row["end_structure"]), row["mechanism"])] = pj
        except Exception as exc:
            typer.echo(f"--atom-mapping: could not read the earlier pair table ({exc}); numbering afresh.")
            prior = {}
    new_structures = list(structures)
    slots = sorted(pj for pj in prior.values() if pj >= len(structures))
    if slots:
        # Hold the earlier product slots (a placeholder until filled below).
        new_structures += [structures[0]] * (slots[-1] - len(structures) + 1)
    new_candidates = []
    for r in rows:
        earlier = prior.get((r["i"], r["j"], r["key"]))
        if r["structure"] is None:
            r["pair_j"] = r["j"]
        elif earlier is not None and earlier >= len(structures):
            new_structures[earlier] = StructureNode(structure=r["structure"])
            r["pair_j"] = earlier
        else:
            new_structures.append(StructureNode(structure=r["structure"]))
            r["pair_j"] = len(new_structures) - 1
        new_candidates.append((r["i"], r["pair_j"]))
    if prior:
        reused = sum(1 for r in rows if (r["i"], r["j"], r["key"]) in prior)
        typer.echo(f"--atom-mapping: {reused} path search(es) keep their earlier labels; "
                   f"{len(rows) - reused} are new.")

    counts = {key: sum(1 for r in rows if r["key"] == key) for key in keys}
    typer.echo(f"{len(keys)} mechanism(s) (atoms are named by element and symmetry class, not by number):")
    for k, key in enumerate(keys, start=1):
        typer.echo(f"  {k}. {_describe_mechanism(key)}: {counts[key]} path search(es)")
    typer.echo(
        f"Path searches: {len(new_candidates)} of the {len(candidates)} pair(s)"
        + (f" -- the {pairs_per_mechanism} best-matching per mechanism (--pairs-per-mechanism)"
           if pairs_per_mechanism > 0 else "")
        + (f"; mapping failed for {n_failed} pair(s)." if n_failed else ".")
    )

    if output is not None:
        import json

        table = [
            {"pair": f"pair_{r['i']}_{r['pair_j']}", "start_conformer": r["i"],
             "end_structure": r["j"], "mechanism": r["key"], "score": r["score"],
             "n_symmetry_variants": r["n_variants"], **({"endpoint_rmsd": r["rmsd"]} if "rmsd" in r else {})}
            for r in rows
        ]
        # Earlier rows this run did not keep (e.g. a lower cap) stay listed, so their labels stay reserved.
        kept_keys = {(r["i"], r["j"], r["key"]) for r in rows}
        for (i, j, key), pj in prior.items():
            if (i, j, key) not in kept_keys:
                table.append({"pair": f"pair_{i}_{pj}", "start_conformer": i, "end_structure": j,
                              "mechanism": key, "score": None, "n_symmetry_variants": 0, "not_in_this_run": True})
        (output / "pair_mechanisms.json").write_text(json.dumps(table, indent=2) + "\n")

    summary = {"n_mechanisms": len(keys), "path_searches_per_mechanism": counts}
    return new_structures, new_candidates, summary


def _gi_near_lowest_rmsd(rows: list[dict], keys: list, score_pair, workers: int, window: float, keep: int,
                         near_lowest) -> list[dict]:
    """--atom-mapping-metric rmsd-geodesic, second stage: per mechanism, the
    (pair, mechanism) rows within `window` standard deviations of the lowest
    endpoint RMSD (at least `keep` of them) get GI paths -- which choose
    their symmetry variant and rank them -- and the rest are dropped.
    On the KAIST direct-only sweep a 1-sigma window held 6% of the pairs and
    56% of what GI paths over every pair would have picked; 2 sigma, 21% and
    82%; 3 sigma, 51% and 96% (see docs/channels_candidates.md)."""
    unmapped = [r for r in rows if r["score"] is None]
    chosen: dict = {}
    n_scored = 0
    for key in keys:
        mine = [r for r in rows if r["key"] == key and r["score"] is not None]
        n_scored += len(mine)
        for k in near_lowest([r["score"] for r in mine], window, keep=keep):
            chosen.setdefault((mine[k]["i"], mine[k]["j"]), {})[key] = mine[k]
    n_chosen = sum(len(v) for v in chosen.values())
    typer.echo(f"Filtered GI path: {n_chosen} of {n_scored} (pair, mechanism) combination(s) are within "
               f"{window:g} standard deviation(s) of their mechanism's lowest endpoint RMSD; computing GI "
               f"paths for those (--atom-mapping-rmsd-window to widen).")
    out = list(unmapped)
    tasks = [(pair, set(by_key)) for pair, by_key in chosen.items()]
    results = _fork_map(lambda t: score_pair(t[0], "rmsd-geodesic", t[1]), tasks, workers)
    for (pair, by_key), (_, choices, error) in zip(chosen.items(), results):
        got = {key: (score, nv, st, is_id) for key, score, nv, st, is_id in (choices or [])}
        for key, row in by_key.items():
            if key in got:
                score, nv, st, is_id = got[key]
                out.append({**row, "rmsd": row["score"], "score": score, "n_variants": nv,
                            "structure": None if is_id else st})
            else:
                # The GI path could not be computed: keep the pair, ranked last.
                out.append({**row, "rmsd": row["score"], "score": None})
                if error:
                    typer.echo(f"  pair {pair}: GI path failed ({error}); ranked last.")
    return out


def _load_conformer_pool(fp: Path, endpoint, label: str, charge: int, multiplicity: int) -> list:
    """A saved final conformer pool (--start-pool / --end-pool) for
    `endpoint`, with its energies. Only conformers whose bonds match the
    endpoint's atom for atom are kept: a pool saved when this molecule's atoms
    were numbered differently (e.g. it was the other end of another pair, so
    the atom-mapping check reordered it) would pair the wrong atoms."""
    from mepd.inputs import ChainInputs

    if not fp.exists():
        raise typer.BadParameter(f"--{label}-pool {fp} does not exist.")
    pool = list(Chain.from_xyz(fp, ChainInputs(), charge=charge, spinmult=multiplicity).nodes)

    def bonds(node):
        return {frozenset(e) for e in node.graph.edges()}

    want_symbols, want_bonds = list(endpoint.symbols), bonds(endpoint)
    kept = [n for n in pool if list(n.symbols) == want_symbols and bonds(n) == want_bonds]
    if len(kept) < len(pool):
        typer.echo(f"  {len(pool) - len(kept)} of {len(pool)} saved {label} conformer(s) are numbered or bonded "
                   f"differently from this run's {label} endpoint; not used.")
    return kept


def _build_partners(sources: list, target, run_inputs: RunInputs) -> tuple[list, list]:
    """--pair-from: for each conformer in `sources`, the other endpoint built
    in its frame -- every atom where it is in that conformer, the target's
    bonds pulled to length (network_expansion.embed_product), then minimized
    at the profile's level. A partner whose bonds change on minimization is
    dropped with its source. `target` (same atom order as the sources) gives
    the bonds, charge and spin. Returns (kept sources, their partners)."""
    import numpy as np
    from qcconst.constants import ANGSTROM_TO_BOHR

    from mepd.discovery.network_expansion import embed_product
    from mepd.nodes.node import StructureNode

    want = {tuple(sorted((int(u), int(v)))) for u, v in target.graph.edges()}
    kept, guesses = [], []
    for node in sources:
        try:
            xyz = embed_product(list(node.structure.symbols),
                                np.asarray(node.structure.geometry) / ANGSTROM_TO_BOHR, want)
        except Exception:
            continue
        kept.append(node)
        guesses.append(StructureNode(structure=target.structure.model_copy(
            update={"geometry": np.asarray(xyz) * ANGSTROM_TO_BOHR})))
    keywords = _geometry_optimizer_keywords(run_inputs)
    batch = getattr(run_inputs.engine, "compute_geometry_optimizations", None)
    if callable(batch):
        try:
            trajectories = batch(guesses, keywords=keywords)
        except TypeError:
            trajectories = batch(guesses)
    else:
        trajectories = [run_inputs.engine.compute_geometry_optimization(g, keywords=keywords) for g in guesses]
    pairs = []
    for src, traj in zip(kept, trajectories):
        if not traj:
            continue
        built = StructureNode(structure=traj[-1].structure)
        if {tuple(sorted((int(u), int(v)))) for u, v in built.graph.edges()} == want:
            pairs.append((src, built))
    return [s for s, _ in pairs], [b for _, b in pairs]


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
            # As in `_run_msmep_pairs`: `pending` was filtered against the disk
            # before the first attempt, and `_fork_map` may re-run it serially
            # after a worker died.
            result = _load_ts_and_irc_from_disk(ts_dir, label, tree_charge, tree_multiplicity)
            if result is None:
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
    # A step that reaches a stereo variant of --start/--end (same bonds,
    # different stereochemistry) is not what was queried: say so, rather
    # than leave it to be read as an off-target exit or a missing channel.
    mismatches = []
    for key in edge_steps:
        for k in key:
            if k in (start_cls, end_cls):
                continue
            for q, name in ((start_node, "--start"), (end_node, "--end")):
                if _is_connectivity_identical(species[k], q, verbose=False, collect_comparison=False,
                                              disregard_stereochem=True):
                    mismatches.append(f"a step reaches {species_labels[k]}, a stereo variant of {name} "
                                      f"({_species_label(q)}), not {name} itself")
    mismatch_warnings = list(dict.fromkeys(mismatches))
    for msg in mismatch_warnings:
        typer.echo(f"WARNING: result does not match the queried endpoints: {msg}.")

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

    # Off-target results: under --direct-only every leg that ran was meant
    # to join --start and --end, so each step that doesn't is a mismatch;
    # in any mode, a run that found nothing but off-target steps is one.
    offtarget_keys = [key for key in edge_steps if key not in on_route_edges]
    if offtarget_keys and (getattr(run_inputs.path_min_inputs, "direct_only", False)
                           or not (channel_clusters or multistep_routes)):
        for key in offtarget_keys:
            i, j = sorted(key)
            mismatch_warnings.append(
                f"an IRC-verified step connects {species_labels[i]} <-> {species_labels[j]}, "
                f"not the queried {species_labels[start_cls]} <-> {species_labels[end_cls]}")
            typer.echo(f"WARNING: result does not match the queried endpoints: {mismatch_warnings[-1]}.")

    n_contributing = sum(len(c) for c in channel_clusters)
    typer.echo(
        f"{len(channel_clusters)} channel(s) found for the requested pair "
        f"({n_contributing} contributing run(s)), "
        f"{len(multistep_routes)} alternate channel(s), "
        f"{n_offtarget} off-target exit channel(s), {n_failed} failed."
    )
    return mismatch_warnings


def channels(
    start: Optional[str] = typer.Option(
        None, "--start", help="Path to the reactant-endpoint xyz file, or a SMILES string (or use --reaction)."),
    end: Optional[str] = typer.Option(
        None, "--end", help="Path to the product-endpoint xyz file, or a SMILES string (or use --reaction)."),
    reaction: Optional[str] = typer.Option(
        None, "--reaction",
        help="Both endpoints as one reaction SMILES, 'reactants>>products' (agents, as in "
        "'reactants>agents>products', are ignored). Atom map numbers ([CH3:1]...) are used when every "
        "heavy atom has one; otherwise SLAPMapper maps the reaction. Instead of --start/--end.",
    ),
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
    explore_mechanisms: int = typer.Option(
        0, "--explore-mechanisms",
        help="Also consider up to this many mechanisms beyond SLAPMapper's minimal-edit ones: relays and "
        "exchanges through other molecules (catalyst, solvent), found by swapping same-element atoms or "
        "groups with at most +2 bond edits per swap, with no rule about which atoms may move. Each is "
        "scored like SLAPMapper's (snap + GI + xtb); ones in which a molecule takes part and is "
        "regenerated are labelled [catalytic: ...]. 0 = off.",
    ),
    relax_mechanisms: Optional[int] = typer.Option(
        None, "--relax-mechanisms",
        help="snap-gi-xtb: rank mappings by relaxed path energy. The N best by raw xtb peak (plus SLAPMapper's own "
        "minimal mapping, always) have their interpolated paths relaxed except for the bonds that break or form, "
        "and the lowest relaxed barrier wins; a mapping with more bond changes must beat SLAPMapper's by more than "
        "--relax-margin. Default: the profile's atom_mapping_inputs.relax_top (4); 0 = raw peaks only.",
    ),
    relax_margin: Optional[float] = typer.Option(
        None, "--relax-margin",
        help="kcal/mol by which a mapping with more bond changes than SLAPMapper's minimal one must beat its relaxed "
        "barrier (else SLAPMapper's is kept). Default: the profile's atom_mapping_inputs.relax_margin (5).",
    ),
    atom_mapping_metric: str = typer.Option(
        "snap-gi-xtb", "--atom-mapping-metric",
        help="--atom-mapping: " + _MAPPING_HELP + " One of: " + ", ".join(_OFFERED_METRICS) + ".",
    ),
    atom_mapping_rmsd_window: Optional[float] = typer.Option(
        None, "--atom-mapping-rmsd-window",
        help="rmsd-geodesic: compute GI paths for the conformer pairs within this many standard deviations "
        "of each mechanism's lowest endpoint RMSD. Default: the profile's atom_mapping_inputs.rmsd_window "
        "(1). Larger keeps more of what GI paths over every pair would pick, at more cost."),
    atom_mapping_gi_variants: Optional[int] = typer.Option(
        None, "--atom-mapping-gi-variants",
        help="rmsd-geodesic: GI paths for this many lowest-endpoint-RMSD symmetry variants per pair and "
        "mechanism (0 = all). Default: atom_mapping_inputs.gi_variant_cap (20)."),
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
    pair_from: str = typer.Option(
        "both", "--pair-from",
        help="both: sample both endpoints' conformers and pair every reactant conformer with every product "
        "conformer. start: sample only the reactant's conformers and build each one's product in its frame "
        "(each atom where it is, the product's bonds pulled to length, minimized), so the product geometries "
        "are sampled through the reactant's and only what reacts moves; each conformer is searched with its "
        "own product, under the endpoint-level atom mapping. end: the same from the product side.",
    ),
    rmsd_cutoff: float = typer.Option(
        0.1, "--rmsd-cutoff",
        help="Minimum pairwise RMSD (bohr) for two conformers of the same endpoint "
        "to count as distinct.",
    ),
    random_seed: int = typer.Option(0, "--random-seed", help="Random seed for conformer embedding."),
    start_pool: Optional[Path] = typer.Option(
        None, "--start-pool",
        help="Use this multi-frame xyz (with its .energies) as the start endpoint's final conformer pool "
        "instead of sampling one: e.g. conformers/start_pool.xyz of an earlier run with the same "
        "sampler settings. With --minimize-ends the pool must already be minimized at this run's "
        "level of theory; it is used as is (no sampling, no minimization).",
    ),
    end_pool: Optional[Path] = typer.Option(
        None, "--end-pool", help="As --start-pool, for the end endpoint.",
    ),
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
    search_budget: float = typer.Option(
        0.0, "--search-budget",
        help="Wall-clock seconds (counted from the start of the command) after "
        "which no new path-search work starts: pairs not yet begun are skipped "
        "and a search that finishes is not split further (it is kept as a "
        "'time_budget' leaf with its own path, so every tree stays continuous). "
        "TS optimization, IRC and channel classification then run on whatever "
        "was found, so a slow reaction still yields its channels instead of "
        "being killed with nothing. Searches already queued at the deadline "
        "still run once. 0 (default) = no budget.",
    ),
    direct_only: Optional[bool] = typer.Option(
        None, "--direct-only/--allow-multistep",
        help="--direct-only: focus compute on the queried pair. When a path search splits, only the pieces "
        "with at least one end at a queried species (the start or end, in any conformer or stereo variant) "
        "are run; a leg between two other species is not. E.g. A->B splitting into A->C, C->D, D->A', A'->B "
        "runs A->C, D->A' and A'->B, not C->D. Default: the profile's path_min_inputs.direct_only, else "
        "every piece is followed.",
    ),
    validate_minima_with_hessian: bool = typer.Option(
        True, "--validate-minima-with-hessian/--no-validate-minima-with-hessian", "-H/-noH",
        help="When a minima-based autosplit is proposed during each pair's MSMEP, "
        "compute Hessians for optimized split candidates and reject candidates "
        "with significant imaginary modes. On by default -- this is a "
        "correctness check, not a convenience.",
    ),
    hessian_minimum_frequency_cutoff: float = typer.Option(
        -20.0, "--hessian-minimum-frequency-cutoff",
        help="Minimum allowed frequency (cm^-1) for --validate-minima-with-hessian.",
    ),
    hessian_minima_rescue_displacement: float = typer.Option(
        0.3, "--hessian-minima-rescue-displacement",
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
    reaction = reaction if isinstance(reaction, str) else None   # called as a plain function
    _check_endpoint_options(start, end, reaction)
    if method != "conformers":
        raise typer.BadParameter(f"Unknown --method '{method}'. Known: 'conformers'.")
    if backend not in ("rdkit", "crest"):
        raise typer.BadParameter(f"Unknown --backend '{backend}'. Known: 'rdkit', 'crest'.")
    if n_conformers < 0:
        raise typer.BadParameter("--n-conformers must be a non-negative integer (0 = no cap).")
    if n_embed < 0:
        raise typer.BadParameter("--n-embed must be a non-negative integer (0 = auto).")
    if not isinstance(pair_from, str):   # typer's OptionInfo when called directly
        pair_from = "both"
    if pair_from not in ("both", "start", "end"):
        raise typer.BadParameter("--pair-from must be both, start or end.")
    if rmsd_cutoff <= 0:
        raise typer.BadParameter("--rmsd-cutoff must be a positive number.")
    if max_pairs < 0:
        raise typer.BadParameter("--max-pairs must be a non-negative integer (0 = no cap).")
    if rdkit_ewin is not None and rdkit_ewin <= 0:
        raise typer.BadParameter("--rdkit-ewin must be positive.")
    if atom_mapping_metric not in _ATOM_MAPPING_METRICS:
        raise typer.BadParameter(
            f"--atom-mapping-metric must be one of {', '.join(_OFFERED_METRICS)}."
        )
    if not isinstance(atom_mapping_rmsd_window, (int, float)):
        atom_mapping_rmsd_window = None   # not given (e.g. called from Python): the profile's value
    if atom_mapping_rmsd_window is not None and atom_mapping_rmsd_window < 0:
        raise typer.BadParameter("--atom-mapping-rmsd-window must be 0 or more (standard deviations).")
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
    if isinstance(direct_only, bool):
        run_inputs.path_min_inputs.direct_only = direct_only
    run_inputs.path_min_inputs.validate_minima_with_hessian = validate_minima_with_hessian
    run_inputs.path_min_inputs.hessian_minimum_frequency_cutoff = hessian_minimum_frequency_cutoff
    run_inputs.path_min_inputs.hessian_minima_rescue_displacement = hessian_minima_rescue_displacement
    run_inputs.atom_mapping_inputs.n_candidates = atom_mapping_candidates
    run_inputs.atom_mapping_inputs.metric = atom_mapping_metric
    if atom_mapping_rmsd_window is not None:
        run_inputs.atom_mapping_inputs.rmsd_window = atom_mapping_rmsd_window
    if isinstance(atom_mapping_gi_variants, int):
        run_inputs.atom_mapping_inputs.gi_variant_cap = max(0, atom_mapping_gi_variants)
    run_inputs.atom_mapping_inputs.veto_margin = atom_mapping_veto_margin
    run_inputs.atom_mapping_inputs.recheck_on_split = atom_mapping_recheck_splits
    run_inputs.atom_mapping_inputs.explore_mechanisms = max(0, explore_mechanisms) if isinstance(explore_mechanisms, int) else 0
    if isinstance(relax_mechanisms, int):
        run_inputs.atom_mapping_inputs.relax_top = max(0, relax_mechanisms)
    if isinstance(relax_margin, (int, float)):
        run_inputs.atom_mapping_inputs.relax_margin = max(0.0, float(relax_margin))
    _echo_run_inputs_summary(run_inputs)

    if reaction is not None:
        start_structure, end_structure = _reaction_endpoints(reaction, charge, multiplicity)
        start_is_smiles = end_is_smiles = True
    else:
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
    search_budget = float(search_budget) if isinstance(search_budget, (int, float)) else 0.0
    if search_budget < 0:
        raise typer.BadParameter("--search-budget must be >= 0.")
    search_deadline = time.time() + search_budget if search_budget > 0 else None
    stats: dict = {"backend": backend, "workers": workers, "conformers": {}}
    output.mkdir(parents=True, exist_ok=True)

    def _write_stats() -> None:
        stats["total_seconds"] = round(time.perf_counter() - run_started, 3)
        (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # Saved final pools of an earlier run (same sampler settings and level of
    # theory; the caller decides): used as is, not re-sampled or re-minimized.
    endpoints = {"start": start_node, "end": end_node}
    reused: dict = {}
    for label, fp in (("start", start_pool), ("end", end_pool)):
        if not isinstance(fp, (str, Path)):   # unset (None, or typer's default when called directly)
            continue
        node = endpoints[label]
        pool = _load_conformer_pool(Path(fp), node, label, node.structure.charge, node.structure.multiplicity)
        if pool:
            reused[label] = pool
            stats["conformers"][label] = {"reused_from": str(fp), "n_reused": len(pool)}
            typer.echo(f"Using {len(pool)} saved {label} conformer(s) from {fp} (not re-sampled or re-minimized).")
        else:
            typer.echo(f"No usable saved {label} conformers in {fp}; sampling them instead.")
    if len(reused) == 2:
        start_confs, end_confs = reused["start"], reused["end"]
    else:
        typer.echo(f"Sampling conformers ({conformer_inputs.backend})...")
        if not reused:
            start_confs, end_confs = generate_seed_pairs(
                method, start_node, end_node, conformer_inputs=conformer_inputs,
                stats=stats["conformers"],
            )
        else:
            from mepd.conformers import generate_conformers

            (label,) = {"start", "end"} - set(reused)
            sampled = generate_conformers(endpoints[label], conformer_inputs,
                                          stats["conformers"].setdefault(label, {}))
            start_confs = reused.get("start", sampled)
            end_confs = reused.get("end", sampled)
        typer.echo(f"  {len(start_confs)} reactant and {len(end_confs)} product conformer(s).")

    pools = {"start": start_confs, "end": end_confs}
    for label in ("start", "end"):
        side = stats["conformers"].setdefault(label, {})
        if minimize_ends and label not in reused and pair_from in ("both", label):
            side_name = "reactant" if label == "start" else "product"
            typer.echo(f"Minimizing the {len(pools[label])} {side_name} conformer(s) at the profile's "
                       "level of theory...")
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
                    f"  Dropped {dropped}: relaxed into the same minimum as a kept one "
                    f"(RMSD < {rmsd_cutoff} bohr)."
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
                    f"  Dropped {side['n_complex_degenerate_merged']}: the same molecules as a kept one, "
                    f"only placed differently (each molecule matches, energy within "
                    f"{complex_energy_tol} kcal/mol)."
                )
            typer.echo(f"  Kept {len(pools[label])}.")

    # Each side's final pool before mirror images are merged (a merge drops
    # conformers from one side only, depending on the other side): what a
    # later run with the same settings can take as its --start/--end-pool.
    from mepd.inputs import ChainInputs as _ChainInputs

    (output / "conformers").mkdir(parents=True, exist_ok=True)
    for label in ("start", "end"):
        if pools[label] and pair_from in ("both", label):   # a built side is never a sampled pool
            Chain.model_validate({"nodes": pools[label], "parameters": _ChainInputs()}).write_to_disk(
                output / "conformers" / f"{label}_pool.xyz")

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
            f"Merged {n_mirrors} {'reactant' if mirror_side == 'start' else 'product'} conformer(s) that are "
            f"mirror images of kept ones (the molecule is achiral, so their paths would be mirror images too): "
            f"{len(pools['start']) * len(pools['end'])} -> {n_if[mirror_side]} pairs."
        )
        pools[mirror_side] = merged[mirror_side]
    for label in pools:
        stats["conformers"][label]["n_mirror_images_merged"] = n_mirrors if label == mirror_side else 0
        stats["conformers"][label]["n_final"] = len(pools[label])
    if pair_from != "both":
        other = "end" if pair_from == "start" else "start"
        side_name = "product" if other == "end" else "reactant"
        typer.echo(f"--pair-from {pair_from}: building a {side_name} in the frame of each of the "
                   f"{len(pools[pair_from])} {'reactant' if pair_from == 'start' else 'product'} conformer(s)...")
        t0 = time.perf_counter()
        pools[pair_from], pools[other] = _build_partners(
            pools[pair_from], end_node if other == "end" else start_node, run_inputs)
        stats["conformers"][other] = {"built_from": pair_from, "n_built": len(pools[other]),
                                      "seconds": round(time.perf_counter() - t0, 3)}
        typer.echo(f"  {len(pools[other])} built (each paired with the conformer it was built from).")
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
    ] if pair_from == "both" else [(i, n_start + i) for i in range(n_start)]
    stats["n_pairs_possible"] = len(candidates)

    if max_pairs and len(candidates) > max_pairs:
        typer.echo(
            f"Pairs: {n_start} reactant x {len(end_confs)} product conformers = {len(candidates)}; "
            f"keeping the first {max_pairs} (capping at --max-pairs={max_pairs})."
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
    # --pair-from: each pair was built from one atom correspondence already.
    structures, candidates, mechanism_summary = _expand_pairs_by_mechanism(
        structures, candidates, atom_mapping and pair_from == "both", run_inputs,
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
    if search_deadline is not None:
        # Read by MSMEP (no further splits) and by _run_msmep_pairs (no new
        # pairs); forked pair workers inherit it with run_inputs.
        setattr(run_inputs.path_min_inputs, "recursive_split_deadline", search_deadline)
        stats["search_budget_seconds"] = search_budget
    _run_msmep_pairs(
        structures, candidates, pairs_dir, run_inputs,
        parallel=parallel, parallel_workers=parallel_workers, workers=workers,
    )
    stats["msmep_seconds"] = round(time.perf_counter() - t0, 3)
    if search_deadline is not None:
        stats["search_budget_reached"] = time.time() > search_deadline
    if getattr(run_inputs.path_min_inputs, "direct_only", False):
        # Legs through other species that were not run; pairs left with
        # nothing to search, and pairs with some legs dropped.
        from mepd.cli_common import direct_only_counts

        stats["direct_only"] = direct_only_counts(pairs_dir)
    _write_stats()

    tree_dirs = _completed_tree_dirs(pairs_dir)
    if not tree_dirs:
        if (stats.get("direct_only") or {}).get("pairs_not_characterized"):
            # Expected with --direct-only, not a failure: every search needed an intermediate.
            typer.echo("No channel: every pair's legs run between species other than the start and end, and "
                       "--direct-only does not run those. Rerun without it to characterize them.")
            return
        typer.echo("No pairs completed successfully; nothing to build a network from.")
        raise typer.Exit(code=1)

    # network.json is a by-product: TS optimization and classification read
    # the pair trees themselves, so a network that can't be built (e.g. no
    # elementary leaf left once --direct-only dropped legs) is reported,
    # not fatal.
    builder = NetworkBuilder(data_dir=output, network_inputs=NetworkInputs())
    try:
        pot = builder.create_rxn_network_from_paths(tree_dirs)
    except Exception as exc:
        typer.echo(f"WARNING: network construction failed ({type(exc).__name__}: {exc}); "
                   "continuing with TS optimization and classification.")
        stats["network_error"] = f"{type(exc).__name__}: {exc}"
    else:
        network_path = output / "network.json"
        pot.write_to_disk(network_path)
        typer.echo(
            f"Wrote network to {network_path} "
            f"({pot.number_of_nodes} nodes, {pot.graph.number_of_edges()} edges)"
        )

    t0 = time.perf_counter()
    mismatch_warnings = _discover_channels(
        output, start_node, end_node, run_inputs, charge, multiplicity, workers=workers,
    )
    if mismatch_warnings:
        stats["endpoint_mismatch_warnings"] = mismatch_warnings
    stats["ts_discovery_seconds"] = round(time.perf_counter() - t0, 3)
    _write_stats()
