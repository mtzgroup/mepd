"""Pick among several candidate atom-to-atom mappings (or no remapping at
all) between two reaction endpoints, using a geodesic-interpolated-path
metric -- generalizes the old single-mapping "reindex or not" veto
(`mepd/cli.py::_check_endpoint_atom_mapping`) into an N-way comparison
across {identity ordering, candidate mapping 1, ..., candidate mapping k}.

Which metric is actually the best predictor of a "correct" mapping is not
settled up front: `"gi-energy"` (an actual QM energy evaluation along the
geodesic path) is the most direct proxy but the most expensive per
candidate; `"geodesic-distance"` and `"path-rmsd"` are effectively free
byproducts of the same interpolation, but unvalidated as mapping-quality
signals; `"endpoint-rmsd"` skips the interpolation altogether, so it is
cheaper again and weaker again (see `score_candidate`). All four are kept
pluggable, and `score_candidate_all_metrics` lets a caller (see `cli.py`'s
`--debug-dump` handling) record every one of them for every candidate so
real runs double as comparative data.
"""
from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qcdata.models.structure import Structure

from mepd.atom_mapping import AtomMapping, realign_end_to_start
from mepd.atom_mapping_metrics import METRICS  # noqa: F401  (re-exported)
from mepd.chain import Chain


@dataclass
class MappingCandidate:
    label: str  # "identity" or "mapping_<i>"
    end_structure: Structure
    atom_map: Optional[AtomMapping]  # None for identity


@dataclass
class SelectionResult:
    winner: MappingCandidate
    scores: dict[str, float] = field(default_factory=dict)  # label -> score under the selection metric
    chains: dict[str, Chain] = field(default_factory=dict)  # label -> its geodesic-interpolated chain
    quantity: str = ""   # what `scores` hold, when not the metric's own score (snap-gi-xtb: the rule that decided)


def build_candidates(
    start_structure: Structure, end_structure: Structure, atom_maps: list[AtomMapping]
) -> list[MappingCandidate]:
    """`end_structure` unchanged ("identity"), plus one candidate per
    non-identity mapping in `atom_maps` -- de-duplicated if two mappings
    happen to produce the same atom order (SLAPMapper's tied candidates can
    coincide after symmetry, even though `_remove_isomorphic_results`
    already dedupes most of those upstream). Identity is left out when the
    end's element order differs from the start's: it would pair up
    mismatched atoms, so its score means nothing."""
    same_order = list(start_structure.symbols) == list(end_structure.symbols)
    candidates = [MappingCandidate(label="identity", end_structure=end_structure, atom_map=None)] if same_order else []
    seen_orders = {tuple(range(len(end_structure.symbols)))}
    i = 0
    for atom_map in atom_maps:
        if atom_map.is_identity:
            continue
        order = tuple(atom_map.as_order())
        if order in seen_orders:
            continue
        seen_orders.add(order)
        candidates.append(
            MappingCandidate(
                label=f"mapping_{i}",
                end_structure=realign_end_to_start(atom_map, end_structure),
                atom_map=atom_map,
            )
        )
        i += 1
    return candidates


def _aligned_rmsd(a: Structure, b: Structure) -> float:
    """Kabsch-aligned RMSD (bohr) between two identically-indexed
    structures -- no interpolation, no engine, microseconds not seconds.
    Backs the "endpoint-rmsd" metric: unlike the other three, it says
    nothing about what happens ALONG the path between `a` and `b`, only
    how far apart the two fixed endpoints are."""
    import numpy as np

    from mepd.rigid_alignment import kabsch_align

    x = np.asarray(a.geometry, dtype=float)
    y = np.asarray(b.geometry, dtype=float)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    aligned = kabsch_align(x, y)
    return float(np.sqrt(((aligned - y) ** 2).sum(axis=1).mean()))


def _interpolate(candidate: MappingCandidate, start_structure: Structure, run_inputs):
    import mepd.chainhelpers as ch
    from mepd.nodes.node import StructureNode

    start_node = StructureNode(structure=start_structure)
    end_node = StructureNode(structure=candidate.end_structure)
    seed_chain = Chain.model_validate({
        "nodes": [start_node, end_node],
        "parameters": copy.deepcopy(run_inputs.chain_inputs),
    })
    chain, smoother = ch.run_geodesic(
        chain=seed_chain,
        chain_inputs=copy.deepcopy(run_inputs.chain_inputs),
        nimages=run_inputs.gi_inputs.nimages,
        friction=run_inputs.gi_inputs.friction,
        nudge=run_inputs.gi_inputs.nudge,
        random_seed=run_inputs.gi_inputs.random_seed,
        return_smoother=True,
    )
    return chain, smoother


def rmsd_window(run_inputs) -> float:
    ami = getattr(run_inputs, "atom_mapping_inputs", None)
    return float(getattr(ami, "rmsd_window", 1.0) if ami is not None else 1.0)


def lowest_rmsd_variants(candidates: list, start_structure: Structure, run_inputs, *, always: tuple = ()) -> list:
    """rmsd-geodesic: the `gi_variant_cap` lowest-endpoint-RMSD candidates (plus `always`)."""
    cap = int(getattr(getattr(run_inputs, "atom_mapping_inputs", None), "gi_variant_cap", 20))
    if cap <= 0 or len(candidates) <= cap:
        return list(candidates)
    rmsd = [_aligned_rmsd(start_structure, c.end_structure) for c in candidates]
    keep = set(sorted(range(len(candidates)), key=rmsd.__getitem__)[:cap])
    keep |= {k for k, c in enumerate(candidates) if c.label in always}
    return [c for k, c in enumerate(candidates) if k in keep]


def near_lowest(values: list[float], window: float = 1.0, *, keep: int = 1) -> list[int]:
    """Indices of the values within `window` standard deviations of the
    lowest (the "degenerate" ones, as far as that score can tell), lowest
    first; at least `keep` of them (the next-lowest fill up), all when they
    are all equal."""
    import numpy as np

    if not values:
        return []
    v = np.asarray(values, dtype=float)
    order = [int(i) for i in np.argsort(v, kind="stable")]
    cut = float(v.min()) + window * float(v.std())
    inside = [i for i in order if v[i] <= cut + 1e-12]
    return inside if len(inside) >= keep else order[:keep]


def score_candidate(
    candidate: MappingCandidate, metric: str, start_structure: Structure, run_inputs,
) -> tuple[float, Optional[Chain]]:
    """Reduces `candidate` to a single score under `metric` -- lower is
    always "better". `endpoint-rmsd` is the odd one out: it skips the
    geodesic interpolation entirely (returns `chain=None`), so it's orders
    of magnitude cheaper than the other three but, per its own docstring
    (`_aligned_rmsd`), a weaker signal -- untested at scale against the
    others before relying on it for something consequential; see
    docs/channels_candidates.md's open-problem note on mapping cost."""
    if metric in ("endpoint-rmsd", "snap", "snap-gi-xtb"):
        # The snap metrics choose among a mechanism's symmetry variants
        # (`snap_choice`); anything else they score is scored by its ends.
        return _aligned_rmsd(start_structure, candidate.end_structure), None

    chain, smoother = _interpolate(candidate, start_structure, run_inputs)

    # "rmsd-geodesic" filters conformer PAIRS by endpoint RMSD (mepd channels);
    # a candidate that is scored at all is scored by its GI path.
    if metric in ("geodesic-distance", "rmsd-geodesic"):
        return float(smoother.length), chain
    if metric == "path-rmsd":
        return float(chain.path_length[-1]), chain
    if metric == "gi-energy":
        run_inputs.engine.compute_energies(chain)
        return float(max(chain.energies_kcalmol)), chain
    raise ValueError(f"Unknown atom-mapping selection metric '{metric}'. Known: {METRICS}.")


def score_candidate_all_metrics(
    candidate: MappingCandidate, start_structure: Structure, run_inputs,
) -> tuple[dict[str, float], Chain]:
    """Like `score_candidate`, but computes every metric from a single
    interpolation (geodesic-distance and path-rmsd are free byproducts of
    it; gi-energy is the only one requiring an extra QM evaluation;
    endpoint-rmsd needs no interpolation at all) -- used for `--debug-dump`
    so every candidate's data is directly comparable across metrics."""
    chain, smoother = _interpolate(candidate, start_structure, run_inputs)
    run_inputs.engine.compute_energies(chain)
    scores = {
        "geodesic-distance": float(smoother.length),
        "path-rmsd": float(chain.path_length[-1]),
        "gi-energy": float(max(chain.energies_kcalmol)),
        "endpoint-rmsd": _aligned_rmsd(start_structure, candidate.end_structure),
    }
    scores["rmsd-geodesic"] = scores["geodesic-distance"]   # a scored candidate is scored by its GI path
    scores["snap"] = scores["snap-gi-xtb"] = scores["endpoint-rmsd"]   # see score_candidate
    return scores, chain


def select_best_candidate(
    candidates: list[MappingCandidate],
    metric: str,
    start_structure: Structure,
    run_inputs,
    veto_margin: float = 0.0,
) -> SelectionResult:
    """Scores every candidate under `metric` and picks the lowest-scoring
    one -- "identity" (don't reindex) is just one more candidate in the
    pool, not a privileged default. A non-identity winner is only adopted
    if it beats identity's score by more than `veto_margin`; otherwise
    identity is kept, even if some other candidate scored marginally
    better (a stability guard against metric noise, opt-in via
    `--atom-mapping-veto-margin`; the default of 0.0 is a pure best-of-N
    with identity winning exact ties)."""
    scores: dict[str, float] = {}
    chains: dict[str, Chain] = {}
    identity = next((c for c in candidates if c.label == "identity"), None)
    if metric in ("snap", "snap-gi-xtb"):
        # Snap's picks from every candidate (each SLAPMapper mapping, and the
        # current numbering), ranked together by GI + xtb.
        picks = _distinct([p for c in candidates for p in _snap_picks(c, start_structure)])
        _, best, _, scores, quantity = _rank_picks(picks, candidates, start_structure, run_inputs,
                                                   use_path=metric == "snap-gi-xtb")
        return SelectionResult(winner=best, scores=scores, chains=chains, quantity=quantity)
    if metric == "rmsd-geodesic":
        candidates = lowest_rmsd_variants(candidates, start_structure, run_inputs,
                                          always=(identity.label,) if identity else ())
    for candidate in candidates:
        score, chain = score_candidate(candidate, metric, start_structure, run_inputs)
        scores[candidate.label] = score
        chains[candidate.label] = chain

    best = min(candidates, key=lambda c: scores[c.label])
    if identity and best is not identity and scores[identity.label] - scores[best.label] <= veto_margin:
        best = identity

    return SelectionResult(winner=best, scores=scores, chains=chains)


def maybe_realign_pair(
    start_structure: Structure, end_structure: Structure, run_inputs,
) -> tuple[Structure, bool]:
    """Best-of-N atom-mapping selection between an arbitrary (start, end)
    pair -- not necessarily the top-level --start/--end -- using
    `run_inputs.atom_mapping_inputs`. Returns `(chosen_end_structure,
    changed)`.

    No-ops (returns `(end_structure, False)`) if slapmapper is
    unavailable, atom counts differ, SLAPMapper finds no mapping, or every
    candidate it finds is the identity mapping. Unlike `check_atom_mapping`,
    this never warns -- meant for `mepd.msmep`'s `--atom-mapping-recheck-splits`,
    which may call this many times per run, deep in the recursion."""
    from mepd.atom_mapping import HAS_SLAPMAPPER, suggest_atom_mapping_candidates

    if not HAS_SLAPMAPPER or len(start_structure.symbols) != len(end_structure.symbols):
        return end_structure, False

    atom_mapping_inputs = run_inputs.atom_mapping_inputs
    try:
        atom_maps = suggest_atom_mapping_candidates(
            start_structure, end_structure, max_candidates=atom_mapping_inputs.n_candidates
        )
    except Exception:
        return end_structure, False

    candidates = build_candidates(start_structure, end_structure, atom_maps)
    if [c.label for c in candidates] in ([], ["identity"]):
        return end_structure, False

    try:
        result = select_best_candidate(
            candidates, atom_mapping_inputs.metric, start_structure, run_inputs,
            veto_margin=atom_mapping_inputs.veto_margin,
        )
    except Exception:
        return end_structure, False

    return result.winner.end_structure, result.winner.label != "identity"


@dataclass
class MechanismChoice:
    key: str  # `mepd.atom_mapping.mechanism_key`
    winner: MappingCandidate  # this mechanism's best-scoring symmetry variant
    score: float
    n_variants: int


def _xtb_engine():
    """GFN2-xTB from whatever xtb `mepd.programs` finds installed (no
    download), to rank interpolations cheaply whatever the run's own level;
    None if there is none."""
    try:
        from mepd.engines.gxtb import GXTBCalculator
        from mepd.programs import xtb_executable

        exe = xtb_executable(download=False)
        return GXTBCalculator(executable=exe, add_gxtb_flag=False, n_parallel=1) if exe else None
    except Exception:
        return None


def _mechanism(label: str) -> str:
    """'mapping_15 snap2' -> 'mapping_15': the symmetric variants of one mechanism."""
    return label.split(" snap")[0]


def _relaxed_choice(picks: list, paths: list, peaks: list, top_frame: list, start_structure: Structure, top: int):
    """Stage 2 of snap-gi-xtb: the best variant of each of the `top` best
    mechanisms (by raw peak) has its path relaxed (`_relaxed_barrier`) and
    the lowest barrier wins. Returns (index into picks, {label: barrier},
    how many were relaxed), or None (no xtb, or none could be relaxed)."""
    from concurrent.futures import ThreadPoolExecutor

    from mepd.programs import xtb_executable

    exe = xtb_executable(download=False)
    if exe is None:
        return None
    best = {}
    for i, p in enumerate(peaks):
        if math.isfinite(p):
            m = _mechanism(picks[i].label)
            if m not in best or p < peaks[best[m]]:
                best[m] = i
    chosen = sorted(best.values(), key=peaks.__getitem__)[:top]
    # SLAPMapper's own (minimal-edit) mechanism is always among them: an
    # explored relay has to beat it relaxed, not just raw.
    minimal = [i for i in best.values() if not getattr(picks[i].atom_map, "explored", False)]
    if minimal and not set(minimal) & set(chosen):
        chosen.append(min(minimal, key=peaks.__getitem__))
    with ThreadPoolExecutor(max_workers=min(4, len(chosen))) as pool:
        barriers = list(pool.map(lambda i: _relaxed_barrier(paths[i][0], start_structure, picks[i].end_structure,
                                                            exe, peak_frame=top_frame[i]), chosen))
    ok = [(b, i) for b, i in zip(barriers, chosen) if b is not None and math.isfinite(b)]
    if not ok:
        return None
    k = min(ok)[1]
    return k, {picks[i].label: (b if b is not None else math.inf) for b, i in zip(barriers, chosen)}, len(chosen)


def _relaxed_barrier(chain: Chain, start_structure: Structure, end_structure: Structure, exe: str,
                     *, stride: int = 2, peak_frame: Optional[int] = None, timeout: float = 300.0) -> Optional[float]:
    """The highest energy along `chain` (kcal/mol above its first frame)
    once each interior frame (every `stride`-th, and the raw maximum) is
    optimized by xtb (GFN2, crude) with the distances of the bonds that
    change between start and end held at that frame's values: the reaction
    stays where the interpolation put it, the clashes relax away. None if
    xtb fails on every frame."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from qcconst.constants import ANGSTROM_TO_BOHR

    from mepd.discovery.nanoreactor import perceive_bonds

    symbols = list(start_structure.symbols)
    xyz = [np.asarray(n.coords, dtype=float).reshape(-1, 3) / ANGSTROM_TO_BOHR for n in chain.nodes]
    a = np.asarray(start_structure.geometry).reshape(-1, 3) / ANGSTROM_TO_BOHR
    b = np.asarray(end_structure.geometry).reshape(-1, 3) / ANGSTROM_TO_BOHR
    changed = sorted(set(perceive_bonds(symbols, a)) ^ set(perceive_bonds(symbols, b)))
    interior = list(range(1, len(xyz) - 1))
    if not interior:
        return None
    frames = set(interior[::stride])
    if peak_frame is not None:
        frames.add(peak_frame)
    charge, uhf = int(start_structure.charge), int(start_structure.multiplicity) - 1
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": os.environ.get("OMP_STACKSIZE", "1G")}

    def energy(coords, constrain) -> Optional[float]:
        tmp = Path(tempfile.mkdtemp(prefix="mepd_relax_"))
        try:
            (tmp / "f.xyz").write_text(f"{len(symbols)}\n\n" + "".join(
                f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(symbols, coords)))
            argv = [exe, "f.xyz", "--gfn", "2", "--chrg", str(charge)] + (["--uhf", str(uhf)] if uhf else [])
            if constrain:
                (tmp / "c.inp").write_text("$constrain\n   force constant=1.0\n" + "".join(
                    f"   distance: {i + 1}, {j + 1}, auto\n" for i, j in changed) + "$end\n")
                argv += ["--opt", "crude", "--input", "c.inp"]
            cmd = "ulimit -s unlimited 2>/dev/null; exec " + " ".join(f"'{x}'" for x in argv)
            done = subprocess.run(["bash", "-c", cmd], cwd=tmp, env=env, capture_output=True, text=True,
                                  timeout=timeout)
            if done.returncode != 0:
                return None
            if constrain:
                head = (tmp / "xtbopt.xyz").read_text().splitlines()[1]
                return float(head.split("energy:")[1].split()[0])
            for line in done.stdout.splitlines():
                if "TOTAL ENERGY" in line:
                    return float(line.split()[3])
            return None
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    e0 = energy(xyz[0], constrain=False)
    if e0 is None:
        return None
    found = [e for e in (energy(xyz[f], constrain=True) for f in sorted(frames)) if e is not None]
    if not found:
        return None
    return (max(found) - e0) * KCAL_PER_HARTREE


def _short_error(exc: Exception) -> str:
    """An exception as one short line (an xtb failure's last words)."""
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[-1][:160]}" if text else type(exc).__name__


KCAL_PER_HARTREE = 627.5094740631
PEAK_TIE = 0.5   # kcal/mol


def _snap_picks(candidate: MappingCandidate, start_structure: Structure) -> list[MappingCandidate]:
    """Snap's picks for one candidate: its relabelings by reactant
    automorphisms (`_snap_variants`, a few at most) that bring its end
    closest to the start, as candidates with the same mechanism. A pick that
    leaves the end's numbering as it was is labelled "identity"."""
    from mepd.atom_mapping import AtomMapping, _snap_variants, realign_end_to_start

    n = len(start_structure.symbols)
    base = candidate.atom_map.mapping if candidate.atom_map is not None else {i: i for i in range(n)}
    out = []
    for k, s in enumerate(_snap_variants(AtomMapping(mapping={i: i for i in range(n)}, cost=0, n_alternatives=1),
                                         start_structure, candidate.end_structure)):
        mapping = {i: base[s.mapping[i]] for i in range(n)}
        same = all(mapping[i] == i for i in range(n))
        out.append(MappingCandidate(
            label="identity" if same else f"{candidate.label} snap{k}",
            end_structure=realign_end_to_start(s, candidate.end_structure),
            atom_map=None if same else AtomMapping(
                mapping=mapping, cost=getattr(candidate.atom_map, "cost", 0),
                n_alternatives=getattr(candidate.atom_map, "n_alternatives", 1),
                explored=getattr(candidate.atom_map, "explored", False))))
    return out


def _distinct(cands: list[MappingCandidate]) -> list[MappingCandidate]:
    seen, out = set(), []
    for c in cands:
        order = tuple(c.atom_map.as_order()) if c.atom_map is not None else ()
        if order not in seen:
            seen.add(order)
            out.append(c)
    return out


def _rank_picks(picks: list, fallback: list, start_structure: Structure, run_inputs, use_path: bool = True):
    """Snap + GI + xtb over `picks`: each interpolated (GI), the one whose
    xtb energy profile peaks lowest; without xtb the shortest
    interpolation; if interpolation fails (or `use_path` is off, the cheap
    pair-ranking stage) the lowest endpoint RMSD among `picks`; with no
    picks (snap unusable) the lowest endpoint RMSD over `fallback`.
    Returns (endpoint RMSD of the choice, the choice, the rule that chose
    it, {label: the quantity it compared}, what that quantity is)."""
    pool, rule = (picks, "snap") if picks else (fallback, "endpoint-rmsd")
    rmsd = [_aligned_rmsd(start_structure, c.end_structure) for c in pool]
    k = min(range(len(pool)), key=rmsd.__getitem__)

    def by_rmsd(why):
        return (rmsd[k], pool[k], rule, {c.label: r for c, r in zip(pool, rmsd)},
                f"endpoint RMSD, bohr ({rule}: {why})")
    if not picks:
        return by_rmsd("snap unavailable")
    if not use_path or len(picks) == 1:
        return by_rmsd("pair ranking" if not use_path else "one snap pick")
    try:
        paths = [_interpolate(c, start_structure, run_inputs) for c in picks]
    except Exception:
        return by_rmsd("interpolation failed")
    length = [float(smoother.length) for _, smoother in paths]
    engine = _xtb_engine()
    if engine is not None:
        # Each path on its own: one xtb cannot evaluate (an SCF that fails on
        # a distorted frame, e.g. a relay dragging an H past other atoms)
        # ranks last instead of costing every path its xtb score.
        peaks, failed, top_frame = [], [], []
        for c, (chain, _) in zip(picks, paths):
            try:
                energies = [float(e) for e in engine.compute_energies(chain)]
                if not energies or not all(math.isfinite(e) for e in energies):
                    raise ValueError("non-finite energies")
                peaks.append((max(energies) - energies[0]) * KCAL_PER_HARTREE)
                top_frame.append(max(range(1, len(energies) - 1), key=energies.__getitem__) if len(energies) > 2 else None)
            except Exception as exc:
                peaks.append(math.inf)
                top_frame.append(None)
                failed.append((c.label, _short_error(exc)))
        scored = [p for p in peaks if math.isfinite(p)]
        top = int(getattr(getattr(run_inputs, "atom_mapping_inputs", None), "relax_top", 0) or 0)
        if top > 0 and len(scored) > 1:
            relaxed = _relaxed_choice(picks, paths, peaks, top_frame, start_structure, top)
            if relaxed is not None:
                k, barriers, n = relaxed
                note = (f"; {len(failed)} path(s) xtb could not evaluate, e.g. {failed[0][0]}: {failed[0][1]}"
                        if failed else "")
                return (rmsd[k], picks[k], "snap-gi-xtb-relaxed", barriers,
                        f"xtb barrier along the path after relaxing all but its changing bonds, kcal/mol, for the "
                        f"best {n} mechanism(s) by raw peak (snap-gi-xtb, relax_top {top}{note})")
        if scored:
            low = min(scored)
            # Peaks within PEAK_TIE of the lowest are a tie (a path that only
            # goes downhill peaks at 0, whatever it does): the shortest wins.
            k = min(range(len(picks)), key=lambda i: (not math.isfinite(peaks[i]), peaks[i] > low + PEAK_TIE, length[i]))
            note = (f"; {len(failed)} of {len(picks)} not evaluated by xtb, ranked last (e.g. {failed[0][0]}: "
                    f"{failed[0][1]})") if failed else ""
            return (rmsd[k], picks[k], "snap-gi-xtb", {c.label: p for c, p in zip(picks, peaks)},
                    f"xtb peak along the interpolation, kcal/mol (snap-gi-xtb{note})")
        why = f"xtb failed on every interpolation, e.g. {failed[0][0]}: {failed[0][1]}"
    else:
        why = "no xtb"
    k = min(range(len(picks)), key=length.__getitem__)
    return rmsd[k], picks[k], "snap-gi", {c.label: v for c, v in zip(picks, length)}, f"GI length (snap-gi: {why})"


def snap_choice(candidates: list, start_structure: Structure, run_inputs,
                *, use_path: bool = True) -> tuple[float, MappingCandidate, str]:
    """One mechanism's symmetry variant by snap + GI + xtb (`_rank_picks`
    over the snap picks of the mechanism's first candidate)."""
    return _rank_picks(_distinct(_snap_picks(candidates[0], start_structure)), candidates,
                       start_structure, run_inputs, use_path)[:3]


def select_per_mechanism(
    start_structure: Structure, end_structure: Structure, metric: str, run_inputs,
    *, max_variants_per_mechanism: int = 200, only_keys: Optional[set] = None,
) -> list[MechanismChoice]:
    """For one (reactant, product) pair: every mechanism SLAPMapper's
    minimal-cost mappings allow, each represented by its best symmetry
    variant under `metric` -- i.e. the geodesic score chooses how to label a
    mechanism's equivalent atoms for THIS pair's geometry, but never chooses
    between mechanisms. Sorted best-scoring first.

    The current ordering ("identity") is scored as one more variant of
    whichever mechanism it implies, or as a mechanism of its own if it
    isn't one of SLAPMapper's. Returns [] if there is nothing to map
    (slapmapper missing, atom counts or compositions differ). `only_keys`:
    score only these mechanisms (the others are left out)."""
    from mepd.atom_mapping import (
        HAS_SLAPMAPPER, mechanism_key, realign_end_to_start, suggest_mechanism_candidates,
    )

    if not HAS_SLAPMAPPER or len(start_structure.symbols) != len(end_structure.symbols):
        return []
    groups = suggest_mechanism_candidates(
        start_structure, end_structure, max_variants_per_mechanism=max_variants_per_mechanism,
        explore=int(getattr(getattr(run_inputs, "atom_mapping_inputs", None), "explore_mechanisms", 0) or 0),
    )
    if not groups:
        return []

    identity_order = tuple(range(len(end_structure.symbols)))
    by_key: dict[str, list[MappingCandidate]] = {}
    for key, atom_maps in groups.items():
        by_key[key] = [
            MappingCandidate(
                label=f"{key} #{n}",
                end_structure=end_structure if tuple(m.as_order()) == identity_order
                else realign_end_to_start(m, end_structure),
                atom_map=None if tuple(m.as_order()) == identity_order else m,
            )
            for n, m in enumerate(atom_maps)
        ]
    if (list(start_structure.symbols) == list(end_structure.symbols)
            and not any(c.atom_map is None for cands in by_key.values() for c in cands)):
        key = mechanism_key(start_structure, end_structure)
        by_key.setdefault(key, []).append(
            MappingCandidate(label=f"{key} identity", end_structure=end_structure, atom_map=None)
        )

    choices = []
    for key, cands in by_key.items():
        if only_keys is not None and key not in only_keys:
            continue
        if metric in ("snap", "snap-gi-xtb"):
            score, best, _ = snap_choice(cands, start_structure, run_inputs, use_path=metric == "snap-gi-xtb")
            choices.append(MechanismChoice(key=key, winner=best, score=score, n_variants=len(cands)))
            continue
        pool = lowest_rmsd_variants(cands, start_structure, run_inputs) if metric == "rmsd-geodesic" else cands
        scored = [(score_candidate(c, metric, start_structure, run_inputs)[0], c) for c in pool]
        score, best = min(scored, key=lambda sc: sc[0])
        choices.append(MechanismChoice(key=key, winner=best, score=score, n_variants=len(cands)))
    choices.sort(key=lambda ch: ch.score)
    return choices
