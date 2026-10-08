"""Turn a job's --output directory into one JSON document the UI renders.

Every collector returns the same shape::

    {
      "headline": "3 channels · lowest ΔE‡ 32.8 kcal/mol",
      "barrier_kcal": 32.8 | None,       # best barrier, for edge badges
      "summary": [{"label": ..., "value": ...}],
      "groups": [{"title", "kind", "entries": [entry, ...]}],
      "warnings": [...],
    }
    entry = {"id", "label", "note", "barrier_kcal", "ts_index",
             "frames": [{"xyz", "energy_kcal", "energy_hartree", "path_length"}]}

`energy_kcal` is relative to a floor chosen per result. For barriers the
floor is the lowest reactant-side energy found anywhere in the result
(reactant conformers, IRC reactant ends, path start), never just the
search's own starting geometry -- a high-energy starting conformer must not
make a barrier look lower than it is.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

from mepd.web.workspace import HARTREE_TO_KCAL


# ------------------------------------------------------------- detection

def detect_operation(path: Path) -> Optional[str]:
    if path.is_dir():
        kind = (_read_json(path / "summary.json") or {}).get("kind")
        if (path / "network.json").exists() and kind == "nanoreactor":
            return "nanoreactor"
        if (path / "routes.json").exists() and kind == "retro":
            return "retrosynthesis"
        if (path / "conformers").is_dir():
            return "channels"
        if (path / "summary.json").exists() and list(path.glob("opt_*.xyz")):
            return "optimize"
        if (path / "mep_output.xyz").exists():
            return "ts"
        if (path / "species.xyz").exists():
            return "graph-enumeration"
        if (path / "unique.xyz").exists():
            return "hessian-sample"
        if (path / "accepted_minima.xyz").exists():
            return "hessian-global"
        if (path / "pairs").is_dir() or (path / "network.json").exists():
            return "network-splits"
        if list(path.glob("ts*.xyz")):
            return "tsopt"
    return None


# --------------------------------------------------------------- helpers

def _load_chain(fp: Path, charge: int, multiplicity: int):
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs

    if not fp.exists():
        return None
    try:
        chain = Chain.from_xyz(fp, ChainInputs(), charge=charge, spinmult=multiplicity)
    except Exception:
        return None
    return chain if len(chain) else None


def _node_energy(node) -> Optional[float]:
    e = getattr(node, "_cached_energy", None)
    return None if e is None or not math.isfinite(e) else float(e)


def _clean(x: Optional[float]) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else round(float(x), 4)


def _one_decimal(x: float) -> str:
    """`x` to one decimal as the page shows the stored value (JavaScript's
    toFixed(1) of `_clean(x)`: ties up), so a headline and its edge agree --
    f"{x:.1f}" gave 65.7 for a stored 65.75 the edge showed as 65.8."""
    from decimal import ROUND_HALF_UP, Decimal

    return str(Decimal(_clean(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _frames(nodes, baseline: Optional[float]) -> list[dict]:
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs
    from mepd.viz import _chain_path_lengths

    nodes = list(nodes)
    try:
        lengths = _chain_path_lengths(Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})) \
            if len(nodes) > 1 else [0.0]
    except Exception:
        lengths = [i / max(1, len(nodes) - 1) for i in range(len(nodes))]
    out = []
    for i, node in enumerate(nodes):
        e = _node_energy(node)
        out.append({
            "xyz": node.structure.to_xyz(),
            "energy_hartree": e,
            "energy_kcal": _clean((e - baseline) * HARTREE_TO_KCAL) if e is not None and baseline is not None else None,
            "path_length": _clean(lengths[i]) if i < len(lengths) else None,
        })
    return out


def _entry(eid: str, label: str, nodes, baseline: Optional[float], *, note: str = "",
           barrier: Optional[float] = None, ts_index: Optional[int] = None) -> dict:
    frames = _frames(nodes, baseline)
    if ts_index is None and len(frames) > 2:
        energies = [f["energy_hartree"] for f in frames]
        if all(e is not None for e in energies):
            ts_index = max(range(len(energies)), key=energies.__getitem__)
    return {"id": eid, "label": label, "note": note, "barrier_kcal": _clean(barrier),
            "ts_index": ts_index, "frames": frames}


def _min(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


# The QM/MM region of the job being read (see `collect`): its structures
# are named and compared by their QM region only.
_REGION = None


def _smiles(node) -> str:
    from mepd.web.chem import perceive_smiles

    if _REGION is not None and len(node.structure.symbols) == _REGION.natoms:
        return perceive_smiles(_REGION.model_structure(node.structure)) or ""
    return perceive_smiles(node.structure) or ""


def _same_connectivity(a, b) -> bool:
    from mepd.irc_network import _same_connectivity as same

    try:
        return bool(same(a, b))
    except Exception:
        return False


def _irc_note(chain) -> str:
    a, b = _smiles(chain[0]), _smiles(chain[-1])
    if a and b:
        return f"{a}  ⇌  {b}" if a != b else f"{a} (both ends same connectivity)"
    return ""


def _group(title: str, kind: str, entries: list[dict]) -> Optional[dict]:
    return {"title": title, "kind": kind, "entries": entries} if entries else None


def _result(headline: str, groups: list, summary: list, barrier: Optional[float] = None,
            warnings: Optional[list] = None) -> dict:
    return {
        "headline": headline,
        "barrier_kcal": _clean(barrier),
        "summary": [s for s in summary if s["value"] not in (None, "")],
        "groups": [g for g in groups if g],
        "warnings": warnings or [],
    }


def _ts_dir_items(out: Path, charge: int, multiplicity: int) -> list[tuple]:
    """(label, ts_node, irc_chain | None) for every TS of a `mepd ts`-shaped
    directory (the bare `ts` label, or `ts_*`)."""
    items = []
    for ts_fp in sorted(p for p in out.glob("ts*.xyz") if not p.stem.endswith("_irc")):
        label = ts_fp.stem
        ts_chain = _load_chain(ts_fp, charge, multiplicity)
        if ts_chain is None:
            continue
        irc = _load_chain(out / ("irc.xyz" if label == "ts" else f"{label}_irc.xyz"), charge, multiplicity)
        items.append((label, ts_chain[0], irc))
    return items


def _irc_failure(out: Path, label: str) -> Optional[str]:
    """Why the IRC of TS `label` failed (the CLI's `<label>_irc_failed.txt`),
    or None (it ran, was not asked for, or the run predates the marker)."""
    fp = Path(out) / ("irc_failed.txt" if label == "ts" else f"{label}_irc_failed.txt")
    try:
        text = fp.read_text().strip()
    except OSError:
        return None
    return (text.splitlines() or ["unknown error"])[0][:200]


def _oriented(irc, start, end):
    """`irc` drawn from `start` to `end` (the job's reactant and product), so
    an IRC reads the same way as the edge it belongs to: flipped when its
    first frame is the product side (see `mepd.cli._irc_needs_reversal`)."""
    from mepd.cli import _irc_needs_reversal

    if irc is None or start is None:
        return irc
    try:
        if _irc_needs_reversal(irc, start, end):
            irc = irc.copy()
            irc.nodes.reverse()
    except Exception:
        pass
    return irc


def _tree_leaf_chains(tree_dir: Path, charge: int, multiplicity: int) -> list[tuple[int, object]]:
    """(node index, final chain) of every leaf of an MSMEP split tree, in
    path order -- read straight from adj_matrix.txt and node_<i>.xyz.

    `TreeNode.read_from_disk` would also parse every node's optimization
    history (node_<i>_history/traj_*.xyz: one file per NEB step), which is
    where nearly all of its time goes and none of which is shown here."""
    import numpy as np

    adj = np.atleast_2d(np.loadtxt(tree_dir / "adj_matrix.txt"))
    n = adj.shape[0]

    def has_data(i: int) -> bool:
        return bool(adj[i].any())

    def children(i: int) -> list[int]:
        return [j for j in range(i + 1, n) if adj[i, j] and has_data(j)]

    leaves: list[int] = []

    def walk(i: int) -> None:
        kids = children(i)
        if not kids:
            leaves.append(i)
        for j in kids:
            walk(j)

    walk(0)
    out = []
    for i in leaves:
        chain = _load_chain(tree_dir / f"node_{i}.xyz", charge, multiplicity)
        if chain is not None:
            out.append((i, chain))
    return out


# ------------------------------------------------------------ collectors

def collect_ts(out: Path, charge: int, multiplicity: int, floor_hint: Optional[float] = None,
               _extra: Optional[dict] = None) -> dict:
    """`mepd run` output: mep_output.xyz, tree/, ts[_leaf_k].xyz (+ _irc),
    network_completion/ + network.json. `floor_hint`: a lower reactant
    energy found elsewhere (runs shown together share one floor); `_extra`
    gets the floor and the route (for collect_ts_extended)."""
    warnings: list[str] = []
    mep = _load_chain(out / "mep_output.xyz", charge, multiplicity)
    ts_items = [(label, ts, _oriented(irc, mep[0], mep[-1]) if mep is not None else irc)
                for label, ts, irc in _ts_dir_items(out, charge, multiplicity)]
    ircs = [(label, irc) for label, _, irc in ts_items if irc is not None]

    # Floor: the path's reactant end, and any IRC end with the same
    # connectivity as it (an IRC often relaxes into a lower conformer).
    reactant = mep[0] if mep is not None else None
    floor_candidates = [_node_energy(reactant)] if reactant is not None else []
    for _, irc in ircs:
        for end in (irc[0], irc[-1]):
            if reactant is not None and _same_connectivity(end, reactant):
                floor_candidates.append(_node_energy(end))
    floor = _min(floor_candidates + [floor_hint])

    groups = []
    if mep is not None:
        groups.append(_group("Minimum-energy path", "path", [_entry("mep", "MEP", mep.nodes, floor)]))

    leaves = []
    tree_dir = out / "tree"
    if (tree_dir / "adj_matrix.txt").exists():
        try:
            for k, (idx, chain) in enumerate(_tree_leaf_chains(tree_dir, charge, multiplicity)):
                leaves.append(_entry(f"leaf_{idx}", f"Step {k + 1} (node {idx})", chain.nodes, floor))
        except Exception as exc:
            warnings.append(f"split tree not loaded: {type(exc).__name__}: {exc}")
    if len(leaves) > 1:
        groups.append(_group("Elementary steps", "path", leaves))

    ts_entries, off_route, unknown, irc_entries = [], [], [], []
    failed = {label: why for label, _, _ in ts_items if (why := _irc_failure(out, label))}
    route = _verify_route(reactant, mep[-1] if mep is not None else None, ts_items, floor, failed)
    if _extra is not None:
        _extra.update(floor=floor, route=route)
    for info in route["items"]:
        label, ts_node, irc, barrier = info["label"], info["ts"], info["irc"], info["barrier"]
        entry = _entry(label, label, [ts_node], floor, barrier=barrier, note=info["note"])
        (ts_entries if info["on_route"] else unknown if irc is None else off_route).append(entry)
        if irc is not None:
            irc_entries.append(_entry(f"{label}_irc", f"{label} IRC", irc.nodes, floor, note=info["note"],
                                      barrier=barrier))
    groups.append(_group("Transition states on the start → end route", "ts", ts_entries))
    groups.append(_group("Other saddle points (do not connect start → end)", "ts_other", off_route))
    # No IRC (it failed, or was not run): what these connect is not known.
    groups.append(_group("Saddle points without an IRC (what they connect is unknown)", "ts_other", unknown))
    if failed:
        warnings.append(f"{len(failed)} IRC(s) failed, so what those TSs connect is unknown; e.g. "
                        f"{next(iter(failed))}: {next(iter(failed.values()))}")
    groups.append(_group("IRC paths", "irc", irc_entries))

    network = _network_group(out / "network.json", floor)
    groups.append(network)

    path_max = None
    if mep is not None and floor is not None and all(_node_energy(n) is not None for n in mep.nodes):
        path_max = (max(_node_energy(n) for n in mep.nodes) - floor) * HARTREE_TO_KCAL
        if groups and groups[0]["kind"] == "path":
            groups[0]["entries"][0]["barrier_kcal"] = _clean(path_max)
    verified = route["barrier"] is not None
    if verified:
        barrier = route["barrier"]
        n_route = route["n_steps"]
        source = f"IRC-verified route start → end, {n_route} step{'s' if n_route != 1 else ''}"
    elif path_max is not None:
        # Nothing IRC-connects the two ends (TS-opt slid into another saddle,
        # e.g. a conformer rotation, or no IRC was run): the path maximum is
        # the only barrier that belongs to *this* start/end pair -- and it is
        # unverified.
        barrier, source = path_max, "path maximum; not verified by TS/IRC"
        if ts_items:
            warnings.append("No optimized TS has an IRC connecting the start and end of this path, so its barrier "
                            "is unverified (see the saddle points listed for what the TS searches found instead).")
    else:
        barrier, source = None, ""
    n_steps = len(leaves) if leaves else (1 if mep is not None else 0)
    approx = "" if verified else "≈ "
    headline = (f"ΔE‡ {approx}{_one_decimal(barrier)} kcal/mol ({source})" if barrier is not None else "No path found") + \
        (f" · {n_steps} path steps" if n_steps > 1 else "")
    summary = [
        {"label": "Elementary steps", "value": n_steps or None},
        {"label": "Optimized TSs", "value": len(ts_items) or None},
        {"label": "Barrier", "value": "verified: IRCs connect start → end" if verified else "not verified by IRC"},
        {"label": "Barrier reference", "value": "lowest reactant-side energy (path start / IRC ends)"},
    ]
    result = _result(headline, groups, summary, barrier, warnings)
    result["barrier_floor_hartree"] = floor     # barrier_kcal = E(TS) - this (summarize: the TS's energy)
    result["barrier_verified"] = verified
    # Explore draws the edge start -> end solid only for a direct path: a
    # verified single step, or an unverified path that did not split.
    result["n_steps"] = route["n_steps"] if verified else n_steps
    result["direct_barrier_kcal"] = _clean(route["direct"]) if route["direct"] is not None else None
    if verified:
        # The TS that sets the edge's barrier (a VRI search on the edge starts here).
        top = [i for i in route["items"] if i["on_route"] and i["barrier"] is not None
               and abs(i["barrier"] - route["barrier"]) < 1e-6]
        if top:
            result["route_ts"] = _route_ts(top[0]["label"], top[0]["ts"], top[0]["barrier"])
    return result


def _connectivity_classes(nodes: list) -> list[int]:
    """Class id per node: same molecular graph (mepd's own connectivity
    comparison, stereo/conformation ignored) -> same id."""
    reps, ids = [], []
    for n in nodes:
        for k, r in enumerate(reps):
            if n is not None and _same_connectivity(n, r):
                ids.append(k)
                break
        else:
            reps.append(n)
            ids.append(len(reps) - 1)
    return ids


def _route_ts(label: str, ts_node, barrier: Optional[float]) -> dict:
    """The TS a result's edge barrier comes from (kept as geometry so a VRI
    search on that edge can start from it without re-reading the result)."""
    return {"label": label, "barrier_kcal": _clean(barrier), "xyz": ts_node.structure.to_xyz()}


def _verify_route(start, end, ts_items: list, floor: Optional[float], failed: Optional[dict] = None) -> dict:
    """Which optimized TSs actually lie on a route from `start` to `end`.

    Every IRC end and the two path ends are grouped by connectivity. A TS
    whose IRC ends are different species is a step between them; one whose
    ends are the same species is a conformer change. The verified barrier is
    the minimax over start->end routes of steps (a route is as good as its
    highest TS). Returns {"barrier", "n_steps", "items": [...]} with a note
    per TS saying what it connects."""
    items = []
    failed = failed or {}

    def no_irc(label: str) -> str:
        why = failed.get(label)
        return (f"the IRC failed ({why}), so what this TS connects is unknown" if why
                else "no IRC was run, so what this TS connects is unknown")

    for label, ts_node, irc in ts_items:
        e = _node_energy(ts_node)
        barrier = (e - floor) * HARTREE_TO_KCAL if e is not None and floor is not None else None
        items.append({"label": label, "ts": ts_node, "irc": irc, "barrier": barrier, "on_route": False, "note": ""})
    if start is None or end is None:
        for it in items:
            it["note"] = _irc_note(it["irc"]) if it["irc"] is not None else no_irc(it["label"])
        return {"barrier": None, "n_steps": 0, "direct": None, "items": items}

    nodes = [start, end]
    for it in items:
        if it["irc"] is not None:
            nodes += [it["irc"][0], it["irc"][-1]]
    classes = _connectivity_classes(nodes)
    s_cls, e_cls = classes[0], classes[1]
    names = {s_cls: "start", e_cls: "end" if e_cls != s_cls else "start"}

    def name(c: int, node) -> str:
        if c not in names:
            names[c] = f"intermediate {_smiles(node) or c}"
        return names[c]

    steps = []  # (barrier, a, b, item)
    k = 2
    for it in items:
        if it["irc"] is None:
            it["note"] = no_irc(it["label"])
            continue
        a, b = classes[k], classes[k + 1]
        k += 2
        na, nb = name(a, it["irc"][0]), name(b, it["irc"][-1])
        if a == b and s_cls != e_cls:
            it["note"] = f"conformer change of the {na} species (both IRC ends are the same molecule): not a reaction step"
        else:
            it["note"] = f"IRC connects {na} ⇌ {nb}"
            if it["barrier"] is not None:
                steps.append((it["barrier"], a, b, it))

    # Minimax route: add steps from lowest barrier up until start and end join.
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # A single TS whose IRC joins start and end: a direct path (the edge
    # start -> end is then one elementary step, not only a route).
    direct = _min(barrier for barrier, a, b, _ in steps if s_cls != e_cls and {a, b} == {s_cls, e_cls})
    bottleneck = None
    for barrier, a, b, _ in sorted(steps, key=lambda t: t[0]):
        parent[find(a)] = find(b)
        if find(s_cls) == find(e_cls):
            bottleneck = barrier
            break
    if s_cls == e_cls and bottleneck is None and steps:
        bottleneck = min(t[0] for t in steps)  # an edge between two conformers
    n_steps = 0
    if bottleneck is not None:
        # One concrete route using only steps up to the bottleneck (BFS).
        adj: dict[int, list] = {}
        for barrier, a, b, it in steps:
            if barrier <= bottleneck + 1e-9:
                adj.setdefault(a, []).append((b, it))
                adj.setdefault(b, []).append((a, it))
        prev = {s_cls: None}
        queue = [s_cls]
        while queue:
            c = queue.pop(0)
            for nxt, it in adj.get(c, []):
                if nxt not in prev:
                    prev[nxt] = (c, it)
                    queue.append(nxt)
        c = e_cls
        while prev.get(c):
            c, it = prev[c]
            it["on_route"] = True
            n_steps += 1
        if s_cls == e_cls:
            for _, _, _, it in steps:
                if it["barrier"] == bottleneck:
                    it["on_route"] = True
                    n_steps = 1
                    break
    for it in items:
        if it["on_route"]:
            it["note"] += " · on the start → end route"
        elif it["irc"] is not None and "conformer change" not in it["note"]:
            it["note"] += " · not on the start → end route"
    return {"barrier": bottleneck, "n_steps": n_steps, "direct": direct, "items": items}


def _network_group(fp: Path, floor: Optional[float]) -> Optional[dict]:
    if not fp.exists():
        return None
    try:
        from mepd.pot import Pot

        pot = Pot.read_from_disk(fp)
    except Exception:
        return None
    entries = []
    for i, j, data in pot.graph.edges(data=True):
        if i > j and pot.graph.has_edge(j, i):
            continue  # NetworkBuilder stores both directions
        chains = [c for c in (data.get("list_of_nebs") or []) if c is not None and len(c)]
        if not chains:
            continue
        best = min(chains, key=lambda c: max((_node_energy(n) or 0.0) for n in c.nodes))
        b = data.get("barrier")
        entries.append(_entry(f"edge_{i}_{j}", f"Network edge {i} ⇌ {j}", best.nodes,
                              floor if floor is not None else _min(_node_energy(n) for n in best.nodes),
                              barrier=b if isinstance(b, (int, float)) else None,
                              note=f"{len(chains)} path(s)"))
    return _group("Network edges", "network", entries)


def collect_tsopt(out: Path, charge: int, multiplicity: int) -> dict:
    ts_items = _ts_dir_items(out, charge, multiplicity)
    ts_entries, irc_entries, floors = [], [], []
    for label, ts_node, irc in ts_items:
        # No reactant is known: the barrier is from the lower IRC end, so
        # draw the IRC from that end (an edge made from its ends then points
        # away from the side its barrier is measured from).
        if irc is not None and len(irc) > 1 and None not in (_node_energy(irc[0]), _node_energy(irc[-1])) \
                and _node_energy(irc[-1]) < _node_energy(irc[0]):
            irc = irc.copy()
            irc.nodes.reverse()
        floor = _min(_node_energy(n) for n in (irc.nodes if irc is not None else []))
        e = _node_energy(ts_node)
        barrier = (e - floor) * HARTREE_TO_KCAL if e is not None and floor is not None else None
        floors.append(floor)
        ts_entries.append(_entry(label, label, [ts_node], floor if floor is not None else e,
                                 barrier=barrier, note=_irc_note(irc) if irc is not None else ""))
        if irc is not None:
            irc_entries.append(_entry(f"{label}_irc", f"{label} IRC", irc.nodes, floor, note=_irc_note(irc),
                                      barrier=barrier))
    headline = f"{len(ts_entries)} TS optimized" if ts_entries else "No TS converged"
    failed = {label: why for label, _, irc in ts_items if irc is None and (why := _irc_failure(out, label))}
    if ts_entries and irc_entries:
        headline += f" · {irc_entries[0]['note']}"
    elif failed:
        headline += " · IRC failed"
    for e in ts_entries:
        if e["id"] in failed:
            e["note"] = f"IRC failed ({failed[e['id']]}): what this TS connects is unknown"
    result = _result(headline, [_group("Transition states", "ts", ts_entries),
                                _group("IRC paths", "irc", irc_entries)],
                     [{"label": "Barrier reference", "value": "lower IRC end"}],
                     ts_entries[0]["barrier_kcal"] if len(ts_entries) == 1 else None)
    result["barrier_floor_hartree"] = floors[0] if len(floors) == 1 else None
    return result


def collect_channels(out: Path, charge: int, multiplicity: int, floor_hint: Optional[float] = None,
                     _extra: Optional[dict] = None) -> dict:
    """`floor_hint`, `_extra`: as for collect_ts."""
    warnings: list[str] = []
    stats = _read_json(out / "stats.json") or {}
    start_pool = _load_chain(out / "conformers" / "start.xyz", charge, multiplicity)
    end_pool = _load_chain(out / "conformers" / "end.xyz", charge, multiplicity)
    ref_start = start_pool[0] if start_pool is not None else None
    ref_end = end_pool[0] if end_pool is not None else None

    def oriented_irc(folder: Path):
        return _oriented(_load_chain(folder / "irc.xyz", charge, multiplicity), ref_start, ref_end)

    def members(folder: Path) -> tuple[str, list[str]]:
        try:
            lines = [ln.strip() for ln in (folder / "members.txt").read_text().splitlines() if ln.strip()]
        except OSError:
            return "", []
        connects = lines[0].removeprefix("connects:").strip() if lines and lines[0].startswith("connects:") else ""
        return connects, [ln for ln in lines if not ln.startswith("connects:")]

    # Classified groups on disk: (kind, title, folder, ts_node, irc, connects, member labels)
    channel_dirs = sorted((out / "channels").glob("channel_*"), key=_trailing_int)
    offtarget_dirs = sorted((out / "offtarget-exit-channels").glob("offtarget_exit_channel_*"), key=_trailing_int)
    alt_dirs = sorted((out / "alternate-channels").glob("alternate_channel_*"), key=_trailing_int)

    loaded: list[dict] = []
    for kind, dirs, title in (("channel", channel_dirs, "Channel"), ("offtarget", offtarget_dirs, "Off-target exit")):
        for d in dirs:
            ts = _load_chain(d / "ts.xyz", charge, multiplicity)
            if ts is None:
                continue
            connects, mem = members(d)
            loaded.append({"kind": kind, "title": f"{title} {_trailing_int(d)}", "ts": ts[0],
                           "irc": oriented_irc(d), "connects": connects, "members": mem})
    alternates = []
    for d in alt_dirs:
        steps = []
        for sd in sorted(d.glob("step_*"), key=_trailing_int):
            ts = _load_chain(sd / "ts.xyz", charge, multiplicity)
            if ts is None:
                continue
            connects, mem = members(sd)
            steps.append({"kind": "alternate", "title": f"Alternate {_trailing_int(d)} · step {_trailing_int(sd)}",
                          "ts": ts[0], "irc": oriented_irc(sd), "connects": connects, "members": mem})
        path_txt = (d / "path.txt").read_text().strip() if (d / "path.txt").exists() else ""
        alternates.append({"title": f"Alternate channel {_trailing_int(d)}", "steps": steps, "path": path_txt})

    # Floor: lowest reactant conformer, and every reactant-side IRC end.
    floor_candidates = [_node_energy(n) for n in (start_pool.nodes if start_pool is not None else [])]
    for item in loaded + [s for a in alternates for s in a["steps"]]:
        irc = item["irc"]
        if irc is None or ref_start is None:
            continue
        for end in (irc[0], irc[-1]):
            if _same_connectivity(end, ref_start):
                floor_candidates.append(_node_energy(end))
    floor = _min(floor_candidates + [floor_hint])

    def to_entries(items: list[dict], prefix: str) -> list[dict]:
        entries = []
        for k, item in enumerate(items):
            e = _node_energy(item["ts"])
            barrier = (e - floor) * HARTREE_TO_KCAL if e is not None and floor is not None else None
            item["barrier"] = barrier
            note = item["connects"] or (_irc_note(item["irc"]) if item["irc"] is not None else "")
            if item["members"]:
                note += f"{' · ' if note else ''}{len(item['members'])} TS search(es) converged here"
            nodes = item["irc"].nodes if item["irc"] is not None else [item["ts"]]
            ts_index = None
            if item["irc"] is not None:
                ts_index = max(range(len(nodes)), key=lambda i: _node_energy(nodes[i]) or -1e9)
            entries.append(_entry(f"{prefix}{k}", item["title"], nodes, floor, note=note,
                                  barrier=barrier, ts_index=ts_index))
        return entries

    channels = [i for i in loaded if i["kind"] == "channel"]
    offtarget = [i for i in loaded if i["kind"] == "offtarget"]
    if _extra is not None:
        _extra.update(floor=floor, channels=channels)
    channel_entries = to_entries(channels, "channel_")
    offtarget_entries = to_entries(offtarget, "offtarget_")
    alt_entries = []
    for a_i, alt in enumerate(alternates):
        entries = to_entries(alt["steps"], f"alt_{a_i}_step_")
        for e in entries:
            e["note"] = (alt["path"].splitlines()[0] + " · " if alt["path"] else "") + e["note"]
        alt_entries += entries

    groups = [
        _group("Direct channels (start → end in one step)", "channel", channel_entries),
        _group("Multi-step channels", "alternate", alt_entries),
        _group("Off-target exits (lead elsewhere)", "offtarget", offtarget_entries),
    ]

    conf_entries = []
    if start_pool is not None:
        conf_entries += [_entry(f"start_conf_{i}", f"Reactant conformer {i}", [n], floor)
                         for i, n in enumerate(start_pool.nodes)]
    if end_pool is not None:
        end_floor = _min(_node_energy(n) for n in end_pool.nodes)
        conf_entries += [_entry(f"end_conf_{i}", f"Product conformer {i}", [n], end_floor)
                         for i, n in enumerate(end_pool.nodes)]
    groups.append(_group("Conformer pools", "conformers", conf_entries))

    # Every individual TS search (unclassified included), for digging in.
    ts_dir = out / "ts"
    if ts_dir.is_dir():
        items = _ts_dir_items(ts_dir, charge, multiplicity)
        classified = {m for i in loaded for m in i["members"]} | \
            {m for a in alternates for s in a["steps"] for m in s["members"]}
        raw = []
        for label, ts_node, irc in items:
            irc = _oriented(irc, ref_start, ref_end)
            e = _node_energy(ts_node)
            barrier = (e - floor) * HARTREE_TO_KCAL if e is not None and floor is not None else None
            tag = "classified" if label in classified else "unclassified"
            raw.append(_entry(f"raw_{label}", label, irc.nodes if irc is not None else [ts_node], floor,
                              barrier=barrier, note=f"{tag}" + (f" · {_irc_note(irc)}" if irc is not None else "")))
        groups.append(_group("All TS searches", "ts", raw))

    mechanisms = _read_json(out / "pair_mechanisms.json") or []
    best = _min(i["barrier"] for i in channels)
    if channels:
        headline = f"{len(channels)} direct channel(s)" + (f" · lowest ΔE‡ {best:.1f} kcal/mol" if best is not None else "")
    elif alternates:
        headline = f"No direct channel · {len(alternates)} multi-step route(s)"
    elif offtarget:
        headline = f"No route to the product · {len(offtarget)} off-target exit(s)"
    elif (out / "network.json").exists():
        headline = "Path searches done; no TS classified"
    elif start_pool is not None:
        headline = "Conformer pools ready"
    else:
        headline = "Nothing written yet"
    conf = stats.get("conformers", {})
    summary = [
        {"label": "Reactant conformers", "value": conf.get("start", {}).get("n_final")},
        {"label": "Product conformers", "value": conf.get("end", {}).get("n_final")},
        {"label": "Conformer pools", "value": ", ".join(
            f"{name} reused from an earlier run" for side, name in (("start", "reactant"), ("end", "product"))
            if conf.get(side, {}).get("reused_from")) or None},
        {"label": "Mechanisms", "value": stats.get("n_mechanisms")},
        {"label": "Path searches", "value": stats.get("n_path_searches")},
        {"label": "Direct only", "value": (
            f"{stats['direct_only']['legs_not_run']} legs between other species not run, "
            f"{stats['direct_only']['pairs_not_characterized']} pairs with nothing left to search"
            if stats.get("direct_only") else None)},
        {"label": "Distinct mechanisms", "value": ", ".join(sorted({m["mechanism"] for m in mechanisms if m.get("mechanism")})) or None},
        {"label": "Wall time", "value": f"{stats['total_seconds']:.0f} s" if stats.get("total_seconds") else None},
        {"label": "Barrier reference", "value": "lowest reactant conformer / reactant-side IRC end"},
    ]
    result = _result(headline, groups, summary, best, warnings)
    result["barrier_floor_hartree"] = floor     # every barrier here, routes too, is E(TS) - this
    lowest = [i for i in channels if i.get("barrier") is not None and best is not None and abs(i["barrier"] - best) < 1e-6]
    if lowest:
        result["route_ts"] = _route_ts(lowest[0]["title"], lowest[0]["ts"], lowest[0]["barrier"])
    # Explore's edge: a direct channel is one step; with none, the lowest
    # multi-step route (as high as its highest step) stands, as a route.
    result["direct_barrier_kcal"] = best
    result["n_steps"] = 1 if channels else None
    routes = [(max(b), len(b)) for b in ([s.get("barrier") for s in a["steps"]] for a in alternates)
              if b and all(x is not None for x in b)]
    if not channels and routes:
        result["barrier_kcal"], result["n_steps"] = _clean(min(routes)[0]), min(routes)[1]
    return result


def collect_ts_extended(ts_out: Path, channels_out: Path, charge: int, multiplicity: int) -> dict:
    """A TS search and the Sample more paths runs added to it (a channels
    folder of its own) as one result: the channels result, with the first
    search's TSs among its channels (or noted on the channel it also found)
    and its path, all on one barrier floor (the lowest reactant energy
    either found)."""
    from mepd.cli_channels import _same_ts
    from mepd.inputs import ChainInputs

    def read(hint=None):
        ts_x, ch_x = {}, {}
        ts = collect_ts(ts_out, charge, multiplicity, floor_hint=hint, _extra=ts_x)
        ch = collect_channels(channels_out, charge, multiplicity, floor_hint=hint, _extra=ch_x) \
            if channels_out.is_dir() else _result("No paths yet", [], [])
        return ts, ts_x, ch, ch_x

    ts, ts_x, ch, ch_x = read()
    shared = _min([ts_x.get("floor"), ch_x.get("floor")])
    if shared is not None and (ts_x.get("floor") != shared or ch_x.get("floor") != shared):
        ts, ts_x, ch, ch_x = read(shared)

    groups = {g["kind"]: g for g in ch["groups"]}
    sampled = any(k in groups for k in ("channel", "alternate", "offtarget"))
    ts_entry = {e["id"]: e for g in ts["groups"] for e in g["entries"]}
    tag = "First search"

    def first(entry: dict, label: str) -> dict:
        return {**entry, "id": f"first_{entry['id']}", "label": f"{tag} · {label}"}

    def add(kind: str, title: str, entries: list, at: Optional[int] = None) -> None:
        if not entries:
            return
        if kind not in groups:
            groups[kind] = {"title": title, "kind": kind, "entries": []}
            ch["groups"].insert(len(ch["groups"]) if at is None else at, groups[kind])
        groups[kind]["entries"] += entries

    cutoffs = ChainInputs()
    channel_nodes = list(zip(groups.get("channel", {}).get("entries", []), ch_x.get("channels", [])))
    route = ts_x.get("route") or {"items": [], "n_steps": 0}
    on_route = [i for i in route["items"] if i["on_route"]]
    direct, steps, other = [], [], []
    for info in route["items"]:
        label = info["label"]
        same = next((e for e, c in channel_nodes if _same_ts(info["ts"], c["ts"], cutoffs.node_rms_thre,
                                                              cutoffs.node_ene_thre)), None)
        if info["on_route"] and same is not None:
            same["note"] += f"{' · ' if same['note'] else ''}also the first search's TS"
            continue
        entry = first(ts_entry.get(f"{label}_irc") or ts_entry[label], label)
        (direct if info["on_route"] and len(on_route) == 1 else steps if info["on_route"] else other).append(entry)
    add("channel", "Direct channels (start → end in one step)", direct, 0)
    add("alternate", "Multi-step channels", steps)
    add("offtarget", "Off-target exits (lead elsewhere)", other)
    paths = [first(e, "path" if e["id"] == "mep" else e["label"]) for g in ts["groups"] if g["kind"] == "path"
             for e in g["entries"]]
    add("path", "First search: path", paths)
    add("ts", "All TS searches", [first(ts_entry[i["label"]], i["label"]) for i in route["items"]])

    chans = groups.get("channel", {}).get("entries", [])
    best = _min(e["barrier_kcal"] for e in chans)
    if not sampled:
        # Nothing classified yet (or at all): the first search's answer stands.
        ch["headline"], ch["barrier_kcal"] = ts["headline"], ts["barrier_kcal"]
        ch["barrier_verified"], ch["route_ts"] = ts.get("barrier_verified", True), ts.get("route_ts")
        ch["n_steps"], ch["direct_barrier_kcal"] = ts.get("n_steps"), ts.get("direct_barrier_kcal")
        ch["warnings"] = ts["warnings"] + ch["warnings"]
    elif chans:
        ch["headline"] = f"{len(chans)} direct channel(s)" + (f" · lowest ΔE‡ {best:.1f} kcal/mol" if best is not None else "")
        ch["barrier_kcal"], ch["barrier_verified"] = best, True
        ch["n_steps"], ch["direct_barrier_kcal"] = 1, best   # direct channels: start -> end in one step
        lowest = next(e for e in chans if e["barrier_kcal"] == best) if best is not None else None
        if lowest is not None and lowest["id"].startswith("first_"):
            ch["route_ts"] = ts.get("route_ts")
    ch["barrier_floor_hartree"] = shared        # both reads are on it
    b = ts["barrier_kcal"]
    ch["summary"].insert(0, {"label": tag, "value": None if b is None else
                             f"ΔE‡ {b:.1f} kcal/mol" + ("" if ts.get("barrier_verified") else ", not IRC-verified")})
    ch["summary"] = [x for x in ch["summary"] if x["value"] not in (None, "")]
    return ch


def _validation_note(v: Optional[dict]) -> str:
    if not v:
        return ""
    if v.get("is_minimum"):
        return f"Hessian ✓ lowest {v['min_frequency']:.0f} cm⁻¹" + (" (rescued)" if v.get("rescued") else "")
    return f"not a minimum: {v.get('validation')}"


def _minima_result(out: Path, xyz: str, charge: int, multiplicity: int, title: str, seed_energy: Optional[float],
                   summary: list, extra_headline: str = "", validation: Optional[list] = None,
                   rejected_validation: Optional[list] = None) -> dict:
    chain = _load_chain(out / xyz, charge, multiplicity)
    nodes = list(chain.nodes) if chain is not None else []
    floor = seed_energy if seed_energy is not None else _min(_node_energy(n) for n in nodes)

    def entries_for(nodes, prefix, label, records):
        out_entries = []
        for i, n in enumerate(nodes):
            e = _node_energy(n)
            rel = (e - floor) * HARTREE_TO_KCAL if e is not None and floor is not None else None
            v = records[i] if records and i < len(records) else None
            note = " · ".join(x for x in (_smiles(n), _validation_note(v)) if x)
            entry = _entry(f"{prefix}{i}", f"{label} {i}" + (f" ({rel:+.1f})" if rel is not None else ""),
                           [n], floor, note=note)
            entry["validation"] = v
            out_entries.append(entry)
        return out_entries

    entries = entries_for(nodes, "min_", "Minimum", validation)
    rejected_chain = _load_chain(out / "rejected.xyz", charge, multiplicity)
    rejected = entries_for(list(rejected_chain.nodes) if rejected_chain is not None else [], "rejected_",
                           "Rejected", rejected_validation)
    headline = f"{len(entries)} {title.lower()}" + extra_headline
    if validation is not None:
        headline += " · Hessian-validated" + (f", {len(rejected)} rejected" if rejected else "")
    return _result(headline, [_group(title, "minima", entries),
                              _group("Rejected: not minima after Hessian check", "rejected", rejected)], summary)


def collect_hessian_sample(out: Path, charge: int, multiplicity: int) -> dict:
    s = _read_json(out / "summary.json") or {}
    hv = (s.get("hessian_validation") or {}).get("enabled")
    return _minima_result(out, "unique.xyz", charge, multiplicity, "Unique minima", s.get("seed_energy"), [
        {"label": "Hessian validation", "value": "on" if hv else "off (minima not verified)"},
        {"label": "Normal modes", "value": s.get("normal_modes_total")},
        {"label": "Displaced candidates", "value": s.get("displaced_candidates")},
        {"label": "Optimized", "value": s.get("optimized_candidates")},
        {"label": "Failed", "value": s.get("failed_candidates")},
        {"label": "Energies relative to", "value": "seed structure"},
    ], validation=s.get("unique_minima_validation") if hv else None,
        rejected_validation=s.get("rejected_minima_validation"))


def collect_hessian_global(out: Path, charge: int, multiplicity: int) -> dict:
    s = _read_json(out / "summary.json") or {}
    hv = (s.get("hessian_validation") or {}).get("enabled")
    return _minima_result(out, "accepted_minima.xyz", charge, multiplicity, "Accepted minima", s.get("start_energy"), [
        {"label": "Hessian validation", "value": "on (only validated minima accepted)" if hv else "off (minima not verified)"},
        {"label": "Rounds", "value": s.get("rounds_run")},
        {"label": "Stopped because", "value": s.get("stopped_reason")},
        {"label": "Energies relative to", "value": "seed structure"},
    ], extra_headline=f" · {s['rounds_run']} rounds" if s.get("rounds_run") else "")


def _steering_note(steering: Optional[dict], steps: Optional[list], species: list) -> Optional[str]:
    if not steering:
        return None
    if steering.get("mode") != "flux":
        return f"energy window ({steering.get('energy_window_kcal')} kcal/mol)"
    fluxed = sum(1 for sp in species[1:] if (sp.get("flux") or 0) >= steering.get("flux_threshold", 0))
    return (f"kinetic flux at {steering.get('temperature_K')} K over {steering.get('time_s'):g} s, threshold "
            f"{steering.get('flux_threshold')} · {len(steps or [])} verified TS steps · {fluxed} species reached")


def collect_graph_enumeration(out: Path, charge: int, multiplicity: int) -> dict:
    s = _read_json(out / "summary.json") or {}
    species = s.get("species") or []
    outcomes = s.get("outcomes") or {}
    result = _minima_result(out, "species.xyz", charge, multiplicity, "Species", s.get("seed_energy"), [
        {"label": "Proposed reactions", "value": sum(outcomes.values()) or None},
        {"label": "Outcomes", "value": _counts(outcomes)},
        {"label": "Rounds", "value": len(s.get("rounds") or []) or None},
        {"label": "Reactions to connect", "value": len(s.get("connections") or []) or None},
        {"label": "Hessian validation", "value": "on" if (s.get("settings") or {}).get("hessian_validation") else "off"},
        {"label": "Energies relative to", "value": "seed structure (species 0)"},
        {"label": "Grown by", "value": _steering_note(s.get("steering"), s.get("steps"), species)},
        # Each stage's method and citations: the References tab.
    ], validation=[sp.get("validation") for sp in species]
        if (s.get("settings") or {}).get("hessian_validation") else None)
    if (out / "pairs").is_dir() or (out / "network.json").exists():
        paths = collect_network_splits(out, charge, multiplicity)
        result["groups"] += paths["groups"]
        result["headline"] += " · " + paths["headline"]
    result["warnings"] += list(s.get("warnings") or [])   # e.g. negative barriers the run found
    ts_entries, irc_entries = _expansion_steps(out, s, charge, multiplicity)
    result["groups"] += [g for g in (_group("Transition states (verified steps)", "ts", ts_entries),
                                     _group("IRC paths", "irc", irc_entries)) if g]
    return result


def _expansion_steps(out: Path, s: dict, charge: int, multiplicity: int) -> tuple[list, list]:
    """Each verified step of a flux-steered expansion: its TS and its IRC
    (drawn from species a to species b), barriers from species a. Entry
    ids are the step label (`<label>` TS, `<label>_irc` IRC), which the
    graph edge for that step points at."""
    names = {sp.get("index"): sp.get("smiles") or f"species {sp.get('index')}" for sp in s.get("species") or []}
    ts_entries, irc_entries = [], []
    for st in s.get("steps") or []:
        label, files = st.get("label") or f"step_{st['a']}_{st['b']}", st.get("files") or {}
        ts = _load_chain(Path(files["ts"]), charge, multiplicity) if files.get("ts") else None
        irc = _load_chain(Path(files["irc"]), charge, multiplicity) if files.get("irc") else None
        if ts is None:
            continue
        barriers = st.get("barrier_kcal") or []
        forward = barriers[0] if barriers else None
        note = f"{names.get(st['a'], st['a'])}  ⇌  {names.get(st['b'], st['b'])}"
        e_ts = _node_energy(ts[0])
        floor = e_ts - forward / HARTREE_TO_KCAL if forward is not None and e_ts is not None else None
        ts_entries.append(_entry(label, f"TS {st['a']} → {st['b']}", [ts[0]], floor, barrier=forward, note=note))
        if irc is not None:
            first, last = _node_energy(irc[0]), _node_energy(irc[-1])
            if len(irc) > 1 and None not in (floor, first, last) and abs(last - floor) < abs(first - floor):
                irc = irc.copy()
                irc.nodes.reverse()   # start at species a's side (its energy), like the edge
            irc_entries.append(_entry(f"{label}_irc", f"IRC {st['a']} → {st['b']}", irc.nodes, floor,
                                      barrier=forward, note=note))
    return ts_entries, irc_entries


def collect_network_splits(out: Path, charge: int, multiplicity: int) -> dict:
    net = _network_group(out / "network.json", None)
    pairs = []
    pairs_dir = out / "pairs"
    if pairs_dir.is_dir():
        for pd in sorted(pairs_dir.iterdir()):
            if not (pd / "tree" / "adj_matrix.txt").exists():
                continue
            try:
                steps = _tree_leaf_chains(pd / "tree", charge, multiplicity)
            except Exception:
                continue
            if not steps:
                continue
            # Stitch the elementary steps into one path (shared ends once).
            nodes = list(steps[0][1].nodes)
            for _, chain in steps[1:]:
                nodes += list(chain.nodes)[1:]
            pairs.append(_entry(pd.name, pd.name.replace("_", " "), nodes, _node_energy(nodes[0])))
    n_edges = len(net["entries"]) if net else 0
    return _result(f"{len(pairs)} pair searches · {n_edges} network edges",
                   [net, _group("Pair paths", "path", pairs)], [])


def collect_optimize(out: Path, charge: int, multiplicity: int) -> dict:
    s = _read_json(out / "summary.json") or {"structures": []}
    entries, failed = [], []
    for rec in s["structures"]:
        chain = _load_chain(out / f"opt_{rec['index']}.xyz", charge, multiplicity)
        if chain is None:
            failed.append(f"{rec['source']}: {rec.get('error') or 'no output'}")
            continue
        entries.append(_entry(f"opt_{rec['index']}", rec["source"], [chain[0]], _node_energy(chain[0]),
                              note=_smiles(chain[0])))
    return _result(f"{len(entries)}/{len(s['structures'])} optimized",
                   [_group("Optimized structures", "minima", entries)],
                   [{"label": "Failed", "value": "; ".join(failed) or None}], warnings=failed)


def collect_complex(out: Path, charge: int, multiplicity: int) -> dict:
    """A `mepd complex` folder: the complex geometries it built, best first
    (screening-level geometries; the workspace minimizes them)."""
    s = _read_json(out / "summary.json") or {}
    chain = _load_chain(out / "complexes.xyz", charge, multiplicity)
    entries = [_entry(f"complex_{k}", f"Complex {k + 1}", [node], None, note=_smiles(node))
               for k, node in enumerate(chain or [])]
    method = s.get("method", "")
    adopted = _read_json(out / "adopted.json") or {}
    warnings = []
    if adopted.get("reacted"):
        warnings.append(f"{adopted['reacted']} of {adopted['total']} geometries were not added: their molecules bonded "
                        "or fell apart on the way (they react with each other at the screening level).")
    if adopted and not adopted.get("kept") and not adopted.get("reacted"):
        warnings.append("Nothing new: Explore already has these geometries.")
    return _result(f"{len(entries)} complex geometr{'y' if len(entries) == 1 else 'ies'}" + (f" ({method})" if method else ""),
                   [_group("Complexes", "complexes", entries)],
                   [{"label": "Method", "value": method or None}, {"label": "Time", "value": f"{s['seconds']} s" if s.get("seconds") is not None else None},
                    {"label": "Added to Explore", "value": str(adopted["kept"]) if adopted else None},
                    {"label": "Energies", "value": "at the workspace level once minimized (see Explore)"}],
                   warnings=warnings)


def collect_conformers(out: Path, charge: int, multiplicity: int) -> dict:
    """A `mepd conformers` folder: the minimized conformers (energies
    relative to the lowest), or -- without --minimize -- the backend's raw
    ones (no mepd energies)."""
    s = _read_json(out / "summary.json") or {}
    opt = _read_json(out / "optimized" / "summary.json")
    nodes, notes, failed = [], [], []
    if opt:
        for rec in opt["structures"]:
            chain = _load_chain(out / "optimized" / f"opt_{rec['index']}.xyz", charge, multiplicity)
            if chain is None:
                failed.append(f"conformer {rec['index']}: {rec.get('error') or 'no output'}")
                continue
            nodes.append((rec["index"], chain[0], rec))
    else:
        raw = _load_chain(out / "conformers.xyz", charge, multiplicity)
        for i, node in enumerate(raw or []):
            node._cached_energy = None   # the backend's MMFF/CREST energy, not comparable to mepd's
            nodes.append((i, node, {}))
    floor = _min(_node_energy(n) for _, n, _ in nodes)
    nodes.sort(key=lambda t: (_node_energy(t[1]) is None, _node_energy(t[1]) or 0.0))
    entries = []
    for i, node, rec in nodes:
        e = _node_energy(node)
        val = {k: rec[k] for k in ("is_minimum", "min_frequency", "rescued", "validation") if k in rec} or None
        note = _smiles(node) + ("" if not val else (" · Hessian ✓" if val.get("is_minimum") else " · not a minimum"))
        entry = _entry(f"conf_{i}", f"Conformer {i}" + (f" ({(e - floor) * HARTREE_TO_KCAL:+.1f})"
                                                        if e is not None and floor is not None else ""),
                       [node], floor, note=note)
        entry["validation"] = val
        entries.append(entry)
    # Minimized into the same minimum (within 0.05 kcal/mol, the rule a node
    # keeps its conformers by): one conformer, as Explore shows it.
    distinct = []
    for entry, (_, node, _) in zip(entries, nodes):
        e = _node_energy(node)
        same = next((d for d in distinct if e is not None and d[1] is not None
                     and abs(e - d[1]) * HARTREE_TO_KCAL < 0.05), None)
        if same is None:
            distinct.append((entry, e))
        else:
            entry["note"] += f" · same minimum as {same[0]['label'].split(' (')[0]}"
    stats = s.get("stats") or {}
    n_shown = len(distinct) if opt else len(entries)
    return _result(f"{n_shown} conformer(s)" + (f" ({len(entries)} minimized, some to the same minimum)"
                                                if n_shown != len(entries) else "") + f" · {s.get('backend', '?')}"
                   + ("" if opt else " (not minimized)"),
                   [_group("Conformers", "conformers", entries)],
                   [{"label": "Backend", "value": s.get("backend")},
                    {"label": "Generated", "value": stats.get("n_generated")},
                    {"label": "Kept (distinct)", "value": stats.get("n_kept")},
                    {"label": "Minimized", "value": "yes" if opt else "no (backend geometries)"},
                    {"label": "Failed", "value": "; ".join(failed) or None}], warnings=failed)


_VRI_VERDICT = {
    "bifurcation": "post-TS bifurcation: the path splits into P1 and P2",
    "second_product_untested": "second product found; not yet checked (run 'Check the bifurcation')",
    "second_product_no_split": "second product found, but sideways pushes all drain into P1",
    "vrt_no_second_product": "valley-ridge transition, but no second product",
    "transient_softening": "a transient soft mode only (no valley-ridge transition)",
    "no_vrt": "no valley-ridge transition",
}


def _kcal(x: Optional[float], digits: int = 1) -> Optional[str]:
    return None if x is None else f"{x:+.{digits}f} kcal/mol"


def _counts(counts: Optional[dict]) -> Optional[str]:
    if not counts:
        return None
    return " · ".join(f"{k} {v}" for k, v in counts.items() if v)


def collect_vri(out: Path, charge: int, multiplicity: int) -> dict:
    """A `mepd discovery vri` folder (also after `vri-check` / `vri-surface`
    have added to it): verdict per IRC branch, TS1/TS2, P1/P2, the IRC and
    the VRT/VRI points, energies relative to TS1."""
    top = _read_json(out / "summary.json")
    if not top:
        return _result("VRI search running (no summary yet)", [], [])
    candidates = top.get("ts1_candidates") or [{"label": "input", "dir": str(out)}]
    many = len([c for c in candidates if (Path(c.get("dir") or out) / "summary.json").exists()]) > 1
    ts_e, prod_e, path_e, point_e, summary, headline_bits = [], [], [], [], [], []
    has_checks = has_surface = False
    for cand in candidates:
        d = Path(cand.get("dir") or out)
        s = _read_json(d / "summary.json")
        if not s:
            continue
        tag = f"{cand.get('label', 'input')} · " if many else ""
        pre = f"{cand.get('label', 'input')}_" if many else ""
        e_ts1 = s.get("ts1_energy")
        scan = _read_json(d / "projected_freqs.json") or {}
        irc = _load_chain(d / "irc.xyz", charge, multiplicity)
        if irc is not None:
            ts_index = scan.get("ts_index")
            ts_index = int(ts_index) if ts_index is not None and 0 <= int(ts_index) < len(irc) else None
            path_e.append(_entry(f"{pre}irc", f"{tag}IRC through TS1", irc.nodes, e_ts1, ts_index=ts_index,
                                 note=_irc_note(irc)))
            if ts_index is not None:
                ts_e.append(_entry(f"{pre}ts1", f"{tag}TS1", [irc.nodes[ts_index]], e_ts1,
                                   note=f"{s.get('ts1_n_imaginary', '?')} imaginary frequency"))
        for name, b in (s.get("branches") or {}).items():
            verdict = b.get("verdict") or "no_vrt"
            label = f"{tag}{name}"
            summary.append({"label": label.capitalize(), "value": _VRI_VERDICT.get(verdict, verdict)})
            if verdict in ("bifurcation", "second_product_untested", "second_product_no_split"):
                headline_bits.append(name)
            vrt = b.get("vrt") or {}
            if vrt:
                summary.append({"label": f"VRT ({label})", "value": f"s = {vrt.get('s', 0):.2f}, "
                                f"{_kcal(vrt.get('rel_ts1_kcal_mol'))} vs TS1"})
            pr = b.get("products") or {}
            p1, p2, ts2 = pr.get("p1_rel_ts1_kcal_mol"), pr.get("p2_rel_ts1_kcal_mol"), pr.get("ts2_rel_ts1_kcal_mol")
            if p1 is not None or p2 is not None:
                summary.append({"label": f"P1 / P2 ({label})",
                                "value": f"{_kcal(p1) or '–'} / {_kcal(p2) or '–'} vs TS1"})
            if ts2 is not None:
                summary.append({"label": f"TS2 ({label})", "value": f"{_kcal(ts2)} vs TS1"
                                + (" (verified)" if pr.get("ts2_verified") else " (not verified)")})
            checks = _read_json(d / f"checks_{name}.json")
            if checks:
                has_checks = True
                vri = checks.get("vri") or {}
                if vri:
                    summary.append({"label": f"Exact VRI ({label})", "value": (
                        f"converged in {vri.get('iterations')} steps" if vri.get("converged") else "did not converge")})
                basin = _counts((checks.get("basin") or {}).get("counts"))
                if basin:
                    summary.append({"label": f"Basin test ({label})", "value": basin})
                traj = checks.get("trajectories") or {}
                if traj.get("counts"):
                    summary.append({"label": f"Trajectories ({label})",
                                    "value": f"{_counts(traj['counts'])} (of {traj.get('n')})"})
            if (d / f"surface_{name}.json").exists():
                has_surface = True
            for key, title, bucket, kind_note in (
                (f"p1_{name}", "P1", prod_e, "product the IRC reaches"),
                (f"p2_{name}", "P2", prod_e, "second product (the other side of the ridge)"),
                (f"ts2_{name}", "TS2", ts_e, "saddle between P1 and P2"),
                (f"vrt_{name}", "VRT", point_e, "where the path's sideways curvature turns negative"),
                (f"vri_{name}", "VRI", point_e, "exact valley-ridge inflection point"),
            ):
                chain = _load_chain(d / f"{key}.xyz", charge, multiplicity)
                if chain is None:
                    continue
                note = kind_note + (f" · {_smiles(chain[0])}" if bucket is prod_e else "")
                bucket.append(_entry(f"{pre}{key}", f"{tag}{title} ({name})", [chain[0]], e_ts1, note=note))
            ts2_irc = _load_chain(d / f"ts2_{name}_irc.xyz", charge, multiplicity)
            if ts2_irc is not None:
                path_e.append(_entry(f"{pre}ts2_{name}_irc", f"{tag}TS2 IRC ({name})", ts2_irc.nodes, e_ts1,
                                     note=_irc_note(ts2_irc)))
    overall = top.get("verdict") or "no_vrt"
    headline = _VRI_VERDICT.get(overall, overall)
    headline = headline[0].upper() + headline[1:]
    if headline_bits:
        headline += f" ({', '.join(headline_bits)} branch{'es' if len(headline_bits) > 1 else ''})"
    result = _result(headline, [
        _group("Transition states", "ts", ts_e),
        _group("Products", "minima", prod_e),
        _group("Ridge points", "points", point_e),
        _group("Paths", "irc", path_e),
    ], summary)
    import importlib.util

    result["vri"] = {"verdict": overall, "checked": has_checks, "surface": has_surface,
                     # The interactive explorer (mepd.viz_vri) ships with newer mepd.
                     "explorer": importlib.util.find_spec("mepd.viz_vri") is not None}
    return result


def collect_solvent(out: Path, charge: int, multiplicity: int) -> dict:
    """`mepd solvent` output: summary.json plus one folder per solvent with
    the profile's files in that solvent. The job's own barrier stays None:
    it is not a gas-phase barrier, so edge badges (which take the lowest
    barrier of an edge's jobs) must never pick it up; the solvent barriers
    live in result["conditions"] instead."""
    data = _read_json(out / "summary.json")
    if not data:
        return _result("No solvent results yet", [], [])
    source = Path(data.get("source") or "")
    entries = []

    def profile_entry(eid: str, label: str, folder: Path, prof: dict) -> Optional[dict]:
        if not prof or prof.get("barrier_kcal") is None or not prof.get("irc_file"):
            return None
        irc = _load_chain(folder / prof["irc_file"], charge, multiplicity)
        if irc is None:
            return None
        nodes = list(reversed(irc.nodes)) if prof.get("reverse") else list(irc.nodes)
        return _entry(eid, f"{label} · {prof['barrier_kcal']:.1f} kcal/mol", nodes, prof.get("floor_hartree"),
                      barrier=prof["barrier_kcal"], note=prof.get("ts_label") or "")

    gas = data.get("gas") or {}
    entries.append(profile_entry("gas", "Gas phase", source, gas))
    for row in data.get("solvents", []):
        entries.append(profile_entry(f"solvent_{row['key']}", row["label"], out / row.get("dir", row["key"]), row))
    entries = [e for e in entries if e]
    rows = [r for r in data.get("solvents", []) if r.get("barrier_kcal") is not None]
    if rows and gas.get("barrier_kcal") is not None:
        best = min(rows, key=lambda r: r["barrier_kcal"])
        headline = (f"Gas {gas['barrier_kcal']:.1f} kcal/mol · lowest in {best['label']}: "
                    f"{best['barrier_kcal']:.1f} ({best['barrier_kcal'] - gas['barrier_kcal']:+.1f})")
    else:
        headline = "No barrier in solvent could be computed"
    mode = data.get("mode")
    summary = [{"label": "Geometries", "value": "gas-phase (single points)" if mode == "single-point"
                else "re-optimized in each solvent"},
               {"label": "Solvent model", "value": f"{data.get('model_label')} "
                + ("(the engine's own)" if data.get("method") == "native" else "via GFN2-xTB correction")},
               {"label": "Temperature", "value": f"{float(data.get('temperature', 298.15)):.0f} K"}]
    result = _result(headline, [_group("Energy profiles", "path", entries)], summary)
    result["conditions"] = {k: data.get(k) for k in ("mode", "model", "model_label", "method", "temperature",
                                                     "gas", "solvents", "insights", "warnings", "kind")}
    _reword_conditions(result["conditions"])
    return result


def _reword_conditions(cond: dict) -> None:
    """Findings, half-lives and warnings in today's wording, rebuilt from
    the numbers a run saved (older runs stored longer sentences)."""
    try:
        from mepd.cli_solvent import modernize_warning
        from mepd.conditions import kinetics_row, solvent_insights

        temperature = float(cond.get("temperature") or 298.15)
        for row in [cond.get("gas") or {}, *(cond.get("solvents") or [])]:
            if row.get("barrier_kcal") is not None:
                row["t_half"] = kinetics_row(row["barrier_kcal"], temperature)["t_half"]
        ok = [r for r in cond.get("solvents") or [] if r.get("barrier_kcal") is not None and r.get("shift_kcal") is not None]
        cond["insights"] = solvent_insights(cond.get("gas") or {}, ok, temperature, mode=cond.get("mode") or "")
        cond["warnings"] = [modernize_warning(w, cond.get("model_label") or "") for w in cond.get("warnings") or []]
    except Exception:
        pass    # keep what the run wrote


def collect_mechanochem(out: Path, charge: int, multiplicity: int) -> dict:
    """`mepd force` output: summary.json (levers per channel, selectivity
    switches, checks under force) plus <channel>_<i>-<j>_<F>nN/ folders
    with the TS and IRC re-optimized under force. Like a solvent job, it
    never reports a barrier_kcal of its own (see collect_solvent)."""
    data = _read_json(out / "summary.json")
    if not data:
        return _result("No force results yet", [], [])
    entries = []
    for row in data.get("efei") or []:
        if not row.get("dir"):
            continue
        irc = _load_chain(out / row["dir"] / "irc.xyz", charge, multiplicity)
        if irc is None:
            continue
        floor = _min(_node_energy(n) for n in irc.nodes[: max(1, len(irc.nodes) // 2)])
        entries.append(_entry(row["dir"], f"{row['label']} · {row['pair_label']} at {row['force_nN']:g} nN",
                              irc.nodes, floor, note="energies on the force-modified surface (E − F·d)"))
    sel = (data.get("selectivity") or [None])[0]
    kinds = {c["id"]: c.get("kind") for c in data.get("channels") or []}
    steer = next((x for x in data.get("selectivity") or [] if kinds.get(x["channel"]) == "direct"), None)
    mechano = {k: data.get(k) for k in ("mode", "temperature", "channels", "selectivity", "insights",
                                         "warnings", "efei", "max_force", "kind")}
    _reword_mechano(mechano)
    top = next((i for i in mechano.get("insights") or [] if i["level"] == "accelerates"), None)
    if steer is not None or sel is not None:
        x = steer or sel
        headline = f"Pulling {x['pair_label']} apart (≥ {x['force_nN']:.1f} nN) lets {x['label']} win over {x['overtakes']}"
    elif top is not None:
        headline = top["text"].split(" (")[0]      # "Best lever: pull C0–C4 apart"
    else:
        headline = "No pulling pair changes these barriers much"
    summary = [{"label": "Method", "value": "Bell estimate" + (" + re-optimized under force"
                                                             if data.get("mode") == "reoptimize" else "")},
               {"label": "Channels", "value": str(len(data.get("channels") or []))}]
    result = _result(headline, [_group("Paths under force", "path", entries)], summary)
    result["mechano"] = mechano
    return result


def _reword_mechano(m: dict) -> None:
    """Findings and warnings in today's wording, rebuilt from the saved
    levers, crossovers and checks under force (see _reword_conditions)."""
    try:
        from mepd.cli_force import _efei_insights
        from mepd.mechanochem import WARNINGS, _insights

        channels = m.get("channels") or []
        if channels:
            insights = _insights(None, channels, m.get("selectivity") or [], float(m.get("temperature") or 298.15))
            m["insights"] = insights + (_efei_insights(m["efei"]) if m.get("efei") else [])
        old = m.get("warnings") or []
        if old and old[0].startswith("Bell's first-order estimate"):
            m["warnings"] = list(WARNINGS) + old[2:]
    except Exception:
        pass    # keep what the run wrote


def collect_substituents(out: Path, charge: int, multiplicity: int) -> dict:
    """`mepd substituents` output: summary.json (shift per channel x site x
    group) plus <channel>/<site>_<h>_<group>.xyz (substituted reactant, TS,
    product). No barrier_kcal of its own (see collect_solvent)."""
    data = _read_json(out / "summary.json")
    if not data:
        return _result("No substituent results yet", [], [])
    channels = data.get("channels") or []
    lead = min(channels, key=lambda c: c["barrier_kcal"])["id"] if channels else None
    groups = []
    for ch in channels:
        rows = sorted((r for r in data.get("variants") or [] if r["channel"] == ch["id"] and r.get("file")
                       and r["status"] == "ok"), key=lambda r: r["barrier_kcal"])
        entries = []
        for r in rows:
            chain = _load_chain(out / r["file"], charge, multiplicity)
            if chain is None:
                continue
            entries.append(_entry(f"{ch['id']}:{r['site']}:{r['group']}",
                                  f"{r['label']} · {r['barrier_kcal']:.1f} ({r['shift']:+.1f})", chain.nodes,
                                  _node_energy(chain[0]), barrier=r["barrier_kcal"], ts_index=1,
                                  note="reactant end · TS · product end"))
        title = "Substituted" + (f": {ch['label']}" if len(channels) > 1 else "")
        grp = _group(title, "path", entries)
        if grp:
            groups.insert(0, grp) if ch["id"] == lead else groups.append(grp)
    top = next((i["text"] for i in data.get("insights") or [] if i["level"] in ("selectivity", "accelerates")), None)
    headline = top or "No substituent shifts this barrier much"
    summary = [{"label": "Geometries", "value": "group relaxed (fast)" if data.get("mode") == "fast"
                else "TS re-optimized"},
               {"label": "Variants", "value": str(sum(1 for r in data.get("variants") or [] if r["status"] == "ok"))}]
    result = _result(headline, groups, summary)
    result["substituents"] = {k: data.get(k) for k in ("mode", "groups", "group_labels", "sites", "channels",
                                                       "insights", "trends", "warnings")}
    result["substituents"]["variants"] = [{k: v for k, v in r.items() if k != "file"}
                                          for r in data.get("variants") or []]
    return result


def _xyz_frames(texts: list, baseline: Optional[float]) -> list[dict]:
    """Frames from xyz texts whose comment line may carry 'energy=<Eh>'
    (the nanoreactor's files: each species has its own charge, so no Chain)."""
    import re

    out = []
    for k, text in enumerate(texts):
        lines = text.splitlines()
        m = re.search(r"energy=(-?[0-9.]+)", lines[1] if len(lines) > 1 else "")
        e = float(m.group(1)) if m else None
        out.append({"xyz": text, "energy_hartree": e, "path_length": float(k),
                    "energy_kcal": _clean((e - baseline) * HARTREE_TO_KCAL) if e is not None and baseline is not None
                    else None})
    return out


def _read_text(fp) -> Optional[str]:
    try:
        return Path(fp).read_text()
    except (OSError, TypeError):
        return None


def collect_nanoreactor(out: Path, charge: int, multiplicity: int) -> dict:
    """Species (each on its own: energies of different molecules are not
    compared), reactions (the optimized subsystem ends; energies from the
    reactant side), and the TSs found on them."""
    data = _read_json(out / "network.json")
    if not data:
        return _result("Reactor running…" if (out / "md").is_dir() else "No output yet", [], [])
    species, reactions = data.get("species") or [], data.get("reactions") or []
    names = {sp["id"]: sp["smiles"] for sp in species}
    sp_entries = []
    for sp in species:
        text = _read_text(sp.get("file")) or _read_text(sp.get("md_file"))
        if not text:
            continue
        note = f"charge {sp['charge']}, mult {sp['multiplicity']}; {sp['initial']} at start, seen {sp['count']}×"
        if not sp.get("file"):
            note += "; as cut from the MD (not optimized" + (f": {sp['note']}" if sp.get("note") else "") + ")"
        sp_entries.append({"id": f"species_{sp['id']}", "label": sp["smiles"], "note": note, "barrier_kcal": None,
                           "ts_index": None, "frames": _xyz_frames([text], None)})
    rx_entries, ts_entries, irc_entries, warnings = [], [], [], []
    for rx in reactions:
        c = rx.get("complex") or {}
        seen = f"seen {rx.get('count', 0)}×" + (f", reverse {rx['reverse_count']}×" if rx.get("reverse_count") else "")
        shuttles = ", ".join(names.get(i, str(i)) for i in rx.get("shuttles") or [])
        dE = rx.get("delta_e_kcal")
        note = "; ".join(x for x in (seen, f"shuttle: {shuttles}" if shuttles else "",
                                     f"ΔE {dE:+.1f} kcal/mol (separated)" if dE is not None else "") if x)
        texts = [_read_text(c.get("reactant")), _read_text(c.get("product"))]
        if all(texts):
            frames = _xyz_frames(texts, c.get("reactant_energy"))
            rx_entries.append({"id": f"reaction_{rx['id']}", "label": rx["label"], "note": note, "barrier_kcal": None,
                               "ts_index": None, "frames": frames})
        elif c.get("error"):
            warnings.append(f"{rx['label']}: {c['error']}")
        ts = rx.get("ts") or {}
        files = ts.get("files") or {}
        if ts.get("error"):
            warnings.append(f"TS of {rx['label']}: {ts['error']}")
        if files.get("ts"):
            chain = _load_chain(Path(files["ts"]), c.get("charge", 0), c.get("multiplicity", 1))
            if chain is not None:
                floor = c.get("reactant_energy")
                ts_entries.append(_entry(f"reaction_{rx['id']}_ts", rx["label"], [chain[0]], floor,
                                         barrier=ts.get("barrier_kcal"), note=note))
                irc = _load_chain(Path(files["irc"]), c.get("charge", 0), c.get("multiplicity", 1)) \
                    if files.get("irc") else None
                if irc is not None:
                    irc_entries.append(_entry(f"reaction_{rx['id']}_irc", f"IRC: {rx['label']}", irc.nodes, floor,
                                              barrier=ts.get("barrier_kcal"), note=note))
    st = data.get("settings") or {}
    n_ts = sum(1 for rx in reactions if (rx.get("ts") or {}).get("barrier_kcal") is not None)
    ps = (data.get("n_frames") or 0) * (data.get("frame_fs") or 0) / 1000 or st.get("time_ps")
    headline = (f"{len(reactions)} reaction{'s' * (len(reactions) != 1)} among {len(species)} species "
                f"({len(data.get('events') or [])} events in {ps:g} ps)")
    if n_ts:
        headline += f" · {n_ts} TS{'s' * (n_ts != 1)}"
    summary = [
        # (an interactive reactor's run names its saved first frame here: not worth showing)
        {"label": "Reactor", "value": ", ".join(m for m in st.get("molecules") or [] if "first_frame.xyz" not in m) or None},
        {"label": "MD", "value": f"{st.get('method', '').upper()} at {st.get('temperature', 0):.0f} K, "
                                 f"{st.get('time_ps')} ps, wall {st['radius']:.1f} → "
                                 f"{st['radius'] * (st.get('compress') or 1):.1f} Å" if st.get("radius") else
         (f"{'an interactive reactor' if '/sandbox/' in str(data.get('trajectory') or '') else 'an existing'} trajectory "
          f"({(data.get('n_frames') or 0) * (data.get('frame_fs') or 0) / 1000:.1f} ps)")},
        {"label": "Energies", "value": "within one reaction only (different reactions have different atoms)"},
    ]
    groups = [_group("Reactions (optimized subsystem: reactants → products)", "path", rx_entries),
              _group("Transition states", "ts", ts_entries), _group("IRC paths", "irc", irc_entries),
              _group("Species", "minima", sp_entries)]
    # No job-level barrier: each reaction's is on its own edge in Explore.
    out = _result(headline, groups, summary, warnings=warnings)
    out["nanoreactor"] = {"reactions": [{k: rx.get(k) for k in ("id", "label", "delta_e_kcal", "count", "reverse_count")}
                                        for rx in reactions]}
    return out


def collect_retro(out: Path, charge: int, multiplicity: int) -> dict:
    """`mepd retro plan` output: routes.json (+ summary.json); while running,
    the routes so far from live_network.json."""
    from mepd.retro.steps import byproducts, coreactants, step_key

    summary = _read_json(out / "summary.json") or {}
    data = _read_json(out / "routes.json")
    live = data is None
    if live:
        net = _read_json(out / "live_network.json") or {}
        return {**_result("Searching…" if net else "No routes yet", [], []),
                "retro": {"live": True, "target": net.get("target"), "routes": [],
                          "steps_so_far": len(net.get("reactions") or [])}}
    routes = data.get("routes") or []
    shown = routes or ([data["best_partial"]] if data.get("best_partial") else [])

    def step_view(s: dict) -> dict:
        v = s.get("verification") or {}
        info = s.get("info") or {}
        return {"key": step_key(s), "product": s["product"], "reactants": s["reactants"], "method": s["method"],
                "score": s.get("score"), "byproducts": byproducts(s), "coreactants": coreactants(s), "reagents": info.get("reagents") or [],
                "reaction": info.get("reaction") or "", "delta_e_kcal": info.get("delta_e_kcal"),
                "roundtrip": info.get("roundtrip"), "barrier_kcal": v.get("barrier_kcal"),
                "verified": v.get("verified"), "check": v.get("status")}

    views = [{"rank": k + 1, "n_steps": r["n_steps"], "score": r.get("score"), "solved": r.get("solved"),
              "highest_barrier_kcal": r.get("highest_barrier_kcal"), "all_verified": r.get("all_verified"),
              # forward order: deepest steps (made first) first
              "steps": [step_view(s) for s in sorted(r["steps"], key=lambda s: -s.get("depth", 0))],
              "leaves": r.get("leaves") or []} for k, r in enumerate(shown)]
    stats = summary.get("stats") or {}
    if routes:
        best = routes[0]
        headline = f"{len(routes)} route{'s' if len(routes) != 1 else ''} · best {best['n_steps']} " \
                   f"step{'s' if best['n_steps'] != 1 else ''}"
    else:
        headline = "No route reaches the stock" + (" · closest shown" if shown else "")
    rows = [{"label": "Method", "value": summary.get("method")},
            {"label": "Building blocks", "value": summary.get("stock")},
            {"label": "Search", "value": f"{stats['seconds']:g} s" + (f", {stats['expansions']} expansions"
                                                                      if stats.get("expansions") is not None else "")
             if stats.get("seconds") is not None else None},
            {"label": "Target in stock", "value": "yes" if summary.get("target_in_stock") else None}]
    out_ = _result(headline, [], rows)
    out_["retro"] = {"live": False, "target": data.get("target"), "solved": bool(routes), "routes": views,
                     "verify": summary.get("verify")}
    return out_


def collect_qmmm_build(out: Path, charge: int, multiplicity: int) -> dict:
    """A `mepd qmmm build` folder: the molecule in its solvent shell (a new
    QM/MM system; the workspace minimizes it embedded)."""
    s = _read_json(out / "summary.json") or {}
    chain = _load_chain(out / "system.xyz", charge, multiplicity)
    entries = [_entry("system", "Molecule in solvent", [chain[0]], None)] if chain else []
    return _result(f"{s.get('qm_atoms', '?')} QM atoms in {s.get('solvent', 'solvent')}: {s.get('natoms', '?')} atoms",
                   [_group("QM/MM system", "qmmm", entries)],
                   [{"label": "QM atoms", "value": s.get("qm_atoms")},
                    {"label": "Moving environment atoms", "value": s.get("active_mm_atoms")},
                    {"label": "Frozen environment atoms", "value": s.get("frozen_atoms")},
                    {"label": "Environment level", "value": s.get("mm")},
                    {"label": "Added to Explore", "value": "as a QM/MM node, minimized embedded"}],
                   warnings=list(s.get("problems") or []))


def collect_protein_sites(out: Path, charge: int, multiplicity: int) -> dict:
    """`mepd qmmm protein-sites`: where the species docks in the protein (the
    view draws the protein with the poses; each site can be built)."""
    data = _read_json(out / "sites.json") or {}
    sites = data.get("sites") or []
    prep = data.get("prepared") or {}
    if not sites:
        return _result("Docking…" if not data else "No sites found", [], [])
    best = sites[0]
    return {**_result(f"{len(sites)} site{'s' if len(sites) != 1 else ''}, best {best['score']:.1f} kcal/mol "
                      f"(Vina score)", [],
                      [{"label": "Protein", "value": f"{prep.get('atoms')} atoms, chains {','.join(prep.get('chains') or [])}, "
                                                     f"charge {prep.get('charge', 0):+d} at pH {prep.get('ph', 7):g}"},
                       {"label": "Removed", "value": ", ".join(prep.get("removed") or [])},
                       {"label": "Gaps not rebuilt", "value": prep.get("gaps_not_rebuilt") or None}]),
            "protein_sites": {"protein": data.get("protein"), "species": data.get("species"),
                              "sites": [{k: s[k] for k in ("id", "score", "centroid", "coords", "hits", "residues")}
                                        for s in sites]}}


def collect_protein_build(out: Path, charge: int, multiplicity: int) -> dict:
    """`mepd qmmm protein-build`: the species at its site, in the protein."""
    s = _read_json(out / "summary.json") or {}
    chain = _load_chain(out / "system.xyz", charge, multiplicity)
    entries = [_entry("system", f"Site {s.get('site', '?')} in the protein", [chain[0]], None)] if chain else []
    return _result(f"Site {s.get('site', '?')}: {s.get('qm_atoms', '?')} QM atoms in {s.get('natoms', '?')}",
                   [_group("QM/MM system", "qmmm", entries)],
                   [{"label": "QM atoms", "value": s.get("qm_atoms")},
                    {"label": "Moving environment atoms", "value": s.get("active_mm_atoms")},
                    {"label": "Frozen environment atoms", "value": s.get("frozen_atoms")},
                    {"label": "Vina score", "value": s.get("score")},
                    {"label": "Added to Explore", "value": "as a QM/MM node, minimized embedded"}],
                   warnings=list(s.get("problems") or []))


def collect_qmmm_reaction(out: Path, charge: int, multiplicity: int) -> dict:
    """A `mepd qmmm reaction` / `mepd qmmm embed` folder: the structures put
    into the solvent (starting geometries: the workspace minimizes them, and
    re-optimizes a TS, embedded)."""
    s = _read_json(out / "summary.json") or {}
    entries = []
    for fname, label in (("system.xyz", "Start in solvent"), ("product.xyz", "End in solvent"),
                         ("ts.xyz", "TS in solvent (to re-optimize)")):
        chain = _load_chain(out / fname, charge, multiplicity)
        if chain:
            entries.append(_entry(fname.split(".")[0], label, [chain[0]], None))
    for k, rec in enumerate(s.get("embedded") or []):
        if rec.get("name", "").startswith("embedded_"):
            chain = _load_chain(out / f"{rec['name']}.xyz", charge, multiplicity)
            if chain:
                entries.append(_entry(rec["name"], "Put into the system", [chain[0]], None))
    contacts = [r.get("closest_contact") for r in s.get("embedded") or [] if r.get("closest_contact") is not None]
    warnings = list(s.get("problems") or []) + [w for r in s.get("embedded") or [] for w in r.get("warnings") or []]
    return _result(f"{len(entries)} structure(s) in {s.get('solvent', 'the solvent')}",
                   [_group("QM/MM structures", "qmmm", entries)],
                   [{"label": "Closest contact after placing", "value": f"{min(contacts):.2f} Å" if contacts else None},
                    {"label": "Added to Explore", "value": "as QM/MM nodes, minimized embedded (a TS re-optimized)"}],
                   warnings=warnings)


def collect_qmmm_inspect(out: Path, charge: int, multiplicity: int) -> dict:
    """A QM/MM energy split: shown on its source job's result page."""
    rep = _read_json(out / "report.json")
    if not rep:
        return _result("No energy split yet", [], [])
    return _result(f"Energy split of {len(rep.get('frames') or [])} frame(s): see the source job's QM/MM checks",
                   [], [], warnings=list(rep.get("warnings") or []))


COLLECTORS = {
    "qmmm-build": collect_qmmm_build,
    "qmmm-protein-sites": collect_protein_sites,
    "qmmm-protein-build": collect_protein_build,
    "qmmm-inspect": collect_qmmm_inspect,
    "qmmm-reaction": collect_qmmm_reaction,
    "qmmm-embed": collect_qmmm_reaction,
    "retrosynthesis": collect_retro,
    "nanoreactor": collect_nanoreactor,
    "nanoreactor-more": collect_nanoreactor,   # Run longer: the same folder, the longer run
    "optimize": collect_optimize,
    "design-optimize": collect_optimize,
    "design-tsopt": collect_tsopt,
    "conformers": collect_conformers,
    "complex": collect_complex,
    "ts": collect_ts,
    "channels": collect_channels,
    # A "Sample more paths" follow-up writes into its source's folder: the same, extended, result.
    "channels-more": collect_channels,
    "tsopt": collect_tsopt,
    "hessian-sample": collect_hessian_sample,
    "hessian-global": collect_hessian_global,
    "graph-enumeration": collect_graph_enumeration,
    "network-splits": collect_network_splits,
    "vri": collect_vri,
    # Follow-ups add to the VRI folder: they show the same, updated, result.
    "vri-check": collect_vri,
    "vri-surface": collect_vri,
    "solvent": collect_solvent,
    "mechanochem": collect_mechanochem,
    "substituents": collect_substituents,
}


def collect(job: dict, job_dir: Optional[Path] = None) -> dict:
    """A job's result. A QM/MM job's structures are named and compared by
    their QM region (the region the job ran with, from its inputs)."""
    global _REGION
    from mepd.nodes.node import StructureNode

    region_fp = Path(job_dir) / "inputs" / "qmmm_region.json" if job_dir is not None and job.get("qmmm") else None
    if region_fp is None and job.get("qmmm") and job.get("source_job"):
        region_fp = Path(job_dir).parent / job["source_job"] / "inputs" / "qmmm_region.json" if job_dir else None
    if region_fp is not None and region_fp.exists():
        from mepd.qmmm import QMMMRegion

        _REGION = QMMMRegion.open(region_fp)
        StructureNode.set_global_graph_atoms(_REGION.qm_atoms, _REGION.natoms)
    try:
        return _collect(job, job_dir)
    finally:
        _REGION = None
        StructureNode.set_global_graph_atoms(None)


def _collect(job: dict, job_dir: Optional[Path] = None) -> dict:
    out = Path(job["output_dir"])
    if not out.exists() and job_dir is not None and (Path(job_dir) / "output").exists() and not job.get("external"):
        # The session folder was moved or copied (e.g. out of the demo's
        # container): the recorded absolute path is stale, the output is here.
        out = Path(job_dir) / "output"
        job = {**job, "output_dir": str(out)}
    if not out.exists():
        return _result("No output yet", [], [])
    fn = COLLECTORS.get(job["op"])
    if fn is None:
        return _result(f"No result reader for {job['op']}", [], [])
    ext = job.get("extension")
    if job["op"] == "ts" and ext and Path(ext["output_dir"]) != out:
        # Sample more paths runs from this search: one result for all of them.
        result = collect_ts_extended(out, Path(ext["output_dir"]), int(job.get("charge") or 0),
                                     int(job.get("multiplicity") or 1))
    else:
        result = fn(out, int(job.get("charge") or 0), int(job.get("multiplicity") or 1))
    if not job.get("external"):
        result["warnings"] += _log_warnings(out.parent / "stdout.log")
    _flag_negative_barriers(result)
    return result


NEGATIVE_BARRIER_TOL = -0.1  # kcal/mol


def negative_barrier_text(barrier: float, what: str = "the barrier") -> str:
    return (f"{what} is negative ({barrier:.1f} kcal/mol): the TS lies below a minimum it connects. That cannot happen "
            "with one level of theory, minimized endpoints and a dense enough path, so check that the endpoints were "
            "minimized at this level (a lower conformer may have been found elsewhere), that every energy comes from "
            "the same level, and that the path has enough images to catch the real maximum.")


def _flag_negative_barriers(result: dict) -> None:
    """Never let a negative barrier pass quietly: every entry with one is
    marked, and the result carries a warning for it."""
    seen = set(result.get("warnings") or [])
    result.setdefault("barrier_warnings", [])
    for g in result.get("groups", []):
        for e in g["entries"]:
            b = e.get("barrier_kcal")
            if b is not None and b < NEGATIVE_BARRIER_TOL:
                e["note"] = ("⚠ negative barrier · " + e["note"]) if e.get("note") else "⚠ negative barrier"
                e["negative_barrier"] = True
                w = negative_barrier_text(b, f"{e['label']}: the barrier")
                if w not in seen:
                    seen.add(w)
                    result["barrier_warnings"].append(w)
    b = result.get("barrier_kcal")
    if b is not None and b < NEGATIVE_BARRIER_TOL:
        w = negative_barrier_text(b, "The reported barrier")
        if w not in seen:
            result["barrier_warnings"].insert(0, w)
    # Warnings the run itself reported about negative barriers (e.g. an expansion's steps) go up there too.
    moved = [w for w in result.get("warnings", []) if "negative barrier" in w.lower()]
    result["warnings"] = [w for w in result.get("warnings", []) if w not in moved]
    result["barrier_warnings"] += [w for w in moved if w not in result["barrier_warnings"]]


_WARNING_MARKERS = ("failed", "warning", "error", "not converged", "did not converge", "skipping")


def _log_warnings(log: Path, limit: int = 8) -> list[str]:
    """Problems mepd reported on stdout but that leave no trace in the output
    files (e.g. an IRC that failed while its TS was kept), so a result that
    looks complete isn't mistaken for one where every step worked."""
    try:
        lines = log.read_text(errors="replace").splitlines()
    except OSError:
        return []
    seen, out = set(), []
    for line in lines:
        text = line.rsplit("\r", 1)[-1].strip()
        if not text or text.startswith("$ ") or not any(m in text.lower() for m in _WARNING_MARKERS):
            continue
        key = text[:120]
        if key not in seen:
            seen.add(key)
            out.append(text if len(text) < 400 else text[:400] + "…")
    return out[-limit:]


# Bump when collectors change what they return, so cached results are rebuilt.
RESULT_VERSION = 29


def collect_cached(job: dict, job_dir: Path) -> dict:
    """Finished jobs are parsed once and cached; running jobs are parsed
    fresh (their output grows), so partial results are always current."""
    cache = job_dir / "result.json"
    ext = job.get("extension")
    terminal = job["status"] in ("done", "failed", "cancelled", "interrupted") and (not ext or ext["status"] == "done")
    # A page with Sample more paths runs: stale once any of them finishes.
    stamp = [job.get("finished"), ext["finished"]] if ext else job.get("finished")
    if terminal and cache.exists():
        try:
            cached = json.loads(cache.read_text())
            if cached.get("_finished") == stamp and cached.get("_version") == RESULT_VERSION:
                return cached
        except Exception:
            pass
    result = collect(job, job_dir)
    route_ts = result.get("route_ts")
    if route_ts and terminal:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "route_ts.xyz").write_text(route_ts["xyz"])
    if terminal:
        result["_finished"] = stamp
        result["_version"] = RESULT_VERSION
        job_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result))
    return result


def summarize(result: dict) -> dict:
    """The bit of a result stored on the job record (list views, edge badges)."""
    counts: dict = {}
    for g in result.get("groups", []):   # two groups can share a kind (saddle points with and without an IRC)
        counts[g["kind"]] = counts.get(g["kind"], 0) + len(g["entries"])
    return {"headline": result.get("headline"), "barrier_kcal": result.get("barrier_kcal"),
            # False: the barrier is not backed by IRCs connecting the job's
            # two ends (edges then show it as unconfirmed).
            "barrier_verified": result.get("barrier_verified", True), "counts": counts,
            # How many elementary steps the barrier's route takes, and the
            # lowest verified direct (one-step) barrier: Explore draws an edge
            # solid only for a direct path. None: not known (a single TS).
            "n_steps": result.get("n_steps"), "direct_barrier_kcal": result.get("direct_barrier_kcal"),
            # The energy (Hartree, the job's level) of the TS that sets barrier_kcal:
            # kinetics puts TSs and species on one scale with it, never adding a
            # barrier measured from one structure to another structure's energy.
            "ts_energy_hartree": ts_energy_hartree(result),
            "barrier_warning": negative_barrier_text(result["barrier_kcal"])
            if result.get("barrier_kcal") is not None and result["barrier_kcal"] < NEGATIVE_BARRIER_TOL else None,
            # Which TS sets that barrier (no geometry: that is in route_ts.xyz).
            "route_ts": {k: v for k, v in result["route_ts"].items() if k != "xyz"} if result.get("route_ts") else None,
            # A solvent job: per-solvent barriers (for Explore's conditions),
            # never in barrier_kcal (see collect_solvent).
            "conditions": _conditions_summary(result.get("conditions")),
            "mechano": {"insight": next((i["text"] for i in result["mechano"].get("insights") or []
                                         if i["level"] in ("selectivity", "accelerates")), None)}
            if result.get("mechano") else None,
            "substituents": {"insight": next((i["text"] for i in result["substituents"].get("insights") or []
                                              if i["level"] in ("selectivity", "accelerates")), None)}
            if result.get("substituents") else None}


def entry_ts_energy(entry: dict) -> Optional[float]:
    """Energy (Hartree) of the TS behind an entry's barrier_kcal: its TS
    frame, or its only frame (a TS listed alone)."""
    frames = entry.get("frames") or []
    if entry.get("barrier_kcal") is None or not frames:
        return None
    k = entry.get("ts_index")
    if k is None and len(frames) == 1:
        k = 0
    return frames[k].get("energy_hartree") if k is not None and 0 <= k < len(frames) else None


def ts_energy_hartree(result: dict) -> Optional[float]:
    """Absolute energy of the TS behind a result's barrier_kcal, or None."""
    b, floor = result.get("barrier_kcal"), result.get("barrier_floor_hartree")
    return floor + b / HARTREE_TO_KCAL if b is not None and floor is not None else None


def _conditions_summary(cond: Optional[dict]) -> Optional[dict]:
    if not cond:
        return None
    gas = cond.get("gas") or {}
    return {"mode": cond.get("mode"), "temperature": cond.get("temperature"), "model": cond.get("model"),
            "gas_kcal": gas.get("barrier_kcal"),
            "gas_reaction_kcal": gas.get("reaction_kcal"),
            "solvents": {r["key"]: {"label": r["label"], "barrier_kcal": r.get("barrier_kcal"),
                                    "shift_kcal": r.get("shift_kcal"), "kind": r.get("kind"),
                                    "reaction_kcal": r.get("reaction_kcal")}
                         for r in cond.get("solvents") or []},
            "insight": next((i["text"] for i in cond.get("insights") or [] if i["level"] == "accelerates"), None)}


def find_entry(result: dict, entry_id: str) -> tuple[dict, dict]:
    """(group, entry) for an entry id."""
    for g in result.get("groups", []):
        for e in g["entries"]:
            if e["id"] == entry_id:
                return g, e
    raise KeyError(entry_id)


# Result groups whose single-frame entries are optimized minima.
MINIMA_KINDS = {"minima", "conformers"}


def _read_json(fp: Path):
    try:
        return json.loads(fp.read_text())
    except Exception:
        return None


def _trailing_int(p: Path) -> int:
    tail = p.name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0
