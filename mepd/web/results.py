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


def _smiles(node) -> str:
    from mepd.web.chem import perceive_smiles

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

def collect_ts(out: Path, charge: int, multiplicity: int) -> dict:
    """`mepd run` output: mep_output.xyz, tree/, ts[_leaf_k].xyz (+ _irc),
    network_completion/ + network.json."""
    warnings: list[str] = []
    mep = _load_chain(out / "mep_output.xyz", charge, multiplicity)
    ts_items = _ts_dir_items(out, charge, multiplicity)
    ircs = [(label, irc) for label, _, irc in ts_items if irc is not None]

    # Floor: the path's reactant end, and any IRC end with the same
    # connectivity as it (an IRC often relaxes into a lower conformer).
    reactant = mep[0] if mep is not None else None
    floor_candidates = [_node_energy(reactant)] if reactant is not None else []
    for _, irc in ircs:
        for end in (irc[0], irc[-1]):
            if reactant is not None and _same_connectivity(end, reactant):
                floor_candidates.append(_node_energy(end))
    floor = _min(floor_candidates)

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

    ts_entries, off_route, irc_entries = [], [], []
    route = _verify_route(reactant, mep[-1] if mep is not None else None, ts_items, floor)
    for info in route["items"]:
        label, ts_node, irc, barrier = info["label"], info["ts"], info["irc"], info["barrier"]
        entry = _entry(label, label, [ts_node], floor, barrier=barrier, note=info["note"])
        (ts_entries if info["on_route"] else off_route).append(entry)
        if irc is not None:
            irc_entries.append(_entry(f"{label}_irc", f"{label} IRC", irc.nodes, floor, note=info["note"],
                                      barrier=barrier))
    groups.append(_group("Transition states on the start → end route", "ts", ts_entries))
    groups.append(_group("Other saddle points (do not connect start → end)", "ts_other", off_route))
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
                            "is unverified (see 'Other saddle points' for what the TS searches found instead).")
    else:
        barrier, source = None, ""
    n_steps = len(leaves) if leaves else (1 if mep is not None else 0)
    approx = "" if verified else "≈ "
    headline = (f"ΔE‡ {approx}{barrier:.1f} kcal/mol ({source})" if barrier is not None else "No path found") + \
        (f" · {n_steps} path steps" if n_steps > 1 else "")
    summary = [
        {"label": "Elementary steps", "value": n_steps or None},
        {"label": "Optimized TSs", "value": len(ts_items) or None},
        {"label": "Barrier", "value": "verified: IRCs connect start → end" if verified else "not verified by IRC"},
        {"label": "Barrier reference", "value": "lowest reactant-side energy (path start / IRC ends)"},
    ]
    result = _result(headline, groups, summary, barrier, warnings)
    result["barrier_verified"] = verified
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


def _verify_route(start, end, ts_items: list, floor: Optional[float]) -> dict:
    """Which optimized TSs actually lie on a route from `start` to `end`.

    Every IRC end and the two path ends are grouped by connectivity. A TS
    whose IRC ends are different species is a step between them; one whose
    ends are the same species is a conformer change. The verified barrier is
    the minimax over start->end routes of steps (a route is as good as its
    highest TS). Returns {"barrier", "n_steps", "items": [...]} with a note
    per TS saying what it connects."""
    items = []
    for label, ts_node, irc in ts_items:
        e = _node_energy(ts_node)
        barrier = (e - floor) * HARTREE_TO_KCAL if e is not None and floor is not None else None
        items.append({"label": label, "ts": ts_node, "irc": irc, "barrier": barrier, "on_route": False, "note": ""})
    if start is None or end is None:
        for it in items:
            it["note"] = _irc_note(it["irc"]) if it["irc"] is not None else "no IRC"
        return {"barrier": None, "n_steps": 0, "items": items}

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
            it["note"] = "no IRC was run, so what this TS connects is unknown"
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
    return {"barrier": bottleneck, "n_steps": n_steps, "items": items}


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
    ts_entries, irc_entries = [], []
    for label, ts_node, irc in ts_items:
        floor = _min(_node_energy(n) for n in (irc.nodes if irc is not None else []))
        e = _node_energy(ts_node)
        barrier = (e - floor) * HARTREE_TO_KCAL if e is not None and floor is not None else None
        ts_entries.append(_entry(label, label, [ts_node], floor if floor is not None else e,
                                 barrier=barrier, note=_irc_note(irc) if irc is not None else ""))
        if irc is not None:
            irc_entries.append(_entry(f"{label}_irc", f"{label} IRC", irc.nodes, floor, note=_irc_note(irc),
                                      barrier=barrier))
    headline = f"{len(ts_entries)} TS optimized" if ts_entries else "No TS converged"
    if ts_entries and irc_entries:
        headline += f" · {irc_entries[0]['note']}"
    return _result(headline, [_group("Transition states", "ts", ts_entries),
                              _group("IRC paths", "irc", irc_entries)],
                   [{"label": "Barrier reference", "value": "lower IRC end"}],
                   ts_entries[0]["barrier_kcal"] if len(ts_entries) == 1 else None)


def collect_channels(out: Path, charge: int, multiplicity: int) -> dict:
    from mepd.cli import _irc_needs_reversal

    warnings: list[str] = []
    stats = _read_json(out / "stats.json") or {}
    start_pool = _load_chain(out / "conformers" / "start.xyz", charge, multiplicity)
    end_pool = _load_chain(out / "conformers" / "end.xyz", charge, multiplicity)
    ref_start = start_pool[0] if start_pool is not None else None
    ref_end = end_pool[0] if end_pool is not None else None

    def oriented_irc(folder: Path):
        irc = _load_chain(folder / "irc.xyz", charge, multiplicity)
        if irc is not None and ref_start is not None:
            try:
                if _irc_needs_reversal(irc, ref_start, ref_end):
                    irc = irc.copy()
                    irc.nodes.reverse()
            except Exception:
                pass
        return irc

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
    floor = _min(floor_candidates)

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
        {"label": "Mechanisms", "value": stats.get("n_mechanisms")},
        {"label": "Path searches", "value": stats.get("n_path_searches")},
        {"label": "Distinct mechanisms", "value": ", ".join(sorted({m["mechanism"] for m in mechanisms if m.get("mechanism")})) or None},
        {"label": "Wall time", "value": f"{stats['total_seconds']:.0f} s" if stats.get("total_seconds") else None},
        {"label": "Barrier reference", "value": "lowest reactant conformer / reactant-side IRC end"},
    ]
    result = _result(headline, groups, summary, best, warnings)
    lowest = [i for i in channels if i.get("barrier") is not None and best is not None and abs(i["barrier"] - best) < 1e-6]
    if lowest:
        result["route_ts"] = _route_ts(lowest[0]["title"], lowest[0]["ts"], lowest[0]["barrier"])
    return result


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
    return result


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
    stats = s.get("stats") or {}
    return _result(f"{len(entries)} conformer(s) · {s.get('backend', '?')}"
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


COLLECTORS = {
    "optimize": collect_optimize,
    "conformers": collect_conformers,
    "ts": collect_ts,
    "channels": collect_channels,
    "tsopt": collect_tsopt,
    "hessian-sample": collect_hessian_sample,
    "hessian-global": collect_hessian_global,
    "graph-enumeration": collect_graph_enumeration,
    "network-splits": collect_network_splits,
    "vri": collect_vri,
    # Follow-ups add to the VRI folder: they show the same, updated, result.
    "vri-check": collect_vri,
    "vri-surface": collect_vri,
}


def collect(job: dict, job_dir: Optional[Path] = None) -> dict:
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
    result = fn(out, int(job.get("charge") or 0), int(job.get("multiplicity") or 1))
    if not job.get("external"):
        result["warnings"] += _log_warnings(out.parent / "stdout.log")
    return result


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
RESULT_VERSION = 11


def collect_cached(job: dict, job_dir: Path) -> dict:
    """Finished jobs are parsed once and cached; running jobs are parsed
    fresh (their output grows), so partial results are always current."""
    cache = job_dir / "result.json"
    terminal = job["status"] in ("done", "failed", "cancelled", "interrupted")
    if terminal and cache.exists():
        try:
            cached = json.loads(cache.read_text())
            if cached.get("_finished") == job.get("finished") and cached.get("_version") == RESULT_VERSION:
                return cached
        except Exception:
            pass
    result = collect(job, job_dir)
    route_ts = result.get("route_ts")
    if route_ts and terminal:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "route_ts.xyz").write_text(route_ts["xyz"])
    if terminal:
        result["_finished"] = job.get("finished")
        result["_version"] = RESULT_VERSION
        job_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result))
    return result


def summarize(result: dict) -> dict:
    """The bit of a result stored on the job record (list views, edge badges)."""
    counts = {g["kind"]: len(g["entries"]) for g in result.get("groups", [])}
    return {"headline": result.get("headline"), "barrier_kcal": result.get("barrier_kcal"),
            # False: the barrier is not backed by IRCs connecting the job's
            # two ends (edges then show it as unconfirmed).
            "barrier_verified": result.get("barrier_verified", True), "counts": counts,
            # Which TS sets that barrier (no geometry: that is in route_ts.xyz).
            "route_ts": {k: v for k, v in result["route_ts"].items() if k != "xyz"} if result.get("route_ts") else None}


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
