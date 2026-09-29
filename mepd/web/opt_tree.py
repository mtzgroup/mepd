"""The optimization tree of an MSMEP run, for the web UI: which path
searches ran, how they relate (a split makes children), what each one's
optimization did step by step, and why each branch ended.

Sources, per path search ("stream": `main` for `mepd run`, `pair_i_j` for
each `mepd channels` pair):

- the finished tree, <output>/tree or <output>/pairs/<stream>/tree:
  adj_matrix.txt, node_<i>.xyz + node_<i>_history/traj_<k>.xyz, and
  tree.json (mepd/tree_log.py) with each node's outcome. Older runs have no
  tree.json; outcomes are then inferred from which files exist.
- the live record, <job>/live/trees/<stream>/ (mepd/tree_log.py), while the
  search runs or when it died before writing its tree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HARTREE_TO_KCAL = 627.509474
STREAM = re.compile(r"^(main|pair_\d+_\d+|[A-Za-z0-9_.-]{1,80})$")


# ------------------------------------------------------------ files
def _read_xyz_frames(fp: Path) -> list[str]:
    """Every frame of a multi-frame xyz, as text."""
    lines = fp.read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        head = lines[i].strip()
        if not head:
            i += 1
            continue
        n = int(head)
        frames.append("\n".join(lines[i:i + n + 2]) + "\n")
        i += n + 2
    return frames


def _coords(frame: str) -> np.ndarray:
    rows = frame.splitlines()[2:]
    return np.array([[float(v) for v in r.split()[1:4]] for r in rows if r.strip()])


def _energies(fp: Path) -> list[float] | None:
    e = fp.with_suffix(".energies")
    if not e.exists():
        return None
    try:
        return [float(v) for v in e.read_text().split()]
    except ValueError:
        return None


def _gradients(fp: Path) -> np.ndarray | None:
    g, shape = fp.with_suffix(".gradients"), fp.parent / f"{fp.stem}_grad_shapes.txt"
    if not g.exists() or not shape.exists():
        return None
    try:
        dims = [int(float(v)) for v in shape.read_text().split()]
        return np.loadtxt(g).reshape(dims[0], -1)
    except Exception:
        return None


def _history_steps(folder: Path) -> list[Path]:
    return sorted(folder.glob("traj_*.xyz"), key=lambda p: int(p.stem.split("_")[1]))


def _barrier(fp: Path) -> float | None:
    e = _energies(fp)
    return (max(e) - e[0]) * HARTREE_TO_KCAL if e and len(e) > 1 else None


def _reaction(fp: Path) -> float | None:
    e = _energies(fp)
    return (e[-1] - e[0]) * HARTREE_TO_KCAL if e and len(e) > 1 else None


# ------------------------------------------------------------ finding trees
def _final_dir(out: Path, stream: str) -> Path:
    return out / "tree" if stream == "main" else out / "pairs" / stream / "tree"


def find_trees(job_dirs: list[Path], out: Path) -> dict[str, dict]:
    """stream -> {final: dir|None, live: dir|None}."""
    found: dict[str, dict] = {}
    if (out / "tree" / "adj_matrix.txt").exists():
        found.setdefault("main", {})["final"] = out / "tree"
    for d in sorted((out / "pairs").glob("pair_*/tree")) if (out / "pairs").is_dir() else []:
        if (d / "adj_matrix.txt").exists():
            found.setdefault(d.parent.name, {})["final"] = d
    for jd in job_dirs:
        for d in sorted((jd / "live" / "trees").glob("*")) if (jd / "live" / "trees").is_dir() else []:
            if (d / "nodes.jsonl").exists():
                prev = found.setdefault(d.name, {}).get("live")
                if prev is None or (d / "nodes.jsonl").stat().st_mtime > (prev / "nodes.jsonl").stat().st_mtime:
                    found[d.name]["live"] = d
    return {k: {"final": v.get("final"), "live": v.get("live")} for k, v in found.items()}


def _use_live(src: dict) -> bool:
    """The live record, when there is no finished tree or it is newer (a
    resumed search that has not finished again)."""
    live, final = src.get("live"), src.get("final")
    if live is None:
        return False
    if final is None:
        return True
    return (live / "nodes.jsonl").stat().st_mtime > (final / "adj_matrix.txt").stat().st_mtime + 1


# ------------------------------------------------------------ the tree
def _final_tree(d: Path) -> list[dict]:
    adj = np.atleast_2d(np.loadtxt(d / "adj_matrix.txt"))
    n = adj.shape[0]
    parent = {}
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j]:
                parent[j] = i
    meta = {}
    if (d / "tree.json").exists():
        try:
            meta = {r["index"]: r for r in json.loads((d / "tree.json").read_text()).get("nodes", [])}
        except Exception:
            meta = {}
    reachable = {0} | set(parent) | set(meta)
    nodes = []
    for i in sorted(reachable):
        depth, p = 0, parent.get(i)
        while p is not None and depth < n:
            depth, p = depth + 1, parent.get(p)
        m = meta.get(i)
        xyz = d / f"node_{i}.xyz"
        kids = [j for j, p in parent.items() if p == i]
        rec = {"key": i, "parent": parent.get(i, m.get("parent") if m else None), "depth": m["depth"] if m else depth}
        if m:
            rec.update({k: m.get(k) for k in ("outcome", "why", "error", "traceback", "converged", "split_criterion",
                                              "leaf_status", "n_rejected_pieces")})
        else:
            failed = d / f"node_{i}_failed.xyz"
            txt = d / f"node_{i}_failed.txt"
            if kids:
                rec.update(outcome="split", why="Not an elementary step; split.")
            elif xyz.exists():
                rec.update(outcome="elementary", why="The path is one elementary step (or ended unresolved: this run "
                                                     "predates the record of why).")
            elif failed.exists() or txt.exists():
                err = txt.read_text() if txt.exists() else ""
                rec.update(outcome="failed", why="This branch failed.", error=err.split("\n\n")[0] or None,
                           traceback=err.split("\n\n", 1)[1] if "\n\n" in err else None)
            elif (d / f"node_{i}_rejected.xyz").exists():
                rec.update(outcome="rejected", why="direct_only: none of the pieces of this split touch a queried "
                                                   "species, so none was run.")
            else:
                rec.update(outcome="skipped", why="No path search was run for this node (e.g. identical endpoints).")
        hist = d / f"node_{i}_history"
        rec["n_steps"] = len(_history_steps(hist)) if hist.is_dir() else 0
        rec["has_path"] = xyz.exists() or (d / f"node_{i}_failed.xyz").exists() or (d / f"node_{i}_rejected.xyz").exists()
        rec["barrier_kcal"] = _barrier(xyz) if xyz.exists() else None
        rec["reaction_kcal"] = _reaction(xyz) if xyz.exists() else None
        rec["running"] = False
        nodes.append(rec)
    return nodes


def _live_tree(d: Path) -> tuple[list[dict], bool]:
    rows = []
    for line in (d / "nodes.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a line being written
    nodes: dict[int, dict] = {}
    for r in rows:
        k = r["key"]
        if r["event"] == "started":
            nodes[k] = {"key": k, "parent": r.get("parent"), "depth": r.get("depth", 0), "outcome": "running",
                        "why": "This path search is running.", "running": True, "started": r.get("time")}
        elif r["event"] == "finished":
            rec = nodes.setdefault(k, {"key": k, "parent": None, "depth": 0})
            rec.update({x: r.get(x) for x in ("outcome", "why", "error", "traceback", "converged", "split_criterion",
                                              "leaf_status", "n_steps", "n_rejected_pieces")})
            rec.update(running=False, finished=r.get("time"))
    for k, rec in nodes.items():
        xyz = d / f"node_{k}.xyz"
        hist = d / f"node_{k}_history"
        rec["n_steps"] = len(_history_steps(hist)) if hist.is_dir() else rec.get("n_steps") or 0
        rec["has_path"] = any((d / f"node_{k}{s}.xyz").exists() for s in ("", "_failed", "_rejected", "_initial"))
        rec["barrier_kcal"] = _barrier(xyz) if xyz.exists() else None
        rec["reaction_kcal"] = _reaction(xyz) if xyz.exists() else None
    return sorted(nodes.values(), key=lambda r: r["key"]), any(r.get("running") for r in nodes.values())


def list_trees(job_dirs: list[Path], out: Path, job_running: bool) -> list[dict]:
    items = []
    for stream, src in sorted(find_trees(job_dirs, out).items(), key=lambda kv: (kv[0] != "main", _natural(kv[0]))):
        try:
            tree = load_tree(job_dirs, out, stream, job_running, src)
        except Exception as exc:  # a half-written file: list it, say why
            items.append({"id": stream, "label": _label(stream), "error": f"{type(exc).__name__}: {exc}"})
            continue
        counts: dict[str, int] = {}
        for n in tree["nodes"]:
            counts[n["outcome"]] = counts.get(n["outcome"], 0) + 1
        items.append({"id": stream, "label": _label(stream), "live": tree["live"], "running": tree["running"],
                      "n_nodes": len(tree["nodes"]), "counts": counts})
    return items


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _label(stream: str) -> str:
    m = re.fullmatch(r"pair_(\d+)_(\d+)", stream)
    return f"Pair {m.group(1)} → {m.group(2)}" if m else ("Path search" if stream == "main" else stream)


def load_tree(job_dirs: list[Path], out: Path, stream: str, job_running: bool, src: dict | None = None) -> dict:
    src = src or find_trees(job_dirs, out).get(stream)
    if not src:
        raise KeyError(stream)
    if _use_live(src):
        nodes, running = _live_tree(src["live"])
        if not job_running:
            # The run is over but these never finished: it died mid-search.
            for n in nodes:
                if n.get("running"):
                    n.update(running=False, outcome="interrupted",
                             why="This path search never finished: the run stopped (crashed or was cancelled) "
                                 "while it was going.")
            running = False
        return {"id": stream, "label": _label(stream), "live": True, "running": running, "nodes": nodes}
    return {"id": stream, "label": _label(stream), "live": False, "running": False, "nodes": _final_tree(src["final"])}


# ------------------------------------------------------------ one node
def _node_files(job_dirs: list[Path], out: Path, stream: str, key: int, job_running: bool) -> tuple[Path, Path]:
    src = find_trees(job_dirs, out).get(stream)
    if not src:
        raise KeyError(stream)
    d = src["live"] if _use_live(src) else src["final"]
    return d, d / f"node_{int(key)}"


def _step_summary(fp: Path, e_ref: float | None) -> dict:
    frames = _read_xyz_frames(fp)
    e = _energies(fp)
    X = np.array([_coords(f) for f in frames])
    seg = np.sqrt(((X[1:] - X[:-1]) ** 2).sum(axis=(1, 2))) if len(X) > 1 else np.array([])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s = (s / s[-1]).tolist() if len(s) > 1 and s[-1] > 0 else [i / max(1, len(X) - 1) for i in range(len(X))]
    g = _gradients(fp)
    rel = [(v - e_ref) * HARTREE_TO_KCAL for v in e] if e and e_ref is not None else None
    ts = int(np.argmax(e)) if e else None
    out = {"s": s, "e": rel, "ts": ts, "n": len(frames)}
    if g is not None and len(g) == len(frames) and len(frames) > 2:
        # |gradient| per image in Hartree/Bohr (interior images: the endpoints are held fixed).
        norms = np.sqrt((g ** 2).mean(axis=1))
        out["grad_rms"] = norms.tolist()
        out["ts_grad_rms"] = float(norms[ts]) if ts is not None else None
    return out, X


def node_detail(job_dirs: list[Path], out: Path, stream: str, key: int, job_running: bool,
                step: int | None = None) -> dict:
    """The node's optimization, step by step: every step's energy profile
    (kcal/mol relative to the reactant end of the last step) and image
    positions (normalized path length), plus how much the path moved and the
    barrier per step. Geometries only for `step` (default: the last)."""
    d, base = _node_files(job_dirs, out, stream, key, job_running)
    hist = Path(f"{base}_history")
    files = _history_steps(hist) if hist.is_dir() else []
    kind = "history"
    if not files:
        for suffix, k in (("", "final"), ("_failed", "failed"), ("_rejected", "rejected"), ("_initial", "initial")):
            fp = Path(f"{base}{suffix}.xyz")
            if fp.exists():
                files, kind = [fp], k
                break
    if not files:
        return {"key": key, "kind": "none", "steps": [], "frames": [], "step": None}
    last_e = _energies(files[-1])
    e_ref = last_e[0] if last_e else None
    steps, prev, moved = [], None, []
    for fp in files:
        summ, X = _step_summary(fp, e_ref)
        moved.append(float(np.sqrt(((X - prev) ** 2).sum(axis=2).mean())) if prev is not None and prev.shape == X.shape
                     else None)
        prev = X
        summ["barrier"] = (max(summ["e"]) - summ["e"][0]) if summ["e"] else None
        summ["moved"] = moved[-1]
        steps.append(summ)
    k = len(files) - 1 if step is None else max(0, min(int(step), len(files) - 1))
    return {"key": key, "kind": kind, "steps": steps, "step": k, "frames": _read_xyz_frames(files[k])}


def running_node(job_dir: Path, stream: str, key: int) -> dict | None:
    """The chain a running node is relaxing right now, from the live
    stream: {frames, e, caption}."""
    fp = job_dir / "live" / f"{stream}.json"
    if not fp.exists():
        return None
    try:
        data = json.loads(fp.read_text())
    except ValueError:
        return None
    monitors = data.get("monitors") or {}
    mon = monitors.get(f"branch-{key}") or data
    geo, plot = mon.get("geometry") or {}, mon.get("plot") or {}
    if not geo.get("frames"):
        return None
    return {"frames": geo["frames"], "ts": geo.get("ts_index"), "s": plot.get("x"), "e": plot.get("y"),
            "caption": mon.get("caption") or plot.get("caption"), "status": mon.get("status_message")}
