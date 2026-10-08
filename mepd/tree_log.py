"""What each node of an MSMEP split tree did, for looking back at a run.

Two records share one shape (`node_record`):

- `tree.json` next to `adj_matrix.txt`, written with the finished tree
  (`TreeNode.write_to_disk`): every node's parent, depth, outcome and why.
- A live copy while the search runs, when a viewer asked for one
  (MEPD_DRIVE_CHAIN_DIR, as `mepd web` sets): <dir>/trees/<stream>/ holds
  `nodes.jsonl` (one "started" and one "finished" line per node) and each
  finished node's path search as node_<key>.xyz plus node_<key>_history/,
  the same files `TreeNode.write_to_disk` writes. So a run that crashes, is
  cancelled or is still going can be inspected node by node. Keys are the
  run's own node numbers; a parallel run renumbers its tree depth-first
  only at the end, so its live keys can differ from the final node_<i>.

Recording never interrupts the search: every write swallows its errors.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

# leaf_status -> (outcome, why), for nodes that end without splitting.
_LEAF = {
    "identical_endpoints": ("skipped", "The two endpoints are the same species, so there is nothing to search."),
    "attempted_elsewhere": ("skipped", "This endpoint pair was already searched elsewhere in the tree."),
    "offtarget_split_rejected": ("rejected", "direct_only: none of the pieces of this split touch a queried species, "
                                             "so none was run."),
    "max_depth_reached": ("elementary", "The split limit (recursive depth) was reached, so the path is not split; "
                                        "it was optimized to convergence and treated as one step."),
    "same_pair_split_limit_reached": ("elementary", "The same endpoint pair kept splitting without finding anything "
                                                    "new, so it is not split again; the path was optimized to "
                                                    "convergence and treated as one step."),
    "time_budget": ("unresolved", "The search budget ran out; the path is kept without splitting it."),
    "cycle": ("elementary", "The endpoint pair repeats an ancestor's (a cycle), so it is not split again; the "
                            "path was optimized to convergence and treated as one step."),
    "electronic_structure_error": ("failed", "The electronic-structure calculation failed."),
    "path_minimization_error": ("failed", "The path search raised an error."),
    "worker_failure": ("failed", "The worker process running this branch failed."),
}

_SPLIT = {
    "minima": "An intermediate minimum was found along the path; the path was split at it.",
    "maxima": "The path has more than one energy maximum; it was split between them.",
}


def _converged(data) -> bool | None:
    explicit = getattr(data, "converged", None)
    if explicit is not None:
        return bool(explicit)
    trajectory = getattr(data, "chain_trajectory", None) or []
    chain = getattr(data, "optimized", None) or (trajectory[-1] if trajectory else None)
    nodes = list(getattr(chain, "nodes", []) or [])
    return all(bool(getattr(n, "converged", False)) for n in nodes) if nodes else None


def node_record(node, n_children: int | None = None) -> dict:
    """{outcome, why, leaf_status, error, traceback, n_steps, converged,
    split_criterion} for one TreeNode. `n_children` overrides
    len(node.children), for a node whose children are not attached yet."""
    data = getattr(node, "data", None)
    leaf = str(getattr(node, "leaf_status", "") or "")
    kids = len(getattr(node, "children", None) or []) if n_children is None else int(n_children)
    criterion = getattr(node, "split_criterion", None)
    rec = {
        "leaf_status": leaf or None,
        "n_steps": len(getattr(data, "chain_trajectory", None) or []) if data is not None else 0,
        "converged": _converged(data) if data is not None else None,
        "split_criterion": str(criterion) if criterion else None,
        "error": None,
        "traceback": None,
        "n_rejected_pieces": len(getattr(node, "rejected_chains", None) or []),
    }
    if getattr(node, "leaf_error", None):
        rec["error"] = f"{getattr(node, 'leaf_error_type', 'Error')}: {node.leaf_error}"
        rec["traceback"] = getattr(node, "leaf_traceback", None) or None
    if kids:
        rec["outcome"] = "split"
        rec["why"] = _SPLIT.get(str(criterion), f"Not an elementary step; split ({criterion})." if criterion
                                else "Not an elementary step; split.")
    elif leaf in _LEAF:
        rec["outcome"], rec["why"] = _LEAF[leaf]
    elif leaf:
        rec["outcome"], rec["why"] = ("failed" if rec["error"] else "unresolved"), leaf.replace("_", " ")
    elif data is None:
        rec["outcome"], rec["why"] = "skipped", "No path search was run for this node."
    else:
        rec["outcome"], rec["why"] = "elementary", "The path is one elementary step."
    if rec["n_rejected_pieces"] and rec["outcome"] != "rejected":
        rec["why"] += (f" direct_only: {rec['n_rejected_pieces']} piece(s) of the split were not run "
                       "(both ends are other species).")
    return rec


def tree_records(tree) -> list[dict]:
    """Every node of a finished tree, depth-first: node_record plus index,
    parent and depth."""
    out: list[dict] = []

    def walk(node, parent, depth):
        out.append({"index": int(node.index), "parent": parent, "depth": depth, **node_record(node)})
        for child in node.children:
            walk(child, int(node.index), depth + 1)

    walk(tree, None, 0)
    return out


def write_tree_json(tree, folder: Path) -> None:
    try:
        (Path(folder) / "tree.json").write_text(json.dumps({"version": 1, "nodes": tree_records(tree)}, indent=1))
    except Exception:
        return


# ------------------------------------------------------------------ live
_lock = threading.Lock()


def _live_dir() -> Path | None:
    directory = os.environ.get("MEPD_DRIVE_CHAIN_DIR", "").strip()
    if not directory:
        return None
    from mepd import progress

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in progress._live_stream) or "main"
    return Path(directory) / "trees" / safe


def _append(folder: Path, row: dict) -> None:
    with _lock, open(folder / "nodes.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({**row, "time": time.time()}) + "\n")


def _quietly_mkdir(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def _quietly(fn) -> None:
    try:
        fn()
    except Exception:
        return


def started(key: int, parent: int | None, depth: int, chain=None) -> None:
    """A node's path search begins. The root (no parent) starts a fresh
    record, so a resumed or repeated search doesn't mix with the last one."""
    folder = _live_dir()
    if folder is None:
        return
    try:
        if parent is None and folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
        _append(folder, {"event": "started", "key": int(key), "parent": parent, "depth": int(depth)})
    except Exception:
        return
    if chain is not None:
        _quietly(lambda: chain.write_to_disk(folder / f"node_{int(key)}_initial.xyz"))


def finished(key: int, node, n_children: int | None = None) -> None:
    """A node's path search ended: save it and what came of it."""
    folder = _live_dir()
    if folder is None:
        return
    if not _quietly_mkdir(folder):
        return
    # The files first, so a viewer that sees "finished" finds them.
    data = getattr(node, "data", None)
    if data is not None and getattr(data, "chain_trajectory", None):
        _quietly(lambda: data.write_to_disk(fp=folder / f"node_{int(key)}.xyz", write_history=True))
    elif getattr(node, "failed_chain", None) is not None:
        _quietly(lambda: node.failed_chain.write_to_disk(folder / f"node_{int(key)}_failed.xyz"))
    elif getattr(node, "rejected_chain", None) is not None:
        _quietly(lambda: node.rejected_chain.write_to_disk(folder / f"node_{int(key)}_rejected.xyz"))
    try:
        _append(folder, {"event": "finished", "key": int(key), **node_record(node, n_children)})
    except Exception:
        return
