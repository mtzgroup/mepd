"""Core transition-state optimization + IRC helpers.

Engine-agnostic and free of CLI/file-I/O concerns, so it can be reused both
by the `mepd` CLI (`cli.py`'s `_optimize_ts_and_irc`, which adds file-writing
and progress echoing on top) and by library/notebook callers such as
`TreeNode.greedy_tsopt`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from mepd.chain import Chain
from mepd.nodes.node import StructureNode


@dataclass
class TSOptResult:
    """Outcome of a single TS-opt (+ optional IRC) attempt.

    Never raises: a failure at any stage is recorded in `error` rather than
    propagated, so a caller iterating many candidates (e.g. greedy tsopt)
    doesn't need to wrap each attempt in its own try/except.
    """

    ts_node: Optional[StructureNode] = None
    irc_chain: Optional[Chain] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.ts_node is not None


def optimize_ts_and_irc(
    ts_guess_node: StructureNode,
    engine: Any,
    *,
    run_irc: bool = False,
) -> TSOptResult:
    """Optimize `ts_guess_node` with `engine.compute_transition_state`, then
    (if `run_irc`) follow up with `engine.compute_irc_chain` -- falling back
    to `mepd.irc.compute_irc_chain_with_geometric` if the engine doesn't
    implement IRC natively.
    """
    compute_ts = getattr(engine, "compute_transition_state", None)
    if not callable(compute_ts):
        return TSOptResult(
            error=f"Engine {type(engine).__name__} does not support "
            "transition-state optimization."
        )

    try:
        result = compute_ts(node=ts_guess_node)
    except Exception as exc:
        return TSOptResult(
            error=f"Transition-state optimization failed: {type(exc).__name__}: {exc}"
        )

    if not isinstance(result, StructureNode):
        return TSOptResult(
            error="Transition-state optimization did not converge to a usable "
            f"structure (engine returned {type(result).__name__})."
        )

    ts_node = result
    if not run_irc:
        return TSOptResult(ts_node=ts_node)

    irc_fn = getattr(engine, "compute_irc_chain", None)
    try:
        if callable(irc_fn):
            irc_chain = irc_fn(ts_node)
        else:
            from mepd.irc import compute_irc_chain_with_geometric

            irc_chain = compute_irc_chain_with_geometric(engine, ts_node)
    except Exception as exc:
        return TSOptResult(
            ts_node=ts_node,
            error=f"IRC computation failed: {type(exc).__name__}: {exc}",
        )

    return TSOptResult(ts_node=ts_node, irc_chain=irc_chain)
