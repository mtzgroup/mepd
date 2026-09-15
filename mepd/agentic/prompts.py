"""Builds the chat messages sent to an LLM backend proposing new mepd NEB
parameter values. Pure string/dict construction, no I/O or network -- kept
separate from `llm_client.py` so both are independently unit-testable.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mepd.agentic.llm_client import TuningContext

_SYSTEM_PROMPT = """You are tuning numeric parameters of a nudged-elastic-band (NEB) \
reaction-path optimizer (the mepd package) to make it converge more reliably and \
cheaply on a fixed benchmark of chemical reactions.

Each iteration you propose a *partial* set of new values for a curated list of \
tunable parameters (given below, grouped by input.toml section). Any parameter you \
omit keeps its current value. Return ONLY a JSON object matching the given schema -- \
no prose, no markdown code fences.

Guidance:
- Prefer small, targeted changes over large jumps; you get many iterations.
- Consider the recent history below: don't repeat a combination that already failed \
or scored worse than the current best, unless you have a specific reason to retry it.
- A higher fitness score is better. Success (every benchmark reaction converging \
without an elementary-step/error failure) dominates the score; wall-clock time and \
gradient-call count are secondary tie-breakers among equally-successful candidates.
"""


def _render_history(history: list[dict]) -> str:
    if not history:
        return "(no iterations yet)"
    lines = []
    for record in history:
        lines.append(
            f"- iteration {record.get('iteration')}: "
            f"fitness={record.get('fitness')!r}, "
            f"accepted={record.get('accepted')!r}, "
            f"changed={record.get('proposed_params')!r}"
        )
    return "\n".join(lines)


def build_messages(context: "TuningContext") -> list[dict[str, str]]:
    history_text = _render_history(context.recent_history)
    best_text = json.dumps(context.current_best_params, indent=2, sort_keys=True)
    seed_text = json.dumps(context.seed_params, indent=2, sort_keys=True)
    user_prompt = (
        f"Iteration {context.iteration_index + 1} of {context.max_iterations}.\n\n"
        f"Tunable parameters:\n{context.schema_text}\n\n"
        f"Seed (original) parameter values:\n{seed_text}\n\n"
        f"Current best parameter values "
        f"(fitness={context.current_best_fitness!r}):\n{best_text}\n\n"
        f"Recent iteration history (most recent last):\n{history_text}\n\n"
        "Propose new values for a subset of the tunable parameters."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
