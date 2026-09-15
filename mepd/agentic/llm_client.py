"""LLM backend abstraction for `mepd agentic tune`: proposes new values for a
curated set of tunable NEB parameters (see `mepd.agentic.schema`), given the
seed parameters, the current best, and recent iteration history.

Backend-agnostic by design -- `LLMBackend.propose_params` is the whole
contract. `OllamaBackend` is the only network-backed implementation for v0:
it talks to a local or Ollama-Cloud-proxied Ollama server over one HTTP API,
so it covers both "local" and "free cloud" model needs from a single
backend. Adding a Gemini-direct or Claude-API backend later is a new
subclass registered in `BACKENDS` -- same `TuningContext -> ParamProposal`
contract, nothing else changes.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mepd.agentic import prompts


@dataclass
class TuningContext:
    """Everything a backend needs to propose the next parameter set."""

    schema_text: str
    json_schema: dict
    seed_params: dict
    current_best_params: dict
    current_best_fitness: float | None
    recent_history: list[dict] = field(default_factory=list)
    iteration_index: int = 0
    max_iterations: int = 1


@dataclass
class ParamProposal:
    """A backend's response. `values` is a *partial* set of new parameter
    values -- nested or flat dotted-path, `schema.clamp_and_validate`
    accepts either -- for only the parameters the backend chose to change.
    `valid=False` means the raw response couldn't be parsed/wasn't usable
    (a network/HTTP failure is a separate case -- see
    `LLMConnectionError`)."""

    values: dict[str, Any]
    raw_response: str = ""
    rationale: str | None = None
    valid: bool = True
    error: str | None = None


class LLMConnectionError(Exception):
    """Raised when the backend itself couldn't be reached (network/HTTP
    failure, timeout) -- distinct from a malformed-but-received response,
    which is reported as an invalid `ParamProposal` instead. Not caught by
    `mepd.agentic.loop`: tuning can't proceed without the LLM, so this is a
    hard stop, not a skip-and-continue case."""


class LLMBackend(ABC):
    @abstractmethod
    def propose_params(self, context: TuningContext) -> ParamProposal:
        """Propose new values for a subset of the tunable parameters."""


class NullBackend(LLMBackend):
    """Always proposes no change. A trivial baseline / test double -- no
    network, fully deterministic."""

    def propose_params(self, context: TuningContext) -> ParamProposal:
        return ParamProposal(values={}, rationale="null backend: no-op")


class ScriptedBackend(LLMBackend):
    """Replays a fixed sequence of `ParamProposal`s -- the seam
    `mepd.agentic.loop` is unit-tested against, with no network involved.
    Raises `IndexError` if asked for more proposals than were scripted (a
    test-writing mistake, not something to swallow silently)."""

    def __init__(self, proposals: list[ParamProposal]):
        self._proposals = list(proposals)
        self._index = 0

    def propose_params(self, context: TuningContext) -> ParamProposal:
        proposal = self._proposals[self._index]
        self._index += 1
        return proposal


class OllamaBackend(LLMBackend):
    """Talks to a local or Ollama-Cloud-proxied Ollama server's `/api/chat`
    endpoint, constraining the response to the tunable-parameter JSON
    schema via Ollama's `format=` structured-output support.

    `httpx` is only imported here (not at module top level), matching the
    friendly-`ImportError`-behind-an-optional-extra convention already used
    for the `ase`/`gxtb` engines in `mepd/inputs.py`.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        temperature: float = 0.4,
        timeout_s: float = 120.0,
    ):
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OllamaBackend requires the 'agentic' extra "
                "(pip install mepd[agentic])."
            ) from exc
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OLLAMA_API_BASE") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s

    def propose_params(self, context: TuningContext) -> ParamProposal:
        messages = prompts.build_messages(context)
        try:
            content = self._chat(messages, context.json_schema)
        except Exception as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self.base_url} "
                f"(model={self.model!r}): {type(exc).__name__}: {exc}"
            ) from exc

        proposal = self._parse_response(content)
        if proposal.valid:
            return proposal

        # One corrective retry: tell the model exactly what was wrong.
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    f"Your previous response was invalid: {proposal.error}. "
                    "Return JSON only, matching the given schema."
                ),
            },
        ]
        try:
            retry_content = self._chat(retry_messages, context.json_schema)
        except Exception as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self.base_url} "
                f"(model={self.model!r}) on retry: {type(exc).__name__}: {exc}"
            ) from exc
        retry_proposal = self._parse_response(retry_content)
        if retry_proposal.valid:
            return retry_proposal
        # Still invalid after a retry: no-op rather than crash the loop --
        # matches the "warning, not fatal" style used elsewhere in mepd's
        # CLI (e.g. --use-tsopt/--greedy-tsopt failures).
        return ParamProposal(
            values={}, raw_response=retry_content, valid=False, error=retry_proposal.error,
        )

    def _chat(self, messages: list[dict[str, str]], json_schema: dict) -> str:
        import httpx

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "format": json_schema,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["message"]["content"]

    @staticmethod
    def _parse_response(content: str) -> ParamProposal:
        try:
            values = json.loads(content)
        except json.JSONDecodeError as exc:
            return ParamProposal(
                values={}, raw_response=content, valid=False,
                error=f"response was not valid JSON: {exc}",
            )
        if not isinstance(values, dict):
            return ParamProposal(
                values={}, raw_response=content, valid=False,
                error=f"expected a JSON object, got {type(values).__name__}",
            )
        return ParamProposal(values=values, raw_response=content, valid=True)


BACKENDS: dict[str, type[LLMBackend]] = {
    "ollama": OllamaBackend,
    "null": NullBackend,
}


def get_backend(name: str, **kwargs: Any) -> LLMBackend:
    try:
        backend_cls = BACKENDS[name]
    except KeyError:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"Unknown agentic-tuning backend {name!r}. Available: {available}."
        ) from None
    # Backends other than OllamaBackend take no constructor kwargs; drop
    # anything that isn't theirs rather than erroring on an unexpected kwarg.
    if backend_cls is OllamaBackend:
        return backend_cls(**kwargs)
    return backend_cls()
