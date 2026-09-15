from __future__ import annotations

import json

import httpx
import pytest

from mepd.agentic.llm_client import (
    LLMConnectionError,
    NullBackend,
    OllamaBackend,
    ParamProposal,
    ScriptedBackend,
    TuningContext,
    get_backend,
)


def _context(**overrides) -> TuningContext:
    defaults = dict(
        schema_text="path | type | bounds | description",
        json_schema={"type": "object", "properties": {}},
        seed_params={"gi_inputs.nimages": 12},
        current_best_params={"gi_inputs.nimages": 12},
        current_best_fitness=None,
        recent_history=[],
        iteration_index=0,
        max_iterations=3,
    )
    defaults.update(overrides)
    return TuningContext(**defaults)


def test_null_backend_always_proposes_no_change():
    backend = NullBackend()
    proposal = backend.propose_params(_context())
    assert proposal.values == {}
    assert proposal.valid is True


def test_scripted_backend_replays_in_order_and_raises_when_exhausted():
    proposals = [
        ParamProposal(values={"gi_inputs.nimages": 14}),
        ParamProposal(values={"gi_inputs.nimages": 16}),
    ]
    backend = ScriptedBackend(proposals)
    assert backend.propose_params(_context()).values == {"gi_inputs.nimages": 14}
    assert backend.propose_params(_context()).values == {"gi_inputs.nimages": 16}
    with pytest.raises(IndexError):
        backend.propose_params(_context())


def test_get_backend_factory_builds_null_and_ollama():
    assert isinstance(get_backend("null"), NullBackend)
    ollama = get_backend("ollama", model="qwen3.5:latest", base_url="http://example.invalid")
    assert isinstance(ollama, OllamaBackend)
    assert ollama.model == "qwen3.5:latest"
    assert ollama.base_url == "http://example.invalid"


def test_get_backend_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown agentic-tuning backend"):
        get_backend("does-not-exist")


class _FakeResponse:
    def __init__(self, content: str, status_code: int = 200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self) -> dict:
        return {"message": {"content": self._content}}


def test_ollama_backend_parses_valid_json_response(monkeypatch):
    backend = OllamaBackend(model="qwen3.5:latest", base_url="http://fake")

    calls = []

    def fake_post(url, *, json, timeout):
        calls.append(json)
        return _FakeResponse(json_dumps_stub)

    json_dumps_stub = json.dumps({"gi_inputs": {"nimages": 15}})
    monkeypatch.setattr("httpx.post", fake_post)

    proposal = backend.propose_params(_context())
    assert proposal.valid is True
    assert proposal.values == {"gi_inputs": {"nimages": 15}}
    assert len(calls) == 1
    assert calls[0]["model"] == "qwen3.5:latest"
    assert calls[0]["format"] == _context().json_schema


def test_ollama_backend_retries_once_then_gives_up_on_malformed_json(monkeypatch):
    backend = OllamaBackend(model="qwen3.5:latest", base_url="http://fake")

    responses = iter(["not json", "still not json"])

    def fake_post(url, *, json, timeout):
        return _FakeResponse(next(responses))

    monkeypatch.setattr("httpx.post", fake_post)

    proposal = backend.propose_params(_context())
    assert proposal.valid is False
    assert proposal.values == {}
    assert proposal.error is not None


def test_ollama_backend_retry_can_succeed(monkeypatch):
    backend = OllamaBackend(model="qwen3.5:latest", base_url="http://fake")

    responses = iter(["not json", json.dumps({"chain_inputs": {"k": 0.2}})])

    def fake_post(url, *, json, timeout):
        return _FakeResponse(next(responses))

    monkeypatch.setattr("httpx.post", fake_post)

    proposal = backend.propose_params(_context())
    assert proposal.valid is True
    assert proposal.values == {"chain_inputs": {"k": 0.2}}


def test_ollama_backend_network_failure_raises_llm_connection_error(monkeypatch):
    backend = OllamaBackend(model="qwen3.5:latest", base_url="http://fake")

    def fake_post(url, *, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("httpx.post", fake_post)

    with pytest.raises(LLMConnectionError):
        backend.propose_params(_context())
