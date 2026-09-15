"""Curated, safety-bounded schema of mepd `input.toml` numeric knobs that
`mepd agentic tune` is allowed to propose new values for.

Deliberately hand-maintained rather than derived by introspecting
`RunInputs`/`NEBInputs`/`ChainInputs`/`GIInputs`: `path_min_inputs` is a
`SimpleNamespace` whose actual shape depends on `path_min_method` (NEB vs
FNEB vs NEB-DLF vs GEOMETRIC-NEB vs GSM each define a different default-kwds
dict in `RunInputs.__post_init__`), and safety here needs an explicit
allowlist -- never `engine_name`, `program`, `program_kwds`, executable
paths, `charge`/`multiplicity` -- rather than introspect-then-blocklist.

Every function here is pure dict-in/dict-out: it never touches a live
`RunInputs` instance. `RunInputs.open()`/`__post_init__` is stateful (it
constructs live `engine`/`optimizer` objects), so building a modified
config means editing a plain dict (the shape `RunInputs.to_dict()`
produces) and constructing a *fresh* `RunInputs(**dict)` -- never mutating
an existing instance in place.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["int", "float", "bool", "categorical"]

_MISSING = object()


@dataclass(frozen=True)
class ParamSpec:
    """One tunable knob. `path` is a dotted path into the dict shape
    `RunInputs.to_dict()` produces, e.g. `"gi_inputs.nimages"`."""

    path: str
    kind: Kind
    description: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    choices: tuple[Any, ...] | None = None
    log_scale: bool = False


def _knobs(*specs: ParamSpec) -> dict[str, ParamSpec]:
    return {spec.path: spec for spec in specs}


# v0 covers path_min_method="NEB" only -- the seed `examples/gxtb.toml` uses
# it, and FNEB/NEB-DLF/GEOMETRIC-NEB/GSM each have differently-shaped
# `path_min_inputs` that would need their own curated knob sets (not yet
# written -- `get_schema` raises a clear error for them). Optimizer
# name/kwds are intentionally excluded: each optimizer in `mepd.inputs`'s
# `optimizer_map` (cg/vpo/lbfgs/adam/fire/...) takes a different kwargs
# shape, so safely mutating it needs a per-optimizer sub-schema -- a v1
# concern, not v0.
SAFE_KNOBS_BY_METHOD: dict[str, dict[str, ParamSpec]] = {
    "NEB": _knobs(
        ParamSpec(
            "gi_inputs.nimages", "int",
            "Number of images in the geodesic-interpolated initial path.",
            minimum=6, maximum=30,
        ),
        ParamSpec(
            "gi_inputs.friction", "float",
            "Geodesic-interpolation friction: penalty for pairwise "
            "distances growing too large.",
            minimum=1e-4, maximum=1.0, log_scale=True,
        ),
        ParamSpec(
            "gi_inputs.nudge", "float",
            "Geodesic-interpolation nudge magnitude.",
            minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            "chain_inputs.k", "float",
            "Maximum NEB spring constant.",
            minimum=0.001, maximum=1.0, log_scale=True,
        ),
        ParamSpec(
            "chain_inputs.delta_k", "float",
            "Energy-weighted spring-constant parameter (chain_inputs.k).",
            minimum=0.0, maximum=1.0, log_scale=True,
        ),
        ParamSpec(
            "path_min_inputs.max_steps", "int",
            "Maximum NEB optimizer steps before giving up.",
            minimum=50, maximum=3000,
        ),
        ParamSpec(
            "path_min_inputs.rms_grad_thre", "float",
            "RMS perpendicular-gradient convergence threshold "
            "(Hartree/Bohr).",
            minimum=1e-4, maximum=0.1, log_scale=True,
        ),
        ParamSpec(
            "path_min_inputs.max_rms_grad_thre", "float",
            "Max(RMS perpendicular gradient) convergence threshold "
            "(Hartree/Bohr).",
            minimum=1e-4, maximum=0.2, log_scale=True,
        ),
        ParamSpec(
            "path_min_inputs.en_thre", "float",
            "Energy-difference convergence threshold (Hartree).",
            minimum=1e-6, maximum=0.01, log_scale=True,
        ),
        ParamSpec(
            "path_min_inputs.negative_steps_thre", "int",
            "Steps a chain can oscillate before its step size is halved.",
            minimum=1, maximum=200000,
        ),
        ParamSpec(
            "path_min_inputs.positive_steps_thre", "int",
            "Stable steps before the step size is increased.",
            minimum=1, maximum=200000,
        ),
        ParamSpec(
            "path_min_inputs.climb", "bool",
            "Whether to use climbing-image NEB.",
        ),
        ParamSpec(
            "path_min_inputs.early_stop_force_thre", "float",
            "Elementary-step early-stop |g_perp| threshold (Hartree/Bohr).",
            minimum=0.0, maximum=0.1,
        ),
        ParamSpec(
            "path_min_inputs.barrier_thre", "float",
            "Barrier-height threshold (kcal/mol) used in elementary-step "
            "checks.",
            minimum=0.01, maximum=20.0, log_scale=True,
        ),
    ),
}


def _normalize_method(path_min_method: str) -> str:
    return str(path_min_method or "").strip().upper().replace("_", "-")


def get_schema(path_min_method: str) -> dict[str, ParamSpec]:
    key = _normalize_method(path_min_method)
    try:
        return SAFE_KNOBS_BY_METHOD[key]
    except KeyError:
        available = ", ".join(sorted(SAFE_KNOBS_BY_METHOD))
        raise ValueError(
            f"No agentic-tuning knob schema defined for "
            f"path_min_method={path_min_method!r} yet. Available: "
            f"{available}."
        ) from None


def _split(path: str) -> list[str]:
    return path.split(".")


def _get_by_path(d: dict, path: str, default: Any = None) -> Any:
    node: Any = d
    for part in _split(path):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set_by_path(d: dict, path: str, value: Any) -> None:
    parts = _split(path)
    node = d
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def extract_current_values(
    run_inputs_dict: dict, schema: dict[str, ParamSpec]
) -> dict[str, Any]:
    """Flat `{dotted_path: value}` snapshot of every tunable knob's current
    value in `run_inputs_dict` (a `RunInputs.to_dict()`-shaped dict)."""
    return {path: _get_by_path(run_inputs_dict, path) for path in schema}


def _flatten(nested: dict, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in nested.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted))
        else:
            flat[dotted] = value
    return flat


def _coerce_and_clamp(spec: ParamSpec, raw_value: Any) -> Any:
    if spec.kind == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str) and raw_value.strip().lower() in {"true", "false"}:
            return raw_value.strip().lower() == "true"
        raise ValueError(f"expected bool, got {raw_value!r}")
    if spec.kind == "categorical":
        if spec.choices and raw_value not in spec.choices:
            raise ValueError(f"expected one of {spec.choices}, got {raw_value!r}")
        return raw_value
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"expected a number, got {raw_value!r}") from None
    if spec.minimum is not None:
        value = max(value, float(spec.minimum))
    if spec.maximum is not None:
        value = min(value, float(spec.maximum))
    if spec.kind == "int":
        value = int(round(value))
    return value


def clamp_and_validate(
    proposed: dict, schema: dict[str, ParamSpec]
) -> tuple[dict[str, Any], list[str]]:
    """Accepts a proposal in either nested (`{"gi_inputs": {"nimages": 14}}`)
    or flat (`{"gi_inputs.nimages": 14}`) form -- an LLM backend's JSON
    response naturally mirrors the nested `to_json_schema` shape, but flat
    dicts are convenient for tests/manual use. Unknown paths are dropped
    with a warning (never applied); numeric values are coerced to their
    declared type and clamped into `[minimum, maximum]` -- defense-in-depth
    even though the LLM's response is already JSON-schema-constrained,
    since range constraints aren't reliably enforced by every backend
    model. Returns `(clamped_flat_dict, warnings)`."""
    flat = _flatten(proposed)
    clamped: dict[str, Any] = {}
    warnings: list[str] = []
    for path, raw_value in flat.items():
        spec = schema.get(path)
        if spec is None:
            warnings.append(f"Ignoring unknown/unsafe parameter '{path}'.")
            continue
        try:
            clamped[path] = _coerce_and_clamp(spec, raw_value)
        except ValueError as exc:
            warnings.append(f"Ignoring '{path}': {exc}")
    return clamped, warnings


def apply_values(
    run_inputs_dict: dict, schema: dict[str, ParamSpec], values: dict[str, Any]
) -> dict:
    """Returns a *new* dict: `run_inputs_dict` with `values` (a flat
    `{dotted_path: value}` mapping, already clamped -- see
    `clamp_and_validate`) applied on top. Only keys present in `values`
    change; everything else is left untouched, i.e. a partial proposal
    means "leave this knob unchanged"."""
    new_dict = copy.deepcopy(run_inputs_dict)
    for path, value in values.items():
        if path not in schema:
            continue
        _set_by_path(new_dict, path, value)
    return new_dict


def _param_spec_json_schema(spec: ParamSpec) -> dict:
    if spec.kind == "bool":
        return {"type": "boolean", "description": spec.description}
    if spec.kind == "categorical":
        return {"enum": list(spec.choices or []), "description": spec.description}
    json_type = "integer" if spec.kind == "int" else "number"
    schema_obj: dict[str, Any] = {"type": json_type, "description": spec.description}
    if spec.minimum is not None:
        schema_obj["minimum"] = spec.minimum
    if spec.maximum is not None:
        schema_obj["maximum"] = spec.maximum
    return schema_obj


def to_json_schema(schema: dict[str, ParamSpec]) -> dict:
    """Nested JSON Schema (grouped by top-level `input.toml` section) for an
    LLM backend's structured-output constraint (e.g. Ollama's `format=`).
    v0's `SAFE_KNOBS_BY_METHOD` paths are exactly two segments deep, so
    grouping only needs one level of nesting."""
    groups: dict[str, dict[str, Any]] = {}
    for path, spec in schema.items():
        top, leaf = path.split(".", 1)
        group = groups.setdefault(
            top, {"type": "object", "properties": {}, "additionalProperties": False}
        )
        group["properties"][leaf] = _param_spec_json_schema(spec)
    return {"type": "object", "properties": groups, "additionalProperties": False}


def render_schema_for_prompt(schema: dict[str, ParamSpec]) -> str:
    """Compact text table of the tunable knobs, for the LLM prompt (and
    `mepd agentic show-schema`)."""
    lines = ["path | type | bounds | description", "-" * 60]
    for path, spec in sorted(schema.items()):
        if spec.kind == "categorical":
            bounds = f"one of {list(spec.choices or [])}"
        elif spec.kind == "bool":
            bounds = "true/false"
        else:
            lo = spec.minimum if spec.minimum is not None else "-inf"
            hi = spec.maximum if spec.maximum is not None else "+inf"
            bounds = f"[{lo}, {hi}]" + (" (log-scale)" if spec.log_scale else "")
        lines.append(f"{path} | {spec.kind} | {bounds} | {spec.description}")
    return "\n".join(lines)
