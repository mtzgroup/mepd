from __future__ import annotations

import pytest

from mepd.agentic import schema


def _seed_dict() -> dict:
    """A plain dict literal shaped like `RunInputs().to_dict()`'s output --
    never an actually-constructed `RunInputs`, since the default
    `engine_name="gxtb"` would require the `gxtb` extra just to build one.
    `schema.py` is deliberately dict-in/dict-out only, so this is a hard
    architectural constraint, not just a test convenience."""
    return {
        "path_min_method": "NEB",
        "gi_inputs": {"nimages": 12, "friction": 0.01, "nudge": 0.1, "align": True},
        "chain_inputs": {"k": 0.1, "delta_k": 0.09},
        "path_min_inputs": {
            "max_steps": 500,
            "rms_grad_thre": 0.01,
            "max_rms_grad_thre": 0.02,
            "en_thre": 0.0005,
            "negative_steps_thre": 100000,
            "positive_steps_thre": 10000,
            "climb": True,
            "early_stop_force_thre": 0.01,
            "barrier_thre": 0.5,
        },
    }


def test_get_schema_returns_neb_knobs():
    result = schema.get_schema("NEB")
    assert "gi_inputs.nimages" in result
    assert "chain_inputs.k" in result
    assert "path_min_inputs.max_steps" in result
    # Never allow an unsafe field into the schema.
    assert not any(path.endswith("engine_name") for path in result)
    assert not any("program_kwds" in path for path in result)


def test_get_schema_normalizes_method_and_rejects_unknown():
    assert schema.get_schema("neb") is schema.get_schema("NEB")
    with pytest.raises(ValueError, match="No agentic-tuning knob schema"):
        schema.get_schema("FNEB")


def test_extract_current_values_reads_nested_dict():
    knobs = schema.get_schema("NEB")
    values = schema.extract_current_values(_seed_dict(), knobs)
    assert values["gi_inputs.nimages"] == 12
    assert values["chain_inputs.k"] == 0.1
    assert values["path_min_inputs.max_steps"] == 500


def test_clamp_and_validate_accepts_nested_and_flat_and_clamps_out_of_range():
    knobs = schema.get_schema("NEB")

    nested_clamped, nested_warnings = schema.clamp_and_validate(
        {"gi_inputs": {"nimages": 14}}, knobs
    )
    assert nested_clamped == {"gi_inputs.nimages": 14}
    assert nested_warnings == []

    flat_clamped, flat_warnings = schema.clamp_and_validate(
        {"gi_inputs.nimages": 14}, knobs
    )
    assert flat_clamped == {"gi_inputs.nimages": 14}
    assert flat_warnings == []

    # Out of range -> clamped into bounds, not rejected.
    clamped, _ = schema.clamp_and_validate({"gi_inputs.nimages": 9999}, knobs)
    assert clamped["gi_inputs.nimages"] == 30

    # int coercion from a float-ish value.
    clamped, _ = schema.clamp_and_validate({"path_min_inputs.max_steps": 123.6}, knobs)
    assert clamped["path_min_inputs.max_steps"] == 124
    assert isinstance(clamped["path_min_inputs.max_steps"], int)


def test_clamp_and_validate_drops_unknown_and_invalid_with_warnings():
    knobs = schema.get_schema("NEB")

    clamped, warnings = schema.clamp_and_validate({"engine_name": "gxtb"}, knobs)
    assert clamped == {}
    assert any("unknown/unsafe" in w for w in warnings)

    clamped, warnings = schema.clamp_and_validate(
        {"gi_inputs": {"nimages": "not-a-number"}}, knobs
    )
    assert clamped == {}
    assert any("gi_inputs.nimages" in w for w in warnings)

    clamped, warnings = schema.clamp_and_validate(
        {"path_min_inputs": {"climb": "sideways"}}, knobs
    )
    assert clamped == {}
    assert any("path_min_inputs.climb" in w for w in warnings)


def test_apply_values_is_partial_and_returns_a_new_dict():
    knobs = schema.get_schema("NEB")
    original = _seed_dict()
    updated = schema.apply_values(original, knobs, {"gi_inputs.nimages": 20})

    assert updated is not original
    assert updated["gi_inputs"]["nimages"] == 20
    # Untouched knobs are preserved unchanged.
    assert updated["chain_inputs"]["k"] == original["chain_inputs"]["k"]
    assert original["gi_inputs"]["nimages"] == 12  # original left untouched


def test_apply_values_ignores_keys_outside_the_schema():
    knobs = schema.get_schema("NEB")
    original = _seed_dict()
    updated = schema.apply_values(original, knobs, {"not.a.real.knob": 1})
    assert updated == original


def test_to_json_schema_is_nested_and_bounded():
    knobs = schema.get_schema("NEB")
    json_schema = schema.to_json_schema(knobs)
    assert json_schema["type"] == "object"
    assert json_schema["additionalProperties"] is False
    gi_props = json_schema["properties"]["gi_inputs"]["properties"]
    assert gi_props["nimages"]["type"] == "integer"
    assert gi_props["nimages"]["minimum"] == 6
    assert gi_props["nimages"]["maximum"] == 30
    climb_prop = json_schema["properties"]["path_min_inputs"]["properties"]["climb"]
    assert climb_prop["type"] == "boolean"


def test_render_schema_for_prompt_is_readable_text():
    knobs = schema.get_schema("NEB")
    text = schema.render_schema_for_prompt(knobs)
    assert "gi_inputs.nimages" in text
    assert "path_min_inputs.max_steps" in text
