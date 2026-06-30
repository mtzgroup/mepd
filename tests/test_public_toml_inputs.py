from dataclasses import fields
from pathlib import Path

import tomli

from mepd.inputs import ChainInputs, GIInputs, NEBInputs, RunInputs


ROOT = Path(__file__).resolve().parents[1]


def _field_names(cls) -> set[str]:
    return {field.name for field in fields(cls)}


def test_bundled_tomls_only_use_active_public_input_keys():
    allowed_top_level = _field_names(RunInputs) - {"engine", "optimizer"}
    allowed_sections = {
        "path_min_inputs": _field_names(NEBInputs),
        "chain_inputs": _field_names(ChainInputs),
        "gi_inputs": _field_names(GIInputs),
        "optimizer_kwds": {"name", "timestep", "max_step_norm"},
    }
    allowed_program_kwds_sections = {
        "cmdline_args",
        "extras",
        "files",
        "keywords",
        "model",
    }

    toml_paths = [
        ROOT / "mepd" / "default_inputs.toml",
        ROOT / "examples" / "example_inputs.toml",
        ROOT / "examples" / "neb_inputs.toml",
    ]

    for fp in toml_paths:
        data = tomli.loads(fp.read_text())
        assert set(data) <= allowed_top_level, fp
        assert not (
            {"nanoreactor_inputs", "geometry_optimizer_kwds", "network_inputs"} & set(data)
        ), fp

        for section, allowed_keys in allowed_sections.items():
            if section in data:
                assert set(data[section]) <= allowed_keys, f"{fp}:{section}"

        if "program_kwds" in data:
            assert set(data["program_kwds"]) <= allowed_program_kwds_sections, fp
            assert set(data["program_kwds"].get("model", {})) <= {
                "method",
                "basis",
                "extras",
            }, fp


def test_runinputs_save_omits_deprecated_path_min_inputs(tmp_path):
    out_fp = tmp_path / "defaults.toml"
    RunInputs(path_min_method="neb").save(out_fp)
    saved = tomli.loads(out_fp.read_text())

    path_min_inputs = saved["path_min_inputs"]
    assert "plateau_exit_window" not in path_min_inputs
    assert "plateau_exit_rtol" not in path_min_inputs
    assert "network_inputs" not in saved
