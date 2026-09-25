"""The Profiles settings form: big choices + choice-dependent Advanced
settings, applied to the TOML without losing anything the user set."""

from __future__ import annotations

import tomllib

import pytest

pytest.importorskip("fastapi")

from mepd.web import profile_form as pf  # noqa: E402
from mepd.web.workspace import WorkspaceError, validate_profile_text  # noqa: E402

BASE = """engine_name = "gxtb"
program = "xtb"
path_min_method = "NEB"
my_custom_key = "keep me"

[path_min_inputs]
climb = true
max_steps = 300
skip_identical_graphs = false

[chain_inputs]
k = 0.2

[gi_inputs]
nimages = 12
"""
VALID = BASE.replace('my_custom_key = "keep me"\n', "")   # (a real RunInputs rejects unknown keys)


def _choices(f):
    return {c["key"]: c["value"] for c in f["basic"]}


def _find(f, key):
    return next(c for c in f["basic"] if c["key"] == key)


def _toml(out):
    return tomllib.loads(out["text"])


def test_form_reads_the_big_choices_and_relevant_groups():
    f = pf.form(BASE)
    assert _choices(f) == {"path_method": "NEB", "engine": "gxtb", "interpolation": "geodesic", "optimizer": "cg"}
    assert [o["value"] for o in _find(f, "interpolation")["options"]] == ["geodesic", "idpp", "lst", "linear"]
    assert [o["value"] for o in _find(f, "engine")["options"]] == ["gxtb", "qccompute", "chemcloud", "mlip", "fairchem", "ase"]
    assert f["images"]["value"] == 12
    titles = [g["title"] for g in f["groups"]]
    assert titles[0] == "NEB settings" and "g-xTB" in titles and "Conjugate gradient optimizer" in titles
    path = {x["path"]: x for x in f["groups"][0]["fields"]}
    assert path["path_min_inputs.max_steps"]["value"] == 300 and path["path_min_inputs.max_steps"]["key"]
    assert "path_min_inputs.validate_minima_with_hessian" not in path   # set per calculation instead


def test_labels_read_as_words():
    labels = {x["path"].rsplit(".", 1)[-1]: x["label"] for g in pf.form(BASE.replace('"NEB"', '"GSM"'))["groups"]
              for x in g["fields"]}
    assert labels["n_threads"] == "Threads per call"
    assert labels["int_thresh"] == "Int thresh" and "rtolerance" not in " ".join(labels.values())


def test_switching_method_keeps_user_values_and_drops_only_untouched_defaults():
    out = pf.apply(BASE, "path_method", "GSM")
    d = _toml(out)
    assert d["path_min_method"] == "GSM"
    assert "climb" not in d["path_min_inputs"]                    # NEB's default, not read by GSM: dropped
    assert d["path_min_inputs"]["max_steps"] == 300               # the user's value: kept (listed as unused)
    assert "max_steps" in out["form"]["unused"]
    assert d["path_min_inputs"]["skip_identical_graphs"] is False  # shared
    assert "nnodes" not in d["path_min_inputs"]                   # defaults aren't frozen into the file
    back = _toml(pf.apply(out["text"], "path_method", "NEB"))
    assert back["path_min_inputs"]["max_steps"] == 300            # NEB -> GSM -> NEB keeps the edit
    assert d["my_custom_key"] == "keep me" and d["chain_inputs"]["k"] == 0.2
    assert "optimizer" not in _choices(out["form"])               # GSM has its own optimizer


def test_method_names_are_checked_and_fsm_is_fneb():
    assert _toml(pf.apply(BASE, "path_method", "FSM"))["path_min_method"] == "FNEB"
    for bad in ("xyz", None, 5):
        with pytest.raises(WorkspaceError):
            pf.apply(BASE, "path_method", bad)
    with pytest.raises(WorkspaceError, match="TeraChem"):
        pf.apply(BASE, "path_method", "NEB-DLF")                  # g-xTB profile


def test_engine_and_program_switches_keep_the_level_of_theory():
    t = pf.apply(BASE, "engine", "qccompute")["text"]
    out = pf.apply(t, "program", "terachem")
    assert _toml(out)["program_kwds"]["model"] == {"method": "ub3lyp", "basis": "3-21g"}
    out = pf.apply(out["text"], "program_kwds.model.basis", "6-31gs")
    dlf = next(o for o in _find(out["form"], "path_method")["options"] if o["value"] == "NEB-DLF")
    assert dlf["disabled"] is None
    for other in ("gxtb", "fairchem", "ase"):
        back = _toml(pf.apply(pf.apply(out["text"], "engine", other)["text"], "engine", "qccompute"))
        assert back["program"] == "terachem" and back["program_kwds"]["model"]["basis"] == "6-31gs"
    gx = pf.form(pf.apply(out["text"], "engine", "gxtb")["text"])
    assert any("[program_kwds]" in n for n in gx["notes"])        # kept, and said so
    # A level the user chose survives a program change; a program's default is swapped.
    assert _toml(pf.apply(out["text"], "program", "psi4"))["program_kwds"]["model"]["basis"] == "6-31gs"
    plain = pf.apply(pf.apply(t, "program", "terachem")["text"], "program", "psi4")
    assert "program_kwds" not in _toml(plain) and any("needs a method" in i for i in plain["form"]["issues"])
    for bad in (None, "", {"x": 1}):
        with pytest.raises(WorkspaceError):
            pf.apply(t, "program", bad)
    for bad in ("foo", None):
        with pytest.raises(WorkspaceError):
            pf.apply(BASE, "engine", bad)


def test_fairchem_and_ase_calculator_engines():
    f = pf.apply(BASE, "engine", "fairchem")["form"]
    grp = next(g for g in f["groups"] if g["id"] == "engine")
    assert grp["title"] == "FAIR-Chem model" and any(x["path"] == "fairchem_engine_kwds.model" for x in grp["fields"])
    out = pf.apply(pf.apply(BASE, "engine", "fairchem")["text"], "fairchem_engine_kwds.device", "cpu")
    assert _toml(out)["fairchem_engine_kwds"]["device"] == "cpu"
    out = pf.apply(out["text"], "fairchem_engine_kwds.device", "")   # "automatic" = the default
    assert "device" not in _toml(out).get("fairchem_engine_kwds", {})
    a = pf.apply(BASE, "engine", "ase")
    assert any("needs a calculator" in i for i in a["form"]["issues"])
    a = pf.apply(a["text"], "calculator", "emt")
    assert not any("calculator" in i for i in a["form"]["issues"])


def test_numbers_are_stored_typed_and_checked():
    out = pf.apply(BASE, "gi_inputs.nimages", "14")
    assert _toml(out)["gi_inputs"]["nimages"] == 14               # was stored as the string "14"
    out = pf.apply(BASE, "path_min_inputs.max_steps", "750")
    assert _toml(out)["path_min_inputs"]["max_steps"] == 750
    assert _toml(pf.apply(BASE, "path_min_inputs.climb", "false"))["path_min_inputs"]["climb"] is False
    assert "max_steps" not in _toml(pf.apply(BASE, "path_min_inputs.max_steps", None))["path_min_inputs"]
    gsm = pf.apply(BASE, "path_method", "GSM")["text"]
    assert _toml(pf.apply(gsm, "path_min_inputs.timeout", "100"))["path_min_inputs"]["timeout"] == 100.0
    assert _toml(pf.apply(BASE, "optimizer_kwds.min_timestep", "0.01"))["optimizer_kwds"]["min_timestep"] == 0.01
    for path, bad in (("path_min_inputs.max_steps", "lots"), ("path_min_inputs.max_steps", "2.7"),
                      ("path_min_inputs.max_steps", "0"), ("gi_inputs.nimages", "1"), ("gi_inputs.nimages", "-4"),
                      ("chain_inputs.k", "nan"), ("chain_inputs.k", "inf"), ("path_min_inputs.climb", "maybe"),
                      ("path_min_inputs.max_steps", "1e30"), ("chain_inputs.k", True)):
        with pytest.raises(WorkspaceError):
            pf.apply(BASE, path, bad)
    qc = pf.apply(BASE, "engine", "qccompute")["text"]
    with pytest.raises(WorkspaceError):
        pf.apply(qc, "geometry_optimizer_kwds.coordsys", "bogus")  # dropdown values must be real options


def test_only_settings_on_the_form_can_be_written():
    for bad in ("engine_name.injected", "program_kwds.model.method.x", "path_min_inputs.max_steps.x",
                "path_min_inputs.", "path_min_inputs..x", "path_min_inputs.__class__", "program_kwds.model"):
        with pytest.raises(WorkspaceError):
            pf.apply(BASE, bad, 1)


def test_optimizer_interpolation_and_the_removed_switch():
    out = pf.apply(BASE, "optimizer", "fire")
    assert _toml(out)["optimizer_kwds"] == {"name": "fire"}
    tuned = pf.apply(out["text"], "optimizer_kwds.timestep", "0.3")
    again = pf.apply(tuned["text"], "optimizer", "fire")          # re-picking the same one is a no-op
    assert _toml(again)["optimizer_kwds"]["timestep"] == 0.3
    old = BASE.replace("[chain_inputs]\n", "[chain_inputs]\nuse_geodesic_interpolation = false\n")
    assert any("has been removed" in i for i in pf.form(old)["issues"])
    out = pf.apply(old, "interpolation", "linear")
    ci = _toml(out)["chain_inputs"]
    assert ci["interpolation"] == "linear" and "use_geodesic_interpolation" not in ci
    assert _choices(pf.apply(out["text"], "interpolation", "idpp")["form"])["interpolation"] == "idpp"
    with pytest.raises(WorkspaceError):
        pf.apply(BASE, "interpolation", "spline")


def test_unused_detection_keeps_every_key_mepd_reads():
    stale = BASE.replace("[path_min_inputs]", "[path_min_inputs]\nnnodes = 9\nrecursive_split_max_depth = 2")
    f = pf.form(stale)
    assert f["unused"] == ["nnodes"]
    cleaned = _toml(pf.apply(stale, "remove_unused", True))["path_min_inputs"]
    assert "nnodes" not in cleaned and cleaned["recursive_split_max_depth"] == 2 and cleaned["max_steps"] == 300
    fneb = 'path_min_method = "FNEB"\n[path_min_inputs]\ndisable_molecular_graphs = true\nrecursive_split_max_depth = 2\n'
    assert pf.form(fneb)["unused"] == []
    geo = 'path_min_method = "GEOMETRIC-NEB"\n[path_min_inputs]\nncimg = 2\nmaxg = 0.1\n'
    assert pf.form(geo)["unused"] == []


def test_bad_profiles_are_reported_not_crashed_on():
    f = pf.form('path_min_inputs = 5\n')
    assert any("[path_min_inputs] is not a table" in i for i in f["issues"])
    f = pf.form('[chain_inputs]\ninterpolation = "cubic"\n')
    assert any("Unknown initial path" in i for i in f["issues"])
    assert "cubic" in [o["value"] for o in _find(f, "interpolation")["options"]]
    f = pf.form('path_min_method = "MADEUP"\n')
    assert any("Unknown path method" in i for i in f["issues"])
    assert pf.form("﻿" + BASE)["basic"]                        # byte-order mark
    with pytest.raises(WorkspaceError):
        pf.form("not = [valid")


def test_comments_are_detected_anywhere_but_not_inside_strings():
    assert pf.form(BASE.replace('program = "xtb"', 'program = "xtb"  # note'))["has_comments"]
    assert pf.form("# top\n" + BASE)["has_comments"]
    assert not pf.form(BASE.replace('"keep me"', '"keep # me"'))["has_comments"]


def test_building_defaults_leaves_the_server_process_alone():
    from mepd.nodes.node import StructureNode

    pf.method_defaults.cache_clear()
    StructureNode.set_global_disable_molecular_graphs(True)
    try:
        pf.method_defaults("GSM")
        assert StructureNode._global_disable_molecular_graphs is True
    finally:
        StructureNode.set_global_disable_molecular_graphs(False)


def test_edited_profiles_still_validate():
    t = pf.apply(pf.apply(VALID, "path_method", "FNEB")["text"], "path_min_inputs.grad_tol", "0.03")["text"]
    assert validate_profile_text(t)["ok"]
    t = pf.apply(pf.apply(VALID, "gi_inputs.nimages", "14")["text"], "interpolation", "idpp")["text"]
    assert validate_profile_text(t)["ok"]
