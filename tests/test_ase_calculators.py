import pytest

from mepd.engines.ase_calculators import CALCULATORS, load_calculator, lookup, supported_message
from mepd.inputs import _import_ase_calculator
from mepd.web.profile_form import apply, form


def test_short_names_aliases_and_paths_resolve():
    assert load_calculator("EMT").__name__ == "EMT"
    assert _import_ase_calculator("ase.calculators.emt:EMT").__name__ == "EMT"
    assert lookup("MACE") is CALCULATORS["mace-off"]
    assert lookup("xtb") is CALCULATORS["tblite"]
    assert lookup("ase.calculators.emt.EMT") is CALCULATORS["emt"]


def test_errors_say_what_is_supported():
    with pytest.raises(ValueError, match="Known names: mace-off"):
        load_calculator("nonsense")
    with pytest.raises(ValueError, match='engine_name = "mlip"'):
        load_calculator("aimnet2")
    with pytest.raises(ValueError, match="has no 'Nope'.*EMT"):
        load_calculator("ase.calculators.emt:Nope")
    with pytest.raises(ImportError, match="not installed here: pip install"):
        _missing("mace-off")
    assert "emt" in supported_message()


def _missing(name):
    import importlib.util

    if importlib.util.find_spec(CALCULATORS[name].package) is not None:
        pytest.skip(f"{name} is installed here")
    load_calculator(name)


def _ase(calc_line=""):
    return 'engine_name = "ase"\n[ase_engine_kwds]\n' + calc_line


def _choice(f, key):
    return next(c for c in f["basic"] if c["key"] == key)


def test_form_offers_known_calculators_and_flags_missing_ones():
    f = form('engine_name = "ase"\n')
    c = _choice(f, "calculator")
    assert c["value"] == "" and {"emt", "mace-off", "tblite", "custom"} <= {o["value"] for o in c["options"]}
    assert any("needs a calculator" in i for i in f["issues"])
    assert _choice(form(_ase('calculator = "MACE"\n')), "calculator")["value"] == "mace-off"
    assert form(_ase('calculator = "emt"\n'))["issues"] == []


def test_form_calculator_switch_custom_path_and_arguments():
    out = apply(_ase('calculator = "emt"\n'), "ase_engine_kwds.calculator_kwds", 'model = "medium", device = "cpu"')
    assert 'device = "cpu"' in out["text"]
    kw = next(x for g in out["form"]["groups"] for x in g["fields"] if x["path"] == "ase_engine_kwds.calculator_kwds")
    assert kw["value"] == 'model = "medium", device = "cpu"'
    out = apply(out["text"], "calculator", "lj")          # a new calculator drops the old one's arguments
    assert "calculator_kwds" not in out["text"]
    out = apply(out["text"], "calculator", "custom")
    assert out["form"]["level"][0]["path"] == "ase_engine_kwds.calculator"
    assert any("import path" in i for i in out["form"]["issues"])
    out = apply(out["text"], "ase_engine_kwds.calculator", "not_a_pkg.calc:Foo")
    assert _choice(out["form"], "calculator")["value"] == "custom"
    assert any("not importable" in i for i in out["form"]["issues"])
    with pytest.raises(Exception, match="key = value"):
        apply(out["text"], "ase_engine_kwds.calculator_kwds", "model = ")
    with pytest.raises(Exception, match="unknown calculator"):
        apply(out["text"], "calculator", "bogus")


def test_form_mlip_engine_gets_a_model_choice():
    out = apply('engine_name = "gxtb"\n', "engine", "mlip")
    assert 'model = "aimnet2-rxn"' in out["text"]
    assert _choice(out["form"], "mlip_model")["value"] == "aimnet2-rxn"
    out = apply(out["text"], "mlip_model", "ani-2x")
    assert 'model = "ani-2x"' in out["text"]
    with pytest.raises(Exception, match="unknown model"):
        apply(out["text"], "mlip_model", "bogus")
