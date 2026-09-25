import copy
import importlib.util
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure
from typer.testing import CliRunner

pytest.importorskip("ase")
from ase.calculators.emt import EMT  # noqa: E402

import mepd.engines.fairchem as fairchem_module  # noqa: E402
from mepd.cli import app  # noqa: E402
from mepd.engines.ase import ASEEngine  # noqa: E402
from mepd.engines.fairchem import FAIRChemEngine  # noqa: E402
from mepd.engines.mlip import FAMILIES, MODELS, MLIPEngine, build_mlip_engine  # noqa: E402
from mepd.inputs import RunInputs  # noqa: E402
from mepd.nodes.node import StructureNode  # noqa: E402


def _co():
    return StructureNode(structure=Structure(symbols=["C", "O"], geometry=np.array([[0, 0, 0], [0, 0, 2.2]]), charge=0, multiplicity=1))


def test_every_registered_model_has_a_known_family():
    for name, spec in MODELS.items():
        assert spec.family in FAMILIES or spec.family == "fairchem", name
        assert spec.install and spec.summary


@pytest.mark.parametrize("name", [n for n, s in MODELS.items() if s.family != "fairchem"])
def test_open_models_build_an_mlip_engine_without_loading(name):
    eng = RunInputs(engine_name="mlip", mlip_engine_kwds={"model": name}).engine
    assert isinstance(eng, MLIPEngine) and eng._model is None


def test_fairchem_names_get_the_batched_engine():
    eng = build_mlip_engine({"model": "uma-s-1p2p1", "device": "cpu", "batch_size": 8})
    assert isinstance(eng, FAIRChemEngine) and (eng.model, eng.device, eng.batch_size) == ("uma-s-1p2p1", "cpu", 8)


def test_unknown_model_lists_the_choices():
    with pytest.raises(ValueError) as err:
        RunInputs(engine_name="mlip", mlip_engine_kwds={"model": "not-a-model"})
    assert "aimnet2-rxn" in str(err.value) and "mepd models" in str(err.value)


def test_missing_package_names_the_install_command():
    if importlib.util.find_spec("aimnet") is not None:
        pytest.skip("aimnet is installed here")
    eng = MLIPEngine(model="aimnet2-rxn")
    with pytest.raises(ImportError) as err:
        eng.calculator
    assert 'pip install "aimnet[ase]"' in str(err.value)


def test_gated_model_explains_how_to_get_access_or_switch(monkeypatch):
    class GatedRepoError(Exception):
        pass

    def refuse(*a, **k):
        raise GatedRepoError("403 Client Error. Access to model facebook/UMA is restricted")

    monkeypatch.setitem(__import__("sys").modules, "fairchem.core", SimpleNamespace(pretrained_mlip=SimpleNamespace(get_predict_unit=refuse)))
    monkeypatch.setitem(__import__("sys").modules, "fairchem", SimpleNamespace())
    with pytest.raises(PermissionError) as err:
        FAIRChemEngine(model="uma-s-1p2p1").predictor
    text = str(err.value)
    assert "huggingface.co/facebook/UMA" in text and "hf auth login" in text and "aimnet2-rxn" in text


def test_custom_calculator_by_import_path():
    eng = RunInputs(engine_name="mlip", mlip_engine_kwds={"calculator": "ase.calculators.emt:EMT"}).engine
    assert isinstance(eng.calculator, EMT)
    assert np.allclose(eng.compute_energies([_co()]), ASEEngine(calculator=EMT()).compute_energies([_co()]))


def test_local_checkpoint_of_a_known_family():
    eng = RunInputs(engine_name="mlip", mlip_engine_kwds={"model": "mine", "family": "fairchem", "checkpoint": "/models/uma.pt"}).engine
    assert isinstance(eng, FAIRChemEngine) and eng.checkpoint == "/models/uma.pt"
    eng = RunInputs(engine_name="mlip", mlip_engine_kwds={"model": "mine", "family": "aimnet2", "checkpoint": "/models/a.pt"}).engine
    assert isinstance(eng, MLIPEngine) and (eng.family, eng.checkpoint) == ("aimnet2", "/models/a.pt")


def test_mlip_engine_pickles_without_the_model_and_copies_share_it():
    eng = MLIPEngine(calculator="ase.calculators.emt:EMT")
    eng.calculator
    loaded = eng._model
    assert pickle.loads(pickle.dumps(eng))._model is None
    assert copy.deepcopy(eng)._model is loaded


def test_mepd_models_lists_every_model():
    result = CliRunner().invoke(app, ["models"], env={"COLUMNS": "220"})
    assert result.exit_code == 0, result.output
    for name in MODELS:
        assert name in result.output
    assert 'aimnet[ase]' in result.output
