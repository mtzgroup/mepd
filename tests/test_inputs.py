from mepd.inputs import RunInputs
from mepd.pathminimizers.mlpgi import _resolve_optimizer_config_values, _resolve_mlpgi_backend
from mepd.nodes.node import StructureNode
from mepd.optimizers.sgd import SGDOptimizer
from mepd.optimizers.gd import DeterministicGradientDescentOptimizer
from qcio import Structure
import pytest
import sys
import types
import numpy as np


def test_runinputs_fsm_uses_empty_path_min_inputs():
    inputs = RunInputs(path_min_method="fsm")

    assert vars(inputs.path_min_inputs) == {}


def test_runinputs_gi_random_seed_default_and_toml(tmp_path):
    inputs = RunInputs(path_min_method="neb")
    assert inputs.gi_inputs.random_seed == 0

    fp = tmp_path / "inputs.toml"
    fp.write_text(
        """
engine_name = "chemcloud"
program = "xtb"
path_min_method = "NEB"

[gi_inputs]
random_seed = 123

[optimizer_kwds]
name = "cg"
""".lstrip()
    )

    loaded = RunInputs.open(fp)

    assert loaded.gi_inputs.random_seed == 123


def test_runinputs_fsm_can_save_defaults(tmp_path):
    inputs = RunInputs(path_min_method="fsm")
    out_fp = tmp_path / "default_inputs.toml"

    inputs.save(out_fp)

    text = out_fp.read_text()
    assert 'path_min_method = "fsm"' in text


def test_runinputs_reads_fraction_freeze_from_toml(tmp_path, monkeypatch):
    class FakeQCOPEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(QCOPEngine=FakeQCOPEngine)
    monkeypatch.setitem(sys.modules, "mepd.engines.qcop", fake_module)

    fp = tmp_path / "inputs.toml"
    fp.write_text(
        """
engine_name = "qcop"
program = "xtb"
path_min_method = "NEB"

[chain_inputs]
fraction_freeze = 0.25

[optimizer_kwds]
name = "cg"
""".lstrip()
    )

    inputs = RunInputs.open(fp)

    assert inputs.chain_inputs.fraction_freeze == pytest.approx(0.25)


def test_runinputs_rejects_removed_friction_optimal_gi(tmp_path):
    fp = tmp_path / "inputs.toml"
    fp.write_text(
        """
engine_name = "chemcloud"
program = "xtb"
path_min_method = "NEB"

[chain_inputs]
friction_optimal_gi = true

[optimizer_kwds]
name = "cg"
""".lstrip()
    )

    with pytest.raises(ValueError, match="friction_optimal_gi has been removed"):
        RunInputs.open(fp)


def test_runinputs_passes_geometry_optimizer_keywords_from_toml(tmp_path, monkeypatch):
    class FakeQCOPEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(QCOPEngine=FakeQCOPEngine)
    monkeypatch.setitem(sys.modules, "mepd.engines.qcop", fake_module)

    fp = tmp_path / "inputs.toml"
    fp.write_text(
        """
engine_name = "chemcloud"
program = "xtb"
path_min_method = "NEB"

[geometry_optimizer_kwds]
coordsys = "tric"
maxit = 75
convergence_energy = 1e-6

[optimizer_kwds]
name = "cg"
""".lstrip()
    )

    inputs = RunInputs.open(fp)

    assert inputs.geometry_optimizer_kwds["coordsys"] == "tric"
    assert inputs.geometry_optimizer_kwds["maxit"] == 75
    assert inputs.geometry_optimizer_kwds["convergence_energy"] == pytest.approx(1e-6)
    assert inputs.engine.kwargs["geometry_optimizer_kwds"] == inputs.geometry_optimizer_kwds


def test_runinputs_neb_dlf_has_expected_defaults():
    inputs = RunInputs(
        engine_name="qcop",
        program="terachem",
        path_min_method="neb-dlf",
    )

    defaults = vars(inputs.path_min_inputs)
    assert defaults["nstep"] == 200
    assert defaults["min_nebk"] == 0.01
    assert defaults["do_elem_step_checks"] is True
    assert defaults["early_stop_stage"] is False
    assert isinstance(defaults["early_stop_loose_overrides"], dict)
    assert defaults["skip_identical_graphs"] is True
    assert defaults["collect_files"] is True
    assert isinstance(defaults["dlfind_keywords"], dict)


def test_runinputs_neb_has_hessian_minima_validation_defaults():
    inputs = RunInputs(path_min_method="neb")

    defaults = vars(inputs.path_min_inputs)
    assert defaults["validate_minima_with_hessian"] is False
    assert defaults["hessian_minimum_frequency_cutoff"] == pytest.approx(0.0)
    assert defaults["hessian_minima_rescue_displacement"] == pytest.approx(0.1)
    assert defaults["network_completion_max_followup_requests"] == 1000
    assert defaults["recursive_split_max_depth"] == 200
    assert defaults["recursive_same_pair_split_limit"] == 5


def test_runinputs_reads_hessian_minima_rescue_displacement_from_toml(tmp_path):
    fp = tmp_path / "inputs.toml"
    fp.write_text(
        """
engine_name = "chemcloud"
program = "xtb"
path_min_method = "NEB"

[path_min_inputs]
hessian_minima_rescue_displacement = 0.025
network_completion_max_followup_requests = 17
recursive_split_max_depth = 9

[optimizer_kwds]
name = "cg"
""".lstrip()
    )

    inputs = RunInputs.open(fp)

    assert inputs.path_min_inputs.hessian_minima_rescue_displacement == pytest.approx(
        0.025
    )
    assert inputs.path_min_inputs.network_completion_max_followup_requests == 17
    assert inputs.path_min_inputs.recursive_split_max_depth == 9


def test_runinputs_mlpgi_has_expected_defaults():
    inputs = RunInputs(path_min_method="mlpgi")

    defaults = vars(inputs.path_min_inputs)
    assert defaults["backend"] == "auto"
    assert defaults["fire_stage1_iter"] == 200
    assert defaults["fire_stage2_iter"] == 500
    assert defaults["variance_penalty_weight"] == pytest.approx(0.0433641)
    assert defaults["fire_conv_geolen_tol"] == pytest.approx(0.25)
    assert defaults["fire_conv_erelpeak_tol"] == pytest.approx(0.25)
    assert defaults["refinement_step_interval"] == 10
    assert defaults["refinement_dynamic_threshold_fraction"] == pytest.approx(0.1)
    assert defaults["do_elem_step_checks"] is True
    assert defaults["skip_identical_graphs"] is True


def test_runinputs_geometric_neb_has_expected_defaults():
    inputs = RunInputs(path_min_method="geometric-neb")

    defaults = vars(inputs.path_min_inputs)
    assert defaults["max_steps"] == 200
    assert defaults["rms_grad_thre"] == pytest.approx(0.02)
    assert defaults["max_rms_grad_thre"] == pytest.approx(0.05)
    assert defaults["do_elem_step_checks"] is True
    assert defaults["batch_engine_calls"] is True
    assert defaults["align"] is True
    assert defaults["nebk"] == pytest.approx(1.0)


def test_mlpgi_backend_auto_uses_engine_for_chemcloud_crest():
    engine = types.SimpleNamespace(program="crest", compute_program="chemcloud")
    requested, resolved = _resolve_mlpgi_backend({"backend": "auto"}, engine)
    assert requested == "auto"
    assert resolved == "engine"


def test_mlpgi_backend_auto_falls_back_to_fairchem_without_crest_chemcloud():
    engine = types.SimpleNamespace(program="xtb", compute_program="qcop")
    requested, resolved = _resolve_mlpgi_backend({"backend": "auto"}, engine)
    assert requested == "auto"
    assert resolved == "fairchem"


def test_mlpgi_backend_chemcloud_alias_maps_to_engine():
    requested, resolved = _resolve_mlpgi_backend({"backend": "chemcloud"}, None)
    assert requested == "chemcloud"
    assert resolved == "engine"


def test_runinputs_propagates_disable_molecular_graphs_to_engine(monkeypatch):
    class FakeQCOPEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(QCOPEngine=FakeQCOPEngine)
    monkeypatch.setitem(sys.modules, "mepd.engines.qcop", fake_module)

    inputs = RunInputs(
        engine_name="qcop",
        program="xtb",
        path_min_method="NEB",
        path_min_inputs={"disable_molecular_graphs": True, "skip_identical_graphs": False},
    )

    assert getattr(inputs.path_min_inputs, "disable_molecular_graphs", False) is True
    assert getattr(inputs.engine, "disable_molecular_graphs", False) is True


def test_runinputs_warns_when_disable_molecular_graphs_and_skip_identical_graphs_enabled(monkeypatch):
    class FakeQCOPEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(QCOPEngine=FakeQCOPEngine)
    monkeypatch.setitem(sys.modules, "mepd.engines.qcop", fake_module)

    with pytest.warns(
        UserWarning, match="disable_molecular_graphs=true.*skip_identical_graphs=true"
    ):
        RunInputs(
            engine_name="qcop",
            program="xtb",
            path_min_method="NEB",
            path_min_inputs={
                "disable_molecular_graphs": True,
                "skip_identical_graphs": True,
            },
        )


def test_runinputs_accepts_sgd_optimizer_names(monkeypatch):
    class FakeQCOPEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(QCOPEngine=FakeQCOPEngine)
    monkeypatch.setitem(sys.modules, "mepd.engines.qcop", fake_module)

    inputs = RunInputs(
        engine_name="qcop",
        program="xtb",
        path_min_method="NEB",
        optimizer_kwds={"name": "sgd", "timestep": 0.03, "momentum": 0.5},
    )
    assert isinstance(inputs.optimizer, SGDOptimizer)
    assert inputs.optimizer.timestep == pytest.approx(0.03)
    assert inputs.optimizer.momentum == pytest.approx(0.5)

    alias_inputs = RunInputs(
        engine_name="qcop",
        program="xtb",
        path_min_method="NEB",
        optimizer_kwds={
            "name": "stochastic_gradient_descent",
            "timestep": 0.02,
        },
    )
    assert isinstance(alias_inputs.optimizer, SGDOptimizer)
    assert alias_inputs.optimizer.timestep == pytest.approx(0.02)


def test_runinputs_accepts_deterministic_gradient_descent_optimizer_names(monkeypatch):
    class FakeQCOPEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(QCOPEngine=FakeQCOPEngine)
    monkeypatch.setitem(sys.modules, "mepd.engines.qcop", fake_module)

    inputs = RunInputs(
        engine_name="qcop",
        program="xtb",
        path_min_method="NEB",
        optimizer_kwds={"name": "gd", "timestep": 0.03, "max_step_norm": 0.2},
    )
    assert isinstance(inputs.optimizer, DeterministicGradientDescentOptimizer)
    assert inputs.optimizer.timestep == pytest.approx(0.03)
    assert inputs.optimizer.max_step_norm == pytest.approx(0.2)

    alias_inputs = RunInputs(
        engine_name="qcop",
        program="xtb",
        path_min_method="NEB",
        optimizer_kwds={
            "name": "deterministic_gradient_descent",
            "timestep": 0.02,
        },
    )
    assert isinstance(alias_inputs.optimizer, DeterministicGradientDescentOptimizer)
    assert alias_inputs.optimizer.timestep == pytest.approx(0.02)


def test_disable_molecular_graphs_prevents_structurenode_graph_construction(monkeypatch):
    class FakeQCOPEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(QCOPEngine=FakeQCOPEngine)
    monkeypatch.setitem(sys.modules, "mepd.engines.qcop", fake_module)

    def _raise_if_called(_structure):
        raise AssertionError("structure_to_molecule should not be called when graphs are disabled")

    monkeypatch.setattr("mepd.nodes.node.structure_to_molecule", _raise_if_called)

    try:
        _ = RunInputs(
            engine_name="qcop",
            program="xtb",
            path_min_method="NEB",
            path_min_inputs={
                "disable_molecular_graphs": True,
                "skip_identical_graphs": False,
            },
        )
        struct = Structure(
            symbols=["H", "H"],
            geometry=np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]),
            charge=0,
            multiplicity=1,
        )
        node = StructureNode(structure=struct)
        assert node.has_molecular_graph is False
        assert node.graph is None
    finally:
        StructureNode.set_global_disable_molecular_graphs(False)


def test_mlpgi_optimizer_fire_conv_tolerances_use_kcal_input_units():
    cfg = _resolve_optimizer_config_values(
        {
            "fire_conv_geolen_tol": 0.25,
            "fire_conv_erelpeak_tol": 0.25,
        }
    )

    assert cfg["fire_conv_geolen_tol"] == pytest.approx(0.010841025)
    assert cfg["fire_conv_erelpeak_tol"] == pytest.approx(0.010841025)


def test_mlpgi_optimizer_aliases_map_to_config_values():
    cfg = _resolve_optimizer_config_values(
        {
            "beta": 1.0,
            "tau_refine": 8,
            "cutoff": 10,
            "convergence_window": 12,
            "path_length_tolerance": 0.25,
            "barrier_height_tolerance": 0.25,
        }
    )

    assert cfg["variance_penalty_weight"] == pytest.approx(0.0433641)
    assert cfg["refinement_step_interval"] == 8
    assert cfg["refinement_dynamic_threshold_fraction"] == pytest.approx(0.1)
    assert cfg["fire_conv_window"] == 12
    assert cfg["fire_conv_geolen_tol"] == pytest.approx(0.010841025)
    assert cfg["fire_conv_erelpeak_tol"] == pytest.approx(0.010841025)


def test_runinputs_ase_omol25_reports_missing_fairchem(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "fairchem.core":
            raise ModuleNotFoundError("No module named 'fairchem'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    try:
        RunInputs(
            engine_name="ase",
            program="omol25",
            program_kwds={},
            path_min_method="fsm",
        )
    except ModuleNotFoundError as exc:
        msg = str(exc)
        assert "fairchem-core" in msg
        assert "Python 3.14" in msg
    else:
        raise AssertionError("Expected ModuleNotFoundError when fairchem.core is unavailable")


def test_runinputs_ase_omol25_uses_configured_model_path_and_device(monkeypatch):
    import sys
    import types

    calls = {}

    def _fake_load_predict_unit(model_path, device):
        calls["model_path"] = model_path
        calls["device"] = device
        return "predictor"

    class _FakeFAIRChemCalculator:
        def __init__(self, predictor, task_name):
            calls["predictor"] = predictor
            calls["task_name"] = task_name

    fairchem_core = types.ModuleType("fairchem.core")
    fairchem_core.pretrained_mlip = types.SimpleNamespace(
        load_predict_unit=_fake_load_predict_unit
    )
    fairchem_core.FAIRChemCalculator = _FakeFAIRChemCalculator

    fairchem_root = types.ModuleType("fairchem")
    fairchem_root.core = fairchem_core

    monkeypatch.setitem(sys.modules, "fairchem", fairchem_root)
    monkeypatch.setitem(sys.modules, "fairchem.core", fairchem_core)

    run_inputs = RunInputs(
        engine_name="ase",
        program="omol25",
        program_kwds={},
        ase_engine_kwds={"geometry_optimizer": "FIRE"},
        path_min_method="fsm",
        path_min_inputs={
            "model_path": "/tmp/custom_omol25_checkpoint.pt",
            "device": "cpu",
        },
    )

    assert run_inputs.engine.__class__.__name__ == "ASEEngine"
    assert calls["model_path"] == "/tmp/custom_omol25_checkpoint.pt"
    assert calls["device"] == "cpu"
    assert calls["predictor"] == "predictor"
    assert calls["task_name"] == "omol"
    assert run_inputs.engine.geometry_optimizer == "FIRE"


def test_runinputs_ase_omol25_raises_with_model_path_context(monkeypatch):
    import sys
    import types

    def _fake_load_predict_unit(model_path, device):
        raise RuntimeError("load failed")

    class _FakeFAIRChemCalculator:
        def __init__(self, predictor, task_name):
            pass

    fairchem_core = types.ModuleType("fairchem.core")
    fairchem_core.pretrained_mlip = types.SimpleNamespace(
        load_predict_unit=_fake_load_predict_unit
    )
    fairchem_core.FAIRChemCalculator = _FakeFAIRChemCalculator

    fairchem_root = types.ModuleType("fairchem")
    fairchem_root.core = fairchem_core

    monkeypatch.setitem(sys.modules, "fairchem", fairchem_root)
    monkeypatch.setitem(sys.modules, "fairchem.core", fairchem_core)

    with pytest.raises(RuntimeError) as excinfo:
        RunInputs(
            engine_name="ase",
            program="omol25",
            program_kwds={},
            path_min_method="fsm",
            path_min_inputs={"model_path": "/tmp/missing.pt", "device": "cpu"},
        )

    msg = str(excinfo.value)
    assert "Failed to load OMol25 model for ASE engine" in msg
    assert "model_path='/tmp/missing.pt'" in msg
    assert "device='cpu'" in msg
