from types import SimpleNamespace

import numpy as np
import pytest
from qcdata import Structure
from qcdata.models.inputs import FileInput

from mepd.engines.qccompute import QCComputeEngine
from mepd.program_args import ProgramArgs
from mepd.nodes.node import StructureNode


def _program_of(inp_obj):
    """Read the target program back off an input model (qcdata >=0.19)."""
    first = inp_obj[0] if isinstance(inp_obj, list) else inp_obj
    return first.program


def _node_at_x(x: float) -> StructureNode:
    struct = Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]], dtype=float),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )
    return StructureNode(structure=struct)


def test_terachem_single_geomopt_preserves_program_keywords():
    engine = QCComputeEngine(
        program="terachem",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 7, "precision": "mixed"},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    assert captured["program"] == "terachem"
    kw = captured["input"].keywords
    assert kw["threads"] == 7
    assert kw["precision"] == "mixed"
    assert kw["purify"] == "no"
    assert kw["new_minimizer"] == "yes"


def test_terachem_batch_geomopt_preserves_program_keywords():
    engine = QCComputeEngine(
        program="terachem",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 3, "gpus": 1},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["inputs"] = inp_obj
        return [SimpleNamespace(), SimpleNamespace()]

    engine.compute_func = _fake_compute_func
    _ = engine.compute_geometry_optimizations([_node_at_x(1.0), _node_at_x(1.2)])

    assert captured["program"] == "terachem"
    assert isinstance(captured["inputs"], list)
    assert len(captured["inputs"]) == 2
    for prog_input in captured["inputs"]:
        kw = prog_input.keywords
        assert kw["threads"] == 3
        assert kw["gpus"] == 1
        assert kw["purify"] == "no"
        assert kw["new_minimizer"] == "yes"


def test_terachem_single_geomopt_with_frozen_atoms_uses_constraints_file_input():
    engine = QCComputeEngine(
        program="terachem",
        compute_program="chemcloud",
        frozen_atom_indices=[0],
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 7},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["input"] = inp_obj
        captured["collect_files"] = kwargs.get("collect_files")
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0), keywords={"maxiter": 10})

    assert captured["program"] == "terachem"
    assert captured["collect_files"] is True
    assert isinstance(captured["input"], FileInput)
    tcin = captured["input"].files["tc.in"]
    assert "run minimize" in tcin
    assert "threads 7" in tcin
    assert "maxit 10" in tcin
    assert "$constraints" in tcin
    assert "atom 1" in tcin
    assert "$end" in tcin
    assert "frozen_atom_indices" not in tcin


def test_terachem_batch_geomopt_with_frozen_atoms_uses_constraints_file_inputs():
    engine = QCComputeEngine(
        program="terachem",
        compute_program="chemcloud",
        frozen_atom_indices=[1],
        program_args=ProgramArgs(
            model={"method": "b3lyp", "basis": "6-31g*"},
            keywords={},
        ),
    )
    captured = {}

    class _Output:
        files = {
            "optim.xyz": (
                "2\n"
                "frame0\n"
                "H 0.000000 0.000000 0.000000\n"
                "H 1.000000 0.000000 0.000000\n"
            )
        }

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["inputs"] = inp_obj
        captured["collect_files"] = kwargs.get("collect_files")
        return [_Output(), _Output()]

    engine.compute_func = _fake_compute_func
    _ = engine.compute_geometry_optimizations([_node_at_x(1.0), _node_at_x(1.2)])

    assert captured["program"] == "terachem"
    assert captured["collect_files"] is True
    assert isinstance(captured["inputs"], list)
    assert len(captured["inputs"]) == 2
    for inp in captured["inputs"]:
        assert isinstance(inp, FileInput)
        tcin = inp.files["tc.in"]
        assert "$constraints" in tcin
        assert "atom 2" in tcin
        assert "$end" in tcin


def test_qccompute_prepare_node_for_comparison_uses_non_frozen_atoms():
    engine = QCComputeEngine(frozen_atom_indices=[1])
    node = _node_at_x(1.0)

    prepared = engine.prepare_node_for_comparison(node)

    assert prepared.comparison_atom_indices == [0]
    assert prepared.graph_atom_indices_source == "qccompute_non_frozen_atoms"
    assert prepared.graph_subset_atom_count == 1
    assert prepared.graph_total_atom_count == 2
    assert prepared.disable_smiles is True


def test_non_terachem_geomopt_default_keywords_use_coordsys():
    engine = QCComputeEngine(
        program="xtb",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    assert captured["program"] == engine.geometry_optimizer
    kw = captured["input"].keywords
    assert kw["coordsys"] == "cart"
    assert kw["maxit"] == 500
    assert kw["convergence_set"] == "GAU_TIGHT"


def test_non_terachem_geomopt_uses_engine_geometry_optimizer_keywords():
    engine = QCComputeEngine(
        program="xtb",
        compute_program="chemcloud",
        geometry_optimizer_kwds={
            "coordsys": "tric",
            "maxit": 120,
            "convergence_set": "GAU_VERYTIGHT",
            "convergence_energy": 1e-6,
        },
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["program"] = _program_of(inp_obj)
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    kw = captured["input"].keywords
    assert kw["coordsys"] == "tric"
    assert kw["maxit"] == 120
    assert kw["convergence_set"] == "GAU_VERYTIGHT"
    assert kw["convergence_energy"] == pytest.approx(1e-6)


def test_non_terachem_geomopt_call_keywords_override_engine_defaults():
    engine = QCComputeEngine(
        program="xtb",
        compute_program="chemcloud",
        geometry_optimizer_kwds={"coordsys": "tric", "maxit": 120},
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0), keywords={"maxit": 25})

    kw = captured["input"].keywords
    assert kw["coordsys"] == "tric"
    assert kw["maxit"] == 25


def test_non_terachem_geomopt_partial_engine_keywords_keep_defaults():
    engine = QCComputeEngine(
        program="xtb",
        compute_program="chemcloud",
        geometry_optimizer_kwds={"maxit": 80},
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(inp_obj, **kwargs):
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    kw = captured["input"].keywords
    assert kw["coordsys"] == "cart"
    assert kw["maxit"] == 80
    assert kw["convergence_set"] == "GAU_TIGHT"
