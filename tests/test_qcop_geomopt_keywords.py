from types import SimpleNamespace

import numpy as np
import pytest
from qcio import Structure
from qcio.models.inputs import ProgramArgs

from mepd.engines.qcop import QCOPEngine
from mepd.nodes.node import StructureNode


def _node_at_x(x: float) -> StructureNode:
    struct = Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]], dtype=float),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )
    return StructureNode(structure=struct)


def test_terachem_single_geomopt_preserves_program_keywords():
    engine = QCOPEngine(
        program="terachem",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 7, "precision": "mixed"},
        ),
    )
    captured = {}

    def _fake_compute_func(program, inp_obj, **kwargs):
        captured["program"] = program
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
    engine = QCOPEngine(
        program="terachem",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "wb97xd3", "basis": "def2-svp"},
            keywords={"threads": 3, "gpus": 1},
        ),
    )
    captured = {}

    def _fake_compute_func(program, inp_obj, **kwargs):
        captured["program"] = program
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


def test_non_terachem_geomopt_default_keywords_use_coordsys():
    engine = QCOPEngine(
        program="xtb",
        compute_program="chemcloud",
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(program, inp_obj, **kwargs):
        captured["program"] = program
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    assert captured["program"] == engine.geometry_optimizer
    kw = captured["input"].keywords
    assert kw["coordsys"] == "cart"
    assert kw["maxit"] == 500


def test_non_terachem_geomopt_uses_engine_geometry_optimizer_keywords():
    engine = QCOPEngine(
        program="xtb",
        compute_program="chemcloud",
        geometry_optimizer_kwds={"coordsys": "tric", "maxit": 120, "convergence_energy": 1e-6},
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(program, inp_obj, **kwargs):
        captured["program"] = program
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    kw = captured["input"].keywords
    assert kw["coordsys"] == "tric"
    assert kw["maxit"] == 120
    assert kw["convergence_energy"] == pytest.approx(1e-6)


def test_non_terachem_geomopt_call_keywords_override_engine_defaults():
    engine = QCOPEngine(
        program="xtb",
        compute_program="chemcloud",
        geometry_optimizer_kwds={"coordsys": "tric", "maxit": 120},
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(program, inp_obj, **kwargs):
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0), keywords={"maxit": 25})

    kw = captured["input"].keywords
    assert kw["coordsys"] == "tric"
    assert kw["maxit"] == 25


def test_non_terachem_geomopt_partial_engine_keywords_keep_defaults():
    engine = QCOPEngine(
        program="xtb",
        compute_program="chemcloud",
        geometry_optimizer_kwds={"maxit": 80},
        program_args=ProgramArgs(
            model={"method": "gfn2xtb"},
            keywords={},
        ),
    )
    captured = {}

    def _fake_compute_func(program, inp_obj, **kwargs):
        captured["input"] = inp_obj
        return SimpleNamespace()

    engine.compute_func = _fake_compute_func
    _ = engine._compute_geom_opt_result(_node_at_x(1.0))

    kw = captured["input"].keywords
    assert kw["coordsys"] == "cart"
    assert kw["maxit"] == 80
