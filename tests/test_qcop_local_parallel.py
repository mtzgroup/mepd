from types import SimpleNamespace

import numpy as np
from qcio import Structure
from qcio.models.inputs import ProgramArgs

from mepd.chain import Chain
from mepd.engines import qcop as qcop_module
from mepd.engines.qcop import QCOPEngine
from mepd.inputs import ChainInputs
from mepd.inputs import RunInputs
from mepd.nodes.node import StructureNode


def _node(x: float) -> StructureNode:
    node = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
            symbols=["H", "H"],
            charge=0,
            multiplicity=1,
        )
    )
    node.has_molecular_graph = False
    node.graph = None
    return node


def test_qcop_crest_gradients_use_process_workers_and_preserve_order(monkeypatch):
    chain = Chain.model_validate(
        {
            "nodes": [_node(0.8), _node(1.0), _node(1.2)],
            "parameters": ChainInputs(do_parallel=True),
        }
    )
    engine = QCOPEngine(
        compute_program="qcop",
        program="crest",
        local_parallel_workers=3,
        program_args=ProgramArgs(
            model={"method": "gfn2", "basis": "gfn2"},
            keywords={"threads": 1},
        ),
    )
    captured_results = []
    captured_executor = {}

    class _FakeProcessPoolExecutor:
        def __init__(self, max_workers):
            captured_executor["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, payloads):
            captured_executor["worker"] = fn
            return [
                SimpleNamespace(image_x=float(payload[1].structure.geometry[1][0]))
                for payload in payloads
            ]

    monkeypatch.setattr(
        qcop_module.concurrent.futures,
        "ProcessPoolExecutor",
        _FakeProcessPoolExecutor,
    )
    monkeypatch.setattr(
        qcop_module,
        "update_node_cache",
        lambda node_list, results: captured_results.extend(results),
    )

    engine._run_calc(chain=chain, calctype="gradient")

    assert captured_executor["max_workers"] == 3
    assert captured_executor["worker"] is qcop_module._compute_local_qcop_input
    assert [result.image_x for result in captured_results] == [0.8, 1.0, 1.2]


def test_qcop_crest_gradients_respect_disabled_chain_parallelism(monkeypatch):
    chain = Chain.model_validate(
        {
            "nodes": [_node(0.8), _node(1.2)],
            "parameters": ChainInputs(do_parallel=False),
        }
    )
    engine = QCOPEngine(
        compute_program="qcop",
        program="crest",
        local_parallel_workers=2,
    )
    calls = []

    monkeypatch.setattr(
        engine,
        "compute_func",
        lambda *args, **kwargs: calls.append(args[1]) or object(),
    )
    monkeypatch.setattr(
        qcop_module.concurrent.futures,
        "ProcessPoolExecutor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("process pool should not be used")
        ),
    )
    monkeypatch.setattr(qcop_module, "update_node_cache", lambda node_list, results: None)

    engine._run_calc(chain=chain, calctype="gradient")

    assert len(calls) == 2


def test_run_inputs_configures_local_qcop_worker_cap():
    inputs = RunInputs(
        engine_name="qcop",
        program="crest",
        qcop_local_parallel_workers=4,
        program_kwds={
            "model": {"method": "gfn2", "basis": "gfn2"},
            "keywords": {"threads": 1},
        },
    )

    assert inputs.engine.local_parallel_workers == 4
