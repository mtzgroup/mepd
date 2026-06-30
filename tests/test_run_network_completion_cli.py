import json
import os
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from qcio import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.TreeNode import TreeNode
from mepd.nodes.node import StructureNode
from mepd.scripts import main_cli


@pytest.fixture(autouse=True)
def _synthetic_connectivity_matches_coordinates(monkeypatch):
    monkeypatch.setattr(
        main_cli,
        "_is_connectivity_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def _node_at_x(x: float) -> StructureNode:
    node = StructureNode(structure=_structure_at_x(x))
    node._cached_energy = float(x)
    node._cached_gradient = np.zeros_like(node.coords)
    return node


def _chain_from_xs(xs: list[float], params: ChainInputs) -> Chain:
    return Chain.model_validate(
        {"nodes": [_node_at_x(x) for x in xs], "parameters": params}
    )


def _chain_signature(chain: Chain) -> list[float]:
    xs: list[float] = []
    for node in chain.nodes:
        x = float(node.coords[1][0])
        if not xs or not np.isclose(xs[-1], x):
            xs.append(x)
    return xs


class _FakeNEB:
    def __init__(self, chain: Chain):
        self.chain_trajectory = [chain]
        self.optimized = chain
        self.grad_calls_made = 0

    def write_to_disk(self, fp, write_history=True, write_qcio=False):
        self.chain_trajectory[-1].write_to_disk(fp)


def _history_from_segments(
    segments: list[tuple[float, float]], params: ChainInputs
) -> TreeNode:
    root_chain = _chain_from_xs([segments[0][0], segments[-1][-1]], params)
    root = TreeNode(data=_FakeNEB(root_chain), children=[], index=0)
    for i, (start, end) in enumerate(segments, start=1):
        leaf_chain = _chain_from_xs([start, end], params)
        root.children.append(TreeNode(data=_FakeNEB(leaf_chain), children=[], index=i))
    return root


class _FakePot:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node(0, td=_node_at_x(0.0))
        self.graph.add_node(1, td=_node_at_x(4.0))
        self.graph.add_edge(0, 1, barrier=1.0, list_of_nebs=[_chain_from_xs([0.0, 4.0], ChainInputs())])

    def write_to_disk(self, fp):
        fp.write_text(
            json.dumps(
                {
                    "nodes": {
                        str(node): {
                            "root": bool(data.get("root")),
                            "requested_target": bool(data.get("requested_target")),
                        }
                        for node, data in self.graph.nodes(data=True)
                    }
                }
            )
        )


class _RegularFakeMSMEP:
    def __init__(self, inputs):
        self.inputs = inputs
        self.recursive_calls = 0
        self.regular_calls = 0

    def run_recursive_minimize(self, input_chain: Chain):
        self.recursive_calls += 1
        return _history_from_segments([(0.0, 1.0), (1.0, 2.0)], self.inputs.chain_inputs)

    def run_minimize_chain(self, input_chain: Chain):
        self.regular_calls += 1
        chain = _chain_from_xs([0.0, 1.0, 2.0], self.inputs.chain_inputs)
        neb = _FakeNEB(chain)
        neb.geom_grad_calls_made = 0
        return neb, SimpleNamespace(is_elem_step=True)


class _ParallelFakeMSMEP:
    def __init__(self, inputs):
        self.inputs = inputs
        self.parallel_calls = 0
        self.recursive_calls = 0
        self.regular_calls = 0
        self.requested_max_workers = None

    def run_parallel_recursive_minimize(self, input_chain: Chain, max_workers: int | None = None):
        self.parallel_calls += 1
        self.requested_max_workers = max_workers
        history = _history_from_segments([(0.0, 1.0), (1.0, 2.0)], self.inputs.chain_inputs)
        history.children.append(TreeNode(data=None, children=[], index=3))
        return history

    def run_recursive_minimize(self, input_chain: Chain):
        self.recursive_calls += 1
        return _history_from_segments([(0.0, 1.0), (1.0, 2.0)], self.inputs.chain_inputs)

    def run_minimize_chain(self, input_chain: Chain):
        self.regular_calls += 1
        chain = _chain_from_xs([0.0, 1.0, 2.0], self.inputs.chain_inputs)
        neb = _FakeNEB(chain)
        neb.geom_grad_calls_made = 0
        return neb, SimpleNamespace(is_elem_step=True)


class _ParallelAllFailedMSMEP:
    def __init__(self, inputs):
        self.inputs = inputs
        self.parallel_calls = 0
        self.requested_max_workers = None

    def run_parallel_recursive_minimize(
        self, input_chain: Chain, max_workers: int | None = None
    ):
        self.parallel_calls += 1
        self.requested_max_workers = max_workers
        root_chain = _chain_from_xs([0.0, 2.0], self.inputs.chain_inputs)
        history = TreeNode(
            data=_FakeNEB(root_chain),
            children=[
                TreeNode(data=None, children=[], index=1),
                TreeNode(data=None, children=[], index=2),
            ],
            index=0,
        )
        history.parallel_failures = [
            "branch-1: worker failed",
            "branch-2: worker failed",
        ]
        return history


class _ParallelIdenticalSkipMSMEP:
    def __init__(self, inputs):
        self.inputs = inputs
        self.parallel_calls = 0

    def run_parallel_recursive_minimize(
        self, input_chain: Chain, max_workers: int | None = None
    ):
        self.parallel_calls += 1
        history = _history_from_segments([(0.0, 1.0)], self.inputs.chain_inputs)
        skipped = TreeNode(data=None, children=[], index=2)
        skipped.leaf_status = "identical_endpoints"
        history.children.append(skipped)
        return history


def test_recursive_split_attempted_pair_registry_uses_connectivity_only(monkeypatch):
    params = ChainInputs(node_rms_thre=0.01, node_ene_thre=0.001)
    registry: list[StructureNode] = []

    first = _node_at_x(1.0)
    second = _node_at_x(9.0)
    first._cached_energy = 1.0
    second._cached_energy = 99.0

    monkeypatch.setattr(main_cli, "is_identical", lambda *args, **kwargs: False)

    def _fake_connectivity_key(node: StructureNode) -> str:
        x = float(node.coords[1][0])
        return "same-graph-and-stereo" if x in {1.0, 9.0} else f"x={x}"

    monkeypatch.setattr(
        main_cli,
        "_is_connectivity_identical",
        lambda a, b, **kwargs: _fake_connectivity_key(a) == _fake_connectivity_key(b),
    )

    first_index = main_cli._register_recursive_split_node(
        first,
        registry=registry,
        chain_inputs=params,
        connectivity_only=True,
    )
    second_index = main_cli._register_recursive_split_node(
        second,
        registry=registry,
        chain_inputs=params,
        connectivity_only=True,
    )

    assert second_index == first_index
    assert len(registry) == 1


def test_run_recursive_network_completion_enqueues_all_nonadjacent_path_pairs(monkeypatch, tmp_path):
    params = ChainInputs()
    expensive_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )

    histories = {
        (0.0, 4.0): _history_from_segments(
            [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)], params
        ),
        (0.0, 2.0): _history_from_segments([(0.0, 2.0)], params),
        (1.0, 4.0): _history_from_segments([(1.0, 5.0), (5.0, 4.0)], params),
    }
    calls: list[tuple[float, float]] = []
    built_dirs: list[str] = []

    class _FakeMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_recursive_minimize(self, input_chain: Chain):
            pair = (
                float(input_chain[0].coords[1][0]),
                float(input_chain[-1].coords[1][0]),
            )
            calls.append(pair)
            return histories[pair]

    class _FakeNetworkBuilder:
        def __init__(self, data_dir, start, end, network_inputs, chain_inputs):
            self.data_dir = data_dir
            self.msmep_data_dir = None

        def create_rxn_network_from_paths(self, msmep_paths):
            nonlocal built_dirs
            built_dirs = sorted(p.name for p in msmep_paths if p.is_dir())
            return _FakePot()

        def create_rxn_network(self, file_pattern="*_msmep"):
            nonlocal built_dirs
            built_dirs = sorted(
                p.name for p in self.msmep_data_dir.glob(file_pattern) if p.is_dir()
            )
            return _FakePot()

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: expensive_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(main_cli, "NetworkBuilder", _FakeNetworkBuilder)
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(4.0)],
    )

    monkeypatch.chdir(tmp_path)
    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=True,
        network_completion=True,
        network_completion_mode="all-to-all",
        name="rgs",
    )

    assert calls == [(0.0, 4.0), (0.0, 2.0), (1.0, 4.0)]
    assert built_dirs == ["request_0_msmep", "request_1_msmep", "request_2_msmep", "rgs"]

    manifest = json.loads(
        (tmp_path / "rgs_network_completion" / "rgs_request_manifest.json").read_text()
    )
    assert manifest["run_state"] == "completed"
    assert manifest["total_requests"] == 3
    assert manifest["counts"]["completed"] == 3
    assert [row["request_id"] for row in manifest["requests"]] == [0, 1, 2]
    assert [row["status"] for row in manifest["requests"]] == [
        "completed",
        "completed",
        "completed",
    ]
    assert manifest["requests"][1]["reason"].startswith("all-to-all mode")
    assert manifest["requests"][2]["reason"].startswith("all-to-all mode")

    network_dump = json.loads(
        (tmp_path / "rgs_network_completion" / "rgs_network.json").read_text()
    )
    assert network_dump["nodes"]["0"]["root"] is True
    assert network_dump["nodes"]["1"]["requested_target"] is True
    best_path_meta = json.loads(
        (tmp_path / "rgs_network_completion" / "rgs_best_path.json").read_text()
    )
    assert best_path_meta["root_index"] == 0
    assert best_path_meta["target_index"] == 1
    assert best_path_meta["path"] == [0, 1]
    assert (tmp_path / "rgs_network_completion" / "rgs_best_path.xyz").exists()
    assert (tmp_path / "rgs_network_completion" / "rgs_best_path.energies").exists()

def test_run_recursive_network_completion_skips_pairs_already_seen_on_completed_paths(monkeypatch, tmp_path):
    params = ChainInputs()
    expensive_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )

    histories = {
        (0.0, 4.0): _history_from_segments(
            [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)], params
        ),
        (0.0, 2.0): _history_from_segments(
            [(0.0, 5.0), (5.0, 2.0)], params
        ),
        (1.0, 4.0): _history_from_segments(
            [(1.0, 5.0), (5.0, 2.0), (2.0, 4.0)], params
        ),
        (5.0, 4.0): _history_from_segments([(5.0, 4.0)], params),
    }
    calls: list[tuple[float, float]] = []

    class _FakeMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_recursive_minimize(self, input_chain: Chain):
            pair = (
                float(input_chain[0].coords[1][0]),
                float(input_chain[-1].coords[1][0]),
            )
            calls.append(pair)
            return histories[pair]

    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(main_cli, "NetworkBuilder", lambda *args, **kwargs: SimpleNamespace(msmep_data_dir=None, create_rxn_network=lambda file_pattern="*_msmep": _FakePot()))
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    request_records, _network_fp, _manifest_fp, _cost_summary = main_cli._run_recursive_network_completion(
        history=histories[(0.0, 4.0)],
        program_input=expensive_inputs,
        initial_start=_node_at_x(0.0),
        initial_end=_node_at_x(4.0),
        output_dir=tmp_path / "splits",
        base_name="rgs",
        split_mode="all-to-all",
    )

    assert calls == [(0.0, 2.0), (1.0, 4.0), (5.0, 4.0)]
    assert [row["request_id"] for row in request_records] == [0, 1, 2, 3]


def test_run_recursive_network_completion_linear_mode_uses_original_endpoints(
    monkeypatch, tmp_path
):
    params = ChainInputs()
    expensive_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )

    histories = {
        (0.0, 4.0): _history_from_segments(
            [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)], params
        ),
        (1.0, 4.0): _history_from_segments(
            [(1.0, 5.0), (5.0, 4.0)], params
        ),
        (0.0, 2.0): _history_from_segments([(0.0, 2.0)], params),
        (0.0, 5.0): _history_from_segments([(0.0, 5.0)], params),
        (5.0, 4.0): _history_from_segments([(5.0, 4.0)], params),
    }
    calls: list[tuple[float, float]] = []

    class _FakeMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_recursive_minimize(self, input_chain: Chain, attempted_pairs_payload=None):
            pair = (
                float(input_chain[0].coords[1][0]),
                float(input_chain[-1].coords[1][0]),
            )
            calls.append(pair)
            return histories[pair]

    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(
        main_cli,
        "NetworkBuilder",
        lambda *args, **kwargs: SimpleNamespace(
            msmep_data_dir=None,
            create_rxn_network=lambda file_pattern="*_msmep": _FakePot(),
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    request_records, _network_fp, _manifest_fp, _cost_summary = (
        main_cli._run_recursive_network_completion(
            history=histories[(0.0, 4.0)],
            program_input=expensive_inputs,
            initial_start=_node_at_x(0.0),
            initial_end=_node_at_x(4.0),
            output_dir=tmp_path / "splits",
            base_name="rgs",
        )
    )

    assert calls == [(1.0, 4.0), (0.0, 2.0), (0.0, 5.0)]
    assert [row["request_id"] for row in request_records] == [0, 1, 2, 3]


def test_run_recursive_network_completion_returns_followup_geometry_costs(
    monkeypatch, tmp_path
):
    params = ChainInputs()
    expensive_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )

    def _with_leaf_costs(history, total: int, geom: int):
        leaf_nebs = [leaf.data for leaf in history.ordered_leaves if leaf.data]
        assert leaf_nebs
        leaf_nebs[0].grad_calls_made = total
        leaf_nebs[0].geom_grad_calls_made = geom
        return history

    initial_history = _history_from_segments(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)], params
    )
    histories = {
        (0.0, 2.0): _with_leaf_costs(
            _history_from_segments([(0.0, 2.0)], params), total=3, geom=1
        ),
        (0.0, 3.0): _with_leaf_costs(
            _history_from_segments([(0.0, 3.0)], params), total=5, geom=2
        ),
        (1.0, 3.0): _with_leaf_costs(
            _history_from_segments([(1.0, 3.0)], params), total=13, geom=5
        ),
        (1.0, 4.0): _with_leaf_costs(
            _history_from_segments([(1.0, 4.0)], params), total=11, geom=4
        ),
        (2.0, 4.0): _with_leaf_costs(
            _history_from_segments([(2.0, 4.0)], params), total=7, geom=3
        ),
    }

    class _FakeMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_recursive_minimize(self, input_chain: Chain, attempted_pairs_payload=None):
            pair = (
                float(input_chain[0].coords[1][0]),
                float(input_chain[-1].coords[1][0]),
            )
            return histories[pair]

    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(
        main_cli,
        "NetworkBuilder",
        lambda *args, **kwargs: SimpleNamespace(
            msmep_data_dir=None,
            create_rxn_network=lambda file_pattern="*_msmep": _FakePot(),
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    request_records, _network_fp, manifest_fp, cost_summary = (
        main_cli._run_recursive_network_completion(
            history=initial_history,
            program_input=expensive_inputs,
            initial_start=_node_at_x(0.0),
            initial_end=_node_at_x(4.0),
            output_dir=tmp_path / "splits",
            base_name="rgs",
            split_mode="all-to-all",
        )
    )

    assert cost_summary["gradient_calls_total"] == 39
    assert cost_summary["gradient_calls_geometry_optimizations"] == 15
    completed_followups = [
        row for row in request_records if int(row["request_id"]) in {1, 2, 3, 4, 5}
    ]
    assert sum(row["gradient_calls_total"] for row in completed_followups) == 39
    assert (
        sum(
            row["gradient_calls_geometry_optimizations"]
            for row in completed_followups
        )
        == 15
    )
    manifest = json.loads(manifest_fp.read_text())
    manifest_followups = [
        row for row in manifest["requests"] if int(row["request_id"]) in {1, 2, 3, 4, 5}
    ]
    assert (
        sum(
            row["gradient_calls_geometry_optimizations"]
            for row in manifest_followups
        )
        == 15
    )


def test_run_recursive_network_completion_excludes_active_request_from_attempted_payload(
    monkeypatch, tmp_path
):
    params = ChainInputs()
    expensive_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )

    initial_history = _history_from_segments(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)], params
    )
    followup_histories = {
        (0.0, 2.0): _history_from_segments([(0.0, 2.0)], params),
        (1.0, 4.0): _history_from_segments([(1.0, 4.0)], params),
    }
    seen_payload_pairs_by_request: dict[tuple[float, float], list[tuple[float, float]]] = {}

    class _FakeMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_recursive_minimize(
            self, input_chain: Chain, attempted_pairs_payload=None
        ):
            pair = (
                float(input_chain[0].coords[1][0]),
                float(input_chain[-1].coords[1][0]),
            )
            request_payload_pairs: list[tuple[float, float]] = []
            for item in list(attempted_pairs_payload or []):
                start_node = StructureNode.from_serializable(item["start"])
                end_node = StructureNode.from_serializable(item["end"])
                request_payload_pairs.append(
                    (float(start_node.coords[1][0]), float(end_node.coords[1][0]))
                )
            seen_payload_pairs_by_request[pair] = request_payload_pairs
            if pair not in followup_histories:
                raise AssertionError(f"Unexpected follow-up request pair {pair}")
            return followup_histories[pair]

    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(
        main_cli,
        "NetworkBuilder",
        lambda *args, **kwargs: SimpleNamespace(
            msmep_data_dir=None,
            create_rxn_network=lambda file_pattern="*_msmep": _FakePot(),
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    request_records, _network_fp, _manifest_fp, _cost_summary = main_cli._run_recursive_network_completion(
        history=initial_history,
        program_input=expensive_inputs,
        initial_start=_node_at_x(0.0),
        initial_end=_node_at_x(4.0),
        output_dir=tmp_path / "splits",
        base_name="rgs",
    )

    for pair, payload_pairs in seen_payload_pairs_by_request.items():
        assert pair not in payload_pairs
    assert [row["status"] for row in request_records] == [
        "completed",
        "completed",
        "completed",
    ]


def test_run_recursive_network_completion_overwrite_rebuilds_existing_output(
    monkeypatch, tmp_path
):
    params = ChainInputs()
    expensive_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    output_dir = tmp_path / "splits"
    output_dir.mkdir(parents=True)
    stale_file = output_dir / "stale.txt"
    stale_file.write_text("old")
    (output_dir / "rgs_request_manifest.json").write_text("{}")
    (output_dir / "request_0_msmep").mkdir()

    class _FakeMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_recursive_minimize(self, input_chain: Chain, attempted_pairs_payload=None):
            start = float(input_chain[0].coords[1][0])
            end = float(input_chain[-1].coords[1][0])
            return _history_from_segments([(start, end)], self.inputs.chain_inputs)

    monkeypatch.setattr(main_cli, "MSMEP", _FakeMSMEP)
    monkeypatch.setattr(
        main_cli,
        "NetworkBuilder",
        lambda *args, **kwargs: SimpleNamespace(
            msmep_data_dir=None,
            create_rxn_network=lambda file_pattern="*_msmep": _FakePot(),
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    request_records, _network_fp, _manifest_fp, _cost_summary = main_cli._run_recursive_network_completion(
        history=_history_from_segments([(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)], params),
        program_input=expensive_inputs,
        initial_start=_node_at_x(0.0),
        initial_end=_node_at_x(4.0),
        output_dir=output_dir,
        base_name="rgs",
        overwrite=True,
    )

    assert not stale_file.exists()
    assert [row["status"] for row in request_records] == [
        "completed",
        "completed",
        "completed",
    ]


def test_ordered_leaf_path_nodes_prunes_trailing_loop_to_repeated_target():
    params = ChainInputs()
    history = _history_from_segments(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 4.0)],
        params,
    )

    path_nodes = main_cli._ordered_leaf_path_nodes(history, params)
    assert [float(node.coords[1][0]) for node in path_nodes] == [0.0, 1.0, 2.0, 4.0]
    assert _chain_signature(history.output_chain) == [0.0, 1.0, 2.0, 4.0]


def test_ordered_leaf_path_nodes_prunes_backtrack_cycle_to_earlier_start():
    params = ChainInputs()
    history = _history_from_segments(
        [(0.0, 1.0), (1.0, 0.0), (0.0, 2.0), (2.0, 4.0)],
        params,
    )

    path_nodes = main_cli._ordered_leaf_path_nodes(history, params)
    assert [float(node.coords[1][0]) for node in path_nodes] == [0.0, 2.0, 4.0]
    assert _chain_signature(history.output_chain) == [0.0, 2.0, 4.0]


def test_write_recursive_split_request_artifacts_tolerates_missing_cached_results(tmp_path):
    params = ChainInputs()
    unevaluated_chain = Chain.model_validate(
        {"nodes": [StructureNode(structure=_structure_at_x(0.0)), StructureNode(structure=_structure_at_x(0.0))], "parameters": params}
    )
    history = TreeNode(data=_FakeNEB(unevaluated_chain), children=[], index=0)
    output_dir = tmp_path / "splits"
    output_dir.mkdir()

    main_cli._write_recursive_split_request_artifacts(
        output_dir=output_dir,
        request_id=7,
        history=history,
        write_qcio=False,
    )

    assert (output_dir / "request_7.xyz").exists()
    assert not (output_dir / "request_7.energies").exists()
    assert not (output_dir / "request_7.gradients").exists()
    assert (output_dir / "request_7_msmep").is_dir()


def test_build_recursive_split_network_summary_returns_none_when_no_valid_leaves(
    monkeypatch, tmp_path
):
    params = ChainInputs()
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    (output_dir / "request_0_msmep").mkdir()

    class _FakeNetworkBuilder:
        def __init__(self, data_dir, start, end, network_inputs, chain_inputs):
            self.data_dir = data_dir
            self.msmep_data_dir = None

        def create_rxn_network(self, file_pattern="*_msmep"):
            raise ValueError(
                "No valid elementary-step leaves were found while building the reaction network."
            )

    monkeypatch.setattr(main_cli, "NetworkBuilder", _FakeNetworkBuilder)

    network_fp = main_cli._build_recursive_split_network_summary(
        output_dir=output_dir,
        base_name="rgs",
        chain_inputs=params,
    )

    assert network_fp is None
    assert not (output_dir / "rgs_network.json").exists()


def test_build_recursive_split_network_summary_includes_source_tree(
    monkeypatch, tmp_path
):
    params = ChainInputs()
    output_dir = tmp_path / "rgs_network_completion"
    output_dir.mkdir()
    source_tree = tmp_path / "rgs"
    source_tree.mkdir()
    (source_tree / "adj_matrix.txt").write_text("0 1\n1 0\n")
    (output_dir / "request_0_msmep").mkdir()
    captured_paths: list[str] = []

    class _FakeNetworkBuilder:
        def __init__(self, data_dir, start, end, network_inputs, chain_inputs):
            self.data_dir = data_dir
            self.msmep_data_dir = None

        def create_rxn_network_from_paths(self, msmep_paths):
            nonlocal captured_paths
            captured_paths = [p.name for p in msmep_paths]
            return _FakePot()

    monkeypatch.setattr(main_cli, "NetworkBuilder", _FakeNetworkBuilder)
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)

    network_fp = main_cli._build_recursive_split_network_summary(
        output_dir=output_dir,
        base_name="rgs",
        chain_inputs=params,
        source_tree_dir=source_tree,
    )

    assert network_fp == output_dir / "rgs_network.json"
    assert captured_paths == ["rgs", "request_0_msmep"]


def test_run_network_completion_forces_recursive_mode(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _RegularFakeMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "NetworkBuilder", lambda *args, **kwargs: SimpleNamespace(msmep_data_dir=None, create_rxn_network=lambda file_pattern="*_msmep": _FakePot()))
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)])
    monkeypatch.setattr(main_cli, "is_identical", lambda a, b, **kwargs: np.allclose(a.coords, b.coords))
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        network_completion=True,
        name="forced",
    )

    assert runner.recursive_calls == 1
    assert runner.regular_calls == 0
    status_payload = json.loads((tmp_path / "forced_status.json").read_text())
    assert status_payload["recursive"] is True
    assert status_payload["network_completion"] is True


def test_run_accepts_network_completion_directory_as_geometries_input(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _RegularFakeMSMEP(program_inputs)
    source_chain = _chain_from_xs([0.0, 1.0, 2.0], params)
    source_dir = tmp_path / "source_network_completion"
    source_dir.mkdir()
    (source_dir / "source_network.json").write_text("{}")

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "_load_best_path_chain_from_network_json",
        lambda _path: source_chain,
    )
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("read_multiple_structure_from_file should not be used for directory input")
        ),
    )
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries=str(source_dir),
        inputs="dummy.toml",
        recursive=False,
        network_completion=False,
        name="dir_input",
    )

    assert runner.regular_calls == 1
    assert (tmp_path / "dir_input.xyz").exists()


def test_run_network_completion_directory_input_resumes_from_request0_history(
    monkeypatch, tmp_path
):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _RegularFakeMSMEP(program_inputs)
    source_chain = _chain_from_xs([0.0, 1.0, 2.0], params)
    source_history = _history_from_segments([(0.0, 1.0), (1.0, 2.0)], params)
    source_dir = tmp_path / "source_network_completion"
    source_dir.mkdir()
    (source_dir / "source_network.json").write_text("{}")
    (source_dir / "source_request_manifest.json").write_text("{}")
    (source_dir / "request_0_msmep").mkdir()

    captured = {}

    def _fake_run_recursive_network_completion(**kwargs):
        captured.update(kwargs)
        return [], None, source_dir / "source_request_manifest.json", {
            "gradient_calls_total": 0,
            "gradient_calls_geometry_optimizations": 0,
        }

    def _fake_load_request_history(
        output_dir, request_id, chain_inputs, engine, charge, multiplicity
    ):
        if Path(output_dir) == source_dir and int(request_id) == 0:
            return source_history
        return None

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "_load_best_path_chain_from_network_json",
        lambda _path: source_chain,
    )
    monkeypatch.setattr(main_cli, "_maybe_resume_recursive_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "_load_recursive_split_request_history",
        _fake_load_request_history,
    )
    monkeypatch.setattr(
        main_cli,
        "_run_recursive_network_completion",
        _fake_run_recursive_network_completion,
    )
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("read_multiple_structure_from_file should not be used for directory input")
        ),
    )
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries=str(source_dir),
        inputs="dummy.toml",
        recursive=False,
        network_completion=True,
        overwrite=True,
        name="dir_resume",
    )

    assert runner.recursive_calls == 0
    assert Path(captured["output_dir"]) == source_dir
    assert captured["base_name"] == "source"
    assert captured["overwrite"] is True
    assert captured["overwrite_followups_only"] is True
    assert captured["split_mode"] == "linear"
    assert captured["history"] is source_history


def test_run_cost_report_includes_network_completion_followup_estimate(
    monkeypatch, tmp_path
):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _RegularFakeMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(
        main_cli,
        "NetworkBuilder",
        lambda *args, **kwargs: SimpleNamespace(
            msmep_data_dir=None,
            create_rxn_network=lambda file_pattern="*_msmep": _FakePot(),
        ),
    )
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)],
    )
    monkeypatch.setattr(
        main_cli,
        "is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )
    monkeypatch.setattr(
        main_cli,
        "_estimate_grad_calls_from_network_completion_dir",
        lambda _network_dir, include_request0=True: 7 if not include_request0 else 9,
    )
    monkeypatch.setattr(
        main_cli,
        "_run_recursive_network_completion",
        lambda **_kwargs: (
            [],
            None,
            tmp_path / "cost_with_splits_request_manifest.json",
            {
                "gradient_calls_total": 7,
                "gradient_calls_geometry_optimizations": 3,
            },
        ),
    )
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        network_completion=True,
        name="cost_with_splits",
    )

    payload = json.loads((tmp_path / "cost_with_splits_cost.json").read_text())
    assert payload["gradient_calls_total"] == 7
    assert payload["gradient_calls_geometry_optimizations"] == 3
    assert payload["metadata"]["gradient_calls_primary"] == 0
    assert payload["metadata"]["gradient_calls_network_completion_followup_estimated"] == 7
    assert (
        payload["metadata"][
            "gradient_calls_geometry_optimizations_network_completion_followup"
        ]
        == 3
    )


def test_parallel_run_rejects_recursive_mode():
    with pytest.raises(Exception, match="--parallel cannot be combined with --recursive"):
        main_cli.run(
            recursive=True,
            parallel=True,
        )


def test_parallel_run_supports_network_completion_and_uses_parallel_runner(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )

    class _ParallelNetworkSplitMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs
            self.parallel_calls = 0
            self.recursive_calls = 0
            self.requested_max_workers: list[int | None] = []

        def run_parallel_recursive_minimize(
            self, input_chain: Chain, max_workers: int | None = None
        ):
            self.parallel_calls += 1
            self.requested_max_workers.append(max_workers)
            pair = (
                float(input_chain[0].coords[1][0]),
                float(input_chain[-1].coords[1][0]),
            )
            if pair == (0.0, 3.0):
                return _history_from_segments(
                    [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
                    self.inputs.chain_inputs,
                )
            if pair == (1.0, 3.0):
                return _history_from_segments([(1.0, 3.0)], self.inputs.chain_inputs)
            if pair == (0.0, 2.0):
                return _history_from_segments([(0.0, 2.0)], self.inputs.chain_inputs)
            raise AssertionError(f"Unexpected pair request: {pair}")

        def run_recursive_minimize(self, input_chain: Chain):
            self.recursive_calls += 1
            raise AssertionError(
                "run_recursive_minimize should not be used when --parallel is enabled."
            )

    runner = _ParallelNetworkSplitMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "NetworkBuilder", lambda *args, **kwargs: SimpleNamespace(msmep_data_dir=None, create_rxn_network=lambda file_pattern="*_msmep": _FakePot()))
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(3.0)])
    monkeypatch.setattr(main_cli, "is_identical", lambda a, b, **kwargs: np.allclose(a.coords, b.coords))
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        parallel=True,
        parallel_workers=6,
        network_completion=True,
        name="parallel_splits",
    )

    assert runner.parallel_calls == 3
    assert runner.recursive_calls == 0
    assert runner.requested_max_workers == [6, 6, 6]

    snapshot = main_cli._load_status_snapshot(str(tmp_path / "parallel_splits.xyz"))
    assert snapshot["run_status"]["parallel"] is True
    assert snapshot["run_status"]["recursive"] is False
    assert snapshot["run_status"]["network_completion"] is True
    manifest = snapshot.get("manifest") or {}
    assert manifest.get("total_requests") == 3
    assert manifest.get("counts", {}).get("completed") == 3


def test_parallel_run_uses_parallel_recursive_runner_and_writes_partial_status(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _ParallelFakeMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)])
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        parallel=True,
        parallel_workers=999,
        network_completion=False,
        name="parallel_case",
    )

    assert runner.parallel_calls == 1
    assert runner.recursive_calls == 0
    assert runner.regular_calls == 0
    assert runner.requested_max_workers == 999

    snapshot = main_cli._load_status_snapshot(str(tmp_path / "parallel_case.xyz"))
    assert snapshot["run_status"]["parallel"] is True
    assert snapshot["run_status"]["recursive"] is False
    assert snapshot["run_status"]["network_completion"] is False


def test_parallel_run_falls_back_to_root_chain_when_all_child_leaves_fail(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _ParallelAllFailedMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)],
    )
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        parallel=True,
        parallel_workers=4,
        network_completion=False,
        name="parallel_all_failed",
    )

    assert runner.parallel_calls == 1
    assert (tmp_path / "parallel_all_failed.xyz").exists()
    snapshot = main_cli._load_status_snapshot(str(tmp_path / "parallel_all_failed.xyz"))
    assert snapshot["run_status"]["run_state"] == "completed"
    assert snapshot["run_status"]["parallel"] is True


def test_parallel_run_reports_identical_endpoint_skips_not_failures(
    monkeypatch, tmp_path, capsys
):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _ParallelIdenticalSkipMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)],
    )
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        parallel=True,
        parallel_workers=2,
        network_completion=False,
        name="parallel_identical_skip",
    )

    out = capsys.readouterr().out
    assert "were skipped because endpoints were identical" in out
    assert "parallel branch(es) failed" not in out


def test_run_network_completion_resumes_from_saved_tree_and_request_artifacts(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="NEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
        write_qcio=False,
    )

    root_history = _history_from_segments(
        [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)], params
    )
    network_dir = tmp_path / "resume_network_completion"
    main_cli._write_recursive_split_request_artifacts(
        output_dir=network_dir,
        request_id=0,
        history=root_history,
        write_qcio=False,
    )
    manifest_fp = main_cli._write_recursive_split_manifest(
        output_dir=network_dir,
        base_name="resume",
        request_records=[
            {
                "request_id": 0,
                "parent_request_id": None,
                "start_index": 0,
                "end_index": 1,
                "status": "completed",
            }
        ],
        run_state="running",
            current_request_id=None,
            network_fp=None,
        )
    main_cli._write_run_status(
        tmp_path / "resume_status.json",
        base_name="resume",
        run_state="running",
        phase="network_completion",
        recursive=True,
        network_completion=True,
        path_min_method="NEB",
        output_chain_path=tmp_path / "resume.xyz",
        tree_path=tmp_path / "resume",
        network_completion_dir=network_dir,
        manifest_fp=manifest_fp,
    )

    child_histories = {
        (0.0, 2.0): _history_from_segments([(0.0, 2.0)], params),
        (0.0, 3.0): _history_from_segments([(0.0, 3.0)], params),
        (1.0, 3.0): _history_from_segments([(1.0, 3.0)], params),
        (1.0, 4.0): _history_from_segments([(1.0, 5.0), (5.0, 4.0)], params),
        (2.0, 4.0): _history_from_segments([(2.0, 4.0)], params),
    }
    calls: list[tuple[float, float]] = []

    class _ResumeMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_recursive_minimize(self, input_chain: Chain):
            pair = (
                float(input_chain[0].coords[1][0]),
                float(input_chain[-1].coords[1][0]),
            )
            calls.append(pair)
            return child_histories[pair]

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", _ResumeMSMEP)
    monkeypatch.setattr(main_cli, "NetworkBuilder", lambda *args, **kwargs: SimpleNamespace(msmep_data_dir=None, create_rxn_network=lambda file_pattern="*_msmep": _FakePot()))
    monkeypatch.setattr(
        main_cli,
        "_match_network_endpoint_indices_by_connectivity",
        lambda pot, start_node, end_node: {"root_index": 0, "target_index": 1},
    )
    monkeypatch.setattr(main_cli, "plot_results_from_pot_obj", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli, "_maybe_resume_recursive_history", lambda *args, **kwargs: root_history)
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(4.0)])
    monkeypatch.setattr(main_cli, "is_identical", lambda a, b, **kwargs: np.allclose(a.coords, b.coords))
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=True,
        network_completion=True,
        network_completion_mode="all-to-all",
        name="resume",
    )

    assert calls == [(0.0, 2.0), (0.0, 3.0), (1.0, 3.0), (1.0, 4.0), (2.0, 4.0)]
    manifest = json.loads((network_dir / "resume_request_manifest.json").read_text())
    assert manifest["counts"]["completed"] == 6


def test_nonrecursive_run_writes_status_snapshot(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="FNEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _RegularFakeMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)])
    monkeypatch.setattr(main_cli, "_write_neb_results_with_history", lambda n, filename: filename.write_text("ok") or True)
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        network_completion=False,
        name="plain",
    )

    snapshot = main_cli._load_status_snapshot(str(tmp_path / "plain.xyz"))
    assert snapshot["run_status"]["recursive"] is False
    assert snapshot["run_status"]["network_completion"] is False
    assert snapshot["run_status"]["path_min_method"] == "FNEB"
    assert snapshot["run_status"]["run_state"] == "completed"


def test_run_minimize_ends_tolerates_empty_batch_endpoint_trajectory(
    monkeypatch, tmp_path, capsys
):
    params = ChainInputs()
    optimized_start = _node_at_x(0.25)

    class _EndpointBatchEngine:
        def compute_geometry_optimizations(self, nodes, keywords=None):
            return [[optimized_start], []]

    class _CaptureMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs
            self.endpoints: tuple[float, float] | None = None

        def run_minimize_chain(self, input_chain: Chain):
            self.endpoints = (
                float(input_chain[0].coords[1][0]),
                float(input_chain[-1].coords[1][0]),
            )
            chain = _chain_from_xs(
                [self.endpoints[0], 1.0, self.endpoints[1]],
                self.inputs.chain_inputs,
            )
            neb = _FakeNEB(chain)
            neb.geom_grad_calls_made = 0
            return neb, SimpleNamespace(is_elem_step=True)

    program_inputs = SimpleNamespace(
        path_min_method="FNEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=_EndpointBatchEngine(),
    )
    runner = _CaptureMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)],
    )
    monkeypatch.setattr(
        main_cli,
        "_write_neb_results_with_history",
        lambda n, filename: filename.write_text("ok") or True,
    )
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        network_completion=False,
        minimize_ends=True,
        name="minends_safe",
    )

    out = capsys.readouterr().out
    assert "empty trajectory" in out
    assert runner.endpoints == (0.25, 2.0)
    assert (tmp_path / "minends_safe.xyz").exists()


def test_run_minimize_ends_prefers_geometry_optimizer_kwds(monkeypatch, tmp_path):
    params = ChainInputs()

    class _TrackingEngine:
        def __init__(self, shift: float):
            self.shift = shift
            self.batch_call_sizes: list[int] = []
            self.batch_call_keywords: list[object] = []

        def compute_geometry_optimization(self, node: StructureNode, keywords=None):
            moved = node.update_coords(node.coords + np.array([[self.shift, 0.0, 0.0]]))
            moved._cached_energy = float(np.linalg.norm(moved.coords[1] - moved.coords[0]))
            moved._cached_gradient = np.zeros_like(moved.coords)
            return [moved]

        def compute_geometry_optimizations(self, nodes: list[StructureNode], keywords=None):
            self.batch_call_sizes.append(len(nodes))
            self.batch_call_keywords.append(keywords)
            return [
                self.compute_geometry_optimization(node=node, keywords=keywords)
                for node in nodes
            ]

    engine = _TrackingEngine(shift=0.25)
    program_inputs = SimpleNamespace(
        path_min_method="FNEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=engine,
        geometry_optimizer_kwds={"coordsys": "tric", "maxiter": 1200},
    )

    class _Runner:
        def __init__(self, inputs):
            self.inputs = inputs

        def run_minimize_chain(self, input_chain: Chain):
            chain = _chain_from_xs([0.0, 1.0, 2.0], self.inputs.chain_inputs)
            neb = _FakeNEB(chain)
            neb.geom_grad_calls_made = 0
            return neb, SimpleNamespace(is_elem_step=True)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: _Runner(inputs))
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_cli,
        "read_multiple_structure_from_file",
        lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)],
    )
    monkeypatch.setattr(
        main_cli,
        "_write_neb_results_with_history",
        lambda n, filename: filename.write_text("ok") or True,
    )
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        network_completion=False,
        minimize_ends=True,
        name="minends_kwds",
    )

    assert engine.batch_call_sizes == [2]
    assert engine.batch_call_keywords == [
        {"coordsys": "tric", "maxiter": 1200}
    ]
    assert (tmp_path / "minends_kwds.xyz").exists()


def test_run_hydrates_cached_xyz_sidecars_when_available(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="FNEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )

    class _CacheAwareMSMEP:
        def __init__(self, inputs):
            self.inputs = inputs
            self.cache_energies = None
            self.cache_gradients = None

        def run_minimize_chain(self, input_chain: Chain):
            self.cache_energies = [node._cached_energy for node in input_chain.nodes]
            self.cache_gradients = [np.array(node._cached_gradient) for node in input_chain.nodes]
            chain = _chain_from_xs([0.0, 1.0, 2.0], self.inputs.chain_inputs)
            neb = _FakeNEB(chain)
            neb.geom_grad_calls_made = 0
            return neb, SimpleNamespace(is_elem_step=True)

    runner = _CacheAwareMSMEP(program_inputs)

    cache_nodes = [StructureNode(structure=_structure_at_x(0.0)), StructureNode(structure=_structure_at_x(2.0))]
    cache_nodes[0]._cached_energy = 111.0
    cache_nodes[1]._cached_energy = 222.0
    cache_nodes[0]._cached_gradient = np.full_like(cache_nodes[0].coords, 1.0)
    cache_nodes[1]._cached_gradient = np.full_like(cache_nodes[1].coords, 2.0)
    cache_chain = Chain.model_validate({"nodes": cache_nodes, "parameters": params})

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)])
    monkeypatch.setattr(main_cli.Chain, "from_xyz", lambda *args, **kwargs: cache_chain)
    monkeypatch.setattr(main_cli, "_write_neb_results_with_history", lambda n, filename: filename.write_text("ok") or True)

    (tmp_path / "dummy.xyz").write_text("placeholder xyz")
    (tmp_path / "dummy.energies").write_text("placeholder")
    (tmp_path / "dummy.gradients").write_text("placeholder")
    (tmp_path / "dummy_grad_shapes.txt").write_text("2 2 3")

    monkeypatch.chdir(tmp_path)
    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        network_completion=False,
        name="with_cache",
    )

    assert runner.cache_energies == [111.0, 222.0]
    assert np.allclose(runner.cache_gradients[0], np.full((2, 3), 1.0))
    assert np.allclose(runner.cache_gradients[1], np.full((2, 3), 2.0))


def test_run_skips_xyz_cache_hydration_when_sidecars_missing(monkeypatch, tmp_path):
    params = ChainInputs()
    program_inputs = SimpleNamespace(
        path_min_method="FNEB",
        path_min_inputs=SimpleNamespace(do_elem_step_checks=True),
        chain_inputs=params,
        engine=SimpleNamespace(),
    )
    runner = _RegularFakeMSMEP(program_inputs)

    monkeypatch.setattr(main_cli.RunInputs, "open", staticmethod(lambda path: program_inputs))
    monkeypatch.setattr(main_cli, "MSMEP", lambda inputs: runner)
    monkeypatch.setattr(main_cli, "_render_runinputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_cli, "read_multiple_structure_from_file", lambda *args, **kwargs: [_structure_at_x(0.0), _structure_at_x(2.0)])
    monkeypatch.setattr(main_cli.Chain, "from_xyz", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("from_xyz should not be called when sidecars are missing")))
    monkeypatch.setattr(main_cli, "_write_neb_results_with_history", lambda n, filename: filename.write_text("ok") or True)

    (tmp_path / "dummy.xyz").write_text("placeholder xyz")
    monkeypatch.chdir(tmp_path)

    main_cli.run(
        geometries="dummy.xyz",
        inputs="dummy.toml",
        recursive=False,
        network_completion=False,
        name="without_cache",
    )

    assert runner.regular_calls == 1


def test_load_status_snapshot_prefers_run_status_and_manifest(tmp_path):
    status_fp = tmp_path / "rgs_status.json"
    manifest_fp = tmp_path / "rgs_network_completion" / "rgs_request_manifest.json"
    manifest_fp.parent.mkdir()

    manifest_fp.write_text(
        json.dumps(
            {
                "base_name": "rgs",
                "run_state": "running",
                "current_request_id": 2,
                "total_requests": 4,
                "counts": {"completed": 1, "queued": 2, "running": 1},
                "requests": [],
                "network_summary": {"node_count": 3, "edge_count": 2, "edges": [["0", "1"], ["1", "2"]]},
            }
        )
    )
    status_fp.write_text(
        json.dumps(
            {
                "base_name": "rgs",
                "run_state": "running",
                "phase": "network_completion",
                "manifest_path": str(manifest_fp),
            }
        )
    )

    snapshot = main_cli._load_status_snapshot(str(status_fp))
    assert snapshot["artifact_kind"] == "run_status"
    assert snapshot["run_status"]["phase"] == "network_completion"
    assert snapshot["manifest"]["current_request_id"] == 2


def test_load_status_snapshot_resolves_xyz_to_status_files(tmp_path):
    xyz_fp = tmp_path / "rgs.xyz"
    xyz_fp.write_text("dummy")
    status_fp = tmp_path / "rgs_status.json"
    status_fp.write_text(
        json.dumps(
            {
                "base_name": "rgs",
                "run_state": "completed",
                "phase": "complete",
            }
        )
    )

    snapshot = main_cli._load_status_snapshot(str(xyz_fp))
    assert snapshot["artifact_kind"] == "run_status"
    assert snapshot["run_status"]["run_state"] == "completed"


def test_load_status_snapshot_resolves_missing_output_target_to_status_file(tmp_path):
    status_fp = tmp_path / "rgs_status.json"
    status_fp.write_text(
        json.dumps(
            {
                "base_name": "rgs",
                "run_state": "running",
                "phase": "initial_recursive_request",
            }
        )
    )

    snapshot = main_cli._load_status_snapshot(str(tmp_path / "rgs.xyz"))
    assert snapshot["artifact_kind"] == "run_status"
    assert snapshot["run_status"]["phase"] == "initial_recursive_request"
