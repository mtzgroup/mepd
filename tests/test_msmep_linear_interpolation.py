from types import SimpleNamespace

import numpy as np
import pytest
from qcio import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
import mepd.msmep as msmep_module
from mepd.msmep import MSMEP
from mepd.nodes.node import StructureNode


def _structure(coords: np.ndarray) -> Structure:
    return Structure(
        geometry=np.array(coords, dtype=float),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def test_linear_interpolation_preserves_endpoint_geometries_and_clears_cache():
    inputs = SimpleNamespace(
        engine=SimpleNamespace(__class__=SimpleNamespace(__name__="FakeEngine")),
        path_min_method="NEB",
        chain_inputs=ChainInputs(use_geodesic_interpolation=False),
        gi_inputs=SimpleNamespace(nimages=5),
    )
    m = MSMEP(inputs=inputs)

    start = StructureNode(structure=_structure([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]))
    end = StructureNode(structure=_structure([[1.0, 0.0, 0.0], [1.0, 0.0, 1.7]]))
    start._cached_energy = -1.0
    start._cached_gradient = np.ones((2, 3))
    start._cached_result = SimpleNamespace(results=SimpleNamespace(energy=-1.0, gradient=start._cached_gradient))
    end._cached_energy = -2.0
    end._cached_gradient = np.full((2, 3), 2.0)
    end._cached_result = SimpleNamespace(results=SimpleNamespace(energy=-2.0, gradient=end._cached_gradient))

    chain = Chain.model_validate({"nodes": [start, end], "parameters": inputs.chain_inputs})

    interpolation = m._create_interpolation(chain)

    assert len(interpolation) == 5
    np.testing.assert_allclose(interpolation[0].coords, start.coords)
    np.testing.assert_allclose(interpolation[-1].coords, end.coords)
    np.testing.assert_allclose(interpolation[2].coords, np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 1.2]]))

    assert interpolation[0] is not start
    assert interpolation[-1] is not end
    assert interpolation[0]._cached_energy is None
    assert interpolation[0]._cached_gradient is None
    assert interpolation[-1]._cached_energy is None
    assert interpolation[-1]._cached_gradient is None


def test_geodesic_interpolation_uses_configured_run_geodesic(monkeypatch):
    chain_inputs = ChainInputs(use_geodesic_interpolation=True)
    gi_inputs = SimpleNamespace(
        nimages=5,
        friction=0.01,
        nudge=0.01,
        random_seed=7,
        align=True,
        extra_kwds={},
    )
    engine = SimpleNamespace(name="configured-engine")
    inputs = SimpleNamespace(
        engine=engine,
        path_min_method="NEB",
        chain_inputs=chain_inputs,
        gi_inputs=gi_inputs,
        path_min_inputs=SimpleNamespace(v=False),
    )
    m = MSMEP(inputs=inputs)

    start = StructureNode(structure=_structure([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]))
    end = StructureNode(structure=_structure([[1.0, 0.0, 0.0], [1.0, 0.0, 1.7]]))
    chain = Chain.model_validate({"nodes": [start, end], "parameters": chain_inputs})

    calls = {}

    def _fake_run_geodesic(**kwargs):
        calls.update(kwargs)
        nodes = [
            start.update_coords(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])),
            start.update_coords(np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 1.2]])),
            start.update_coords(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.7]])),
        ]
        interpolation = Chain.model_validate(
            {"nodes": nodes, "parameters": kwargs["chain_inputs"]}
        )
        return interpolation, SimpleNamespace(path=np.array([node.coords for node in nodes]))

    monkeypatch.setattr(
        msmep_module.ch,
        "run_geodesic",
        _fake_run_geodesic,
    )

    interpolation = m._create_interpolation(chain)

    assert calls["chain"] is chain
    assert calls["chain_inputs"] is not chain_inputs
    assert calls["nimages"] == 5
    assert calls["friction"] == pytest.approx(0.01)
    assert calls["nudge"] == pytest.approx(0.01)
    assert calls["random_seed"] == 7
    assert calls["align"] is True
    assert calls["return_smoother"] is True
    assert len(interpolation) == 3
    np.testing.assert_allclose(interpolation[-1].coords, end.coords)


def test_recursive_minimize_stops_at_configured_max_depth(monkeypatch):
    inputs = SimpleNamespace(
        engine=SimpleNamespace(compute_gradients=lambda chain: None),
        path_min_method="NEB",
        chain_inputs=ChainInputs(),
        gi_inputs=SimpleNamespace(nimages=2),
        path_min_inputs=SimpleNamespace(
            v=False,
            skip_identical_graphs=False,
            recursive_split_max_depth=0,
        ),
    )
    m = MSMEP(inputs=inputs)

    start = StructureNode(structure=_structure([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]))
    end = StructureNode(structure=_structure([[1.0, 0.0, 0.0], [1.0, 0.0, 1.7]]))
    chain = Chain.model_validate({"nodes": [start, end], "parameters": ChainInputs()})

    class FakeNEB:
        def __init__(self):
            self.chain_trajectory = [chain]

    monkeypatch.setattr(
        m,
        "run_minimize_chain",
        lambda input_chain: (
            FakeNEB(),
            SimpleNamespace(
                is_elem_step=False,
                splitting_criterion="minima",
                minimization_results=[chain[0]],
            ),
        ),
    )
    monkeypatch.setattr(
        m,
        "make_sequence_of_chains",
        lambda **kwargs: [chain],
    )

    history = m.run_recursive_minimize(chain)

    assert getattr(history, "leaf_status", None) == "max_depth_reached"
    assert history.children == []
