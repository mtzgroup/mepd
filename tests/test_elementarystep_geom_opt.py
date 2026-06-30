from types import SimpleNamespace

import numpy as np
import pytest
from qcio import Structure

from mepd.elementarystep import _run_geom_opt
from mepd.errors import ElectronicStructureError
from mepd.nodes.node import StructureNode


def _node() -> StructureNode:
    n = StructureNode(
        structure=Structure(
            geometry=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            symbols=["H", "H"],
            charge=0,
            multiplicity=1,
        )
    )
    n.has_molecular_graph = False
    n.graph = None
    return n


def test_run_geom_opt_engine_without_geometry_optimizer_attribute():
    captured = {}

    class _Engine:
        def compute_geometry_optimization(self, node, keywords=None):
            captured["keywords"] = keywords
            return [node]

    traj = _run_geom_opt(_node(), _Engine())
    assert len(traj) == 1
    assert captured["keywords"] == {}


def test_run_geom_opt_geometric_engine_uses_geometric_defaults():
    captured = {}

    class _Engine(SimpleNamespace):
        geometry_optimizer = "geometric"

        def compute_geometry_optimization(self, node, keywords=None):
            captured["keywords"] = keywords
            return [node]

    traj = _run_geom_opt(_node(), _Engine())
    assert len(traj) == 1
    assert captured["keywords"] == {"coordsys": "cart", "maxiter": 1000}


def test_run_geom_opt_rejects_empty_failed_trajectory():
    class _Engine:
        def compute_geometry_optimization(self, node, keywords=None):
            return []

    with pytest.raises(ElectronicStructureError, match="did not produce a converged trajectory"):
        _run_geom_opt(_node(), _Engine())
