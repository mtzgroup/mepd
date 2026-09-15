from __future__ import annotations

import numpy as np
from qcdata.models.structure import Structure

from mepd.engines.engine import Engine
from mepd.nodes.node import StructureNode


class _QuadraticEngine(Engine):
    """Toy engine: E(x) = sum(x^2), grad = 2x. Simple enough for
    steepest_descent to make real, checkable progress in a few steps."""

    def compute_gradients(self, chain):
        return [2.0 * np.asarray(node.coords) for node in chain]

    def compute_energies(self, chain):
        return [float(np.sum(np.asarray(node.coords) ** 2)) for node in chain]


def _node(distance: float) -> StructureNode:
    structure = Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )
    return StructureNode(structure=structure)


def test_steepest_descent_computes_missing_gradient_and_energy_up_front():
    """Regression test: steepest_descent used to read node.gradient/.energy
    on its very first iteration before ever computing anything for the
    passed-in node, assuming the caller had already cached both -- which
    crashed with GradientsNotComputedError whenever the caller (e.g.
    elementarystep._converges_to_an_endpoints, checking a freshly-built split
    candidate) passed in a node with nothing cached yet."""
    engine = _QuadraticEngine()
    node = _node(2.0)
    assert node._cached_gradient is None
    assert node._cached_energy is None

    history = engine.steepest_descent(node=node, ss=0.1, max_steps=5)

    assert len(history) > 0
    # Energy should have decreased -- real steepest-descent progress, not a
    # trivial no-op trajectory.
    assert history[-1].energy < float(np.sum(node.coords ** 2))


def test_steepest_descent_reuses_already_cached_gradient_and_energy():
    """When the caller already computed gradient/energy (the common case for
    a chain node coming out of NEB/GSM), steepest_descent must not silently
    recompute and overwrite them with something inconsistent before starting."""
    engine = _QuadraticEngine()
    node = _node(2.0)
    node._cached_gradient = np.array([999.0, 0.0, 0.0] * 2).reshape(2, 3)
    node._cached_energy = -12345.0

    history = engine.steepest_descent(node=node, ss=0.1, max_steps=5)

    assert len(history) > 0
