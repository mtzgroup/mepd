from __future__ import annotations

import numpy as np
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode


def _water(x_offset: float = 0.0) -> Structure:
    return Structure(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.43355001758932 + x_offset, 0.0, 0.95295864902809],
                [-1.43355001758932, 0.0, 0.95295864902809],
            ],
            dtype=float,
        ),
        charge=0,
        multiplicity=1,
    )


def _node_with_energy(x_offset: float, energy: float) -> StructureNode:
    node = StructureNode(structure=_water(x_offset))
    node._cached_energy = energy
    node._cached_gradient = np.zeros((3, 3))
    return node


def test_chain_from_xyz_round_trips_multi_node_chain(tmp_path):
    chain = Chain.model_validate({
        "nodes": [_node_with_energy(0.0, -76.0), _node_with_energy(0.1, -75.9)],
        "parameters": ChainInputs(),
    })
    fp = tmp_path / "chain.xyz"
    chain.write_to_disk(fp)

    reloaded = Chain.from_xyz(fp, ChainInputs())

    assert len(reloaded) == 2
    assert np.allclose(reloaded.energies, [-76.0, -75.9])


def test_chain_from_xyz_round_trips_single_node_chain(tmp_path):
    """Regression test: np.loadtxt collapses a single-value .energies/.gradients
    file to a 0-d array, which used to raise `TypeError: iteration over a 0-d
    array` when reloading a one-node chain (e.g. mepd discovery hessian-sample's
    unique.xyz when only one minimum was found)."""
    chain = Chain.model_validate({
        "nodes": [_node_with_energy(0.0, -76.0)],
        "parameters": ChainInputs(),
    })
    fp = tmp_path / "chain.xyz"
    chain.write_to_disk(fp)

    reloaded = Chain.from_xyz(fp, ChainInputs())

    assert len(reloaded) == 1
    assert np.allclose(reloaded.energies, [-76.0])
