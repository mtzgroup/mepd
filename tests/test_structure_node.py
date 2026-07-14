import numpy as np
from qcio import Structure

from mepd.nodes.node import StructureNode


def test_update_coords_rebuilds_molecular_graph():
    node = StructureNode(
        structure=Structure(
            symbols=["H", "H"],
            geometry=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
            charge=0,
            multiplicity=1,
        )
    )

    updated = node.update_coords(np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=float))

    assert node.graph.number_of_edges() == 1
    assert updated.graph.number_of_edges() == 0
