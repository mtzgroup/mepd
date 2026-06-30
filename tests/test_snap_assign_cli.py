import numpy as np
from qcio import Structure
from typer.testing import CliRunner

from mepd.nodes.node import StructureNode
from mepd.scripts import main_cli


def _structure(symbols: list[str], coords: list[list[float]]) -> Structure:
    return Structure(
        geometry=np.array(coords, dtype=float),
        symbols=symbols,
        charge=0,
        multiplicity=1,
    )


def test_snap_assign_endpoint_nodes_reorders_identical_graph_endpoint():
    start = StructureNode(
        structure=_structure(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
    )
    shuffled_end = StructureNode(
        structure=_structure(["O", "C"], [[0.0, 0.0, 1.1], [0.0, 0.0, 0.0]])
    )

    result = main_cli._snap_assign_endpoint_nodes([start, shuffled_end])

    assert result[0] is start
    assert result[1].structure.symbols == ["C", "O"]
    np.testing.assert_allclose(result[1].coords, start.coords)


def test_snap_assign_endpoint_nodes_skips_different_graphs():
    start = StructureNode(
        structure=_structure(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
    )
    different_end = StructureNode(
        structure=_structure(["C", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    )

    result = main_cli._snap_assign_endpoint_nodes([start, different_end])

    assert result == [start, different_end]
    np.testing.assert_allclose(result[1].coords, different_end.coords)


def test_run_help_exposes_snap_assign_flag():
    runner = CliRunner()

    result = runner.invoke(main_cli.app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--snap-assign" in result.output
