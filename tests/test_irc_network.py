from pathlib import Path

import numpy as np

from mepd.irc_network import build_irc_network
from mepd.pot import Pot


def _write_xyz(path: Path, frames: list[list[tuple[str, float]]]) -> None:
    lines = []
    for index, atoms in enumerate(frames):
        lines.extend(
            [
                str(len(atoms)),
                f"Frame {index}",
                *(f"{symbol} {x:.6f} 0.0 0.0" for symbol, x in atoms),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_irc_network_scans_pairs_and_round_trips(tmp_path):
    xyz_ab = tmp_path / "edge_90_91_chain_0.irc.xyz"
    xyz_bc = tmp_path / "edge_5_7_chain_0.irc.xyz"
    species_a = [("C", 0.0), ("C", 1.4), ("C", 2.8)]
    species_b = [("C", 0.0), ("C", 1.4), ("C", 5.0)]
    species_c = [("C", 0.0), ("C", 4.0), ("C", 8.0)]
    _write_xyz(xyz_ab, [species_a, species_b, species_b])
    _write_xyz(xyz_bc, [species_b, species_b, species_c])
    np.savetxt(xyz_ab.with_suffix(".energies"), [-10.0, -9.9, -10.1])
    np.savetxt(xyz_bc.with_suffix(".energies"), [-10.1, -9.8, -10.2])
    _write_xyz(tmp_path / "edge_2_3_chain_0.ts.xyz", [species_c])

    scan = build_irc_network(tmp_path)

    assert set(scan.pot.graph.nodes) == {0, 1, 2}
    assert len(scan.pot.graph.edges) == 4
    edge_ab = next(
        edge
        for edge in scan.pot.graph.edges
        if scan.pot.graph.edges[edge]["filename_edges"] == [(90, 91)]
    )
    edge_bc = next(
        edge
        for edge in scan.pot.graph.edges
        if scan.pot.graph.edges[edge]["filename_edges"] == [(5, 7)]
    )
    assert edge_ab[1] == edge_bc[0]
    assert len(scan.xyz_files) == 2
    assert len(scan.skipped_xyz_files) == 1
    assert np.isclose(
        scan.pot.graph.edges[edge_ab]["barrier"],
        0.1 * 627.509474,
    )
    assert sum(bool(data["root"]) for _, data in scan.pot.graph.nodes(data=True)) == 1
    assert sum(
        bool(data["requested_target"])
        for _, data in scan.pot.graph.nodes(data=True)
    ) == 1

    output = tmp_path / "network.json"
    scan.pot.write_to_disk(output)
    loaded = Pot.read_from_disk(output)
    assert len(loaded.graph.edges[edge_ab]["list_of_nebs"][0].nodes) == 3
    assert np.allclose(
        loaded.graph.edges[edge_ab]["list_of_nebs"][0].energies,
        [-10.0, -9.9, -10.1],
    )


def test_build_irc_network_strict_rejects_bad_frame_energy_count(tmp_path):
    xyz = tmp_path / "edge_0_1.irc.xyz"
    _write_xyz(xyz, [[("H", 0.0)], [("H", 1.0)]])
    np.savetxt(xyz.with_suffix(".energies"), [-1.0])

    try:
        build_irc_network(tmp_path, strict=True)
    except ValueError as exc:
        assert "2 frames" in str(exc)
    else:
        raise AssertionError("Expected mismatched frame/energy counts to fail")
