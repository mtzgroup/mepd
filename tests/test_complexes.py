"""Building complexes from molecules (mepd.complexes)."""
import numpy as np
import pytest

from mepd import complexes
from mepd.web import chem
from mepd.web.compose import _fragments

WATER = "3\nwater\nO 0 0 0\nH 0.758 0 0.504\nH -0.758 0 0.504\n"
ACETALDEHYDE = ("7\nacetaldehyde\nC -1.18 0 0\nC 0.31 0 0\nO 0.95 1.04 0\nH -1.55 1.03 0\nH -1.55 -0.51 0.89\n"
                "H -1.55 -0.51 -0.89\nH 0.80 -0.98 0\n")


def _mols():
    (w,) = chem.structures_from_xyz_text(WATER)
    (a,) = chem.structures_from_xyz_text(ACETALDEHYDE)
    return [a, w, w]


def _pieces(s):
    return sorted(chem.perceive_smiles(f) for f in _fragments(s))


def test_packing_keeps_every_molecule_whole_apart_and_in_input_order():
    (s,) = complexes.build(_mols(), "packed")
    assert len(s.symbols) == 13 and s.charge == 0 and s.multiplicity == 1
    assert _pieces(s) == ["CC=O", "O", "O"]
    xyz = np.asarray(s.geometry).reshape(-1, 3)
    assert np.allclose(xyz.mean(axis=0), 0, atol=1e-6)
    assert list(s.symbols) == [x for m in _mols() for x in m.symbols]


def test_packing_is_compact_and_seeded():
    a, = complexes.build(_mols(), "packed", seed=1)
    b, = complexes.build(_mols(), "packed", seed=1)
    c, = complexes.build(_mols(), "packed", seed=2)
    assert np.allclose(a.geometry, b.geometry) and not np.allclose(a.geometry, c.geometry)
    xyz = np.asarray(a.geometry).reshape(-1, 3) / 1.8897259886
    extent = max(np.linalg.norm(m - m.mean(axis=0), axis=1).max()
                 for m in (np.asarray(x.geometry).reshape(-1, 3) / 1.8897259886 for x in _mols()))
    assert np.linalg.norm(xyz, axis=1).max() < 3 * (extent + 2.0)   # inside a sphere near the packing radius


def test_a_smiles_of_several_molecules_is_packed_not_lined_up():
    from mepd.cli_common import _load_structure_from_smiles_or_xyz

    s = _load_structure_from_smiles_or_xyz("CC=O.O.O", None, None)
    assert _pieces(s) == ["CC=O", "O", "O"]
    assert list(s.symbols)[:3] == ["C", "C", "O"] and list(s.symbols).count("O") == 3
    xyz = np.asarray(s.geometry).reshape(-1, 3)
    spans = np.sort(np.ptp(xyz, axis=0))
    assert spans[-1] < 2.5 * spans[0] + 4.0                        # a cluster, not a row along one axis


def test_bad_requests_are_explained():
    with pytest.raises(ValueError, match="at least two"):
        complexes.build(_mols()[:1], "packed")
    with pytest.raises(ValueError, match="one solute"):
        complexes.build(_mols()[:2], "qcg", workdir="/nonexistent")     # one of each: no solvent copies
    with pytest.raises(ValueError, match="unknown method"):
        complexes.build(_mols(), "nope")


@pytest.mark.skipif(complexes.xtb_dock_executable(download=False) is None, reason="needs xtb")
def test_docking_adds_the_molecules_one_at_a_time(tmp_path):
    poses = complexes.build(_mols(), "dock", tmp_path, keep=2)
    assert 1 <= len(poses) <= 2
    assert all(_pieces(p) == ["CC=O", "O", "O"] for p in poses)
    assert (tmp_path / "step_1").is_dir() and (tmp_path / "step_2").is_dir()


def test_the_cli_keeps_each_complex_s_charge_and_spin(tmp_path):
    from typer.testing import CliRunner

    from mepd.cli import app

    (tmp_path / "oh.xyz").write_text("2\nqcdata_charge=-1 qcdata_multiplicity=1\nO 0 0 0\nH 0 0 0.97\n")
    (tmp_path / "ch3.xyz").write_text("4\nqcdata_charge=0 qcdata_multiplicity=2\nC 0 0 0\nH 1.08 0 0\nH -0.54 0.94 0\n"
                                      "H -0.54 -0.94 0\n")
    res = CliRunner().invoke(app, ["complex", str(tmp_path / "oh.xyz"), str(tmp_path / "ch3.xyz"), "--method", "packed",
                                   "-o", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output
    (s,) = chem.structures_from_xyz_text((tmp_path / "out" / "complexes.xyz").read_text())
    assert (int(s.charge), int(s.multiplicity)) == (-1, 2)


def test_the_live_view_reads_what_the_programs_have_written_so_far(tmp_path):
    from mepd.web.complex_live import complex_live

    xyz = lambda rows, c="": f"{len(rows)}\n{c}\n" + "".join(f"{s} {x} {y} {z}\n" for s, x, y, z in rows)
    water = [("O", 0, 0, 0), ("H", 0.76, 0, 0.5), ("H", -0.76, 0, 0.5)]
    two = water + [(s, x + 2.8, y, z) for s, x, y, z in water]
    step = tmp_path / "work" / "step_1"
    step.mkdir(parents=True)
    assert complex_live(tmp_path, "dock", 2)["stage"] == "Building the complex…"
    (step / "host.xyz").write_text(xyz(water))
    (step / "guest.xyz").write_text(xyz(water))
    v = complex_live(tmp_path, "dock", 2)
    assert [f["caption"] for f in v["frames"]] == ["Molecule 2 of 2: finding where it binds…"]
    assert len(v["frames"][0]["symbols"]) == 6
    (step / "optimized_structures.xyz").write_text(xyz(two, " -10.0") + xyz(two, " -10.5") + "6\n half")
    v = complex_live(tmp_path, "dock", 2)
    assert v["frames"][-1]["caption"] == "Molecule 2 of 2: settled" and v["frames"][-1]["tween"]
    # QCG: the solute, then the shell as it grows (complete frames only).
    q = tmp_path / "q" / "work"
    (q / "grow").mkdir(parents=True)
    (q / "solute.xyz").write_text(xyz(water))
    (q / "solvent.xyz").write_text(xyz(water))
    (q / "grow" / "qcg_grow.xyz").write_text(xyz(two) + "9\n")
    caps = [f["caption"] for f in complex_live(tmp_path / "q", "qcg", 3)["frames"]]
    assert caps == ["The solute", "Solvent molecule 1 of 2 added"]
