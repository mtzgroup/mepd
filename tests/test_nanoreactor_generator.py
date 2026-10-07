"""The nanoreactor as a network-expansion product generator: the structure
right after each reaction event of its hot piston MD is a product (whole, in
the input's atom order); on a QM/MM system it runs on the capped QM region."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
from qcconst.constants import ANGSTROM_TO_BOHR
from qcdata import Structure

# Two waters: H1 hops from O0 to O3 (OH- + H3O+), then everything flies apart.
SYMBOLS = ["O", "H", "H", "O", "H", "H"]


def _frames():
    a = np.array([[0.0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [2.7, 0, 0], [3.0, 0.9, 0], [3.0, -0.9, 0]])
    hop = a.copy()
    hop[1] = [1.75, 0, 0]
    apart = np.arange(18.0).reshape(6, 3) * 3.0
    return np.array([a] * 40 + [hop] * 100 + [apart] * 40)   # events further apart than the merge window


def _fake_md(frames):
    def run(symbols, xyz, *, workdir, **kw):
        fp = workdir / "trajectory.xyz"
        fp.write_text("".join(f"{len(symbols)}\n\n" + "".join(f"{s} {x:.5f} {y:.5f} {z:.5f}\n"
                                                              for s, (x, y, z) in zip(symbols, f)) for f in frames))
        return fp
    return run


def test_products_are_states_after_events_without_free_atoms(monkeypatch, tmp_path):
    import mepd.discovery.nanoreactor as nr
    from mepd.discovery import nanoreactor_generator as gen

    monkeypatch.setattr(nr, "run_reactor_md", _fake_md(_frames()))
    monkeypatch.setattr(gen, "_md_method", lambda name: ("gfn2", None))
    s = Structure(symbols=SYMBOLS, geometry=_frames()[0] * ANGSTROM_TO_BOHR)
    products = gen.reactor_products(s, max_products=5, workdir=str(tmp_path))
    assert len(products) == 1                       # the hop; the atomized end is dropped
    x = np.asarray(products[0].geometry) / ANGSTROM_TO_BOHR
    assert abs(np.linalg.norm(x[1] - x[3]) - 0.95) < 1e-6
    assert list(products[0].symbols) == SYMBOLS and products[0].charge == s.charge


def test_registered_and_options_checked():
    from mepd.discovery.generators import get_generator

    g = get_generator("nanoreactor")
    assert g.kind == "structures" and "temperature" in g.options
    with pytest.raises(ValueError, match="nanoreactor options"):
        g.propose(Structure(symbols=["H", "H"], geometry=np.zeros((2, 3))), max_products=1, options={"bogus": 1})


def test_web_offers_it_only_for_qmmm_and_builds_real_flags():
    from mepd.web.operations import ExpandParams, OPERATIONS, _cli_options

    op = OPERATIONS["graph-enumeration"]
    meth = [m for m in op.describe()["methods"] if m["fixed"].get("generator") == "nanoreactor"]
    assert meth and meth[0]["qmmm_only"]
    p = ExpandParams(generator="nanoreactor", reactor_temperature=3200, reactor_time_ps=5)

    class Ctx:   # just enough of a JobContext for the argv
        structures = [{}]

        def snapshot_structure(self, rec, name):
            return "seed.xyz"

        def common_flags(self):
            return []

        output_dir = "out"

    argv = op.build(Ctx(), p)
    assert "temperature=3200" in argv and "time_ps=5" in argv and "embedded=true" not in argv
    assert "embedded=true" in op.build(Ctx(), ExpandParams(generator="nanoreactor", reactor_embedded=True))
    known = _cli_options(("discovery", "expand"))
    assert not [w for w in argv if w.startswith("--") and w not in known]


@pytest.mark.skipif(shutil.which("xtb") is None, reason="needs the xtb program")
def test_real_hot_md_turns_vinyl_alcohol_into_something(tmp_path):
    from mepd.cli import _load_structure_from_smiles_or_xyz
    from mepd.discovery.nanoreactor_generator import reactor_products

    s = _load_structure_from_smiles_or_xyz("C=CO", 0, 1)
    out = reactor_products(s, max_products=3, temperature=3000, time_ps=2.0, md_method="gfn2",
                           workdir=str(tmp_path))
    for p in out:
        assert list(p.symbols) == list(s.symbols)


def test_qmmm_reactor_md_holds_frozen_atoms_and_keeps_two_temperatures(tmp_path, monkeypatch):
    from mepd.programs import xtb_executable

    exe = xtb_executable(download=False)
    if not exe:
        pytest.skip("needs the xtb program")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    from test_qmmm import QM_PROPANOL, _propanol

    from mepd.discovery.nanoreactor import ReactorSettings, read_xyz_frames
    from mepd.discovery.qmmm_reactor import qmmm_reactor_products, run_qmmm_reactor_md
    from mepd.engines.frozen import FrozenAtomsEngine
    from mepd.engines.gfnff import GFNFFEngine
    from mepd.engines.qmmm import QMMMEngine
    from mepd.qmmm import QMMMRegion

    s = _propanol()
    region = QMMMRegion.build(s, QM_PROPANOL, active_radius=None, frozen_atoms=[4, 5, 6])
    engine = FrozenAtomsEngine(base=QMMMEngine(base=GFNFFEngine(method="gfn2", executable=exe, n_parallel=1),
                                               region=region, n_parallel=1), frozen=region.frozen_atoms)
    seen = []
    settings = ReactorSettings(temperature=1500, time_ps=0.04, period_ps=0.02, method="level", radius=4.0)
    traj = run_qmmm_reactor_md(s, engine, region, settings=settings, workdir=tmp_path / "md",
                               on_event=lambda ev, p: seen.append(p) if ev == "md_segment" else None)
    _, frames, _ = read_xyz_frames(traj)
    x0 = np.asarray(s.geometry) / ANGSTROM_TO_BOHR
    assert len(frames) >= 10
    assert np.abs(frames[:, [4, 5, 6]] - x0[[4, 5, 6]]).max() < 1e-6       # frozen stay put
    assert np.abs(frames[:, QM_PROPANOL] - x0[QM_PROPANOL]).max() > 0.01    # the QM region moves
    assert seen and all("environment_temperature" in p for p in seen)
    products, counts = qmmm_reactor_products(s, engine, region, temperature=1500, time_ps=0.04, period_ps=0.02,
                                             workdir=str(tmp_path / "md2"))
    assert set(counts) == {"events", "products", "mm_changed", "boundary", "torn"}
    assert all(len(p.symbols) == len(s.symbols) for p in products)
