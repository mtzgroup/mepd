from __future__ import annotations

import json

import numpy as np
import pytest
import typer
from qcdata import Structure

import mepd.discovery.cli as discovery_cli
from mepd.chain import Chain
from mepd.discovery import vri
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode


def _node(dx: float, energy: float) -> StructureNode:
    node = StructureNode(structure=Structure(
        symbols=["O", "H", "H"],
        geometry=np.array([[0.0, 0.0, 0.0], [1.43 + dx, 0.0, 0.95], [-1.43, 0.0, 0.95]]),
        charge=0, multiplicity=1,
    ))
    node._cached_energy = energy
    node._cached_gradient = np.zeros((3, 3))
    return node


def _modes(freqs):
    freqs = np.asarray(freqs, dtype=float)
    return vri.ProjectedModes(
        eigvals=np.sign(freqs) * freqs**2, freqs=freqs, modes_mw=np.eye(9)[:, : len(freqs)],
        modes_cart=[np.eye(9)[:, i].reshape(3, 3) for i in range(len(freqs))],
        grad_norm=0.0, tangent_source="none",
    )


def _call_vri(**overrides):
    """Typer options are not resolved when the command is called as a plain
    function, so every parameter is given explicitly."""
    kwargs = dict(
        ts="ts.xyz", irc=None, inputs=None, charge=None, multiplicity=None,
        irc_step=0.05, irc_fmax=0.01, stride=1, n_bisect=4, vrt_threshold=20.0, persist=2,
        imaginary_cutoff=50.0, grad_floor=1e-4, push_amplitude=0.3, n_push_points=3,
        branches="both", skip_ts2=True, output=None,
    )
    kwargs.update(overrides)
    return discovery_cli.vri_search(**kwargs)


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    ts_fp = tmp_path / "ts.xyz"
    ts_fp.write_text(_node(0.0, 0.0).structure.to_xyz())
    irc_nodes = [_node(-0.2 + 0.05 * i, -0.01 * abs(i - 4)) for i in range(9)]
    irc_nodes[4]._cached_energy = 0.0
    state = {"ts_freqs": [-400.0, 150.0, 300.0]}

    monkeypatch.setattr(vri, "stationary_point_modes", lambda node, engine: _modes(state["ts_freqs"]))
    monkeypatch.setattr(
        discovery_cli, "_run_vri_irc",
        lambda engine, ts_node, **kw: Chain.model_validate({"nodes": irc_nodes, "parameters": ChainInputs()}),
    )

    def fake_scan(nodes, engine, **kw):
        branches = {}
        for name, branch_nodes in vri.split_irc_branches(nodes, 4).items():
            branch = vri.BranchScan(name=name, nodes=branch_nodes, s=np.linspace(0, 1, len(branch_nodes)))
            if name == "forward":
                branch.vrt = vri.VRT(
                    s=0.4, node=branch_nodes[2], ridge_mode_cart=np.eye(9)[:, 3].reshape(3, 3),
                    freq_before=30.0, freq_after=-60.0, bracket_chain_indices=(1, 2),
                )
            branches[name] = branch
        return vri.VRTScan(ts_index=4, ts_modes=_modes([-400.0]), branches=branches, n_hessians=10, freq_units="cm^-1")

    def fake_products(scan, name, engine, **kw):
        return vri.BranchProducts(
            branch=name, endpoint_n_imaginary=1, p1=irc_nodes[-1], p2=irc_nodes[0], ts2=irc_nodes[-2],
            ts2_source="irc_endpoint", ts2_verified=True,
            ts2_checks={"n_imaginary": 1, "below_ts1": True, "connects_p1_p2": True},
        )

    monkeypatch.setattr(vri, "scan_irc_for_vrt", fake_scan)
    monkeypatch.setattr(vri, "find_bifurcation_products", fake_products)
    return ts_fp, tmp_path, state


def test_vri_command_writes_outputs_and_summary(fake_pipeline):
    ts_fp, tmp_path, _ = fake_pipeline
    out = tmp_path / "out"
    _call_vri(ts=str(ts_fp), output=out)

    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "bifurcation"
    assert summary["ts1_n_imaginary"] == 1
    assert summary["branches"]["forward"]["verdict"] == "bifurcation"
    assert summary["branches"]["reverse"]["verdict"] == "no_vrt"
    assert summary["branches"]["forward"]["products"]["ts2_source"] == "irc_endpoint"
    for name in ("irc.xyz", "projected_freqs.json", "vrt_forward.xyz", "ridge_mode_forward.xyz",
                 "p1_forward.xyz", "p2_forward.xyz", "ts2_forward.xyz"):
        assert (out / name).exists(), name
    frames = Chain.from_xyz(out / "ridge_mode_forward.xyz", ChainInputs())
    assert len(frames.nodes) == 21
    scan = json.loads((out / "projected_freqs.json").read_text())
    assert scan["branches"]["forward"]["vrt"]["s"] == 0.4


def test_vri_command_refuses_a_structure_without_imaginary_mode(fake_pipeline):
    ts_fp, tmp_path, state = fake_pipeline
    state["ts_freqs"] = [120.0, 300.0]
    with pytest.raises(typer.Exit):
        _call_vri(ts=str(ts_fp), output=tmp_path / "out")


def test_vri_command_rejects_bad_branch_option(fake_pipeline):
    ts_fp, tmp_path, _ = fake_pipeline
    with pytest.raises(typer.BadParameter):
        _call_vri(ts=str(ts_fp), output=tmp_path / "out", branches="sideways")
