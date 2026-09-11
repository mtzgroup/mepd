from __future__ import annotations

import numpy as np
import pytest
import typer
from qcdata import Structure

from mepd.cli import visualize as cli_visualize
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


def _write_chain_xyz(fp, energies):
    nodes = []
    for i, energy in enumerate(energies):
        node = StructureNode(structure=_water(x_offset=0.05 * i))
        node._cached_energy = energy
        node._cached_gradient = np.zeros((3, 3))
        nodes.append(node)
    chain = Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})
    chain.write_to_disk(fp)


def _call_visualize(**overrides):
    """Same rationale as the other `_call_*` helpers in this test suite --
    calling a Typer-decorated function directly does not resolve
    `typer.Option(...)` defaults, so every parameter must be explicit."""
    kwargs = dict(
        result_path=None,
        output=None,
        charge=0,
        multiplicity=1,
        no_open=True,
        show_atom_indices=False,
    )
    kwargs.update(overrides)
    return cli_visualize(**kwargs)


def test_visualize_writes_html_with_viewer_and_energy_plot(tmp_path):
    xyz_fp = tmp_path / "chain.xyz"
    _write_chain_xyz(xyz_fp, [-76.0, -75.9, -76.1])

    _call_visualize(result_path=xyz_fp)

    out_fp = tmp_path / "chain_visualize.html"
    assert out_fp.exists()
    html = out_fp.read_text()
    assert "3Dmol-min.js" in html
    assert 'data:image/png' in html


def test_visualize_labels_every_frame_and_flags_the_ts_guess(tmp_path):
    """The whole point of per-frame labeling: a specific frame (e.g. the
    apparent TS) must be identifiable directly in the viewer, not just by
    cross-referencing node index against the plain xyz file."""
    xyz_fp = tmp_path / "chain.xyz"
    _write_chain_xyz(xyz_fp, [-76.0, -75.5, -76.1])  # node 1 is the energy maximum

    _call_visualize(result_path=xyz_fp)

    html = (tmp_path / "chain_visualize.html").read_text()
    assert "Frame 0" in html
    assert "Frame 1" in html
    assert "Frame 2" in html
    assert "TS guess" in html
    # relative energies (kcal/mol vs. frame 0) should appear somewhere
    assert "kcal/mol" in html


def test_visualize_show_atom_indices_flag_is_forwarded(tmp_path, monkeypatch):
    xyz_fp = tmp_path / "chain.xyz"
    _write_chain_xyz(xyz_fp, [-76.0, -75.9])

    seen = {}
    import mepd.viz as viz_module
    real_render = viz_module.render_chain_html

    def spy(chain, title="mepd chain visualization", show_atom_indices=False):
        seen["show_atom_indices"] = show_atom_indices
        return real_render(chain, title=title, show_atom_indices=show_atom_indices)

    monkeypatch.setattr(viz_module, "render_chain_html", spy)

    _call_visualize(result_path=xyz_fp, show_atom_indices=True)

    assert seen["show_atom_indices"] is True


def test_visualize_handles_single_node_chain(tmp_path):
    """Regression coverage for the Chain.from_xyz single-node fix -- a
    one-minimum unique.xyz from hessian-sample must still visualize."""
    xyz_fp = tmp_path / "single.xyz"
    _write_chain_xyz(xyz_fp, [-76.0])

    _call_visualize(result_path=xyz_fp)

    out_fp = tmp_path / "single_visualize.html"
    assert out_fp.exists()
    html = out_fp.read_text()
    assert "3Dmol-min.js" in html
    # A single-frame chain has no energy profile to plot.
    assert 'data:image/png' not in html


def test_visualize_respects_custom_output_path(tmp_path):
    xyz_fp = tmp_path / "chain.xyz"
    _write_chain_xyz(xyz_fp, [-76.0, -75.9])
    custom_out = tmp_path / "custom" / "report.html"

    _call_visualize(result_path=xyz_fp, output=custom_out)

    assert custom_out.exists()
