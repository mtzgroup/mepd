"""GSM's live string carries each image's gradient (not only its energy), so
the channels path map can fit slopes for GSM paths too."""

import os
from types import SimpleNamespace

import numpy as np
from qcdata import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.pathminimizers.gsm import GSM, _ENGINE_SERVER_SOURCE
from mepd.progress import _chain_geometry_payload

WATER = "3\n\nO 0.000 0.000 0.000\nH 0.758 0.000 0.504\nH -0.758 0.000 0.504\n"


class _Engine:
    def compute_energies(self, nodes):
        return [-76.0 - 0.001 * float(np.asarray(n.coords)[1, 0]) for n in nodes]

    def compute_gradients(self, nodes):
        return [np.full((3, 3), 0.01 * (k + 1)) for k, _ in enumerate(nodes)]


def test_gsm_live_string_keeps_each_images_gradient(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scratch").mkdir()
    server = {"__name__": "gsm_engine_server"}        # not __main__: no socket server
    exec(compile(_ENGINE_SERVER_SOURCE, "engine_server", "exec"), server)

    reactant = StructureNode(structure=Structure.from_xyz(WATER))
    reactant._cached_energy = -76.0
    for idx, dx in ((1, 0.0), (2, 0.1), (3, 0.2)):
        xyz = WATER.replace("H 0.758", f"H {0.758 + dx:.3f}", 1)
        (tmp_path / "scratch" / f"structure_x.{idx}").write_text(xyz)
        server["handle"](f"_x.{idx}", _Engine(), reactant, -76.0)

    live = tmp_path / server["LIVE_PATH"]
    gsm = SimpleNamespace(_parse_string_blocks=GSM._parse_string_blocks)
    chain = GSM._build_live_chain(gsm, live, reactant, reactant, -76.0, ChainInputs())
    assert len(chain) == 3
    # The pinned endpoints are the real nodes; the interior image has the
    # gradient the engine returned for it (Hartree/bohr, per atom).
    np.testing.assert_allclose(chain[1]._cached_gradient, np.full((3, 3), 0.01))

    payload = _chain_geometry_payload(chain, [n.energy for n in chain])
    assert payload["gradients"][0] is None and payload["gradients"][2] is None   # endpoints: none cached
    assert payload["gradients"][1] == [0.01] * 9

    # A gradient file from another snapshot (other length) is ignored, not misapplied.
    (live.parent / (live.name + ".grad")).write_text("[[0.0], [0.0]]")
    chain = GSM._build_live_chain(gsm, live, reactant, reactant, -76.0, ChainInputs())
    assert chain[1]._cached_gradient is None


def test_live_payload_without_any_gradient_has_none():
    nodes = [StructureNode(structure=Structure.from_xyz(WATER)) for _ in range(3)]
    chain = Chain.model_validate({"nodes": nodes, "parameters": ChainInputs()})
    assert "gradients" not in _chain_geometry_payload(chain, [0.0, 1.0, 0.0])
