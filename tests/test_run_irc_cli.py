from types import SimpleNamespace

import numpy as np
from qcio import Structure

from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.scripts import main_cli


def _node_at_x(x: float, energy: float) -> StructureNode:
    node = StructureNode(
        structure=Structure(
            symbols=["H"],
            geometry=np.array([[x, 0.0, 0.0]], dtype=float),
            charge=0,
            multiplicity=1,
        )
    )
    node.has_molecular_graph = False
    node.graph = None
    node._cached_energy = float(energy)
    node._cached_gradient = [[0.0, 0.0, 0.0]]
    return node


def test_run_irc_compat_dispatch_forces_geometric_program(monkeypatch, tmp_path):
    input_fp = tmp_path / "mep_output.xyz"
    input_fp.write_text("placeholder\n")
    output_fp = tmp_path / "irc.xyz"
    chain_inputs = ChainInputs()
    source_chain = Chain.model_validate(
        {"nodes": [_node_at_x(0.0, 0.0), _node_at_x(1.0, 2.0)], "parameters": chain_inputs}
    )
    irc_chain = Chain.model_validate(
        {"nodes": [_node_at_x(-1.0, 0.5), _node_at_x(1.0, 0.0)], "parameters": chain_inputs}
    )
    calls = {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        main_cli.RunInputs,
        "open",
        staticmethod(
                lambda fp: SimpleNamespace(
                    chain_inputs=chain_inputs,
                    engine=SimpleNamespace(),
                    write_qcio=False,
                    path_min_method="NEB",
                    path_min_inputs=SimpleNamespace(),
                    gi_inputs=SimpleNamespace(),
                    engine_name="qcop",
                    program="crest",
                    program_kwds={},
                    optimizer_kwds={},
            )
        ),
    )
    monkeypatch.setattr(main_cli.Chain, "from_xyz", classmethod(lambda cls, **kwargs: source_chain))
    monkeypatch.setattr(main_cli, "_ascii_profile_for_chain", lambda chain: None)

    def _fake_compute_irc_chain_for_program(**kwargs):
        calls.update(kwargs)
        return irc_chain

    monkeypatch.setattr(
        main_cli,
        "_compute_irc_chain_for_program",
        _fake_compute_irc_chain_for_program,
    )

    main_cli.run(
        geometries="irc",
        inputs="crest-neb.toml",
        program="geometric",
        irc_output=str(output_fp),
    )

    assert calls["program"] == "geometric"
    assert calls["ts_node"] is source_chain.nodes[1]
    assert output_fp.exists()
