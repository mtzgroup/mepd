import json
from types import SimpleNamespace

import numpy as np
from qcio import Structure

from mepd.chain import Chain
from mepd.elementarystep import ElemStepResults
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode
from mepd.scripts import main_cli


def _node_at_x(x: float) -> StructureNode:
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
    return node


def test_check_elem_step_cli_writes_result_and_new_structures(monkeypatch, tmp_path):
    geom_fp = tmp_path / "path.xyz"
    geom_fp.write_text("placeholder\n", encoding="utf-8")
    output_fp = tmp_path / "elem.json"
    new_structures_fp = tmp_path / "new.xyz"
    chain_inputs = ChainInputs()
    chain = Chain.model_validate(
        {"nodes": [_node_at_x(0.0), _node_at_x(1.0)], "parameters": chain_inputs}
    )
    new_node = _node_at_x(5.0)
    calls = {}

    class FakeEngine:
        def compute_energies(self, input_chain):
            calls["compute_energies_chain"] = input_chain
            for index, node in enumerate(input_chain.nodes):
                node._cached_energy = float(index)
                node._cached_gradient = np.zeros_like(node.coords)

    monkeypatch.setattr(main_cli, "BANNER", "")
    monkeypatch.setattr(
        main_cli.RunInputs,
        "open",
        staticmethod(
            lambda _fp: SimpleNamespace(
                chain_inputs=chain_inputs,
                path_min_inputs=SimpleNamespace(
                    hessian_minima_rescue_displacement=0.1
                ),
                engine=FakeEngine(),
            )
        ),
    )
    monkeypatch.setattr(
        main_cli.Chain,
        "from_xyz",
        classmethod(lambda cls, **kwargs: chain),
    )

    def fake_check_if_elem_step(**kwargs):
        calls["check_kwargs"] = kwargs
        return ElemStepResults(
            is_elem_step=False,
            is_concave=False,
            splitting_criterion="minima",
            minimization_results=[new_node],
            number_grad_calls=7,
            new_structures=[new_node],
        )

    monkeypatch.setattr(main_cli, "check_if_elem_step", fake_check_if_elem_step)

    main_cli.check_elem_step_cli(
        geometries=str(geom_fp),
        inputs="inputs.toml",
        output=str(output_fp),
        new_structures_output=str(new_structures_fp),
        validate_minima_with_hessian=True,
        hessian_minimum_frequency_cutoff=12.5,
        hessian_minima_rescue_displacement=0.03,
        verbose=False,
    )

    payload = json.loads(output_fp.read_text(encoding="utf-8"))

    assert calls["compute_energies_chain"] is chain
    assert calls["check_kwargs"]["inp_chain"] is chain
    assert calls["check_kwargs"]["validate_minima_with_hessian"] is True
    assert calls["check_kwargs"]["hessian_minimum_frequency_cutoff"] == 12.5
    assert calls["check_kwargs"]["hessian_minima_rescue_displacement"] == 0.03
    assert payload["is_elem_step"] is False
    assert payload["splitting_criterion"] == "minima"
    assert payload["number_grad_calls"] == 7
    assert payload["new_structure_count"] == 1
    assert payload["chain_energies_hartree"] == [0.0, 1.0]
    assert payload["new_structures_xyz"] == str(new_structures_fp)
    assert new_structures_fp.exists()
