from types import SimpleNamespace

import numpy as np
import pytest
from qcio import Structure

from mepd.chain import Chain
from mepd.elementarystep import (
    _chain_is_concave,
    _converges_to_an_endpoints,
    _get_ts_neighbor_pair_indices,
    _validate_hessian_minimum,
    _write_nodes_xyz,
    check_if_elem_step,
    is_approx_elem_step,
    pseudo_irc,
)
from mepd.errors import EnergiesNotComputedError
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode


def _structure_at_x(x: float) -> Structure:
    return Structure(
        geometry=np.array([[0.0, 0.0, 0.0], [x, 0.0, 0.0]]),
        symbols=["H", "H"],
        charge=0,
        multiplicity=1,
    )


def _make_chain_with_energies(energies: list[float]) -> Chain:
    chain = Chain.model_validate(
        {
            "nodes": [
                StructureNode(structure=_structure_at_x(float(index)))
                for index in range(len(energies))
            ],
            "parameters": ChainInputs(),
        }
    )
    for node, energy in zip(chain, energies):
        node._cached_energy = float(energy)
    return chain


class ExternalProgramError(Exception):
    pass


class ProgramNotFoundError(Exception):
    pass


def test_ts_neighbor_pair_uses_highest_image_and_highest_energy_neighbor():
    chain = _make_chain_with_energies([0.0, 1.0, 2.0, 5.0, 4.0, 3.0, 0.0])

    assert _get_ts_neighbor_pair_indices(chain) == (3, 4)


def test_approx_elem_step_minimizes_highest_image_and_highest_energy_neighbor(monkeypatch):
    chain = _make_chain_with_energies([0.0, 1.0, 2.0, 5.0, 4.0, 3.0, 0.0])
    calls = []

    def fake_converges_to_endpoints(**kwargs):
        calls.append((kwargs["node_index"], kwargs["direction"]))
        return True, [chain[kwargs["node_index"]]]

    monkeypatch.setattr(
        "mepd.elementarystep._converges_to_an_endpoints",
        fake_converges_to_endpoints,
    )
    monkeypatch.setattr(
        "mepd.elementarystep._is_connectivity_identical",
        lambda *args, **kwargs: True,
    )

    passed, _ = is_approx_elem_step(chain=chain, engine=SimpleNamespace(), verbose=False)

    assert passed is True
    assert calls == [(3, -1), (4, 1)]


def test_approx_elem_step_chemcloud_falls_through_to_expensive_check():
    chain = _make_chain_with_energies([0.0, 1.0, 2.0, 5.0, 4.0, 3.0, 0.0])

    passed, grad_calls = is_approx_elem_step(
        chain=chain,
        engine=SimpleNamespace(compute_program="chemcloud"),
        verbose=False,
    )

    assert passed is False
    assert grad_calls == 0


def test_pseudo_irc_minimizes_highest_image_and_highest_energy_neighbor(monkeypatch):
    chain = _make_chain_with_energies([0.0, 1.0, 2.0, 5.0, 4.0, 3.0, 0.0])
    optimized_indices = []

    def fake_run_geom_opt(node, engine):
        optimized_indices.append(next(i for i, curr in enumerate(chain.nodes) if curr is node))
        return [node]

    monkeypatch.setattr("mepd.elementarystep._run_geom_opt", fake_run_geom_opt)

    result = pseudo_irc(chain=chain, engine=SimpleNamespace())

    assert optimized_indices == [3, 4]
    assert result.found_reactant is chain[3]
    assert result.found_product is chain[4]


def test_pseudo_irc_fallback_handles_missing_chain_for_opt_energies(monkeypatch):
    chain = _make_chain_with_energies([0.0, 1.0, 5.0, 4.0, 0.0])

    def fake_get_ts_neighbor_pair_indices(_chain):
        raise EnergiesNotComputedError(msg="Energies have not been computed.")

    monkeypatch.setattr(
        "mepd.elementarystep._get_ts_neighbor_pair_indices",
        fake_get_ts_neighbor_pair_indices,
    )

    result = pseudo_irc(chain=chain, engine=SimpleNamespace())

    assert result.optimization_succeeded is False
    assert result.found_reactant is chain[1]
    assert result.found_product is chain[3]


def test_check_if_elem_step_runs_pseudo_irc_for_chemcloud(monkeypatch):
    chain = Chain.model_validate(
        {
            "nodes": [
                StructureNode(structure=_structure_at_x(0.0)),
                StructureNode(structure=_structure_at_x(1.0)),
                StructureNode(structure=_structure_at_x(2.0)),
            ],
            "parameters": ChainInputs(),
        }
    )
    for node, energy in zip(chain, [0.0, 2.0, 0.0]):
        node._cached_energy = energy

    monkeypatch.setattr(
        "mepd.elementarystep._chain_is_concave",
        lambda chain, engine, verbose=True: SimpleNamespace(
            is_not_concave=False,
            is_concave=True,
            minimization_results=[],
            number_grad_calls=0,
        ),
    )
    pseudo_irc_called = False

    def fake_pseudo_irc(chain, engine):
        nonlocal pseudo_irc_called
        pseudo_irc_called = True
        return SimpleNamespace(
            found_reactant=chain[0],
            found_product=chain[-1],
            number_grad_calls=2,
            optimization_succeeded=True,
        )

    monkeypatch.setattr("mepd.elementarystep.pseudo_irc", fake_pseudo_irc)
    monkeypatch.setattr(
        "mepd.elementarystep.is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    engine = SimpleNamespace(compute_program="chemcloud")
    result = check_if_elem_step(chain, engine=engine, verbose=False)

    assert pseudo_irc_called is True
    assert result.is_elem_step is True
    assert result.number_grad_calls == 2


def test_chain_is_concave_rejects_hessian_saddle_minima_split(monkeypatch):
    chain = _make_chain_with_energies([1.0, 0.0, 1.0])

    monkeypatch.setattr(
        "mepd.elementarystep._converges_to_an_endpoints",
        lambda **kwargs: (False, [chain[kwargs["node_index"]]]),
    )
    monkeypatch.setattr(
        "mepd.elementarystep.is_identical",
        lambda a, b, **kwargs: False,
    )

    class FakeEngine:
        compute_program = "qcop"

        def compute_geometry_optimization(self, node, keywords=None):
            return [node]

        def _compute_hessian_result(self, node):
            return SimpleNamespace(
                results=SimpleNamespace(freqs_wavenumber=[-250.0, 100.0, 150.0])
            )

    result = _chain_is_concave(
        chain=chain,
        engine=FakeEngine(),
        verbose=False,
        validate_minima_with_hessian=True,
    )

    assert result.is_concave is True
    assert result.minimization_results == [chain[0], chain[-1]]
    assert result.rejected_minimization_results == [chain[1]]


def test_chain_is_concave_accepts_hessian_true_minimum_split(monkeypatch):
    chain = _make_chain_with_energies([1.0, 0.0, 1.0])

    monkeypatch.setattr(
        "mepd.elementarystep._converges_to_an_endpoints",
        lambda **kwargs: (False, [chain[kwargs["node_index"]]]),
    )
    monkeypatch.setattr(
        "mepd.elementarystep.is_identical",
        lambda a, b, **kwargs: False,
    )

    class FakeEngine:
        compute_program = "qcop"

        def compute_geometry_optimization(self, node, keywords=None):
            return [node]

        def _compute_hessian_result(self, node):
            return SimpleNamespace(
                results=SimpleNamespace(freqs_wavenumber=[15.0, 100.0, 150.0])
            )

    result = _chain_is_concave(
        chain=chain,
        engine=FakeEngine(),
        verbose=False,
        validate_minima_with_hessian=True,
    )

    assert result.is_concave is False
    assert result.minimization_results == [chain[1]]
    assert result.rejected_minimization_results == []


def test_hessian_minimum_validation_rejects_negative_matrix_curvature():
    node = _make_chain_with_energies([0.0])[0]

    class FakeEngine:
        def _compute_hessian_result(self, node):
            return SimpleNamespace(
                results=SimpleNamespace(
                    hessian=np.diag([-1.0, 2.0, 3.0]),
                    freqs_wavenumber=[-1.0, 1.414, 1.732],
                )
            )

    result = _validate_hessian_minimum(
        node,
        engine=FakeEngine(),
        frequency_cutoff=0.0,
    )

    assert result.is_minimum is False
    assert result.min_hessian_eigenvalue == pytest.approx(-1.0)


def test_check_if_elem_step_forces_maxima_split_when_pseudo_irc_optimization_fails(
    monkeypatch,
):
    chain = Chain.model_validate(
        {
            "nodes": [
                StructureNode(structure=_structure_at_x(0.0)),
                StructureNode(structure=_structure_at_x(0.5)),
                StructureNode(structure=_structure_at_x(1.0)),
            ],
            "parameters": ChainInputs(),
        }
    )
    for index, node in enumerate(chain):
        node._cached_energy = float(index)

    monkeypatch.setattr(
        "mepd.elementarystep._chain_is_concave",
        lambda chain, engine, verbose=True: SimpleNamespace(
            is_not_concave=False,
            is_concave=True,
            minimization_results=[],
            number_grad_calls=0,
        ),
    )
    monkeypatch.setattr(
        "mepd.elementarystep.is_approx_elem_step",
        lambda chain, engine, slope_thresh=0.1, verbose=True: (False, 0),
    )
    monkeypatch.setattr(
        "mepd.elementarystep.pseudo_irc",
        lambda chain, engine: SimpleNamespace(
            found_reactant=chain[0],
            found_product=chain[-1],
            number_grad_calls=0,
            optimization_succeeded=False,
        ),
    )
    monkeypatch.setattr(
        "mepd.elementarystep.is_identical",
        lambda a, b, **kwargs: np.allclose(a.coords, b.coords),
    )

    result = check_if_elem_step(chain, engine=SimpleNamespace(), verbose=False)

    assert result.is_elem_step is False
    assert result.splitting_criterion == "maxima"


def test_chain_is_concave_runs_endpoint_check_for_engines_without_compute_program(
    monkeypatch,
):
    chain = Chain.model_validate(
        {
            "nodes": [
                StructureNode(structure=_structure_at_x(0.0)),
                StructureNode(structure=_structure_at_x(1.0)),
                StructureNode(structure=_structure_at_x(2.0)),
            ],
            "parameters": ChainInputs(),
        }
    )
    chain[0]._cached_energy = 0.0
    chain[1]._cached_energy = -1.0
    chain[2]._cached_energy = 0.0

    monkeypatch.setattr(
        "mepd.elementarystep._converges_to_an_endpoints",
        lambda **kwargs: (True, [chain[1], chain[0]]),
    )
    monkeypatch.setattr(
        "mepd.elementarystep._run_geom_opt",
        lambda node, engine: (_ for _ in ()).throw(
            AssertionError("ASE-like engines should not take chemcloud fallback path")
        ),
    )

    result = _chain_is_concave(chain=chain, engine=SimpleNamespace(), verbose=False)

    assert result.is_concave is True


def test_endpoint_probe_treats_external_program_failure_as_inconclusive():
    chain = _make_chain_with_energies([0.0, -1.0, 0.0])

    engine = SimpleNamespace(
        steepest_descent=lambda **kwargs: (_ for _ in ()).throw(
            ExternalProgramError("crest failed")
        )
    )

    converged, traj = _converges_to_an_endpoints(
        chain=chain,
        engine=engine,
        node_index=1,
        direction=-1,
        slope_thresh=0.1,
        verbose=False,
    )

    assert converged is False
    assert traj == [chain[1]]


def test_endpoint_probe_still_raises_when_backend_is_missing():
    chain = _make_chain_with_energies([0.0, -1.0, 0.0])

    engine = SimpleNamespace(
        steepest_descent=lambda **kwargs: (_ for _ in ()).throw(
            ProgramNotFoundError("crest not found")
        )
    )

    try:
        _converges_to_an_endpoints(
            chain=chain,
            engine=engine,
            node_index=1,
            direction=-1,
            slope_thresh=0.1,
            verbose=False,
        )
    except ProgramNotFoundError:
        pass
    else:
        raise AssertionError("ProgramNotFoundError should not be swallowed")


def test_chain_is_concave_treats_external_program_failure_as_inconclusive(
    monkeypatch,
):
    chain = _make_chain_with_energies([0.0, -1.0, 0.0])

    monkeypatch.setattr(
        "mepd.elementarystep._run_geom_opt",
        lambda node, engine: (_ for _ in ()).throw(
            ExternalProgramError("crest failed")
        ),
    )

    result = _chain_is_concave(
        chain=chain,
        engine=SimpleNamespace(compute_program="chemcloud"),
        verbose=False,
    )

    assert result.is_concave is True


def test_non_elementary_result_exposes_new_structures_and_writes_xyz(monkeypatch, tmp_path):
    chain = Chain.model_validate(
        {
            "nodes": [
                StructureNode(structure=_structure_at_x(0.0), has_molecular_graph=False),
                StructureNode(structure=_structure_at_x(0.5), has_molecular_graph=False),
                StructureNode(structure=_structure_at_x(1.0), has_molecular_graph=False),
            ],
            "parameters": ChainInputs(node_rms_thre=0.01),
        }
    )
    for index, node in enumerate(chain):
        node._cached_energy = float(index)

    new_node = StructureNode(structure=_structure_at_x(5.0), has_molecular_graph=False)
    monkeypatch.setattr(
        "mepd.elementarystep._chain_is_concave",
        lambda chain, engine, verbose=True: SimpleNamespace(
            is_not_concave=True,
            is_concave=False,
            minimization_results=[new_node],
            number_grad_calls=0,
        ),
    )

    result = check_if_elem_step(chain, engine=SimpleNamespace(), verbose=False)

    assert result.is_elem_step is False
    assert result.splitting_criterion == "minima"
    assert result.new_structures == [new_node]

    out_fp = _write_nodes_xyz(result.new_structures, tmp_path / "new_structures.xyz")
    assert out_fp.read_text(encoding="utf-8").count("\n2\n") == 0
    assert out_fp.read_text(encoding="utf-8").startswith("2\n")
