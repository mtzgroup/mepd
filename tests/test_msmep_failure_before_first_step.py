"""A path minimizer that fails before its first step -- e.g. GSM, whose
geodesic seed energies raised a g-xTB error -- must leave a failed branch
with the path it was given, not an IndexError from its empty trajectory
that hides the real error."""

import numpy as np
import pytest

from mepd.chain import Chain
from mepd.errors import ElectronicStructureError
from mepd.inputs import ChainInputs, NEBInputs, RunInputs
from mepd.msmep import MSMEP
from mepd.nodes.node import XYNode
from mepd.engines.flower import FlowerPotential
from mepd.optimizers.cg import ConjugateGradient


class _FailsBeforeFirstStep:
    """Stands in for GSM failing while computing its seed's energies."""

    def __init__(self):
        self.chain_trajectory = []

    def optimize_chain(self):
        raise ElectronicStructureError(msg="g-xTB calculation failed with exit code 128.")


def _msmep_and_chain():
    coords = np.linspace([-2.59807434, -1.499999], [2.5980755, 1.49999912], 7)
    chain_inputs = ChainInputs(k=10, delta_k=9, interpolation="linear", node_ene_thre=10)
    neb_inputs = NEBInputs(v=False, max_steps=5, climb=False, do_elem_step_checks=True)
    run_inputs = RunInputs(path_min_inputs=neb_inputs.__dict__, chain_inputs=chain_inputs.__dict__)
    run_inputs.engine = FlowerPotential()
    run_inputs.optimizer = ConjugateGradient(timestep=0.1)
    chain = Chain.model_validate({"nodes": [XYNode(structure=xy) for xy in coords], "parameters": chain_inputs})
    return MSMEP(run_inputs), chain


def test_the_real_error_reaches_the_caller(monkeypatch):
    msmep, chain = _msmep_and_chain()
    monkeypatch.setattr(msmep, "_construct_path_minimizer", lambda initial_chain: _FailsBeforeFirstStep())
    with pytest.raises(ElectronicStructureError, match="exit code 128"):
        msmep.run_minimize_chain(chain)


def test_a_recursive_search_records_the_failed_step_with_its_real_reason(monkeypatch):
    msmep, chain = _msmep_and_chain()
    monkeypatch.setattr(msmep, "_construct_path_minimizer", lambda initial_chain: _FailsBeforeFirstStep())
    tree = msmep.run_recursive_minimize(chain, max_depth=1)
    assert tree.leaf_status == "electronic_structure_error"
    assert "exit code 128" in tree.leaf_error
    assert tree.ordered_leaves == []          # nothing is TS-optimized from the raw guess
    assert len(tree.failed_chain) == len(chain)   # the path it was given, for node_0_failed.xyz
