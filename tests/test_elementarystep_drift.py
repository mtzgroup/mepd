"""An interior minimum that drifts toward an endpoint is only that endpoint
if it is the same molecule: a distinct species near it (e.g. acetylene +
1-butene on a path to acetylene + 2-butene) must be optimized and kept as a
split point, not waved through as "the product"."""

import numpy as np
from qcdata import Structure

import mepd.elementarystep as es
from mepd.chain import Chain
from mepd.inputs import ChainInputs
from mepd.nodes.node import StructureNode

ETHANOL = """9

C -0.9250 0.0000 0.0000
C 0.5750 0.0000 0.0000
O 1.0500 1.3400 0.0000
H -1.2950 1.0300 0.0000
H -1.2950 -0.5150 0.8920
H -1.2950 -0.5150 -0.8920
H 0.9450 -0.5150 0.8920
H 0.9450 -0.5150 -0.8920
H 2.0200 1.3400 0.0000
"""
DIMETHYL_ETHER = """9

C -1.1600 0.0000 0.0000
O 0.0000 0.7800 0.0000
C 1.1600 0.0000 0.0000
H -2.0500 0.6300 0.0000
H -1.1800 -0.6400 0.8900
H -1.1800 -0.6400 -0.8900
H 2.0500 0.6300 0.0000
H 1.1800 -0.6400 0.8900
H 1.1800 -0.6400 -0.8900
"""


def _node(xyz):
    n = StructureNode(structure=Structure.from_xyz(xyz))
    n._cached_energy = -154.0
    return n


def _run(monkeypatch, drift_end):
    """A 3-image chain ethanol -> ? -> dimethyl ether whose middle image is
    a minimum that drifts toward the product (RMSD) and ends as `drift_end`."""
    a, b = _node(ETHANOL), _node(DIMETHYL_ETHER)
    chain = Chain.model_validate({"nodes": [a, _node(ETHANOL), b], "parameters": ChainInputs()})
    monkeypatch.setattr(es, "_get_ind_minima", lambda chain: [1])
    monkeypatch.setattr(es, "_converges_to_an_endpoints",
                        lambda **kw: (False, [chain[1], drift_end]))
    # Moving 0.5 away from the reactant and 0.5 toward the product: a clear drift.
    monkeypatch.setattr(es, "_distances_to_refs",
                        lambda ref1, ref2, raw_node: [1.0, 1.0] if raw_node is chain[1] else [1.5, 0.5])
    optimized = []
    monkeypatch.setattr(es, "_run_geom_opt", lambda node, engine: optimized.append(node) or [drift_end])
    es._chain_is_concave(chain=chain, engine=object(), verbose=False)
    return optimized


def test_a_different_molecule_drifting_toward_the_product_is_optimized(monkeypatch):
    assert len(_run(monkeypatch, _node(ETHANOL))) == 1      # ethanol is not the dimethyl-ether product


def test_the_product_itself_is_still_waved_through(monkeypatch):
    assert _run(monkeypatch, _node(DIMETHYL_ETHER)) == []    # unchanged: no optimization needed
