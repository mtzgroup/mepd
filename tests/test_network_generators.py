"""The product-generator registry for network expansion."""
import pytest

from mepd.discovery import generators
from mepd.discovery.network_expansion import enumerate_bond_changes, graph_edges
from tests.test_network_expansion import _ang, _node

COMMON = dict(charge=0, multiplicity=1, n_break=2, n_form=2, form_distance=4.0, max_products=50,
              allow_radicals=False, allow_zwitterions=False, source=0)


def test_registry_names_the_generators_and_rejects_unknown_ones():
    assert set(generators.GENERATORS) == {"bond-rules"}
    with pytest.raises(ValueError, match="bond-rules.*package.module:function"):
        generators.get_generator("nope")


def test_bond_rules_through_the_registry_is_the_built_in_enumeration():
    node = _node("CCO")
    props, _ = generators.propose("bond-rules", list(node.symbols), _ang(node), graph_edges(node), **COMMON)
    direct, _ = enumerate_bond_changes(list(node.symbols), _ang(node), graph_edges(node))
    assert [(p.broken, p.formed, p.smiles) for p in props] == [(p.broken, p.formed, p.smiles) for p in direct]
