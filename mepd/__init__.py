import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mepd.chain import Chain
    from mepd.inputs import (
        ChainInputs,
        GIInputs,
        NEBInputs,
        NetworkInputs,
        PathMinInputs,
        RunInputs,
    )
    from mepd.msmep import MSMEP
    from mepd.neb import NEB
    from mepd.nodes.node import Node, StructureNode

_EXPORTS = {
    "Node": ("mepd.nodes.node", "Node"),
    "StructureNode": ("mepd.nodes.node", "StructureNode"),
    "Chain": ("mepd.chain", "Chain"),
    "NEB": ("mepd.neb", "NEB"),
    "MSMEP": ("mepd.msmep", "MSMEP"),
    "PathMinInputs": ("mepd.inputs", "PathMinInputs"),
    "NEBInputs": ("mepd.inputs", "NEBInputs"),
    "ChainInputs": ("mepd.inputs", "ChainInputs"),
    "GIInputs": ("mepd.inputs", "GIInputs"),
    "NetworkInputs": ("mepd.inputs", "NetworkInputs"),
    "RunInputs": ("mepd.inputs", "RunInputs"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
