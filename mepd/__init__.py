import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mepd.chain import Chain
    from mepd.inputs import (
        ChainInputs,
        GIInputs,
        NEBInputs,
        PathMinInputs,
        RunInputs,
    )
    from mepd.neb import NEB
    from mepd.nodes.node import Node, StructureNode, XYNode

_CORE_EXPORTS = {
    "Node": ("mepd.nodes.node", "Node"),
    "StructureNode": ("mepd.nodes.node", "StructureNode"),
    "XYNode": ("mepd.nodes.node", "XYNode"),
    "Chain": ("mepd.chain", "Chain"),
    "NEB": ("mepd.neb", "NEB"),
    "PathMinInputs": ("mepd.inputs", "PathMinInputs"),
    "NEBInputs": ("mepd.inputs", "NEBInputs"),
    "ChainInputs": ("mepd.inputs", "ChainInputs"),
    "GIInputs": ("mepd.inputs", "GIInputs"),
    "RunInputs": ("mepd.inputs", "RunInputs"),
}

__all__ = sorted([*_CORE_EXPORTS, "engines"])


def __getattr__(name: str):
    if name == "engines":
        import mepd.engines as engines_module
        globals()["engines"] = engines_module
        return engines_module
    if name in _CORE_EXPORTS:
        module_name, attr_name = _CORE_EXPORTS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
