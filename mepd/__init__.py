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
    from mepd.msmep import MSMEP
    from mepd.TreeNode import TreeNode

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
    "MSMEP": ("mepd.msmep", "MSMEP"),
}
# `TreeNode` (the class) is intentionally not re-exported here: the module is
# also named `mepd.TreeNode`, and Python's import machinery registers that
# submodule as an attribute on the `mepd` package as soon as anything imports
# it -- which permanently shadows a `__getattr__`-based lazy export of the
# class under the same name. Use `from mepd.TreeNode import TreeNode` instead.

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
