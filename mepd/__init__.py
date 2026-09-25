"""mepd -- reaction path discovery.

Note the Tcl/Tk guard below: it has to run before any `mepd.*` submodule
(and so before `openbabel.pybel`) is imported, which is why it sits at the
very top of the package's `__init__`.
"""

import os
import sys

# macOS fork safety. `mepd.cli_common._fork_map` runs pairs/TS-opts/mechanism
# searches in *forked* children (fork, not spawn, so the closure's structures,
# engine and RunInputs reach each child for free instead of being pickled).
#
# Importing `tkinter` registers Tcl's macOS notifier `pthread_atfork` handlers.
# After a fork-without-exec, the child's copy of Tcl's `notifierInitLock` is
# recorded as owned by a thread that does not exist there, so the next time
# that child forks -- which is the first thing it does, to launch the engine's
# subprocess -- `AtForkPrepare` tries to take the lock and the kernel SIGKILLs
# the child ("BUG IN CLIENT OF LIBPLATFORM: os_unfair_lock is corrupt"). No
# Python-level exception is raised, so the whole run dies with an opaque
# `BrokenProcessPool` a few milliseconds into the first parallel stage.
#
# Nothing in mepd draws with Tk: the import arrives transitively through
# `openbabel.pybel`, which wants it only for `Molecule.draw()` and already
# falls back to `tk = None` when the import fails. Blocking it keeps Tcl's
# atfork handlers unregistered and the children alive. `None` in `sys.modules`
# is the documented way to make an import fail; set MEPD_ALLOW_TKINTER=1 to
# keep Tk (and run the parallel stages with `--workers 1`).
if (
    sys.platform == "darwin"
    and "_tkinter" not in sys.modules
    and not os.environ.get("MEPD_ALLOW_TKINTER")
):
    for _tk_module in ("_tkinter", "tkinter"):
        sys.modules.setdefault(_tk_module, None)
    del _tk_module

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
    from mepd.neb import NEB
    from mepd.nodes.node import Node, StructureNode, XYNode
    from mepd.msmep import MSMEP
    from mepd.TreeNode import TreeNode
    from mepd.pot import Pot
    from mepd.NetworkBuilder import NetworkBuilder

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
    "NetworkInputs": ("mepd.inputs", "NetworkInputs"),
    "RunInputs": ("mepd.inputs", "RunInputs"),
    "MSMEP": ("mepd.msmep", "MSMEP"),
    "Pot": ("mepd.pot", "Pot"),
}
# `TreeNode` and `NetworkBuilder` (the classes) are intentionally not
# re-exported here: their modules are named `mepd.TreeNode`/
# `mepd.NetworkBuilder` too, and Python's import machinery registers each
# submodule as an attribute on the `mepd` package as soon as anything imports
# it -- which permanently shadows a `__getattr__`-based lazy export of the
# class under the same name. Use `from mepd.TreeNode import TreeNode` /
# `from mepd.NetworkBuilder import NetworkBuilder` instead.

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
