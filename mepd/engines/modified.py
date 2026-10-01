"""An engine that is another engine plus an extra energy term.

`ModifiedEngine` subclasses give `_term(nodes) -> (dE, dgrad)` (or None),
and get everything else on the modified surface: energies, gradients,
finite-difference Hessians, and geometry / TS optimizations and IRCs (through
ASE on this engine's own energies). Used for implicit solvent
(mepd.solvation.SolvatedEngine) and external force (mepd.mechanochem.
ForcedEngine); they stack, e.g. a force in solvent.
"""
from __future__ import annotations

import os
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from mepd.chain import Chain
from mepd.engines.engine import Engine
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache


def fresh(node: StructureNode) -> StructureNode:
    """A copy of `node` with nothing cached (so an inner engine cannot hand
    back a modified energy cached on it)."""
    return node.update_coords(node.coords)


class ModifiedEngine(Engine):
    """Subclasses are dataclasses with a `base: Engine` field."""

    base: Engine

    def _term(self, nodes: list[StructureNode]) -> Optional[tuple[NDArray, NDArray]]:
        raise NotImplementedError

    def _inherit(self) -> None:
        for attr in ("disable_molecular_graphs",):
            if hasattr(self.base, attr):
                setattr(self, attr, getattr(self.base, attr))

    # --- energies / gradients -------------------------------------------
    def _run(self, chain: Union[Chain, List]) -> list[StructureNode]:
        nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [n for n in nodes if n._cached_energy is None or n._cached_gradient is None]
        if todo:
            copies = [fresh(n) for n in todo]
            g_base = np.asarray(self.base.compute_gradients(copies), dtype=float)
            e_base = np.array([n.energy for n in copies], dtype=float)
            term = self._term(todo)
            if term is None:
                e, g = e_base, g_base
            else:
                e, g = e_base + term[0], g_base + term[1]
            results = [FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                {"energy": float(ei), "gradient": np.asarray(gi)})}) for ei, gi in zip(e, g)]
            update_node_cache(node_list=todo, results=results)
        return nodes

    def compute_energies(self, chain: Union[Chain, List]) -> NDArray:
        return np.array([n.energy for n in self._run(chain)])

    def compute_gradients(self, chain: Union[Chain, List]) -> NDArray:
        return np.array([n.gradient for n in self._run(chain)])

    # --- everything else through ASE on this engine's own surface -------
    def _ase(self, node: StructureNode):
        from mepd.engines.ase import ASEEngine
        from mepd.engines.gxtb import _GXTBASEResultsCalculator

        calc = _GXTBASEResultsCalculator(self, charge=int(node.structure.charge),
                                         multiplicity=int(node.structure.multiplicity))
        kw = {}
        for key in ("geometry_optimizer", "transition_state_optimizer"):
            if isinstance(getattr(self.base, key, None), str):
                kw[key] = getattr(self.base, key)
        return ASEEngine(calculator=calc, **kw)

    def compute_hessian(self, node: StructureNode, step_size: float | None = None) -> NDArray:
        """Central differences of the (solvated) gradients, batched."""
        h = float(step_size or self.finite_difference_hessian_step_size)
        x0 = np.asarray(node.coords, dtype=float)
        n = x0.size
        displaced = []
        for i in range(n):
            for sign in (1.0, -1.0):
                d = np.zeros(n)
                d[i] = sign * h
                displaced.append(node.update_coords((x0.reshape(-1) + d).reshape(x0.shape)))
        grads = self.compute_gradients(displaced).reshape(n, 2, -1)
        hess = ((grads[:, 0] - grads[:, 1]) / (2 * h)).T
        return 0.5 * (hess + hess.T)

    def _compute_hessian_result(self, node: StructureNode, **kwargs):
        from mepd.engines.engine import build_hessian_result_from_matrix

        return build_hessian_result_from_matrix(node=node, hessian=self.compute_hessian(node, kwargs.get("step_size")))

    def compute_geometry_optimization(self, node: StructureNode, keywords: dict | None = None) -> list[StructureNode]:
        return self._ase(node).compute_geometry_optimization(node, keywords=dict(keywords or {}))

    def compute_geometry_optimizations(self, nodes: list[StructureNode], keywords: dict | None = None):
        from concurrent.futures import ThreadPoolExecutor

        width = max(1, min(len(nodes), os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=width) as pool:
            return list(pool.map(lambda nd: self.compute_geometry_optimization(nd, keywords), nodes))

    def compute_transition_state(self, node: StructureNode, keywords: dict | None = None) -> StructureNode:
        return self._ase(node).compute_transition_state(node=node, keywords=keywords)

    def compute_irc_chain(self, ts_node: StructureNode, keywords: dict | None = None) -> Chain:
        return self._ase(ts_node).compute_irc_chain(ts_node=ts_node, keywords=keywords)


