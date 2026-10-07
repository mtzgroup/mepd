"""Frozen atoms for any engine.

`FrozenAtomsEngine` wraps an engine so that a set of atoms never moves:

* their gradient is zero, so NEB/GSM chains, steepest descent and every
  gradient-driven optimizer leave them where they are;
* geometry optimizations, TS searches (Sella) and IRCs run through ASE with
  a `FixAtoms` constraint on them;
* Hessians are *partial*: finite differences over the moving atoms only
  (or a given subset, e.g. a QM region), with frequencies of that block
  (partial Hessian vibrational analysis). Frozen atoms add no zero modes,
  and there are no free translations/rotations to remove.

Used for chain_inputs.frozen_atom_indices on any engine, and for the frozen
environment of a QM/MM region (mepd.qmmm).

Partial Hessian vibrational analysis: H. Li, J. H. Jensen, Theor. Chem. Acc.
107, 211 (2002), doi:10.1007/s00214-002-0356-6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from mepd.chain import Chain
from mepd.engines.engine import Engine, FiniteDifferenceHessianOutput, FiniteDifferenceHessianResults
from mepd.engines.engine import _HESSIAN_EIGENVALUE_TO_CM2
from mepd.engines.modified import ModifiedEngine, fresh
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.helper_functions import get_mass
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache


def partial_hessian_result(node, hessian_block: np.ndarray, atoms: list[int]) -> FiniteDifferenceHessianOutput:
    """Frequencies and modes of the Hessian block over `atoms`; modes are
    returned on the full system (zero on every other atom), the Hessian as
    the full 3N matrix with only that block filled."""
    atoms = list(atoms)
    coords = np.asarray(node.coords, dtype=float)
    n = coords.shape[0]
    h = 0.5 * (hessian_block + hessian_block.T)
    masses = np.repeat(np.sqrt([get_mass(node.symbols[i]) for i in atoms]), 3)
    w, v = np.linalg.eigh(h / np.outer(masses, masses))
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    freqs = [float(np.sign(x) * np.sqrt(abs(x) * _HESSIAN_EIGENVALUE_TO_CM2)) for x in w]
    modes = []
    for k in range(v.shape[1]):
        full = np.zeros((n, 3))
        full[atoms] = (v[:, k] / masses).reshape(-1, 3)   # mass-weighted -> Cartesian displacement
        full /= max(np.linalg.norm(full), 1e-300)
        modes.append(full)
    full_h = np.zeros((3 * n, 3 * n))
    dof = np.array([3 * a + c for a in atoms for c in range(3)])
    full_h[np.ix_(dof, dof)] = h
    return FiniteDifferenceHessianOutput(
        input_data=SimpleNamespace(structure=getattr(node, "structure", None)),
        results=FiniteDifferenceHessianResults(hessian=full_h, normal_modes_cartesian=modes, freqs_wavenumber=freqs))


@dataclass
class FrozenAtomsEngine(ModifiedEngine):
    """`frozen`: atom indices that never move. `hessian_atoms`: the atoms a
    Hessian is taken over (default: every atom that is not frozen)."""

    base: Engine
    frozen: list[int] = field(default_factory=list)
    hessian_atoms: Optional[list[int]] = None

    def __post_init__(self):
        self.frozen = sorted({int(i) for i in self.frozen})
        self._inherit()

    def __getattr__(self, name):
        # Anything the wrapped engine offers (n_parallel, region, decompose,
        # prepare_node_for_comparison, ...) stays reachable.
        if name.startswith("__") or name in ("base", "frozen", "hessian_atoms"):
            raise AttributeError(name)
        return getattr(self.__dict__["base"] if "base" in self.__dict__ else None, name)

    def __repr__(self) -> str:
        return f"FrozenAtomsEngine({self.base!r}, {len(self.frozen)} frozen atoms)"

    def _mask(self, natoms: int) -> np.ndarray:
        idx = [i for i in self.frozen if i < natoms]
        return np.asarray(idx, dtype=int)

    def _run(self, chain: Union[Chain, List]) -> list[StructureNode]:
        nodes = chain.nodes if isinstance(chain, Chain) else list(chain)
        todo = [n for n in nodes if n._cached_energy is None or n._cached_gradient is None]
        if todo:
            copies = [fresh(n) for n in todo]
            g = np.array(self.base.compute_gradients(copies), dtype=float)
            e = np.array([n.energy for n in copies], dtype=float)
            if len(g) and self.frozen:
                g[:, self._mask(g.shape[1])] = 0.0
            update_node_cache(node_list=todo, results=[
                FakeQCIOOutput.model_validate({"results": FakeQCIOResults.model_validate(
                    {"energy": float(ei), "gradient": gi})}) for ei, gi in zip(e, g)])
        else:
            for n in nodes:   # gradients cached by someone else (an ASE trajectory) carry frozen forces
                if self.frozen and n._cached_gradient is not None:
                    grad = np.array(n._cached_gradient, dtype=float)
                    grad[self._mask(len(grad))] = 0.0
                    n._cached_gradient = grad
        return nodes

    def _ase(self, node: StructureNode):
        eng = super()._ase(node)
        eng.fixed_atoms = list(self.frozen)
        return eng

    def _micro(self):
        """Microiterations for a QM/MM engine with a moving environment."""
        from mepd.engines.qmmm import QMMMEngine

        base = self.base
        if not isinstance(base, QMMMEngine) or not getattr(base.region, "microiterations", True):
            return None
        from mepd.engines.qmmm import _OpenMMLow

        if not isinstance(base._low, _OpenMMLow):
            # An xtb environment costs a program start per inner step (~50 ms,
            # hundreds of steps per relaxation): slower than plain optimization.
            return None
        from mepd.engines.microiter import MicroIterations

        micro = MicroIterations(base, self.frozen)
        return micro if micro.inner else None

    def compute_geometry_optimization(self, node: StructureNode, keywords: dict | None = None) -> list[StructureNode]:
        micro = self._micro()
        if micro is not None:
            return micro.minimize(node, keywords)
        return super().compute_geometry_optimization(node, keywords)

    def compute_transition_state(self, node: StructureNode, keywords: dict | None = None) -> StructureNode:
        micro = self._micro()
        if micro is not None:
            return micro.transition_state(node, keywords)
        keywords = {k: v for k, v in (keywords or {}).items() if k not in ("v0", "echo", "exact_hessian")} or None
        return super().compute_transition_state(node, keywords)

    def compute_irc_chain(self, ts_node: StructureNode, keywords: dict | None = None):
        micro = self._micro()
        if micro is not None:
            return micro.irc(ts_node, keywords)
        keywords = {k: v for k, v in (keywords or {}).items() if k != "echo"} or None
        return super().compute_irc_chain(ts_node, keywords)

    def _hessian_atoms(self, natoms: int) -> list[int]:
        if self.hessian_atoms is not None:
            return [i for i in self.hessian_atoms if i < natoms]
        frozen = set(self.frozen)
        return [i for i in range(natoms) if i not in frozen]

    def hessian_block(self, node: StructureNode, step_size: float | None = None) -> tuple[np.ndarray, list[int]]:
        """Central differences of gradients over the Hessian atoms only."""
        h = float(step_size or self.finite_difference_hessian_step_size)
        x0 = np.asarray(node.coords, dtype=float)
        atoms = self._hessian_atoms(x0.shape[0])
        dof = [3 * a + c for a in atoms for c in range(3)]
        displaced = []
        for k in dof:
            for sign in (1.0, -1.0):
                d = np.zeros(x0.size)
                d[k] = sign * h
                displaced.append(node.update_coords((x0.reshape(-1) + d).reshape(x0.shape)))
        # The base engine's gradients: the frozen atoms' rows are needed for
        # nothing here, but zeroing them must not hide the block's couplings.
        grads = np.array(self.base.compute_gradients([fresh(n) for n in displaced]), dtype=float)
        grads = grads.reshape(len(dof), 2, -1)[:, :, dof]
        block = ((grads[:, 0] - grads[:, 1]) / (2 * h)).T
        return 0.5 * (block + block.T), atoms

    def compute_hessian(self, node: StructureNode, step_size: float | None = None) -> NDArray:
        block, atoms = self.hessian_block(node, step_size)
        n = np.asarray(node.coords).shape[0]
        full = np.zeros((3 * n, 3 * n))
        dof = np.array([3 * a + c for a in atoms for c in range(3)])
        full[np.ix_(dof, dof)] = block
        return full

    def _compute_hessian_result(self, node: StructureNode, **kwargs):
        block, atoms = self.hessian_block(node, kwargs.get("step_size"))
        return partial_hessian_result(node, block, atoms)
