"""Microiterations for QM/MM optimizations: the environment is relaxed with
cheap force-field steps between QM steps, so the expensive optimizer only
moves the QM region.

Each outer step (one QM/MM energy and gradient):
  1. the QM atoms (and the MM atoms of cut bonds) are where the outer
     optimizer put them;
  2. the moving environment is minimized with the QM region held fixed, on
     the low level's energy plus, for electrostatic embedding, the Coulomb
     energy of the environment's point charges with the QM region's atomic
     charges from the last QM calculation;
  3. the QM/MM energy and gradient are computed there; the outer optimizer
     sees the forces on the QM region only.

The outer surface is the environment-relaxed (adiabatic) one, E(x_QM) =
min over x_MM of E(x_QM, x_MM), so minima are those of the full QM/MM
surface, a TS search finds a saddle in the QM coordinates with the
environment relaxed, and an IRC follows it the same way. Outer optimizers:
LBFGS (minima), Sella (TS, IRC).

QM/MM microiterative optimization: F. Maseras, K. Morokuma, J. Comput.
Chem. 16, 1170 (1995), doi:10.1002/jcc.540160911; for TS searches, e.g.
J. Kästner, S. Thiel, H. M. Senn, P. Sherwood, W. Thiel, J. Chem. Theory
Comput. 3, 1064 (2007), doi:10.1021/ct600346p.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR

from mepd.chain import Chain
from mepd.errors import ElectronicStructureError
from mepd.fakeoutputs import FakeQCIOOutput, FakeQCIOResults
from mepd.nodes.node import StructureNode
from mepd.nodes.nodehelpers import update_node_cache

HARTREE_EV = 27.211386245988


class MicroIterations:
    def __init__(self, qmmm, frozen=(), inner_gtol: float = 1e-5, inner_maxiter: int = 2000):
        r = qmmm.region
        self.qmmm, self.region = qmmm, r
        frozen = set(int(i) for i in frozen)
        partners = {m for _, m in r.links}
        self.outer = sorted(set(r.qm_atoms) | (partners - frozen))
        self.inner = [i for i in r.active_mm_atoms if i not in frozen and i not in partners]
        self.frozen = sorted(frozen)
        self.electrostatic = bool(getattr(qmmm._low, "electrostatic", False))
        self.inner_gtol, self.inner_maxiter = inner_gtol, inner_maxiter
        self.max_move = 0.5
        self.calls = {"qm": 0, "inner_steps": 0}
        self._reduced = None

    # ------------------------------------------------------------ inner
    def _inner_energy(self, x: np.ndarray, qm_charges: Optional[np.ndarray], charge: int):
        if self._reduced is None and self.inner and hasattr(self.qmmm._low, "reduced"):
            # A cut-off force field: only the atoms near the moving ones matter
            # to their forces (a protein: a few thousand atoms fewer per step).
            self._reduced = self.qmmm._low.reduced(sorted(set(self.inner) | set(self.outer))) or False
        if self._reduced:
            e, g = self._reduced.terms(x)
        else:
            low = self.qmmm._low.terms([x], charge)[0]
            e, g = float(low["energy"]), np.array(low["gradient"], dtype=float)
        if self.electrostatic and qm_charges is not None:
            model = self.region.model_coords(x)
            idx, q = self.qmmm._low.point_charges()
            if len(idx):
                d = x[idx][:, None, :] - model[None, :, :]
                r = np.linalg.norm(d, axis=2)
                qq = q[:, None] * np.asarray(qm_charges)[None, : len(model)]
                e += float(np.sum(qq / r))
                g[idx] += np.sum((-qq / r ** 3)[:, :, None] * d, axis=1)
        return e, g

    def relax(self, x: np.ndarray, qm_charges, charge: int, correction: Optional[np.ndarray] = None) -> np.ndarray:
        """The environment minimized with the QM region fixed. `correction`
        (environment atoms x 3, Eh/bohr): a constant force added so the cheap
        energy's gradient equals the full QM/MM gradient where the last QM
        calculation was made (the approximate QM charges then shift the
        environment's minimum only to second order)."""
        from scipy.optimize import minimize

        if not self.inner:
            return x
        idx = np.asarray(self.inner, dtype=int)
        x = np.array(x, dtype=float)
        x_ref = x[idx].copy()
        corr = np.zeros_like(x_ref) if correction is None else np.asarray(correction)

        def fg(flat):
            y = x.copy()
            y[idx] = flat.reshape(-1, 3)
            e, g = self._inner_energy(y, qm_charges, charge)
            self.calls["inner_steps"] += 1
            return e + float(np.sum(corr * (y[idx] - x_ref))), (g[idx] + corr).ravel()

        # At most `max_move` bohr per coordinate per relaxation: the correction
        # is a constant force, trustworthy only near where it was computed.
        start = x[idx].ravel()
        bounds = [(v - self.max_move, v + self.max_move) for v in start]
        res = minimize(fg, start, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.inner_maxiter, "gtol": self.inner_gtol})
        x[idx] = res.x.reshape(-1, 3)
        return x

    # ------------------------------------------------------------ outer
    def _node(self, structure, x, res) -> StructureNode:
        g = np.array(res["gradient"], dtype=float)
        if self.frozen:
            g[self.frozen] = 0.0
        node = StructureNode(structure=structure.model_copy(update={"geometry": x}), has_molecular_graph=False)
        update_node_cache(node_list=[node], results=[FakeQCIOOutput.model_validate(
            {"results": FakeQCIOResults.model_validate({"energy": res["energy"], "gradient": g})})])
        return node

    def _calculator(self, structure, x0):
        from ase.calculators.calculator import Calculator, all_changes

        engine = self
        charge = int(structure.charge)
        node0 = StructureNode(structure=structure.model_copy(update={"geometry": x0}), has_molecular_graph=False)
        first = self.qmmm.evaluate([node0])[0]
        self.calls["qm"] += 1

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self):
                super().__init__()
                self.x = np.array(x0, dtype=float)
                self.charges = first.get("qm_charges")
                self.trajectory = []
                self.correction = engine._correction(x0, first, self.charges, charge)

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                x = self.x.copy()
                x[engine.outer] = self.atoms.get_positions() * ANGSTROM_TO_BOHR
                x = engine.relax(x, self.charges, charge, self.correction)
                res = engine.qmmm.evaluate([StructureNode(structure=structure.model_copy(update={"geometry": x}),
                                                          has_molecular_graph=False)])[0]
                engine.calls["qm"] += 1
                self.x = x
                if res.get("qm_charges") is not None:
                    self.charges = res["qm_charges"]
                self.correction = engine._correction(x, res, self.charges, charge)
                self.trajectory.append(engine._node(structure, x, res))
                engine._report(len(self.trajectory), res)
                self.results["energy"] = res["energy"] * HARTREE_EV
                self.results["forces"] = -np.asarray(res["gradient"])[engine.outer] * HARTREE_EV * ANGSTROM_TO_BOHR

        return _Calc()

    def _report(self, step: int, res: dict) -> None:
        """A line every 10 QM/MM steps (a log shows the optimization moving)."""
        import time

        now = time.time()
        if step == 1:
            self._t0, self._inner0 = now, self.calls["inner_steps"]
        if step % 10:
            return
        f = np.abs(np.asarray(res["gradient"])[self.outer]).max() * HARTREE_EV * ANGSTROM_TO_BOHR
        inner = (self.calls["inner_steps"] - self._inner0) / max(step, 1)
        print(f"  QM/MM step {step}: energy {res['energy']:.6f} Eh, max QM force {f:.3f} eV/Å, "
              f"{inner:.0f} environment steps per QM step, {(now - self._t0) / step:.1f} s per step", flush=True)

    def _correction(self, x, res, qm_charges, charge: int) -> Optional[np.ndarray]:
        """Full QM/MM gradient minus the cheap one, on the environment atoms."""
        if not self.inner:
            return None
        idx = np.asarray(self.inner, dtype=int)
        _, g_cheap = self._inner_energy(np.asarray(x, dtype=float), qm_charges, charge)
        return np.asarray(res["gradient"], dtype=float)[idx] - g_cheap[idx]

    def _atoms(self, structure, x):
        from ase import Atoms

        return Atoms(symbols=[structure.symbols[i] for i in self.outer], positions=np.asarray(x)[self.outer] / ANGSTROM_TO_BOHR)

    @staticmethod
    def _run_kwargs(keywords, fmax_default, steps_default):
        kw = dict(keywords or {})
        fmax = float(kw.pop("fmax", fmax_default))
        steps = int(kw.pop("maxiter", kw.pop("maxit", kw.pop("steps", steps_default))))
        return fmax, steps, kw

    def minimize(self, node: StructureNode, keywords: dict | None = None) -> list[StructureNode]:
        from ase.optimize import LBFGS

        fmax, steps, _ = self._run_kwargs(keywords, 0.01, 500)
        x0 = np.asarray(node.coords, dtype=float)
        calc = self._calculator(node.structure, x0)
        calc.x = self.relax(x0, calc.charges, int(node.structure.charge), calc.correction)
        atoms = self._atoms(node.structure, calc.x)
        atoms.calc = calc
        try:
            LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        except Exception as exc:
            raise ElectronicStructureError(msg=f"microiterative minimization failed: {exc}") from exc
        return calc.trajectory

    def transition_state(self, node: StructureNode, keywords: dict | None = None) -> StructureNode:
        from sella import Sella

        fmax, steps, kw = self._run_kwargs(keywords, 0.01, 500)
        opt_kw = dict(kw.pop("optimizer_kwds", kw.pop("optimizer_kwargs", {})) or {})
        opt_kw.setdefault("order", 1)
        # The QM region sits in a fixed environment: moving or turning it as a
        # whole is real motion, not a free molecule's to project out.
        opt_kw.setdefault("proj_trans", False)
        opt_kw.setdefault("proj_rot", False)
        direction = kw.pop("v0", None)
        if direction is not None:
            # The reaction direction (e.g. a path's tangent at its peak), on
            # the atoms Sella moves: its first guess for the mode to climb.
            v = np.asarray(direction, dtype=float).reshape(-1, 3)[self.outer].ravel()
            if np.linalg.norm(v) > 1e-8:
                opt_kw["v0"] = v / np.linalg.norm(v)
        exact = kw.pop("exact_hessian", True)
        echo = kw.pop("echo", None) or (lambda *_: None)
        x0 = np.asarray(node.coords, dtype=float)
        calc = self._calculator(node.structure, x0)
        calc.x = self.relax(x0, calc.charges, int(node.structure.charge), calc.correction)
        atoms = self._atoms(node.structure, calc.x)
        atoms.calc = calc
        if exact and "H0" not in opt_kw:
            # Sella's model Hessian (and the tangent as v0) loses the reaction
            # mode on the environment-relaxed surface within a few steps; an
            # exact Hessian of the QM region at the guess starts it on it.
            opt_kw["H0"] = self.qm_hessian(node.structure, calc.x, direction, echo=echo)
            opt_kw.setdefault("eig", False)
        try:
            Sella(atoms, logfile=None, **opt_kw).run(fmax=fmax, steps=steps)
        except Exception as exc:
            raise ElectronicStructureError(msg=f"microiterative TS search failed: {exc}") from exc
        if not calc.trajectory:
            raise ElectronicStructureError(msg="microiterative TS search produced no structure")
        return calc.trajectory[-1]

    def qm_hessian(self, structure, x, direction=None, step: float = 0.005, echo=None) -> np.ndarray:
        """Central finite-difference Hessian (eV/Å², Sella's units) over the
        atoms the outer optimizer moves, with the environment held where it
        is: 6 x (QM atoms + link partners) QM/MM gradients.

        With a reaction direction (e.g. the path tangent at its peak), the
        negative mode overlapping it most is kept as the one to climb and any
        other negative eigenvalues are made positive, so order-1 Sella follows
        the reaction rather than, say, a methyl rotation."""
        echo = echo or (lambda *_: None)
        x = np.asarray(x, dtype=float)
        n = 3 * len(self.outer)
        h = step * ANGSTROM_TO_BOHR
        echo(f"Exact QM-region Hessian at the TS guess: {2 * n} QM/MM gradients...")
        cols = []
        for k in range(n):
            a, c = divmod(k, 3)
            grads = []
            for sgn in (1, -1):
                y = x.copy()
                y[self.outer[a], c] += sgn * h
                res = self.qmmm.evaluate([StructureNode(structure=structure.model_copy(update={"geometry": y}),
                                                        has_molecular_graph=False)])[0]
                self.calls["qm"] += 1
                grads.append(np.asarray(res["gradient"], dtype=float)[self.outer].ravel())
            cols.append((grads[0] - grads[1]) / (2 * h))
            if (k + 1) % 12 == 0 or k + 1 == n:
                echo(f"  {k + 1}/{n} coordinates")
        H = np.array(cols)
        H = 0.5 * (H + H.T) * HARTREE_EV * ANGSTROM_TO_BOHR ** 2      # Eh/bohr² -> eV/Å²
        w, V = np.linalg.eigh(H)
        neg = [i for i in range(len(w)) if w[i] < 0]
        keep = None
        if direction is not None and neg:
            d = np.asarray(direction, dtype=float).reshape(-1, 3)[self.outer].ravel()
            if np.linalg.norm(d) > 1e-8:
                d = d / np.linalg.norm(d)
                ov = [abs(V[:, i] @ d) for i in neg]
                keep = neg[int(np.argmax(ov))]
                echo(f"  negative modes: {', '.join(f'{w[i]:.3f}' for i in neg)} eV/Å²; climbing {w[keep]:.3f} "
                     f"(overlap with the path tangent {max(ov):.2f})")
        elif neg:
            echo(f"  negative modes: {', '.join(f'{w[i]:.3f}' for i in neg)} eV/Å²")
        else:
            echo("  no negative mode at the guess: Sella will have to find one")
        if keep is not None:
            for i in neg:
                if i != keep:
                    w[i] = abs(w[i])
        return (V * w) @ V.T

    def irc(self, ts_node: StructureNode, keywords: dict | None = None) -> Chain:
        from sella import IRC

        fmax, steps, kw = self._run_kwargs(keywords, 0.1, 1000)
        echo = kw.pop("echo", None) or (lambda *_: None)
        opt_kw = dict(kw.pop("optimizer_kwds", kw.pop("optimizer_kwargs", {})) or {})
        for key in ("dx", "eta", "gamma", "keep_going"):
            if key in kw:
                opt_kw[key] = kw.pop(key)
        # The environment relaxes between steps, so the forces carry a little
        # noise and Sella's corrector can miss its tolerance on a step; go on
        # (the IRC is judged by where its ends land).
        opt_kw.setdefault("keep_going", True)
        x_ts = np.asarray(ts_node.coords, dtype=float)
        branches = {}
        for direction in ("reverse", "forward"):
            calc = self._calculator(ts_node.structure, x_ts)
            atoms = self._atoms(ts_node.structure, x_ts)
            atoms.calc = calc
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        IRC(atoms, logfile=None, trajectory=str(Path(tmp) / "irc.traj"), **opt_kw).run(
                            fmax=fmax, steps=steps, direction=direction)
                except Exception as exc:
                    raise ElectronicStructureError(
                        msg=f"microiterative IRC failed: {type(exc).__name__}: {exc}") from exc
            n_loose = sum("inner loop failed" in str(w.message) for w in caught)
            if n_loose:
                echo(f"  IRC {direction}: the corrector missed its tolerance on {n_loose} step(s); went on")
            branches[direction] = calc.trajectory
        ts = ts_node.copy()
        if ts._cached_energy is None:
            ts = self._node(ts_node.structure, x_ts, self.qmmm.evaluate([ts_node])[0])
        nodes = list(reversed(branches["reverse"])) + [ts] + list(branches["forward"])
        return Chain.model_validate({"nodes": nodes})
