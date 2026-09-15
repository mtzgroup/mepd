from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Union, List
from mepd.chain import Chain
import numpy as np


from mepd.nodes.node import Node
from mepd.fakeoutputs import FakeQCIOOutput
from mepd.helper_functions import get_mass
from qcdata import ProgramOutput
import qcconst.constants as _qcconst_constants

# Hartree/(bohr^2 * amu) -> (cm^-1)^2, derived from CODATA constants (matches
# the standard ~5140.487 cm^-1 per sqrt(Hartree/(bohr^2*amu)) conversion used
# throughout quantum chemistry). Needed to turn mass-weighted Hessian
# eigenvalues into physically meaningful wavenumbers -- see
# build_hessian_result_from_matrix.
_HESSIAN_EIGENVALUE_TO_CM2 = (
    float(_qcconst_constants.phys["Hartree energy"])
    / float(_qcconst_constants.phys["Bohr radius"]) ** 2
    / float(_qcconst_constants.phys["atomic mass constant"])
) / (2 * np.pi * float(_qcconst_constants.phys["speed of light in vacuum"]) * 100) ** 2


@dataclass
class FiniteDifferenceHessianResults:
    hessian: np.ndarray
    normal_modes_cartesian: list[np.ndarray]
    freqs_wavenumber: list[float]


@dataclass
class FiniteDifferenceHessianOutput:
    input_data: Any
    results: FiniteDifferenceHessianResults
    success: bool = True

    @property
    def return_result(self) -> np.ndarray:
        return self.results.hessian

    def save(self, filename: str | Path) -> None:
        payload = {
            "success": bool(self.success),
            "input_data": {
                "structure": (
                    self.input_data.structure.model_dump()
                    if getattr(self.input_data, "structure", None) is not None
                    and hasattr(self.input_data.structure, "model_dump")
                    else None
                ),
            },
            "results": {
                "hessian": np.asarray(self.results.hessian, dtype=float).tolist(),
                "normal_modes_cartesian": [
                    np.asarray(mode, dtype=float).tolist()
                    for mode in self.results.normal_modes_cartesian
                ],
                "freqs_wavenumber": [float(freq) for freq in self.results.freqs_wavenumber],
            },
        }
        Path(filename).write_text(json.dumps(payload, indent=2))


def build_hessian_result_from_matrix(node: Node, hessian: np.ndarray) -> FiniteDifferenceHessianOutput:
    hessian_arr = np.asarray(hessian, dtype=float)
    if hessian_arr.ndim != 2 or hessian_arr.shape[0] != hessian_arr.shape[1]:
        raise ValueError("Hessian must be a square 2D array.")

    # Numerical finite differences are not exactly symmetric; enforce symmetry.
    hessian_arr = 0.5 * (hessian_arr + hessian_arr.T)

    refshape = np.asarray(node.coords).shape
    natoms = refshape[0]

    # Mass-weighting is required before these eigenvalues mean anything as
    # vibrational frequencies. Without it: (a) the 6 (5 for linear molecules)
    # translation/rotation eigenvalues -- which should be exactly zero -- come
    # out as small positive *or negative* numerical noise, indistinguishable
    # from a genuine imaginary mode, so any minima-validation check comparing
    # against a frequency_cutoff near zero rejects virtually every real
    # minimum; (b) the remaining eigenvalues aren't in real cm^-1 units at
    # all. Confirmed directly: an unweighted Hessian on a relaxed water
    # molecule (an unambiguous minimum) reported min "frequency" -9.7e-6 and
    # vibrational "frequencies" of 0.4/0.9/1.0 (real values: ~1600/3650/3750
    # cm^-1). After mass-weighting, only the first 6 modes are near-zero
    # (trans/rot) and are excluded from `freqs_wavenumber`, so a
    # `frequency_cutoff` in cm^-1 is now comparing against a genuine
    # vibrational spectrum, not translation/rotation noise.
    try:
        symbols = list(node.symbols)
        masses_amu = np.array([get_mass(s) for s in symbols], dtype=float)
        sqrt_mass = np.repeat(np.sqrt(masses_amu), 3)
        mass_weighted = hessian_arr / np.outer(sqrt_mass, sqrt_mass)
    except Exception:
        # No atomic masses available (e.g. a non-molecular toy-potential
        # node) -- fall back to the raw Hessian; frequencies won't be
        # physically meaningful, but this keeps non-molecular engines working.
        mass_weighted = hessian_arr

    eigvals, eigvecs = np.linalg.eigh(mass_weighted)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    modes = [eigvecs[:, i].reshape(refshape) for i in range(eigvecs.shape[1])]
    freqs_cm = [
        float(np.sign(v) * np.sqrt(abs(v) * _HESSIAN_EIGENVALUE_TO_CM2)) for v in eigvals
    ]

    # Drop the lowest-magnitude modes as translation/rotation rather than
    # genuine vibrations -- they're the smallest in magnitude by construction
    # (true zero modes plus noise), while real vibrational force constants
    # are much larger. Diatomics (5 trans/rot dof) are special-cased; other
    # linear molecules (also 5, not 6) aren't detected, so one genuine very
    # low-frequency bending mode would be discarded alongside rotation for
    # a linear polyatomic -- a known limitation, not exercised by anything
    # this package currently targets.
    n_trans_rot = 5 if natoms == 2 else 6
    n_trans_rot = min(n_trans_rot, len(freqs_cm))
    keep = sorted(range(len(freqs_cm)), key=lambda i: abs(eigvals[i]))[n_trans_rot:]
    keep.sort()
    freqs = [freqs_cm[i] for i in keep]
    modes = [modes[i] for i in keep]

    return FiniteDifferenceHessianOutput(
        input_data=SimpleNamespace(structure=getattr(node, "structure", None)),
        results=FiniteDifferenceHessianResults(
            hessian=hessian_arr,
            normal_modes_cartesian=modes,
            freqs_wavenumber=freqs,
        ),
        success=True,
    )


@dataclass
class Engine(ABC):
    finite_difference_hessian_step_size: ClassVar[float] = 1e-3

    @abstractmethod
    def compute_gradients(
        self, chain: Union[Chain, List]
    ) -> Union[FakeQCIOOutput, ProgramOutput]:
        """
        returns the gradients for each node in the chain as
        specified by the object inputs
        """
        ...

    @abstractmethod
    def compute_energies(
        self, chain: Union[Chain, List]
    ) -> Union[FakeQCIOOutput, ProgramOutput]:
        """
        returns the energies for each node in the chain as
        specified by the object inputs
        """
        ...

    def _warn_finite_difference_hessian_fallback(
        self,
        *,
        node: Node,
        step_size: float,
        expected_energy_evaluations: int,
    ) -> None:
        message = (
            "Using default finite-difference Hessian fallback "
            f"for `{self.__class__.__name__}` (step={step_size:g} Bohr; "
            f"~{expected_energy_evaluations} energy evaluations)."
        )
        logging.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def compute_hessian(
        self,
        node: Node,
        step_size: float | None = None,
    ) -> np.ndarray:
        h = float(step_size if step_size is not None else self.finite_difference_hessian_step_size)
        if h <= 0:
            raise ValueError("finite-difference Hessian step size must be positive.")

        coords = np.asarray(node.coords, dtype=float)
        refshape = coords.shape
        x0 = coords.reshape(-1)
        ndof = x0.size
        if ndof == 0:
            raise ValueError("Cannot compute Hessian for a node with zero coordinates.")

        expected_energy_evaluations = 1 + 2 * ndof * ndof
        self._warn_finite_difference_hessian_fallback(
            node=node,
            step_size=h,
            expected_energy_evaluations=expected_energy_evaluations,
        )

        def _energy_at(displacement: np.ndarray) -> float:
            displaced = node.update_coords((x0 + displacement).reshape(refshape))
            return float(self.compute_energies([displaced])[0])

        h2 = h * h
        e0 = _energy_at(np.zeros(ndof, dtype=float))
        hessian = np.zeros((ndof, ndof), dtype=float)

        e_plus = np.zeros(ndof, dtype=float)
        e_minus = np.zeros(ndof, dtype=float)
        for i in range(ndof):
            disp = np.zeros(ndof, dtype=float)
            disp[i] = h
            e_plus[i] = _energy_at(disp)
            disp[i] = -h
            e_minus[i] = _energy_at(disp)
            hessian[i, i] = (e_plus[i] - 2.0 * e0 + e_minus[i]) / h2

        for i in range(ndof):
            for j in range(i + 1, ndof):
                disp = np.zeros(ndof, dtype=float)
                disp[i], disp[j] = h, h
                e_pp = _energy_at(disp)
                disp[i], disp[j] = h, -h
                e_pm = _energy_at(disp)
                disp[i], disp[j] = -h, h
                e_mp = _energy_at(disp)
                disp[i], disp[j] = -h, -h
                e_mm = _energy_at(disp)
                value = (e_pp - e_pm - e_mp + e_mm) / (4.0 * h2)
                hessian[i, j] = value
                hessian[j, i] = value

        return hessian

    def _compute_hessian_result(
        self,
        node: Node,
        **kwargs,
    ) -> FiniteDifferenceHessianOutput:
        step_size = kwargs.pop("step_size", None)
        hessian = self.compute_hessian(node=node, step_size=step_size)
        return build_hessian_result_from_matrix(node=node, hessian=hessian)

    def steepest_descent(
        self,
        node: Node,
        ss=1.0,
        max_steps=500,
        ene_thre: float = 1e-6,
        grad_thre: float = 1e-4,
        mass_weighted: bool = False,
    ) -> list[Node]:
        # print("************\n\n\n\nRUNNING STEEPEST DESCENT\n\n\n\nn***********")
        history = []
        last_node = node.copy()
        # make sure the node isn't frozen so it returns a gradient
        last_node.converged = False

        # The loop below reads last_node.gradient/.energy starting on its very
        # first iteration, before ever calling compute_gradients/compute_energies
        # itself (those calls are only made for the NEW node each step) -- so a
        # caller passing in a node with no cached gradient/energy yet (e.g. a
        # freshly-built split-candidate structure, as opposed to a chain node
        # that already went through NEB/GSM) would otherwise crash immediately
        # with GradientsNotComputedError. Compute them here if missing instead
        # of assuming the caller already did.
        if last_node._cached_gradient is None:
            last_node._cached_gradient = self.compute_gradients([last_node])[0]
        if last_node._cached_energy is None:
            last_node._cached_energy = self.compute_energies([last_node])[0]

        curr_step = 0
        converged = False
        natom = node.coords.shape[0]
        while curr_step < max_steps and not converged:
            grad = np.array(last_node.gradient)
            if mass_weighted:
                masses = [get_mass(s) for s in node.structure.symbols]
                grad = np.array([atom*np.sqrt(mass)
                                for atom, mass in zip(grad, masses)])

            grad_mag = np.linalg.norm(grad) / np.sqrt(natom)
            # print(f"Step {curr_step}: Gradient magnitude {grad_mag:.4e}")
            if grad_mag > ss:
                logging.getLogger(__name__).debug(
                    "Step %s: gradient magnitude %.4e greater than step size %.4e; scaling step.",
                    curr_step,
                    grad_mag,
                    ss,
                )
                # normalize the gradient

                grad = grad / np.linalg.norm(grad)
                grad = grad / np.sqrt(natom)
                grad = grad*ss

            new_coords = last_node.coords - ((1.0 * ss) * grad)
            node_new = last_node.update_coords(new_coords)
            grads = self.compute_gradients([node_new])
            ene = self.compute_energies([node_new])
            node_new._cached_gradient = grads[0]
            node_new._cached_energy = ene[0]

            history.append(node_new)

            delta_en = node_new.energy - last_node.energy
            grad_inf_norm = np.amax(np.abs(node_new.gradient))
            converged = delta_en <= ene_thre and grad_inf_norm <= grad_thre

            last_node = node_new.copy()
            curr_step += 1

        return history
