"""Microkinetics on an expanding reaction network, to decide which species
are worth expanding next (flux-steered exploration).

Every verified elementary step A <-> B (a TS whose IRC connects A and B)
gets Eyring rate constants in both directions from the one TS energy,

    k(A->B) = (kB T / h) exp(-(E_TS - E_A) / RT),

so detailed balance holds by construction (energies are electronic, each
species at its lowest conformer found -- the barrier floor rule). Starting
from the seed at concentration 1, the linear first-order network
dc/dt = K c is integrated to `time_s`. Each proposed product counts as one
species, a pair of fragments included (as in the expansion itself), so all
steps are first order.

A species' *concentration flux* is the total amount that flowed into it
over the simulated time, integral_0^T sum_i k(i->j) c_i(t) dt. Species
whose flux reaches `threshold` are expanded in the next round; the network
stops growing once none does. This is the concentration-flux criterion of
Bensberg & Reiher (Isr. J. Chem. 63, e202200123 (2023)), the same idea as
the rate-based model enlargement of Susnow, Dean, Green & Peczak (J. Phys.
Chem. A 101, 3731 (1997)) used by RMG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.constants import Boltzmann, Planck, gas_constant

from qcconst.constants import HARTREE_TO_KCAL_PER_MOL

_KCAL_TO_J = 4184.0


@dataclass
class Step:
    """One verified elementary step between species `a` and `b`."""
    a: int
    b: int
    ts_energy: float  # Hartree
    label: str = ""
    files: dict = field(default_factory=dict)


@dataclass
class KineticsResult:
    time_s: float
    temperature: float
    flux: list[float]          # integrated inflow per species
    max_concentration: list[float]
    final_concentration: list[float]
    barriers_kcal: list[tuple[float, float]]  # (forward a->b, reverse b->a) per step


def rate_constants(steps: Sequence[Step], energies: Sequence[float], temperature: float) -> list[tuple[float, float]]:
    """(k(a->b), k(b->a)) in 1/s per step. The TS is never taken below
    either end (a TS energy below a minimum means the minimum's lowest
    conformer was found elsewhere), keeping detailed balance exact."""
    prefactor = Boltzmann * temperature / Planck
    rt = gas_constant * temperature / _KCAL_TO_J  # kcal/mol
    out = []
    for s in steps:
        ea, eb = float(energies[s.a]), float(energies[s.b])
        ts = max(float(s.ts_energy), ea, eb)
        out.append(tuple(prefactor * np.exp(-(ts - e) * HARTREE_TO_KCAL_PER_MOL / rt) for e in (ea, eb)))
    return out


def barriers(steps: Sequence[Step], energies: Sequence[float]) -> list[tuple[float, float]]:
    return [((max(s.ts_energy, energies[s.a], energies[s.b]) - energies[s.a]) * HARTREE_TO_KCAL_PER_MOL,
             (max(s.ts_energy, energies[s.a], energies[s.b]) - energies[s.b]) * HARTREE_TO_KCAL_PER_MOL)
            for s in steps]


def simulate(n_species: int, steps: Sequence[Step], energies: Sequence[float], *, temperature: float = 298.15,
             time_s: float = 3600.0, start: int = 0, n_samples: int = 80) -> KineticsResult:
    """Integrate dc/dt = K c (plus the running inflow of every species)
    from c = e_start to `time_s` with an implicit (Radau) solver -- rates
    here span many orders of magnitude."""
    from scipy.integrate import solve_ivp

    ks = rate_constants(steps, energies, temperature)
    K = np.zeros((n_species, n_species))
    for s, (kf, kr) in zip(steps, ks):
        if s.a == s.b:
            continue
        K[s.b, s.a] += kf
        K[s.a, s.a] -= kf
        K[s.a, s.b] += kr
        K[s.b, s.b] -= kr
    inflow = np.where(np.eye(n_species, dtype=bool), 0.0, K)  # k(i->j) at [j, i]
    n = n_species
    J = np.zeros((2 * n, 2 * n))
    J[:n, :n] = K
    J[n:, :n] = inflow  # d(flux_j)/dt = sum_i k(i->j) c_i
    y0 = np.zeros(2 * n)
    y0[start] = 1.0
    if not steps or not np.any(K):
        c = y0[:n]
        return KineticsResult(time_s, temperature, [0.0] * n, c.tolist(), c.tolist(), barriers(steps, energies))
    fastest = max(max(k) for k in ks)
    t_eval = np.unique(np.concatenate([[0.0], np.geomspace(min(1e-3 / fastest, time_s), time_s, n_samples)]))
    sol = solve_ivp(lambda t, y: J @ y, (0.0, time_s), y0, method="Radau", jac=J, t_eval=t_eval,
                    rtol=1e-8, atol=1e-14)
    c = np.clip(sol.y[:n], 0.0, None)
    return KineticsResult(time_s, temperature, np.clip(sol.y[n:, -1], 0.0, None).tolist(),
                          c.max(axis=1).tolist(), c[:, -1].tolist(), barriers(steps, energies))


def select_for_expansion(result: KineticsResult, expanded: set[int], threshold: float,
                         limit: Optional[int] = None) -> list[int]:
    """Species not expanded yet whose concentration flux reached
    `threshold`, most flux first."""
    picks = sorted((j for j, f in enumerate(result.flux) if j not in expanded and f >= threshold),
                   key=lambda j: -result.flux[j])
    return picks[:limit] if limit else picks
