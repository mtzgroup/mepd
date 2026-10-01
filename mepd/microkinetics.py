"""Microkinetics of a reaction network with steps of any order.

Species carry an energy and steps a transition-state energy, all on one
scale (kcal/mol here; the caller converts). Every step is reversible, with
mass-action rates from transition-state theory:

    k_f = (k_B T / h) exp(-(G_top - sum G_reactants) / RT)   [M^(1-n) s^-1]
    k_r = (k_B T / h) exp(-(G_top - sum G_products) / RT)

where G_top = max(G_TS, sum G_reactants, sum G_products): a TS computed
below one of its sides counts as level with it, so forward and reverse
rates always obey detailed balance with the same TS (1 M standard state).
A species on both sides of a step (a shuttle: water relaying a proton)
enters its rate and is not consumed.

What it computes:
  * amounts over time from an initial composition, in a closed flask
    ("batch") or with some species held fixed ("constant feed": a
    chemostat, whose long-time limit is a true steady state);
  * net flux and the integrated extent of every step (where material went);
  * the degree of rate control of every TS and the thermodynamic degree of
    control of every species for a chosen quantity -- the formation rate of
    a target, or its amount at the end -- X_i = d ln Q / d(-G_i / RT)
    (Campbell): which barriers to lower, which intermediates to stabilize;
  * temperature sweeps: the same at several temperatures, and the apparent
    activation energy of the target's formation.

Energies are taken as given: electronic energies stand in for free
energies unless free energies are supplied, so absolute rates (bimolecular
ones especially, which lack the entropy cost of association) are orders of
magnitude; ratios, controlling steps and trends are what to read.

References: degree of rate control, C. T. Campbell, J. Catal. 204, 520
(2001), doi:10.1006/jcat.2001.3396, and ACS Catal. 7, 2770 (2017),
doi:10.1021/acscatal.7b00115; thermodynamic degree of rate control,
C. Stegelmann, A. Andreasen, C. T. Campbell, J. Am. Chem. Soc. 131, 8077
(2009), doi:10.1021/ja9000097.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.constants import Boltzmann, Planck, gas_constant

KCAL = 4184.0


def rt_kcal(temperature: float) -> float:
    return gas_constant * temperature / KCAL


@dataclass
class Step:
    reactants: list          # species indices (repeated for 2 A)
    products: list
    g_ts: float              # kcal/mol, the species' scale
    label: str = ""
    info: dict = field(default_factory=dict)


@dataclass
class Network:
    names: list              # species names
    g: np.ndarray            # kcal/mol per species
    steps: list

    def __post_init__(self):
        self.g = np.asarray(self.g, dtype=float)
        n, m = len(self.names), len(self.steps)
        self.nu = np.zeros((n, m))                 # net stoichiometry
        for j, st in enumerate(self.steps):
            for i in st.reactants:
                self.nu[i, j] -= 1
            for i in st.products:
                self.nu[i, j] += 1

    def rate_constants(self, temperature: float, g: Optional[np.ndarray] = None,
                       g_ts: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        g = self.g if g is None else np.asarray(g)
        g_ts = np.array([st.g_ts for st in self.steps]) if g_ts is None else np.asarray(g_ts)
        rt = rt_kcal(temperature)
        pre = Boltzmann * temperature / Planck
        kf, kr = np.zeros(len(self.steps)), np.zeros(len(self.steps))
        for j, st in enumerate(self.steps):
            gr, gp = g[st.reactants].sum(), g[st.products].sum()
            top = max(g_ts[j], gr, gp)
            kf[j] = pre * math.exp(-(top - gr) / rt)
            kr[j] = pre * math.exp(-(top - gp) / rt)
        return kf, kr

    def barriers(self) -> list[tuple[float, float]]:
        """(forward, reverse) barriers as computed (may be negative)."""
        out = []
        for st in self.steps:
            out.append((st.g_ts - self.g[st.reactants].sum(), st.g_ts - self.g[st.products].sum()))
        return out


@dataclass
class Result:
    times: np.ndarray            # s
    c: np.ndarray                # M, [time, species]
    net_rate: np.ndarray         # M/s per step at the end (forward - reverse)
    extent: np.ndarray           # M per step, integrated net over the run
    steady: bool                 # all non-held species stopped changing
    temperature: float


def _rates(c, kf, kr, steps):
    rf = kf.copy()
    rr = kr.copy()
    for j, st in enumerate(steps):
        for i in st.reactants:
            rf[j] *= c[i]
        for i in st.products:
            rr[j] *= c[i]
    return rf, rr


def simulate(net: Network, c0: Sequence[float], temperature: float, time_s: float, *,
             held: Sequence[int] = (), g: Optional[np.ndarray] = None, g_ts: Optional[np.ndarray] = None,
             n_points: int = 80, rtol: float = 1e-8, atol: float = 1e-14) -> Result:
    """Amounts from `c0` (M) over `time_s` (log-spaced output). `held`
    species stay at their initial amount (a constant feed)."""
    from scipy.integrate import solve_ivp

    n, m = len(net.names), len(net.steps)
    kf, kr = net.rate_constants(temperature, g, g_ts)
    free = np.ones(n)
    free[list(held)] = 0.0
    steps = net.steps

    def f(_t, y):
        c = np.maximum(y[:n], 0.0)
        rf, rr = _rates(c, kf, kr, steps)
        r = rf - rr
        return np.concatenate([(net.nu @ r) * free, r])

    def jac(_t, y):
        c = np.maximum(y[:n], 0.0)
        dr = np.zeros((m, n))            # d(net rate_j)/d c_i
        for j, st in enumerate(steps):
            for side, k, sign in ((st.reactants, kf[j], 1.0), (st.products, kr[j], -1.0)):
                for pos, i in enumerate(side):
                    prod = k
                    for q, i2 in enumerate(side):
                        if q != pos:
                            prod *= c[i2]
                    dr[j, i] += sign * prod
        top = (net.nu @ dr) * free[:, None]
        J = np.zeros((n + m, n + m))
        J[:n, :n] = top
        J[n:, :n] = dr
        return J

    y0 = np.concatenate([np.asarray(c0, dtype=float), np.zeros(m)])
    t_first = min(1e-12, time_s * 1e-6)
    t_eval = np.concatenate([[0.0], np.logspace(math.log10(t_first), math.log10(time_s), n_points - 1)])
    sol = solve_ivp(f, (0.0, time_s), y0, method="BDF", t_eval=t_eval, jac=jac, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"the kinetics did not integrate: {sol.message}")
    c = np.maximum(sol.y[:n].T, 0.0)
    rf, rr = _rates(c[-1], kf, kr, steps)
    dcdt = (net.nu @ (rf - rr)) * free
    scale = np.maximum(c[-1], 1e-12)
    steady = bool(np.all(np.abs(dcdt) * time_s < 1e-3 * scale + 1e-12))
    return Result(sol.t, c, rf - rr, sol.y[n:, -1], steady, temperature)


def formation_rate(net: Network, res: Result, target: int) -> float:
    """Net rate (M/s) at which the target forms at the end of the run."""
    return float(net.nu[target] @ res.net_rate)


def _quantity(net, res, target, what):
    if what == "rate":
        return formation_rate(net, res, target)
    return float(res.c[-1, target])


def degree_of_control(net: Network, c0, temperature: float, time_s: float, target: int, *,
                      what: str = "amount", held: Sequence[int] = (), delta_kcal: float = 0.2) -> dict:
    """Degree of rate control of each TS and thermodynamic degree of control
    of each species for the target's `what` ("amount" at the end, or
    formation "rate"): X = d ln Q / d(-G/RT), by central differences of
    +/- delta. X_TS > 0: lowering that TS raises Q. X_species < 0:
    stabilizing that species lowers Q (it is a trap or a sink)."""
    rt = rt_kcal(temperature)
    g_ts0 = np.array([st.g_ts for st in net.steps])

    def q(g=None, g_ts=None):
        v = _quantity(net, simulate(net, c0, temperature, time_s, held=held, g=g, g_ts=g_ts, n_points=8), target,
                      what)
        return v

    base = q()
    out = {"base": base, "what": what, "steps": [], "species": []}
    if not (base > 0 and math.isfinite(base)):
        return out
    for j in range(len(net.steps)):
        lo, hi = g_ts0.copy(), g_ts0.copy()
        lo[j] -= delta_kcal
        hi[j] += delta_kcal
        a, b = q(g_ts=lo), q(g_ts=hi)
        x = (math.log(a) - math.log(b)) / (2 * delta_kcal / rt) if a > 0 and b > 0 else float("nan")
        out["steps"].append(x)
    for i in range(len(net.names)):
        lo, hi = net.g.copy(), net.g.copy()
        lo[i] -= delta_kcal
        hi[i] += delta_kcal
        a, b = q(g=lo), q(g=hi)
        x = (math.log(a) - math.log(b)) / (2 * delta_kcal / rt) if a > 0 and b > 0 else float("nan")
        out["species"].append(x)
    return out


def sweep(net: Network, c0, temperatures: Sequence[float], time_s: float, target: Optional[int] = None, *,
          held: Sequence[int] = ()) -> dict:
    """The run at each temperature: final amounts, the target's formation
    rate, and its apparent activation energy (kcal/mol, from d ln r / dT)."""
    finals, rates = [], []
    for t in temperatures:
        res = simulate(net, c0, t, time_s, held=held, n_points=8)
        finals.append(res.c[-1].tolist())
        rates.append(formation_rate(net, res, target) if target is not None else None)
    ea = []
    for k in range(len(temperatures)):
        lo, hi = max(0, k - 1), min(len(temperatures) - 1, k + 1)
        r1, r2, t1, t2 = rates[lo], rates[hi], temperatures[lo], temperatures[hi]
        if target is None or lo == hi or not (r1 and r2 and r1 > 0 and r2 > 0):
            ea.append(None)
            continue
        # Arrhenius: ln r = -Ea/(R T) + const
        ea.append(-(math.log(r2) - math.log(r1)) / (1 / t2 - 1 / t1) * gas_constant / KCAL)
    return {"temperatures": list(temperatures), "final": finals, "rate": rates, "apparent_ea_kcal": ea}
