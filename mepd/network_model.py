"""Whole-network properties of a reaction network, and how sensitive they are
to the energies of its parts.

A `NetworkModel` is species (free-energy stand-ins G_i, kcal/mol) joined by
steps (a TS energy G_TS on the same scale), all first order, plus the
provenance of every number: which job, level of theory, solvent, variant.
From it:

* kinetics: amounts after a reaction time, conversion, selectivity;
* equilibrium: Boltzmann populations of the species the start can reach,
  and whether the outcome at that time is kinetically or thermodynamically
  controlled (how far it is from equilibrium);
* timescales: relaxation times of the network (eigenvalues of its rate
  matrix); the slowest says when it has equilibrated;
* formation time of each product: when it first reaches half of its peak
  (a mean first-passage time exists too, but rare detours into deep traps
  dominate it, so it is not what an experiment sees);
* bottleneck paths: to each product, the route whose highest TS is lowest,
  and that effective barrier;
* sensitivities: for any of the above (as a scalar P), the generalized
  degree of rate control of every TS, X_i = d ln P / d(-G_TS,i / RT), and
  of every species (the thermodynamic analogue) -- which atomic-scale
  numbers the network-scale property actually depends on;
* comparison: two models of one network (another solvent, level of
  theory, temperature, substituent...) -- each property's change, and to
  first order how much of it each step's energy change explains.

Rates are Eyring's from the energies given (electronic barriers stand in
for free energies unless free energies are supplied), so absolute rates
are orders of magnitude; ratios, sensitivities and trends are more robust.

References: degree of rate control, C. T. Campbell, ACS Catal. 7, 2770
(2017), doi:10.1021/acscatal.7b00115 (and J. Catal. 204, 520 (2001),
doi:10.1006/jcat.2001.3396); mean first-passage times of a Markov chain,
e.g. D. J. Wales, Int. Rev. Phys. Chem. 25, 237 (2006),
doi:10.1080/01442350600676921.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.constants import Boltzmann, Planck, gas_constant

KCAL = 4184.0


# Amounts / rates below this are "does not happen": no sensitivities of them.
TINY = 1e-12


def _finite(x: float) -> Optional[float]:
    return None if not math.isfinite(x) else float(x)


def rt_kcal(temperature: float) -> float:
    return gas_constant * temperature / KCAL


@dataclass
class Species:
    id: str
    name: str
    G: float                      # kcal/mol, on the model's one scale
    provenance: dict = field(default_factory=dict)


@dataclass
class Step:
    id: str
    a: str
    b: str
    G_ts: float                   # kcal/mol, same scale as the species
    provenance: dict = field(default_factory=dict)


@dataclass
class NetworkModel:
    species: list[Species]
    steps: list[Step]
    temperature: float = 298.15
    time_s: float = 3600.0
    start: dict = field(default_factory=dict)   # species id -> initial amount
    warnings: list = field(default_factory=list)
    spec: dict = field(default_factory=dict)    # what the model was built from (setup, level, ...)

    # ---- basic pieces ------------------------------------------------------
    @property
    def index(self) -> dict:
        return {s.id: i for i, s in enumerate(self.species)}

    def barriers(self) -> list[tuple[float, float]]:
        """(forward, reverse) raw barriers per step, kcal/mol (may be negative)."""
        G = {s.id: s.G for s in self.species}
        return [(st.G_ts - G[st.a], st.G_ts - G[st.b]) for st in self.steps]

    def rate_matrix(self, G_species=None, G_ts=None) -> np.ndarray:
        """K with dc/dt = K c; a TS below an end counts as level with it
        (only inside the rates -- raw barriers are reported as they are)."""
        n = len(self.species)
        idx = self.index
        Gs = np.array([s.G for s in self.species]) if G_species is None else np.asarray(G_species)
        Gt = np.array([st.G_ts for st in self.steps]) if G_ts is None else np.asarray(G_ts)
        rt = rt_kcal(self.temperature)
        pre = Boltzmann * self.temperature / Planck
        K = np.zeros((n, n))
        for st, gt in zip(self.steps, Gt):
            a, b = idx[st.a], idx[st.b]
            if a == b:
                continue
            top = max(gt, Gs[a], Gs[b])
            kf, kr = pre * math.exp(-(top - Gs[a]) / rt), pre * math.exp(-(top - Gs[b]) / rt)
            K[b, a] += kf
            K[a, a] -= kf
            K[a, b] += kr
            K[b, b] -= kr
        return K

    def c0(self) -> np.ndarray:
        c = np.zeros(len(self.species))
        idx = self.index
        total = sum(self.start.values()) or 1.0
        for sid, amount in self.start.items():
            if sid in idx:
                c[idx[sid]] += amount / total
        return c

    def reachable(self) -> set:
        adj: dict = {}
        for st in self.steps:
            adj.setdefault(st.a, set()).add(st.b)
            adj.setdefault(st.b, set()).add(st.a)
        seen, todo = set(self.start), list(self.start)
        while todo:
            x = todo.pop()
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        return seen

    # ---- properties ----------------------------------------------------------
    def amounts_at(self, t: float, K=None) -> np.ndarray:
        """Amounts at time t, exact for first order and stable at any t: the
        rates obey detailed balance (they come from energies), so with
        D = diag(sqrt(p_eq)) the matrix D^-1 K D is symmetric and
        c(t) = D V exp(L t) V^T D^-1 c0."""
        K = self.rate_matrix() if K is None else K
        D, V, L, sel = self._eig(K)
        c0 = self.c0()
        out = np.zeros_like(c0)
        if len(sel):
            y = V.T @ (c0[sel] / D)
            out[sel] = D * (V @ (np.exp(np.minimum(L * t, 0.0)) * y))
        return np.clip(out, 0.0, None)

    def _eig(self, K):
        key = id(K)
        cache = self.__dict__.setdefault("_eig_cache", {})
        if key in cache and cache[key][0] is K:
            return cache[key][1]
        reach = self.reachable()
        sel = np.array([i for i, sp in enumerate(self.species) if sp.id in reach], dtype=int)
        G = np.array([self.species[i].G for i in sel])
        # sqrt of Boltzmann weights, relative to the lowest (log space: no underflow for sane spans)
        D = np.exp(-(G - G.min()) / (2 * rt_kcal(self.temperature)))
        Ks = K[np.ix_(sel, sel)]
        S = Ks * D[None, :] / D[:, None]
        S = 0.5 * (S + S.T)
        L, V = np.linalg.eigh(S)
        res = (D, V, L, sel)
        cache.clear()
        cache[key] = (K, res)
        return res

    def equilibrium(self, G_species=None) -> np.ndarray:
        Gs = np.array([s.G for s in self.species]) if G_species is None else np.asarray(G_species)
        reach = self.reachable()
        mask = np.array([s.id in reach for s in self.species])
        p = np.zeros(len(self.species))
        if mask.any():
            g = Gs[mask]
            w = np.exp(-(g - g.min()) / rt_kcal(self.temperature))
            p[mask] = w / w.sum()
        return p

    def timescales(self, K=None, n: int = 3) -> list[float]:
        """The slowest relaxation times (s) of the part the start reaches."""
        K = self.rate_matrix() if K is None else K
        reach = self.reachable()
        sel = [i for i, s in enumerate(self.species) if s.id in reach]
        if len(sel) < 2:
            return []
        L = self._eig(K)[2]
        rates = sorted(-L[L < -1e-300])
        return [1.0 / r for r in rates[:n]]

    def mfpt(self, target: str, K=None) -> Optional[float]:
        """Mean first-passage time (s) from the start to `target`."""
        K = self.rate_matrix() if K is None else K
        idx = self.index
        if target not in idx or target not in self.reachable():
            return None
        Q = K.T                                   # CTMC generator, rows sum to 0
        rest = [i for i in range(len(self.species)) if i != idx[target] and self.species[i].id in self.reachable()]
        if not rest:
            return 0.0
        try:
            m = np.linalg.solve(Q[np.ix_(rest, rest)], -np.ones(len(rest)))
        except np.linalg.LinAlgError:
            return None
        c = self.c0()[rest]
        return float(np.dot(c / max(c.sum(), 1e-300), m)) if c.sum() > 0 else 0.0

    def formation_time(self, target: str, K=None) -> Optional[float]:
        """When `target` first reaches half of its peak amount (s), the peak
        taken over the network's own timescales. Unlike the mean first-
        passage time, it is not dominated by rare detours into deep traps."""
        K = self.rate_matrix() if K is None else K
        i = self.index.get(target)
        if i is None or target not in self.reachable():
            return None
        slow = self.timescales(K, n=1)
        t_max = min(1e30, 10 * slow[0]) if slow else 1.0
        grid = np.geomspace(1e-13, t_max, 160)
        amounts = np.array([self.amounts_at(t, K)[i] for t in grid])
        k = int(np.argmax(amounts))
        peak = amounts[k]
        if peak < 1e-6:
            return None
        half = 0.5 * peak
        lo_i = int(np.nonzero(amounts[: k + 1] >= half)[0][0])
        if lo_i == 0:
            return float(grid[0])
        lo, hi = math.log(grid[lo_i - 1]), math.log(grid[lo_i])
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.amounts_at(math.exp(mid), K)[i] >= half:
                hi = mid
            else:
                lo = mid
        return float(math.exp(0.5 * (lo + hi)))

    def bottleneck(self, target: str) -> Optional[dict]:
        """The start -> target route whose highest TS is lowest."""
        G = {s.id: s.G for s in self.species}
        adj: dict = {}
        for st in self.steps:
            adj.setdefault(st.a, []).append((st.b, st))
            adj.setdefault(st.b, []).append((st.a, st))
        best: dict = {}
        heap = []
        for s in self.start:
            best[s] = G.get(s, 0.0)
            heapq.heappush(heap, (best[s], s, []))
        while heap:
            top, x, path = heapq.heappop(heap)
            if x == target:
                ref = min(G[s] for s in self.start if s in G)
                worst = max(path, key=lambda st: st.G_ts) if path else None
                return {"effective_barrier": top - ref, "steps": [st.id for st in path],
                        "bottleneck_step": worst.id if worst else None}
            if top > best.get(x, math.inf):
                continue
            for y, st in adj.get(x, []):
                t = max(top, st.G_ts, G[y])
                if t < best.get(y, math.inf) - 1e-12:
                    best[y] = t
                    heapq.heappush(heap, (t, y, path + [st]))
        return None

    # ---- the report -----------------------------------------------------------
    def products(self, final: np.ndarray, eq: np.ndarray, n: int = 4) -> list[str]:
        """Species other than the start that matter: largest at the end or at equilibrium."""
        score = np.maximum(final, eq)
        order = [i for i in np.argsort(-score) if self.species[i].id not in self.start and score[i] > 1e-3]
        return [self.species[i].id for i in order[:n]]

    def properties(self, products: Optional[list[str]] = None) -> dict:
        K = self.rate_matrix()
        final = self.amounts_at(self.time_s, K)
        eq = self.equilibrium()
        products = products or self.products(final, eq)
        idx = self.index
        tv = 0.5 * float(np.abs(final - eq).sum())
        conv = 1.0 - float(sum(final[idx[s]] for s in self.start if s in idx))
        out = {
            "final": {s.id: float(final[i]) for i, s in enumerate(self.species)},
            "equilibrium": {s.id: float(eq[i]) for i, s in enumerate(self.species)},
            "conversion": conv,
            "distance_from_equilibrium": tv,
            "control": "thermodynamic" if tv < 0.1 else "kinetic",
            "timescales_s": self.timescales(K),
            "products": products,
            "formation_time_s": {p: self.formation_time(p, K) for p in products},
            "bottleneck": {p: self.bottleneck(p) for p in products},
        }
        if len(products) >= 2 and final[idx[products[1]]] > 0:
            out["selectivity"] = {"of": products[0], "over": products[1],
                                  "ratio": float(final[idx[products[0]]] / final[idx[products[1]]])}
        return out

    # ---- sensitivities ----------------------------------------------------------
    def sensitivity(self, quantity: Callable[["NetworkModel"], float], delta: float = 0.1) -> dict:
        """Degree of control of every TS and every species over a positive
        scalar `quantity(model)`: X = d ln P / d(-G/RT), central differences
        of +-delta kcal/mol. X_TS > 0: lowering that TS raises P."""
        rt = rt_kcal(self.temperature)

        def at(Gs=None, Gt=None) -> float:
            m = self.copy_with(Gs, Gt)
            v = quantity(m)
            # A vanishing quantity (a product that never forms) has no
            # meaningful log-derivative: nan, reported as undefined.
            return math.log(v) if v > TINY else math.nan

        Gs0 = np.array([s.G for s in self.species])
        Gt0 = np.array([st.G_ts for st in self.steps])
        ts, sp = {}, {}
        for k, st in enumerate(self.steps):
            up, dn = Gt0.copy(), Gt0.copy()
            up[k] += delta
            dn[k] -= delta
            ts[st.id] = _finite((at(Gt=dn) - at(Gt=up)) * rt / (2 * delta))
        for k, s in enumerate(self.species):
            up, dn = Gs0.copy(), Gs0.copy()
            up[k] += delta
            dn[k] -= delta
            sp[s.id] = _finite((at(Gs=dn) - at(Gs=up)) * rt / (2 * delta))
        return {"ts": ts, "species": sp}

    def copy_with(self, G_species=None, G_ts=None) -> "NetworkModel":
        sp = [Species(s.id, s.name, float(g), s.provenance) for s, g in
              zip(self.species, G_species if G_species is not None else [s.G for s in self.species])]
        st = [Step(x.id, x.a, x.b, float(g), x.provenance) for x, g in
              zip(self.steps, G_ts if G_ts is not None else [x.G_ts for x in self.steps])]
        return NetworkModel(sp, st, self.temperature, self.time_s, dict(self.start), [], dict(self.spec))


# Scalar quantities for sensitivity and comparison ---------------------------
def yield_of(product: str) -> Callable[[NetworkModel], float]:
    def f(m: NetworkModel) -> float:
        return float(m.amounts_at(m.time_s)[m.index[product]])
    return f


def rate_to(product: str) -> Callable[[NetworkModel], float]:
    """1 / formation time: how fast `product` appears (see formation_time)."""
    def f(m: NetworkModel) -> float:
        t = m.formation_time(product)
        return 1.0 / t if t else 0.0
    return f


def compare(base: NetworkModel, other: NetworkModel, quantities: dict[str, Callable]) -> dict:
    """Each quantity in both models, its change, and a first-order split of
    d ln P over the steps and species whose energies changed:
    d ln P ~ sum_i X_i * (-dG_i / RT) (X from the base model)."""
    rt = rt_kcal(base.temperature)
    Gs_b = {s.id: s.G for s in base.species}
    Gt_b = {st.id: st.G_ts for st in base.steps}
    dGs = {s.id: s.G - Gs_b[s.id] for s in other.species if s.id in Gs_b}
    dGt = {st.id: st.G_ts - Gt_b[st.id] for st in other.steps if st.id in Gt_b}
    # Energies are only defined up to a constant: measure changes against
    # the start's, which is what every property is relative to.
    ref = np.mean([dGs[s] for s in base.start if s in dGs]) if base.start else 0.0
    dGs = {k: v - ref for k, v in dGs.items()}
    dGt = {k: v - ref for k, v in dGt.items()}
    out = {}
    for name, q in quantities.items():
        pb, po = q(base), q(other)
        if pb <= TINY or po <= TINY:
            # Something that (nearly) does not happen in one of them: the
            # change is "from nothing" / "to nothing", with no first-order split.
            out[name] = {"base": pb, "other": po, "dlnP": None, "dlnP_linear": None, "contributions": [],
                         "note": "negligible in " + ("both" if max(pb, po) <= TINY else "base" if pb <= TINY else "other")}
            continue
        sens = base.sensitivity(q)
        parts = [("ts", k, x * -dGt.get(k, 0.0) / rt) for k, x in sens["ts"].items() if x is not None] + \
                [("species", k, x * -dGs.get(k, 0.0) / rt) for k, x in sens["species"].items() if x is not None]
        parts = sorted((p for p in parts if abs(p[2]) > 1e-3), key=lambda p: -abs(p[2]))
        exact = math.log(po) - math.log(pb)
        linear = float(sum(p[2] for p in parts))
        # How much of the actual change the first-order split accounts for:
        # far from 1, the response is nonlinear (e.g. a yield saturated in the
        # base that an equilibrium shift then drains) and the split misleads.
        explained = linear / exact if abs(exact) > 1e-6 else (1.0 if abs(linear) < 1e-6 else None)
        out[name] = {"base": pb, "other": po, "dlnP": exact, "dlnP_linear": linear, "explained": explained,
                     "linear_ok": explained is not None and 0.5 <= explained <= 2.0,
                     "contributions": [{"kind": k, "id": i, "dlnP": float(v)} for k, i, v in parts[:8]]}
    return {"quantities": out, "dG_ts": dGt, "dG_species": dGs,
            "temperature_changed": abs(base.temperature - other.temperature) > 1e-9}
