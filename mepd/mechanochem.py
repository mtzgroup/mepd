"""Mechanical force on existing transition states: which atoms to pull on
to favor or disfavor a reaction channel, and how hard.

A constant force F pulling atoms i and j apart adds -F * d_ij to the
energy (the force-modified surface; EFEI). To first order in F the
stationary points stay where they are, so a barrier changes by

    dE++(F) = dE++(0) - F * dq++,   dq++ = d_ij(TS) - d_ij(reactant)

(Bell's model): a pair the TS stretches is a lever that speeds the
reaction up, a pair it compresses one that holds it back. 1 nN * 1 Å =
14.39 kcal/mol, so a pair stretched by 0.5 Å at the TS gives ~7 kcal/mol
per nN. Across several channels from one reactant each channel has its own
dq++, and a pair whose dq++ favors a slower channel can make it win above a
crossover force.

The first-order estimate ignores that the reactant and the TS deform
under force (the extended Bell theory adds that compliance term, from
Hessians) and that a TS can vanish at high force. `ForcedEngine` computes
the force-modified surface itself, so TSs, IRCs and minima can be
re-optimized under force to check it (`mepd force --mode reoptimize`).

In an experiment the force enters through handles: polymer chains or a
tether attached at the pulled atoms (sonication, AFM, a stiff-stilbene
photoswitch). Pairs are ranked for any heavy atoms; which ones can carry a
handle is the chemist's call.

References: G. I. Bell, Science 200, 618 (1978), doi:10.1126/science.347575;
EFEI, J. Ribas-Arino, M. Shiga, D. Marx, Angew. Chem. Int. Ed. 48, 4190
(2009), doi:10.1002/anie.200900673; force-modified PES, M. T. Ong et al.,
J. Am. Chem. Soc. 131, 6377 (2009), doi:10.1021/ja8095834; COGEF, M. K.
Beyer, J. Chem. Phys. 112, 7307 (2000), doi:10.1063/1.481330; extended Bell
theory, S. S. M. Konda et al., J. Chem. Phys. 135, 164103 (2011),
doi:10.1063/1.3656367.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from mepd.engines.engine import Engine
from mepd.engines.modified import ModifiedEngine

# 1 nN * 1 Å = 1e-19 J per molecule, in kcal/mol
NN_ANGSTROM_KCAL = 1e-19 * 6.02214076e23 / 4184.0
# 1 nN in Hartree/Bohr (atomic unit of force: 8.2387235e-8 N)
NN_TO_HARTREE_PER_BOHR = 1e-9 / 8.2387235e-8
BOHR_TO_ANGSTROM = 0.529177210903
# Bell's first-order picture is trusted to about this force (nN); beyond,
# re-optimize under force.
BELL_RANGE_NN = 1.0
# Forces are scanned up to this for crossovers (single-molecule experiments
# and polymer mechanochemistry reach a few nN; covalent bonds fail at ~4-6).
MAX_FORCE_NN = 2.5


def atom_label(symbols, i: int) -> str:
    return f"{symbols[i]}{i}"


def pair_label(symbols, pair) -> str:
    i, j = pair
    return f"{atom_label(symbols, i)}–{atom_label(symbols, j)}"


def distances(coords_bohr: NDArray, pairs) -> NDArray:
    c = np.asarray(coords_bohr, dtype=float)
    return np.array([np.linalg.norm(c[i] - c[j]) for i, j in pairs]) * BOHR_TO_ANGSTROM


def candidate_pairs(symbols, hydrogens: bool = False) -> list[tuple[int, int]]:
    idx = [i for i, s in enumerate(symbols) if hydrogens or s != "H"]
    return [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]]


def bell_barrier(barrier: float, dq: float, force: float) -> float:
    return barrier - NN_ANGSTROM_KCAL * force * dq


def force_for_barrier(barrier: float, dq: float, target: float) -> Optional[float]:
    """Pulling force (nN) that brings `barrier` down to `target` (Bell)."""
    if dq <= 1e-6 or barrier <= target:
        return None
    return (barrier - target) / (NN_ANGSTROM_KCAL * dq)


@dataclass
class Channel:
    """One reaction channel: its barrier and the geometries (Bohr) of its
    reactant end, TS and product end, all with the same atom order."""
    id: str
    label: str
    barrier_kcal: float
    reactant: NDArray
    ts: NDArray
    product: Optional[NDArray] = None
    kind: str = "direct"    # "direct" (start -> end) | "offtarget" (leads elsewhere)


def analyze(symbols, channels: list[Channel], *, pairs=None, hydrogens: bool = False, top: int = 5,
            temperature: float = 298.15, max_force: float = MAX_FORCE_NN) -> dict:
    """Bell analysis of every candidate pair over every channel.

    Returns {"pairs": [...], "channels": [...], "selectivity": [...],
    "insights": [...], "warnings": [...]}; each channel gets its best
    levers ("favor": largest dq++ > 0, "disfavor": most negative) and the
    force that brings its half-life to 1 h at `temperature`."""
    from mepd.conditions import format_duration, half_life

    pairs = [tuple(p) for p in (pairs or candidate_pairs(symbols, hydrogens))]
    if not pairs or not channels:
        return {"pairs": [], "channels": [], "selectivity": [], "insights": [], "warnings": []}
    dq = {c.id: distances(c.ts, pairs) - distances(c.reactant, pairs) for c in channels}
    d_ts = {c.id: distances(c.ts, pairs) for c in channels}
    bench = _bench_barrier(temperature)
    rows = []
    for c in channels:
        order = np.argsort(-dq[c.id])
        favor = [_lever(symbols, pairs[k], dq[c.id][k], c.barrier_kcal, bench) for k in order[:top] if dq[c.id][k] > 0.05]
        disfavor = [_lever(symbols, pairs[k], dq[c.id][k], c.barrier_kcal, bench)
                    for k in order[::-1][:top] if dq[c.id][k] < -0.05]
        rows.append({"id": c.id, "label": c.label, "kind": c.kind, "barrier_kcal": c.barrier_kcal, "favor": favor,
                     "disfavor": disfavor,
                     "t_half": format_duration(half_life(max(c.barrier_kcal, 0.0), temperature))})
    # Every pair that is a lever for some channel, with its dq++ everywhere.
    keep = sorted({tuple(x["pair"]) for r in rows for x in r["favor"] + r["disfavor"]})
    index = {p: k for k, p in enumerate(pairs)}
    table = [{"pair": list(p), "label": pair_label(symbols, p),
              "dq": {c.id: float(dq[c.id][index[p]]) for c in channels},
              "d_ts": {c.id: float(d_ts[c.id][index[p]]) for c in channels}} for p in keep]
    selectivity = _crossovers(symbols, channels, pairs, dq, max_force) if len(channels) > 1 else []
    insights = _insights(symbols, rows, selectivity, temperature)
    warnings = list(WARNINGS)
    return {"pairs": table, "channels": rows, "selectivity": selectivity, "insights": insights,
            "warnings": warnings, "max_force": max_force}


def _bench_barrier(temperature: float) -> float:
    """The barrier with a 1 h half-life at `temperature` (kcal/mol)."""
    from mepd.conditions import BENCH_HALF_LIFE_S, half_life

    lo, hi = 0.0, 80.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if half_life(mid, temperature) < BENCH_HALF_LIFE_S:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _lever(symbols, pair, dq: float, barrier: float, bench: float) -> dict:
    return {"pair": list(pair), "label": pair_label(symbols, pair), "dq": float(dq),
            "per_nN_kcal": float(NN_ANGSTROM_KCAL * dq),
            "barrier_at_1nN": float(bell_barrier(barrier, dq, 1.0)),
            "force_for_1h": force_for_barrier(barrier, dq, bench)}


def _crossovers(symbols, channels: list[Channel], pairs, dq: dict, max_force: float) -> list[dict]:
    """For every channel that is not the fastest: the pair and the smallest
    force (<= max_force) at which it becomes the lowest barrier (Bell)."""
    base = {c.id: c.barrier_kcal for c in channels}
    leader = min(channels, key=lambda c: c.barrier_kcal)
    forces = np.linspace(0.0, max_force, 251)
    out = []
    for c in channels:
        if c.id == leader.id:
            continue
        best = None
        for k, p in enumerate(pairs):
            if dq[c.id][k] <= 0:
                continue
            e = {o.id: base[o.id] - NN_ANGSTROM_KCAL * forces * dq[o.id][k] for o in channels}
            others = np.min(np.array([e[o.id] for o in channels if o.id != c.id]), axis=0)
            wins = np.nonzero(e[c.id] < others - 0.5)[0]     # a clear 0.5 kcal/mol margin
            if len(wins) and (best is None or forces[wins[0]] < best["force_nN"]):
                f = float(forces[wins[0]])
                best = {"channel": c.id, "label": c.label, "pair": list(p), "pair_label": pair_label(symbols, p),
                        "force_nN": f, "barrier_then": float(e[c.id][wins[0]]),
                        "overtakes": leader.label, "leader_then": float(e[leader.id][wins[0]]),
                        "dq_self": float(dq[c.id][k]), "dq_leader": float(dq[leader.id][k]),
                        "gap_kcal": float(base[c.id] - base[leader.id])}
        if best is not None:
            out.append(best)
    return sorted(out, key=lambda x: x["force_nN"])


WARNINGS = (f"Bell estimate (ΔE‡ − F·Δq‡, geometries fixed): reliable to ~{BELL_RANGE_NN:g} nN; beyond that, "
            "or to confirm, re-optimize under force.",
            "Pairs are ranked over all heavy atoms; experimentally, pick pairs that can carry a handle "
            "(polymer chain, tether).")


def _insights(symbols, rows: list[dict], selectivity: list[dict], temperature: float) -> list[dict]:
    out = []
    t_c = temperature - 273.15
    lead = min(rows, key=lambda r: r["barrier_kcal"])
    for r in rows[:1] if len(rows) == 1 else [lead]:
        if r["favor"]:
            f = r["favor"][0]
            text = (f"Best lever: pull {f['label']} apart (−{f['per_nN_kcal']:.0f} kcal/mol per nN; "
                    f"{r['barrier_kcal']:.1f} → {f['barrier_at_1nN']:.1f} at 1 nN)")
            if f["force_for_1h"] is not None and f["force_for_1h"] <= MAX_FORCE_NN:
                text += f". 1 h t½ at {t_c:.0f} °C: ~{f['force_for_1h']:.1f} nN"
            elif f["force_for_1h"] is not None:
                text += (f". 1 h t½ at {t_c:.0f} °C needs ~{f['force_for_1h']:.0f} nN: out of reach alone "
                         "(near bond rupture)")
            out.append({"level": "accelerates", "text": text, "pair": f["pair"], "channel": r["id"]})
        if r["disfavor"]:
            f = r["disfavor"][0]
            out.append({"level": "slows", "pair": f["pair"], "channel": r["id"], "text":
                        f"Pull {f['label']} apart to hold it back (+{-f['per_nN_kcal']:.0f} kcal/mol per nN)"})
    kinds = {r["id"]: r.get("kind") for r in rows}
    if kinds.get(lead["id"]) == "offtarget":
        steer = [s for s in selectivity if kinds.get(s["channel"]) == "direct"]
        if steer:
            s = steer[0]
            out.append({"level": "selectivity", "pair": s["pair"], "channel": s["channel"], "text":
                        f"Without force {lead['label']} (off-target) is fastest. Pulling {s['pair_label']} apart "
                        f"with ≥ {s['force_nN']:.1f} nN lets the intended {s['label']} win "
                        f"({s['barrier_then']:.1f} vs {s['leader_then']:.1f}, Bell)"})
            selectivity = [x for x in selectivity if x is not s]
    for s in selectivity[:3]:
        out.append({"level": "selectivity", "pair": s["pair"], "channel": s["channel"], "text":
                    f"{s['label']} overtakes {s['overtakes']} (+{s['gap_kcal']:.1f} without force) at ≥ "
                    f"{s['force_nN']:.1f} nN on {s['pair_label']}: {s['barrier_then']:.1f} vs {s['leader_then']:.1f} "
                    "(Bell)" + (f", mostly by holding {s['overtakes']} back" if s["dq_self"] < -s["dq_leader"] else "")})
    if len(rows) > 1 and not selectivity:
        out.append({"level": "note", "text": f"Selectivity is robust: no pair lets another channel beat "
                    f"{lead['label']} below {MAX_FORCE_NN:g} nN (Bell)", "pair": None,
                    "channel": lead["id"]})
    return out


@dataclass
class ForcedEngine(ModifiedEngine):
    """`base` with a constant force pulling each pair apart:
    E_F = E - F * sum(d_ij) (EFEI). Negative force pushes them together."""

    base: Engine
    pairs: list = field(default_factory=list)
    force_nN: float = 0.0

    def __post_init__(self):
        self.pairs = [tuple(int(x) for x in p) for p in self.pairs]
        self._inherit()

    @property
    def label(self) -> str:
        return f"{self.force_nN:g} nN on {', '.join(f'{i}–{j}' for i, j in self.pairs)}"

    def _term(self, nodes):
        f = self.force_nN * NN_TO_HARTREE_PER_BOHR
        de = np.zeros(len(nodes))
        dgrad = []
        for k, n in enumerate(nodes):
            c = np.asarray(n.coords, dtype=float)
            g = np.zeros_like(c)
            for i, j in self.pairs:
                r = c[i] - c[j]
                d = float(np.linalg.norm(r))
                de[k] -= f * d
                if d > 0:
                    g[i] -= f * r / d
                    g[j] += f * r / d
            dgrad.append(g)
        return de, np.array(dgrad)


def force_term_kcal(coords_bohr: NDArray, pairs, force_nN: float) -> float:
    """-F * sum(d_ij) in kcal/mol (the force's own part of E_F)."""
    return -NN_ANGSTROM_KCAL * force_nN * float(np.sum(distances(coords_bohr, pairs)))

