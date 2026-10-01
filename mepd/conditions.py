"""What a computed barrier means for an experiment: rates, half-lives and
temperatures under given conditions, and plain-language insights when a
solvent (or another condition) changes the picture.

Rates are Eyring's, k = (kB T / h) exp(-dE / RT), with the electronic
barrier (plus the implicit solvation free energy, in solvent) in place of
dG++: no thermal or entropic corrections, so they are order-of-magnitude
estimates -- an activation entropy of +-10 cal/(mol K) alone moves dG++ by
+-3 kcal/mol at room temperature, a factor of ~150 in rate.

Reference: H. Eyring, J. Chem. Phys. 3, 107 (1935), doi:10.1063/1.1749604.
"""
from __future__ import annotations

import math
from typing import Optional

from scipy.constants import Boltzmann, Planck, gas_constant

_KCAL_TO_J = 4184.0
ZERO_C = 273.15

# A solvent that moves the barrier by at least this much (kcal/mol, ~160x
# at room temperature) is worth telling the user about.
NOTABLE_SHIFT = 3.0
# Single points on gas-phase geometries stop being a fair approximation
# once solvation shifts the barrier by this much: the TS geometry then
# usually moves as well.
REOPTIMIZE_SHIFT = 8.0
# One hour: the half-life a bench chemist would call "goes overnight at
# most", used to turn a barrier into a temperature.
BENCH_HALF_LIFE_S = 3600.0
# Below this a barrier is negative, not small (as in mepd.discovery.kinetics).
NEGATIVE_BARRIER_TOL = -0.1


def rate_constant(barrier_kcal: float, temperature: float) -> float:
    """First-order Eyring rate constant (1/s)."""
    return Boltzmann * temperature / Planck * math.exp(-barrier_kcal * _KCAL_TO_J / (gas_constant * temperature))


def half_life(barrier_kcal: float, temperature: float) -> float:
    """Seconds (first order)."""
    exponent = barrier_kcal * _KCAL_TO_J / (gas_constant * temperature)
    if exponent > 700:
        return math.inf
    return math.log(2) / rate_constant(barrier_kcal, temperature)


def temperature_for_half_life(barrier_kcal: float, t_half_s: float = BENCH_HALF_LIFE_S) -> Optional[float]:
    """Temperature (K) at which the half-life is `t_half_s`; None when it is
    shorter than that even at 50 K or needs more than 5000 K."""
    if barrier_kcal is None or not math.isfinite(barrier_kcal):
        return None
    lo, hi = 50.0, 5000.0
    if half_life(barrier_kcal, lo) <= t_half_s or half_life(barrier_kcal, hi) > t_half_s:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if half_life(barrier_kcal, mid) > t_half_s:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _pow10(exponent: float) -> str:
    """10 to a whole power, written with superscripts: '10¹⁰'."""
    return "10" + f"{exponent:.0f}".translate(_SUP)


def format_duration(seconds: float) -> str:
    """Compact: '3.2 h', '10⁸ yr', '> 10¹⁰ yr' (longer than the universe's age), '< 1 µs'."""
    if seconds is None or math.isnan(seconds):
        return "?"
    if math.isinf(seconds) or seconds > 3.15e7 * 1e6:
        years = seconds / 3.15e7 if math.isfinite(seconds) else math.inf
        return "> 10¹⁰ yr" if years > 1.4e10 else f"{_pow10(math.log10(years))} yr"
    for unit, size in (("yr", 3.15e7), ("d", 86400.0), ("h", 3600.0), ("min", 60.0), ("s", 1.0)):
        if seconds >= size:
            v = seconds / size
            return f"{v:.0f} {unit}" if v >= 10 else f"{v:.1f} {unit}"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f} ms"
    if seconds >= 1e-6:
        return f"{seconds * 1e6:.1f} µs"
    return "< 1 µs"


def _celsius(t_k: Optional[float]) -> Optional[float]:
    return None if t_k is None else t_k - ZERO_C


def kinetics_row(barrier_kcal: Optional[float], temperature: float) -> dict:
    """Half-life at `temperature` and the temperature for a 1 h half-life."""
    if barrier_kcal is None:
        return {"t_half_s": None, "t_half": None, "t_1h_c": None}
    t = half_life(barrier_kcal, temperature)
    return {"t_half_s": t if math.isfinite(t) else None, "t_half": format_duration(t),
            "t_1h_c": _celsius(temperature_for_half_life(barrier_kcal))}


def _kirkwood(eps: float) -> float:
    """Onsager/Kirkwood polarity function (eps - 1) / (2 eps + 1)."""
    return (eps - 1.0) / (2.0 * eps + 1.0)


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else 0.0


def solvent_insights(gas: dict, rows: list[dict], temperature: float, *, mode: str) -> list[dict]:
    """Plain-language findings from a gas-phase barrier and the same
    reaction's barriers in several solvents.

    `gas`: {"barrier_kcal", "reaction_kcal"}; each row: a solvent's
    {"key", "label", "kind", "epsilon", "bp_c", "mp_c", "barrier_kcal",
    "shift_kcal", "reaction_kcal", "t_1h_c", "peak_shift"}. Returns
    [{"level": "accelerates" | "slows" | "trend" | "caution" | "note", "text", "solvents"}].
    """
    out: list[dict] = []
    g = gas.get("barrier_kcal")
    t_c = temperature - ZERO_C
    done = [r for r in rows if r.get("barrier_kcal") is not None]
    if g is None or not done:
        return out

    cautions: list[dict] = []
    negative = [r for r in done if r["barrier_kcal"] < NEGATIVE_BARRIER_TOL]
    if negative:
        cautions.append({"level": "caution", "solvents": [r["key"] for r in negative], "text":
                    f"Negative ΔE‡ in {', '.join(r['label'] for r in negative[:4])} "
                    f"({min(r['barrier_kcal'] for r in negative):.1f}): TS below reactant. "
                    + ("Re-optimize in solvent." if mode == "single-point" else "Check the solvent re-minimization.")})
        done = [r for r in done if r not in negative]

    # When single points on gas-phase geometries are not good enough.
    unsure: set = set()
    if mode == "single-point":
        big = [r for r in done if abs(r["shift_kcal"]) >= REOPTIMIZE_SHIFT]
        moved = [r for r in done if r.get("peak_shift")]
        if big:
            cautions.append({"level": "caution", "solvents": [r["key"] for r in big], "text":
                             f"Shift up to {max(abs(r['shift_kcal']) for r in big):.0f} kcal/mol "
                             f"({', '.join(r['label'] for r in big[:4])}): the TS likely moves. Re-optimize in solvent."})
        if moved:
            cautions.append({"level": "caution", "solvents": [r["key"] for r in moved], "text":
                             f"{', '.join(r['label'] for r in moved[:4])}: the solvated peak is off the gas-phase TS "
                             "(ΔE‡ given at that peak). Re-optimize in solvent."})
        unsure = {r["key"] for r in big + moved}

    def est(r: dict) -> str:
        return " (single-point estimate)" if r["key"] in unsure else ""

    faster = sorted((r for r in done if r["shift_kcal"] <= -NOTABLE_SHIFT), key=lambda r: r["shift_kcal"])
    slower = sorted((r for r in done if r["shift_kcal"] >= NOTABLE_SHIFT), key=lambda r: -r["shift_kcal"])

    def factor(shift: float) -> str:
        x = -shift * _KCAL_TO_J / (gas_constant * temperature) / math.log(10)
        return f"~{_pow10(x)}×" if abs(x) >= 3 else f"~{10 ** x:.0f}×" if x > 0 else f"~{10 ** -x:.0f}×"

    if faster:
        best = faster[0]
        kinds = sorted({r["kind"] for r in faster})
        text = (f"Faster in {best['label']}: ΔE‡ {g:.1f} → {best['barrier_kcal']:.1f} ({best['shift_kcal']:+.1f}), "
                f"{factor(best['shift_kcal'])} at {t_c:.0f} °C{est(best)}")
        others = [f"{r['label']} {r['barrier_kcal']:.1f}" for r in faster[1:4]]
        text += (f". Also {', '.join(others)}" if others else "")
        if len(kinds) == 1 and len(faster) > 1:
            text += f" (all {kinds[0]})"
        out.append({"level": "accelerates", "text": text, "solvents": [r["key"] for r in faster]})
        gas_t, best_t = half_life(g, temperature), half_life(best["barrier_kcal"], temperature)
        out.append({"level": "note", "solvents": [best["key"]],
                    "text": f"t½ at {t_c:.0f} °C: {format_duration(gas_t)} (gas) → {format_duration(best_t)} "
                            f"({best['label']}){est(best)}"})
    if slower:
        worst = slower[0]
        out.append({"level": "slows", "solvents": [r["key"] for r in slower],
                    "text": f"Slower in {', '.join(r['label'] for r in slower[:4])}: up to ΔE‡ "
                            f"{worst['barrier_kcal']:.1f} ({worst['shift_kcal']:+.1f}, {worst['label']}); the reactant "
                            "is better solvated than the TS"})

    # Polarity trend over the solvents computed.
    polar = [r for r in done if r.get("epsilon")]
    if len(polar) >= 3:
        xs = [_kirkwood(r["epsilon"]) for r in polar]
        ys = [r["shift_kcal"] for r in polar]
        rho = _spearman(xs, ys)
        spread = max(ys) - min(ys)
        if spread >= NOTABLE_SHIFT and abs(rho) >= 0.7:
            if rho < 0:
                out.append({"level": "trend", "solvents": [], "text":
                            f"ΔE‡ falls with polarity (ρ = {rho:.2f}): the TS is more polar than the reactant"})
            else:
                out.append({"level": "trend", "solvents": [], "text":
                            f"ΔE‡ rises with polarity (ρ = {rho:+.2f}): nonpolar solvents are best"})

    # Is it a room-temperature reaction anywhere, and can the solvent take the heat?
    for r in (faster[:1] if faster else []):
        t1h = r.get("t_1h_c")
        if t1h is None:
            continue
        if r.get("bp_c") is not None and t1h > r["bp_c"]:
            alt = sorted((o for o in done if o["kind"] == r["kind"] and o.get("bp_c") is not None
                          and o.get("t_1h_c") is not None and o["bp_c"] >= o["t_1h_c"]),
                         key=lambda o: o["barrier_kcal"])
            text = (f"{r['label']}: 1 h t½ at ~{t1h:.0f} °C, above its bp ({r['bp_c']:.0f} °C): sealed vessel")
            text += (f", or {alt[0]['label']} (bp {alt[0]['bp_c']:.0f} °C, ~{alt[0]['t_1h_c']:.0f} °C)" if alt else "")
            out.append({"level": "note", "solvents": [r["key"]], "text": text + est(r)})
        elif r.get("mp_c") is not None and t1h < r["mp_c"]:
            out.append({"level": "note", "solvents": [r["key"]], "text":
                        f"{r['label']}: 1 h t½ at ~{t1h:.0f} °C, below its mp: fast even cold{est(r)}"})
        else:
            out.append({"level": "note", "solvents": [r["key"]], "text":
                        f"{r['label']}: 1 h t½ at ~{t1h:.0f} °C (bp {r['bp_c']:.0f} °C){est(r)}"})

    # Thermodynamics can flip too.
    gr = gas.get("reaction_kcal")
    if gr is not None:
        flips = [r for r in done if r.get("reaction_kcal") is not None and (r["reaction_kcal"] < 0) != (gr < 0)
                 and abs(r["reaction_kcal"] - gr) >= 1.0]
        if flips:
            word = "downhill" if gr >= 0 else "uphill"
            out.append({"level": "note", "solvents": [r["key"] for r in flips], "text":
                        f"ΔE_rxn changes sign in {', '.join(r['label'] for r in flips[:4])}: {word} there "
                        f"(gas {gr:+.1f})"})

    if any(r["kind"] == "protic" for r in done):
        out.append({"level": "note", "solvents": [r["key"] for r in done if r["kind"] == "protic"], "text":
                    "Protic solvents can relay protons or H-bond the TS, which implicit solvent misses: "
                    "try an explicit water or methanol in Design"})
    return cautions + out


def simulate_first_order(n_species: int, steps: list[tuple[int, int, float, float]], c0: list[float],
                         time_s: float, n_samples: int = 60) -> dict:
    """Integrate dc/dt = K c for first-order steps (a, b, k(a->b), k(b->a))
    from `c0` to `time_s` (Radau: rates span many orders of magnitude).
    Returns {"t": [...], "c": [[species at each t], ...], "final": [...]}."""
    import numpy as np
    from scipy.integrate import solve_ivp

    K = np.zeros((n_species, n_species))
    for a, b, kf, kr in steps:
        if a == b:
            continue
        K[b, a] += kf
        K[a, a] -= kf
        K[a, b] += kr
        K[b, b] -= kr
    y0 = np.asarray(c0, dtype=float)
    fastest = max([max(kf, kr) for _, _, kf, kr in steps] or [0.0])
    if fastest <= 0:
        return {"t": [0.0, time_s], "c": [y0.tolist(), y0.tolist()], "final": y0.tolist()}
    t_eval = np.unique(np.concatenate([[0.0], np.geomspace(min(1e-3 / fastest, time_s), time_s, n_samples)]))
    sol = solve_ivp(lambda t, y: K @ y, (0.0, time_s), y0, method="Radau", jac=K, t_eval=t_eval,
                    rtol=1e-8, atol=1e-14)
    c = np.clip(sol.y, 0.0, None)
    return {"t": sol.t.tolist(), "c": c.T.tolist(), "final": c[:, -1].tolist()}
