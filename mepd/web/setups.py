"""Experimental setups for the Explore graph (a prototype).

A setup is the conditions an experiment would run under: a solvent (or
the gas phase), a temperature, a reaction time, and the structures it
starts from. Under a setup every edge of the graph gets

* the barrier that applies there: the edge's barrier in that solvent when
  a "Solvent effects" job has computed it, else its gas-phase barrier
  (flagged as such -- the phases are never silently mixed);
* a half-life at the setup's temperature (Eyring), which says whether the
  step runs within the reaction time;

and the whole graph can be run forward in time from the starting
structures (first-order kinetics over its edges, mepd.conditions), to
predict what the flask would contain at the end.

Every step is first order: a node is one species or one complex (as the
path searches treat them), so concentrations do not enter the rates.
Barriers are electronic (+ solvation free energy), not free energies, so
the predicted amounts are qualitative.
"""
from __future__ import annotations

import math
import secrets
from typing import Optional

from mepd.web.workspace import HARTREE_TO_KCAL, WorkspaceError

DEFAULT_SETUP = {"name": "", "solvent": None, "temperature": 298.15, "time_s": 3600.0,
                 "start": [], "level": None, "variant": None}


def normalize(setup: dict) -> dict:
    from mepd.solvation import get_solvent

    s = {**DEFAULT_SETUP, **{k: v for k, v in (setup or {}).items() if v is not None or k == "solvent"}}
    if s.get("solvent"):
        try:
            s["solvent"] = get_solvent(s["solvent"]).key
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from None
    else:
        s["solvent"] = None
    s["temperature"] = float(s["temperature"])
    s["time_s"] = float(s["time_s"])
    if not (0 < s["temperature"] < 5000):
        raise WorkspaceError("temperature must be between 0 and 5000 K")
    if s["time_s"] <= 0:
        raise WorkspaceError("reaction time must be positive")
    s["start"] = [str(x) for x in (s.get("start") or [])]
    # Level of theory: a profile name (only barriers computed with it), or None (any).
    s["level"] = str(s["level"]) if s.get("level") else None
    # A substituent variant, {site: atom number, group}: each edge then uses its
    # "Substituent effects" shift for that site and group.
    v = s.get("variant") or None
    if v:
        from mepd.substituents import group_smiles

        if v.get("group") not in group_smiles():
            raise WorkspaceError(f"unknown group {v.get('group')!r}")
        try:
            v = {"site": int(v["site"]), "group": str(v["group"])}
        except (KeyError, TypeError, ValueError):
            raise WorkspaceError("a variant needs an atom number (site) and a group") from None
    s["variant"] = v
    s["name"] = str(s.get("name") or "").strip() or _auto_name(s)
    s["id"] = str(s.get("id") or "") or "s_" + secrets.token_hex(4)
    return {k: s[k] for k in ("id", "name", "solvent", "temperature", "time_s", "start", "level", "variant")}


def _auto_name(s: dict) -> str:
    from mepd.solvation import SOLVENTS

    medium = SOLVENTS[s["solvent"]].label if s.get("solvent") else "Gas phase"
    name = f"{medium}, {s['temperature'] - 273.15:.0f} °C"
    if s.get("variant"):
        from mepd.substituents import SHORT

        name += f", {SHORT.get(s['variant']['group'], s['variant']['group'])} on atom {s['variant']['site']}"
    if s.get("level"):
        name += f" ({s['level']})"
    return name


def _gas_barrier(edge: dict, jobs: dict, level_key: Optional[str] = None) -> Optional[dict]:
    """The edge's lowest verified gas-phase barrier, in its direction (only
    from jobs at `level_key`, when given)."""
    best = None

    def at_level(j) -> bool:
        return level_key is None or ((j or {}).get("level") or {}).get("key") == level_key

    for j in jobs.values():
        summ = j.get("summary") or {}
        if j.get("status") != "done" or j.get("op") == "solvent" or summ.get("barrier_kcal") is None:
            continue
        if edge["id"] not in (j.get("targets") or {}).get("edges", []) or summ.get("barrier_verified") is False:
            continue
        if not at_level(j):
            continue
        cand = {"barrier_kcal": float(summ["barrier_kcal"]), "job": j["id"], "level": j.get("level")}
        if best is None or cand["barrier_kcal"] < best["barrier_kcal"]:
            best = cand
    origin = edge.get("origin") or {}
    ojob = jobs.get(origin.get("job") or "")
    if origin.get("barrier_kcal") is not None and not origin.get("proposed") and at_level(ojob) \
            and (best is None or origin["barrier_kcal"] < best["barrier_kcal"]):
        best = {"barrier_kcal": float(origin["barrier_kcal"]), "job": origin.get("job"),
                "level": (ojob or {}).get("level")}
    return best


def _solvent_barrier(edge: dict, jobs: dict, solvent: str) -> Optional[dict]:
    """The edge's barrier in `solvent` from its latest finished solvent job
    (re-optimized results preferred over single points)."""
    origin_job = (edge.get("origin") or {}).get("job")
    found = []
    for j in jobs.values():
        cond = (j.get("summary") or {}).get("conditions")
        if j.get("op") != "solvent" or j.get("status") != "done" or not cond:
            continue
        on_edge = edge["id"] in (j.get("targets") or {}).get("edges", []) \
            or (origin_job and j.get("source_job") == origin_job)
        row = (cond.get("solvents") or {}).get(solvent)
        if on_edge and row and row.get("barrier_kcal") is not None:
            found.append((cond.get("mode") == "reoptimize", j.get("finished") or 0, row, j["id"], cond))
    if not found:
        return None
    _, _, row, jid, cond = max(found, key=lambda x: (x[0], x[1]))
    return {"barrier_kcal": float(row["barrier_kcal"]), "reaction_kcal": row.get("reaction_kcal"),
            "job": jid, "mode": cond.get("mode")}


def edge_under(edge: dict, structures: dict, jobs: dict, setup: dict) -> dict:
    """What an edge looks like under `setup`: barrier, where it comes from
    ("solvent" | "gas" | None), half-life, and whether it runs in time."""
    from mepd.conditions import format_duration, half_life

    T, solvent = setup["temperature"], setup.get("solvent")
    out = {"edge": edge["id"], "source": edge["source"], "target": edge["target"], "barrier_kcal": None,
           "reverse_kcal": None, "phase": None, "job": None, "mode": None}
    hit = _solvent_barrier(edge, jobs, solvent) if solvent else None
    if hit is not None:
        out.update(barrier_kcal=hit["barrier_kcal"], phase="solvent", job=hit["job"], mode=hit["mode"])
        if hit.get("reaction_kcal") is not None:
            out["reverse_kcal"] = hit["barrier_kcal"] - hit["reaction_kcal"]
    else:
        gas = _gas_barrier(edge, jobs)
        if gas is not None:
            out.update(barrier_kcal=gas["barrier_kcal"], phase="gas", job=gas["job"])
            rxn = _reaction_energy(structures.get(edge["source"]), structures.get(edge["target"]))
            if rxn is not None:
                out["reverse_kcal"] = gas["barrier_kcal"] - rxn
    if out["barrier_kcal"] is not None:
        t = half_life(max(out["barrier_kcal"], 0.0), T)
        out["t_half_s"] = t if math.isfinite(t) else None
        out["t_half"] = format_duration(t)
        out["runs"] = "yes" if t <= setup["time_s"] else "slow" if t <= 100 * setup["time_s"] else "no"
    return out


def _reaction_energy(a: Optional[dict], b: Optional[dict]) -> Optional[float]:
    """E(b) - E(a) in kcal/mol, only when both are minima at one level."""
    if not a or not b or a.get("energy") is None or b.get("energy") is None:
        return None
    if (a.get("level") or {}).get("key") != (b.get("level") or {}).get("key"):
        return None
    return (float(b["energy"]) - float(a["energy"])) * HARTREE_TO_KCAL


def predict(snapshot: dict, jobs: dict, setup: dict) -> dict:
    """Run the graph forward from the setup's starting structures."""
    from mepd.conditions import rate_constant, simulate_first_order

    structures = {sid: r for sid, r in snapshot["structures"].items() if r.get("role") != "ts"}
    start = [sid for sid in setup.get("start") or [] if sid in structures]
    if not start:
        raise WorkspaceError("choose the structure(s) the experiment starts from")
    ids = list(structures)
    index = {sid: i for i, sid in enumerate(ids)}
    T = setup["temperature"]
    steps, used, missing, warnings = [], [], [], []
    for edge in snapshot["edges"].values():
        if edge["source"] not in index or edge["target"] not in index:
            continue
        info = edge_under(edge, structures, jobs, setup)
        if info["barrier_kcal"] is None:
            missing.append(edge["id"])
            continue
        fwd, rev = info["barrier_kcal"], info["reverse_kcal"]
        for label, value in (("forward", fwd), ("reverse", rev)):
            if value is not None and value < -0.1:
                warnings.append(f"Edge {edge['id']}: negative {label} barrier ({value:.1f} kcal/mol), used as zero "
                                "in the rate only. Check its endpoints' minimization and level of theory.")
        kf = rate_constant(max(fwd, 0.0), T)
        kr = rate_constant(max(rev, 0.0), T) if rev is not None else 0.0
        steps.append((index[edge["source"]], index[edge["target"]], kf, kr))
        used.append(info)
    c0 = [0.0] * len(ids)
    for sid in start:
        c0[index[sid]] = 1.0 / len(start)
    sim = simulate_first_order(len(ids), steps, c0, setup["time_s"])
    gas_used = [u for u in used if u["phase"] == "gas"]
    if setup.get("solvent") and gas_used:
        warnings.append(f"{len(gas_used)} of {len(used)} step(s) have no barrier in this solvent yet and use their "
                        "gas-phase barrier: compute solvent effects on them for a consistent prediction.")
    irreversible = [u for u in used if u["reverse_kcal"] is None]
    if irreversible:
        warnings.append(f"{len(irreversible)} step(s) run one way only: their reaction energy is not known at one "
                        "level of theory, so no reverse rate could be set.")
    notes = []
    for u in used:
        if u["reverse_kcal"] is None:
            continue
        rxn = u["barrier_kcal"] - u["reverse_kcal"]
        if rxn > 2.0:
            a, b = (structures[u[k]].get("name") or u[k] for k in ("source", "target"))
            notes.append(f"{a} → {b} is uphill by {rxn:.1f} kcal/mol under these conditions: it goes back faster "
                         f"than it goes forward, so its equilibrium lies on the {a} side.")
    species = sorted(({"id": sid, "name": structures[sid].get("name"), "final": sim["final"][index[sid]],
                       "start": c0[index[sid]]} for sid in ids), key=lambda r: -r["final"])
    return {"setup": setup, "species": [s for s in species if s["final"] > 1e-4 or s["start"] > 0],
            "steps": used, "missing": missing, "warnings": warnings, "notes": notes,
            "curve": {"t": sim["t"], "ids": ids, "c": sim["c"]}}


def fill_requests(snapshot: dict, jobs: dict, setup: dict) -> list[dict]:
    """The solvent jobs to queue so that every edge with a gas-phase TS
    also has a barrier in the setup's solvent: [{source_job, edge}]."""
    solvent = setup.get("solvent")
    if not solvent:
        return []
    out, seen = [], set()
    for edge in snapshot["edges"].values():
        if _solvent_barrier(edge, jobs, solvent) is not None:
            continue
        gas = _gas_barrier(edge, jobs)
        src = jobs.get((gas or {}).get("job") or "")
        if not src or src.get("op") not in ("ts", "tsopt", "design-tsopt") or src["id"] in seen:
            continue
        busy = any(j.get("op") == "solvent" and j.get("source_job") == src["id"]
                   and j.get("status") in ("queued", "running") for j in jobs.values())
        if busy:
            continue
        seen.add(src["id"])
        out.append({"source_job": src["id"], "edge": edge["id"]})
    return out


# ------------------------------------------------------------ network model

def _variant_shift(gas_job: Optional[str], jobs: dict, variant: dict, channel: Optional[str] = None) -> Optional[dict]:
    """The latest "Substituent effects" result on `gas_job` for this site
    and group: {shift, reaction_shift, job, mode}, for `channel` (an edge
    made from one channel of a channels run) or else the lowest one."""
    import json as _json
    from pathlib import Path

    if not gas_job:
        return None
    runs = sorted((j for j in jobs.values() if j.get("op") == "substituents" and j.get("status") == "done"
                   and j.get("source_job") == gas_job), key=lambda j: j.get("finished") or 0, reverse=True)
    for j in runs:
        try:
            data = _json.loads((Path(j["output_dir"]) / "summary.json").read_text())
        except (OSError, ValueError, KeyError):
            continue
        chans = data.get("channels") or []
        if not chans:
            continue
        ids = {c["id"] for c in chans}
        lead = channel if channel in ids else min(chans, key=lambda c: c["barrier_kcal"])["id"]
        for r in data.get("variants") or []:
            hit = r["site"] == variant["site"] or r.get("h") == variant["site"]
            if r["channel"] == lead and hit and r["group"] == variant["group"] and r["status"] == "ok" \
                    and not r.get("clash") and r.get("shift") is not None:
                return {"shift": float(r["shift"]), "reaction_shift": r.get("reaction_shift"), "job": j["id"],
                        "mode": data.get("mode")}
    return None


def build_model(snapshot: dict, jobs: dict, setup: dict, levels: Optional[dict] = None):
    """A mepd.network_model.NetworkModel of the Explore graph under `setup`:
    every edge with a usable barrier becomes a step, with its provenance
    (job, level, phase, variant); species energies are placed on one scale
    from the start outwards, through each step's reaction energy."""
    from mepd.network_model import NetworkModel, Species, Step

    structures = {sid: r for sid, r in snapshot["structures"].items() if r.get("role") != "ts"}
    level_key = (levels or {}).get(setup["level"], {}).get("key") if setup.get("level") else None
    if setup.get("level") and level_key is None:
        raise WorkspaceError(f"unknown level of theory {setup['level']!r}")
    solvent, variant = setup.get("solvent"), setup.get("variant")
    edges, warnings = [], []
    for edge in snapshot["edges"].values():
        if edge["source"] not in structures or edge["target"] not in structures:
            continue
        gas = _gas_barrier(edge, jobs, level_key)
        hit = _solvent_barrier(edge, jobs, solvent) if solvent and gas is not None else None
        if gas is None:
            continue
        prov = {"edge": edge["id"], "job": gas["job"], "level": (gas.get("level") or {}).get("label"),
                "level_key": (gas.get("level") or {}).get("key"), "phase": "gas", "mode": None, "variant": None}
        fwd = gas["barrier_kcal"]
        rxn = _reaction_energy(structures[edge["source"]], structures[edge["target"]])
        if hit is not None:
            fwd, rxn = hit["barrier_kcal"], hit.get("reaction_kcal")
            prov.update(phase=solvent, mode=hit["mode"], solvent_job=hit["job"])
        if variant:
            origin = edge.get("origin") or {}
            v = _variant_shift(gas["job"], jobs, variant,
                               channel=origin.get("entry") if origin.get("job") == gas["job"] else None)
            if v is not None:
                fwd += v["shift"]
                rxn = rxn + v["reaction_shift"] if rxn is not None and v.get("reaction_shift") is not None else rxn
                prov.update(variant=f"{variant['group']}@{variant['site']}", variant_job=v["job"],
                            variant_mode=v["mode"], additive=hit is not None)
        edges.append((edge, fwd, rxn, prov))
    start = [s for s in setup.get("start") or [] if s in structures]
    if not start:
        raise WorkspaceError("choose the structure(s) the experiment starts from")
    # Species energies from the start outwards, through each step's reaction energy.
    # One anchor per connected part: the first start in it is zero, every
    # other species (other starts included) gets its energy through the
    # steps. A second start is never pinned to zero as well: that would
    # invent an energy difference.
    G: dict = {}
    adj: dict = {}
    for e, fwd, rxn, prov in edges:
        if rxn is None:
            continue
        adj.setdefault(e["source"], []).append((e["target"], rxn))
        adj.setdefault(e["target"], []).append((e["source"], -rxn))
    anchors = 0
    for root in start:
        if root in G:
            continue
        G[root] = 0.0
        anchors += 1
        todo = [root]
        while todo:
            x = todo.pop(0)
            for y, d in adj.get(x, []):
                if y not in G:
                    G[y] = G[x] + d
                    todo.append(y)
    if anchors > 1:
        warnings.append(f"The starting structures sit in {anchors} unconnected parts of the network: their energies "
                        "are not compared with each other.")
    steps, used = [], set()
    no_rxn = 0
    for e, fwd, rxn, prov in edges:
        a, b = e["source"], e["target"]
        if a not in G or b not in G:
            no_rxn += rxn is None
            continue
        if rxn is not None and abs(G[b] - G[a] - rxn) > 1.0:
            warnings.append(f"Energies around a cycle disagree by {abs(G[b] - G[a] - rxn):.1f} kcal/mol at edge "
                            f"{structures[a].get('name')} → {structures[b].get('name')} (mixed sources).")
        steps.append(Step(e["id"], a, b, G[a] + fwd, prov))
        used.update((a, b))
    species = [Species(sid, structures[sid].get("name") or sid, G[sid],
                       {"level": (structures[sid].get("level") or {}).get("label")})
               for sid in structures if sid in used or sid in start]
    # Counted over the steps in the model only (not edges the start never reaches).
    gas_fallback = sum(1 for st in steps if solvent and st.provenance.get("phase") == "gas")
    missing_variant = sum(1 for st in steps if variant and not st.provenance.get("variant"))
    if gas_fallback:
        warnings.append(f"{gas_fallback} of {len(steps)} step(s) have no {solvent} barrier: gas-phase values used.")
    if missing_variant:
        warnings.append(f"{missing_variant} of {len(steps)} step(s) have no result for this substituent: "
                        "unsubstituted values used.")
    if no_rxn:
        warnings.append(f"{no_rxn} step(s) left out: no reaction energy links them to the start at one level.")
    levels_used = {p.provenance.get("level_key") for p in steps}
    if len(levels_used) > 1:
        warnings.append(f"Barriers from {len(levels_used)} levels of theory are mixed: pick one level to compare.")
    return NetworkModel(species, steps, setup["temperature"], setup["time_s"], {s: 1.0 for s in start},
                        warnings, {"setup": setup})


def analyze(snapshot: dict, jobs: dict, setup: dict, levels: Optional[dict] = None) -> dict:
    """Whole-network properties under a setup, and what controls them."""
    from mepd.network_model import rate_to, yield_of

    m = build_model(snapshot, jobs, setup, levels)
    names = {s.id: s.name for s in m.species}
    props = m.properties()
    sens = {}
    for p in props["products"][:2]:
        sens[p] = {"yield": _top(m.sensitivity(yield_of(p))), "rate": _top(m.sensitivity(rate_to(p)))}
    return {"setup": setup, "names": names, "properties": props, "sensitivity": sens,
            "steps": [{"id": st.id, "a": st.a, "b": st.b, "barrier": fr, "reverse": rv, **st.provenance}
                      for st, (fr, rv) in zip(m.steps, m.barriers())],
            "species": [{"id": s.id, "name": s.name, "G": s.G} for s in m.species],
            "warnings": m.warnings}


def _top(sens: dict, n: int = 5) -> dict:
    """The largest degrees of control, TS and species."""
    return {k: sorted(({"id": i, "x": float(x)} for i, x in v.items() if x is not None and abs(x) > 0.02),
                      key=lambda r: -abs(r["x"]))[:n] for k, v in sens.items()}


def compare_setups(snapshot: dict, jobs: dict, base: dict, other: dict, levels: Optional[dict] = None) -> dict:
    """How the network's properties change from `base` to `other`, and which
    steps' and species' energy changes account for it."""
    from mepd.network_model import compare, rate_to, yield_of

    mb = build_model(snapshot, jobs, base, levels)
    mo = build_model(snapshot, jobs, other, levels)
    common_steps = {st.id for st in mb.steps} & {st.id for st in mo.steps}
    common_species = {s.id for s in mb.species} & {s.id for s in mo.species}
    # One network to compare: the steps both setups have.
    mb = _restrict(mb, common_steps, common_species)
    mo = _restrict(mo, common_steps, common_species)
    props_b, props_o = mb.properties(), mo.properties(products=None)
    products = list(dict.fromkeys(props_b["products"][:2] + props_o["products"][:2]))
    q = {}
    for p in products:
        q[f"yield:{p}"] = yield_of(p)
        q[f"rate:{p}"] = rate_to(p)
    res = compare(mb, mo, q)
    return {"base": base, "other": other, "names": {s.id: s.name for s in mb.species},
            "properties": {"base": props_b, "other": props_o}, **res,
            "warnings": list(dict.fromkeys(mb.warnings + mo.warnings)),
            "steps": [{"id": st.id, "a": st.a, "b": st.b} for st in mb.steps]}


def _restrict(m, steps: set, species: set):
    from mepd.network_model import NetworkModel

    return NetworkModel([s for s in m.species if s.id in species], [st for st in m.steps if st.id in steps],
                        m.temperature, m.time_s, {k: v for k, v in m.start.items() if k in species},
                        list(m.warnings), m.spec)
