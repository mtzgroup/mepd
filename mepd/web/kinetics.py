"""Analyze › Kinetics: the workspace's network as a microkinetic model
(mepd.microkinetics) -- every reaction (nanoreactor or composed) and every
plain edge whose TS search gave a barrier, among species with energies at
one level of theory -- and the analyses the page shows."""
from __future__ import annotations

from typing import Optional

import numpy as np

from mepd import microkinetics as mk

HARTREE_KCAL = 627.509474


def edge_barrier(edge: dict, jobs: dict, include_unverified: bool) -> tuple[Optional[float], bool, Optional[str]]:
    """(barrier from the edge's source side, verified?, job id) -- as the graph
    shows it: the lowest IRC-verified barrier, else (if allowed) the lowest
    path maximum."""
    ver, unver = [], []
    for j in jobs.values():
        if edge["id"] not in j["targets"]["edges"] or j["status"] != "done":
            continue
        b = (j.get("summary") or {}).get("barrier_kcal")
        if b is None:
            continue
        (unver if (j.get("summary") or {}).get("barrier_verified") is False else ver).append((b, j["id"]))
    o = edge.get("origin") or {}
    if o.get("barrier_kcal") is not None:
        ver.append((o["barrier_kcal"], o.get("job")))
    if ver:
        b, jid = min(ver, key=lambda x: x[0])
        return b, True, jid
    if include_unverified and unver:
        b, jid = min(unver, key=lambda x: x[0])
        return b, False, jid
    return None, False, None


def build(snap: dict, jobs: dict, level_key: Optional[str], *, include_unverified: bool = False):
    """(Network, species ids, step info, excluded) for the workspace."""
    st = snap["structures"]

    def usable(rec):
        return rec and rec.get("energy") is not None and (level_key is None or (rec.get("level") or {}).get("key") == level_key)

    ids = [sid for sid, r in st.items() if r.get("role") not in ("ts", "complex") and usable(r)]
    index = {sid: k for k, sid in enumerate(ids)}
    steps, info, excluded = [], [], []
    for r in (snap.get("reactions") or {}).values():
        label = r.get("label") or "reaction"
        edge = snap["edges"].get(r.get("edge") or "")
        if edge is None:
            excluded.append({"label": label, "reason": "no TS endpoints (" + (r.get("complex_reason") or "none") + ")",
                             "reaction": r["id"]})
            continue
        b, verified, jid = edge_barrier(edge, jobs, include_unverified)
        if b is None:
            excluded.append({"label": label, "reason": "no barrier yet", "reaction": r["id"]})
            continue
        rc = st.get(edge["source"])
        missing = [st.get(s, {}).get("name", s) for s in r["reactants"] + r["products"] if s not in index]
        if not usable(rc) or missing:
            excluded.append({"label": label, "reaction": r["id"],
                             "reason": "energies missing at this level: " + ", ".join(missing or ["reactant complex"])})
            continue
        steps.append(mk.Step([index[s] for s in r["reactants"]], [index[s] for s in r["products"]],
                             rc["energy"] * HARTREE_KCAL + b, label))
        info.append({"kind": "reaction", "reaction": r["id"], "edge": edge["id"], "label": label,
                     "verified": verified, "job": jid, "shuttles": r.get("shuttles") or []})
    complexes = {sid for sid, rec in st.items() if rec.get("role") == "complex"}
    for e in snap["edges"].values():
        if e["source"] in complexes or e["target"] in complexes:
            continue
        b, verified, jid = edge_barrier(e, jobs, include_unverified)
        if b is None:
            continue
        a, c = e["source"], e["target"]
        label = f"{st.get(a, {}).get('name', '?')} -> {st.get(c, {}).get('name', '?')}"
        if a not in index or c not in index:
            excluded.append({"label": label, "reason": "an end has no energy at this level", "edge": e["id"]})
            continue
        steps.append(mk.Step([index[a]], [index[c]], st[a]["energy"] * HARTREE_KCAL + b, label))
        info.append({"kind": "edge", "edge": e["id"], "label": label, "verified": verified, "job": jid,
                     "shuttles": []})
    # The same transformation from several runs (same reactants and products,
    # shuttles included) is one channel: keep its lowest TS, not two copies
    # that would double its rate.
    best: dict = {}
    for k, stp in enumerate(steps):
        a, b = tuple(sorted(stp.reactants)), tuple(sorted(stp.products))
        key = min((a, b), (b, a))
        if key not in best or stp.g_ts < steps[best[key]].g_ts:
            best[key] = k
    keep = sorted(best.values())
    merged = len(steps) - len(keep)
    for k in keep:
        a, b = tuple(sorted(steps[k].reactants)), tuple(sorted(steps[k].products))
        info[k]["copies"] = sum(1 for s2 in steps if min((tuple(sorted(s2.reactants)), tuple(sorted(s2.products))),
                                                          (tuple(sorted(s2.products)), tuple(sorted(s2.reactants))))
                                == min((a, b), (b, a)))
    steps, info = [steps[k] for k in keep], [info[k] for k in keep]
    names = [st[s]["name"] for s in ids]
    g = np.array([st[s]["energy"] * HARTREE_KCAL for s in ids])
    return mk.Network(names, g, steps), ids, info, excluded


def analyze(snap: dict, jobs: dict, level_key: Optional[str], *, initial: dict, held: list, temperature: float,
            time_s: float, target: Optional[str], include_unverified: bool = False, control: bool = True,
            sweep: Optional[list] = None, control_what: str = "amount") -> dict:
    net, ids, info, excluded = build(snap, jobs, level_key, include_unverified=include_unverified)
    warnings = ["Electronic energies stand in for free energies: absolute rates (bimolecular ones especially, "
                "which lack the entropy cost of meeting) are orders of magnitude; read ratios, controlling steps "
                "and trends."]
    if not net.steps:
        return {"error": "No reaction in this workspace has a barrier yet: run TS searches first"
                + (" (or include unverified barriers)" if not include_unverified else "") + ".",
                "excluded": excluded}
    # Only species a step touches, or the user put in.
    used = sorted({i for s in net.steps for i in s.reactants + s.products}
                  | {ids.index(s) for s in initial if s in ids})
    keep = {old: new for new, old in enumerate(used)}
    steps = [mk.Step([keep[i] for i in s.reactants], [keep[i] for i in s.products], s.g_ts, s.label)
             for s in net.steps]
    net = mk.Network([net.names[i] for i in used], net.g[used], steps)
    sids = [ids[i] for i in used]
    pos = {sid: k for k, sid in enumerate(sids)}
    c0 = np.zeros(len(sids))
    for sid, conc in (initial or {}).items():
        if sid in pos:
            c0[pos[sid]] = float(conc)
    unknown = [snap["structures"].get(s, {}).get("name", s) for s in initial if s not in pos]
    if unknown:
        warnings.append("Not in the model (no energy at this level, or no step with a barrier): "
                        + ", ".join(unknown))
    held_idx = [pos[s] for s in held or [] if s in pos]
    res = mk.simulate(net, c0, temperature, time_s, held=held_idx)
    for (bf, br), s in zip(net.barriers(), net.steps):
        if bf < -0.1 or br < -0.1:
            warnings.append(f"{s.label}: its TS lies below one side ({bf:.1f} / {br:.1f} kcal/mol); it counts as "
                            "barrierless that way.")
    out = {
        "species": [{"id": sid, "name": net.names[k], "c0": float(c0[k]), "final": float(res.c[-1, k]),
                     "max": float(res.c[:, k].max()), "held": k in held_idx} for k, sid in enumerate(sids)],
        "times": res.times.tolist(),
        "series": {sid: res.c[:, k].tolist() for k, sid in enumerate(sids)},
        "steps": [{**i, "barrier_fwd": bf, "barrier_rev": br, "net_rate": float(nr), "extent": float(ex)}
                  for i, (bf, br), nr, ex in zip(info, net.barriers(), res.net_rate, res.extent)],
        "steady": res.steady, "temperature": temperature, "time_s": time_s, "excluded": excluded,
        "warnings": warnings, "mode": "constant feed" if held_idx else "batch",
    }
    if target in pos:
        t = pos[target]
        out["target"] = {"id": target, "name": net.names[t], "final": float(res.c[-1, t]),
                         "rate": mk.formation_rate(net, res, t),
                         "yield_of": float(c0.sum()) or None}
        if control:
            what = control_what if control_what in ("amount", "rate") else "amount"
            drc = mk.degree_of_control(net, c0, temperature, time_s, t, what=what, held=held_idx)
            out["control"] = {"what": what, "steps": drc["steps"], "species": drc["species"]}
        if sweep:
            out["sweep"] = mk.sweep(net, c0, [float(x) for x in sweep], time_s, t, held=held_idx)
    elif sweep:
        out["sweep"] = mk.sweep(net, c0, [float(x) for x in sweep], time_s, None, held=held_idx)
    return out
