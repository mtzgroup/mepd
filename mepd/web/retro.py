"""A retrosynthesis job's routes in Explore.

`mepd retro plan` writes network.json (and live_network.json while it
searches): species by SMILES with a 3D guess, and one reaction per distinct
step of the routes it reports. Here each species becomes a node (a molecule
already in the graph is reused, the target is the job's own node) and each
step a workspace reaction: its reactants and its products (+ byproducts)
packed as two complexes (mepd.complexes.packed) joined by a 'proposed' edge, where "Find TS"
maps the atoms and runs, like on any composed reaction. A step the job
path-searched itself (--verify) carries that barrier.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from mepd.web import chem
from mepd.web.workspace import WorkspaceError


def _read_json(fp: Path) -> Optional[dict]:
    try:
        return json.loads(fp.read_text())
    except (OSError, ValueError):
        return None


def adopt_retro(ws, job: dict) -> bool:
    """Bring a finished job's routes into the workspace (network.json; the
    live file's routes change as the search goes, so they stay in the job
    view). Returns True when the workspace changed."""
    data = _read_json(Path(job.get("output_dir") or "") / "network.json")
    if not data:
        return False
    nodes: dict = dict(job.get("retro_nodes") or {})       # SMILES -> structure id
    changed = False
    known = ws.snapshot()["structures"]
    seeds = (job.get("targets") or {}).get("structures") or []
    for sp in data.get("species") or []:
        smi = sp["smiles"]
        if nodes.get(smi) in known:
            continue
        if sp.get("role") == "target" and seeds and seeds[0] in known:
            nodes[smi] = seeds[0]
            continue
        rec = ws.find_molecule(smi, sp["charge"], sp["multiplicity"])
        if rec is None:
            try:
                text = Path(sp["md_file"]).read_text()
            except (OSError, KeyError):
                continue
            (s,) = chem.structures_from_xyz_text(text, sp["charge"], sp["multiplicity"])
            note = {"stock": "in stock", "small": "small, counted as available"}.get(sp.get("in_stock") or "")
            rec = ws.add_or_merge(s, name=smi, smiles=smi, optimized=False,
                                  origin={"kind": "job", "job": job["id"], "entry": f"species_{sp['id']}",
                                          "label": "building block" if note else sp.get("role", "species"),
                                          "retro": True, **({"note": note} if note else {})})["rec"]
        nodes[smi] = rec["id"]
        changed = True
        known = ws.snapshot()["structures"]
    by_id = {sp["id"]: sp["smiles"] for sp in data.get("species") or []}
    for rx in data.get("reactions") or []:
        r_ids = [nodes.get(by_id.get(i)) for i in rx["reactants"]]
        p_ids = [nodes.get(by_id.get(i)) for i in rx["products"]]
        if not all(x in known for x in [*r_ids, *p_ids]):
            continue
        changed |= _put_step(ws, job, rx, r_ids, p_ids)
    job["retro_nodes"] = nodes
    return changed


def _put_step(ws, job: dict, rx: dict, r_ids: list, p_ids: list) -> bool:
    from mepd.web.compose import place, species_structures
    from mepd.web.nanoreactor import _label_of

    existing = ws.find_reaction(job["id"], rx["key"], key=rx["key"])
    names = {sid: ws.structure(sid)["name"] for sid in [*r_ids, *p_ids]}
    label = _label_of(r_ids, p_ids, names)
    ts = rx.get("ts") or {}
    fields = {"label": label, "count": 0, "reverse_count": 0, "delta_e_kcal": rx.get("delta_e_kcal"), "events": [],
              "retro": {"routes": rx.get("routes"), "method": rx.get("method"), "score": rx.get("score"),
                        "reagents": rx.get("reagents") or [], "balanced": rx.get("balanced")},
              "complexes": (existing or {}).get("complexes") or [], "edge": (existing or {}).get("edge")}
    snap = ws.snapshot()
    if rx.get("balanced") and not (fields["edge"] and fields["edge"] in snap["edges"]):
        if len(r_ids) == 1 and len(p_ids) == 1:
            edge = ws.find_edge(r_ids[0], p_ids[0])
            try:
                edge = edge or ws.add_edge(r_ids[0], p_ids[0], reaction=False, origin=_edge_origin(job, rx))
                fields["edge"] = edge["id"]
            except WorkspaceError:
                pass
        else:
            sids = []
            try:
                for side, ids in (("reactant", r_ids), ("product", p_ids)):
                    s = place(species_structures(ws, ids))
                    sids.append(ws.add_structure(
                        s, name=f"{label} [{side}s]", smiles=chem.perceive_smiles(s), optimized=False,
                        role="complex", merge=False,
                        origin={"kind": "job", "job": job["id"], "entry": f"step_{rx['id']}_{side}s",
                                "label": f"retrosynthesis step {side}s", "retro": True})["id"])
                edge = ws.add_edge(sids[0], sids[1], reaction=False, origin=_edge_origin(job, rx))
                fields["complexes"], fields["edge"] = sids, edge["id"]
            except WorkspaceError:
                pass
    if fields["edge"] and ts.get("barrier_kcal") is not None:
        _put_barrier(ws, job, fields["edge"], rx)
    before = json.dumps({k: (existing or {}).get(k) for k in fields}, sort_keys=True, default=str)
    rec = ws.put_reaction(reactants=r_ids, products=p_ids,
                          origin={"kind": "job", "job": job["id"], "index": rx["key"], "key": rx["key"],
                                  "retro": True}, **fields)
    return existing is None or json.dumps({k: rec.get(k) for k in fields}, sort_keys=True, default=str) != before


def _edge_origin(job: dict, rx: dict) -> dict:
    routes = rx.get("routes") or []
    where = f"route {routes[0]}" if len(routes) == 1 else f"routes {', '.join(map(str, routes))}"
    return {"kind": "job", "job": job["id"], "proposed": True, "retro": True, "reaction": rx["key"],
            "headline": f"retrosynthesis step ({where}, {rx.get('method')}): run a TS search to check it"}


def _put_barrier(ws, job: dict, eid: str, rx: dict) -> None:
    ts = rx["ts"]
    with ws._lock:
        edge = ws._data["edges"].get(eid)
        if edge is None:
            return
        o = edge.get("origin") or {}
        if o.get("barrier_kcal") == ts["barrier_kcal"] and not o.get("proposed"):
            return
        approx = "" if ts.get("verified") else "≈ "
        edge["origin"] = {**o, "proposed": False, "barrier_kcal": ts["barrier_kcal"],
                          "barrier_verified": bool(ts.get("verified")), "retro": True,
                          "headline": f"retrosynthesis step, path-searched: ΔE‡ {approx}{ts['barrier_kcal']:.1f} "
                                      f"kcal/mol" + ("" if ts.get("verified") else " (not IRC-verified)")}
        ws._save()
