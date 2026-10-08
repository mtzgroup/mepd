"""A nanoreactor job's network (network.json) in Explore.

Species become ordinary nodes (merged with a molecule already there, as
one more conformer); each reaction becomes a workspace reaction (drawn as
a dot joined to its reactants and products, shuttles dashed). A
reaction's optimized subsystem ends are hidden structures (role
"complex") joined by an ordinary edge: selecting the dot selects that
edge, so every pair operation (TS search, channels) runs on exactly the
atoms the reaction needs. A TS the job found itself (--connect) is put on
that edge like a flux-steered expansion's steps.

The import is idempotent and incremental: it runs whenever network.json
changes (after the analysis, the refinement and each TS search) and at
the end of the job.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from mepd.web import chem
from mepd.web.workspace import WorkspaceError


def _read(fp: Path) -> Optional[str]:
    try:
        return Path(fp).read_text()
    except OSError:
        return None


def _sp_key(sp: dict) -> str:
    return sp.get("key") or f"{sp['smiles']}|{sp['charge']}|{sp['multiplicity']}"


def _rx_key(rx: dict, keys: dict) -> str:
    """A reaction's identity, whichever way it is written (as the CLI's reaction_key)."""
    if rx.get("key"):
        return rx["key"]
    a = "+".join(sorted(keys[i] for i in rx["reactants"]))
    b = "+".join(sorted(keys[i] for i in rx["products"]))
    return min(f"{a}>{b}", f"{b}>{a}")


def adopt_nanoreactor(ws, job: dict, *, final: bool = False) -> bool:
    """Bring the job's species and reactions into the workspace: from
    live_network.json (reactions refined while the MD runs, --refine-live)
    and network.json (the run's analysis and final refinement). Species and
    reactions are matched by identity (SMILES/charge/spin; reactant and
    product sets), so the live and final versions of one reaction are one
    workspace reaction. Returns True when the workspace changed."""
    out = Path(job.get("output_dir") or "")
    files = [out / "live_network.json", out / "network.json"]
    stamps = []
    for fp in files:
        try:
            stamps.append(fp.stat().st_mtime)
        except OSError:
            stamps.append(0.0)
    if not any(stamps) or (stamps == list(job.get("nano_mtimes") or []) and not final):
        return False
    datas = []
    for fp, st in zip(files, stamps):
        if not st:
            continue
        try:
            datas.append((fp.name, json.loads(fp.read_text())))
        except ValueError:
            return False   # being written; next time
    job["nano_mtimes"] = stamps
    nodes: dict = dict(job.get("nano_keys") or {})          # species key -> structure id
    refined: dict = dict(job.get("nano_refined_keys") or {})
    changed = False
    for name, data in datas:
        changed |= _adopt_one(ws, job, data, nodes, refined, final=final and name == "network.json")
    job["nano_keys"], job["nano_refined_keys"] = nodes, refined
    return _sync_events(ws, job, out) or changed


def _sync_events(ws, job: dict, out: Path) -> bool:
    """Each of the job's reactions with the MD events that are that
    reaction now. A reaction refined while the MD ran kept the numbers its
    events had then; the live analysis renumbers them as the trajectory
    grows (an H lost and then taken up again became one event), so its card
    played a different reaction. Events carry their reaction's key; runs
    from before that keep their numbers."""
    evs, final = _events(out)
    if final and not any(e.get("key") for e in evs):
        # The final analysis lists its species: the key from them.
        data = _json(out / "network.json") or {}
        keys = {sp["id"]: _sp_key(sp) for sp in data.get("species") or []}
        for e in evs:
            if e.get("reaction") is not None and all(i in keys for i in e.get("reactants", []) + e.get("products", [])):
                e["key"] = _rx_key({"reactants": e["reactants"], "products": e["products"]}, keys)
    if not any(e.get("key") for e in evs):
        return False
    by_key: dict = {}
    for e in evs:
        if e.get("key"):
            by_key.setdefault(e["key"], []).append(e["event"])
    changed = False
    with ws._lock:
        for r in ws._data.get("reactions", {}).values():
            o = r.get("origin") or {}
            if o.get("job") != job["id"] or not o.get("key"):
                continue
            want = by_key.get(o["key"], [])
            if r.get("events") != want:
                r["events"], changed = want, True
        if changed:
            ws._save()
    return changed


def _adopt_one(ws, job: dict, data: dict, nodes: dict, refined: dict, *, final: bool) -> bool:
    refine = (job.get("params") or {}).get("refine", True)
    level = job.get("level")
    known = ws.snapshot()["structures"]
    keys = {sp["id"]: _sp_key(sp) for sp in data.get("species") or []}
    changed = False
    for sp in data.get("species") or []:
        k = keys[sp["id"]]
        have = nodes.get(k) in known
        energy = sp.get("energy")
        if have and (refined.get(k) or energy is None):
            continue
        if energy is not None and sp.get("file"):
            text, opt = _read(Path(sp["file"])), True
        elif (not refine or final) and not have and sp.get("md_file"):
            text, opt, energy = _read(Path(sp["md_file"])), False, None   # as cut from the MD
        else:
            continue
        if not text:
            continue
        (s,) = chem.structures_from_xyz_text(text, sp["charge"], sp["multiplicity"])
        smiles = sp["smiles"] if not str(sp["smiles"]).startswith("?") and "unusual bonding" not in str(sp["smiles"]) \
            else None
        res = ws.add_or_merge(
            s, name=chem.common_name(smiles) or smiles or sp["smiles"] or chem.formula(s), smiles=smiles, energy=energy,
            optimized=opt,
            level=level if opt else None,
            origin={"kind": "job", "job": job["id"], "entry": f"species_{sp['id']}", "label": f"Species {sp['id']}",
                    "nanoreactor": True})
        nodes[k], changed = res["rec"]["id"], True
        if opt:
            refined[k] = True
        known = ws.snapshot()["structures"]
    for rx in data.get("reactions") or []:
        ids = rx["reactants"] + rx["products"]
        if not all(nodes.get(keys.get(i)) in known for i in ids):
            continue
        rkey = _rx_key(rx, keys)
        existing = ws.find_reaction(job["id"], rx["id"], key=rkey)
        events = [inst.get("event") for inst in rx.get("instances") or [] if inst.get("event") is not None]
        c = rx.get("complex") or {}
        if not c.get("reactant") and rx.get("instances"):
            # refined before ends that changed bonds but still differ were kept: take them now
            from mepd.discovery.nanoreactor import relaxed_complex

            c = relaxed_complex(rx["instances"]) or c
        fields = {"label": _label(rx, data), "count": rx.get("count", 0), "reverse_count": rx.get("reverse_count", 0),
                  "delta_e_kcal": rx.get("delta_e_kcal"), "first_fs": rx.get("first_fs"),
                  "complex_delta_e_kcal": c.get("delta_e_kcal"), "complex_error": c.get("error"),
                  "complex_reason": c.get("reason"),
                  "complexes": (existing or {}).get("complexes") or [], "edge": (existing or {}).get("edge"),
                  # For the reaction card: the MD events it was seen in (to replay), and the
                  # energy ladder on one scale, kcal/mol from the separated reactants.
                  "events": events or (existing or {}).get("events") or [],
                  "ladder": _ladder(rx, data), "live": bool(data.get("live"))}
        if c.get("reactant") and not (fields["edge"] and fields["edge"] in ws.snapshot()["edges"]):
            made = _complex_pair(ws, job, rx, c, level, fields["label"])
            if made:
                fields["complexes"], fields["edge"] = made
        if fields["edge"] and (rx.get("ts") or {}).get("barrier_kcal") is not None:
            _put_ts(ws, job, fields["edge"], rx)
        r_ids, p_ids = [nodes[keys[i]] for i in rx["reactants"]], [nodes[keys[i]] for i in rx["products"]]
        if c.get("relaxed") and c.get("reactant"):
            # Its ends changed bonds on optimization but still differ: the reaction is what they are
            # (its molecules from the relaxed ends), the sampled one kept as `relaxed_from`.
            if existing and existing.get("relaxed_from"):
                r_ids, p_ids = existing["reactants"], existing["products"]
            else:
                r_ids, p_ids = _relaxed_species(ws, job, c)
            fields["label"] = c["relaxed"]["label"]
            fields["relaxed_from"] = _label(rx, data)
            fields["relaxed_why"] = c["relaxed"].get("why") or []
        before = json.dumps({k: (existing or {}).get(k) for k in fields}, sort_keys=True, default=str)
        rec = ws.put_reaction(
            reactants=r_ids, products=p_ids,
            origin={"kind": "job", "job": job["id"], "index": rx["id"], "key": rkey, "nanoreactor": True}, **fields)
        if existing is None or json.dumps({k: rec.get(k) for k in fields}, sort_keys=True, default=str) != before:
            changed = True
    return changed


def spawn_ts_searches(manager, job: dict) -> list:
    """With 'Find each reaction's TS': an ordinary TS search per reaction of
    this run (on its subsystem edge), each its own job -- queued, watchable
    live, its barrier landing on the reaction. Smallest subsystems first;
    reactions that already have a search are skipped, so calling this
    again (as more reactions arrive) starts nothing twice. Returns the new
    job ids."""
    import logging

    params = job.get("params") or {}
    if not params.get("connect"):
        return []
    snap = manager.ws.snapshot()
    busy = {e for j in manager.jobs.values() if j["op"] in ("ts", "channels") for e in j["targets"]["edges"]}
    mine = [r for r in snap.get("reactions", {}).values()
            if r.get("origin", {}).get("job") == job["id"] and r.get("edge") in snap["edges"]]
    spawned = list(job.get("spawned") or [])
    room = int(params.get("max_connect") or 20) - len(spawned)
    size = lambda r: snap["structures"].get((r.get("complexes") or [None])[0], {}).get("natoms", 999)
    todo = [r for r in sorted(mine, key=size) if r["edge"] not in busy][:max(0, room)]
    new = []
    for r in todo:
        try:
            created = manager.submit("ts", structure_ids=[], edge_ids=[r["edge"]], params={},
                                     profile=job.get("profile"), label=f"TS: {r['label']}")
        except WorkspaceError as exc:
            logging.getLogger(__name__).warning("could not start the TS search of %s: %s", r["label"], exc)
            continue
        for c in created:
            manager._update(c, parent_job=job["id"])
            new.append(c["id"])
    if new:
        manager._update(job, spawned=spawned + new)
    return new


def _ladder(rx: dict, data: dict) -> Optional[dict]:
    """Separated reactants (0), reactant complex, product complex and
    separated products, kcal/mol, all over the same atoms (the reaction's
    subsystem), so they compare. None where an energy is missing."""
    k = 627.509474
    e = {sp["id"]: sp.get("energy") for sp in data.get("species") or []}
    er = [e.get(i) for i in rx["reactants"]]
    ep = [e.get(i) for i in rx["products"]]
    if any(x is None for x in er):
        return None
    base = sum(er)
    c = rx.get("complex") or {}
    return {"reactants": 0.0,
            "reactant_complex": (c["reactant_energy"] - base) * k if c.get("reactant_energy") is not None else None,
            "product_complex": (c["product_energy"] - base) * k if c.get("product_energy") is not None else None,
            "products": (sum(ep) - base) * k if all(x is not None for x in ep) else None}


def _label_of(reactants: list, products: list, names: dict) -> str:
    """'A + B -> C + B' from structure ids: the species that change first, shuttles last."""
    from collections import Counter

    both = Counter(reactants) & Counter(products)

    def side(ids):
        c = Counter(ids) - both
        parts = [f"{n} {names[i]}" if n > 1 else names[i] for i, n in c.items()]
        parts += [f"{n} {names[i]}" if n > 1 else names[i] for i, n in both.items()]
        return " + ".join(parts)
    return f"{side(reactants)} -> {side(products)}"


def _label(rx: dict, data: dict) -> str:
    return rx.get("label") or f"reaction {rx['id']}"


def _complex_pair(ws, job: dict, rx: dict, c: dict, level, label: str):
    """The reaction's optimized subsystem ends as hidden structures joined
    by an edge (a 'proposed' one: no TS searched yet)."""
    sids = []
    for side in ("reactant", "product"):
        text = _read(Path(c[side]))
        if not text:
            return None
        (s,) = chem.structures_from_xyz_text(text, c["charge"], c["multiplicity"])
        rec = ws.add_structure(
            s, name=f"{label} [{side}s]", smiles=chem.perceive_smiles(s), energy=c.get(f"{side}_energy"),
            optimized=True, level=level, role="complex", merge=False,
            origin={"kind": "job", "job": job["id"], "entry": f"reaction_{rx['id']}_{side}",
                    "label": f"Reaction {rx['id']} {side}s", "nanoreactor": True})
        sids.append(rec["id"])
    try:
        edge = ws.add_edge(sids[0], sids[1], reaction=False, label="", origin={
            "kind": "job", "job": job["id"], "proposed": True, "nanoreactor": True, "reaction": rx["id"],
            "headline": "reaction from the nanoreactor: run a TS search on its subsystem"})
    except WorkspaceError:
        return None
    return sids, edge["id"]


def _relaxed_species(ws, job: dict, c: dict) -> tuple[list, list]:
    """The molecules of a reaction's relaxed ends (complex files), as
    species (merged with known ones): its reactants and products."""
    from mepd.web.compose import _fragments

    out = []
    for side in ("reactant", "product"):
        text = _read(Path(c[side]))
        if not text:
            out.append([])
            continue
        (s,) = chem.structures_from_xyz_text(text, c["charge"], c["multiplicity"])
        out.append([ws.add_or_merge(f, optimized=False, origin={
            "kind": "job", "job": job["id"], "label": f"relaxed {side} end (molecule)", "nanoreactor": True})["rec"]["id"]
                    for f in _fragments(s)])
    return out[0], out[1]


def _put_ts(ws, job: dict, eid: str, rx: dict) -> None:
    ts = rx["ts"]
    with ws._lock:
        edge = ws._data["edges"].get(eid)
        if edge is None:
            return
        o = edge.get("origin") or {}
        if o.get("barrier_kcal") == ts["barrier_kcal"] and not o.get("proposed"):
            return
        edge["origin"] = {"kind": "job", "job": job["id"], "entry": f"reaction_{rx['id']}_irc", "group": "irc",
                          "has_ts": True, "barrier_kcal": ts["barrier_kcal"], "label": ts.get("label"),
                          "nanoreactor": True, "reaction": rx["id"],
                          "headline": "TS + IRC on the reaction's subsystem (nanoreactor)"}
        ws._save()


# --------------------------------------------------------------------------
# Live reactor view: the trajectory (as it grows) and its reaction events.

_SEGMENTS: dict = {}   # path -> (mtime, symbols, frames): a finished segment never changes


def _segment(fp: Path):
    import numpy as np

    from mepd.discovery.nanoreactor import read_xyz_frames

    m = fp.stat().st_mtime
    hit = _SEGMENTS.get(str(fp))
    if hit is None or hit[0] != m:
        syms, xyz, _ = read_xyz_frames(fp)
        hit = _SEGMENTS[str(fp)] = (m, syms, np.round(xyz, 2))
        if len(_SEGMENTS) > 2000:
            _SEGMENTS.pop(next(iter(_SEGMENTS)))
    return hit[1], hit[2]


def _trajectory(out: Path):
    """(symbols, frames) of the reactor so far: finished MD segments, or the
    trajectory a --trajectory run analyzed."""
    import numpy as np

    segs = sorted((out / "md").glob("segment_*.xyz"))
    symbols = _segment(segs[0])[0] if segs else _md_symbols(out / "md")
    if symbols:
        parts = [_segment(fp) for fp in segs]
        frames = [p[1] for p in parts if len(p[1])]
        # The segment xtb is running now (md/xtb.trj, moved to segment_k.xyz
        # when it ends): its complete frames too, so a long segment does not
        # hold the live view at the edge. Dropped if a segment finished while
        # we read (its frames then come from segment_k.xyz next time).
        running = _running_frames(out / "md", len(symbols))
        if running is not None and len(running) and len(sorted((out / "md").glob("segment_*.xyz"))) == len(segs):
            frames.append(running)
        if frames:
            return symbols, np.concatenate(frames)
    data = _json(out / "network.json") or {}
    fp = Path(data.get("trajectory") or "")
    if fp.is_file():
        return _segment(fp)
    return [], np.zeros((0, 0, 3))


def _md_symbols(md: Path) -> list:
    """The reactor's atoms before any MD segment has finished (the first
    segment is running): from the relaxed (or packed) reactor."""
    from mepd.discovery.nanoreactor import read_xyz_frames

    for name in ("reactor.xyz", "packed.xyz"):
        try:
            return list(read_xyz_frames(md / name)[0])
        except (OSError, ValueError, IndexError):
            continue
    return []


def _running_frames(md: Path, n_atoms: int):
    frames, _ = _growing_xyz(md / "xtb.trj", n_atoms)
    return frames


def _growing_xyz(fp: Path, n_atoms: int):
    """(frames rounded to 0.01 A, comment lines) of a multi-frame xyz a
    program is still writing: its complete frames only. (None, []) if none."""
    import numpy as np

    try:
        lines = fp.read_text().splitlines()
    except OSError:
        return None, []
    block = n_atoms + 2
    whole = len(lines) // block
    if not whole:
        return None, []
    out = np.empty((whole, n_atoms, 3))
    try:
        for f in range(whole):
            rows = lines[f * block + 2:(f + 1) * block]
            out[f] = [[float(v) for v in r.split()[1:4]] for r in rows]
    except (ValueError, IndexError):   # a frame being written
        return None, []
    return np.round(out, 2), [lines[f * block + 1] for f in range(whole)]


def _prep(out: Path, max_frames: int = 150):
    """Before the MD: the packed reactor and its relaxation inside the wall
    (xtb's xtbopt.log, or relax.xyz from an engine MD), frame by frame, with
    the energy of each step (kcal/mol from the first) when known. None
    before packing."""
    import re

    import numpy as np

    from mepd.discovery.nanoreactor import read_xyz_frames

    md = out / "md"
    try:
        symbols, packed, _ = read_xyz_frames(md / "packed.xyz")
    except (OSError, ValueError, IndexError):
        return None
    frames, energies = [np.round(packed[0], 2)], [None]
    for name in ("xtbopt.log", "relax.xyz"):
        got, comments = _growing_xyz(md / name, len(symbols))
        if got is not None:
            frames += list(got)
            for c in comments:
                m = re.search(r"energy:\s*(-?\d+\.\d+)", c)
                energies.append(float(m.group(1)) if m else None)
            break
    known = [e for e in energies if e is not None]
    rel = [None if e is None else round((e - known[0]) * 627.509474, 1) for e in energies]
    idx = np.unique(np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int)).tolist()
    sched = _json(md / "schedule.json") or {}
    radius = (sched.get("segments") or [[None, None]])[0][1]
    return {"stage": "relaxed" if (md / "reactor.xyz").exists() else ("relaxing" if len(frames) > 1 else "packed"),
            "symbols": list(symbols), "steps": len(frames) - 1, "radius": radius,
            "frames": [frames[k].reshape(-1).tolist() for k in idx], "step": idx,
            "energy_kcal": [rel[k] for k in idx]}


def _json(fp: Path):
    try:
        return json.loads(Path(fp).read_text())
    except (OSError, ValueError):
        return None


def _events(out: Path) -> tuple[list, bool]:
    """The run's events: the final analysis when there is one (charges from
    partial charges), else the live one (so far)."""
    final = _json(out / "network.json")
    if final and final.get("events") is not None:
        names = {sp["id"]: sp["smiles"] for sp in final.get("species") or []}
        evs = final["events"]
        for e in evs:   # runs from before labels were stored per event
            if "label" not in e:
                from mepd.discovery.nanoreactor import reaction_label

                e["label"] = reaction_label(e["reactants"], e["products"], names)
        return evs, True
    live = _json(out / "live_events.json") or {}
    return live.get("events") or [], False


def reactor_view(out: Path, start: int = 0) -> dict:
    """Frames from raw frame `start` on (subsampled to about 10 fs, or
    coarser for long runs, so the page stays light), the wall radius per
    frame, and every event. Coordinates in Angstrom, 2 decimals."""
    import math

    import numpy as np
    from qcconst.constants import ANGSTROM_TO_BOHR

    out = Path(out)
    symbols, frames = _trajectory(out)
    sched = _json(out / "md" / "schedule.json") or {}
    settings = (_json(out / "network.json") or {}).get("settings") or {}
    dump_fs = float(sched.get("dump_fs") or settings.get("dump_fs") or 2.0)
    total_ps = float(sched.get("time_ps") or settings.get("time_ps") or len(frames) * dump_fs / 1000.0)
    step_fs = max(10.0, total_ps * 1000.0 / 4000.0)       # at most ~4000 frames in the page
    stride = max(1, int(round(step_fs / dump_fs)))
    first = int(math.ceil(max(0, start) / stride)) * stride
    idx = list(range(first, len(frames), stride))
    # Wall radius at each frame's time (the piston schedule).
    segments = sched.get("segments")
    if not segments and settings.get("radius"):   # runs from before schedule.json: rebuild it
        from mepd.discovery.nanoreactor import ReactorSettings, piston_schedule

        known = {k: v for k, v in settings.items() if k in ReactorSettings.__dataclass_fields__}
        try:
            segments = piston_schedule(ReactorSettings(**known))
        except Exception:
            segments = []
    bounds, t = [], 0.0
    for dur, r in segments or []:
        t += float(dur) * 1000.0
        bounds.append((t, float(r)))

    def radius(k):
        tf = (k + 1) * dump_fs
        return next((r for end, r in bounds if tf <= end + 1e-6), bounds[-1][1] if bounds else None)

    events, final = _events(out)
    # A QM/MM reactor: which atoms react (QM) and which never move (frozen),
    # so the view can fade the environment.
    region = {"qm_atoms": sched["qm_atoms"], "frozen_atoms": sched.get("frozen_atoms")} if sched.get("qm_atoms") else {}
    if sched.get("centre_bohr") is not None and len(frames):
        # Its piston wall is around where the QM region started, not the
        # origin: shift the frames so it is (the wall drawn, and the view
        # framed, around the atoms that react).
        frames = frames - np.asarray(sched["centre_bohr"], dtype=float) / ANGSTROM_TO_BOHR
    return {**region, "prep": _prep(out) if not len(frames) else None, "symbols": list(symbols), "dump_fs": dump_fs, "stride": stride, "total_ps": total_ps,
            "n_frames": int(len(frames)), "start": first, "frames": [frames[k].reshape(-1).tolist() for k in idx],
            "radius": [radius(k) for k in idx], "events": events, "final": final}


def event_view(out: Path, k: int, pad_fs: float = 60.0) -> dict:
    """One event at full time resolution, cut to its atoms: frames from a
    little before its reactant frame to a little after its product frame,
    centred on the subsystem, its bond changes in local atom indices, and
    the optimized ends of its reaction (if refined)."""
    import numpy as np

    out = Path(out)
    events, _ = _events(out)
    ev = next((e for e in events if e.get("event") == k), None)
    if ev is None:
        raise KeyError(k)
    symbols, frames = _trajectory(out)
    dump_fs = float((_json(out / "md" / "schedule.json") or {}).get("dump_fs") or 2.0)
    pad = int(round(pad_fs / dump_fs))
    a, b = max(0, ev["reactant_frame"] - pad), min(len(frames) - 1, ev["product_frame"] + pad)
    atoms = list(ev["atoms"])
    local = {x: i for i, x in enumerate(atoms)}
    cut = frames[a:b + 1][:, atoms]
    cut = cut - cut.mean(axis=1, keepdims=True)   # each frame on its own centre: no drift across the view
    network = _json(out / "network.json") or {}
    rx = next((r for r in network.get("reactions") or [] if r["id"] == ev.get("reaction")), None)
    ends = {}
    for side in ("reactant", "product"):
        text = _read((rx or {}).get("complex", {}).get(side) or "") if rx else None
        if text:
            ends[side] = text
    return {"event": k, "label": ev.get("label"), "symbols": [symbols[x] for x in atoms], "first_frame": a,
            "dump_fs": dump_fs, "frames": np.round(cut, 2).reshape(len(cut), -1).tolist(),
            "reactant_frame": ev["reactant_frame"] - a, "product_frame": ev["product_frame"] - a,
            "changes": [{"frame": c["frame"] - a, "atoms": [local[c["atoms"][0]], local[c["atoms"][1]]],
                         "formed": c["formed"]} for c in ev.get("bond_changes") or []
                        if c["atoms"][0] in local and c["atoms"][1] in local],   # runs from before that was filtered
            "reaction": ev.get("reaction"), "optimized": ends}
