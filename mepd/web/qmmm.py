"""QM/MM systems in the web workspace (see mepd/qmmm.py).

A QM/MM system is a fixed list of atoms (a molecule in a solvent shell, an
active site in a protein) with one region: QM atoms, link atoms, the moving
shell and the frozen rest. Every structure with exactly those atoms in that
order belongs to it (`Workspace.qmmm_system_of`); its node is named by the
QM region (capped with its link hydrogens), never split into molecules, and
every calculation on it runs embedded (the job's profile gets a [qmmm]
table, see operations.JobContext.common_flags).

This module: creating/editing systems, the region data the 3D views style
atoms by, and the checks shown next to every QM/MM result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from qcdata import Structure

from mepd.qmmm import QMMMRegion, diagnose, format_indices
from mepd.web import chem
from mepd.web.workspace import Workspace, WorkspaceError


def region_view(region: QMMMRegion, sysrec: Optional[dict] = None) -> dict:
    """What the browser needs to draw and edit a region."""
    return {
        **(sysrec or {}),
        "natoms": region.natoms, "qm_atoms": list(region.qm_atoms), "frozen_atoms": list(region.frozen_atoms),
        "links": [list(x) for x in region.links], "link_ratios": region.link_ratios(),
        "qm_text": format_indices(region.qm_atoms),
        "qm_charge": region.qm_charge, "qm_multiplicity": region.qm_multiplicity, "mm": region.mm,
        "active_radius": region.active_radius, "summary": region.summary(), "name": region.name,
        "embedding": region.embedding,
    }


def _region_args(body: dict, region: Optional[QMMMRegion] = None) -> dict:
    out = {}
    for key in ("qm_charge", "qm_multiplicity", "active_radius", "mm", "embedding"):
        if body.get(key) is not None:
            out[key] = body[key]
        elif region is not None:
            out[key] = getattr(region, key)
    if "active_radius" in body and body["active_radius"] in (None, 0, ""):
        out["active_radius"] = None
    # TIP3P water goes with electrostatic embedding, xTB/GFN-FF with mechanical;
    # AMBER takes either.
    if out.get("mm") == "tip3p":
        out["embedding"] = "electrostatic"
    elif out.get("mm") in ("gfnff", "gfn2", "gfn1"):
        out["embedding"] = "mechanical"
    return out


def check_limits(region: QMMMRegion, limits: Optional[dict]) -> None:
    """The public demo's bounds on a system ({"max_qm", "max_atoms",
    "max_active_radius"}; None: no bounds), so one visitor's system stays a
    few CPU-minutes per calculation."""
    if not limits:
        return
    if len(region.qm_atoms) > limits["max_qm"]:
        raise ValueError(f"the demo allows at most {limits['max_qm']} QM atoms (got {len(region.qm_atoms)})")
    if region.natoms > limits["max_atoms"]:
        raise ValueError(f"the demo allows QM/MM systems of at most {limits['max_atoms']} atoms "
                         f"(got {region.natoms})")
    if region.active_radius is None or region.active_radius > limits["max_active_radius"]:
        raise ValueError(f"the demo needs a moving shell of at most {limits['max_active_radius']} Å "
                         "(the rest of the environment frozen)")


def preview(ws: Workspace, sid: str, body: dict) -> dict:
    """The region `body` describes on structure `sid`, not saved."""
    s = ws.load_structure(sid)
    region = QMMMRegion.build(s, body.get("qm_atoms") or "", **_region_args(body), name=body.get("name") or "")
    return {**region_view(region), "problems": region.check(s) + _level_problems(ws, region)}


def _level_problems(ws: Workspace, region: QMMMRegion) -> list[str]:
    """Electrostatic embedding with no profile in the workspace that can run
    it: say so before the system is built, not when its first job fails."""
    from mepd.qmmm import embedding_problem
    from mepd.web.jobs import _profile_engine

    if region.embedding != "electrostatic":
        return []
    engines = [_profile_engine(ws, n) for n in ws.profile_names()] or ["gxtb"]
    if any(embedding_problem("electrostatic", e) is None for e in engines):
        return []
    return ["Electrostatic embedding needs a QM level that takes point charges (xTB or Psi4), and no profile in "
            "this workspace uses one: add an xTB (GFN2) or Psi4 profile in Settings before running jobs on this "
            "system, or pick a GFN-FF environment (mechanical embedding)."]


def create(ws: Workspace, sid: str, body: dict, limits: Optional[dict] = None) -> dict:
    """Make structure `sid` (and every structure with its atoms) a QM/MM
    system with the region in `body`."""
    s = ws.load_structure(sid)
    rec = ws.structure(sid)
    if rec.get("qmmm"):
        return update(ws, rec["qmmm"], body, limits)
    region = QMMMRegion.build(s, body.get("qm_atoms") or "", **_region_args(body),
                              name=body.get("name") or f"{rec['name']} (QM/MM)")
    check_limits(region, limits)
    sysrec = ws.put_qmmm_system(region, name=region.name)
    retag(ws, sysrec["id"])
    return sysrec


def update(ws: Workspace, sysid: str, body: dict, limits: Optional[dict] = None) -> dict:
    """Change a system's region (QM atoms, charge, moving shell, low level).
    Energies computed with the old region keep their old level key, so they
    are never compared with the new ones."""
    old = ws.qmmm_region(sysid)
    ref = old.reference_structure()
    qm = body.get("qm_atoms") if body.get("qm_atoms") not in (None, "") else old.qm_atoms
    extra = {k: getattr(old, k) for k in ("prmtop", "pdb", "forcefield", "tcin")}
    region = QMMMRegion.build(ref, qm, **_region_args(body, old), name=body.get("name") or old.name, **extra)
    check_limits(region, limits)
    sysrec = ws.put_qmmm_system(region, name=region.name, sysid=sysid)
    retag(ws, sysid)
    return sysrec


def retag(ws: Workspace, sysid: str) -> int:
    """Give every structure of the system its `qmmm` id and QM-region name."""
    region = ws.qmmm_region(sysid)
    label = region.name
    n = 0
    with ws._lock:
        for rec in ws._data["structures"].values():
            if rec.get("natoms") != region.natoms:
                continue
            try:
                s = ws.load_structure(rec["id"])
            except Exception:
                continue
            if ws.qmmm_system_of(s) != sysid:
                continue
            smiles = chem.perceive_smiles(region.model_structure(s))
            rec["qmmm"] = sysid
            rec["smiles"] = smiles
            rec["name"] = f"{smiles or 'QM region'} · {label}" + (" [TS]" if rec.get("role") == "ts" else "")
            rec.pop("members", None)
            if rec.get("role") == "complex":
                rec["role"] = "minimum"
            n += 1
        ws._save()
    return n


def adopt_build(ws: Workspace, job: dict) -> Optional[str]:
    """A finished `qmmm-build` job: its system as a new QM/MM node (to be
    minimized). Returns the structure id."""
    out = Path(job["output_dir"])
    if not (out / "region.json").exists() or not (out / "system.xyz").exists():
        return None
    region = QMMMRegion.open(out / "region.json")
    known = ws.snapshot()["structures"]
    solute = next((known[s]["name"] for s in job["targets"]["structures"] if s in known), "molecule")
    solvent = (job.get("params") or {}).get("solvent", "solvent")
    sysrec = ws.put_qmmm_system(region, name=f"in {solvent}" if solute else region.name)
    (s,) = chem.structures_from_xyz_text((out / "system.xyz").read_text(), job.get("charge"), job.get("multiplicity"))
    res = ws.add_or_merge(s, optimized=False, origin={"kind": "job", "job": job["id"], "entry": "system",
                                                      "label": f"{solute} in {solvent} (QM/MM)"})
    return res["rec"]["id"]


def adopt_protein_build(ws: Workspace, job: dict) -> dict:
    """A finished `qmmm-protein-build` job: its system as a new QM/MM node
    (the protein's force-field PDB goes with the region); with a reaction
    brought along, its other end (and TS) too, the ends joined by an edge.
    Returns {"start", "end", "ts", "edge"} like `adopt_reaction` (the caller
    minimizes the ends and re-optimizes the TS)."""
    out = Path(job["output_dir"])
    if not (out / "region.json").exists() or not (out / "system.xyz").exists():
        return {}
    region = QMMMRegion.open(out / "region.json")
    where = region.name or "in protein"
    ws.put_qmmm_system(region, name=where, base_dir=out)
    origin = {"kind": "job", "job": job["id"]}

    def add(fname, label, role="minimum"):
        fp = out / fname
        if not fp.exists():
            return None
        # The whole system's charge: the protein's plus the species'.
        (s,) = chem.structures_from_xyz_text(fp.read_text(), region.charge, job.get("multiplicity"))
        res = ws.add_or_merge(s, optimized=False, role=role, origin={**origin, "entry": fname.split(".")[0],
                                                                      "label": label})
        return res["rec"]["id"]

    start = add("system.xyz", f"{where} (QM/MM)")
    end = add("product.xyz", f"the other end, {where}")
    ts = add("ts.xyz", f"TS {where} (from the gas phase, to re-optimize)", role="ts")
    edge = None
    if start and end and start != end:
        edge = (ws.find_edge(start, end) or ws.add_edge(start, end, label=where, origin={
            **origin, "headline": "the reaction inside the protein (QM/MM)",
            "gas_edges": [e for e in [(job.get("params") or {}).get("edge")] if e]}))["id"]
    return {"start": start, "end": end, "ts": ts, "edge": edge}


def from_terachem(ws: Workspace, path: str, mm: str = "amber", active_radius: Optional[float] = None) -> dict:
    """A TeraChem QM/MM input on this machine as a QM/MM system + node."""
    from mepd.qmmm_build import from_terachem as convert

    fp = Path(path).expanduser()
    if not fp.is_file():
        raise WorkspaceError(f"{fp} does not exist")
    system, region, qm = convert(fp, mm=mm, active_radius=active_radius)
    region.name = fp.parent.name or "TeraChem system"
    sysrec = ws.put_qmmm_system(region, name=region.name)
    res = ws.add_or_merge(system, optimized=False,
                          origin={"kind": "qmmm", "input": str(fp), "label": f"TeraChem input {fp.name}"})
    return {"system": sysrec, "structure": res["rec"]["id"], "terachem_method": qm}


def from_upload(ws: Workspace, text: str, filename: str, body: dict, limits: Optional[dict] = None) -> dict:
    """An uploaded XYZ/PDB of a whole system with a region on it."""
    import tempfile

    from mepd.cli_qmmm import _load_system

    suffix = ".pdb" if filename.lower().endswith(".pdb") else ".xyz"
    with tempfile.TemporaryDirectory() as tmp:
        fp = Path(tmp) / f"system{suffix}"
        fp.write_text(text)
        s = _load_system(str(fp), body.get("charge"), body.get("multiplicity"))
    region = QMMMRegion.build(s, body.get("qm_atoms") or "", **_region_args(body),
                              name=body.get("name") or Path(filename).stem)
    check_limits(region, limits)
    sysrec = ws.put_qmmm_system(region, name=region.name)
    res = ws.add_or_merge(s, optimized=False, origin={"kind": "qmmm", "input": filename,
                                                      "label": f"uploaded {filename}"})
    return {"system": sysrec, "structure": res["rec"]["id"]}


def check_frames(ws: Workspace, sysid: str, frames: list, energies: Optional[list] = None) -> dict:
    region = ws.qmmm_region(sysid)
    structures = [Structure.from_xyz(f) if isinstance(f, str) else f for f in frames]
    return diagnose(region, structures, energies)


def energy_split(ws: Workspace, jobs: dict, jid: str, entry: str) -> Optional[dict]:
    """The finished `qmmm-inspect` follow-up of this job's entry, if any:
    per-frame QM / environment energies (kcal/mol from the first frame)."""
    runs = sorted((j for j in jobs.values() if j.get("op") == "qmmm-inspect" and j.get("source_job") == jid
                   and (j.get("params") or {}).get("entry") == entry), key=lambda j: j["created"], reverse=True)
    for j in runs:
        fp = Path(j["output_dir"]) / "report.json"
        if j["status"] == "done" and fp.exists():
            rep = json.loads(fp.read_text())
            fr = rep.get("frames") or []
            if not fr or "qm" not in fr[0]:
                continue
            q0, e0 = fr[0]["qm"], fr[0]["environment"]
            return {"job": j["id"], "qm_kcal": [(f["qm"] - q0) * 627.509474 for f in fr],
                    "environment_kcal": [(f["environment"] - e0) * 627.509474 for f in fr],
                    "total_kcal": [f.get("rel_kcal") for f in fr]}
    if runs:
        return {"job": runs[0]["id"], "status": runs[0]["status"]}
    return None


def adopt_reaction(ws: Workspace, job: dict) -> dict:
    """A finished `qmmm-reaction` job: the system, its start and end as QM/MM
    nodes joined by an edge, and the TS (if one was brought along) as a TS
    node. Returns {"start", "end", "ts", "edge"} (structure/edge ids; the
    caller minimizes the ends and re-optimizes the TS)."""
    out = Path(job["output_dir"])
    if not (out / "region.json").exists():
        return {}
    region = QMMMRegion.open(out / "region.json")
    known = ws.snapshot()["structures"]
    names = [known[s]["name"] for s in job["targets"]["structures"] if s in known]
    solvent = (job.get("params") or {}).get("solvent", "solvent")
    ws.put_qmmm_system(region, name=f"in {solvent}")
    origin = {"kind": "job", "job": job["id"]}

    def add(fname, label, role="minimum"):
        fp = out / fname
        if not fp.exists():
            return None
        (s,) = chem.structures_from_xyz_text(fp.read_text(), job.get("charge"), job.get("multiplicity"))
        res = ws.add_or_merge(s, optimized=False, role=role, origin={**origin, "entry": fname.split(".")[0],
                                                                      "label": label})
        return res["rec"]["id"]

    start = add("system.xyz", f"{names[0] if names else 'start'} in {solvent}")
    end = add("product.xyz", f"{names[1] if len(names) > 1 else 'end'} in {solvent}")
    ts = add("ts.xyz", f"TS in {solvent} (from the gas phase, to re-optimize)", role="ts")
    edge = None
    if start and end and start != end:
        edge = ws.find_edge(start, end) or ws.add_edge(start, end, label=f"in {solvent}", origin={
            **origin, "headline": "the gas-phase reaction in explicit solvent (QM/MM)",
            "gas_edges": list(job["targets"].get("edges") or []),
            "gas_structures": list(job["targets"].get("structures") or [])})
        edge = edge["id"]
    return {"start": start, "end": end, "ts": ts, "edge": edge}


def adopt_embed(ws: Workspace, job: dict) -> dict:
    """A finished `qmmm-embed` job: the embedded structure as a node of the
    host's QM/MM system (a TS stays a TS), joined to the host by an edge."""
    out = Path(job["output_dir"])
    fp = out / "embedded_0.xyz"
    if not fp.exists():
        return {}
    known = ws.snapshot()["structures"]
    recs = [known[s] for s in job["targets"]["structures"] if s in known]
    host = next((r for r in recs if r.get("qmmm")), None)
    gas = next((r for r in recs if not r.get("qmmm")), None)
    role = "ts" if gas is not None and gas.get("role") == "ts" else "minimum"
    (s,) = chem.structures_from_xyz_text(fp.read_text(), job.get("charge"), job.get("multiplicity"))
    if host is not None:
        s = s.model_copy(update={"charge": host["charge"], "multiplicity": host["multiplicity"]})
    res = ws.add_or_merge(s, optimized=False, role=role, origin={
        "kind": "job", "job": job["id"], "entry": "embedded_0",
        "label": f"{gas['name'] if gas else 'structure'} put into the system"})
    sid = res["rec"]["id"]
    edge = None
    if host is not None and role == "minimum" and sid != host["id"]:
        edge = (ws.find_edge(host["id"], sid) or ws.add_edge(host["id"], sid, origin={
            "kind": "job", "job": job["id"], "headline": "embedded from the gas phase: run a TS search"}))["id"]
    return {"structure": sid, "role": role, "edge": edge}


# ------------------------------------------------------------- barriers
HARTREE_KCAL = 627.509474


def _first_energy(fp: Path) -> Optional[float]:
    try:
        return float(fp.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _gas_barrier(ws: Workspace, jobs: dict, origin: dict) -> Optional[tuple]:
    """(barrier, verified) of the gas-phase edge this QM/MM edge came from."""
    gas_edges = set(origin.get("gas_edges") or [])
    pair = set(origin.get("gas_structures") or [])
    if not gas_edges and not pair:
        return None
    verified, unverified = [], []
    for j in jobs.values():
        t = j.get("targets") or {}
        if j.get("status") != "done" or j.get("qmmm") or not (set(t.get("edges") or []) & gas_edges
                                                             or (pair and set(t.get("structures") or []) == pair)):
            continue
        b = (j.get("summary") or {}).get("barrier_kcal")
        if b is not None and j.get("op") in ("ts", "channels"):
            (verified if (j.get("summary") or {}).get("barrier_verified", True) else unverified).append(b)
    if verified:
        return min(verified), True
    if unverified:
        return min(unverified), False
    return None


def _irc_ends(ws: Workspace, sysid: str, out: Path) -> Optional[dict]:
    """{QM-region key: energy} of the two ends of a TS job's IRC (its own
    energies, at the TS's level), or None (no IRC, or both ends the same)."""
    try:
        irc = Structure.open_multi(str(out / "irc.xyz"))
        energies = [float(x) for x in (out / "irc.energies").read_text().split()]
    except (OSError, ValueError, Exception):
        return None
    if len(irc) < 2 or len(energies) != len(irc):
        return None
    region = ws.qmmm_region(sysid)
    ends = {chem.canonical_key(chem.perceive_smiles(region.model_structure(s)) or ""): e
            for s, e in ((irc[0], energies[0]), (irc[-1], energies[-1]))}
    return ends if len(ends) == 2 else None


def attach_ts(ws: Workspace, job: dict, jobs: dict) -> list[str]:
    """A finished TS optimization (with IRC) in a QM/MM system: on every
    QM/MM edge of that system whose two ends its IRC connects (by their QM
    regions), record the TS and its IRC's end energies, then set the
    edge's barrier from them (`edge_barrier`). Returns the edges updated."""
    sysid = job.get("qmmm")
    out = Path(job.get("output_dir") or "")
    e_ts = _first_energy(out / "ts.energies")
    if not sysid or e_ts is None or not (out / "irc.xyz").exists():
        return []
    ends = _irc_ends(ws, sysid, out)
    if ends is None:
        return []
    keys = set(ends)
    snap = ws.snapshot()
    updated = []
    with ws._lock:
        for eid, e in ws._data["edges"].items():
            a, b = (snap["structures"].get(e["source"]), snap["structures"].get(e["target"]))
            if not a or not b or a.get("qmmm") != sysid or b.get("qmmm") != sysid:
                continue
            if {chem.canonical_key(a.get("smiles") or ""), chem.canonical_key(b.get("smiles") or "")} != keys:
                continue
            origin = e.setdefault("origin", {})
            old = origin.get("qmmm_ts") or {}
            if old.get("energy") is not None and old.get("level") == (job.get("level") or {}).get("key") \
                    and old["energy"] <= e_ts and old.get("job") != job["id"]:
                continue        # a lower TS between these two is already known
            origin["qmmm_ts"] = {"job": job["id"], "energy": e_ts, "level": (job.get("level") or {}).get("key"),
                                 "source_energy": ends[chem.canonical_key(a.get("smiles") or "")],
                                 "target_energy": ends[chem.canonical_key(b.get("smiles") or "")]}
            updated.append(eid)
        ws._save()
    for eid in updated:
        edge_barrier(ws, jobs, eid)
    return updated


def edge_barrier(ws: Workspace, jobs: dict, eid: str) -> Optional[float]:
    """The barrier of a QM/MM edge with a known TS: TS energy minus its own
    IRC's start end (not the species' stored energies, which may come from
    another structure or level), plus the reverse barrier, the reaction
    energy and the gas-phase barrier for comparison, in the edge's origin
    (what Explore shows)."""
    snap = ws.snapshot()
    e = snap["edges"].get(eid) or {}
    t = (e.get("origin") or {}).get("qmmm_ts")
    if t and t.get("source_energy") is None:
        # Recorded before the IRC's ends were kept: read them from the TS job.
        job = jobs.get(t.get("job")) or {}
        ends = _irc_ends(ws, job["qmmm"], Path(job.get("output_dir") or "")) if job.get("qmmm") else None
        a, b = snap["structures"].get(e["source"]) or {}, snap["structures"].get(e["target"]) or {}
        if ends:
            ka, kb = chem.canonical_key(a.get("smiles") or ""), chem.canonical_key(b.get("smiles") or "")
            if ka in ends and kb in ends:
                t = {**t, "source_energy": ends[ka], "target_energy": ends[kb]}
    with ws._lock:
        e = ws._data["edges"].get(eid)
        if e is None:
            return None
        origin = e.setdefault("origin", {})
        if not origin.get("qmmm_ts"):
            return None
        if t.get("source_energy") is None:
            origin.pop("barrier_kcal", None)
            origin["headline"] = "TS found in the solvent; its IRC's ends are not known"
            ws._save()
            return None
        origin["qmmm_ts"] = t
        gas = _gas_barrier(ws, jobs, origin)
        gas_text = "" if gas is None else f" (gas phase {'' if gas[1] else '≈'}{gas[0]:.1f})"
        fwd = (t["energy"] - t["source_energy"]) * HARTREE_KCAL
        rev = (t["energy"] - t["target_energy"]) * HARTREE_KCAL
        rxn = (t["target_energy"] - t["source_energy"]) * HARTREE_KCAL
        origin.pop("ends_minimized", None)
        origin.update(barrier_kcal=fwd, job=t["job"], reverse_barrier_kcal=rev, reaction_kcal=rxn,
                      gas_barrier_kcal=gas and gas[0],
                      headline=f"in solvent: ΔE‡ {fwd:.1f}{gas_text}, reverse {rev:.1f}, ΔE {rxn:+.1f} kcal/mol, "
                               "from the TS's IRC ends (TS re-optimized in the solvent)")
        ws._save()
        return fwd


def refresh_edges_of(ws: Workspace, jobs: dict, sids) -> None:
    """Ends re-minimized: recompute the barriers of their QM/MM edges."""
    sids = set(sids)
    for eid, e in list(ws.snapshot()["edges"].items()):
        if (e.get("origin") or {}).get("qmmm_ts") and (e["source"] in sids or e["target"] in sids):
            edge_barrier(ws, jobs, eid)
