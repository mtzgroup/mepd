"""On-disk workspace backing the web UI.

Layout::

    <root>/
      workspace.json          structures, edges, node positions
      structures/<id>.xyz     the structure's representative geometry (qcdata xyz):
                              its lowest-energy conformer
      structures/<id>/<c>.xyz every conformer of that molecule
      profiles/<name>.toml    RunInputs TOML profiles
      jobs/<id>/job.json      job record (see mepd.web.jobs)
      jobs/<id>/output/       the mepd command's --output directory

The library *is* the reaction graph: every structure is a graph node, and
edges are pairs the user (or an imported result) declared connected. A node
is one molecule (connectivity + stereo, charge, spin): every conformer found
for it -- by RDKit or CREST, an optimization, an imported result, a network
expansion -- is kept in the node's `conformers`, and the lowest-energy one
(at the workspace level of theory when it has energies there) represents
it. An edge may name the conformer to use at either end; otherwise jobs use
the representative. Transition states are never merged. Jobs
point at structures/edges; results can be imported back as new structures
and edges, so exploration results grow the same graph the user draws.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from qcdata import Structure

from mepd.web import chem

HARTREE_TO_KCAL = 627.509474


def new_id(prefix: str) -> str:
    return f"{prefix}{secrets.token_hex(4)}"


def _atomic_write(fp: Path, text: str) -> None:
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, fp)


class WorkspaceError(ValueError):
    pass


def is_ts(rec: dict) -> bool:
    """TS structures are never geometry-minimized. Records from before the
    `role` field existed are recognized by the " [TS]" name the importer gave them."""
    return rec.get("role") == "ts" or (rec.get("role") is None and str(rec.get("name", "")).endswith("[TS]"))


# RunInputs fields that define the potential energy surface. Two profiles
# that agree on these give comparable geometries and energies, whatever
# their path-minimizer or optimizer settings.
LEVEL_FIELDS = ("engine_name", "program", "program_kwds", "gxtb_engine_kwds", "ase_engine_kwds")


def level_key(profile_text: Optional[str]) -> str:
    """Fingerprint of a profile's level of theory (see LEVEL_FIELDS). `None`
    means mepd's built-in defaults."""
    import hashlib

    import tomli

    data = tomli.loads(profile_text) if profile_text else {}
    picked = {k: data.get(k) for k in LEVEL_FIELDS if data.get(k) not in (None, "", {})}
    return hashlib.sha1(json.dumps(picked, sort_keys=True, default=str).encode()).hexdigest()[:10]


def path_summary(profile_text: Optional[str]) -> dict:
    """What a profile's path search will actually do, in one line plus
    warnings -- mainly so settings that look effective but are not (GSM's
    `nnodes` while it is seeded from the geodesic path) are visible before a
    job runs. Defaults mirror mepd.inputs (GIInputs.nimages=10, GSM nnodes=9,
    seed_with_geodesic_interpolation=True); `validate_profile_text` reports
    the authoritative resolved values."""
    import tomli

    try:
        data = tomli.loads(profile_text) if profile_text else {}
    except Exception:
        return {"text": "invalid TOML", "warnings": []}
    method = str(data.get("path_min_method", "NEB")).upper()
    pmi = data.get("path_min_inputs") or {}
    nimages = int((data.get("gi_inputs") or {}).get("nimages", 10))
    warnings = []
    if method == "GSM":
        seeded = bool(pmi.get("seed_with_geodesic_interpolation", True))
        nnodes = int(pmi.get("nnodes", 9))
        if seeded:
            text = f"GSM · {nimages}-node string, seeded from the geodesic path"
            if "nnodes" in pmi and nnodes != nimages:
                warnings.append(
                    f"path_min_inputs.nnodes = {nnodes} is ignored: GSM is seeded from the geodesic path, "
                    f"so the string has gi_inputs.nimages = {nimages} nodes. Set [gi_inputs] nimages = {nnodes}, "
                    "or seed_with_geodesic_interpolation = false to let GSM grow to nnodes itself.")
        else:
            text = f"GSM · grows its own string to {nnodes} nodes"
        if bool(pmi.get("early_stop_on_minima", False)):
            text += " · early stop on"
            warnings.append("early_stop_on_minima = true: GSM may be stopped at the first intermediate that looks "
                            "stable, before the string has converged. mepd's default is false.")
        else:
            text += " · early stop off"
    else:
        text = f"{method} · {nimages} images"
    return {"text": text, "warnings": warnings}


def level_label(profile_text: Optional[str]) -> str:
    """Short human description, e.g. 'gxtb' or 'chemcloud/terachem ub3lyp/3-21g'."""
    import tomli

    data = tomli.loads(profile_text) if profile_text else {}
    parts = [str(data.get("engine_name", "gxtb"))]
    if data.get("engine_name") in ("chemcloud", "qccompute"):
        parts[0] += f"/{data.get('program', 'xtb')}"
    model = (data.get("program_kwds") or {}).get("model") if isinstance(data.get("program_kwds"), dict) else None
    if model:
        parts.append("/".join(str(model.get(k)) for k in ("method", "basis") if model.get(k)))
    return " ".join(parts)


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.structures_dir = self.root / "structures"
        self.profiles_dir = self.root / "profiles"
        self.jobs_dir = self.root / "jobs"
        for d in (self.root, self.structures_dir, self.profiles_dir, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._fp = self.root / "workspace.json"
        if self._fp.exists():
            self._data = json.loads(self._fp.read_text())
        else:
            self._data = {"version": 1, "structures": {}, "edges": {}, "positions": {}}
            self._save()
        self._data.setdefault("positions", {})
        self._data.setdefault("level_profile", None)
        # Hessian check applied when structures are optimized on entry.
        self._data.setdefault("validate_minima", True)
        if any([self._ensure_conformers(rec) for rec in self._data["structures"].values()]):
            self._save()   # an older workspace: each structure becomes its own first conformer

    # ------------------------------------------------------------------ io
    def _save(self) -> None:
        _atomic_write(self._fp, json.dumps(self._data, indent=1))

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))

    # ---------------------------------------------------------- structures
    def structure(self, sid: str) -> dict:
        try:
            return self._data["structures"][sid]
        except KeyError:
            raise WorkspaceError(f"unknown structure {sid!r}") from None

    def structure_path(self, sid: str) -> Path:
        self.structure(sid)
        return self.structures_dir / f"{sid}.xyz"

    def load_structure(self, sid: str) -> Structure:
        rec = self.structure(sid)
        s = Structure.open(str(self.structure_path(sid)))
        return s.model_copy(update={"charge": rec["charge"], "multiplicity": rec["multiplicity"]})

    def add_structure(
        self,
        structure: Structure,
        *,
        name: Optional[str] = None,
        origin: dict,
        energy: Optional[float] = None,
        smiles: Optional[str] = None,
        optimized: bool = False,
        level: Optional[dict] = None,
        status: str = "ready",
        role: str = "minimum",
        validation: Optional[dict] = None,
        merge: bool = True,
    ) -> dict:
        """`optimized`: the geometry is a known minimum at the level of theory
        that produced it (a sampled minimum, a relaxed conformer). SMILES
        embeddings, uploads and path/IRC frames are not, so jobs minimize
        them first (see operations._build_ts). A minimum of a molecule already
        in the graph becomes a conformer of that node (see `add_or_merge`)."""
        return self.add_or_merge(structure, name=name, origin=origin, energy=energy, smiles=smiles,
                                 optimized=optimized, level=level, status=status, role=role,
                                 validation=validation, merge=merge)["rec"]

    def add_or_merge(self, structure: Structure, *, name: Optional[str] = None, origin: dict,
                     energy: Optional[float] = None, smiles: Optional[str] = None, optimized: bool = False,
                     level: Optional[dict] = None, status: str = "ready", role: str = "minimum",
                     validation: Optional[dict] = None, merge: bool = True) -> dict:
        """Add a structure, or -- a minimum of a molecule already in the graph
        -- a conformer of that molecule's node. Returns {"rec", "conformer"
        (its id), "merged" (added to an existing node), "duplicate" (that
        conformer was already there)}."""
        smiles = smiles or chem.perceive_smiles(structure)
        with self._lock:
            match = self.find_molecule(smiles, int(structure.charge), int(structure.multiplicity)) \
                if merge and role == "minimum" else None
            if match is not None:
                cid, duplicate = self.add_conformer(match["id"], structure, energy=energy, level=level,
                                                    optimized=optimized, validation=validation, origin=origin)
                return {"rec": match, "conformer": cid, "merged": True, "duplicate": duplicate}
        sid = new_id("s_")
        rec = {
            "id": sid,
            "name": name or smiles or chem.formula(structure),
            "smiles": smiles,
            "formula": chem.formula(structure),
            "natoms": len(structure.symbols),
            "charge": int(structure.charge),
            "multiplicity": int(structure.multiplicity),
            "energy": energy,
            "optimized": bool(optimized),
            # {"profile", "key", "label"} of the level the geometry/energy
            # belong to; None = not at any QM level (force field / as given).
            "level": level,
            "status": status,  # ready | optimizing | opt_failed
            # "ts": a saddle point -- never minimized (that would destroy it).
            "role": role,
            "validation": validation,  # Hessian check {is_minimum, min_frequency, rescued, validation}
            "origin": origin,
            "created": time.time(),
        }
        with self._lock:
            structure.save(str(self.structures_dir / f"{sid}.xyz"))
            self._data["structures"][sid] = rec
            self._ensure_conformers(rec)
            self._save()
        return {"rec": rec, "conformer": rec["conformer"], "merged": False, "duplicate": False}

    # ---------------------------------------------------------- conformers
    def conformer_path(self, sid: str, cid: Optional[str] = None) -> Path:
        rec = self.structure(sid)
        cid = cid or rec.get("conformer")
        if not any(c["id"] == cid for c in rec.get("conformers", [])):
            raise WorkspaceError(f"{rec['name']} has no conformer {cid!r}")
        return self.structures_dir / sid / f"{cid}.xyz"

    def _ensure_conformers(self, rec: dict) -> bool:
        """Give a structure its conformer list (its own geometry as the first
        conformer) if it has none yet. True when something changed."""
        if rec.get("conformers"):
            return False
        src = self.structures_dir / f"{rec['id']}.xyz"
        (self.structures_dir / rec["id"]).mkdir(exist_ok=True)
        if src.exists():
            shutil.copyfile(src, self.structures_dir / rec["id"] / "c0.xyz")
        rec["conformers"] = [{"id": "c0", "energy": rec.get("energy"), "level": rec.get("level"),
                              "optimized": bool(rec.get("optimized")), "validation": rec.get("validation"),
                              "origin": rec.get("origin"), "created": rec.get("created", time.time())}]
        rec["conformer"] = "c0"
        return True

    def find_molecule(self, smiles: Optional[str], charge: int, multiplicity: int) -> Optional[dict]:
        """The node (a minimum, not a TS) of the molecule `smiles`, if any."""
        if not smiles:
            return None
        key = chem.canonical_key(smiles)
        for rec in self._data["structures"].values():
            if (rec.get("role") != "ts" and rec.get("smiles") and rec["charge"] == charge
                    and rec["multiplicity"] == multiplicity and not rec.get("reacted")
                    and chem.canonical_key(rec["smiles"]) == key):
                return rec
        return None

    def _same_conformer(self, rec: dict, conf: dict, structure: Structure, energy, level) -> bool:
        """Is this geometry one we already have? Same level and energy within
        0.05 kcal/mol, or (no energies) within 0.1 Å RMSD after alignment."""
        if (conf.get("level") or {}).get("key") != (level or {}).get("key"):
            return False
        if energy is not None and conf.get("energy") is not None:
            return abs(conf["energy"] - energy) * HARTREE_TO_KCAL < 0.05
        if energy is None and conf.get("energy") is None:
            try:
                other = Structure.open(str(self.structures_dir / rec["id"] / f"{conf['id']}.xyz"))
                return _aligned_rmsd(other.geometry, structure.geometry) < 0.1 / 0.529177
            except Exception:
                return False
        return False

    def add_conformer(self, sid: str, structure: Structure, *, energy: Optional[float] = None,
                      level: Optional[dict] = None, optimized: bool = False,
                      validation: Optional[dict] = None, origin: Optional[dict] = None) -> tuple[str, bool]:
        """Add a geometry of `sid`'s molecule; returns (conformer id,
        whether it was already there)."""
        with self._lock:
            rec = self.structure(sid)
            self._ensure_conformers(rec)
            if list(structure.symbols) and len(structure.symbols) != rec["natoms"]:
                raise WorkspaceError(f"a conformer of {rec['name']} needs {rec['natoms']} atoms")
            structure = structure.model_copy(update={"charge": rec["charge"], "multiplicity": rec["multiplicity"]})
            for conf in rec["conformers"]:
                if self._same_conformer(rec, conf, structure, energy, level):
                    return conf["id"], True
            cid = new_id("c_")
            structure.save(str(self.structures_dir / sid / f"{cid}.xyz"))
            rec["conformers"].append({"id": cid, "energy": energy, "level": level, "optimized": bool(optimized),
                                      "validation": validation, "origin": origin, "created": time.time()})
            self._pick_representative(rec)
            self._save()
            return cid, False

    def delete_conformer(self, sid: str, cid: str) -> dict:
        with self._lock:
            rec = self.structure(sid)
            if len(rec.get("conformers", [])) <= 1:
                raise WorkspaceError("a structure keeps at least one conformer (delete the structure instead)")
            self.conformer_path(sid, cid)   # exists?
            rec["conformers"] = [c for c in rec["conformers"] if c["id"] != cid]
            (self.structures_dir / sid / f"{cid}.xyz").unlink(missing_ok=True)
            if rec["conformer"] == cid:
                self._set_representative(rec, rec["conformers"][0])
            self._pick_representative(rec)
            for e in self._data["edges"].values():
                if (e.get("conformers") or {}).get(sid) == cid:
                    e["conformers"].pop(sid)
            self._save()
            return rec

    def _set_representative(self, rec: dict, conf: dict) -> None:
        shutil.copyfile(self.structures_dir / rec["id"] / f"{conf['id']}.xyz",
                        self.structures_dir / f"{rec['id']}.xyz")
        rec.update(conformer=conf["id"], energy=conf.get("energy"), level=conf.get("level"),
                   optimized=bool(conf.get("optimized")), validation=conf.get("validation"))
        if rec.get("status") != "optimizing":
            bad = (conf.get("validation") or {}).get("is_minimum") is False
            rec["status"] = "not_minimum" if bad else "ready"
            if not bad and not rec.get("reacted"):
                rec["status_error"] = None

    def _pick_representative(self, rec: dict) -> None:
        """Represent the node by its lowest-energy conformer: among those at
        the workspace level of theory if any have energies there, else at the
        current representative's level. Energies from different levels are
        never compared; saddle points (failed Hessian check) never win."""
        eligible = [c for c in rec.get("conformers", []) if c.get("energy") is not None
                    and (c.get("validation") or {}).get("is_minimum") is not False]
        if not eligible:
            return
        try:
            ws_key = self.level_of(self.level_profile)["key"]
        except Exception:
            ws_key = None
        current = next((c for c in rec["conformers"] if c["id"] == rec.get("conformer")), None)
        by_level = lambda key: [c for c in eligible if (c.get("level") or {}).get("key") == key]  # noqa: E731
        pool = by_level(ws_key) or (by_level((current.get("level") or {}).get("key")) if current else []) \
            or ([] if current and current.get("energy") is not None else eligible)
        if not pool:
            return
        best = min(pool, key=lambda c: c["energy"])
        if best["id"] != rec.get("conformer"):
            self._set_representative(rec, best)

    def refresh_representatives(self) -> None:
        """Re-pick every node's representative (e.g. after the workspace level
        of theory changed)."""
        with self._lock:
            for rec in self._data["structures"].values():
                self._pick_representative(rec)
            self._save()

    def structure_view(self, sid: str, conformer: Optional[str] = None) -> dict:
        """The structure record as a job sees it: with `conformer` (not the
        representative), that conformer's energy, level and flags, and its
        id under "conformer_id" (JobContext copies that geometry)."""
        rec = self.structure(sid)
        if not conformer or conformer == rec.get("conformer"):
            return rec
        conf = next((c for c in rec.get("conformers", []) if c["id"] == conformer), None)
        if conf is None:
            raise WorkspaceError(f"{rec['name']} has no conformer {conformer!r}")
        view = dict(rec)
        view.update(energy=conf.get("energy"), level=conf.get("level"), optimized=bool(conf.get("optimized")),
                    validation=conf.get("validation"), conformer_id=conformer)
        return view

    def merge_duplicates(self) -> dict:
        """Fold nodes that are the same molecule (e.g. from before conformers
        were merged) into the oldest one: their conformers join it, their
        edges move to it (an edge that would join the node to itself -- a
        conformer change -- is dropped)."""
        with self._lock:
            groups: dict[tuple, list[dict]] = {}
            for rec in sorted(self._data["structures"].values(), key=lambda r: r.get("created", 0)):
                if rec.get("role") == "ts" or not rec.get("smiles") or rec.get("reacted"):
                    continue
                key = (chem.canonical_key(rec["smiles"]), rec["charge"], rec["multiplicity"])
                groups.setdefault(key, []).append(rec)
            merged = 0
            for keeper, *dups in groups.values():
                self._ensure_conformers(keeper)
                for dup in dups:
                    self._ensure_conformers(dup)
                    remap = {}
                    for conf in dup["conformers"]:
                        s = Structure.open(str(self.structures_dir / dup["id"] / f"{conf['id']}.xyz"))
                        remap[conf["id"]], _ = self.add_conformer(
                            keeper["id"], s, energy=conf.get("energy"), level=conf.get("level"),
                            optimized=conf.get("optimized", False), validation=conf.get("validation"),
                            origin=conf.get("origin"))
                    for eid, e in list(self._data["edges"].items()):
                        if dup["id"] not in (e["source"], e["target"]):
                            continue
                        sel = e.get("conformers") or {}
                        if dup["id"] in sel:
                            sel[keeper["id"]] = remap.get(sel.pop(dup["id"]))
                        e["source"] = keeper["id"] if e["source"] == dup["id"] else e["source"]
                        e["target"] = keeper["id"] if e["target"] == dup["id"] else e["target"]
                        twin = next((o for oid, o in self._data["edges"].items() if oid != eid
                                     and {o["source"], o["target"]} == {e["source"], e["target"]}), None)
                        if e["source"] == e["target"] or twin is not None:
                            del self._data["edges"][eid]
                    del self._data["structures"][dup["id"]]
                    self._data["positions"].pop(dup["id"], None)
                    (self.structures_dir / f"{dup['id']}.xyz").unlink(missing_ok=True)
                    shutil.rmtree(self.structures_dir / dup["id"], ignore_errors=True)
                    merged += 1
            self._save()
            return {"merged": merged}

    def update_structure(self, sid: str, **fields: Any) -> dict:
        allowed = {"name", "charge", "multiplicity", "notes"}
        with self._lock:
            rec = self.structure(sid)
            for k, v in fields.items():
                if k in allowed and v is not None:
                    rec[k] = int(v) if k in ("charge", "multiplicity") else v
            self._save()
            return rec

    def delete_structure(self, sid: str) -> None:
        with self._lock:
            self.structure(sid)
            del self._data["structures"][sid]
            self._data["positions"].pop(sid, None)
            for eid in [e for e, rec in self._data["edges"].items() if sid in (rec["source"], rec["target"])]:
                del self._data["edges"][eid]
            (self.structures_dir / f"{sid}.xyz").unlink(missing_ok=True)
            shutil.rmtree(self.structures_dir / sid, ignore_errors=True)
            self._save()

    # --------------------------------------------------------------- edges
    def edge(self, eid: str) -> dict:
        try:
            return self._data["edges"][eid]
        except KeyError:
            raise WorkspaceError(f"unknown edge {eid!r}") from None

    def find_edge(self, a: str, b: str) -> Optional[dict]:
        for rec in self._data["edges"].values():
            if (rec["source"], rec["target"]) in ((a, b), (b, a)):
                return rec
        return None

    def add_edge(self, source: str, target: str, *, label: str = "", origin: Optional[dict] = None) -> dict:
        if source == target:
            raise WorkspaceError("an edge needs two different structures")
        with self._lock:
            s, t = self.structure(source), self.structure(target)
            if s["natoms"] != t["natoms"]:
                raise WorkspaceError(
                    f"{s['name']} has {s['natoms']} atoms but {t['name']} has {t['natoms']}: "
                    "a reaction edge must conserve atoms"
                )
            existing = self.find_edge(source, target)
            if existing is not None:
                return existing
            eid = new_id("e_")
            rec = {
                "id": eid,
                "source": source,
                "target": target,
                "label": label,
                "origin": origin or {"kind": "manual"},
                "created": time.time(),
            }
            self._data["edges"][eid] = rec
            self._save()
            return rec

    def update_edge(self, eid: str, **fields: Any) -> dict:
        with self._lock:
            rec = self.edge(eid)
            if fields.get("label") is not None:
                rec["label"] = fields["label"]
            if fields.get("reverse"):
                rec["source"], rec["target"] = rec["target"], rec["source"]
            for sid, cid in (fields.get("conformers") or {}).items():
                if sid not in (rec["source"], rec["target"]):
                    raise WorkspaceError(f"{sid!r} is not an end of this edge")
                sel = rec.setdefault("conformers", {})
                if cid:
                    self.conformer_path(sid, cid)   # exists?
                    sel[sid] = cid
                else:
                    sel.pop(sid, None)              # back to the lowest-energy conformer
            self._save()
            return rec

    def delete_edge(self, eid: str) -> None:
        with self._lock:
            self.edge(eid)
            del self._data["edges"][eid]
            self._save()

    def delete_many(self, structure_ids: list[str], edge_ids: list[str]) -> dict:
        """Remove several structures (with their edges) and edges in one save."""
        with self._lock:
            edges = self._data["edges"]
            gone_s = [sid for sid in structure_ids if sid in self._data["structures"]]
            gone_e = {eid for eid in edge_ids if eid in edges}
            gone_e |= {eid for eid, rec in edges.items() if rec["source"] in gone_s or rec["target"] in gone_s}
            for eid in gone_e:
                del edges[eid]
            for sid in gone_s:
                del self._data["structures"][sid]
                self._data["positions"].pop(sid, None)
                (self.structures_dir / f"{sid}.xyz").unlink(missing_ok=True)
                shutil.rmtree(self.structures_dir / sid, ignore_errors=True)
            self._save()
        return {"structures": gone_s, "edges": sorted(gone_e)}

    def set_status(self, sids: list[str], status: str, error: Optional[str] = None) -> None:
        with self._lock:
            for sid in sids:
                rec = self._data["structures"].get(sid)
                if rec is not None:
                    rec["status"] = status
                    rec["status_error"] = error
            self._save()

    def replace_geometry(self, sid: str, structure: Structure, *, energy: Optional[float], level: dict,
                         validation: Optional[dict] = None, conformer: Optional[str] = None) -> Optional[dict]:
        """Swap in an optimized geometry, keeping the structure's identity
        (id, name, edges, position): of `conformer`, or of the representative
        one. The node is then represented by whichever conformer is lowest."""
        with self._lock:
            rec = self._data["structures"].get(sid)
            if rec is None:
                return None  # deleted while optimizing
            self._ensure_conformers(rec)
            structure = structure.model_copy(update={"charge": rec["charge"], "multiplicity": rec["multiplicity"]})
            cid = conformer or rec["conformer"]
            conf = next((c for c in rec["conformers"] if c["id"] == cid), None)
            if conf is None:
                return rec  # that conformer was deleted meanwhile
            structure.save(str(self.structures_dir / sid / f"{cid}.xyz"))
            conf.update(energy=energy, level=level, optimized=not (validation and validation.get("is_minimum") is False),
                        validation=validation)
            twin = next((c for c in rec["conformers"] if c["id"] != cid
                         and self._same_conformer(rec, c, structure, energy, level)), None)
            if twin is not None:
                # It relaxed onto a conformer we already had: keep one.
                rec["conformers"] = [c for c in rec["conformers"] if c["id"] != cid]
                (self.structures_dir / sid / f"{cid}.xyz").unlink(missing_ok=True)
                for e in self._data["edges"].values():
                    if (e.get("conformers") or {}).get(sid) == cid:
                        e["conformers"][sid] = twin["id"]
                if rec["conformer"] == cid:
                    self._set_representative(rec, twin)
                self._pick_representative(rec)
                self._save()
                return rec
            if cid != rec["conformer"]:
                self._pick_representative(rec)
                self._save()
                return rec
            structure.save(str(self.structures_dir / f"{sid}.xyz"))
            new_smiles = chem.perceive_smiles(structure)
            old_smiles = rec.get("smiles")
            if new_smiles and old_smiles and chem.canonical_key(new_smiles) != chem.canonical_key(old_smiles):
                # The minimization changed connectivity (a proton moved, or the
                # input reacted downhill, e.g. OH- + CH3Br -> CH3OH + Br-): keep
                # the result, rename a structure still carrying its automatic
                # name so the graph shows what it now is, and flag it.
                rec["status_error"] = (f"reacted while being minimized: {old_smiles} -> {new_smiles}. "
                                       "This is now a different species.")
                rec["reacted"] = {"from": old_smiles, "to": new_smiles}
                origin_input = (rec.get("origin") or {}).get("input")
                if rec.get("name") in (old_smiles, origin_input):
                    rec["name"] = new_smiles
            else:
                rec["status_error"] = None
                rec.pop("reacted", None)
            rec.update(smiles=new_smiles or rec.get("smiles"), energy=energy, optimized=True,
                       level=level, status="ready", validation=validation)
            if validation and validation.get("is_minimum") is False:
                # Converged, but on a saddle point even after the rescue push:
                # keep the geometry, never treat it as a minimum.
                rec["optimized"] = False
                rec["status"] = "not_minimum"
                rec["status_error"] = f"not a minimum: {validation.get('validation')}"
            self._pick_representative(rec)
            self._save()
            return rec

    # ------------------------------------------------------ level of theory
    @property
    def level_profile(self) -> Optional[str]:
        name = self._data.get("level_profile")
        if name and name in self.profile_names():
            return name
        return "default" if "default" in self.profile_names() else None

    @property
    def validate_minima(self) -> bool:
        return bool(self._data.get("validate_minima", True))

    def set_validate_minima(self, value: bool) -> None:
        with self._lock:
            self._data["validate_minima"] = bool(value)
            self._save()

    def set_level_profile(self, name: Optional[str]) -> None:
        if name is not None and name not in self.profile_names():
            raise WorkspaceError(f"unknown profile {name!r}")
        with self._lock:
            self._data["level_profile"] = name
            self._save()
        self.refresh_representatives()   # a node's lowest conformer is judged at the new level

    def level_of(self, profile: Optional[str]) -> dict:
        text = self.read_profile(profile) if profile else None
        return {"profile": profile, "key": level_key(text), "label": level_label(text)}

    def path_summaries(self) -> dict[str, dict]:
        out = {"": path_summary(None)}
        for name in self.profile_names():
            try:
                out[name] = path_summary(self.read_profile(name))
            except Exception:
                continue
        return out

    def levels(self) -> dict[str, dict]:
        """Current level fingerprint of every profile (plus built-in defaults
        under ""), so the UI can tell which structures are off-level."""
        out = {"": self.level_of(None)}
        for name in self.profile_names():
            try:
                out[name] = self.level_of(name)
            except Exception:
                continue
        return out

    # -------------------------------------------------------------- design
    @property
    def design(self) -> Optional[dict]:
        """The Design tab's molecule: {molblock, name, charge, multiplicity,
        source, energy, level, rev} (energy/level only while it is exactly
        the geometry minimized at that level)."""
        return self._data.get("design")

    def set_design(self, design: Optional[dict]) -> dict:
        with self._lock:
            old = self._data.get("design") or {}
            if design is not None:
                design = {**design, "rev": int(old.get("rev") or 0) + 1}
            self._data["design"] = design
            self._save()
            return design

    # ----------------------------------------------------------- positions
    def set_positions(self, positions: dict[str, dict]) -> None:
        with self._lock:
            for sid, pos in positions.items():
                if sid in self._data["structures"]:
                    self._data["positions"][sid] = {"x": float(pos["x"]), "y": float(pos["y"])}
            self._save()

    # ------------------------------------------------------------ profiles
    def profile_names(self) -> list[str]:
        return sorted(p.stem for p in self.profiles_dir.glob("*.toml"))

    def profile_path(self, name: str) -> Path:
        if not name or "/" in name or name.startswith("."):
            raise WorkspaceError(f"invalid profile name {name!r}")
        return self.profiles_dir / f"{name}.toml"

    def read_profile(self, name: str) -> str:
        fp = self.profile_path(name)
        if not fp.exists():
            raise WorkspaceError(f"unknown profile {name!r}")
        return fp.read_text()

    def write_profile(self, name: str, text: str) -> None:
        import tomli

        try:
            tomli.loads(text)
        except Exception as exc:
            raise WorkspaceError(f"not valid TOML: {exc}") from None
        _atomic_write(self.profile_path(name), text)

    def delete_profile(self, name: str) -> None:
        self.profile_path(name).unlink(missing_ok=True)

    def ensure_default_profile(self) -> None:
        """Seed `profiles/default.toml` from `mepd init` the first time a
        workspace is opened. Run as a subprocess: constructing RunInputs
        builds an engine and flips process-global node settings, neither of
        which belongs in the server process."""
        fp = self.profiles_dir / "default.toml"
        if fp.exists() or self.profile_names():
            return
        try:
            subprocess.run(
                [sys.executable, "-m", "mepd.cli", "init", "--output", str(fp)],
                check=True, capture_output=True, timeout=300,
            )
        except Exception:
            fp.write_text('engine_name = "gxtb"\npath_min_method = "NEB"\n')


def validate_profile_text(text: str, timeout: float = 120.0) -> dict:
    """Check a profile by actually building RunInputs from it (engine
    included) in a throwaway subprocess. Returns {ok, message}."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fp = Path(tmp) / "profile.toml"
        fp.write_text(text)
        code = (
            "import sys\n"
            "from mepd.inputs import RunInputs\n"
            "ri = RunInputs.open(sys.argv[1])\n"
            "pmi = ri.path_min_inputs\n"
            "print(f'engine={ri.engine_name} program={ri.program} engine_class={type(ri.engine).__name__}')\n"
            "line = f'path_min_method={ri.path_min_method} gi_inputs.nimages={ri.gi_inputs.nimages}'\n"
            "if str(ri.path_min_method).upper() == 'GSM':\n"
            "    seeded = bool(getattr(pmi, 'seed_with_geodesic_interpolation', True))\n"
            "    line += f' nnodes={getattr(pmi, \"nnodes\", None)} seed_with_geodesic_interpolation={seeded}'\n"
            "    line += (f' -> string length {ri.gi_inputs.nimages} (seeded; nnodes ignored)' if seeded\n"
            "             else f' -> string grows to {getattr(pmi, \"nnodes\", None)}')\n"
            "print(line)\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, str(fp)], capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "message": f"validation timed out after {timeout:.0f}s"}
    if proc.returncode == 0:
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith(("engine=", "path_min_method="))]
        extra = path_summary(text)["warnings"]
        return {"ok": True, "message": "\n".join(lines + [f"warning: {w}" for w in extra]) or "ok"}
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
    return {"ok": False, "message": "\n".join(tail)}


def find_duplicate(ws: Workspace, smiles: Optional[str], energy: Optional[float],
                    level: Optional[dict], job_id: str) -> Optional[dict]:
    """Same connectivity, same level of theory and the same energy to within
    0.05 kcal/mol -> treat as the structure already in the library. Energies
    from different levels are never compared; when the level is unknown
    (an imported external output) only structures from that same job count."""
    if not smiles:
        return None
    for rec in ws.snapshot()["structures"].values():
        if rec.get("smiles") != smiles:
            continue
        if level:
            if (rec.get("level") or {}).get("key") != level.get("key"):
                continue
        elif (rec.get("origin") or {}).get("job") != job_id:
            continue
        if energy is not None and rec.get("energy") is not None:
            if abs(rec["energy"] - energy) * HARTREE_TO_KCAL < 0.05:
                return rec
    return None


def _aligned_rmsd(a, b) -> float:
    """RMSD of two geometries (same atom order) after the best rigid overlay."""
    import numpy as np

    a = np.asarray(a, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1, 3)
    if a.shape != b.shape:
        return float("inf")
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rot - b) ** 2, axis=1))))


def remove_tree(fp: Path) -> None:
    shutil.rmtree(fp, ignore_errors=True)
