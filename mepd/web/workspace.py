"""On-disk workspace backing the web UI.

Layout::

    <root>/
      workspace.json          structures, edges, node positions
      structures/<id>.xyz     one geometry per library structure (qcdata xyz)
      profiles/<name>.toml    RunInputs TOML profiles
      jobs/<id>/job.json      job record (see mepd.web.jobs)
      jobs/<id>/output/       the mepd command's --output directory

The library *is* the reaction graph: every structure is a graph node, and
edges are pairs the user (or an imported result) declared connected. Jobs
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
    ) -> dict:
        """`optimized`: the geometry is a known minimum at the level of theory
        that produced it (a sampled minimum, a relaxed conformer). SMILES
        embeddings, uploads and path/IRC frames are not, so jobs minimize
        them first (see operations._build_ts)."""
        sid = new_id("s_")
        smiles = smiles or chem.perceive_smiles(structure)
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
            self._save()
        return rec

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
                         validation: Optional[dict] = None) -> Optional[dict]:
        """Swap in an optimized geometry, keeping the structure's identity
        (id, name, edges, position)."""
        with self._lock:
            rec = self._data["structures"].get(sid)
            if rec is None:
                return None  # deleted while optimizing
            structure = structure.model_copy(update={"charge": rec["charge"], "multiplicity": rec["multiplicity"]})
            structure.save(str(self.structures_dir / f"{sid}.xyz"))
            new_smiles = chem.perceive_smiles(structure)
            if new_smiles and rec.get("smiles") and chem.canonical_key(new_smiles) != chem.canonical_key(rec["smiles"]):
                # The optimization changed connectivity (e.g. a proton moved):
                # keep the result but say so rather than hiding it.
                rec["status_error"] = f"connectivity changed on optimization: {rec['smiles']} -> {new_smiles}"
            else:
                rec["status_error"] = None
            rec.update(smiles=new_smiles or rec.get("smiles"), energy=energy, optimized=True,
                       level=level, status="ready", validation=validation)
            if validation and validation.get("is_minimum") is False:
                # Converged, but on a saddle point even after the rescue push:
                # keep the geometry, never treat it as a minimum.
                rec["optimized"] = False
                rec["status"] = "not_minimum"
                rec["status_error"] = f"not a minimum: {validation.get('validation')}"
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


def remove_tree(fp: Path) -> None:
    shutil.rmtree(fp, ignore_errors=True)
