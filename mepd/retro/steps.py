"""Routes into mepd: each step as a balanced reaction, its path-search check, and
the network file the web UI adopts.

A template keeps its leaving groups on the precursors (acyl *chloride* +
amine -> amide), so a step's atoms balance only with its byproducts (HCl),
and a proposal often leaves out what it consumes (a reduction's H2, a
hydrolysis's water): chem.balance_step adds both. A step that can't be balanced is shown but not verified.

Verification is `mepd run --reaction "A.B>>P.HCl" --recursive --use-tsopt
--irc --minimize-ends` per step (SLAPMapper maps the atoms, as for any
reaction SMILES), in `<output>/verify/step_<k>`, read back the way the web
UI reads any TS job: a barrier counts as verified only when IRCs connect
start to end.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from mepd.retro import chem


def step_key(step: dict) -> str:
    return ".".join(sorted(step["reactants"])) + ">>" + step["product"]


def balanced(step: dict) -> Optional[tuple[list[str], list[str]]]:
    """(co-reactants, byproducts) that balance the step, None if none do
    (see chem.balance_step)."""
    info = step.get("info") or {}
    if "byproducts" in info:
        return list(info.get("coreactants") or []), list(info["byproducts"])
    try:
        return chem.balance_step(list(step["reactants"]), step["product"])
    except ValueError:
        return None


def byproducts(step: dict) -> Optional[list[str]]:
    b = balanced(step)
    return None if b is None else b[1]


def coreactants(step: dict) -> list[str]:
    b = balanced(step)
    return [] if b is None else b[0]


def reaction_smiles(step: dict) -> Optional[str]:
    """'A.B>>P.HCl' (with any co-reactant: 'K.H2.H2>>P.H2O'), balanced; None
    if it can't be balanced."""
    b = balanced(step)
    if b is None:
        return None
    co, extra = b
    return ".".join([*step["reactants"], *co]) + ">>" + ".".join([step["product"], *extra])


def unique_steps(routes: list[dict]) -> list[dict]:
    """Every distinct step of the routes, first-seen order, with the routes it is in."""
    seen: dict = {}
    for rank, route in enumerate(routes, start=1):
        for st in route["steps"]:
            k = step_key(st)
            if k not in seen:
                seen[k] = {**st, "key": k, "routes": []}
            seen[k]["routes"].append(rank)
    return list(seen.values())


def verify(step: dict, inputs: Optional[str], out: Path, *, workers: int = 1, timeout: float = 6 * 3600,
           extra_flags: tuple = (), say: Callable[[str], None] = print) -> dict:
    """Path search + TS + IRC for one step. Returns {status, barrier_kcal,
    verified, headline, output}; resumable (a finished folder is read back)."""
    rxn = reaction_smiles(step)
    if rxn is None:
        return {"status": "unbalanced", "headline": "atoms don't balance with a known byproduct; not verified"}
    out.mkdir(parents=True, exist_ok=True)
    done = out / "retro_verify.json"
    if done.exists():
        try:
            return json.loads(done.read_text())
        except ValueError:
            pass
    argv = [sys.executable, "-m", "mepd.cli", "run", "--reaction", rxn, "--recursive", "--use-tsopt", "--irc",
            "--minimize-ends", "--output", str(out), *extra_flags]
    if inputs:
        argv += ["--inputs", str(inputs)]
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    say(f"Verifying {rxn} ...")
    with open(out / "run.log", "w") as log:
        try:
            proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    res = {"status": "failed" if code else "done", "reaction": rxn, "output": str(out)}
    try:
        from mepd.web.results import collect_ts

        charge = int(chem.formula([step["product"]]).get("+", 0))
        got = collect_ts(out, charge, 1)
        res.update(barrier_kcal=got.get("barrier_kcal"), verified=bool(got.get("barrier_verified")),
                   headline=got.get("headline"))
    except Exception as exc:
        res.update(barrier_kcal=None, verified=False, headline=f"could not read the path search: {exc}")
    if code:
        res["headline"] = f"path search exited with {code}; " + (res.get("headline") or "")
    done.write_text(json.dumps(res, indent=1))
    return res


def network(target: str, routes: list[dict], steps: list[dict], out: Path, *, live: bool = False) -> dict:
    """network.json for the web UI: species (SMILES, a 3D guess) and one
    reaction per distinct step, reactants -> product (+ byproducts)."""
    from mepd.web.chem import structure_from_smiles

    sp_dir = out / "species"
    sp_dir.mkdir(parents=True, exist_ok=True)
    species, ids = [], {}

    def sid(smiles: str, role: str) -> Optional[int]:
        if smiles in ids:
            return ids[smiles]
        fp = sp_dir / (hashlib.sha1(smiles.encode()).hexdigest()[:16] + ".xyz")
        if not fp.exists():
            try:
                fp.write_text(structure_from_smiles(smiles).to_xyz())
            except Exception:
                return None
        k = len(species)
        q = int(chem.formula([smiles]).get("+", 0))
        species.append({"id": k, "smiles": smiles, "charge": q, "multiplicity": 1, "md_file": str(fp), "role": role})
        ids[smiles] = k
        return k

    sid(target, "target")
    leaves = {l["smiles"]: l.get("in_stock") for r in routes for l in r["leaves"]}
    reactions = []
    for k, st in enumerate(steps):
        co, extra = balanced(st) or ([], [])
        r_ids = [sid(s, "stock" if leaves.get(s) else "intermediate") for s in st["reactants"]] + \
            [sid(s, "coreactant") for s in co]
        p_ids = [sid(s, "target" if s == target else "byproduct") for s in [st["product"], *extra]]
        if None in r_ids or None in p_ids:
            continue
        info = st.get("info") or {}
        v = st.get("verification") or {}
        rx = {"id": k, "key": st["key"], "reactants": r_ids, "products": p_ids, "routes": st["routes"],
              "method": st["method"], "score": st["score"], "balanced": balanced(st) is not None,
              "label": " + ".join([*st["reactants"], *co]) + " -> " + " + ".join([st["product"], *extra]),
              "reagents": info.get("reagents") or [], "delta_e_kcal": info.get("delta_e_kcal")}
        if v.get("barrier_kcal") is not None:
            rx["ts"] = {"barrier_kcal": v["barrier_kcal"], "verified": v.get("verified"), "output": v.get("output"),
                        "label": v.get("headline")}
        reactions.append(rx)
    for s in species:
        s["in_stock"] = leaves.get(s["smiles"])
    data = {"kind": "retro", "live": live, "target": target, "species": species, "reactions": reactions}
    fp = out / ("live_network.json" if live else "network.json")
    tmp = fp.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(fp)
    return data
