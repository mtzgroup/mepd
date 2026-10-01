"""AiZynthFinder as a separate program: its own environment, its own data.

AiZynthFinder needs Python < 3.13 and numpy < 2 (its RDKit is built against
numpy 1), so it can't share mepd's environment. `setup()` makes one with uv under the retro data folder
(aizynth-env), installs aizynthfinder there, downloads its public data
(USPTO ONNX models and templates, CC-BY 4.0; ZINC in-stock, MIT; ~790 MB)
and converts the ZINC stock to an InChIKey list mepd's own stock reads too
(stock name "zinc"). `plan()` runs `aizynthcli` on one target and returns
its routes in mepd's route format.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from mepd.retro.data import data_dir

VERSION = "4.4.1"


def env_dir() -> Path:
    return data_dir() / "aizynth-env"


def _bin(name: str) -> Path:
    sub = "Scripts" if sys.platform == "win32" else "bin"
    return env_dir() / sub / name


def model_dir() -> Path:
    return data_dir() / "aizynth-data"


def installed() -> bool:
    return _bin("aizynthcli").exists() and (model_dir() / "config.yml").exists()


def _run(argv, say, **kw) -> None:
    say("$ " + " ".join(map(str, argv)))
    subprocess.run([str(a) for a in argv], check=True, **kw)


def setup(say: Callable[[str], None] = print) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("setting up AiZynthFinder needs uv (https://docs.astral.sh/uv/) to make its environment")
    if not _bin("aizynthcli").exists():
        # --no-config: run from inside a uv project (mepd's own checkout), uv would apply that
        # project's overrides (numpy >= 2) here too. Its RDKit (2023.9) is built against numpy 1
        # and segfaults under numpy 2.
        _run([uv, "venv", "--no-config", "--python", "3.11", env_dir()], say)
        _run([uv, "pip", "install", "--no-config", "--python", _bin("python"), f"aizynthfinder=={VERSION}",
              "numpy<2"], say)
    model_dir().mkdir(parents=True, exist_ok=True)
    if not (model_dir() / "config.yml").exists():
        say("Downloading AiZynthFinder's public data (~790 MB: USPTO models and templates, CC-BY 4.0; "
            "ZINC in-stock, MIT) ...")
        _run([_bin("download_public_data"), model_dir()], say)
    zinc_keys()


def zinc_keys(say: Callable[[str], None] = print) -> Path:
    """The ZINC stock (AiZynthFinder's HDF5) as an InChIKey list for mepd's stock."""
    from mepd.retro.stock import ZINC_KEYS

    out = data_dir() / ZINC_KEYS
    src = model_dir() / "zinc_stock.hdf5"
    if out.exists() or not src.exists():
        return out
    say("Converting the ZINC stock to an InChIKey list ...")
    code = ("import pandas as pd, sys; df = pd.read_hdf(sys.argv[1], 'table'); "
            "open(sys.argv[2], 'w').write('\\n'.join(df['inchi_key'].astype(str)) + '\\n')")
    _run([_bin("python"), "-c", code, src, out.with_suffix(".tmp")], say)
    out.with_suffix(".tmp").replace(out)
    return out


def _routes(trees: list, max_routes: int) -> list[dict]:
    """AiZynthFinder's route dicts -> mepd route records."""
    import math

    from mepd.retro import chem
    from mepd.retro.search import route_record

    def mol(node: dict) -> dict:
        d = {"smiles": chem.canonical(node["smiles"]) or node["smiles"],
             "in_stock": "stock" if node.get("in_stock") else None}
        kids = node.get("children") or []
        if kids:
            rx = kids[0]
            meta = rx.get("metadata") or {}
            p = float(meta.get("policy_probability") or 1.0)
            d["children"] = [{"product": d["smiles"], "score": p, "method": "aizynthfinder",
                              "info": {k: meta[k] for k in ("template_hash", "classification", "policy_name")
                                       if k in meta},
                              "reactants": [mol(c) for c in rx.get("children") or []]}]
        return d

    out = []
    for tree in trees[:max_routes]:
        t = mol(tree)
        cost = 0.0

        def walk(m):
            nonlocal cost
            for rx in m.get("children") or []:
                cost += -math.log(max(rx["score"], 1e-6))
                for c in rx["reactants"]:
                    walk(c)

        walk(t)
        rec = route_record(t, cost)
        if not rec["n_steps"]:
            continue   # the target itself is in stock: not a route (summary.json says so)
        rec["aizynth_scores"] = tree.get("scores")
        out.append(rec)
    return out


def plan(target: str, workdir: Path, *, max_iterations: int = 100, time_limit: float = 120, max_depth: int = 6,
         routes: int = 5, stock_files: Optional[list] = None, say: Callable[[str], None] = print) -> dict:
    if not installed():
        raise RuntimeError("AiZynthFinder is not set up: run `mepd retro setup --aizynthfinder` (~790 MB)")
    import yaml

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((model_dir() / "config.yml").read_text())
    cfg["search"] = {"iteration_limit": int(max_iterations), "time_limit": int(time_limit),
                     "max_transforms": int(max_depth)}
    if stock_files:
        cfg["stock"] = {f"s{k}": str(f) for k, f in enumerate(stock_files)}
    (workdir / "config.yml").write_text(yaml.safe_dump(cfg))
    out = workdir / "trees.json"
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    _run([_bin("aizynthcli"), "--config", workdir / "config.yml", "--smiles", target, "--output", out], say,
         cwd=workdir, env=env)
    trees = json.loads(out.read_text())
    return {"routes": _routes(trees, routes), "n_trees": len(trees)}
