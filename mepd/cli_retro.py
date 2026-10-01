"""`mepd retro`: retrosynthetic routes to a target, from a stock of building blocks.

    mepd retro plan "CC(=O)Nc1ccc(OCc2ccccc2)cc1"                     # templates, seconds
    mepd retro plan TARGET --method reactiont5                        # local language model
    mepd retro plan TARGET --method local-llm -o model=qwen3:8b       # LLM on a local server
    mepd retro plan TARGET --method aizynthfinder                     # after `mepd retro setup --aizynthfinder`
    mepd retro plan TARGET -i gxtb.toml --verify top                  # path search + TS + IRC per step

Writes routes.json (every route, cheapest first), summary.json (settings,
statistics, methods and their citations), network.json (species and one
reaction per distinct step, which the web UI adds to Explore) and, while
searching, live_network.json. See mepd/retro/__init__.py for the methods.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import typer

retro_app = typer.Typer(help="Retrosynthesis: routes to a target from purchasable building blocks.",
                        no_args_is_help=True)


def _options(values: List[str]) -> dict:
    out = {}
    for v in values or []:
        if "=" not in v:
            raise typer.BadParameter(f"--option takes key=value (got {v!r})")
        k, val = v.split("=", 1)
        k = k.strip().replace("-", "_")
        low = val.strip().lower()
        if low in ("true", "false"):
            out[k] = low == "true"
        else:
            try:
                out[k] = int(val) if val.strip().lstrip("-").isdigit() else float(val)
            except ValueError:
                out[k] = val
    return out


def _target_smiles(target: str) -> str:
    from mepd.retro import chem

    fp = Path(target)
    if fp.exists():
        from mepd.web.chem import perceive_smiles, structures_from_xyz_text

        (s,) = structures_from_xyz_text(fp.read_text())[:1]
        smi = perceive_smiles(s)
        if not smi:
            raise typer.BadParameter(f"could not read a molecule from {fp}")
        target = smi
    smi = chem.canonical(target)
    if smi is None:
        raise typer.BadParameter(f"not a valid SMILES: {target!r}")
    if "." in smi:
        raise typer.BadParameter("the target must be one molecule")
    return smi


def _default_stock() -> list[str]:
    from mepd.retro.data import data_dir
    from mepd.retro.stock import ZINC_KEYS

    return ["zinc"] if (data_dir() / ZINC_KEYS).exists() else ["paroutes-n1", "paroutes-n5"]


def _route_line(r: dict) -> str:
    return " | ".join(f"{'.'.join(s['reactants'])} >> {s['product']}" for s in reversed(r["steps"]))


@retro_app.command("plan")
def plan(
    target: str = typer.Argument(..., help="Target molecule: SMILES or an xyz file."),
    method: str = typer.Option("templates", "--method", "-m",
                               help="Single-step method: templates | reactiont5 | local-llm | aizynthfinder."),
    option: List[str] = typer.Option([], "--option", "-o", help="Method option key=value (repeatable), e.g. "
                                     "model=qwen3:8b, roundtrip=false, filter_cutoff=0.1."),
    stock: List[str] = typer.Option([], "--stock", "-s", help="Building blocks: paroutes-n1, paroutes-n5, zinc, or a "
                                    "file of SMILES/InChIKeys (repeatable). Default: zinc if set up, else both "
                                    "PaRoutes sets."),
    max_heavy: int = typer.Option(2, "--max-heavy", help="Also count any molecule with at most this many heavy atoms "
                                  "as available (H2, CO, ethylene...: the stocks list common reagents themselves); "
                                  "0 turns it off."),
    max_depth: int = typer.Option(6, "--max-depth", help="Most steps from a building block to the target."),
    iterations: int = typer.Option(100, "--iterations", help="Search budget: molecules expanded."),
    time_limit: float = typer.Option(120.0, "--time-limit", help="Search budget: seconds."),
    width: int = typer.Option(10, "--width", help="Proposals kept per molecule."),
    routes: int = typer.Option(5, "--routes", "-n", help="Routes to report."),
    value: str = typer.Option("sa", "--value", help="Cost estimate of an unsolved molecule: sa (synthetic "
                              "accessibility) or zero (Retro*-0)."),
    stop_when_solved: bool = typer.Option(False, "--stop-when-solved", help="Stop at the first solved route."),
    verify: str = typer.Option("none", "--verify", help="Check route steps by path search + TS + IRC (at "
                               "the level of -i): none | top (the best route's steps) | all."),
    inputs: Optional[str] = typer.Option(None, "--inputs", "-i", help="RunInputs TOML (level of theory) for "
                                         "--verify."),
    workers: int = typer.Option(4, "--workers", help="Parallel processes for optimizations and verification."),
    output: Optional[Path] = typer.Option(None, "--output", help="Output folder (default retro_<method>)."),
    charge: Optional[int] = typer.Option(None, "--charge", help="The target's charge (checked against its SMILES)."),
    multiplicity: Optional[int] = typer.Option(None, "--multiplicity", help="The target's spin multiplicity; "
                                               "routes are for closed-shell molecules (1)."),
):
    """Find routes from building blocks to TARGET."""
    from mepd.retro import REFERENCES, steps as steps_mod
    from mepd.retro.proposers import PROPOSERS, make
    from mepd.retro.search import RetroSearch, best_partial, extract_routes
    from mepd.retro.stock import load

    if method not in PROPOSERS:
        raise typer.BadParameter(f"--method: choose from {', '.join(PROPOSERS)}")
    if verify not in ("none", "top", "all"):
        raise typer.BadParameter("--verify: none, top or all")
    smi = _target_smiles(target)
    from mepd.retro import chem as _chem

    if charge is not None and int(_chem.formula([smi]).get("+", 0)) != charge:
        raise typer.BadParameter(f"--charge {charge} disagrees with the target's SMILES {smi}")
    if multiplicity not in (None, 1):
        raise typer.BadParameter("retrosynthesis plans routes to closed-shell molecules (multiplicity 1)")
    out = Path(output or f"retro_{method}")
    out.mkdir(parents=True, exist_ok=True)
    say = typer.echo
    opts = _options(option)
    if verify != "none" and not inputs:
        say("--verify without -i: the path searches run at mepd's default level of theory.")

    stock_names = list(stock) or _default_stock()
    st = load(stock_names, max_heavy=max_heavy, say=say)
    say(f"Target {smi}; stock: {st.describe()}; method: {PROPOSERS[method].label}")
    t0 = time.time()
    stats: dict = {}
    if PROPOSERS[method].multistep:
        from mepd.retro import aizynth
        from mepd.retro.stock import BUILTIN, _keys_file
        from mepd.retro.data import path as data_path

        files = []
        for name in stock_names:
            if name == "zinc":
                files.append(str(aizynth.model_dir() / "zinc_stock.hdf5"))
            elif name in BUILTIN:
                files.append(str(_keys_file(data_path(BUILTIN[name], say=say))))
            else:
                files.append(str(Path(name).expanduser().resolve()))
        res = aizynth.plan(smi, out / "aizynthfinder", max_iterations=iterations, time_limit=time_limit,
                           max_depth=max_depth, routes=routes, stock_files=files, say=say)
        found, partial, solved = res["routes"], None, any(r["solved"] for r in res["routes"])
        stats = {"trees": res["n_trees"]}
    else:
        proposer = make(method, say=say, **opts)
        search = RetroSearch(proposer, st, max_depth=max_depth, expansion_width=width, value=value)
        last = [0.0]

        def on_iteration(it, root):
            if time.time() - last[0] < 2.0:
                return
            last[0] = time.time()
            now = extract_routes(root, routes)
            shown = now or ([best_partial(root)] if root.children else [])
            shown = [r for r in shown if r]
            steps_mod.network(smi, shown, steps_mod.unique_steps(shown), out, live=True)
            say(f"  iteration {it}: {len(now)} solved route(s), {search.expansions} expansions")

        result = search.run(smi, max_iterations=iterations, time_limit=time_limit, routes=routes,
                            stop_when_solved=stop_when_solved, on_iteration=on_iteration)
        found, partial, solved = result.routes, result.best_partial, result.solved
        stats = {"iterations": result.iterations, "expansions": result.expansions, "molecules": result.molecules}
    stats["seconds"] = round(time.time() - t0, 2)

    shown = found or ([partial] if partial else [])
    uniq = steps_mod.unique_steps(shown)
    if verify != "none" and shown:
        todo = [s for s in uniq if verify == "all" or 1 in s["routes"]]
        say(f"Verifying {len(todo)} step(s) with path search + TS + IRC ...")
        from mepd.cli_common import _fork_map

        def check(s):
            return steps_mod.verify(s, inputs, out / "verify" / f"step_{uniq.index(s)}", say=say)

        for s, v in zip(todo, _fork_map(check, todo, max(1, min(workers, len(todo))))):
            s["verification"] = v
        # each route's steps carry their verification too
        by_key = {s["key"]: s.get("verification") for s in uniq}
        for r in shown:
            for s in r["steps"]:
                if by_key.get(steps_mod.step_key(s)):
                    s["verification"] = by_key[steps_mod.step_key(s)]
            bars = [s.get("verification", {}).get("barrier_kcal") for s in r["steps"]]
            if bars and all(b is not None for b in bars):
                r["highest_barrier_kcal"] = max(bars)
                r["all_verified"] = all(s["verification"].get("verified") for s in r["steps"])
        # Routes whose every step was checked first, lowest highest barrier first; the rest keep their order.
        checked = sorted([r for r in found if r.get("highest_barrier_kcal") is not None],
                         key=lambda r: r["highest_barrier_kcal"])
        found[:] = checked + [r for r in found if r.get("highest_barrier_kcal") is None]
    steps_mod.network(smi, shown, uniq, out)

    used = {"search", *PROPOSERS[method].references} - ({"search"} if PROPOSERS[method].multistep else set())
    if verify != "none":
        used.add("verification")
    if value == "sa" and not PROPOSERS[method].multistep:
        used.add("synthetic_accessibility")
    summary = {"kind": "retro", "target": smi, "method": method, "options": opts,
               "stock": st.describe(), "stock_sources": stock_names, "max_heavy": max_heavy, "max_depth": max_depth,
               "solved": solved, "n_routes": len(found), "target_in_stock": st.why(smi), "verify": verify, "stats": stats,
               "methods": {k: REFERENCES[k] for k in sorted(used)}}
    (out / "routes.json").write_text(json.dumps({"target": smi, "routes": found, "best_partial": partial}, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    if found:
        say(f"{len(found)} route(s) to {smi} in {stats['seconds']} s:")
        for k, r in enumerate(found, start=1):
            bar = f", highest barrier {r['highest_barrier_kcal']:.1f} kcal/mol" if r.get("highest_barrier_kcal") \
                is not None else ""
            say(f"  {k}. {r['n_steps']} step(s), score {r['score']:.3g}{bar}: {_route_line(r)}")
    else:
        say(f"No route reaches the stock in this budget ({stats['seconds']} s)."
            + (f" Closest: {_route_line(partial)}" if partial else ""))
    say(f"Written to {out}/")


@retro_app.command("setup")
def setup(
    aizynthfinder: bool = typer.Option(False, "--aizynthfinder", help="Make AiZynthFinder's own environment and "
                                       "download its public data and ZINC stock (~790 MB)."),
    data: bool = typer.Option(True, "--data/--no-data", help="Download the template method's data (USPTO "
                              "templates, policy and filter networks, PaRoutes stocks; ~112 MB, CC-BY 4.0)."),
):
    """Download the data the retrosynthesis methods use (done on first use otherwise)."""
    from mepd.retro import data as data_mod

    if data:
        for name in data_mod.FILES:
            fp = data_mod.path(name, say=typer.echo)
            typer.echo(f"  {fp}")
    if aizynthfinder:
        from mepd.retro import aizynth

        aizynth.setup(say=typer.echo)
        typer.echo(f"AiZynthFinder {aizynth.VERSION} is set up in {aizynth.env_dir()}; stock 'zinc' is available.")


@retro_app.command("methods")
def methods():
    """List the methods and whether each can run here."""
    from mepd.retro.proposers import PROPOSERS

    for p in PROPOSERS.values():
        problem = p.problem()
        typer.echo(f"{p.name:14s} {'ok' if problem is None else 'unavailable'}  {p.summary}")
        if problem:
            typer.echo(f"{'':14s}   {problem}")
