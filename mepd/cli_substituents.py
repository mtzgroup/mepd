"""`mepd substituents`: swap hydrogens of a finished reaction for
functional groups and see how its barriers (and its channel selectivity)
respond (mepd/substituents.py).

Reads a `mepd run`, `mepd ts` or `mepd channels` output folder. For every
site x group x channel it writes the substituted reactant end, TS and
product end (<output>/<channel>/<site>_<group>.xyz, energies alongside) and
summary.json: barrier shifts, a Hammett-style slope per site, and
insights, including groups that change which channel wins.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

from qcconst.constants import HARTREE_TO_KCAL_PER_MOL as HARTREE_TO_KCAL

FAST_WARNING = ("Fast estimate: only the new group is relaxed, the rest stays at the parent's geometries. "
                "Re-optimize to confirm large shifts.")
LARGE_SHIFT = 8.0
SCALE_NOTE = "σp is a para-aryl scale: elsewhere it is a rough donor/acceptor ranking, and sterics mix in."


def substituents(
    source: Path = typer.Argument(..., help="A finished `mepd run`, `mepd ts` or `mepd channels` output folder."),
    groups: Optional[List[str]] = typer.Option(
        None, "--group", "-g", help="Group to try (repeatable; names as in Design, e.g. methyl, nitro). "
        "Default: NH2, OH, OMe, Me, F, Cl, CF3, CN, NO2."),
    sites: Optional[List[int]] = typer.Option(
        None, "--site", "-s", help="Only these atoms (0-based): a hydrogen, or the heavy atom whose hydrogens to replace. "
        "Default: every hydrogen that is not transferred, one per symmetry-equivalent set."),
    mode: str = typer.Option("fast", "--mode", help="fast: relax only the new group. reoptimize: TS search + IRC and "
                             "a minimized reactant for every variant (and the parent)."),
    max_sites: int = typer.Option(8, "--max-sites", help="At most this many sites (closest to the reacting atoms first)."),
    workers: int = typer.Option(0, "--workers", help="Variants computed at once (0: up to 4)."),
    temperature: float = typer.Option(298.15, "--temperature", "-T", help="Temperature (K) for half-lives."),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="RunInputs TOML: the level the source ran at."),
    charge: int = typer.Option(0, "--charge"),
    multiplicity: int = typer.Option(1, "--multiplicity"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Default: <source>/substituents."),
) -> None:
    """How swapping hydrogens for functional groups shifts this reaction's barriers."""
    from mepd import substituents as X
    from mepd.cli_common import _open_run_inputs
    from mepd.cli_force import load_channels

    source = Path(source)
    if mode not in ("fast", "reoptimize"):
        raise typer.BadParameter("--mode must be fast or reoptimize")
    known = X.group_smiles()
    groups = list(dict.fromkeys(groups or X.DEFAULT_GROUPS))
    bad = [g for g in groups if g not in known]
    if bad:
        raise typer.BadParameter(f"unknown group(s) {', '.join(bad)}; known: {', '.join(known)}")
    symbols, channels, kind = load_channels(source, charge, multiplicity, purpose="substitute")
    lead = min(channels, key=lambda c: c.barrier_kcal)
    found = X.find_sites(symbols, lead.reactant, lead.product, only=set(sites) if sites else None)
    if not found:
        raise typer.BadParameter("no hydrogen to replace (every H takes part in the reaction, or --site matched none)")
    found = _nearest_to_center(symbols, lead, found)[:max_sites]
    output = Path(output) if output is not None else source / "substituents"
    output.mkdir(parents=True, exist_ok=True)
    run_inputs = _open_run_inputs(inputs)
    engine = run_inputs.engine
    width = workers or min(4, os.cpu_count() or 1)   # modest by default: a laptop overheats under sustained full load

    typer.echo(f"{len(found)} site(s) x {len(groups)} group(s) x {len(channels)} channel(s), {mode}...")
    parents = {c.id: _parent(c, symbols, engine, run_inputs, mode, charge, multiplicity) for c in channels}
    tasks = [(c, s, g) for c in channels for s in found for g in groups]

    def run(task):
        c, s, g = task
        try:
            return _variant(c, s, g, symbols, engine, run_inputs, mode, parents[c.id], output, charge, multiplicity)
        except Exception as exc:
            return {"channel": c.id, "site": s.anchor, "h": s.h, "group": g, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=width) as pool:
        rows = list(pool.map(run, tasks))
    for r in rows:
        r["label"] = f"{X.SHORT.get(r['group'], r['group'])} on {symbols[r['site']]}{r['site']}"
        r["sigma_p"] = X.SIGMA_P.get(r["group"])
    result = _summarize(symbols, channels, found, groups, rows, parents, mode)
    summary = {"source": str(source), "kind": kind, "mode": mode, "temperature": temperature, "groups": groups,
               "group_labels": {g: X.SHORT.get(g, g) for g in groups},
               "sites": [{"h": s.h, "anchor": s.anchor, "equivalent": s.equivalent, "label": s.label(symbols)}
                         for s in found],
               "channels": [{"id": c.id, "label": c.label, "kind": c.kind, "barrier_kcal": c.barrier_kcal,
                             "parent_barrier": parents[c.id]["barrier"]} for c in channels],
               "variants": rows, **result, "references": ["hammett-sigma"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    for ins in summary["insights"]:
        typer.echo(f"[{ins['level']}] {ins['text']}")
    typer.echo(f"Wrote {output / 'summary.json'}")


def _json_default(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _nearest_to_center(symbols, channel, sites):
    """Sites ordered by distance of their anchor from the atoms whose bonds
    change (the substituent effects that matter most come first)."""
    from mepd.substituents import adjacency

    a_r, a_p = adjacency(symbols, channel.reactant), adjacency(symbols, channel.product)
    center = [i for i in range(len(symbols)) if not np.array_equal(a_r[i], a_p[i])]
    if not center:
        return sites
    c = np.asarray(channel.ts)
    return sorted(sites, key=lambda s: min(np.linalg.norm(c[s.anchor] - c[k]) for k in center))


def _node(symbols, coords, charge, multiplicity):
    from qcdata import Structure

    from mepd.nodes.node import StructureNode

    return StructureNode(structure=Structure(symbols=list(symbols), geometry=np.asarray(coords, dtype=float),
                                             charge=charge, multiplicity=multiplicity))


def _parent(c, symbols, engine, run_inputs, mode, charge, multiplicity) -> dict:
    """The parent through the same protocol as its variants, so that shifts
    compare like with like."""
    from mepd.cli_common import _geometry_optimizer_keywords

    r, ts, p = (_node(symbols, x, charge, multiplicity) for x in (c.reactant, c.ts, c.product))
    if mode == "reoptimize":
        r = engine.compute_geometry_optimization(r, keywords=_geometry_optimizer_keywords(run_inputs))[-1]
    e = [float(v) for v in engine.compute_energies([r, ts, p])]
    return {"barrier": (e[1] - e[0]) * HARTREE_TO_KCAL, "reaction": (e[2] - e[0]) * HARTREE_TO_KCAL}


def _variant(c, site, group, symbols, engine, run_inputs, mode, parent, output, charge, multiplicity) -> dict:
    from mepd import substituents as X
    from mepd.cli_common import _geometry_optimizer_keywords
    from mepd.cli_solvent import _write
    from mepd.web import results as R

    built, reacted = {}, []
    for key, geo in (("r", c.reactant), ("ts", c.ts), ("p", c.product)):
        sy, xyz, free = X.attach(symbols, geo, site, group)
        before = X.adjacency(sy, xyz)
        built[key] = X.relax_group(engine, _node(sy, xyz, charge, multiplicity), free)
        # The group must keep its own bonds (and make no new ones): otherwise
        # it reacted with the rest while relaxing, which is not a substituent effect.
        after = X.adjacency(sy, built[key].coords)
        if not np.array_equal(before[free], after[free]):
            reacted.append({"r": "reactant", "ts": "TS", "p": "product"}[key])
    row = {"channel": c.id, "site": site.anchor, "h": site.h, "group": group, "status": "ok"}
    if reacted:
        row["status"] = "reacted"
        row["error"] = f"the group formed or broke bonds while relaxing ({', '.join(reacted)})"
    # A group squeezed against the rest (< 1.2 Å from a non-bonded atom) is a clash, not a substituent effect.
    row["clash"] = _clash(built["ts"], len(symbols), free)
    r, ts, p = built["r"], built["ts"], built["p"]
    if mode == "reoptimize":
        r = engine.compute_geometry_optimization(r, keywords=_geometry_optimizer_keywords(run_inputs))[-1]
        ts = engine.compute_transition_state(node=ts)
        irc = engine.compute_irc_chain(ts)
        engine.compute_energies(irc.nodes)
        irc = R._oriented(irc, built["r"], built["p"])
        if not (R._same_connectivity(irc[0], built["r"]) and R._same_connectivity(irc[-1], built["p"])):
            row["status"] = "other_minima"
    e = [float(v) for v in engine.compute_energies([r, ts, p])]
    row["barrier"] = (e[1] - e[0]) * HARTREE_TO_KCAL
    row["shift"] = row["barrier"] - parent["barrier"]
    row["reaction_shift"] = (e[2] - e[0]) * HARTREE_TO_KCAL - parent["reaction"]
    # On the parent's reported scale (its barrier floor), for tables and kinetics.
    row["barrier_kcal"] = c.barrier_kcal + row["shift"]
    fp = output / c.id / f"{site.anchor}_{site.h}_{group.replace(' ', '_').replace('(', '').replace(')', '')}.xyz"
    _write([r, ts, p], e, fp)
    row["file"] = str(fp.relative_to(output))
    return row


def _clash(node, n_parent: int, free: list) -> bool:
    c = np.asarray(node.coords) * 0.529177210903
    fixed = [i for i in range(n_parent) if i not in set(free)]
    new = [i for i in free if i >= n_parent]
    if not new or not fixed:
        return False
    return float(np.min(np.linalg.norm(c[new][:, None] - c[fixed][None], axis=-1))) < 1.2


def _summarize(symbols, channels, sites, groups, rows, parents, mode) -> dict:
    from mepd import substituents as X

    ok = [r for r in rows if r["status"] == "ok" and not r.get("clash") and r["barrier_kcal"] >= -0.1]
    negative = [r for r in rows if r["status"] == "ok" and r.get("barrier_kcal") is not None
                and r["barrier_kcal"] < -0.1]
    lead = min(channels, key=lambda c: c.barrier_kcal)
    insights, trends = [], []
    mine = sorted((r for r in ok if r["channel"] == lead.id), key=lambda r: r["shift"])
    where = f" ({lead.label})" if len(channels) > 1 else ""
    if mine and mine[0]["shift"] < -0.5:
        r = mine[0]
        insights.append({"level": "accelerates", "text": f"Fastest{where}: {r['label']}, ΔE‡ "
                         f"{r['barrier_kcal']:.1f} ({r['shift']:+.1f})", "site": r["site"], "group": r["group"],
                         "channel": r["channel"]})
    if mine and mine[-1]["shift"] > 0.5:
        r = mine[-1]
        insights.append({"level": "slows", "text": f"Slowest{where}: {r['label']}, ΔE‡ "
                         f"{r['barrier_kcal']:.1f} ({r['shift']:+.1f})", "site": r["site"], "group": r["group"],
                         "channel": r["channel"]})
    for s in sites:
        fit = X.hammett({r["group"]: r["shift"] for r in mine if r["site"] == s.anchor})
        if fit is None:
            continue
        trends.append({"site": s.anchor, "label": s.label(symbols), **fit})
        if fit["r2"] >= 0.7 and abs(fit["slope"]) >= 2.0:
            who = "donors" if fit["slope"] > 0 else "acceptors"
            insights.append({"level": "trend", "site": s.anchor, "group": None, "text":
                             f"{s.label(symbols)}: ΔΔE‡ ≈ {fit['slope']:+.1f}·σp (r² {fit['r2']:.2f}); {who} speed it up"})
    # Which variants change the winning channel.
    if len(channels) > 1:
        kinds = {c.id: c.kind for c in channels}
        by_variant: dict = {}
        for r in ok:
            by_variant.setdefault((r["site"], r["group"]), {})[r["channel"]] = r
        switches = []
        for (site, group), per in by_variant.items():
            if len(per) < len(channels):
                continue
            win = min(per.values(), key=lambda r: r["barrier_kcal"])
            if win["channel"] != lead.id:
                runner = min((r for r in per.values() if r is not win), key=lambda r: r["barrier_kcal"])
                switches.append((runner["barrier_kcal"] - win["barrier_kcal"], win, runner))
        for margin, win, runner in sorted(switches, key=lambda x: -x[0])[:3]:
            label = next(c.label for c in channels if c.id == win["channel"])
            steer = kinds.get(lead.id) == "offtarget" and kinds.get(win["channel"]) == "direct"
            insights.append({"level": "selectivity", "site": win["site"], "group": win["group"],
                             "channel": win["channel"], "text":
                             f"{win['label']}: {label} wins{' (intended product)' if steer else ''}, "
                             f"{win['barrier_kcal']:.1f} vs {runner['barrier_kcal']:.1f}"})
    warnings = ([FAST_WARNING] if mode == "fast" else []) + [SCALE_NOTE]
    if mode == "fast":
        # Checked on Menshutkin: fast shifts within ~2 kcal/mol of re-optimized ones,
        # except where the variant's TS turned into another mechanism (a -12 that was
        # really a proton transfer). Large shifts are where that happens.
        big = [r for r in ok if abs(r["shift"]) >= LARGE_SHIFT]
        if big:
            warnings.append(f"Re-optimize before trusting shifts of {LARGE_SHIFT:g}+ kcal/mol: "
                            + ", ".join(f"{r['label']} ({r['shift']:+.1f})" for r in big[:5])
                            + (f" and {len(big) - 5} more" if len(big) > 5 else "") + ". The TS may change mechanism.")
        for ins in insights:
            r = next((x for x in ok if x["site"] == ins.get("site") and x["group"] == ins.get("group")
                      and x["channel"] == ins.get("channel", x["channel"])), None)
            if r is not None and abs(r["shift"]) >= LARGE_SHIFT:
                ins["text"] += " (check)"
    clashes = [r for r in rows if r.get("clash")]
    if clashes:
        warnings.append(f"{len(clashes)} variant(s) left out: the group clashes with the rest.")
    reacted = [r for r in rows if r["status"] == "reacted"]
    if reacted:
        warnings.append(f"{len(reacted)} variant(s) left out: the group reacted with the rest while relaxing.")
    if negative:
        warnings.append(f"{len(negative)} variant(s) left out with a negative ΔE‡ (e.g. {negative[0]['label']}, "
                        f"{negative[0]['barrier_kcal']:.1f}): the frozen core is no longer a TS there. Re-optimize.")
    failed = [r for r in rows if r["status"] in ("failed", "other_minima")]
    if failed:
        warnings.append(f"{len(failed)} variant(s) failed or lost the reaction's TS.")
    return {"insights": insights, "trends": trends, "warnings": warnings}
