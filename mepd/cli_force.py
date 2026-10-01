"""`mepd force`: how mechanical force on chosen atoms would favor or
disfavor the reaction channels of a finished search (mepd/mechanochem.py).

Reads a `mepd run`, `mepd ts` or `mepd channels` output folder. Every
channel (a TS with its IRC) gives, for every heavy-atom pair, the change
in that pair's distance from reactant to TS; Bell's model turns it into a
barrier change per nN. The summary names the pairs that speed each channel
up or hold it back, the force that brings it to a 1 h half-life, and --
with several channels -- the force at which a slower channel overtakes the
fastest one.

`--mode reoptimize` checks the estimates on the force-modified surface
itself (E - F * d_ij): at each force the reactant is re-minimized and the
TS re-optimized (with its IRC) under force, for the strongest lever of the
fastest channel and for the leading selectivity switch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

from qcconst.constants import HARTREE_TO_KCAL_PER_MOL as HARTREE_TO_KCAL


def _coords(xyz: str) -> np.ndarray:
    from qcdata import Structure

    return np.asarray(Structure.from_xyz(xyz).geometry, dtype=float)


def load_channels(source: Path, charge: int, multiplicity: int, purpose: str = "put under force"):
    """(symbols, [Channel], kind) from a finished output folder, oriented
    and floored as the web shows it (one reactant floor for all channels)."""
    from mepd.mechanochem import Channel
    from mepd.web import results as R

    op = R.detect_operation(source)
    if op not in ("channels", "ts", "tsopt"):
        raise typer.BadParameter(f"{source}: not a `mepd run`, `mepd ts` or `mepd channels` output")
    res = R.COLLECTORS[op](source, charge, multiplicity)
    kinds = ("channel", "offtarget") if op == "channels" else ("irc",)
    channels, symbols = [], None
    for g in res["groups"]:
        if g["kind"] not in kinds:
            continue
        for e in g["entries"]:
            frames = e["frames"]
            if e.get("barrier_kcal") is None or len(frames) < 3 or e.get("ts_index") is None:
                continue
            k = e["ts_index"]
            if symbols is None:
                from qcdata import Structure

                symbols = list(Structure.from_xyz(frames[0]["xyz"]).symbols)
            label = e["label"] if op == "channels" else e["label"].removesuffix(" IRC")
            channels.append(Channel(id=e["id"], label=label, barrier_kcal=float(e["barrier_kcal"]),
                                    reactant=_coords(frames[0]["xyz"]), ts=_coords(frames[k]["xyz"]),
                                    product=_coords(frames[-1]["xyz"]),
                                    kind="offtarget" if g["kind"] == "offtarget" else "direct"))
    if op != "channels":
        for c in channels:
            c.label = "the reaction" if len(channels) == 1 else f"TS {c.label}"
    if not channels:
        why = ("no channel was classified: no TS search found a TS whose IRC connects this reactant and product"
               if op == "channels" else "no TS with an IRC and a barrier")
        raise typer.BadParameter(f"{source}: {why}, so there is nothing to {purpose}.")
    return symbols, channels, op


def _parse_pair(text: str, natoms: int) -> tuple[int, int]:
    try:
        i, j = (int(x) for x in text.replace("-", ",").split(","))
    except ValueError:
        raise typer.BadParameter(f"--pair {text!r}: give two atom indices, e.g. 0,7") from None
    if not (0 <= i < natoms and 0 <= j < natoms) or i == j:
        raise typer.BadParameter(f"--pair {text!r}: atoms are numbered 0..{natoms - 1}")
    return i, j


def force(
    source: Path = typer.Argument(..., help="A finished `mepd run`, `mepd ts` or `mepd channels` output folder."),
    pairs: Optional[List[str]] = typer.Option(
        None, "--pair", "-p", help="Atoms to pull apart, 'i,j' (0-based, as the viewer numbers them; repeatable). "
        "Default: rank every heavy-atom pair."),
    forces: Optional[List[float]] = typer.Option(
        None, "--force", "-f", help="Forces (nN) to check with --mode reoptimize (repeatable). Default: 0.5, 1, 1.5."),
    mode: str = typer.Option("bell", "--mode", help="bell: first-order estimates from the geometries (instant). "
                             "reoptimize: also re-optimize TS, IRC and reactant under force for the main levers."),
    top: int = typer.Option(5, "--top", help="Levers to list per channel and direction."),
    hydrogens: bool = typer.Option(False, "--hydrogens/--no-hydrogens", help="Also rank pairs involving H."),
    max_force: float = typer.Option(2.5, "--max-force", help="Largest force (nN) scanned for selectivity switches."),
    temperature: float = typer.Option(298.15, "--temperature", "-T", help="Temperature (K) for half-lives."),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="RunInputs TOML: the level the source ran at."),
    charge: int = typer.Option(0, "--charge"),
    multiplicity: int = typer.Option(1, "--multiplicity"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Default: <source>/force."),
) -> None:
    """Which atoms to pull on to favor or disfavor each reaction channel, and how hard."""
    from mepd.mechanochem import analyze

    source = Path(source)
    if mode not in ("bell", "reoptimize"):
        raise typer.BadParameter("--mode must be bell or reoptimize")
    symbols, channels, kind = load_channels(source, charge, multiplicity)
    chosen = [_parse_pair(p, len(symbols)) for p in pairs] if pairs else None
    output = Path(output) if output is not None else source / "force"
    output.mkdir(parents=True, exist_ok=True)
    result = analyze(symbols, channels, pairs=chosen, hydrogens=hydrogens, top=top, temperature=temperature,
                     max_force=max_force)
    efei = []
    if mode == "reoptimize":
        from mepd.cli_common import _open_run_inputs

        efei = _reoptimize(symbols, channels, result, _open_run_inputs(inputs), forces or [0.5, 1.0, 1.5],
                           output, charge, multiplicity)
        result["insights"] += _efei_insights(efei)
    summary = {"source": str(source), "kind": kind, "mode": mode, "temperature": temperature,
               "symbols": symbols, "chosen_pairs": [list(p) for p in chosen] if chosen else None,
               **result, "efei": efei, "references": ["bell", "efei", "fmpes", "cogef", "extended-bell"],
               "channel_geometries": {c.id: {"reactant": c.reactant.tolist(), "ts": c.ts.tolist()} for c in channels}}
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


def _reoptimize(symbols, channels, result, run_inputs, forces, output: Path, charge: int, multiplicity: int):
    """Barriers on the force-modified surface for the main levers: the
    fastest channel's strongest pulling pair, and the leading selectivity
    switch (its channel and the fastest one, on its pair)."""
    from qcdata import Structure

    from mepd.chain import Chain
    from mepd.cli_common import _geometry_optimizer_keywords
    from mepd.inputs import ChainInputs
    from mepd.mechanochem import ForcedEngine, bell_barrier, distances, pair_label
    from mepd.nodes.node import StructureNode
    from mepd.web import results as R

    by_id = {c.id: c for c in channels}
    lead = min(channels, key=lambda c: c.barrier_kcal)
    lead_row = next(r for r in result["channels"] if r["id"] == lead.id)
    todo = []
    if lead_row["favor"]:
        todo.append((lead, tuple(lead_row["favor"][0]["pair"])))
    if result["selectivity"]:
        s = result["selectivity"][0]
        todo += [(by_id[s["channel"]], tuple(s["pair"])), (lead, tuple(s["pair"]))]
    todo = list({(c.id, p): (c, p) for c, p in todo}.values())

    def node(coords):
        return StructureNode(structure=Structure(symbols=symbols, geometry=np.asarray(coords), charge=charge,
                                                 multiplicity=multiplicity))

    out = []
    for c, pair in todo:
        zero_r, zero_ts = node(c.reactant), node(c.ts)
        dq = float(distances(c.ts, [pair])[0] - distances(c.reactant, [pair])[0])
        for f in forces:
            eng = ForcedEngine(run_inputs.engine, pairs=[pair], force_nN=float(f))
            row = {"channel": c.id, "label": c.label, "pair": list(pair), "pair_label": pair_label(symbols, pair),
                   "force_nN": float(f), "barrier_bell": bell_barrier(c.barrier_kcal, dq, f),
                   "barrier_zero": c.barrier_kcal, "barrier_efei": None, "status": None}
            typer.echo(f"{c.label}, {row['pair_label']} at {f:g} nN...")
            try:
                r_traj = eng.compute_geometry_optimization(zero_r.update_coords(zero_r.coords),
                                                           keywords=_geometry_optimizer_keywords(run_inputs))
                r_f = r_traj[-1]
                e_r = float(eng.compute_energies([r_f])[0])
                ts_f = eng.compute_transition_state(node=zero_ts.update_coords(zero_ts.coords))
                e_ts = float(eng.compute_energies([ts_f])[0])
                irc = eng.compute_irc_chain(ts_f)
                eng.compute_energies(irc.nodes)
                irc = R._oriented(irc, zero_r, node(c.product) if c.product is not None else None)
                same = R._same_connectivity(irc[0], zero_r) and (
                    c.product is None or R._same_connectivity(irc[-1], node(c.product)))
                row["barrier_efei"] = (e_ts - e_r) * HARTREE_TO_KCAL
                row["status"] = "ok" if same else "other_minima"
                folder = output / f"{c.id}_{pair[0]}-{pair[1]}_{f:g}nN"
                folder.mkdir(parents=True, exist_ok=True)
                irc.write_to_disk(folder / "irc.xyz")
                Chain.model_validate({"nodes": [ts_f], "parameters": ChainInputs()}).write_to_disk(folder / "ts.xyz")
                row["dir"] = folder.name
                typer.echo(f"  under force {row['barrier_efei']:.1f} kcal/mol (Bell {row['barrier_bell']:.1f})"
                           + ("" if same else "; its IRC connects other minima"))
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                typer.echo(f"  failed: {row['error']}")
            out.append(row)
    return out


def _efei_insights(efei: list[dict]) -> list[dict]:
    out = []
    for key in dict.fromkeys((r["channel"], tuple(r["pair"])) for r in efei):
        rows = [r for r in efei if (r["channel"], tuple(r["pair"])) == key and r["status"] == "ok"]
        bad = [r for r in efei if (r["channel"], tuple(r["pair"])) == key and r["status"] != "ok"]
        if rows:
            worst = max(rows, key=lambda r: abs(r["barrier_efei"] - r["barrier_bell"]))
            dev = worst["barrier_efei"] - worst["barrier_bell"]
            text = (f"{worst['label']}, {worst['pair_label']} re-optimized under force: "
                    + ", ".join(f"{r['barrier_efei']:.1f} at {r['force_nN']:g} nN" for r in rows)
                    + " (Bell " + ", ".join(f"{r['barrier_bell']:.1f}" for r in rows) + ")")
            text += ("; Bell holds" if abs(dev) < 2.0 else f"; Bell off by {dev:+.1f}: use the re-optimized values")
            out.append({"level": "check", "text": text, "pair": list(key[1]), "channel": key[0]})
        for r in bad:
            what = ("TS connects other minima (force changes the mechanism)"
                    if r["status"] == "other_minima" else "no TS under force (barrier may vanish)")
            out.append({"level": "caution", "pair": r["pair"], "channel": r["channel"],
                        "text": f"{r['label']}, {r['pair_label']} at {r['force_nN']:g} nN: {what}"})
    return out
