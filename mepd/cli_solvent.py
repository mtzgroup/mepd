"""`mepd solvent`: a finished reaction profile, recomputed in implicit solvents.

Reads a `mepd run` / `mepd ts` output folder (mep_output.xyz, ts*.xyz and
their IRCs, all at one level of theory in the gas phase) and, for every
solvent asked for, writes `<output>/<solvent>/` with the same files and
the same frames, energies in that solvent:

* `--mode single-point` (default): every frame keeps its gas-phase
  geometry; its energy becomes E_gas + dG_solv (GFN2-xTB with ALPB/GBSA/
  CPCM-X, see mepd/solvation.py). Fast, and fair while the solvent does
  not move the stationary points much; the summary says when it probably
  does.
* `--mode reoptimize`: each TS is re-optimized in solvent from its
  gas-phase geometry, its IRC recomputed there, and the path's two ends
  re-minimized, so the stationary points are those of the solvated
  surface.

summary.json holds the barriers (gas and per solvent, one floor per
phase: the barrier floor rule), their shifts, half-lives and temperatures
for a 1 h half-life, and plain-language insights (mepd/conditions.py).
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

from qcconst.constants import HARTREE_TO_KCAL_PER_MOL as HARTREE_TO_KCAL

DEFAULT_SOLVENTS = ("water", "methanol", "dmso", "acetonitrile", "thf", "toluene")

SINGLE_POINT_WARNING = (
    "Single points on gas-phase geometries (not re-optimized in solvent): a first estimate. For charged or "
    "strongly polar TSs, confirm with --mode reoptimize.")
COMPOSITE_NOTE = (
    "Composite model: E(engine) + ΔG_solv(GFN2-xTB/{model}). Barriers are electronic plus solvation, not free "
    "energies.")
UNVERIFIED_WARNING = (
    "Gas-phase barrier not IRC-verified, and the solvent barriers inherit that: confirm the TS (TS opt + IRC) first.")


def modernize_warning(text: str, model_label: str = "") -> str:
    """Older runs' warning sentences -> today's shorter ones (results written before the rewording)."""
    if text.startswith("Single points on gas-phase geometries") and text != SINGLE_POINT_WARNING:
        return SINGLE_POINT_WARNING
    if text.startswith("Solvent energies are E(engine)"):
        return COMPOSITE_NOTE.format(model=model_label or "ALPB")
    if text.startswith("The gas-phase barrier is not verified"):
        return UNVERIFIED_WARNING
    return text


def _chains(source: Path) -> list[tuple[str, Path]]:
    """(role, file) for every chain the profile consists of."""
    files = []
    if (source / "mep_output.xyz").exists():
        files.append(("mep", source / "mep_output.xyz"))
    for fp in sorted(source.glob("ts*.xyz")):
        files.append(("irc" if fp.stem.endswith("_irc") else "ts", fp))
    if (source / "irc.xyz").exists():
        files.append(("irc", source / "irc.xyz"))
    return files


def _load(fp: Path, charge: int, multiplicity: int):
    from mepd.chain import Chain
    from mepd.inputs import ChainInputs

    return Chain.from_xyz(fp, ChainInputs(), charge=charge, spinmult=multiplicity)


def _energies(chain) -> list[Optional[float]]:
    return [None if n._cached_energy is None else float(n._cached_energy) for n in chain.nodes]


def _write(chain_nodes, energies, fp: Path) -> None:
    from mepd.geodesic_interpolation2.fileio import write_xyz
    from qcconst.constants import BOHR_TO_ANGSTROM

    fp.parent.mkdir(parents=True, exist_ok=True)
    coords = np.array([n.coords for n in chain_nodes]) * BOHR_TO_ANGSTROM
    write_xyz(filename=fp, atoms=chain_nodes[0].symbols, coords=coords)
    np.savetxt(fp.parent / f"{fp.stem}.energies", np.array(energies, dtype=float))


def _profile(folder: Path, kind: str, charge: int, multiplicity: int, orient: Optional[dict] = None) -> dict:
    """Barriers of one folder of files: forward barrier (from the reactant
    floor), reaction energy, and the TS that sets it. `kind` "run": as the
    web shows a `mepd run` result (reactant = the path's start). "tsopt": a
    TS without a known reactant; the side called reactant is fixed by
    `orient` (from the gas phase) so that solvents are compared in one
    direction even when the lower end changes."""
    from mepd.web import results as R

    if kind == "run":
        res = R.collect_ts(folder, charge, multiplicity)
        mep = R._load_chain(folder / "mep_output.xyz", charge, multiplicity)
        route_ts = (res.get("route_ts") or {}).get("label")
        out = {"barrier_kcal": res.get("barrier_kcal"), "verified": bool(res.get("barrier_verified")),
               "ts_label": route_ts, "reaction_kcal": None, "barrier_at_ts_kcal": res.get("barrier_kcal")}
        # As for a TS alone: the barrier is the highest point along the
        # route TS's IRC (the TS itself unless these are single points).
        items = {label: (ts, irc) for label, ts, irc in R._ts_dir_items(folder, charge, multiplicity)}
        if out["barrier_kcal"] is not None and route_ts in items and items[route_ts][1] is not None:
            ts, irc = items[route_ts]
            e = [R._node_energy(n) for n in irc.nodes]
            if R._node_energy(ts) is not None and None not in e:
                out["barrier_kcal"] += max(0.0, max(e) - R._node_energy(ts)) * HARTREE_TO_KCAL
            from mepd.cli import _irc_needs_reversal

            out["irc_file"] = _irc_of(f"{route_ts}.xyz")
            out["reverse"] = bool(mep is not None and _irc_needs_reversal(irc, mep[0], mep[-1]))
        if mep is not None:
            e0, e1 = R._node_energy(mep[0]), R._node_energy(mep[-1])
            lo = [e0] + [R._node_energy(end) for _, _, irc in R._ts_dir_items(folder, charge, multiplicity)
                         if irc is not None for end in (irc[0], irc[-1]) if R._same_connectivity(end, mep[0])]
            lo = [x for x in lo if x is not None]
            if lo:
                out["floor_hartree"] = min(lo)
            if lo and e1 is not None:
                out["reaction_kcal"] = (e1 - min(lo)) * HARTREE_TO_KCAL
        return out
    best = None
    for label, ts, irc in R._ts_dir_items(folder, charge, multiplicity):
        if irc is None or len(irc) < 3:
            continue
        nodes = list(irc.nodes)
        if orient is not None and label in orient:
            if orient[label]:
                nodes.reverse()
        k = int(np.argmin([np.linalg.norm(n.coords - ts.coords) for n in nodes]))
        e = [R._node_energy(n) for n in nodes]
        e_ts = R._node_energy(ts)
        if e_ts is None or None in e:
            continue
        # The TS itself is on neither side: a floor that included it would
        # clamp a negative barrier (TS below the reactant) to zero.
        floor = min(e[:k]) if k > 0 else e[0]
        top = min(e[k + 1:]) if k + 1 < len(e) else e[-1]
        # The highest point along the IRC: the TS itself on its own surface,
        # but with single points in solvent along a gas-phase IRC the
        # solvated maximum can sit elsewhere, and the barrier is that one.
        peak = max(e_ts, max(e))
        row = {"barrier_kcal": (peak - floor) * HARTREE_TO_KCAL, "verified": True, "ts_label": label,
               "barrier_at_ts_kcal": (e_ts - floor) * HARTREE_TO_KCAL,
               "reaction_kcal": (top - floor) * HARTREE_TO_KCAL,
               "reverse_kcal": (peak - top) * HARTREE_TO_KCAL, "floor_hartree": floor,
               "irc_file": _irc_of(f"{label}.xyz"), "reverse": bool(orient and orient.get(label))}
        if best is None or row["barrier_kcal"] < best["barrier_kcal"]:
            best = row
    return best or {"barrier_kcal": None, "verified": False, "ts_label": None, "reaction_kcal": None}


def _tsopt_orientation(source: Path, charge: int, multiplicity: int) -> dict:
    """Per TS label: reverse its IRC so the gas-phase lower end comes first
    (as the web draws a TS-only result)."""
    from mepd.web import results as R

    out = {}
    for label, ts, irc in R._ts_dir_items(source, charge, multiplicity):
        if irc is None or len(irc) < 2:
            continue
        a, b = R._node_energy(irc[0]), R._node_energy(irc[-1])
        out[label] = bool(a is not None and b is not None and b < a)
    return out


def _peak_moved(gas_e: list[float], solv_e: list[float], ts_index: int) -> bool:
    """Does the solvated energy along a gas-phase IRC peak somewhere other
    than at (next to) the gas-phase TS, by more than 0.5 kcal/mol?"""
    k = int(np.argmax(solv_e))
    return abs(k - ts_index) > 1 and (solv_e[k] - solv_e[ts_index]) * HARTREE_TO_KCAL > 0.5


def solvent(
    source: Path = typer.Argument(..., help="A finished `mepd run` or `mepd ts` output folder (gas phase)."),
    solvents: Optional[List[str]] = typer.Option(
        None, "--solvent", "-s", help=f"Solvent (repeatable). Default: {', '.join(DEFAULT_SOLVENTS)}."),
    model: str = typer.Option("alpb", "--model", help="Implicit solvent model: alpb, gbsa or cpcmx (GFN2-xTB)."),
    mode: str = typer.Option(
        "single-point", "--mode",
        help="single-point: solvent energies on the gas-phase geometries (fast; a first estimate). "
        "reoptimize: re-optimize each TS, its IRC and the path's ends in solvent."),
    temperature: float = typer.Option(298.15, "--temperature", "-T", help="Temperature (K) for rates and half-lives."),
    inputs: Optional[Path] = typer.Option(None, "--inputs", "-i", help="RunInputs TOML: the level the source was run at."),
    charge: int = typer.Option(0, "--charge"),
    multiplicity: int = typer.Option(1, "--multiplicity"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Default: <source>/solvent."),
) -> None:
    """Recompute a reaction profile in implicit solvents, and say what changes."""
    from mepd.cli_common import _open_run_inputs
    from mepd.conditions import kinetics_row, solvent_insights
    from mepd.solvation import MODELS, SolvationCorrection, get_solvent, solvate_engine

    source = Path(source)
    if mode not in ("single-point", "reoptimize"):
        raise typer.BadParameter("--mode must be single-point or reoptimize")
    model = model.lower()
    if model not in MODELS:
        raise typer.BadParameter(f"--model must be one of {', '.join(MODELS)}")
    chains = _chains(source)
    ts_files = [fp for role, fp in chains if role == "ts"]
    if not ts_files:
        raise typer.BadParameter(f"{source} has no ts*.xyz: run `mepd run --use-tsopt --irc` or `mepd ts` first")
    kind = "run" if (source / "mep_output.xyz").exists() else "tsopt"
    output = Path(output) if output is not None else source / "solvent"
    output.mkdir(parents=True, exist_ok=True)
    run_inputs = _open_run_inputs(inputs)
    if run_inputs.solvation:
        raise typer.BadParameter("the --inputs profile already has a [solvation] table: give the gas-phase profile")
    try:
        picked = [get_solvent(s) for s in (solvents or DEFAULT_SOLVENTS)]
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    orient = _tsopt_orientation(source, charge, multiplicity) if kind == "tsopt" else None
    gas = _profile(source, kind, charge, multiplicity, orient)
    gas.update(kinetics_row(gas.get("barrier_kcal"), temperature))
    typer.echo(f"Gas phase: barrier {gas['barrier_kcal']:.1f} kcal/mol" if gas.get("barrier_kcal") is not None
               else "Gas phase: no barrier could be read from the source")

    loaded = {fp.name: _load(fp, charge, multiplicity) for _, fp in chains}
    missing = [name for name, ch in loaded.items() if None in _energies(ch)]
    if missing:
        typer.echo(f"Computing gas-phase energies for {', '.join(missing)} (not stored with the source)...")
        for name in missing:
            run_inputs.engine.compute_energies(loaded[name].nodes)

    rows, warnings = [], []
    for sv in picked:
        folder = output / sv.key
        row = {"key": sv.key, "label": sv.label, "kind": sv.kind, "epsilon": sv.epsilon, "bp_c": sv.bp_c,
               "mp_c": sv.mp_c, "model": model, "dir": sv.key, "peak_shift": False, "error": None}
        typer.echo(f"{sv.label} ({MODELS[model]}, {mode})...")
        try:
            if model not in sv.models:
                raise ValueError(f"no {MODELS[model]} parameters for {sv.label}")
            # Single points first, in either mode: in "reoptimize" they are
            # kept in single-point/ for comparison, and their solvated peak
            # along each gas-phase IRC is where the solvent TS search starts.
            sp_folder = folder if mode == "single-point" else folder / "single-point"
            corr = SolvationCorrection(sv.key, model)
            sp = _single_points(corr, loaded, sp_folder, row)
            if corr.fallbacks:
                row["notes"] = [f"GFN2-xTB needed {', '.join(sorted({f for _, f in corr.fallbacks}))} to converge on "
                                f"{len(corr.fallbacks)} frame(s); their solvation energies are slightly less precise"]
            if mode == "reoptimize":
                sp_prof = _profile(sp_folder, kind, charge, multiplicity, orient)
                row["single_point_barrier_kcal"] = sp_prof.get("barrier_kcal")
                _reoptimize(sv.key, model, run_inputs, loaded, sp, folder, row)
            prof = _profile(folder, kind, charge, multiplicity, orient)
            row.update(prof)
            row.update(kinetics_row(prof.get("barrier_kcal"), temperature))
            row["shift_kcal"] = (prof["barrier_kcal"] - gas["barrier_kcal"]) \
                if prof.get("barrier_kcal") is not None and gas.get("barrier_kcal") is not None else None
            if row["shift_kcal"] is not None:
                typer.echo(f"  barrier {prof['barrier_kcal']:.1f} kcal/mol ({row['shift_kcal']:+.1f} vs gas)")
            elif mode == "reoptimize" and row.get("single_point_barrier_kcal") is not None:
                row["error"] = ("no TS in solvent connects the gas-phase minima; single points on the gas-phase "
                                f"path give {row['single_point_barrier_kcal']:.1f} kcal/mol")
                typer.echo(f"  {row['error']}")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            typer.echo(f"  failed: {row['error']}")
        rows.append(row)

    if mode == "single-point":
        warnings.append(SINGLE_POINT_WARNING)
    if gas.get("barrier_kcal") is not None and not gas.get("verified", True):
        warnings.append(UNVERIFIED_WARNING)
    engine_name = type(run_inputs.engine).__name__
    native = False
    try:
        native = type(solvate_engine(run_inputs.engine, {"solvent": "water", "model": model})).__name__ != "SolvatedEngine"
    except Exception:
        pass
    if not native:
        warnings.append(COMPOSITE_NOTE.format(model=MODELS[model]))
    ok = [r for r in rows if r.get("barrier_kcal") is not None and r.get("shift_kcal") is not None]
    insights = solvent_insights(gas, ok, temperature, mode=mode)
    summary = {
        "source": str(source), "kind": kind, "mode": mode, "model": model, "model_label": MODELS[model],
        "method": "native" if native else "correction", "engine": engine_name, "temperature": temperature,
        "gas": gas, "solvents": rows, "insights": insights, "warnings": warnings,
        "references": ["gfn2", "alpb", "eyring"] + (["cpcmx"] if model == "cpcmx" else []),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    for ins in insights:
        typer.echo(f"[{ins['level']}] {ins['text']}")
    for w in warnings:
        typer.echo(f"Note: {w}")
    typer.echo(f"Wrote {output / 'summary.json'}")


def _json_default(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)


def _single_points(corr, loaded: dict, folder: Path, row: dict) -> dict[str, list[float]]:
    """E_gas + dG_solv for every frame of every file, written to `folder`;
    flags `row["peak_shift"]` when the solvated maximum along an IRC is not
    at its gas-phase TS. Returns the energies per file name."""
    flat = [(name, n) for name, ch in loaded.items() for n in ch.nodes]
    dg, _ = corr.compute([n for _, n in flat])
    by_file: dict[str, list] = {}
    for (name, n), d in zip(flat, dg):
        by_file.setdefault(name, []).append(float(n._cached_energy) + float(d))
    for name, ch in loaded.items():
        _write(ch.nodes, by_file[name], folder / name)
        ts = loaded.get(_ts_of(name)) if _is_irc(name) else None
        if ts is not None:
            k = _nearest(ch.nodes, ts[0])
            row["peak_shift"] |= _peak_moved(_energies(ch), by_file[name], k)
    return by_file


def _is_irc(name: str) -> bool:
    return name == "irc.xyz" or name.endswith("_irc.xyz")


def _ts_of(irc_name: str) -> str:
    return "ts.xyz" if irc_name == "irc.xyz" else irc_name.replace("_irc.xyz", ".xyz")


def _irc_of(ts_name: str) -> str:
    return "irc.xyz" if ts_name == "ts.xyz" else ts_name.replace(".xyz", "_irc.xyz")


def _nearest(nodes, target) -> int:
    return int(np.argmin([np.linalg.norm(n.coords - target.coords) for n in nodes]))


def _reoptimize(solvent_key: str, model: str, run_inputs, loaded: dict, sp: dict, folder: Path, row: dict) -> None:
    """Stationary points on the solvated surface. Each TS is searched from
    the solvated single-point peak along its gas-phase IRC (where the TS
    has moved to, to first order), then from the gas-phase TS; a TS counts
    only if its IRC connects minima with the same bonds as the gas-phase
    IRC's ends. The path's two ends are re-minimized (a 2-frame
    mep_output.xyz)."""
    from mepd.cli_common import _geometry_optimizer_keywords
    from mepd.solvation import solvate_engine
    from mepd.web import results as R

    engine = solvate_engine(run_inputs.engine, {"solvent": solvent_key, "model": model})
    folder.mkdir(parents=True, exist_ok=True)
    notes = []
    mep = loaded.get("mep_output.xyz")
    if mep is not None:
        ends = []
        for node in (mep[0], mep[-1]):
            traj = engine.compute_geometry_optimization(node.update_coords(node.coords),
                                                        keywords=_geometry_optimizer_keywords(run_inputs))
            ends.append(traj[-1])
        engine.compute_energies(ends)
        _write(ends, [n.energy for n in ends], folder / "mep_output.xyz")
    for name, ch in loaded.items():
        if not name.startswith("ts") or _is_irc(name):
            continue
        label = Path(name).stem
        gas_irc = loaded.get(_irc_of(name))
        guesses = [("gas-phase TS", ch[0])]
        if gas_irc is not None and _irc_of(name) in sp:
            peak = int(np.argmax(sp[_irc_of(name)]))
            if peak != _nearest(gas_irc.nodes, ch[0]):
                guesses.insert(0, ("solvated peak of the gas-phase IRC", gas_irc[peak]))
        found = False
        for how, guess in guesses:
            try:
                ts = engine.compute_transition_state(node=guess.update_coords(guess.coords))
                engine.compute_energies([ts])
                irc = engine.compute_irc_chain(ts)
                engine.compute_energies(irc.nodes)
            except Exception as exc:
                notes.append(f"{label}: TS search from the {how} failed ({type(exc).__name__}: {exc})")
                continue
            if gas_irc is not None:
                irc = R._oriented(irc, gas_irc[0], gas_irc[-1])
                if not (R._same_connectivity(irc[0], gas_irc[0]) and R._same_connectivity(irc[-1], gas_irc[-1])):
                    notes.append(f"{label}: from the {how}, the solvent TS's IRC connects other minima "
                                 "(not this reaction's saddle)")
                    continue
            _write([ts], [ts.energy], folder / name)
            _write(irc.nodes, [n.energy for n in irc.nodes], folder / _irc_of(name))
            notes.append(f"{label}: TS found in solvent from the {how}")
            found = True
            break
        if not found:
            notes.append(f"{label}: no TS of this reaction found in solvent")
    row["notes"] = row.get("notes", []) + notes
