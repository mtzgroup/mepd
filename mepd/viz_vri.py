"""Interactive view of a `mepd discovery vri` result, for `mepd visualize`.

One self-contained HTML page per VRI output directory, built to make the
result understandable rather than just browsable:

- the IRC in a 3D viewer, with two plots against the signed arc length s
  from TS1 (reverse branch s < 0, forward branch s > 0) that follow the
  frame slider: energy relative to TS1, and the lowest frequency
  perpendicular to the path (the VRT scan), with the VRT marked;
- per branch: the verdict in plain language, one-click views of TS1, the
  VRT, P1, P2 and TS2, the ridge mode animated, TS2's IRC, the TS2 checks,
  the ridge-check energy profile, and where each push went.

Reads only files the VRI search writes (xyz + .energies sidecars and the
two JSON files); no QC engine or molecular graphs are needed.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

H2K = 627.509474
BOHR_TO_ANGSTROM = 0.529177210903


def is_vri_output(path: Path) -> bool:
    return path.is_dir() and (path / "summary.json").exists() and (path / "projected_freqs.json").exists()


def _read_xyz_frames(fp: Path) -> list[str]:
    """Multi-frame xyz file -> list of single-frame xyz texts."""
    lines = fp.read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        frames.append("\n".join(lines[i : i + n + 2]) + "\n")
        i += n + 2
    return frames


def _frame_arrays(frame: str):
    rows = frame.splitlines()[2:]
    symbols = [r.split()[0] for r in rows if r.strip()]
    coords = np.array([[float(x) for x in r.split()[1:4]] for r in rows if r.strip()])
    return symbols, coords


def _read_energies(fp: Path) -> Optional[np.ndarray]:
    sidecar = fp.with_suffix(".energies")
    if not sidecar.exists():
        return None
    try:
        values = np.atleast_1d(np.loadtxt(sidecar, dtype=float))
        return values if values.size else None
    except Exception:
        return None


def _structure(fp: Path, e_ref: Optional[float]) -> Optional[dict]:
    if not fp.exists():
        return None
    frames = _read_xyz_frames(fp)
    energies = _read_energies(fp)
    e = None if energies is None or e_ref is None else float((energies[-1] - e_ref) * H2K)
    return {"xyz": frames[-1], "e_rel": e}


def _sequence(fp: Path, e_ref: Optional[float]) -> Optional[dict]:
    if not fp.exists():
        return None
    frames = _read_xyz_frames(fp)
    energies = _read_energies(fp)
    rel = None
    if energies is not None and len(energies) == len(frames) and e_ref is not None:
        rel = [float((e - e_ref) * H2K) for e in energies]
    return {"frames": frames, "e_rel": rel}


VERDICT_TEXT = {
    "bifurcation": "Sideways pushes off the IRC drain into both P1 and P2: the path from TS1 splits.",
    "second_product_no_split": "A second product exists, but every sideways push off the IRC drains back to P1: not a bifurcation.",
    "second_product_untested": "A second product exists, but the basin test did not run.",
    "vrt_no_second_product": "The valley turns into a ridge along the IRC, but the ridge leads back to P1 or a conformer of it.",
    "transient_softening": "Only a brief, noise-level dip in the perpendicular curvature.",
    "no_vrt": "Every direction perpendicular to the IRC stays a valley.",
}


def _ridge_outcome(branch: dict) -> str:
    if not branch.get("vrt"):
        return "no VRT"
    pr = branch.get("products") or {}
    outs = [o for o in pr.get("push_outcomes") or [] if not o.get("failed")]
    if branch.get("verdict") == "bifurcation":
        return "ridge leads to a different product (verified TS2)"
    if pr.get("p2_energy") is not None:
        return "ridge leads to a different product (TS2 not verified)"
    if any(o.get("stereo_differs") for o in outs):
        return "ridge leads to a stereoisomer of P1"
    if any(o.get("same_bonds") and (o.get("rmsd_to_p1") or 0) > 0.3 for o in outs):
        return "ridge leads to another conformer of P1"
    if outs:
        return "ridge collapses back to P1"
    return "no push results recorded"


def _path_curvature(mw: np.ndarray, arc: np.ndarray, half_window: int = 2) -> list:
    """How sharply the IRC bends at each frame, |dt/ds| (per amu^1/2 bohr),
    from the geometry alone: unit tangents over +-`half_window` frames,
    differenced over the same window. Where the path turns a corner the
    transverse modes mix strongly, which is where narrow dips in the
    projected frequencies tend to sit."""
    n = len(mw)
    if n < 2 * half_window + 3:
        return [None] * n
    tangents = []
    for k in range(n):
        lo, hi = max(0, k - half_window), min(n - 1, k + half_window)
        t = mw[hi] - mw[lo]
        norm = np.linalg.norm(t)
        tangents.append(t / norm if norm > 1e-12 else None)
    out = []
    for k in range(n):
        lo, hi = max(0, k - half_window), min(n - 1, k + half_window)
        if tangents[lo] is None or tangents[hi] is None or arc[hi] - arc[lo] < 1e-9:
            out.append(None)
            continue
        out.append(float(np.linalg.norm(tangents[hi] - tangents[lo]) / (arc[hi] - arc[lo])))
    return out


def _branch_frames(coords_mw: np.ndarray, ts_index: int) -> dict:
    """Global frame indices of each TS-outward branch, with the same
    near-duplicate dropping as `vri.split_irc_branches` (so the scan's
    `chain_index` maps onto these lists)."""
    def dedupe(idx):
        out = [idx[0]]
        for i in idx[1:]:
            if np.linalg.norm(coords_mw[i] - coords_mw[out[-1]]) > 1e-6:
                out.append(i)
        return out

    n = len(coords_mw)
    return {
        "forward": dedupe(list(range(ts_index, n))),
        "reverse": dedupe(list(range(ts_index, -1, -1))),
    }


def _load_surface(fp: Path) -> Optional[dict]:
    if not fp.exists():
        return None
    try:
        surf = json.loads(fp.read_text())
    except Exception:
        return None
    surf.pop("grid_reached", None)
    return surf


def _quick_surface(d: Path, branch: str, summary: dict) -> Optional[dict]:
    """The instant map from saved data (see `vri_surface.quick_surface`)."""
    if not ((d / f"p1_{branch}.xyz").exists() and (d / f"p2_{branch}.xyz").exists()):
        return None
    try:
        from mepd.discovery.vri_surface import quick_surface

        return quick_surface(d, branch, charge=int(summary.get("charge", 0)),
                             multiplicity=int(summary.get("multiplicity", 1)))
    except Exception:
        return None


@dataclass
class VriCandidate:
    label: str
    payload: dict = field(default_factory=dict)


@dataclass
class VriResult:
    """A `mepd discovery vri` output directory: the input TS1 and any
    symmetric TS1 candidates, each with its IRC scan and branch results."""

    path: Path
    summary: dict
    candidates: list = field(default_factory=list)  # list[VriCandidate]


def load_vri_result(path: Path) -> VriResult:
    from mepd.discovery.vri_surface import candidate_dir
    from mepd.helper_functions import get_mass

    path = Path(path)
    top = json.loads((path / "summary.json").read_text())
    entries = top.get("ts1_candidates") or [{"label": "input", "dir": str(path)}]
    settings = top.get("settings") or {}
    result = VriResult(path=path, summary=top)
    for entry in entries:
        d = candidate_dir(path, entry)
        if not (d / "summary.json").exists() or not (d / "projected_freqs.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        scan = json.loads((d / "projected_freqs.json").read_text())
        ts_index = int(scan["ts_index"])
        e_ts1 = s.get("ts1_energy")

        irc_frames = _read_xyz_frames(d / "irc.xyz") if (d / "irc.xyz").exists() else []
        irc_e = _read_energies(d / "irc.xyz")
        coords = []
        symbols = None
        for f in irc_frames:
            symbols, xyz = _frame_arrays(f)
            coords.append(xyz)
        if coords:
            sqrt_m = np.sqrt(np.repeat([get_mass(x) for x in symbols], 3))
            mw = np.array([c.reshape(-1) / BOHR_TO_ANGSTROM * sqrt_m for c in coords])
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(mw, axis=0), axis=1))])
            s_signed = (arc - arc[ts_index]).tolist()
            branch_frames = _branch_frames(mw, ts_index)
            curvature = _path_curvature(mw, arc)
        else:
            s_signed, branch_frames, curvature = [], {"forward": [], "reverse": []}, []
        irc_rel = None
        if irc_e is not None and len(irc_e) == len(irc_frames) and e_ts1 is not None:
            irc_rel = [float((e - e_ts1) * H2K) for e in irc_e]

        branches = {}
        for name, b in (s.get("branches") or {}).items():
            sign = 1.0 if name == "forward" else -1.0
            bscan = (scan.get("branches") or {}).get(name) or {}
            frames_of = branch_frames.get(name, [])

            def point(p):
                k = p.get("chain_index")
                frame = frames_of[k] if k is not None and k < len(frames_of) else None
                return {
                    "s": sign * float(p["s"]),
                    "f1": float(p["lowest_freqs"][0]),
                    "f2": float(p["lowest_freqs"][1]) if len(p["lowest_freqs"]) > 1 else None,
                    "l1": float(p.get("lowest_eigval", 0.0)),
                    "e_rel": None if e_ts1 is None else float((p["energy"] - e_ts1) * H2K),
                    "frame": frame,
                }

            pr = b.get("products") or {}
            checks = pr.get("ts2_checks") or {}
            from mepd.discovery.vri_surface import _legacy_ts2_label

            ts2_label = _legacy_ts2_label(pr, name)
            ts2_irc = _sequence(d / f"{ts2_label}_irc.xyz", e_ts1) if ts2_label else None
            if ts2_irc is None and pr.get("ts2_source") == "irc_endpoint":
                ts2_irc = None  # TS2 is the IRC's own endpoint: no separate IRC

            vrt = b.get("vrt")
            scan_pts = [point(p) for p in bscan.get("points", [])]
            deepest = None
            if vrt:
                after = [p for p in scan_pts if abs(p["s"]) >= float(vrt["s"])]
                if after:
                    run = []
                    for p in sorted(after, key=lambda q: abs(q["s"])):
                        if p["f1"] >= 0:
                            break
                        run.append(p)
                    if run:
                        low = min(run, key=lambda q: q["f1"])
                        deepest = {"f": low["f1"], "s": low["s"], "e_rel": low["e_rel"],
                                   "ridge_length": abs(run[-1]["s"]) - float(vrt["s"])}
            branches[name] = {
                "verdict": b.get("verdict"),
                "verdict_text": VERDICT_TEXT.get(b.get("verdict"), ""),
                "ridge_outcome": _ridge_outcome(b),
                "scan": scan_pts,
                "deepest": deepest,
                "bisection": [point(p) for p in bscan.get("bisection_points", [])],
                "vrt": None if not vrt else {
                    "s": sign * float(vrt["s"]),
                    "e_rel": vrt.get("rel_ts1_kcal_mol"),
                    "freq_before": vrt.get("freq_before"),
                    "freq_after": vrt.get("freq_after"),
                },
                "late_dips": [sign * float(x) for x in b.get("late_dips_s") or []],
                "transient_dips": [sign * float(x) for x in b.get("transient_dips_s") or []],
                "valley_reforms": bool(b.get("valley_reforms")),
                "vrt_structure": _structure(d / f"vrt_{name}.xyz", e_ts1),
                "ridge_mode": _sequence(d / f"ridge_mode_{name}.xyz", e_ts1),
                "p1": _structure(d / f"p1_{name}.xyz", e_ts1),
                "p2": _structure(d / f"p2_{name}.xyz", e_ts1),
                "ts2": _structure(d / f"ts2_{name}.xyz", e_ts1),
                "ts2_irc": ts2_irc,
                "ts2_source": pr.get("ts2_source"),
                "evidence": pr.get("evidence"),
                "degenerate": pr.get("degenerate"),
                "checks": {k: checks.get(k) for k in ("n_imaginary", "below_ts1", "connects_p1_p2", "distinct_from_ts1", "method")},
                "ridge": (checks.get("ridge") or None),
                "notes": pr.get("notes") or [],
                "push_outcomes": pr.get("push_outcomes") or [],
                "surface": _quick_surface(d, name, s),
                "checks_run": pr.get("checks") or (json.loads((d / f"checks_{name}.json").read_text())
                                                    if (d / f"checks_{name}.json").exists() else None),
            }

        ts1_xyz = None
        if irc_frames:
            ts1_xyz = irc_frames[ts_index]
        elif (d / "ts1.xyz").exists():
            ts1_xyz = _read_xyz_frames(d / "ts1.xyz")[-1]

        result.candidates.append(VriCandidate(label=entry.get("label", "input"), payload={
            "label": entry.get("label", "input"),
            "verdict": s.get("verdict"),
            "ts1_freqs": s.get("ts1_lowest_freqs_cm"),
            "ts1_xyz": ts1_xyz,
            "n_hessians": s.get("n_hessians"),
            "irc": {"frames": irc_frames, "s": s_signed, "e_rel": irc_rel, "ts_frame": ts_index,
                    "curvature": curvature},
            "branches": branches,
        }))
    result.summary["settings"] = settings
    return result


_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&'
    'family=Source+Sans+3:wght@400;600&family=Geist+Mono&display=swap" rel="stylesheet">'
)


def render_vri_html(result: VriResult, title: str = "VRI search", show_atom_indices: bool = False) -> str:
    """A two-panel page: the energy surface of each branch that has a second
    product (clickable markers and paths) and a molecule viewer."""
    from mepd.viz import _3DMOL_CDN_SCRIPT

    if not result.candidates:
        raise ValueError("No VRI scan found in this directory (need summary.json + projected_freqs.json).")
    maps = []
    ts1_xyz = None
    for c in result.candidates:
        pl = c.payload
        ts1_xyz = ts1_xyz or pl.get("ts1_xyz")
        for name in ("reverse", "forward"):
            b = pl["branches"].get(name)
            if b and b.get("surface"):
                prefix = "" if pl["label"] == "input" else f"TS1 {pl['label']}, "
                maps.append({"label": f"{prefix}{name} branch", "verdict": b.get("verdict"), "surface": b["surface"],
                             "ircFrames": pl["irc"]["frames"], "tsFrame": pl["irc"]["ts_frame"],
                             "checks": b.get("checks_run")})
    data = {"title": title, "verdict": result.summary.get("verdict"), "maps": maps, "ts1_xyz": ts1_xyz}
    payload = json.dumps(data).replace("</", "<\\/")
    return (_PAGE.replace("{{TITLE}}", html.escape(title)).replace("{{FONTS}}", _FONTS)
            .replace("{{THREEDMOL}}", _3DMOL_CDN_SCRIPT).replace("{{DATA}}", payload)
            .replace("{{ATOM_INDICES}}", "true" if show_atom_indices else "false"))


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
{{FONTS}}
{{THREEDMOL}}
<style>
  :root { --paper: #f8f7f3; --ink: #17324a; --muted: #5d6b78; --blue: #3868b8; --rule: #dcd9d0; --good: #2f7d4f; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 "Source Sans 3", -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1320px; margin: 0 auto; padding: 22px 20px 40px; }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 18px; margin-bottom: 12px; }
  h1 { font: 600 26px/1.2 "Source Serif 4", Georgia, serif; margin: 0; }
  .pill { display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 12.5px; font-weight: 600; background: #e8edf5; }
  .pill.bifurcation { background: #dcefe3; color: var(--good); }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
  .tabs button { border: 1px solid var(--rule); background: #fff; color: var(--ink); padding: 4px 11px; border-radius: 4px; font: inherit; cursor: pointer; }
  .tabs button.on { border-color: var(--blue); background: #e7eefb; }
  .grid { display: grid; grid-template-columns: minmax(0, 7fr) minmax(0, 5fr); gap: 16px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .panel { background: #fff; border: 1px solid var(--rule); border-radius: 4px; padding: 12px; }
  .surf { position: relative; width: 100%; }
  .surf canvas { width: 100%; height: auto; display: block; }
  .surf svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  .surf .tip { position: absolute; pointer-events: none; background: rgba(23,50,74,.92); color: #fff; font-size: 12px;
               padding: 3px 7px; border-radius: 3px; white-space: nowrap; display: none; }
  svg text { fill: var(--muted); font-size: 11px; font-family: "Source Sans 3", sans-serif; }
  .legend { font-size: 13px; color: var(--ink); margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }
  .legend .row { display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: center; }
  .legend .item { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
  .legend svg { flex: none; }
  .legend canvas { width: 160px; height: 10px; border-radius: 2px; }
  .legend .muted { color: var(--muted); }
  #viewer { width: 100%; height: 460px; position: relative; }
  #caption { font-weight: 600; margin-bottom: 4px; min-height: 1.5em; }
  #energy { color: var(--muted); font-size: 14px; }
  .empty { color: var(--muted); padding: 40px 10px; text-align: center; }
  .controls { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
  .checks { font-size: 14px; color: var(--ink); margin: -2px 0 10px; }
  .checks .muted { color: var(--muted); }
  .viewer-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
  .labels-toggle { font-size: 13.5px; color: var(--muted); white-space: nowrap; cursor: pointer; }
  .axes-help { margin-top: 8px; font-size: 13.5px; }
  .axes-help summary { cursor: pointer; color: var(--blue); font-weight: 600; }
  .axes-help p { margin: 6px 0; max-width: 80ch; }
  .axes-help code { font-family: "Geist Mono", ui-monospace, monospace; font-size: 12.5px; }
  .controls input[type=range] { flex: 1; }
  .toggles { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 13.5px; color: var(--muted); margin-bottom: 6px; }
  .toggles:empty { display: none; }
  .toggles label { cursor: pointer; white-space: nowrap; }
  .controls button, .controls select { border: 1px solid var(--rule); background: #fff; color: var(--ink); padding: 4px 10px;
                                       border-radius: 4px; font: inherit; cursor: pointer; }
</style>
</head>
<body>
<main>
  <header><h1 id="title"></h1><span id="verdict" class="pill"></span></header>
  <div class="tabs" id="tabs"></div>
  <div id="checks" class="checks"></div>
  <div class="grid">
    <section class="panel" aria-label="Energy surface"><div id="toggles" class="toggles"></div><div id="map"></div><div class="legend" id="legend"></div>
      <details class="axes-help"><summary>How the axes are computed</summary><div id="axesHelp"></div></details></section>
    <section class="panel" aria-label="Molecule">
      <div class="viewer-head"><div><div id="caption"></div><div id="energy"></div></div>
        <label class="labels-toggle"><input type="checkbox" id="labelsToggle"> Show atom labels</label></div>
      <div id="viewer"></div>
      <div id="pathControls" class="controls">
        <select id="pathSelect" aria-label="Path to scrub"></select>
        <button id="play" type="button">Play</button>
        <input id="slider" type="range" min="0" max="0" value="0" step="1" aria-label="Frame along the path">
      </div>
    </section>
  </div>
</main>
<script>
const DATA = {{DATA}};
const SHOW_IDX = {{ATOM_INDICES}};
let viewer = null;
function fmt(x, d = 1) { return (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d); }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }
let cursorFn = null;  // (q | null) -> moves the dot on the current map
let paths = [], pathIdx = 0, frameIdx = 0, playTimer = null;
let showLabels = SHOW_IDX, model = null, zoomed = false;
let pathHook = null;  // called after the selected path changes (redraws the highlighted check path)
const showCheck = {basin: false, trajectory: false};
function outcomeColor(lab) {
  return lab === "P1" ? "#2166ac" : lab === "P2" ? "#d6452a" : (lab === "R" || lab === "recrossed") ? "#6b6b6b"
    : lab === "failed" ? "#111111" : "#8e44ad";
}
function toXyz(symbols, X) {
  let t = `${symbols.length}\n\n`;
  for (let i = 0; i < symbols.length; i++) t += `${symbols[i]} ${X[3 * i]} ${X[3 * i + 1]} ${X[3 * i + 2]}\n`;
  return t;
}
function checkPathName(cp) {
  const i = cp.info || {}, side = String(i.side || "");
  if (cp.kind === "basin")
    return `Push at s ${fmt(i.s, 1)}, mode ${side.slice(1)} ${side[0] || ""}, ${fmt(i.displacement, 1)} bohr → ${cp.label}`;
  return `Trajectory ${i.index} → ${cp.label}` + (i.commit_fs !== null && i.commit_fs !== undefined ? ` (settled by ${Math.round(i.commit_fs)} fs)` : "");
}
function drawLabels() {
  viewer.removeAllLabels();
  if (showLabels && model) model.selectedAtoms({}).forEach((a, i) => viewer.addLabel(`${a.elem}${i}`, {
    position: {x: a.x, y: a.y, z: a.z}, fontSize: 11, fontColor: "#17324a",
    backgroundColor: "white", backgroundOpacity: 0.75, borderThickness: 0, inFront: true, alignment: "center"}));
  viewer.render();
}
document.getElementById("labelsToggle").checked = SHOW_IDX;
document.getElementById("labelsToggle").addEventListener("change", (e) => { showLabels = e.target.checked; if (viewer) drawLabels(); });
function showStructure(xyz, caption, energy, q) {
  if (!viewer) viewer = $3Dmol.createViewer(document.getElementById("viewer"), {backgroundColor: "white"});
  viewer.clear();
  model = viewer.addModel(xyz, "xyz");
  viewer.setStyle({}, {stick: {radius: 0.14}, sphere: {scale: 0.25}});
  if (!zoomed) { viewer.zoomTo(); zoomed = true; }  // keep the user's rotation/zoom while scrubbing
  drawLabels();
  document.getElementById("caption").textContent = caption;
  document.getElementById("energy").textContent = energy === null || energy === undefined ? "" : `${fmt(energy, 1)} kcal/mol relative to TS1`;
  if (cursorFn) cursorFn(q || null);
}
function stopPlay() { if (playTimer) { clearInterval(playTimer); playTimer = null; document.getElementById("play").textContent = "Play"; } }
function showFrame(i) {
  const path = paths[pathIdx]; if (!path) return;
  frameIdx = Math.max(0, Math.min(i, path.frames.length - 1));
  document.getElementById("slider").value = String(frameIdx);
  const f = path.frames[frameIdx];
  if (!f.xyz && f.X) f.xyz = toXyz(path.symbols, f.X);
  showStructure(f.xyz, `${path.name}, frame ${frameIdx + 1} of ${path.frames.length}`, f.e, f.q);
  if (path.note) document.getElementById("energy").textContent = path.note;
}
function selectPath(i, start) {
  stopPlay(); pathIdx = i; const path = paths[i];
  const slider = document.getElementById("slider"); slider.max = String(path.frames.length - 1);
  document.getElementById("pathSelect").value = String(i);
  showFrame(start === undefined ? 0 : start);
  if (pathHook) pathHook();
}
document.getElementById("slider").addEventListener("input", (e) => { stopPlay(); showFrame(parseInt(e.target.value, 10)); });
document.getElementById("pathSelect").addEventListener("change", (e) => {
  const i = parseInt(e.target.value, 10); selectPath(i);
  if (paths[i].loop) document.getElementById("play").click();  // the ridge motion only makes sense moving
});
document.getElementById("play").addEventListener("click", () => {
  if (playTimer) return stopPlay();
  const path = paths[pathIdx]; if (!path) return;
  if (frameIdx >= path.frames.length - 1) showFrame(0);
  document.getElementById("play").textContent = "Pause";
  playTimer = setInterval(() => {
    const cur = paths[pathIdx];
    if (frameIdx >= cur.frames.length - 1) { if (cur.loop) return showFrame(0); return stopPlay(); }
    showFrame(frameIdx + 1);
  }, path.loop ? 90 : 40);
});
const PALETTE = [[23,50,74],[31,82,120],[45,118,150],[80,152,160],[135,183,160],[196,210,160],[241,229,178],[250,244,222]];
function color(t) {
  t = Math.max(0, Math.min(1, t)) * (PALETTE.length - 1);
  const i = Math.min(PALETTE.length - 2, Math.floor(t)), f = t - i, a = PALETTE[i], b = PALETTE[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function niceTicks(a, b, n) {
  const span = b - a || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = []; for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) out.push(+t.toFixed(10)); return out;
}
function bilinear(G, xs, ys, x, y) {
  const nx = xs.length, ny = ys.length;
  const fx = (x - xs[0]) / (xs[nx - 1] - xs[0]) * (nx - 1), fy = (y - ys[0]) / (ys[ny - 1] - ys[0]) * (ny - 1);
  const i = Math.max(0, Math.min(nx - 2, Math.floor(fx))), j = Math.max(0, Math.min(ny - 2, Math.floor(fy)));
  const tx = fx - i, ty = fy - j, v = [G[j][i], G[j][i + 1], G[j + 1][i], G[j + 1][i + 1]];
  if (v.some((q) => q === null)) return null;
  return (1 - tx) * (1 - ty) * v[0] + tx * (1 - ty) * v[1] + (1 - tx) * ty * v[2] + tx * ty * v[3];
}
function contourSegments(G, xs, ys, lev) {
  const seg = [];
  for (let j = 0; j < ys.length - 1; j++) for (let i = 0; i < xs.length - 1; i++) {
    const c = [[xs[i], ys[j], G[j][i]], [xs[i + 1], ys[j], G[j][i + 1]], [xs[i + 1], ys[j + 1], G[j + 1][i + 1]], [xs[i], ys[j + 1], G[j + 1][i]]];
    if (c.some((p) => p[2] === null)) continue;
    const pts = [];
    for (let k = 0; k < 4; k++) { const a = c[k], b = c[(k + 1) % 4];
      if ((a[2] - lev) * (b[2] - lev) < 0) { const t = (lev - a[2]) / (b[2] - a[2]); pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]); } }
    if (pts.length === 2) seg.push(pts); else if (pts.length === 4) { seg.push([pts[0], pts[1]]); seg.push([pts[2], pts[3]]); }
  }
  return seg;
}
const LINES = [["IRC through TS1", "#ffffff", 2.4, ""], ["IRC through TS2", "#f0a24b", 2.2, ""],
               ["TS1 → TS2 path", "#ffffff", 1.5, "5 4"], ["P1 → P2 path", "#f0a24b", 1.5, "5 4"]];
function drawChecks(m) {
  const c = m.checks, el = document.getElementById("checks");
  if (!c) { el.innerHTML = ""; return; }
  const parts = [];
  if (c.trajectories) {
    const k = c.trajectories.counts, prods = Object.keys(k).filter((x) => /^P\d+$/.test(x)).sort((a, b) => +a.slice(1) - +b.slice(1));
    const nP = prods.reduce((a, x) => a + k[x], 0) || 1;
    parts.push(`<b>Trajectories from TS1:</b> ` + prods.map((x) => `${x} ${k[x]} (${Math.round(100 * k[x] / nP)}%)`).join(" · ")
      + ` <span class="muted">· recrossed ${k.recrossed || 0}${k.other ? ` · other ${k.other}` : ""}${k.failed ? ` · failed ${k.failed}` : ""} of ${c.trajectories.n}</span>`);
  }
  if (c.basin) {
    const k = c.basin.counts, prods = Object.keys(k).filter((x) => k[x] > 0 && (/^P\d+$/.test(x) || x.startsWith("other")));
    const desc = Object.entries(k).map(([x, v]) => `${x.replace("other_", "other product ")} ${v}`).join(", ");
    parts.push(`<b>Sideways pushes:</b> ${prods.length >= 2 ? "drain into different products" : "all drain into one product"} (${desc})`);
  }
  const extra = (c.products || []).filter((p) => !["P1", "P2"].includes(p.label));
  if (extra.length) parts.push(`<b>Further products:</b> ` + extra.map((p) => `${p.label} ${esc(p.smiles || "")}`
      + (p.equivalent_to ? ` (same molecule as ${p.equivalent_to})` : "")).join("; "));
  el.innerHTML = parts.join("  &nbsp;·&nbsp;  ");
}
const MARK_INFO = {
  R: ["#17324a", "the other end of the IRC from TS1"], TS1: ["#b3452f", "transition state"],
  VRT: ["#c0561f", "where the IRC's ridge starts"], VRI: ["#7a3fa0", "exact valley-ridge inflection point"],
  TS2: ["#b3452f", "saddle between P1 and P2"], P1: ["#17324a", "product the IRC reaches"], P2: ["#17324a", "second product"],
};
function lineSwatch(color, dash) {
  return `<svg width="34" height="12" aria-hidden="true"><line x1="1" y1="6" x2="33" y2="6" stroke="rgba(23,50,74,.5)" stroke-width="4.5" stroke-dasharray="${dash}"/>`
    + `<line x1="1" y1="6" x2="33" y2="6" stroke="${color}" stroke-width="2.6" stroke-dasharray="${dash}"/></svg>`;
}
function drawLegend(s, marks, vmin, vmax) {
  const has = (role) => s.points.some((p) => p.role === role);
  const lines = [["IRC from TS1", "#ffffff", "", has("IRC through TS1")], ["IRC from TS2", "#f0a24b", "", has("IRC through TS2")],
                 ["TS1 → TS2 path", "#ffffff", "5 4", has("TS1 → TS2 path")], ["P1 → P2 path", "#f0a24b", "5 4", has("P1 → P2 path")]];
  const lineItems = lines.filter((l) => l[3]).map(([label, c, dash]) => `<span class="item">${lineSwatch(c, dash)}${label}</span>`);
  lineItems.push(`<span class="item"><svg width="22" height="22" aria-hidden="true"><circle cx="11" cy="11" r="8" fill="none" stroke="#ffd23f" stroke-width="3" style="filter:drop-shadow(0 0 1px #17324a)"/></svg>structure shown on the right</span>`);
  if (s.instant) lineItems.push(`<span class="item"><svg width="22" height="14" aria-hidden="true"><defs><pattern id="hatchL" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#f4f2ec"/><rect width="1.5" height="6" fill="#d6d1c7"/></pattern></defs><rect width="22" height="14" rx="2" fill="url(#hatchL)" stroke="#dcd9d0"/></svg>no computed data nearby</span>`);
  const shown = (s.check_paths || []).filter((cp) => showCheck[cp.kind]);
  if (shown.length) {
    const labs = [...new Set(shown.map((cp) => cp.label))].sort();
    const name = (l) => l === "R" ? "back to R" : l === "recrossed" ? "recrossed TS1" : l === "failed" ? "failed" : l;
    lineItems.push(`<span class="item muted">${showCheck.basin && showCheck.trajectory ? "Basin descents and trajectories" : showCheck.basin ? "Basin descents" : "Trajectories"} ending in:</span>`
      + labs.map((l) => `<span class="item"><svg width="30" height="10" aria-hidden="true"><line x1="1" y1="5" x2="29" y2="5" stroke="${outcomeColor(l)}" stroke-width="2.2"${l === "failed" ? ' stroke-dasharray="3 3"' : ""}/></svg>${name(l)}</span>`).join(""));
  }
  const present = new Set(marks.map((m) => m.role));
  const markItems = Object.entries(MARK_INFO).filter(([r]) => present.has(r)).map(([r, [c, desc]]) =>
    `<span class="item"><svg width="24" height="24" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="${c}" stroke="#fff" stroke-width="1.5"/><text x="12" y="15.5" text-anchor="middle" style="fill:#fff;font-size:8.5px;font-weight:600">${r}</text></svg><span><b>${r}</b> <span class="muted">${desc}</span></span></span>`);
  document.getElementById("legend").innerHTML =
    `<div class="row"><span class="item"><span class="muted">Energy</span> ${fmt(vmin, 0)} <canvas id="legendBar" width="160" height="10"></canvas> ${fmt(vmax, 0)} <span class="muted">kcal/mol relative to TS1</span></span></div>`
    + `<div class="row">${lineItems.join("")}</div>`
    + `<div class="row">${markItems.join("")}</div>`
    + `<div class="row muted">Drag the slider or press Play to move along a path; click a marker or the map to show a structure.</div>`;
  const bar = document.getElementById("legendBar").getContext("2d");
  for (let k = 0; k < 160; k++) { const c = color(k / 159); bar.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; bar.fillRect(k, 0, 1, 10); }
}
function drawMap(m) {
  drawChecks(m);
  const s = m.surface, W = 640, H = 500, L = 58, R = 14, T = 12, B = 44;
  const el = document.getElementById("map");
  el.innerHTML = `<div class="surf"><canvas width="${W}" height="${H}"></canvas><svg viewBox="0 0 ${W} ${H}"></svg><div class="tip"></div></div>`;
  const cv = el.querySelector("canvas"), ctx = cv.getContext("2d"), svg = el.querySelector("svg"), tip = el.querySelector(".tip");
  const xs = s.x, ys = s.y, x0 = xs[0], x1 = xs[xs.length - 1], y0 = ys[0], y1 = ys[ys.length - 1];
  const marks = s.points.filter((p) => ["R", "TS1", "VRT", "VRI", "TS2", "P1", "P2"].includes(p.role) && p.xyz);
  const vals = s.E.flat().filter((v) => v !== null);
  const vmin = Math.min(...vals), vmax = Math.min(Math.max(...vals), Math.max(15, ...marks.map((q) => q.e + 10)));
  const E = s.E.map((row, j) => row.map((v, i) => (v === null || (s.ok && !s.ok[j][i])) ? null : Math.min(v, vmax)));
  const sx = (x) => L + (x - x0) / (x1 - x0) * (W - L - R), sy = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const ix = (px) => x0 + (px - L) / (W - L - R) * (x1 - x0), iy = (py) => y0 + (1 - (py - T) / (H - T - B)) * (y1 - y0);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
  const img = ctx.createImageData(W - L - R, H - T - B);
  for (let py = 0; py < H - T - B; py++) for (let px = 0; px < W - L - R; px++) {
    const e = bilinear(E, xs, ys, ix(px + L), iy(py + T)), o = 4 * (py * (W - L - R) + px);
    if (e === null) { img.data.set(((px + py) % 8) < 2 ? [214, 209, 199, 255] : [244, 242, 236, 255], o); continue; }
    const c = color((e - vmin) / (vmax - vmin)); img.data.set([c[0], c[1], c[2], 255], o);
  }
  ctx.putImageData(img, L, T);
  const ticks = niceTicks(vmin, vmax, 14), step = (ticks[1] - ticks[0]) || 5;
  let g = `<defs><clipPath id="plotClip"><rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}"/></clipPath></defs>`;
  for (let lev = Math.ceil(vmin / step) * step; lev < vmax; lev += step)
    for (const [a, b] of contourSegments(E, xs, ys, lev))
      g += `<line x1="${sx(a[0]).toFixed(1)}" y1="${sy(a[1]).toFixed(1)}" x2="${sx(b[0]).toFixed(1)}" y2="${sy(b[1]).toFixed(1)}" stroke="rgba(23,50,74,.3)" stroke-width="0.7"/>`;
  for (const t of niceTicks(x0, x1, 6)) g += `<text x="${sx(t)}" y="${H - B + 16}" text-anchor="middle">${fmt(t, 1)}</text>`;
  for (const t of niceTicks(y0, y1, 6)) g += `<text x="${L - 6}" y="${sy(t) + 4}" text-anchor="end">${fmt(t, 1)}</text>`;
  const ax = s.axes || {}, b1 = (ax.bonds_p1 || []), b2 = (ax.bonds_p2 || []);
  const dist = (bonds) => bonds.length === 1 ? `d(${bonds[0]})` : `mean d(${bonds.join(", ")})`;
  const yTitle = b1.length && b2.length ? `${dist(b2)} − ${dist(b1)} (Å)   P2 ↓  ↑ P1` : "which product (Å)   P2 ↓  ↑ P1";
  g += `<text x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">distance along TS1's reaction direction (Å):   R ←  → products</text>`
     + `<text x="14" y="${(T + H - B) / 2}" transform="rotate(-90 14 ${(T + H - B) / 2})" text-anchor="middle">${esc(yTitle)}</text>`;
  document.getElementById("axesHelp").innerHTML =
    `<p><b>x: how far through TS1.</b> The IRC leaves TS1 along one direction in atomic coordinates, TS1's reaction direction (its imaginary vibration). `
    + `x is how far a structure is displaced from TS1 along that direction, in Å: 0 at TS1, negative toward R, positive toward the products. `
    + `Computed as <code>x = u · (r − r<sub>TS1</sub>)</code>, with r all atom positions and u the unit vector along the IRC at TS1.</p>`
    + (b1.length && b2.length
      ? `<p><b>y: which product.</b> P1 and P2 differ in a few bonds: P1 has ${esc(b1.join(", "))}, P2 has ${esc(b2.join(", "))}. `
        + `y is the length of P2's bond${b2.length > 1 ? "s (averaged)" : ""} minus the length of P1's bond${b1.length > 1 ? "s (averaged)" : ""}, in Å. `
        + `At P1 its own bond is short and P2's is long, so y is positive; at P2 it is negative; structures on the ridge between them sit near 0.</p>`
      : "")
    + `<p><b>Colors</b> are energies interpolated from the computed structures on the lines (their energies and gradients)`
    + ((s.check_paths || []).some((cp) => cp.kind === "basin") ? `, including the basin-test descents` : "")
    + `; hatched areas are too far from any computed structure to show. `
    + `A 2D map of a many-atom molecule is a projection: it is exact at the computed points and approximate between them.</p>`;
  g += `<g id="checkPaths" clip-path="url(#plotClip)"></g>`;
  for (const [role, colr, width, dash] of LINES) {
    const pts = s.points.filter((p) => p.role === role).map((p) => `${sx(p.q[0]).toFixed(1)},${sy(p.q[1]).toFixed(1)}`).join(" ");
    if (pts) g += `<polyline points="${pts}" fill="none" stroke="rgba(23,50,74,.5)" stroke-width="${width + 1.6}" stroke-dasharray="${dash}"/>`
                + `<polyline points="${pts}" fill="none" stroke="${colr}" stroke-width="${width}" stroke-dasharray="${dash}"/>`;
  }
  for (const q of marks) {
    const X = sx(q.q[0]), Y = sy(q.q[1]), fill = (q.role.startsWith("P") || q.role === "R") ? "#17324a"
      : q.role === "VRT" ? "#c0561f" : q.role === "VRI" ? "#7a3fa0" : "#b3452f";
    g += `<g data-mark="${q.role}" style="cursor:pointer"><circle cx="${X}" cy="${Y}" r="11" fill="${fill}" stroke="#fff" stroke-width="1.5"/>`
       + `<text x="${X}" y="${Y + 3.5}" text-anchor="middle" style="fill:#fff;font-size:9.5px;font-weight:600;pointer-events:none">${q.role}</text>`
       + `<text x="${X + 14}" y="${Y - 9}" style="fill:#17324a;font-size:11.5px;font-weight:600;paint-order:stroke;stroke:#fff;stroke-width:3px;pointer-events:none">${fmt(q.e, 0)}</text></g>`;
  }
  g += `<circle id="cursor" r="13" fill="none" stroke="#ffd23f" stroke-width="3.5" style="pointer-events:none;display:none;filter:drop-shadow(0 0 1.5px #17324a)"/>`;
  svg.innerHTML = g;
  const cursor = svg.querySelector("#cursor");
  cursorFn = (q) => { if (!q) { cursor.style.display = "none"; return; }
    cursor.style.display = ""; cursor.setAttribute("cx", sx(q[0])); cursor.setAttribute("cy", sy(q[1])); };
  // scrubbable paths: the IRC from TS1 (every frame has a map position) and the IRC from TS2
  paths = [];
  const irc1 = s.points.filter((p) => p.role === "IRC through TS1" && p.frame !== undefined).sort((a, b) => a.frame - b.frame);
  if (irc1.length) paths.push({name: "IRC from TS1", frames: irc1.map((p) => ({xyz: m.ircFrames[p.frame], e: p.e, q: p.q}))});
  const irc2 = s.points.filter((p) => p.role === "IRC through TS2" && p.xyz);
  if (irc2.length) paths.push({name: "IRC from TS2", frames: irc2.map((p) => ({xyz: p.xyz, e: p.e, q: p.q}))});
  const ridge = s.points.filter((p) => p.role === "ridge mode" && p.xyz);
  if (ridge.length) paths.push({name: "Ridge motion at the VRT", loop: true,
    note: "sideways motion that becomes unstable at the VRT (energies not computed)",
    frames: ridge.map((p) => ({xyz: p.xyz, e: null, q: p.q}))});
  // Basin-test descents and trajectories (optional overlays, also scrubbable)
  const sel = document.getElementById("pathSelect");
  const basePaths = paths.slice(), cps = s.check_paths || [], cpG = svg.querySelector("#checkPaths");
  const cpPaths = cps.map((cp, ci) => ({name: checkPathName(cp), cp: ci, symbols: s.symbols,
    frames: cp.q.map((q, k) => ({X: cp.X[k], e: cp.e[k], q}))}));
  const tsPos0 = irc1.findIndex((p) => p.frame === m.tsFrame);
  function buildPathList() {
    const cur = paths[pathIdx];
    paths = basePaths.slice();
    let html = basePaths.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("");
    for (const [kind, label] of [["basin", "Basin test (steepest descent from sideways pushes)"], ["trajectory", "Trajectories from TS1"]]) {
      if (!showCheck[kind]) continue;
      const opts = [];
      cpPaths.forEach((p, ci) => { if (cps[ci].kind === kind) { opts.push(`<option value="${paths.length}">${esc(p.name)}</option>`); paths.push(p); } });
      if (opts.length) html += `<optgroup label="${label}">${opts.join("")}</optgroup>`;
    }
    sel.innerHTML = html;
    const ni = paths.indexOf(cur);
    if (ni >= 0) { pathIdx = ni; sel.value = String(ni); }
    else if (paths.length) selectPath(0, tsPos0 >= 0 ? tsPos0 : 0);
  }
  function drawCheckLines() {
    const selCp = paths[pathIdx] ? paths[pathIdx].cp : undefined;
    const order = cps.map((cp, ci) => ci).filter((ci) => showCheck[cps[ci].kind]).sort((a, b) => (a === selCp) - (b === selCp));
    let h = "";
    for (const ci of order) {
      const cp = cps[ci], on = ci === selCp;
      const pts = cp.q.map((q) => `${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join(" ");
      if (on) h += `<polyline points="${pts}" fill="none" stroke="#fff" stroke-width="5.4"/>`;
      h += `<polyline points="${pts}" fill="none" stroke="${outcomeColor(cp.label)}" stroke-width="${on ? 3 : 1.2}" stroke-opacity="${on ? 1 : 0.8}"${cp.label === "failed" ? ' stroke-dasharray="3 3"' : ""}/>`
         + `<polyline data-cp="${ci}" points="${pts}" fill="none" stroke="transparent" stroke-width="7" style="cursor:pointer;pointer-events:stroke"/>`;
    }
    cpG.innerHTML = h;
  }
  pathHook = drawCheckLines;
  const toggles = document.getElementById("toggles");
  toggles.innerHTML = [["basin", "Basin-test paths"], ["trajectory", "Trajectories"]].map(([kind, label]) => {
    const n = cps.filter((cp) => cp.kind === kind).length;
    return n ? `<label><input type="checkbox" data-kind="${kind}"${showCheck[kind] ? " checked" : ""}> ${label} (${n})</label>` : "";
  }).join("");
  toggles.querySelectorAll("input").forEach((box) => box.addEventListener("change", () => {
    showCheck[box.dataset.kind] = box.checked;
    buildPathList(); drawCheckLines(); drawLegend(s, marks, vmin, vmax);
  }));
  buildPathList();
  document.getElementById("pathControls").style.display = paths.length ? "" : "none";
  const toData = (ev) => { const r = svg.getBoundingClientRect(); return [ix((ev.clientX - r.left) / r.width * W), iy((ev.clientY - r.top) / r.height * H)]; };
  svg.addEventListener("mousemove", (ev) => {
    const [x, y] = toData(ev), r = svg.getBoundingClientRect();
    if (x < x0 || x > x1 || y < y0 || y > y1) { tip.style.display = "none"; return; }
    const e = bilinear(E, xs, ys, x, y);
    tip.style.display = "block"; tip.style.left = `${ev.clientX - r.left + 12}px`; tip.style.top = `${ev.clientY - r.top + 12}px`;
    tip.textContent = e === null ? "no computed data here" : `${fmt(e, 1)} kcal/mol`;
  });
  svg.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  svg.addEventListener("click", (ev) => {
    const cpHit = ev.target.closest("[data-cp]");
    if (cpHit) { const i = paths.indexOf(cpPaths[+cpHit.getAttribute("data-cp")]); if (i >= 0) { selectPath(i); document.getElementById("play").click(); } return; }
    const mk = ev.target.closest("[data-mark]");
    if (mk) { stopPlay(); const q = marks.find((p) => p.role === mk.getAttribute("data-mark")); showStructure(q.xyz, q.role, q.e, q.q); return; }
    stopPlay();
    const [x, y] = toData(ev);
    if (s.instant) {
      let best = null, bd = Infinity;
      for (const p of s.points) { if (!p.xyz && p.frame === undefined) continue; const d = (p.q[0] - x) ** 2 + (p.q[1] - y) ** 2; if (d < bd) { bd = d; best = p; } }
      if (best && best.frame !== undefined) { selectPath(0, best.frame); return; }
      if (best) showStructure(best.xyz, `${best.role} (nearest computed point)`, best.e, best.q);
      return;
    }
    let bi = -1, bd = Infinity;
    xs.forEach((xv, i) => ys.forEach((yv, j) => { const d = (xv - x) ** 2 + (yv - y) ** 2; if (d < bd && s.E[j][i] !== null) { bd = d; bi = j * xs.length + i; } }));
    if (bi >= 0 && s.grid_xyz && s.grid_xyz[bi]) showStructure(s.grid_xyz[bi], "Relaxed structure at this point", s.E[Math.floor(bi / xs.length)][bi % xs.length]);
  });
  drawLegend(s, marks, vmin, vmax);
  if (paths.length) {  // open on TS1, positioned on the IRC so Play runs toward the products
    selectPath(0, tsPos0 >= 0 ? tsPos0 : 0);
  } else {
    const ts1 = marks.find((p) => p.role === "TS1");
    if (ts1) showStructure(ts1.xyz, "TS1", 0, ts1.q);
  }
}
document.getElementById("title").textContent = DATA.title;
const v = document.getElementById("verdict"); v.textContent = DATA.verdict || ""; v.classList.add(DATA.verdict || "");
const tabs = document.getElementById("tabs");
if (!DATA.maps.length) {
  pathHook = null;
  document.getElementById("map").innerHTML = '<div class="empty">No bifurcation was found, so there is no second product to map.</div>';
  document.getElementById("pathControls").style.display = "none";
  if (DATA.ts1_xyz) showStructure(DATA.ts1_xyz, "TS1", 0);
} else {
  if (DATA.maps.length > 1) DATA.maps.forEach((m, i) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = m.label;
    b.addEventListener("click", () => { tabs.querySelectorAll("button").forEach((x, j) => x.classList.toggle("on", j === i)); drawMap(m); });
    tabs.appendChild(b);
  });
  if (tabs.firstChild) tabs.firstChild.classList.add("on");
  drawMap(DATA.maps[0]);
}
</script>
</body>
</html>
"""
