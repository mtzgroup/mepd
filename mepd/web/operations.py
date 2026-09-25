"""Registry of the calculations the web UI can launch.

Each `Operation` declares:

* what it acts on (`target`): one structure, a pair (an edge, or two
  selected structures), or a set of structures;
* its parameters, as a pydantic model whose fields carry the CLI flag they
  map to (`cli=`) plus UI hints (`group=`, `advanced=`). The UI renders its
  forms straight from `model_json_schema()`, so adding a knob here is the
  only change needed to expose it;
* how to turn (targets, params, profile) into a `mepd ...` argv.

Adding a new mepd capability to the web UI = one params model + one
`Operation(...)` entry below (and, if its output layout is new, a collector
in `mepd.web.results`).
"""

from __future__ import annotations

import functools
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from mepd.atom_mapping_metrics import METRICS as _ATOM_MAPPING_METRICS
from mepd.web.workspace import Workspace, WorkspaceError, is_ts

Target = Literal["structure", "pair", "set", "job"]  # "job": a follow-up on another job's output
# Subscripting with a tuple lists its items, so this stays in step with
# mepd's metric registry instead of drifting behind it (it did: the UI
# offered three of the four for a while). `Literal[*METRICS]` would be
# tidier but needs 3.11, and this package still supports 3.10.
AtomMappingMetric = Literal[_ATOM_MAPPING_METRICS]


def P(default, title: str, help: str = "", *, cli: Optional[str] = None, kind: str = "value",
      group: str = "Basic", advanced: bool = False, requires: Optional[str] = None, **kw):
    """A parameter field. `kind`:
    * "value"  -> `--flag <value>` (omitted when the value is None)
    * "switch" -> `--flag` when true, nothing when false
    * "toggle" -> `--flag` / `--no-flag`
    * "custom" -> handled by the operation's argv builder

    `requires` names the field this one only matters for -- "other" (other
    is truthy) or "other=a|b" -- so it is left off the command line and
    hidden in the form otherwise.
    """
    extra = {"cli": cli, "cli_kind": kind, "group": group, "advanced": advanced, "requires": requires}
    return Field(default, title=title, description=help, json_schema_extra=extra, **kw)


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


def requirement_met(params: Params, requires: Optional[str]) -> bool:
    if not requires:
        return True
    name, _, allowed = requires.partition("=")
    value = getattr(params, name)
    return str(value) in allowed.split("|") if allowed else bool(value)


def generic_flags(params: Params) -> list[str]:
    argv: list[str] = []
    for name, finfo in type(params).model_fields.items():
        extra = finfo.json_schema_extra or {}
        cli, kind = extra.get("cli"), extra.get("cli_kind", "value")
        if not cli or kind == "custom" or not requirement_met(params, extra.get("requires")):
            continue
        value = getattr(params, name)
        if kind == "switch":
            if value:
                argv.append(cli)
        elif kind == "toggle":
            argv.append(cli if value else "--no-" + cli.removeprefix("--"))
        elif value is not None:
            argv += [cli, str(value)]
    return argv


# ---------------------------------------------------------------- params

EndpointMode = Literal["auto", "smiles", "xyz"]
_ENDPOINTS_HELP = (
    "How endpoints are handed to mepd. 'auto' passes the stored geometries once either "
    "structure has been minimized (so the path starts from those minima), and the SMILES "
    "only for two raw SMILES embeddings (mepd then builds an atom-mapped pair itself)."
)
_MAPPING_METRIC_HELP = (
    "How each candidate atom mapping is scored. geodesic-distance / path-rmsd: interpolate a path per "
    "candidate. gi-energy: adds one energy per candidate. endpoint-rmsd: aligned RMSD between the two "
    "endpoints, no interpolation (orders of magnitude cheaper, but blind to what happens along the path; "
    "experimental in mepd)."
)


class TsParams(Params):
    path_mode: Literal["recursive", "single", "parallel"] = P(
        "recursive", "Path search", "recursive: MSMEP splits multi-step paths into elementary "
        "steps. single: one path minimization. parallel: recursive with branches run concurrently.",
        kind="custom")
    use_tsopt: bool = P(True, "Optimize TS", "Refine each path's highest-energy image to a true saddle.",
                        cli="--use-tsopt", kind="switch")
    irc: bool = P(True, "Run IRC", "Follow each optimized TS downhill both ways (needs 'Optimize TS').",
                  cli="--irc", kind="switch")
    minimize_ends: Literal["auto", "yes", "no"] = P(
        "auto", "Minimize endpoints", "auto: minimize unless both endpoints are known minima (e.g. sampled "
        "minima or relaxed conformers). SMILES embeddings and uploaded geometries get minimized.", kind="custom")
    atom_mapping: bool = P(True, "Check atom mapping",
                           "Verify (and fix) the start/end atom correspondence before searching.",
                           cli="--atom-mapping", kind="switch")
    endpoints: EndpointMode = P("auto", "Endpoint source", _ENDPOINTS_HELP, kind="custom",
                                group="Endpoints", advanced=True)
    atom_mapping_candidates: int = P(200, "Mapping candidates", cli="--atom-mapping-candidates",
                                     group="Atom mapping", advanced=True, ge=1, requires="atom_mapping")
    atom_mapping_metric: AtomMappingMetric = P(
        "geodesic-distance", "Mapping metric", _MAPPING_METRIC_HELP,
        cli="--atom-mapping-metric", group="Atom mapping", advanced=True,
        requires="atom_mapping")
    validate_minima_with_hessian: bool = P(
        True, "Validate minima with Hessian", "Every intermediate minimum a recursive split proposes must "
        "have no imaginary frequency; a failing one is pushed along its unstable mode and re-optimized.",
        cli="--validate-minima-with-hessian", kind="toggle", requires="path_mode=recursive|parallel")
    hessian_minimum_frequency_cutoff: float = P(
        0.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.1, "Rescue push (bohr)", "First push along the unstable mode before re-optimizing; escalates to "
        "0.3 and 0.5 bohr if the rescue fails.", cli="--hessian-minima-rescue-displacement", group="Hessian validation", advanced=True,
        requires="validate_minima_with_hessian", gt=0)
    same_pair_split_limit: int = P(5, "Same-pair split limit", cli="--same-pair-split-limit",
                                   group="Recursive splitting", advanced=True, ge=0,
                                   requires="path_mode=recursive|parallel")
    parallel_workers: Optional[int] = P(None, "Parallel workers", cli="--parallel-workers",
                                        group="Recursive splitting", advanced=True, ge=1,
                                        requires="path_mode=parallel")
    network_completion: bool = P(False, "Network completion",
                                 "After the search, also connect intermediates the split tree found.",
                                 cli="--network-completion", kind="switch", group="Network completion",
                                 advanced=True)
    network_completion_mode: Literal["linear", "all-to-all"] = P(
        "linear", "Completion mode", cli="--network-completion-mode", group="Network completion", advanced=True,
        requires="network_completion")
    network_max_followups: int = P(25, "Max follow-ups", cli="--network-max-followups",
                                   group="Network completion", advanced=True, ge=0, requires="network_completion")


class ChannelsParams(Params):
    backend: Literal["rdkit", "crest"] = P("rdkit", "Conformer backend",
                                           "Where the reactant/product conformer pools come from.",
                                           cli="--backend")
    pairs_per_mechanism: int = P(3, "Searches per mechanism",
                                 "Best-scoring conformer pairs kept per bond-change mechanism (0 = all).",
                                 cli="--pairs-per-mechanism", ge=0)
    workers: int = P(4, "Workers", "Processes for atom mapping, path searches and TS/IRC.",
                     cli="--workers", ge=1)
    conformers_only: bool = P(False, "Conformers only", "Stop after conformer pools and pair selection.",
                              cli="--conformers-only", kind="switch")
    minimize_ends: bool = P(True, "Minimize endpoints", cli="--minimize-ends", kind="toggle",
                            group="Endpoints", advanced=True)
    endpoints: EndpointMode = P("auto", "Endpoint source", _ENDPOINTS_HELP, kind="custom",
                                group="Endpoints", advanced=True)
    n_conformers: int = P(0, "Max conformers", "0 = no cap.", cli="--n-conformers",
                          group="Conformers", advanced=True, ge=0)
    n_embed: int = P(0, "RDKit embeddings", "0 = automatic by size.", cli="--n-embed",
                     group="Conformers", advanced=True, ge=0, requires="backend=rdkit")
    rdkit_ewin: Optional[float] = P(None, "RDKit energy window (kcal/mol)", cli="--rdkit-ewin",
                                    group="Conformers", advanced=True, requires="backend=rdkit")
    crest_ewin: float = P(6.0, "CREST energy window (kcal/mol)", cli="--crest-ewin",
                          group="Conformers", advanced=True, requires="backend=crest")
    crest_method: Literal["--gfn2", "--gfnff", "--gfn2//gfnff"] = P(
        "--gfn2", "CREST method", cli="--crest-method", group="Conformers", advanced=True,
        requires="backend=crest")
    crest_timeout: float = P(3600.0, "CREST timeout (s)", cli="--crest-timeout", group="Conformers",
                             advanced=True, requires="backend=crest")
    rmsd_cutoff: float = P(0.5, "Dedup RMSD (bohr)", cli="--rmsd-cutoff", group="Conformers", advanced=True)
    random_seed: int = P(0, "Random seed", cli="--random-seed", group="Conformers", advanced=True)
    max_pairs: int = P(0, "Max pairs", "0 = no cap.", cli="--max-pairs", group="Pairs", advanced=True, ge=0)
    atom_mapping: bool = P(True, "Atom mapping per pair", cli="--atom-mapping", kind="toggle",
                           group="Pairs", advanced=True)
    atom_mapping_metric: AtomMappingMetric = P(
        "geodesic-distance", "Mapping metric", _MAPPING_METRIC_HELP,
        cli="--atom-mapping-metric", group="Pairs", advanced=True,
        requires="atom_mapping")
    validate_minima_with_hessian: bool = P(
        True, "Validate minima with Hessian", "Every intermediate minimum a recursive split proposes must "
        "have no imaginary frequency; a failing one is pushed along its unstable mode and re-optimized.",
        cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        0.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.1, "Rescue push (bohr)", "First push along the unstable mode before re-optimizing; escalates to "
        "0.3 and 0.5 bohr if the rescue fails.", cli="--hessian-minima-rescue-displacement", group="Hessian validation", advanced=True,
        requires="validate_minima_with_hessian", gt=0)


class TsOptParams(Params):
    irc: bool = P(True, "Run IRC", cli="--irc", kind="switch")


class HessianSampleParams(Params):
    amplitude_policy: Literal["fixed-cartesian", "energy"] = P(
        "fixed-cartesian", "Displacement policy",
        "fixed-cartesian: move every mode by the same RMS distance. energy: move each mode to a "
        "target harmonic energy.", cli="--amplitude-policy")
    dr: float = P(0.1, "Displacement (bohr RMS)", cli="--dr", gt=0, requires="amplitude_policy=fixed-cartesian")
    target_energy_kcal: float = P(25.0, "Target energy (kcal/mol)", cli="--target-energy-kcal", gt=0,
                                  requires="amplitude_policy=energy")
    max_candidates: int = P(100, "Max candidates", cli="--max-candidates", ge=1)
    imaginary_mode_amplitude: float = P(0.3, "Imaginary-mode amplitude (bohr)",
                                        cli="--imaginary-mode-amplitude", advanced=True, group="Advanced",
                                        requires="amplitude_policy=energy")
    maxiter: int = P(500, "Max optimizer iterations", cli="--maxiter", advanced=True, group="Advanced")
    validate_minima_with_hessian: bool = P(
        True, "Validate minima with Hessian", "Every minimum found must have no imaginary frequency; one "
        "that stopped on a saddle point is pushed along its unstable mode and re-optimized, and dropped "
        "(listed as rejected) if that fails too.", cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        0.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.1, "Rescue push (bohr)", "First push along the unstable mode; escalates to 0.3 and 0.5 bohr if the rescue fails.", cli="--hessian-minima-rescue-displacement", group="Hessian validation",
        advanced=True, requires="validate_minima_with_hessian", gt=0)


class HessianGlobalParams(Params):
    max_rounds: int = P(100, "Max rounds", cli="--max-rounds", ge=1)
    temperature: float = P(298.15, "Temperature (K)", "Metropolis acceptance temperature.",
                           cli="--temperature", gt=0)
    acceptance_baseline: Literal["connected", "seed", "running_best"] = P(
        "connected", "Acceptance baseline", cli="--acceptance-baseline")
    dr: float = P(0.1, "Displacement (bohr RMS)", cli="--dr", gt=0)
    max_candidates: int = P(100, "Max candidates per source", cli="--max-candidates", ge=1)
    full_dr_scan: bool = P(False, "Full displacement scan", cli="--full-dr-scan", kind="toggle",
                           advanced=True, group="Advanced")
    dr_scan_values: Optional[str] = P(None, "Scan values", "Comma-separated, e.g. 0.1,0.3,0.6",
                                      cli="--dr-scan-values", advanced=True, group="Advanced",
                                      requires="full_dr_scan")
    energy_tolerance_kcal: float = P(1e-4, "Energy tolerance (kcal/mol)", cli="--energy-tolerance-kcal",
                                     advanced=True, group="Advanced")
    random_seed: Optional[int] = P(None, "Random seed", cli="--random-seed", advanced=True, group="Advanced")
    maxiter: int = P(500, "Max optimizer iterations", cli="--maxiter", advanced=True, group="Advanced")
    validate_minima_with_hessian: bool = P(
        True, "Validate minima with Hessian", "Every minimum found must have no imaginary frequency; one "
        "that stopped on a saddle point is pushed along its unstable mode and re-optimized, and dropped "
        "(listed as rejected) if that fails too.", cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        0.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.1, "Rescue push (bohr)", "First push along the unstable mode; escalates to 0.3 and 0.5 bohr if the rescue fails.", cli="--hessian-minima-rescue-displacement", group="Hessian validation",
        advanced=True, requires="validate_minima_with_hessian", gt=0)


class OptimizeParams(Params):
    validate_minima_with_hessian: bool = P(
        True, "Verify minima with Hessian", "After optimizing, require no imaginary frequency; a structure "
        "that stopped on a saddle point is pushed along its unstable mode and re-optimized, and flagged "
        "'not a minimum' if that fails too.", cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        0.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.1, "Rescue push (bohr)", "First push along the unstable mode; escalates to 0.3 and 0.5 bohr if the rescue fails.", cli="--hessian-minima-rescue-displacement", group="Hessian validation",
        advanced=True, requires="validate_minima_with_hessian", gt=0)


class NetworkSplitsParams(Params):
    max_pairs: int = P(100, "Max pairs", cli="--max-pairs", ge=1)
    parallel: bool = P(False, "Parallel branches", cli="--parallel", kind="switch")
    validate_minima_with_hessian: bool = P(
        True, "Validate minima with Hessian", "Every intermediate minimum a recursive split proposes must "
        "have no imaginary frequency; a failing one is pushed along its unstable mode and re-optimized.",
        cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        0.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.1, "Rescue push (bohr)", "First push along the unstable mode before re-optimizing; escalates to "
        "0.3 and 0.5 bohr if the rescue fails.", cli="--hessian-minima-rescue-displacement", group="Hessian validation", advanced=True,
        requires="validate_minima_with_hessian", gt=0)

    same_pair_split_limit: int = P(5, "Same-pair split limit", cli="--same-pair-split-limit",
                                   advanced=True, group="Recursive splitting", ge=0)


# --------------------------------------------------------------- context

@dataclass
class JobContext:
    ws: Workspace
    job_dir: Path
    output_dir: Path
    structures: list[dict]  # resolved structure records, in target order
    profile: Optional[str]
    source: Optional[dict] = None  # follow-up operations: the job whose output they work on
    edge_ids: list = field(default_factory=list)   # the edge(s) a pair job runs on
    jobs: dict = field(default_factory=dict)       # the session's jobs (to find an edge's TS)

    def snapshot_structure(self, rec: dict, name: str) -> Path:
        """Copy a library geometry into the job folder, so the job stays
        reproducible even if the library entry is later edited/deleted."""
        dst = self.job_dir / "inputs" / f"{name}.xyz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.ws.structure_path(rec["id"]), dst)
        return dst

    def common_flags(self) -> list[str]:
        # A follow-up whose structure has since left the graph (or an
        # imported folder, which never had one) uses its source job's.
        first = self.structures[0] if self.structures else {
            "charge": (self.source or {}).get("charge") or 0,
            "multiplicity": (self.source or {}).get("multiplicity") or 1}
        argv = ["--charge", str(first["charge"]), "--multiplicity", str(first["multiplicity"])]
        if self.profile:
            prof = self.job_dir / "inputs" / "profile.toml"
            prof.parent.mkdir(parents=True, exist_ok=True)
            prof.write_text(self.ws.read_profile(self.profile))
            argv += ["--inputs", str(prof)]
        return argv

    def level(self) -> dict:
        """Level of theory this job runs at (its profile's fingerprint)."""
        return self.ws.level_of(self.profile)

    def at_job_level(self, rec: dict) -> bool:
        """Is this structure a minimum at the level of theory this job uses?"""
        lvl = rec.get("level") or {}
        return bool(rec.get("optimized")) and rec.get("status") == "ready" and lvl.get("key") == self.level()["key"]

    def endpoint_flags(self, mode: str) -> list[str]:
        a, b = self.structures
        for key in ("charge", "multiplicity"):
            if a[key] != b[key]:
                raise WorkspaceError(f"endpoints disagree on {key}: {a['name']}={a[key]}, {b['name']}={b[key]}")
        smiles = [rec.get("origin", {}).get("kind") == "smiles" and rec["origin"].get("input") for rec in (a, b)]
        # Once a structure has a minimized geometry, that geometry *is* the
        # structure: re-embedding its SMILES would throw the minimum away (and
        # the input SMILES may not even describe it any more, if it reacted
        # while being minimized).
        minimized = any(rec.get("level") for rec in (a, b))
        use_smiles = mode == "smiles" or (mode == "auto" and all(smiles) and not minimized)
        if use_smiles:
            if not all(smiles):
                raise WorkspaceError("endpoint source 'smiles' needs both structures to have been entered as SMILES")
            return ["--start", smiles[0], "--end", smiles[1]]
        return ["--start", str(self.snapshot_structure(a, "start")),
                "--end", str(self.snapshot_structure(b, "end"))]


# ----------------------------------------------------------- operations

@dataclass
class Operation:
    key: str
    title: str
    summary: str
    target: Target
    category: str
    params_model: Optional[type[Params]] = None
    build: Optional[Callable[[JobContext, Params], list[str]]] = None
    available: bool = True
    unavailable_reason: str = ""
    min_structures: int = 1
    # What a finished job contributes to the graph (used by the UI's import hints).
    produces: list[str] = field(default_factory=list)
    # "ts": offered only when every selected structure is a transition state.
    structure_role: Optional[str] = None
    # target "job": the operations whose finished output this follows up on.
    source_ops: tuple = ()
    # target "pair": runs from the edge's IRC-verified TS (so needs one).
    needs_route_ts: bool = False
    # The mepd command this runs (e.g. ("discovery", "vri")) and flags its
    # builder adds beyond the params' own: checked against the installed CLI,
    # so an operation whose command or flags this mepd lacks is shown as
    # unavailable instead of failing when run.
    cli_path: tuple = ()
    cli_extra_flags: tuple = ()

    def cli_problem(self) -> Optional[str]:
        if not self.cli_path:
            return None
        opts = _cli_options(self.cli_path)
        if opts is None:
            return f"this mepd has no `mepd {' '.join(self.cli_path)}` command yet"
        wanted = set(self.cli_extra_flags)
        for finfo in (self.params_model.model_fields.values() if self.params_model else []):
            extra = finfo.json_schema_extra or {}
            if extra.get("cli") and extra.get("cli_kind") != "custom":
                wanted.add(extra["cli"])
                if extra.get("cli_kind") == "toggle":
                    wanted.add("--no-" + extra["cli"].removeprefix("--"))
        missing = sorted(wanted - opts)
        if missing:
            return f"this mepd's `mepd {' '.join(self.cli_path)}` lacks {', '.join(missing)} (update mepd)"
        return None

    @property
    def usable(self) -> bool:
        return self.available and self.cli_problem() is None

    def describe(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "target": self.target,
            "category": self.category,
            "available": self.usable,
            "unavailable_reason": self.unavailable_reason or self.cli_problem() or "",
            "min_structures": self.min_structures,
            "produces": self.produces,
            "structure_role": self.structure_role,
            "source_ops": list(self.source_ops),
            "needs_route_ts": self.needs_route_ts,
            "schema": self.params_model.model_json_schema() if self.params_model else None,
        }

    def parse_params(self, raw: Optional[dict]) -> Params:
        if self.params_model is None:
            return Params()
        return self.params_model.model_validate(raw or {})


def _build_ts(ctx: JobContext, p: TsParams) -> list[str]:
    if p.irc and not p.use_tsopt:
        raise WorkspaceError("'Run IRC' needs 'Optimize TS'")
    argv = ["run", *ctx.endpoint_flags(p.endpoints), *ctx.common_flags()]
    if p.path_mode == "recursive" or (p.network_completion and p.path_mode == "single"):
        argv.append("--recursive")
    elif p.path_mode == "parallel":
        argv.append("--parallel")
    if p.minimize_ends == "auto":
        # mepd's own auto only fires for SMILES --start/--end; endpoints handed
        # over as xyz (anything not both-SMILES) would otherwise go into the
        # path search unrelaxed, which costs far more NEB steps than it saves.
        # A structure only counts as a minimum at the level it was optimized
        # at: one from a different profile (or a force-field embedding) is
        # re-minimized here, so both ends and the path share one PES.
        needs = not all(ctx.at_job_level(rec) for rec in ctx.structures)
        argv.append("--minimize-ends" if needs else "--no-minimize-ends")
    else:
        argv.append("--minimize-ends" if p.minimize_ends == "yes" else "--no-minimize-ends")
    argv += generic_flags(p)
    return argv + ["--output", str(ctx.output_dir)]


def _build_channels(ctx: JobContext, p: ChannelsParams) -> list[str]:
    argv = ["channels", *ctx.endpoint_flags(p.endpoints), *ctx.common_flags(), *generic_flags(p)]
    return argv + ["--output", str(ctx.output_dir)]


def _build_tsopt(ctx: JobContext, p: TsOptParams) -> list[str]:
    guess = ctx.snapshot_structure(ctx.structures[0], "guess")
    return ["ts", "--guess", str(guess), *ctx.common_flags(), *generic_flags(p), "--output", str(ctx.output_dir)]


def _build_discovery(command: str):
    def build(ctx: JobContext, p: Params) -> list[str]:
        seed = ctx.snapshot_structure(ctx.structures[0], "seed")
        return ["discovery", command, str(seed), *ctx.common_flags(), *generic_flags(p),
                "--output", str(ctx.output_dir)]
    return build


class VriParams(Params):
    branches: Literal["both", "forward", "reverse"] = P(
        "both", "IRC branches", "Which side(s) of TS1 to scan for a valley-ridge transition.", cli="--branches")
    stride: int = P(2, "Hessian every Nth IRC point", "1 resolves the scan best; 2 halves the cost (the "
                    "VRT is still bracketed and then bisected).", cli="--stride", ge=1)
    workers: int = P(4, "Parallel Hessians", "Hessians/optimizations at once; each engine call stays "
                     "single-threaded.", cli="--workers", ge=1)
    find_ts2: bool = P(True, "Find TS2", "Search for the second saddle (TS2) between P1 and P2.", kind="custom")
    symmetric_ts: bool = P(True, "Try symmetrized TS1", "Also start from TS1 candidates made by symmetrizing "
                           "TS1 under near-symmetries of its bond graph (finds ridge saddles).",
                           cli="--symmetric-ts", kind="toggle")
    exact_vri: bool = P(True, "Converge the exact VRI", "From the VRT to the exact valley-ridge inflection "
                        "point, on every branch with a second product (a few seconds per branch).",
                        cli="--exact-vri", kind="toggle")
    trajectories: int = P(0, "Trajectories from TS1", "Quasiclassical trajectories into each confirmed "
                          "bifurcation, for the P1:P2 ratio (100 resolves shares above a few percent; 0 = none).",
                          cli="--trajectories", ge=0, group="Advanced", advanced=True)
    hessian: Literal["auto", "gradients", "engine"] = P(
        "auto", "Hessian source", "gradients: central differences of engine gradients; engine: the "
        "engine's own Hessian.", cli="--hessian", group="Advanced", advanced=True)
    irc_step: float = P(0.05, "IRC step (Å·amu½)", cli="--irc-step", gt=0, group="Advanced", advanced=True)
    irc_fmax: float = P(0.01, "IRC stopping force (eV/Å)", cli="--irc-fmax", gt=0, group="Advanced", advanced=True)
    n_bisect: int = P(4, "Bisection steps", cli="--n-bisect", ge=0, group="Advanced", advanced=True)
    vrt_threshold: float = P(50.0, "Imaginary threshold (cm⁻¹)", cli="--vrt-threshold", gt=0,
                             group="Advanced", advanced=True)
    persist: int = P(2, "Consecutive imaginary points", cli="--persist", ge=1, group="Advanced", advanced=True)
    push_amplitude: float = P(0.3, "Ridge push (bohr)", cli="--push-amplitude", gt=0,
                              group="Advanced", advanced=True)
    max_vrt_depth: float = P(30.0, "Max VRT depth below TS1 (kcal/mol)", cli="--max-vrt-depth", gt=0,
                             group="Advanced", advanced=True)
    ts2_method: Literal["auto", "geodesic", "neb"] = P("auto", "TS2 search", cli="--ts2-method",
                                                       group="Advanced", advanced=True, requires="find_ts2")
    save_hessians: bool = P(True, "Save Hessians", cli="--save-hessians", kind="toggle",
                            group="Advanced", advanced=True)


class VriCheckParams(Params):
    trajectories: int = P(150, "Trajectories from TS1", "Quasiclassical trajectories per branch, to count "
                          "P1 vs P2 (0 = skip).", cli="--trajectories", ge=0)
    refine: bool = P(True, "Converge the exact VRI", cli="--refine", kind="toggle")
    basin: bool = P(True, "Basin test", "Steepest descent from sideways pushes off the IRC: a real "
                    "bifurcation drains into both P1 and P2.", cli="--basin", kind="toggle")
    workers: int = P(4, "Parallel engine calls", cli="--workers", ge=1)
    traj_fs: float = P(400.0, "Trajectory length (fs)", cli="--traj-fs", gt=0, group="Advanced", advanced=True)


class VriSurfaceParams(Params):
    grid: int = P(13, "Grid nodes per axis", "Each node is one restrained optimization.", cli="--grid", ge=3)
    workers: int = P(4, "Parallel gradient calls", cli="--workers", ge=1)


# Result groups whose entries are IRCs (or carry one): their TS connects the
# two ends of an edge added from them ("Add ends + edge to Graph").
IRC_GROUP_KINDS = ("irc", "channel", "alternate", "offtarget")


def _origin_ts(ws, edge: dict) -> Optional[tuple]:
    """(barrier, xyz) of the TS of the IRC an edge was added from, if any."""
    origin = edge.get("origin") or {}
    if origin.get("kind") != "job" or not origin.get("entry"):
        return None
    fp = ws.jobs_dir / origin["job"] / "result.json"
    try:
        result = json.loads(fp.read_text())
    except (OSError, ValueError):
        return None
    for group in result.get("groups", []):
        for entry in group.get("entries", []):
            if entry.get("id") != origin["entry"]:
                continue
            is_irc = group.get("kind") in IRC_GROUP_KINDS or str(entry["id"]).endswith("_irc")
            k = entry.get("ts_index")
            if not is_irc or k is None or not (0 <= k < len(entry.get("frames") or [])):
                return None
            barrier = origin.get("barrier_kcal", entry.get("barrier_kcal"))
            return (barrier if barrier is not None else float("inf"), entry["frames"][k]["xyz"])
    return None


def edge_route_ts(ws, jobs: dict, structure_ids: list, edge_ids: list) -> Optional[tuple]:
    """(barrier, label, xyz) of the lowest IRC-verified TS known between these
    two structures: from a finished TS / channels job on them, or from the
    IRC an edge was added from."""
    pair = set(structure_ids)
    best = None
    for job in jobs.values():
        if job["status"] != "done" or job["op"] not in ("ts", "channels"):
            continue
        targets = job.get("targets") or {}
        if not (set(targets.get("edges") or []) & set(edge_ids) or set(targets.get("structures") or []) == pair):
            continue
        info = (job.get("summary") or {}).get("route_ts")
        fp = ws.jobs_dir / job["id"] / "route_ts.xyz"
        if not info or not fp.exists():
            continue
        barrier = info.get("barrier_kcal")
        key = barrier if barrier is not None else float("inf")
        if best is None or key < best[0]:
            best = (key, f"{info.get('label')} of {job['title']}", fp.read_text())
    for eid in edge_ids:
        try:
            found = _origin_ts(ws, ws.edge(eid))
        except WorkspaceError:
            found = None
        if found is not None and (best is None or found[0] < best[0]):
            best = (found[0], "the TS of the IRC this edge was added from", found[1])
    return best


def _build_vri(ctx: JobContext, p: VriParams) -> list[str]:
    a, b = ctx.structures
    found = edge_route_ts(ctx.ws, ctx.jobs, [a["id"], b["id"]], ctx.edge_ids)
    if found is None:
        raise WorkspaceError(f"no transition state connects {a['name']} and {b['name']} yet: run 'Transition "
                             "state' or 'Reaction channels' on this edge first (the VRI search starts from the "
                             "IRC-verified TS that sets the edge's barrier)")
    _, _label, xyz = found
    ts = ctx.job_dir / "inputs" / "ts1.xyz"
    ts.parent.mkdir(parents=True, exist_ok=True)
    ts.write_text(xyz)
    argv = ["discovery", "vri", str(ts), *ctx.common_flags(), *generic_flags(p)]
    if not p.find_ts2:
        argv.append("--skip-ts2")
    return argv + ["--output", str(ctx.output_dir)]


def _build_vri_followup(command: str):
    def build(ctx: JobContext, p: Params) -> list[str]:
        if ctx.source is None:
            raise WorkspaceError(f"{command} follows up on a finished VRI search")
        return ["discovery", command, ctx.source["output_dir"], *ctx.common_flags(), *generic_flags(p)]
    return build


def _build_optimize(ctx: JobContext, p: OptimizeParams) -> list[str]:
    ts = [r["name"] for r in ctx.structures if is_ts(r)]
    if ts:
        raise WorkspaceError(f"{', '.join(ts)}: transition states are not minimized (use 'Optimize TS from guess')")
    if len({(r["charge"], r["multiplicity"]) for r in ctx.structures}) > 1:
        raise WorkspaceError("optimize structures with different charge/multiplicity in separate jobs")
    files = [str(ctx.snapshot_structure(r, f"structure_{i}")) for i, r in enumerate(ctx.structures)]
    return ["optimize", *files, *ctx.common_flags(), *generic_flags(p), "--output", str(ctx.output_dir)]


def _build_network_splits(ctx: JobContext, p: NetworkSplitsParams) -> list[str]:
    charges = {(r["charge"], r["multiplicity"]) for r in ctx.structures}
    if len(charges) > 1:
        raise WorkspaceError("all structures must share charge and multiplicity")
    natoms = {r["natoms"] for r in ctx.structures}
    if len(natoms) > 1:
        raise WorkspaceError("all structures must have the same atoms")
    minima = [str(ctx.snapshot_structure(r, f"minimum_{i}")) for i, r in enumerate(ctx.structures)]
    return ["network-splits", *minima, *ctx.common_flags(), *generic_flags(p), "--output", str(ctx.output_dir)]


PAIR = "Connect two structures"
EXPLORE = "Explore around a structure"
SET = "Across a set of structures"
FROM_TS = "Past the transition state"

OPERATIONS: dict[str, Operation] = {op.key: op for op in [
    Operation(
        "ts", "Transition state", "Find the minimum-energy path and its transition state(s) between "
        "two structures, optionally refined by TS optimization and IRC. (`mepd run`)",
        "pair", PAIR, TsParams, _build_ts, min_structures=2,
        produces=["transition states", "IRC endpoints", "intermediates"]),
    Operation(
        "channels", "Reaction channels", "Sample reactant/product conformers and atom mappings, search "
        "every distinct mechanism, and classify the TSs into direct, multi-step and off-target "
        "channels. (`mepd channels`)",
        "pair", PAIR, ChannelsParams, _build_channels, min_structures=2,
        produces=["transition states per channel", "conformers", "off-target products"]),
    Operation(
        "optimize", "Optimize geometry", "Minimize the selected structures at the chosen profile's level of "
        "theory, replacing their geometry and energy in place. (`mepd optimize`)",
        "set", EXPLORE, OptimizeParams, _build_optimize, min_structures=1,
        produces=["optimized geometries (in place)"]),
    Operation(
        "tsopt", "Optimize TS from guess", "Treat the structure as a TS guess: saddle optimization, "
        "optionally followed by IRC. (`mepd ts`)",
        "structure", EXPLORE, TsOptParams, _build_tsopt, produces=["transition state", "IRC endpoints"]),
    Operation(
        "hessian-sample", "Hessian sampling", "Displace along every normal mode and re-optimize to find "
        "nearby minima. (`mepd discovery hessian-sample`)",
        "structure", EXPLORE, HessianSampleParams, _build_discovery("hessian-sample"),
        produces=["nearby minima"]),
    Operation(
        "hessian-global", "Hessian basin hopping", "Repeated Hessian sampling with Metropolis acceptance: "
        "a global search over minima reachable from the seed. (`mepd discovery hessian-global`)",
        "structure", EXPLORE, HessianGlobalParams, _build_discovery("hessian-global"),
        produces=["accepted minima"]),
    Operation(
        "nanoreactor", "Nanoreactor", "Reactive MD / CREST msreact products around a structure.",
        "structure", EXPLORE, available=False,
        unavailable_reason="The engine can generate nanoreactor candidates "
        "(QCComputeEngine.compute_nanoreactor_candidates), but mepd has no CLI command for it yet."),
    Operation(
        "graph-enumeration", "Graph enumeration", "Enumerate products by bond-breaking/forming rules.",
        "structure", EXPLORE, available=False,
        unavailable_reason="Not implemented in mepd yet."),
    Operation(
        "network-splits", "All-pairs network", "Run recursive path searches between every pair of the "
        "selected minima and assemble a reaction network. (`mepd network-splits`)",
        "set", SET, NetworkSplitsParams, _build_network_splits, min_structures=2,
        produces=["network edges", "intermediates"]),
    Operation(
        "vri", "Valley-ridge inflection", "Scan the IRC from this edge's transition state for a valley-ridge "
        "transition, where the path can split after the TS (a post-TS bifurcation into two products, P1 and P2). "
        "(`mepd discovery vri`)",
        "pair", FROM_TS, VriParams, _build_vri, min_structures=2, needs_route_ts=True,
        produces=["products P1 and P2", "TS2 between them"],
        cli_path=("discovery", "vri"), cli_extra_flags=("--skip-ts2", "--charge", "--multiplicity", "--inputs", "--output")),
    Operation(
        "vri-check", "Check the bifurcation", "Converge the exact VRI, test that sideways pushes off the IRC "
        "drain into both P1 and P2, and count trajectories into each. (`mepd discovery vri-check`)",
        "job", FROM_TS, VriCheckParams, _build_vri_followup("vri-check"), source_ops=("vri",),
        cli_path=("discovery", "vri-check")),
    Operation(
        "vri-surface", "Map the 2D surface", "Reconstruct the energy surface around each bifurcation "
        "(TS1, VRI, TS2, P1, P2 on one map). (`mepd discovery vri-surface`)",
        "job", FROM_TS, VriSurfaceParams, _build_vri_followup("vri-surface"), source_ops=("vri",),
        cli_path=("discovery", "vri-surface")),
]}


def get_operation(key: str) -> Operation:
    try:
        op = OPERATIONS[key]
    except KeyError:
        raise WorkspaceError(f"unknown operation {key!r}") from None
    if not op.usable:
        raise WorkspaceError(f"{op.title} is not available: {op.unavailable_reason or op.cli_problem()}")
    return op


@functools.lru_cache(maxsize=None)
def _cli_options(path: tuple) -> Optional[frozenset]:
    """Every option of the installed `mepd <path...>` command, or None if
    there is no such command."""
    import click
    import typer

    from mepd.cli import app

    cmd = typer.main.get_command(app)
    for word in path:
        sub = cmd.get_command(click.Context(cmd), word) if hasattr(cmd, "get_command") else None
        if sub is None:
            return None
        cmd = sub
    opts = set()
    for prm in cmd.params:
        opts.update(getattr(prm, "opts", []))
        opts.update(getattr(prm, "secondary_opts", []))
    return frozenset(opts)
