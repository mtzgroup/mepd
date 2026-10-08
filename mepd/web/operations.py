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

from mepd.atom_mapping_metrics import HELP as _MAPPING_HELP, LABELS as _MAPPING_LABELS, OFFERED as _OFFERED_METRICS
from mepd.web.workspace import Workspace, WorkspaceError, is_ts

# "job": a follow-up on another job's output; "design": the Design tab's molecule
Target = Literal["structure", "pair", "set", "job", "design", "complex"]
# The three offered: endpoint RMSD, GI path, filtered GI path (mepd still
# accepts the other two from older profiles, but no form offers them).
# Subscripting with a tuple lists its items, so this stays in step with
# mepd's list (`Literal[*OFFERED]` would need 3.11).
AtomMappingMetric = Literal[_OFFERED_METRICS]


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
    if "labels" in kw:   # {value: what the form shows for it}
        extra["labels"] = kw.pop("labels")
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
_MAPPING_METRIC_HELP = _MAPPING_HELP


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
    pair_from: Literal["both", "start", "end"] = P(
        "both", "Build endpoint", "Use both endpoints as given, or replace one with a geometry built in the "
        "other's frame (each atom where it is, the new bonds pulled to length, minimized): only what reacts "
        "moves.", cli="--pair-from", group="Endpoints", advanced=True,
        labels={"both": "Both as given", "start": "Product built from reactant",
                "end": "Reactant built from product"})
    rebuild_splits_from: Literal["none", "start", "end"] = P(
        "none", "Rebuild at splits", "Experimental: at every split, rebuild each new leg's product from its "
        "start (or its start from its product), kept only if bonds and stereochemistry survive.",
        cli="--rebuild-splits-from", group="Endpoints", advanced=True,
        labels={"none": "Off", "start": "Product from start", "end": "Start from product"})
    atom_mapping_candidates: int = P(200, "Mapping candidates", cli="--atom-mapping-candidates",
                                     group="Atom mapping", advanced=True, ge=1, requires="atom_mapping")
    atom_mapping_metric: AtomMappingMetric = P(
        "snap-gi-xtb", "Mapping metric", _MAPPING_METRIC_HELP,
        cli="--atom-mapping-metric", group="Atom mapping", advanced=True,
        requires="atom_mapping", labels=_MAPPING_LABELS)
    explore_mechanisms: int = P(
        0, "Explore mechanisms", ("Also consider up to this many mechanisms beyond SLAPMapper's minimal-edit ones: relays and exchanges "
        "through other molecules (a catalyst, solvent), found by swapping same-element atoms or groups at most "
        "+2 bond edits per swap, with no rule about which atoms may move. Each is scored like SLAPMapper's "
        "(snap + GI + xtb); ones in which a molecule takes part and is regenerated are labelled "
        "[catalytic: ...]. 0 = off; 20 is a reasonable start."),
        cli="--explore-mechanisms", group="Atom mapping", advanced=True, ge=0, requires="atom_mapping")
    relax_mechanisms: int = P(
        4, "Rank mappings by relaxed path energy (top N)", ("Candidate atom mappings are compared by the energy "
        "along their interpolated path. With N > 0, the N best by raw xtb peak (plus SLAPMapper's own minimal "
        "mapping, always) are first relaxed -- everything except the bonds that break or form -- and compared by "
        "that relaxed barrier. Only this way can a mapping with more bond changes (e.g. from Explore mechanisms) "
        "win, and only by more than the margin below. 0 = raw interpolation peaks only."),
        cli="--relax-mechanisms", group="Atom mapping", advanced=True, ge=0, requires="atom_mapping")
    relax_margin: float = P(
        5.0, "Margin for extra bond changes (kcal/mol)", ("A mapping with more bond changes than SLAPMapper's minimal "
        "one must beat its relaxed barrier by more than this, or SLAPMapper's mapping is kept. The log lists every "
        "compared mechanism and says which rule decided."),
        cli="--relax-margin", group="Atom mapping", advanced=True, ge=0, requires="atom_mapping")
    validate_minima_with_hessian: bool = P(
        True, "Validate minima with Hessian", "Every intermediate minimum a recursive split proposes must "
        "have no imaginary frequency beyond the cutoff; a failing one is pushed along its unstable mode and re-optimized.",
        cli="--validate-minima-with-hessian", kind="toggle", requires="path_mode=recursive|parallel")
    hessian_minimum_frequency_cutoff: float = P(
        -20.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.3, "Rescue push (bohr)", "Push along the unstable mode, both ways, before re-optimizing.", cli="--hessian-minima-rescue-displacement", group="Hessian validation", advanced=True,
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
    direct_only: bool = P(False, "Only legs touching start or end", "When a path search splits, run only the "
                          "pieces with at least one end at the start or end species (any conformer or stereoisomer): "
                          "A→B splitting into A→C, C→D, D→A′, A′→B runs A→C, D→A′ and A′→B, not C→D.",
                          cli="--direct-only", kind="switch")
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
    rmsd_cutoff: float = P(0.1, "Dedup RMSD (bohr)", "Conformers closer than this (snap-RMSD, bohr) count as one.",
                           cli="--rmsd-cutoff", group="Conformers", advanced=True, gt=0)
    random_seed: int = P(0, "Random seed", cli="--random-seed", group="Conformers", advanced=True)
    reuse_conformers: bool = P(True, "Reuse conformer pools",
                               "Take an endpoint's conformers from an earlier Reaction channels run on the same "
                               "molecule with the same sampler settings and level of theory, instead of "
                               "sampling and minimizing them again.", kind="custom", group="Conformers")
    pair_from: Literal["both", "start", "end", "start+end"] = P(
        "both", "Pairs from", "Every reactant conformer with every product conformer, or sample one side only "
        "and build each conformer's partner in its frame (each atom where it is, the new bonds pulled to "
        "length, minimized): only what reacts moves, and the partner's geometries are sampled through the "
        "conformer's.", cli="--pair-from", group="Pairs", advanced=True,
        labels={"both": "Both sides sampled", "start": "Reactant conformers, products built",
                "end": "Product conformers, reactants built",
                "start+end": "Each side's conformers, partners built"})
    rebuild_splits_from: Literal["none", "start", "end"] = P(
        "none", "Rebuild at splits", "Experimental: at every split, rebuild each new leg's product from its "
        "start (or its start from its product), kept only if bonds and stereochemistry survive.",
        cli="--rebuild-splits-from", group="Pairs", advanced=True,
        labels={"none": "Off", "start": "Product from start", "end": "Start from product"})
    max_pairs: int = P(0, "Max pairs", "0 = no cap.", cli="--max-pairs", group="Pairs", advanced=True, ge=0)
    atom_mapping: bool = P(True, "Atom mapping per pair", cli="--atom-mapping", kind="toggle",
                           group="Pairs", advanced=True)
    atom_mapping_metric: AtomMappingMetric = P(
        "snap-gi-xtb", "Mapping metric", _MAPPING_METRIC_HELP,
        cli="--atom-mapping-metric", group="Pairs", advanced=True,
        requires="atom_mapping", labels=_MAPPING_LABELS)
    explore_mechanisms: int = P(
        0, "Explore mechanisms", ("Also consider up to this many mechanisms beyond SLAPMapper's minimal-edit ones: relays and exchanges "
        "through other molecules (a catalyst, solvent), found by swapping same-element atoms or groups at most "
        "+2 bond edits per swap, with no rule about which atoms may move. Each is scored like SLAPMapper's "
        "(snap + GI + xtb); ones in which a molecule takes part and is regenerated are labelled "
        "[catalytic: ...]. 0 = off; 20 is a reasonable start."),
        cli="--explore-mechanisms", group="Pairs", advanced=True, ge=0, requires="atom_mapping")
    relax_mechanisms: int = P(
        4, "Rank mappings by relaxed path energy (top N)", ("Candidate atom mappings are compared by the energy "
        "along their interpolated path. With N > 0, the N best by raw xtb peak (plus SLAPMapper's own minimal "
        "mapping, always) are first relaxed -- everything except the bonds that break or form -- and compared by "
        "that relaxed barrier. Only this way can a mapping with more bond changes (e.g. from Explore mechanisms) "
        "win, and only by more than the margin below. 0 = raw interpolation peaks only."),
        cli="--relax-mechanisms", group="Pairs", advanced=True, ge=0, requires="atom_mapping")
    relax_margin: float = P(
        5.0, "Margin for extra bond changes (kcal/mol)", ("A mapping with more bond changes than SLAPMapper's minimal "
        "one must beat its relaxed barrier by more than this, or SLAPMapper's mapping is kept. The log lists every "
        "compared mechanism and says which rule decided."),
        cli="--relax-margin", group="Pairs", advanced=True, ge=0, requires="atom_mapping")
    atom_mapping_rmsd_window: float = P(
        1.0, "RMSD window (σ)", "Filtered GI path: GI paths for the conformer pairs within this many standard "
        "deviations of each mechanism's lowest endpoint RMSD. On the KAIST set 1σ kept 6% of the pairs and 56% of "
        "what GI paths over every pair would pick; 2σ 21% / 82%; 3σ 51% / 96%.",
        cli="--atom-mapping-rmsd-window", group="Pairs", advanced=True, ge=0,
        requires="atom_mapping_metric=rmsd-geodesic")
    validate_minima_with_hessian: bool = P(
        True, "Validate minima with Hessian", "Every intermediate minimum a recursive split proposes must "
        "have no imaginary frequency beyond the cutoff; a failing one is pushed along its unstable mode and re-optimized.",
        cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        -20.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.3, "Rescue push (bohr)", "Push along the unstable mode, both ways, before re-optimizing.", cli="--hessian-minima-rescue-displacement", group="Hessian validation", advanced=True,
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
        True, "Validate minima with Hessian", "Every minimum found must have no imaginary frequency beyond the cutoff; one "
        "that stopped on a saddle point is pushed along its unstable mode and re-optimized, and dropped "
        "(listed as rejected) if that fails too.", cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        -20.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.3, "Rescue push (bohr)", "Push along the unstable mode, both ways, before re-optimizing.", cli="--hessian-minima-rescue-displacement", group="Hessian validation",
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
        True, "Validate minima with Hessian", "Every minimum found must have no imaginary frequency beyond the cutoff; one "
        "that stopped on a saddle point is pushed along its unstable mode and re-optimized, and dropped "
        "(listed as rejected) if that fails too.", cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        -20.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.3, "Rescue push (bohr)", "Push along the unstable mode, both ways, before re-optimizing.", cli="--hessian-minima-rescue-displacement", group="Hessian validation",
        advanced=True, requires="validate_minima_with_hessian", gt=0)


class OptimizeParams(Params):
    validate_minima_with_hessian: bool = P(
        True, "Verify minima with Hessian", "After optimizing, require no imaginary frequency beyond the cutoff; a structure "
        "that stopped on a saddle point is pushed along its unstable mode and re-optimized, and flagged "
        "'not a minimum' if that fails too.", cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        -20.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.3, "Rescue push (bohr)", "Push along the unstable mode, both ways, before re-optimizing.", cli="--hessian-minima-rescue-displacement", group="Hessian validation",
        advanced=True, requires="validate_minima_with_hessian", gt=0)


class NetworkSplitsParams(Params):
    max_pairs: int = P(100, "Max pairs", cli="--max-pairs", ge=1)
    parallel: bool = P(False, "Parallel branches", cli="--parallel", kind="switch")
    validate_minima_with_hessian: bool = P(
        True, "Validate minima with Hessian", "Every intermediate minimum a recursive split proposes must "
        "have no imaginary frequency beyond the cutoff; a failing one is pushed along its unstable mode and re-optimized.",
        cli="--validate-minima-with-hessian", kind="toggle")
    hessian_minimum_frequency_cutoff: float = P(
        -20.0, "Lowest allowed frequency (cm⁻¹)", cli="--hessian-minimum-frequency-cutoff",
        group="Hessian validation", advanced=True, requires="validate_minima_with_hessian")
    hessian_minima_rescue_displacement: float = P(
        0.3, "Rescue push (bohr)", "Push along the unstable mode, both ways, before re-optimizing.", cli="--hessian-minima-rescue-displacement", group="Hessian validation", advanced=True,
        requires="validate_minima_with_hessian", gt=0)

    same_pair_split_limit: int = P(5, "Same-pair split limit", cli="--same-pair-split-limit",
                                   advanced=True, group="Recursive splitting", ge=0)


class ConformersParams(Params):
    backend: Literal["rdkit", "crest"] = P("rdkit", "Sampler", "RDKit: ETKDG embeddings relaxed with MMFF "
                                           "(seconds). CREST: metadynamics at GFN2/GFN-FF (minutes, single-threaded).",
                                           cli="--backend")
    n_conformers: int = P(20, "Keep at most", "Distinct conformers kept (0 = no cap).", cli="--n-conformers", ge=0)
    minimize: bool = P(True, "Minimize at the profile's level", "So their energies are comparable and the "
                       "lowest one can represent the molecule. Off: the sampler's geometries, without energies.",
                       cli="--minimize", kind="toggle")
    validate_minima_with_hessian: bool = P(False, "Hessian-check each", cli="--validate-minima-with-hessian",
                                           kind="toggle", requires="minimize")
    rmsd_cutoff: float = P(0.1, "Distinct beyond RMSD (bohr)", cli="--rmsd-cutoff", gt=0, advanced=True,
                           group="Sampling")
    crest_method: Literal["--gfn2", "--gfnff", "--gfn2//gfnff"] = P(
        "--gfn2", "CREST level", cli="--crest-method", advanced=True, group="Sampling", requires="backend=crest")
    crest_ewin: float = P(6.0, "CREST window (kcal/mol)", cli="--crest-ewin", gt=0, advanced=True,
                          group="Sampling", requires="backend=crest")


def _build_conformers(ctx: JobContext, p: ConformersParams) -> list[str]:
    rec = ctx.structures[0]
    if is_ts(rec):
        raise WorkspaceError(f"{rec['name']} is a transition state: conformers are sampled for minima")
    if p.backend == "crest" and shutil.which("crest") is None:
        raise WorkspaceError("CREST isn't installed here (no `crest` on PATH); use the RDKit sampler")
    seed = ctx.snapshot_structure(rec, "seed")
    return ["conformers", str(seed), *ctx.common_flags(), *generic_flags(p), "--output", str(ctx.output_dir)]


class ExpandParams(Params):
    # Chosen by the method switch (Bond rules / CREST msreact), not in the form.
    generator: Literal["bond-rules", "crest-msreact", "nanoreactor"] = P("bond-rules", "Product generator", kind="custom",
                                                          group="Hidden")
    rounds: int = P(1, "Rounds", "Round 2 proposes products of round 1's new species, and so on.", cli="--rounds", ge=1)
    steer: Literal["auto", "flux", "window"] = P(
        "auto", "Grow the network by", "flux: find a verified TS for every reaction, simulate the kinetics from "
        "the seed, and expand only species that enough material flows into (the network stops growing on its "
        "own). window: expand every new species within the energy window. auto: flux when rounds > 1.",
        cli="--steer")
    temperature: float = P(298.15, "Temperature (K)", cli="--temperature", gt=0, requires="steer=auto|flux",
                           group="Kinetics")
    time_s: float = P(3600.0, "Simulated time (s)", cli="--time", gt=0, requires="steer=auto|flux", group="Kinetics")
    flux_threshold: float = P(0.01, "Flux threshold", "Expand a species once the material that flowed into it "
                              "(as a fraction of the seed) reaches this.", cli="--flux-threshold", gt=0,
                              requires="steer=auto|flux", group="Kinetics")
    n_break: int = P(2, "Bonds broken (max)", cli="--n-break", ge=0, requires="generator=bond-rules")
    n_form: int = P(2, "Bonds formed (max)", cli="--n-form", ge=0, requires="generator=bond-rules")
    msreact_mode: Literal["all", "fragments", "isomers"] = P(
        "all", "Products", "fragments: dissociated products only (e.g. to read as precursors, backwards); "
        "isomers: non-dissociated only; all: both.", kind="custom", requires="generator=crest-msreact")
    msreact_nbonds: int = P(3, "Bias bonds up to (bonds apart)", "msreact's repulsive potential acts on atom pairs "
                            "up to this many bonds apart (CREST default 3).", kind="custom", ge=1,
                            requires="generator=crest-msreact", advanced=True, group="CREST msreact")
    msreact_nshifts: int = P(0, "Random-shift optimizations", "Extra optimizations from randomly shifted atoms, for "
                             "more products (slower).", kind="custom", ge=0, requires="generator=crest-msreact",
                             advanced=True, group="CREST msreact")
    reactor_temperature: float = P(3000.0, "MD temperature (K)", "Hot enough for bonds to break within "
                                   "picoseconds (2500-3500 K). Much hotter and molecules fall apart into atoms, "
                                   "which give no products.", kind="custom", gt=0,
                                   requires="generator=nanoreactor")
    reactor_time_ps: float = P(20.0, "MD time (ps)", kind="custom", gt=0, requires="generator=nanoreactor")
    reactor_embedded: bool = P(False, "MD with QM/MM forces", "Run the hot MD with the environment present "
                               "(QM/MM forces every step; the solvent is held near room temperature). Off: the QM "
                               "region alone in vacuum, its products embedded afterwards (much faster). On is about "
                               "30-60 min per 10 ps for a small QM region.", kind="custom",
                               requires="generator=nanoreactor")
    reactor_compress: float = P(0.5, "Squeeze to (of the wall radius)", "The piston closes the wall to this "
                                "fraction of its radius once per picosecond.", kind="custom", gt=0, le=1,
                                requires="generator=nanoreactor", advanced=True, group="Nanoreactor")
    max_products: int = P(50, "Proposals per species", "Bond rules: fewest bond changes first. CREST msreact: "
                          "lowest GFN2-xTB energy first.", cli="--max-products", ge=1)
    energy_window: float = P(60.0, "Expand species within (kcal/mol)", "Only species this close to the seed "
                             "are expanded in the next round.", cli="--energy-window", requires="steer=auto|window")
    explore_within: Optional[float] = P(
        None, "Add to Explore within (kcal/mol)", "Only species at most this far above the seed become nodes in "
        "Explore; the rest stay in the result, to add by hand. Empty: the 'Expand species within' window (every "
        "species when growing by flux).", ge=0, group="Explore")
    form_distance: float = P(4.0, "Form bonds within (Å)", "Raise it to also close rings between atoms that "
                             "start far apart.", cli="--form-distance", gt=0, advanced=True, group="Rules",
                             requires="generator=bond-rules")
    allow_radicals: bool = P(False, "Allow radicals and carbenes", cli="--allow-radicals", kind="toggle",
                             advanced=True, group="Rules", requires="generator=bond-rules")
    allow_zwitterions: bool = P(False, "Allow charge-separated products", "Needed for e.g. isocyanides and CO.",
                                cli="--allow-zwitterions", kind="switch", advanced=True, group="Rules",
                                requires="generator=bond-rules")
    validate_minima_with_hessian: bool = P(
        True, "Validate species with Hessian", "Every new species must have no imaginary frequency beyond the cutoff; one that "
        "stopped on a saddle point is pushed along its unstable mode and re-optimized, and dropped if that fails.",
        cli="--validate-minima-with-hessian", kind="toggle")
    connect: bool = P(False, "Connect by path search", "Run a recursive path search for every proposed reaction "
                      "that reached a new species, and build the network from the paths.", cli="--connect", kind="toggle")
    max_pairs: int = P(50, "Max path searches", cli="--max-pairs", ge=1, requires="connect")
    maxiter: int = P(500, "Max optimizer iterations", cli="--maxiter", advanced=True, group="Advanced")
    workers: int = P(4, "Parallel workers", "Processes building product guesses and their animations.",
                     cli="--workers", ge=1, advanced=True, group="Advanced")


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
        # A chosen conformer (the edge's, or the caller's), else the node's
        # lowest-energy one.
        src = self.ws.conformer_path(rec["id"], rec["conformer_id"]) if rec.get("conformer_id") \
            else self.ws.structure_path(rec["id"])
        shutil.copyfile(src, dst)
        return dst

    @property
    def qmmm(self) -> Optional[str]:
        """The QM/MM system this job runs on (its structures', or its source
        job's), or None. Mixing systems, or QM/MM with plain structures, is
        refused."""
        if not self.structures:
            return (self.source or {}).get("qmmm")
        ids = {r.get("qmmm") for r in self.structures}
        if len(ids) > 1:
            raise WorkspaceError("these structures are not all in the same QM/MM system" if None not in ids else
                                 "QM/MM structures cannot be mixed with structures outside that system")
        return ids.pop()

    def common_flags(self) -> list[str]:
        # A follow-up whose structure has since left the graph (or an
        # imported folder, which never had one) uses its source job's.
        first = self.structures[0] if self.structures else {
            "charge": (self.source or {}).get("charge") or 0,
            "multiplicity": (self.source or {}).get("multiplicity") or 1}
        if not self.structures and self.source is None and self.ws.design:
            first = self.ws.design
        argv = ["--charge", str(first["charge"]), "--multiplicity", str(first["multiplicity"])]
        qmmm = self.qmmm
        if self.profile or qmmm:
            prof = self.job_dir / "inputs" / "profile.toml"
            prof.parent.mkdir(parents=True, exist_ok=True)
            text = self.ws.read_profile(self.profile) if self.profile else ""
            if qmmm:
                # The profile is the QM level; the system's region embeds it.
                import tomllib

                if "qmmm" in tomllib.loads(text or ""):
                    raise WorkspaceError(f"profile {self.profile!r} has its own [qmmm] table; the QM/MM system "
                                         "sets the region: remove it from the profile")
                shutil.copyfile(self.ws.qmmm_region_path(qmmm), prof.parent / "qmmm_region.json")
                text = text.rstrip() + '\n\n[qmmm]\nfile = "qmmm_region.json"\n'
            prof.write_text(text)
            argv += ["--inputs", str(prof)]
        return argv

    def level(self) -> dict:
        """Level of theory this job runs at (its profile's fingerprint; for a
        QM/MM system, also its region's: energies of different regions, or
        of a region and the gas phase, are never compared)."""
        level = self.ws.level_of(self.profile)
        try:
            qmmm = self.qmmm
        except WorkspaceError:   # a mixed selection (e.g. putting a structure into a system)
            qmmm = None
        if qmmm:
            sysrec = self.ws.snapshot()["qmmm_systems"][qmmm]
            level = {**level, "key": f"{level['key']}+qmmm:{sysrec['sig']}",
                     "label": f"{level['label']} / QM/MM", "qmmm": qmmm}
        return level

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
    # target "job": write into this job's own output folder instead of the
    # source's (for follow-ups that may run many times side by side).
    own_output: bool = False
    # Several operations shown as one card with a method switch (e.g. the
    # ways of exploring a reaction network). Each method: {label, summary,
    # fixed: params it sets (hidden in the form), programs: executables it
    # needs on PATH}.
    family: Optional[dict] = None
    methods: tuple = ()
    # target "set": every selected structure must have the same atoms (isomers).
    same_atoms: bool = False

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
            "family": self.family,
            "methods": [_method_view(m) for m in self.methods],
            "same_atoms": self.same_atoms,
            "qmmm": self.key in QMMM_OPS,
        }

    def parse_params(self, raw: Optional[dict]) -> Params:
        if self.params_model is None:
            return Params()
        return self.params_model.model_validate(raw or {})


def _method_view(m: dict) -> dict:
    import shutil

    missing = [prog for prog in m.get("programs", ()) if shutil.which(prog) is None]
    reason = f"needs {' and '.join(missing)} on the server's PATH ({m['install']})" if missing else ""
    if not reason and callable(m.get("check")):   # e.g. a Python package or a one-time setup
        reason = m["check"]() or ""
    return {"label": m["label"], "summary": m.get("summary", ""), "fixed": m.get("fixed", {}), "rank": m.get("rank", 0),
            "qmmm_only": bool(m.get("qmmm_only")),
            "group": m.get("group", ""), "available": not reason, "reason": reason}


TS_FAMILY = {"key": "ts", "title": "Transition state",
             "summary": "Find the transition state(s) between two structures: from these endpoints only, or "
                        "sampling their conformers and atom mappings for every distinct channel."}

NETWORK = {"key": "network", "title": "Reaction network expansion",
           "summary": "Explore the reactions and minima around a structure (or, with the nanoreactor, among "
                      "several). Pick a method; every species found joins Explore."}


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


# Settings that decide what a channels run's conformer pools contain.
_POOL_KEYS = ("backend", "n_conformers", "rmsd_cutoff", "random_seed", "minimize_ends")
_POOL_KEYS_BY_BACKEND = {"rdkit": ("n_embed", "rdkit_ewin"), "crest": ("crest_method", "crest_ewin", "crest_timeout")}
_POOL_FILES = (".xyz", ".energies", ".gradients", "_grad_shapes.txt")


def _pool_settings(params: dict) -> tuple:
    p = ChannelsParams.model_validate({k: v for k, v in params.items() if k in ChannelsParams.model_fields})
    keys = _POOL_KEYS + _POOL_KEYS_BY_BACKEND[p.backend]
    return tuple((k, getattr(p, k)) for k in keys)


def _saved_pool(job: dict, side: str) -> Optional[Path]:
    """A finished channels job's final pool for `side` ("start"/"end"),
    before mirror images were merged (so it is complete on its own)."""
    conf = Path(job.get("output_dir") or "") / "conformers"
    if (conf / f"{side}_pool.xyz").exists():
        return conf / f"{side}_pool.xyz"
    # Older runs saved only the merged pool: usable when nothing was merged.
    try:
        stats = json.loads((Path(job["output_dir"]) / "stats.json").read_text())
    except (OSError, ValueError, KeyError):
        return None
    merged = ((stats.get("conformers") or {}).get(side) or {}).get("n_mirror_images_merged")
    return conf / f"{side}.xyz" if merged == 0 and (conf / f"{side}.xyz").exists() else None


def _reusable_pool(ctx: JobContext, p: ChannelsParams, rec: dict) -> Optional[tuple[Path, dict]]:
    """The newest finished channels job that sampled `rec`'s molecule (as
    either endpoint) with the same pool settings -- and, when its pools were
    minimized, at this job's level of theory -- and its saved pool file."""
    want = _pool_settings(p.model_dump())
    level = ctx.level().get("key")
    best = None
    for job in (ctx.jobs or {}).values():
        sids = job.get("targets", {}).get("structures") or []
        if job.get("op") != "channels" or job.get("status") != "done" or rec["id"] not in sids[:2]:
            continue
        if job.get("charge") != rec["charge"] or job.get("multiplicity") != rec["multiplicity"]:
            continue
        try:
            if _pool_settings(job.get("params") or {}) != want:
                continue
        except Exception:
            continue
        if p.minimize_ends and (job.get("level") or {}).get("key") != level:
            continue
        fp = _saved_pool(job, "start" if sids[0] == rec["id"] else "end")
        if fp is not None and (best is None or (job.get("finished") or 0) > (best[1].get("finished") or 0)):
            best = (fp, job)
    return best


def _build_channels(ctx: JobContext, p: ChannelsParams) -> list[str]:
    argv = ["channels", *ctx.endpoint_flags(p.endpoints), *ctx.common_flags(), *generic_flags(p)]
    if p.reuse_conformers:
        for side, rec in zip(("start", "end"), ctx.structures):
            found = _reusable_pool(ctx, p, rec)
            if found is None:
                continue
            src, _ = found
            dst = ctx.job_dir / "inputs" / f"{side}_pool.xyz"   # a copy: the job stays reproducible
            dst.parent.mkdir(parents=True, exist_ok=True)
            for suffix in _POOL_FILES:
                f = src.with_name(src.stem + suffix)
                if f.exists():
                    shutil.copyfile(f, dst.with_name(dst.stem + suffix))
            argv += [f"--{side}-pool", str(dst)]
    return argv + ["--output", str(ctx.output_dir)]


class ChannelsMoreParams(Params):
    pairs_per_mechanism: int = P(6, "Searches per mechanism", "The new cap: more of each mechanism's "
                                 "best-scoring conformer pairs (0 = all of them). Pairs already searched are "
                                 "kept and skipped.", kind="custom", ge=0)
    workers: int = P(4, "Workers", "Processes for atom mapping, path searches and TS/IRC.", kind="custom", ge=1)
    rmsd_window: Optional[float] = P(
        None, "RMSD window (σ)", "Filtered GI path only: widen the window to GI-score more conformer pairs "
        "(empty: as before).", kind="custom", ge=0, advanced=True)


def _replace_flags(argv: list[str], flags: dict) -> list[str]:
    """`argv` with each `--flag value` in `flags` removed, then appended with
    its new value (None: dropped)."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in flags:
            skip = True
            continue
        out.append(a)
    for flag, value in flags.items():
        if value is not None:
            out += [flag, str(value)]
    return out


def _build_channels_more(ctx: JobContext, p: ChannelsMoreParams) -> list[str]:
    """Rerun a finished channels search in its own folder with a higher cap:
    the same endpoints, level and settings, the conformer pools it saved (so
    every conformer keeps its number, and every pair its label), and more
    pairs per mechanism. Pairs already searched are skipped (their folders
    are done); only the new ones run, then TS/IRC and classification over
    all of them."""
    src = ctx.source
    if src is None:
        raise WorkspaceError("Sample more paths follows up on a finished Reaction channels run")
    old = int((src.get("params") or {}).get("pairs_per_mechanism", 0) or 0)
    if old == 0:
        raise WorkspaceError("that run already searched every pair of every mechanism (0 = all)")
    if p.pairs_per_mechanism != 0 and p.pairs_per_mechanism <= old:
        raise WorkspaceError(f"that run already searched {old} per mechanism: ask for more than {old} (or 0 = all)")
    out = Path(src["output_dir"])
    pools = {side: out / "conformers" / f"{side}_pool.xyz" for side in ("start", "end")}
    missing = [side for side, fp in pools.items() if not fp.exists()]
    if missing:
        raise WorkspaceError(f"that run saved no {' or '.join(missing)} conformer pool (an older run): new pairs "
                             "could not be numbered like its earlier ones, so it cannot be extended safely")
    if not (out / "pair_mechanisms.json").exists():
        raise WorkspaceError("that run has no pair table (pair_mechanisms.json) to extend")
    flags = {"--pairs-per-mechanism": p.pairs_per_mechanism, "--workers": p.workers,
             "--start-pool": pools["start"], "--end-pool": pools["end"], "--output": out}
    if p.rmsd_window is not None:
        flags["--atom-mapping-rmsd-window"] = f"{p.rmsd_window:g}"
    return _replace_flags(list(src["argv"]), flags)


def _build_tsopt(ctx: JobContext, p: TsOptParams) -> list[str]:
    guess = ctx.snapshot_structure(ctx.structures[0], "guess")
    return ["ts", "--guess", str(guess), *ctx.common_flags(), *generic_flags(p), "--output", str(ctx.output_dir)]


def _build_expand(ctx: JobContext, p: ExpandParams) -> list[str]:
    argv = _build_discovery("expand")(ctx, p)
    if p.generator == "crest-msreact":
        out = argv.index("--output")
        argv[out:out] = ["--generator", "crest-msreact", "--generator-option", f"mode={p.msreact_mode}",
                         "--generator-option", f"nbonds={p.msreact_nbonds}",
                         "--generator-option", f"nshifts={p.msreact_nshifts}"]
    elif p.generator == "nanoreactor":
        out = argv.index("--output")
        argv[out:out] = ["--generator", "nanoreactor", "--generator-option", f"temperature={p.reactor_temperature:g}",
                         "--generator-option", f"time_ps={p.reactor_time_ps:g}",
                         "--generator-option", f"compress={p.reactor_compress:g}"]
        if p.reactor_embedded:
            argv[out:out] = ["--generator-option", "embedded=true"]
    return argv


def _build_discovery(command: str):
    def build(ctx: JobContext, p: Params) -> list[str]:
        seed = ctx.snapshot_structure(ctx.structures[0], "seed")
        return ["discovery", command, str(seed), *ctx.common_flags(), *generic_flags(p),
                "--output", str(ctx.output_dir)]
    return build


class RetroParams(Params):
    # Chosen by the method switch, not in the form.
    method: Literal["templates", "reactiont5", "local-llm", "aizynthfinder"] = P(
        "templates", "Method", kind="custom", group="Hidden")
    routes: int = P(5, "Routes", "How many routes to report, cheapest first. The best is added to Explore; the "
                    "others from the result page.", cli="--routes",
                    ge=1, le=50)
    max_depth: int = P(6, "Most steps", "Longest route from a building block to the target.", cli="--max-depth",
                       ge=1, le=15)
    iterations: int = P(100, "Search budget (molecules expanded)", cli="--iterations", ge=1)
    time_limit: float = P(120.0, "Search budget (s)", cli="--time-limit", gt=0)
    stock: Literal["auto", "paroutes", "zinc"] = P(
        "auto", "Building blocks", "What a route may start from. PaRoutes: ~20k building blocks and common reagents "
        "(downloaded on first use, 0.8 MB). ZINC: AiZynthFinder's in-stock set (needs `mepd retro setup "
        "--aizynthfinder`). Automatic: ZINC if set up, else PaRoutes.", kind="custom",
        labels={"auto": "Automatic", "paroutes": "PaRoutes", "zinc": "ZINC in-stock"})
    stock_file: str = P("", "Also these (file)", "A file on the server with your own building blocks: one SMILES "
                        "or InChIKey per line, or a CSV with a smiles column.", kind="custom", advanced=True,
                        group="Building blocks")
    max_heavy: int = P(2, "Small molecules count as available up to (heavy atoms)", "H2, CO, ethylene... (the "
                       "stocks list common reagents themselves); 0: only what is in the stock.", cli="--max-heavy",
                       ge=0, advanced=True, group="Building blocks")
    verify: Literal["none", "top", "all"] = P(
        "none", "Check steps with path searches", "Off: routes come back in seconds as proposed reactions in "
        "Explore, where you can run 'Find TS' on any step. Best route / every route: each step also gets a path "
        "search + TS + IRC at the profile's level of theory (slow: minutes to hours per step).", cli="--verify",
        labels={"none": "Off", "top": "Best route", "all": "Every route"})
    llm_url: str = P("http://localhost:11434/v1", "Local LLM server", "An OpenAI-compatible endpoint on this "
                     "machine: Ollama (http://localhost:11434/v1), llama.cpp server (http://localhost:8080/v1), "
                     "vLLM, LM Studio.", kind="custom", requires="method=local-llm")
    llm_model: str = P("", "Model", "Its name on the server (e.g. qwen3:8b); empty: the first one it serves.",
                       kind="custom", requires="method=local-llm")
    llm_samples: int = P(1, "Answers per molecule", "Ask several times and pool the answers (a proposal given "
                         "repeatedly counts more).", kind="custom", ge=1, le=10, requires="method=local-llm")
    roundtrip: bool = P(True, "Round-trip check", "Ask the forward model to predict the product of each proposal; "
                        "one that doesn't give the target back is ranked lower.", kind="custom",
                        requires="method=reactiont5")
    width: int = P(10, "Proposals per molecule", cli="--width", ge=1, advanced=True, group="Search")
    workers: int = P(4, "Parallel workers", "Step checks run side by side.", cli="--workers", ge=1,
                     advanced=True, group="Search")


def _build_retro(ctx: JobContext, p: RetroParams) -> list[str]:
    rec = ctx.structures[0]
    target = rec.get("smiles") or str(ctx.snapshot_structure(rec, "target"))
    argv = ["retro", "plan", target, "--method", p.method, *ctx.common_flags(), *generic_flags(p)]
    stocks = {"auto": [], "paroutes": ["paroutes-n1", "paroutes-n5"], "zinc": ["zinc"]}[p.stock]
    if p.stock_file.strip():
        stocks = (stocks or ["auto"]) + [p.stock_file.strip()]
    for s in stocks:
        argv += ["--stock", s]
    opts = {"local-llm": {"url": p.llm_url, "model": p.llm_model, "samples": p.llm_samples},
            "reactiont5": {"roundtrip": str(p.roundtrip).lower()}}.get(p.method, {})
    for k, v in opts.items():
        if v != "":
            argv += ["--option", f"{k}={v}"]
    return argv + ["--output", str(ctx.output_dir)]


class NanoreactorParams(Params):
    copies: str = P("4", "Copies of each", "Molecules of each selected structure in the reactor: one number for all, "
                    "or by name, e.g. 'CC=O: 2, O: 6'.", kind="custom")
    temperature: float = P(2000.0, "Temperature (K)", "Hot on purpose: reactions that take hours at room "
                           "temperature happen within picoseconds.", cli="--temperature", gt=0)
    time_ps: float = P(5.0, "Simulated time (ps)", "With g-xTB, a few dozen atoms take about 4 min per ps (5 ps: about "
                       "20 min); an MLIP is much faster. Run longer from the result if it found too little. The status "
                       "line shows the time left.", cli="--time", gt=0)
    compress: float = P(0.6, "Piston squeeze", "Narrow wall radius as a fraction of the wide one: smaller pushes "
                        "the molecules harder together.", cli="--compress", gt=0, le=1)
    md_method: Literal["level", "auto", "gfn2", "gfn1", "gxtb"] = P(
        "level", "MD level", "What drives the discovery MD. 'Your level of theory' (the default) runs it on the "
        "compute profile below, with any calculator (MLIPs, ASE, g-xTB...): as fast as that calculator, and the "
        "MD sees the same chemistry as the rest of your workspace. The xtb methods run xtb's own MD (fast), at a "
        "different level. Automatic: GFN2-xTB if xtb is installed, else g-xTB, else your level of theory. Every "
        "species, reaction and TS is then refined at the profile's level.", cli="--md-method",
        labels={"level": "Your level of theory", "auto": "Automatic", "gfn2": "GFN2-xTB", "gfn1": "GFN1-xTB",
                "gxtb": "g-xTB (slower)"})
    # Not passed to the CLI: when the run ends, the web app queues one
    # ordinary TS search per reaction (its own job, watchable live) instead.
    connect: bool = P(False, "Find each reaction's TS", "When the run ends, start a TS search on every reaction's "
                      "subsystem (only the molecules it needs): one calculation per reaction, each watchable live, "
                      "whose barrier lands on its reaction. Or start them later on the reactions you pick.",
                      kind="custom")
    max_connect: int = P(20, "Max TS searches", "At most this many are started (fewest atoms first).", kind="custom",
                         ge=1, requires="connect")
    refine_live: bool = P(False, "Refine reactions as they appear", "Each reaction is refined (its species and "
                          "subsystem) as soon as its event is over, in parallel with the MD, and joins Explore and "
                          "Analyze at once: you can start its TS search while the MD goes on (and with 'Find each "
                          "reaction's TS', it starts by itself).", cli="--refine-live", kind="toggle")
    live_workers: int = P(2, "Refinement workers", "Reactions refined at the same time while the MD runs.",
                          cli="--live-workers", ge=1, requires="refine_live", advanced=True, group="Advanced")
    period: float = P(1.0, "Piston period (ps)", cli="--period", gt=0, advanced=True, group="Reactor")
    duty: float = P(0.75, "Time wide", "Fraction of each period at the wide radius.", cli="--duty", gt=0, le=1,
                    advanced=True, group="Reactor")
    radius: Optional[float] = P(None, "Wide radius (Å)", "Empty: from the number of atoms.", cli="--radius", gt=0,
                                advanced=True, group="Reactor")
    seed: int = P(0, "Packing seed", "Another seed packs the molecules differently: a new, independent run.",
                  cli="--seed", advanced=True, group="Reactor")
    min_lifetime: float = P(20.0, "Bond lifetime (fs)", "Bonds that live shorter than this are vibrations, not "
                            "chemistry.", cli="--min-lifetime", gt=0, advanced=True, group="Events")
    merge_window: float = P(100.0, "Merge window (fs)", "Bond changes this close in time, on shared molecules, are "
                            "one reaction (e.g. both halves of a proton relay).", cli="--merge-window", gt=0,
                            advanced=True, group="Events")
    instances: int = P(3, "Occurrences refined", "Per reaction and species: more gives more chances of a clean "
                       "optimization.", cli="--instances", ge=1, advanced=True, group="Events")
    # An interactive reactor's trajectory to analyze instead of running an MD (Sandbox › Stop & analyze).
    trajectory: str = P("", "Trajectory", kind="custom", group="Hidden")
    trajectory_start: str = P("", "Its first frame", kind="custom", group="Hidden")


def _parse_copies(text: str, recs: list[dict]) -> list[int]:
    """Copies per structure: '3' (all), '2, 6' (in selection order) or
    'CC=O: 2, O: 6' (by name or SMILES; unnamed ones get 1)."""
    text = str(text).strip()
    if ":" in text:
        counts = [1] * len(recs)
        for item in [x for x in text.replace(";", ",").split(",") if x.strip()]:
            name, _, n = item.rpartition(":")
            hits = [k for k, r in enumerate(recs) if name.strip() in (r.get("name"), r.get("smiles"))]
            if not hits or not n.strip().isdigit() or int(n) < 1:
                raise WorkspaceError(f"copies: {item.strip()!r} names no selected structure, or its count is not "
                                     f"a whole number >= 1")
            for k in hits:
                counts[k] = int(n)
        return counts
    parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p]
    try:
        counts = [int(p) for p in parts]
    except ValueError:
        raise WorkspaceError(f"copies must be whole numbers, e.g. '2, 6', or 'name: count' pairs; got {text!r}") from None
    if len(counts) == 1:
        counts *= len(recs)
    if len(counts) != len(recs) or any(c < 1 for c in counts):
        raise WorkspaceError(f"give one count (>= 1) per selected structure ({len(recs)}), one for all, or "
                             f"'name: count' pairs; got {text!r}")
    return counts


def _retro_methods():
    """The retrosynthesis methods as the family card's method switch; each
    says why it can't run here (a package or the AiZynthFinder setup missing)."""
    from mepd.retro.proposers import PROPOSERS

    summaries = {
        "templates": "USPTO reaction templates ranked by AiZynthFinder's policy network, run inside mepd: routes "
                     "in seconds. Data (~110 MB, CC-BY) downloads on first use.",
        "reactiont5": "ReactionT5, a local chemistry language model, writes the reactants of each molecule; a "
                      "forward model checks them. Runs on this machine (GPU if there is one); ~1.6 GB of weights "
                      "on first use.",
        "local-llm": "An open-weight LLM you run locally (Ollama, llama.cpp, vLLM) proposes disconnections; RDKit "
                     "checks every one. Nothing leaves your machine.",
        "aizynthfinder": "AiZynthFinder's own tree search with its USPTO models and the ZINC stock, in its own "
                         "environment.",
    }
    for rank, (name, prop) in enumerate(PROPOSERS.items()):
        yield {"label": prop.label, "rank": 10 + rank, "fixed": {"method": name}, "summary": summaries[name],
               "group": "Retrosynthesis: routes back to building blocks", "check": prop.problem}


def _build_nanoreactor(ctx: JobContext, p: NanoreactorParams) -> list[str]:
    if p.trajectory:
        # An interactive reactor's trajectory (frames 2 fs apart): analyzed and refined like an MD of our own.
        # Its first frame stands for the molecules (its charge and spin are in its comment line).
        flags = ctx.common_flags()
        flags = flags[flags.index("--inputs"):flags.index("--inputs") + 2] if "--inputs" in flags else []
        return ["discovery", "nanoreactor", p.trajectory_start, "--trajectory", p.trajectory, "--dump", "2.0",
                *flags, "--instances", str(p.instances), "--min-lifetime", str(p.min_lifetime),
                "--merge-window", str(p.merge_window), "--output", str(ctx.output_dir)]
    counts = _parse_copies(p.copies, ctx.structures)
    mols = [f"{ctx.snapshot_structure(r, f'molecule_{i}')}*{n}" for i, (r, n) in enumerate(zip(ctx.structures, counts))]
    flags = ctx.common_flags()
    flags = flags[flags.index("--inputs"):] if "--inputs" in flags else []   # total charge below, spin from electrons
    charge = sum(int(r["charge"]) * n for r, n in zip(ctx.structures, counts))
    return ["discovery", "nanoreactor", *mols, "--charge", str(charge), *flags, *generic_flags(p),
            "--output", str(ctx.output_dir)]


class ComplexParams(Params):
    method: Literal["dock", "nci", "qcg", "packed"] = P(
        "dock", "How", "dock: xtb's aISS docking; nci: a CREST ensemble of arrangements; qcg: a CREST solvation "
        "shell (one solute, copies of one solvent); packed: placed without a search.", cli="--method")
    counts: str = P("1", "How many of each", "One count per selected structure, in selection order.", kind="custom")
    keep: int = P(3, "Geometries kept", "dock and nci: the best few, each a geometry of the complex.",
                  cli="--keep", ge=1, le=10)


def _build_complex(ctx: JobContext, p: ComplexParams) -> list[str]:
    counts = _parse_copies(p.counts, ctx.structures)
    mols = [f"{ctx.snapshot_structure(r, f'molecule_{i}')}*{n}" for i, (r, n) in enumerate(zip(ctx.structures, counts))]
    return ["complex", *mols, *generic_flags(p), "--output", str(ctx.output_dir)]


class NanoreactorMoreParams(Params):
    more_ps: float = P(10.0, "More time (ps)", "Added to the MD, continuing from its last frame: the same reactor, "
                       "piston and settings. Rounded up to whole piston periods.", kind="custom", gt=0)


def _build_nanoreactor_more(ctx: JobContext, p: NanoreactorMoreParams) -> list[str]:
    """Rerun a nanoreactor in its own folder with a longer --time: the MD
    resumes from its last segment (finished segments are kept), then the
    whole trajectory is analyzed again; species, reactions and TSs refined
    before are reused (nanoreactor.PreviousRefinement), only new ones are
    computed."""
    import math

    src = ctx.source
    if src is None:
        raise WorkspaceError("Run longer follows up on a finished nanoreactor run")
    if src.get("external") or not src.get("argv"):
        raise WorkspaceError("that run was read in place (not run here): there is no command to continue it with")
    argv = list(src["argv"])
    if "--trajectory" in argv:
        raise WorkspaceError("that run analyzed an existing trajectory: there is no MD to continue")
    out = Path(src["output_dir"])
    try:
        sched = json.loads((out / "md" / "schedule.json").read_text())
        segments = sched["segments"]
    except (OSError, ValueError, KeyError):
        raise WorkspaceError("that run has no MD schedule (md/schedule.json) to continue") from None
    now = float(sched.get("time_ps") or sum(d for d, _ in segments))
    period = float(argv[argv.index("--period") + 1]) if "--period" in argv else 1.0
    if abs(now / period - round(now / period)) > 1e-6:
        raise WorkspaceError(f"that run ended partway through a piston period ({now:g} ps, period {period:g} ps): "
                             "continuing it would change its last segment")
    more = math.ceil(p.more_ps / period - 1e-9) * period
    return _replace_flags(argv, {"--time": f"{now + more:g}", "--output": out})


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


def _build_design_optimize(ctx: JobContext, p: OptimizeParams) -> list[str]:
    from mepd.web import design

    d = ctx.ws.design
    if not d:
        raise WorkspaceError("there is no design to minimize")
    fp = ctx.job_dir / "inputs" / "design.xyz"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(design.to_xyz(d["molblock"], d["charge"], d["multiplicity"]))
    return ["optimize", str(fp), *ctx.common_flags(), *generic_flags(p), "--output", str(ctx.output_dir)]


def _build_design_tsopt(ctx: JobContext, p: TsOptParams) -> list[str]:
    from mepd.web import design

    d = ctx.ws.design
    if not d:
        raise WorkspaceError("there is no design to optimize")
    fp = ctx.job_dir / "inputs" / "guess.xyz"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(design.to_xyz(d["molblock"], d["charge"], d["multiplicity"]))
    return ["ts", "--guess", str(fp), *ctx.common_flags(), *generic_flags(p), "--output", str(ctx.output_dir)]


def _build_network_splits(ctx: JobContext, p: NetworkSplitsParams) -> list[str]:
    charges = {(r["charge"], r["multiplicity"]) for r in ctx.structures}
    if len(charges) > 1:
        raise WorkspaceError("all structures must share charge and multiplicity")
    natoms = {r["natoms"] for r in ctx.structures}
    if len(natoms) > 1:
        raise WorkspaceError("all structures must have the same atoms")
    minima = [str(ctx.snapshot_structure(r, f"minimum_{i}")) for i, r in enumerate(ctx.structures)]
    return ["network-splits", *minima, *ctx.common_flags(), *generic_flags(p), "--output", str(ctx.output_dir)]


def _solvent_labels() -> dict:
    from mepd.solvation import SOLVENTS

    return {k: f"{v.label} (ε {v.epsilon:g}, {v.kind})" for k, v in SOLVENTS.items()}


def _solvent_keys() -> tuple:
    from mepd.solvation import SOLVENTS

    return tuple(SOLVENTS)


SolventKey = Literal[_solvent_keys()]


class SolventParams(Params):
    solvents: list[SolventKey] = P(
        ["water", "methanol", "dmso", "acetonitrile", "thf", "toluene"], "Solvents",
        "Each is an implicit (continuum) solvent. Pick several kinds (protic, polar aprotic, nonpolar) to see a trend.",
        kind="custom", labels=_solvent_labels())
    mode: Literal["single-point", "reoptimize"] = P(
        "single-point", "Geometries",
        "Keep gas-phase: solvent energies along the gas-phase path (seconds; a first estimate, and mepd says when "
        "it does not hold). Re-optimize: search the TS again in each solvent, with its IRC and minimized ends "
        "(minutes per solvent).", cli="--mode",
        labels={"single-point": "Keep gas-phase (fast)", "reoptimize": "Re-optimize in solvent"})
    temperature: float = P(298.15, "Temperature (K)", "For half-lives and rates (298.15 K = 25 °C).",
                           cli="--temperature", gt=0)
    model: Literal["alpb", "gbsa", "cpcmx"] = P(
        "alpb", "Solvent model", "GFN2-xTB's implicit models: ALPB (default), GBSA, CPCM-X.", cli="--model",
        advanced=True)


def _build_solvent(ctx: JobContext, p: SolventParams) -> list[str]:
    if ctx.source is None:
        raise WorkspaceError("Solvent effects follow up on a finished TS search")
    if not p.solvents:
        raise WorkspaceError("pick at least one solvent")
    text = ctx.ws.read_profile(ctx.profile) if ctx.profile else ""
    if "[solvation]" in (text or ""):
        raise WorkspaceError("this job already ran in solvent (its profile has a [solvation] table)")
    argv = ["solvent", ctx.source["output_dir"], *ctx.common_flags()]
    for key in dict.fromkeys(p.solvents):
        argv += ["--solvent", key]
    return [*argv, *generic_flags(p), "--output", str(ctx.output_dir)]


class MechanoParams(Params):
    mode: Literal["bell", "reoptimize"] = P(
        "bell", "Method", "Estimate: Bell's first-order barrier shifts from the geometries you have (instant). "
        "Check under force: re-optimize the reactant, TS and IRC on the force-modified surface for the main levers "
        "(minutes).", cli="--mode", labels={"bell": "Estimate (instant)", "reoptimize": "Check under force"})
    pair: str = P("", "Atoms to pull", "Two atom numbers 'i,j' (0-based, as the viewer's atom indices show). "
                  "Empty: rank every heavy-atom pair.", kind="custom")
    forces: str = P("0.5, 1, 1.5", "Forces to check (nN)", "Used by 'Check under force'.", kind="custom",
                    requires="mode=reoptimize")
    max_force: float = P(2.5, "Largest force (nN)", "Scanned for selectivity switches. Covalent bonds break at ~4–6 nN.",
                         cli="--max-force", gt=0, le=6, advanced=True)
    hydrogens: bool = P(False, "Pull on hydrogens too", cli="--hydrogens", kind="toggle", advanced=True)
    temperature: float = P(298.15, "Temperature (K)", "For half-lives (298.15 K = 25 °C).", cli="--temperature",
                           gt=0, advanced=True)


def _build_mechano(ctx: JobContext, p: MechanoParams) -> list[str]:
    if ctx.source is None:
        raise WorkspaceError("Mechanical force follows up on a finished TS search or channels run")
    argv = ["force", ctx.source["output_dir"], *ctx.common_flags()]
    if p.pair.strip():
        parts = p.pair.replace("-", ",").split(",")
        if len(parts) != 2 or not all(x.strip().isdigit() for x in parts):
            raise WorkspaceError(f"atoms to pull: give two atom numbers like 0,7 (got {p.pair!r})")
        argv += ["--pair", ",".join(x.strip() for x in parts)]
    if p.mode == "reoptimize":
        try:
            forces = [float(x) for x in p.forces.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            raise WorkspaceError(f"forces: give numbers in nN, e.g. 0.5, 1, 1.5 (got {p.forces!r})") from None
        if not forces or any(not 0 < f <= 6 for f in forces):
            raise WorkspaceError("forces must be between 0 and 6 nN")
        for f in forces:
            argv += ["--force", f"{f:g}"]
    return [*argv, *generic_flags(p), "--output", str(ctx.output_dir)]


def _group_keys() -> tuple:
    from mepd.substituents import group_smiles

    return tuple(group_smiles())


def _group_labels() -> dict:
    from mepd.substituents import SHORT, SIGMA_P

    return {g: SHORT.get(g, g) + (f" (σp {SIGMA_P[g]:+.2f})" if g in SIGMA_P else "") for g in _group_keys()}


GroupKey = Literal[_group_keys()]


class SubstituentParams(Params):
    groups: list[GroupKey] = P(
        ["amino", "hydroxyl", "methoxy", "methyl", "fluoro", "chloro", "trifluoromethyl", "cyano", "nitro"],
        "Groups", "Each replaces one hydrogen at a time. Donors to acceptors shows the electronic trend.",
        kind="custom", labels=_group_labels())
    mode: Literal["fast", "reoptimize"] = P(
        "fast", "Geometries", "Fast: relax only the new group (seconds per variant). Re-optimize: TS search, IRC "
        "and minimized reactant for every variant (about a minute each).", cli="--mode",
        labels={"fast": "Relax the group (fast)", "reoptimize": "Re-optimize the TS"})
    sites: str = P("", "Where", "Atom numbers (0-based, as the viewer shows): a hydrogen, or the heavy atom whose "
                   "hydrogens to replace. Empty: every hydrogen that is not transferred.", kind="custom")
    max_sites: int = P(8, "Most sites", "Closest to the reacting atoms first.", cli="--max-sites", ge=1, le=40,
                       advanced=True)


def _build_substituents(ctx: JobContext, p: SubstituentParams) -> list[str]:
    if ctx.source is None:
        raise WorkspaceError("Substituent effects follow up on a finished TS search or channels run")
    if not p.groups:
        raise WorkspaceError("pick at least one group")
    argv = ["substituents", ctx.source["output_dir"], *ctx.common_flags()]
    for g in dict.fromkeys(p.groups):
        argv += ["--group", g]
    for x in [t.strip() for t in p.sites.replace(";", ",").split(",") if t.strip()]:
        if not x.isdigit():
            raise WorkspaceError(f"where: give atom numbers like 4, 7 (got {p.sites!r})")
        argv += ["--site", x]
    return [*argv, *generic_flags(p), "--output", str(ctx.output_dir)]


PAIR = "Connect two structures"
EXPLORE = "Explore around a structure"
SET = "Across a set of structures"
FROM_TS = "Past the transition state"

# ------------------------------------------------------------------ QM/MM

QMMM = "QM/MM"
# Operations that run on a QM/MM system (the rest need whole molecules:
# atom mappings, conformers, SMILES, implicit solvent...).
QMMM_OPS = {"ts", "optimize", "tsopt", "hessian-sample", "graph-enumeration", "network-splits", "qmmm-inspect",
            "qmmm-embed"}
# Operations that take a mix of QM/MM and gas-phase structures (their own rules).
QMMM_MIXED_OPS = {"qmmm-embed"}

ShellSolvent = Literal["water", "methanol", "ethanol", "acetonitrile", "dmso", "acetone", "thf",
                       "dichloromethane", "chloroform", "benzene", "hexane"]


class QmmmBuildParams(Params):
    solvent: ShellSolvent = P("water", "Solvent", "Explicit solvent molecules around the molecule.",
                              cli="--solvent")
    shell: float = P(6.0, "Shell thickness (Å)", "How far the solvent reaches beyond the molecule.",
                     cli="--shell", gt=1.0, le=20.0)
    active_radius: float = P(5.0, "Moving shell (Å)", "Solvent within this distance of the molecule moves in "
                             "optimizations; the rest is frozen.", cli="--active-radius", ge=0.0, le=20.0)
    mm: Literal["gfnff", "tip3p", "gfn2", "gfn1"] = P(
        "gfnff", "Environment level", "How the solvent is described. TIP3P water: fixed charges that the QM "
        "calculation feels (electrostatic embedding: needed when charges separate, e.g. ions forming); it needs "
        "a QM level that takes point charges (Psi4, in Settings). The others: mechanical embedding, any QM level.",
        cli="--mm", labels={"gfnff": "GFN-FF force field", "tip3p": "TIP3P water, electrostatic embedding",
                            "gfn2": "GFN2-xTB", "gfn1": "GFN1-xTB"})
    relax: bool = P(True, "Relax the solvent", "GFN-FF minimization of the solvent around the fixed molecule.",
                    cli="--relax", kind="toggle", advanced=True, group="Advanced")
    seed: int = P(0, "Random seed", cli="--seed", advanced=True, group="Advanced")


def _build_qmmm_build(ctx: JobContext, p: QmmmBuildParams) -> list[str]:
    (rec,) = ctx.structures
    if rec.get("qmmm"):
        raise WorkspaceError(f"{rec['name']} is already a QM/MM system")
    if is_ts(rec):
        raise WorkspaceError("embed a minimum (a TS is better found again inside the solvent)")
    xyz = ctx.snapshot_structure(rec, "solute")
    return ["qmmm", "build", str(xyz), "--charge", str(rec["charge"]), "--multiplicity", str(rec["multiplicity"]),
            *generic_flags(p), "--output", str(ctx.output_dir)]


class QmmmReactionParams(QmmmBuildParams):
    use_ts: bool = P(True, "Bring the TS along", "If a TS between these two is known (an IRC-verified one from a "
                     "TS search), put it into the solvent too and re-optimize it there, with its IRC: the most "
                     "reliable way to the solvated reaction's TS.", kind="custom")


def _build_qmmm_reaction(ctx: JobContext, p: QmmmReactionParams) -> list[str]:
    a, b = ctx.structures
    for r in (a, b):
        if r.get("qmmm"):
            raise WorkspaceError(f"{r['name']} is already in a QM/MM system: select two gas-phase structures")
    if a["natoms"] != b["natoms"] or a["charge"] != b["charge"] or a["multiplicity"] != b["multiplicity"]:
        raise WorkspaceError("the two ends must have the same atoms, charge and spin")
    argv = ["qmmm", "reaction", "--start", str(ctx.snapshot_structure(a, "start")),
            "--end", str(ctx.snapshot_structure(b, "end")),
            "--charge", str(a["charge"]), "--multiplicity", str(a["multiplicity"])]
    if p.use_ts:
        found = edge_route_ts(ctx.ws, ctx.jobs, [a["id"], b["id"]], ctx.edge_ids)
        if found is not None:
            ts = ctx.job_dir / "inputs" / "ts.xyz"
            ts.write_text(found[2])
            argv += ["--ts", str(ts)]
    return argv + [*generic_flags(p), "--output", str(ctx.output_dir)]


class QmmmEmbedParams(Params):
    pass


def _qmmm_embed_pair(ctx: JobContext) -> tuple[dict, dict]:
    sys_recs = [r for r in ctx.structures if r.get("qmmm")]
    gas = [r for r in ctx.structures if not r.get("qmmm")]
    if len(sys_recs) != 1 or len(gas) != 1:
        raise WorkspaceError("select one structure of a QM/MM system and one gas-phase structure of its solute")
    return sys_recs[0], gas[0]


def _build_qmmm_embed(ctx: JobContext, p: QmmmEmbedParams) -> list[str]:
    host, gas = _qmmm_embed_pair(ctx)
    region = ctx.ws.qmmm_region(host["qmmm"])
    solute = region.solute
    if gas["natoms"] != len(solute):
        raise WorkspaceError(f"{gas['name']} has {gas['natoms']} atoms, the solute of this QM/MM system {len(solute)}")
    inputs = ctx.job_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ctx.ws.qmmm_region_path(host["qmmm"]), inputs / "qmmm_region.json")
    return ["qmmm", "embed", str(ctx.snapshot_structure(gas, "solute")), "--region", str(inputs / "qmmm_region.json"),
            "--into", str(ctx.snapshot_structure(host, "system")), "--output", str(ctx.output_dir)]


class ProteinSitesParams(Params):
    protein: str = P("", "Protein", "A PDB ID to download (e.g. 2CHT), or the path of a PDB file on this machine.",
                     cli="--protein", kind="custom")
    chains: str = P("", "Chains", "Chains to keep, e.g. A,B,C (empty: all). A big crystal often holds several copies.",
                    cli="--chains")
    ph: float = P(7.0, "pH", "Protonation states of the protein's residues.", cli="--ph", ge=0.0, le=14.0)
    exhaustiveness: int = P(8, "Search effort", "Vina's exhaustiveness in each docking box.",
                            cli="--exhaustiveness", ge=1, advanced=True, group="Advanced")
    spacing: float = P(12.0, "Box spacing (Å)", "Spacing of the docking boxes over the protein (each 20 Å).",
                       cli="--spacing", gt=0.0, advanced=True, group="Advanced")
    max_sites: int = P(20, "Sites kept", cli="--max-sites", ge=1, advanced=True, group="Advanced")
    workers: int = P(4, "Docking processes", cli="--workers", ge=1, advanced=True, group="Advanced")


def _build_protein_sites(ctx: JobContext, p: ProteinSitesParams) -> list[str]:
    (rec,) = ctx.structures
    if rec.get("qmmm"):
        raise WorkspaceError(f"{rec['name']} is already a QM/MM system: dock the molecule (or complex) itself")
    if not p.protein.strip():
        raise WorkspaceError("name the protein: a PDB ID or the path of a PDB file")
    argv = ["qmmm", "protein-sites", str(ctx.snapshot_structure(rec, "species")), "--protein", p.protein.strip(),
            "--charge", str(rec["charge"]), "--multiplicity", str(rec["multiplicity"])]
    return argv + [*generic_flags(p), "--output", str(ctx.output_dir)]


class ProteinBuildParams(Params):
    site: int = P(0, "Site", "Which docked site (its number in the result).", cli="--site", ge=0)
    qm_residues: str = P("", "Residues in the QM region", "Side chains that take part in the chemistry, e.g. "
                         "A:ARG90 A:GLU78 (cut at CA-CB, with link atoms).", cli="--qm-residues")
    water_shell: float = P(8.0, "Water shell (Å)", "TIP3P water within this distance of the species.",
                           cli="--water-shell", ge=0.0)
    active_radius: float = P(6.0, "Moving shell (Å)", "Water and protein atoms within this distance of the QM "
                             "region move in optimizations; the rest is frozen.", cli="--active-radius", ge=0.0)
    freeze_protein: bool = P(False, "Freeze the protein", "Only the water around the species moves; the protein "
                             "still acts on the QM region through its charges.", cli="--freeze-protein",
                             kind="switch")
    cutoff: float = P(12.0, "Force-field cutoff (Å)", "Non-bonded cutoff for the force field (0: none). The QM "
                      "region feels every charge regardless.", cli="--cutoff", ge=0.0, advanced=True,
                      group="Advanced")
    edge: str = P("", "Bring a reaction along", "An edge of the docked species (its id): its other end, and its "
                  "TS if one is known, are put into the same site, as for a reaction taken into solvent.",
                  kind="custom")


def _build_protein_build(ctx: JobContext, p: ProteinBuildParams) -> list[str]:
    src = ctx.source
    if src is None or src.get("op") != "qmmm-protein-sites" or src.get("status") != "done":
        raise WorkspaceError("build from a finished protein-site search")
    argv = ["qmmm", "protein-build", str(src["output_dir"]), *generic_flags(p)]
    if p.edge:
        edge = ctx.ws.snapshot()["edges"].get(p.edge)
        species = (src.get("targets") or {}).get("structures") or []
        if edge is None or not species or species[0] not in (edge["source"], edge["target"]):
            raise WorkspaceError("that reaction is not one of the docked species' edges")
        other = edge["target"] if edge["source"] == species[0] else edge["source"]
        rec = ctx.ws.structure_view(other)
        if rec.get("qmmm"):
            raise WorkspaceError(f"{rec['name']} is already in a QM/MM system: bring a gas-phase reaction")
        argv += ["--end", str(ctx.snapshot_structure(rec, "end"))]
        found = edge_route_ts(ctx.ws, ctx.jobs, [species[0], other], [p.edge])
        if found is not None:
            ts = ctx.job_dir / "inputs" / "ts.xyz"
            ts.parent.mkdir(parents=True, exist_ok=True)
            ts.write_text(found[2])
            argv += ["--ts", str(ts)]
    return argv + ["--output", str(ctx.output_dir)]


class QmmmInspectParams(Params):
    entry: str = P("", "Result entry", "Which path of the result (its id).", kind="custom")


def _build_qmmm_inspect(ctx: JobContext, p: QmmmInspectParams) -> list[str]:
    src = ctx.source
    if src is None or not src.get("qmmm"):
        raise WorkspaceError("the energy split is for QM/MM jobs")
    fp = ctx.ws.jobs_dir / src["id"] / "result.json"
    try:
        result = json.loads(fp.read_text())
    except (OSError, ValueError):
        raise WorkspaceError("open the job's result first (it is read once, then cached)") from None
    entry = next((e for g in result.get("groups", []) for e in g["entries"] if e["id"] == p.entry), None)
    if entry is None:
        raise WorkspaceError(f"no result entry {p.entry!r}")
    frames = ctx.job_dir / "inputs" / "frames.xyz"
    frames.parent.mkdir(parents=True, exist_ok=True)
    frames.write_text("".join(f["xyz"] if f["xyz"].endswith("\n") else f["xyz"] + "\n" for f in entry["frames"]))
    flags = ctx.common_flags()
    inputs = flags[flags.index("--inputs") + 1]
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    return ["qmmm", "inspect", str(Path(inputs).parent / "qmmm_region.json"), str(frames), "--inputs", inputs,
            "--output", str(ctx.output_dir / "report.json")]


OPERATIONS: dict[str, Operation] = {op.key: op for op in [
    Operation(
        "ts", "Transition state", "Find the minimum-energy path and its transition state(s) between "
        "two structures, optionally refined by TS optimization and IRC. (`mepd run`)",
        "pair", PAIR, TsParams, _build_ts, min_structures=2,
        produces=["transition states", "IRC endpoints", "intermediates"], family=TS_FAMILY, methods=({
            "label": "These endpoints", "rank": 0,
            "summary": "One path search between these two structures (the chosen conformers): its TS(s), "
                       "optimized and followed by IRC. Quick. Sample more paths from the result later if you want "
                       "other conformers or atom mappings."},)),
    Operation(
        "channels", "Reaction channels", "Sample reactant/product conformers and atom mappings, search "
        "every distinct mechanism, and classify the TSs into direct, multi-step and off-target "
        "channels. (`mepd channels`)",
        "pair", PAIR, ChannelsParams, _build_channels, min_structures=2,
        produces=["transition states per channel", "conformers", "off-target products"],
        cli_extra_flags=("--charge", "--multiplicity", "--inputs", "--output", "--start-pool", "--end-pool"),
        family=TS_FAMILY, methods=({
            "label": "Sample conformers and mappings", "rank": 1,
            "summary": "Reaction channels: sample both ends' conformers and the atom mappings, search every "
                       "distinct mechanism, and sort the TSs into direct, multi-step and off-target channels. "
                       "Slower; finds the lowest channel rather than the nearest one."},)),
    Operation(
        "optimize", "Optimize geometry", "Minimize the selected structures at the chosen profile's level of "
        "theory, replacing their geometry and energy in place. (`mepd optimize`)",
        "set", EXPLORE, OptimizeParams, _build_optimize, min_structures=1,
        produces=["optimized geometries (in place)"]),
    Operation(
        "design-optimize", "Minimize the design", "Minimize the Design tab's molecule at the chosen profile's "
        "level of theory; its geometry and energy are replaced in place. (`mepd optimize`)",
        "design", "Design", OptimizeParams, _build_design_optimize, min_structures=0,
        produces=["the minimized design (in place)"]),
    Operation(
        "design-tsopt", "Optimize the design as a TS", "Treat the Design tab's molecule as a TS guess: saddle "
        "optimization, then an IRC to the reactant and product it connects. If it does not converge, the design "
        "goes back to the structure submitted. (`mepd ts --guess`)",
        "design", "Design", TsOptParams, _build_design_tsopt, min_structures=0,
        produces=["a TS and its IRC ends, from the design"]),
    Operation(
        "complex", "Build a complex", "Several molecules together (not bonded), from species in the graph: docked, "
        "an ensemble, a solvation shell or packed; then minimized at the workspace level. (`mepd complex`)",
        "complex", "Complexes", ComplexParams, _build_complex, min_structures=1, cli_path=("complex",),
        produces=["complexes"], methods=(
            {"label": "Dock", "rank": 0, "fixed": {"method": "dock"}, "programs": ["xtb"],
             "install": "conda install -c conda-forge xtb",
             "summary": "xtb's aISS docking finds where the molecules stick (H-bonds, stacking). Seconds."},
            {"label": "Ensemble", "rank": 1, "fixed": {"method": "nci"}, "programs": ["xtb", "crest"],
             "install": "conda install -c conda-forge crest",
             "summary": "CREST searches arrangements of the complex; the best few are kept. Minutes."},
            {"label": "Solvation shell", "rank": 2, "fixed": {"method": "qcg"}, "programs": ["xtb", "crest"],
             "install": "conda install -c conda-forge crest",
             "summary": "CREST grows the copies of one solvent around one solute. A minute or so."},
            {"label": "Packed", "rank": 3, "fixed": {"method": "packed"},
             "summary": "Random orientations in a small sphere, like the nanoreactor (Packmol if installed). Instant."})),
    Operation(
        "tsopt", "Optimize TS from guess", "Treat the structure as a TS guess: saddle optimization, "
        "optionally followed by IRC. (`mepd ts`)",
        "structure", EXPLORE, TsOptParams, _build_tsopt, produces=["transition state", "IRC endpoints"]),
    Operation(
        "hessian-sample", "Hessian sampling", "Displace along every normal mode and re-optimize to find "
        "nearby minima. (`mepd discovery hessian-sample`)",
        "structure", EXPLORE, HessianSampleParams, _build_discovery("hessian-sample"),
        produces=["nearby minima"], family=NETWORK, methods=({
            "label": "Hessian sampling", "rank": 2, "summary": "Displace along every normal mode and re-optimize: the minima "
            "one vibration away (conformers, nearby isomers)."},)),
    Operation(
        "hessian-global", "Hessian basin hopping", "Repeated Hessian sampling with Metropolis acceptance: "
        "a global search over minima reachable from the seed. (`mepd discovery hessian-global`)",
        "structure", EXPLORE, HessianGlobalParams, _build_discovery("hessian-global"),
        produces=["accepted minima"], family=NETWORK, methods=({
            "label": "Basin hopping", "rank": 3, "summary": "Repeated Hessian sampling with Metropolis acceptance: a global "
            "search over the minima reachable from the seed."},)),
    Operation(
        "conformers", "Conformers", "Sample this molecule's conformers with RDKit or CREST and minimize them; "
        "they are added to the node, and its lowest-energy conformer represents it. (`mepd conformers`)",
        "structure", EXPLORE, ConformersParams, _build_conformers,
        produces=["conformers of the node"], cli_path=("conformers",),
        cli_extra_flags=("--charge", "--multiplicity", "--inputs", "--output")),
    Operation(
        "graph-enumeration", "Reaction network expansion", "Propose products by breaking and forming up to two "
        "bonds on the molecular graph, keep those with a valid Lewis structure, optimize them, and grow the network "
        "from the species the kinetics reach. (`mepd discovery expand`)",
        "structure", EXPLORE, ExpandParams, _build_expand,
        produces=["product species", "proposed reactions", "network edges (with path search)"],
        cli_path=("discovery", "expand"),
        cli_extra_flags=("--charge", "--multiplicity", "--inputs", "--output", "--generator", "--generator-option"),
        family=NETWORK, methods=(
            {"label": "Bond rules", "rank": 0, "fixed": {"generator": "bond-rules"},
             "summary": "Break and form up to n bonds on the molecular graph, keep products with a valid Lewis "
                        "structure, optimize them, and grow from the species the kinetics reach."},
            {"label": "CREST msreact", "rank": 1, "fixed": {"generator": "crest-msreact"}, "programs": ("crest", "xtb"),
             "install": "conda install -c conda-forge crest xtb",
             "summary": "CREST's fragment generator (msreact): biased GFN2-xTB optimizations find the fragments and "
                        "isomers the molecule can reach, e.g. likely precursors (read backwards) or nearby products. "
                        "They are re-optimized at your level of theory and grown like any other species. One molecule at a "
                        "time: a cluster of several gets no products."},
            {"label": "Nanoreactor", "rank": -1, "fixed": {"generator": "nanoreactor"}, "qmmm_only": True,
             "summary": "Hot molecular dynamics of the QM region (capped with its link hydrogens) in a periodically "
                        "squeezing wall: its state after each reaction event is put back into the environment, "
                        "optimized embedded and grown like any other species."})),
    Operation(
        "retrosynthesis", "Retrosynthesis", "Routes from purchasable building blocks to this molecule: a "
        "tree search over single steps proposed by the chosen method; each route's steps join Explore as "
        "reactions coming back from the target, ready for 'Find TS'. (`mepd retro plan`)",
        "structure", EXPLORE, RetroParams, _build_retro,
        produces=["routes", "precursor species", "proposed reactions (optionally path-searched)"],
        cli_path=("retro", "plan"), cli_extra_flags=("--charge", "--multiplicity", "--inputs", "--output", "--stock",
                                                     "--option", "--method"),
        family=NETWORK, methods=tuple(_retro_methods())),
    Operation(
        "nanoreactor", "Nanoreactor", "Put the selected molecules in a hot box whose wall periodically squeezes "
        "them, and watch what reacts. Each reaction is cut out with only the molecules it needs (a water that "
        "relays a proton is part of it; bystanders are not), refined at your level of theory, and joins Explore "
        "as a reaction dot. (`mepd discovery nanoreactor`)",
        "set", EXPLORE, NanoreactorParams, _build_nanoreactor, min_structures=1,
        produces=["species", "reactions with only the molecules they need", "TSs on those subsystems"],
        cli_path=("discovery", "nanoreactor"), cli_extra_flags=("--charge", "--inputs", "--output"),
        family=NETWORK, methods=({
            "label": "Nanoreactor", "rank": -1,
            "summary": "Hot, periodically squeezed molecular dynamics of the selected molecules (several copies, "
                       "solvent, partners): whatever reacts becomes a reaction dot with only the molecules it needs, "
                       "refined at your level of theory. Select several structures to put them in together."},)),
    Operation(
        "network-splits", "Paths between isomers", "Search a path between every pair of the selected isomers (same "
        "atoms, same charge and spin) and join the steps it finds, with any intermediates, into one network. For "
        "molecules with different atoms, use the Nanoreactor. (`mepd network-splits`)",
        "set", SET, NetworkSplitsParams, _build_network_splits, min_structures=2, same_atoms=True,
        produces=["network edges", "intermediates"]),
    Operation(
        "vri", "Valley-ridge inflection", "Scan the IRC from this edge's transition state for a valley-ridge "
        "transition, where the path can split after the TS (a post-TS bifurcation into two products, P1 and P2). "
        "(`mepd discovery vri`)",
        "pair", FROM_TS, VriParams, _build_vri, min_structures=2, needs_route_ts=True,
        produces=["products P1 and P2", "TS2 between them"],
        cli_path=("discovery", "vri"), cli_extra_flags=("--skip-ts2", "--charge", "--multiplicity", "--inputs", "--output")),
    Operation(
        "channels-more", "Sample more paths", "Search more conformer pairs per mechanism for a finished "
        "Reaction channels run, reusing everything it computed (its conformers, and every path already searched) "
        "and running only the new pairs; the result then covers all of them. (`mepd channels`)",
        "job", PAIR, ChannelsMoreParams, _build_channels_more,
        source_ops=("channels",), cli_path=("channels",)),
    Operation(
        "nanoreactor-more", "Run longer", "Continue the reactor's MD from its last frame for more time: new "
        "reactions join this page and Explore; everything refined before is reused. (`mepd discovery nanoreactor`)",
        "job", "Reaction discovery", NanoreactorMoreParams, _build_nanoreactor_more,
        source_ops=("nanoreactor",), cli_path=("discovery", "nanoreactor")),
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
    Operation(
        "solvent", "Solvent effects", "Recompute this reaction's barrier in implicit solvents and see which "
        "conditions speed it up or slow it down: barriers, half-lives, and the temperature each solvent needs. "
        "(`mepd solvent`)",
        "job", "Reaction conditions", SolventParams, _build_solvent, source_ops=("ts", "tsopt", "design-tsopt"),
        cli_path=("solvent",), cli_extra_flags=("--solvent", "--charge", "--multiplicity", "--inputs", "--output"),
        own_output=True, produces=["barriers in each solvent", "insights on conditions"]),
    Operation(
        "mechanochem", "Mechanical force", "Which atoms to pull on to speed a channel up or hold it back, how hard, "
        "and when pulling makes a slower channel win. (`mepd force`)",
        "job", "Reaction conditions", MechanoParams, _build_mechano,
        source_ops=("ts", "tsopt", "design-tsopt", "channels"), cli_path=("force",),
        cli_extra_flags=("--pair", "--force", "--charge", "--multiplicity", "--inputs", "--output"),
        own_output=True, produces=["pulling pairs per channel", "selectivity switches under force"],
        # Off in the web UI until its results are trusted (the user's call, 2026-09-29): not offered, not
        # run from here, results not shown. `mepd force` still works on the command line.
        available=False, unavailable_reason="Mechanical force is off in the web UI for now; `mepd force` runs it "
        "on the command line."),
    Operation(
        "substituents", "Substituent effects", "Swap hydrogens for functional groups, one at a time, and see how "
        "each shifts the barrier and which channel wins. (`mepd substituents`)",
        "job", "Reaction conditions", SubstituentParams, _build_substituents,
        source_ops=("ts", "tsopt", "design-tsopt", "channels"), cli_path=("substituents",),
        cli_extra_flags=("--group", "--site", "--charge", "--multiplicity", "--inputs", "--output"),
        own_output=True, produces=["barrier shift per site and group", "substituted TSs"]),
    Operation(
        "qmmm-build", "Put in explicit solvent (QM/MM)", "Surround this molecule with explicit solvent: the "
        "molecule is computed at the profile's level (QM), the solvent with a force field (MM). A new QM/MM "
        "system; every calculation on it then runs embedded. (`mepd qmmm build`)",
        "structure", QMMM, QmmmBuildParams, _build_qmmm_build, min_structures=1,
        produces=["a QM/MM system (molecule in solvent)"], cli_path=("qmmm", "build"),
        cli_extra_flags=("--charge", "--multiplicity", "--output")),
    Operation(
        "qmmm-inspect", "QM/MM energy split", "Split each frame's energy into the QM region and its environment. "
        "(`mepd qmmm inspect`)",
        "job", QMMM, QmmmInspectParams, _build_qmmm_inspect, own_output=True,
        source_ops=("ts", "optimize", "tsopt", "hessian-sample", "graph-enumeration", "network-splits"),
        cli_path=("qmmm", "inspect"), cli_extra_flags=("--inputs", "--output")),
    Operation(
        "qmmm-reaction", "Model in solvent (QM/MM)", "Take this gas-phase reaction into explicit solvent: the start "
        "is surrounded by solvent, the end (and the known TS) are put into that same solvent shell, all minimized "
        "embedded (the TS re-optimized there, with its IRC). A QM/MM edge, ready for a TS search. "
        "(`mepd qmmm reaction`)",
        "pair", QMMM, QmmmReactionParams, _build_qmmm_reaction, min_structures=2,
        produces=["the reaction in solvent: both ends and its TS as QM/MM structures"],
        cli_path=("qmmm", "reaction"), cli_extra_flags=("--start", "--end", "--ts", "--charge", "--multiplicity",
                                                        "--output")),
    Operation(
        "qmmm-protein-sites", "Place in a protein (QM/MM)", "Dock this molecule or complex into a protein, rigid "
        "in its own geometry, with AutoDock Vina over the whole protein; see where it binds, then build a QM/MM "
        "system at the site you choose (the protein and a water shell as the environment). "
        "(`mepd qmmm protein-sites`)",
        "structure", QMMM, ProteinSitesParams, _build_protein_sites, min_structures=1,
        produces=["the sites where it binds"], cli_path=("qmmm", "protein-sites"),
        cli_extra_flags=("--protein", "--charge", "--multiplicity", "--output")),
    Operation(
        "qmmm-protein-build", "QM/MM system at this site", "The species at the chosen docked site as a QM/MM "
        "system: protein (AMBER ff14SB) and TIP3P water around it as the environment, whose charges act on the QM "
        "region (needs a QM level that takes point charges: xTB or Psi4). (`mepd qmmm protein-build`)",
        "job", QMMM, ProteinBuildParams, _build_protein_build, own_output=True,
        source_ops=("qmmm-protein-sites",), produces=["a QM/MM system (species in the protein)"],
        cli_path=("qmmm", "protein-build"), cli_extra_flags=("--end", "--ts", "--output")),
    Operation(
        "qmmm-embed", "Put into this QM/MM system", "Put the gas-phase structure (a product, a TS: the same atoms "
        "as this system's solute) into the selected QM/MM structure's solvent, in place of its solute, then minimize "
        "it embedded (a TS is re-optimized as a TS). (`mepd qmmm embed`)",
        "set", QMMM, QmmmEmbedParams, _build_qmmm_embed, min_structures=2,
        produces=["the structure in the QM/MM system"], cli_path=("qmmm", "embed"),
        cli_extra_flags=("--region", "--into", "--output")),
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
