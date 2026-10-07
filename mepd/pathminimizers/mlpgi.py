"""MLP-GI path minimization (mepd.mlp_geodesic) as a mepd path method.

The path is optimized on one surface and reported on the profile's engine:

  mlp_model unset   the profile's own engine (every step costs one gradient
                    per node and per midpoint, ~2x the images)
  mlp_model = name  a machine-learned potential from `mepd models` (e.g.
                    "uma-s-1p1", "aimnet2"), optionally from `mlp_checkpoint`;
                    the finished path is then re-evaluated on the engine

neb-dynamics' `backend` / `model_path` settings are read too: backend
"engine" (or "auto", "chemcloud", ...) is the profile's engine, "fairchem" /
"mace" / "egret" with `model_path` a local checkpoint or a model name.
Tolerances follow 10.1021/acs.jctc.5c01221 (Table 1); the two energy
tolerances are given in kcal/mol, as in neb-dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from qcconst.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM

from mepd.chain import Chain
from mepd.elementarystep import IS_ELEM_STEP, ElemStepResults, check_if_elem_step, elem_step_check_kwargs
from mepd.engines.engine import Engine
from mepd.mlp_geodesic import GeodesicOptimizer, OptimizerConfig, PathData
from mepd.nodes.node import StructureNode
from mepd.pathminimizers.pathminimizer import PathMinimizer
from mepd.progress import print_chain_step, print_persistent, update_status

HARTREE_TO_EV = 27.211386245988
KCAL_MOL_TO_EV = 0.0433641
_FORCE_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANGSTROM          # Hartree/bohr -> eV/Angstrom
_ENGINE_BACKENDS = {"", "auto", "engine", "target-engine", "chemcloud", "qcop", "qccompute", "crest"}
# neb-dynamics' checkpoint file names -> mepd model names.
_CHECKPOINT_NAMES = {"esen_sm_conserving_all.pt": "esen-sm-conserving-all-omol",
                     "esen_md_direct_all.pt": "esen-md-direct-all-omol"}


def _as_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return dict(getattr(value, "__dict__", {}))


def optimizer_config(params: dict) -> OptimizerConfig:
    """OptimizerConfig from path_min_inputs: energy tolerances given in
    kcal/mol, the paper's names (beta, tau_refine, cutoff, ...) accepted,
    a refinement cutoff above 1 read as a percentage."""
    cfg = OptimizerConfig()
    kcal = {"fire_conv_geolen_tol", "fire_conv_erelpeak_tol"}
    for f in fields(OptimizerConfig):
        if params.get(f.name) is not None:
            value = params[f.name]
            setattr(cfg, f.name, float(value) * KCAL_MOL_TO_EV if f.name in kcal else type(f.default)(value))
    aliases = {"beta": ("variance_penalty_weight", KCAL_MOL_TO_EV), "tau_refine": ("refinement_step_interval", None),
               "cutoff": ("refinement_dynamic_threshold_fraction", None),
               "convergence_window": ("fire_conv_window", None),
               "path_length_tolerance": ("fire_conv_geolen_tol", KCAL_MOL_TO_EV),
               "barrier_height_tolerance": ("fire_conv_erelpeak_tol", KCAL_MOL_TO_EV)}
    for alias, (name, scale) in aliases.items():
        if params.get(alias) is not None and params.get(name) is None:
            value = float(params[alias]) * scale if scale else params[alias]
            setattr(cfg, name, type(getattr(OptimizerConfig, name))(value))
    if cfg.refinement_dynamic_threshold_fraction > 1.0:
        cfg.refinement_dynamic_threshold_fraction /= 100.0
    return cfg


def surrogate_kwds(params: dict) -> dict | None:
    """`mlip_engine_kwds` for the potential the path is optimized on, or
    None for the profile's engine."""
    model = params.get("mlp_model")
    checkpoint = params.get("mlp_checkpoint")
    device = params.get("mlp_device") or params.get("device")
    backend = str(params.get("backend") or "").strip().lower()
    if not model and backend not in _ENGINE_BACKENDS:
        path = str(params.get("model_path") or "")
        family = "mace" if backend in ("mace", "egret") else backend
        if path and Path(path).expanduser().is_file():
            checkpoint = str(Path(path).expanduser())
            model = {"fairchem": "esen-sm-conserving-all-omol", "mace": "mace-off"}.get(family, family)
        else:
            model = _CHECKPOINT_NAMES.get(Path(path).name, path) or None
        if not model:
            raise ValueError(f"MLP-GI backend {backend!r} needs `mlp_model` (a name from `mepd models`).")
    if not model:
        return None
    kwds = {"model": str(model)}
    if checkpoint:
        kwds["checkpoint"] = str(checkpoint)
    if device:
        kwds["device"] = str(device)
    return kwds


def _bare_node(template: StructureNode, coords_bohr: np.ndarray) -> StructureNode:
    """A node at new coordinates without a bond graph (evaluation only)."""
    node = template.copy()
    node._cached_result = None
    node._cached_gradient = None
    node._cached_energy = None
    node.structure = node.structure.model_copy(update={"geometry": coords_bohr})
    node.graph = None
    return node


def engine_evaluator(engine: Engine, template: StructureNode):
    """evaluate(coords [n, N, 3] Angstrom) -> (eV [n], eV/Angstrom [n, N, 3]) on `engine`."""
    def evaluate(coords: np.ndarray):
        nodes = [_bare_node(template, c * ANGSTROM_TO_BOHR) for c in coords]
        grads = np.asarray(engine.compute_gradients(nodes), dtype=float).reshape(coords.shape)
        energies = [n._cached_energy for n in nodes]
        if any(e is None for e in energies):
            energies = engine.compute_energies(nodes)
        return np.asarray(energies, dtype=float) * HARTREE_TO_EV, -grads * _FORCE_TO_EV_ANG
    return evaluate


@dataclass
class MLPGI(PathMinimizer):
    initial_chain: Chain
    engine: Engine
    parameters: object = None

    optimized: Chain = None
    chain_trajectory: list[Chain] = field(default_factory=list)
    grad_calls_made: int = 0
    geom_grad_calls_made: int = 0
    surrogate_calls_made: int = 0

    def __post_init__(self):
        self._params = _as_dict(self.parameters)
        self.config = optimizer_config(self._params)
        self._verbose = bool(self._params.get("v", False))
        self._surrogate = surrogate_kwds(self._params)

    def _status(self, message: str, persistent: bool = False) -> None:
        if self._verbose:
            print(message, flush=True)
        elif persistent:
            print_persistent(message=message)
        else:
            update_status(message)

    def _chain(self, pd: PathData, template: Chain, with_graphs: bool = False) -> Chain:
        """The path in `pd` as a Chain whose nodes carry its energies and gradients."""
        base = template.nodes[0]
        nodes = []
        for xyz, e, f in zip(pd.nodes, pd.energies, pd.forces):
            bohr = xyz * ANGSTROM_TO_BOHR
            node = base.update_coords(bohr) if with_graphs else _bare_node(base, bohr)
            node._cached_energy = float(e) / HARTREE_TO_EV
            node._cached_gradient = -np.asarray(f) / _FORCE_TO_EV_ANG
            nodes.append(node)
        out = template.copy()
        out.nodes = nodes
        return out

    def optimize_chain(self) -> ElemStepResults:
        chain = self.initial_chain.copy()
        self._status("MLP-GI: energies of the initial path")
        self.engine.compute_energies(chain)
        self.grad_calls_made += len(chain)
        self.chain_trajectory.append(chain)

        frozen = list(getattr(chain.parameters, "frozen_atom_indices", None) or [])
        on_engine = self._surrogate is None
        if on_engine:
            surface, label = self.engine, "the profile's engine"
        else:
            from mepd.engines.mlip import build_mlip_engine

            surface, label = build_mlip_engine(self._surrogate), self._surrogate["model"]
        self._status(f"MLP-GI on {label}", persistent=True)
        evaluate = engine_evaluator(surface, chain.nodes[0])
        if frozen:
            def evaluate(coords, _inner=evaluate):
                e, f = _inner(coords)
                f[:, frozen] = 0.0
                return e, f

        def on_step(pd: PathData, caption: str) -> None:
            snapshot = self._chain(pd, chain)
            if not on_engine:
                caption += f" (on {label})"
            self.chain_trajectory.append(snapshot)
            print_chain_step(snapshot, caption, force_update=True)

        frames = np.stack([np.asarray(n.coords) * BOHR_TO_ANGSTROM for n in chain.nodes])
        opt = GeodesicOptimizer(frames, list(chain.nodes[0].structure.symbols), evaluate, self.config,
                                on_step=on_step, on_status=self._status, align=not frozen)
        try:
            final = opt.optimize()
        finally:
            if on_engine:
                self.grad_calls_made += opt.evaluations
            else:
                self.surrogate_calls_made += opt.evaluations

        if on_engine:
            new_chain = self._chain(final, chain, with_graphs=True)
        else:
            self._status(f"MLP-GI: the path on the profile's engine ({len(final.nodes)} images)")
            new_chain = chain.copy()
            new_chain.nodes = [chain.nodes[0].update_coords(x * ANGSTROM_TO_BOHR) for x in final.nodes]
            self.engine.compute_gradients(new_chain)
            self.grad_calls_made += len(new_chain)
        # The ends exactly as given (alignment only ever turns the product).
        new_chain.nodes[0], new_chain.nodes[-1] = chain.nodes[0], chain.nodes[-1]
        self.chain_trajectory.append(new_chain)
        self.optimized = new_chain
        self._status(f"MLP-GI: done after {opt.steps} steps, {len(new_chain)} images", persistent=True)

        if not bool(self._params.get("do_elem_step_checks", True)):
            return IS_ELEM_STEP
        self._status("MLP-GI: elementary-step checks")
        results = check_if_elem_step(inp_chain=new_chain, engine=self.engine,
                                     **elem_step_check_kwargs(self._params))
        self.geom_grad_calls_made += int(results.number_grad_calls)
        return results
