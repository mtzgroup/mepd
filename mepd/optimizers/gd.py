from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mepd.optimizers.optimizer import Optimizer, scale_step_imagewise


@dataclass
class DeterministicGradientDescentOptimizer(Optimizer):
    """Deterministic full-gradient descent optimizer for path updates."""

    timestep: float = 0.05
    weight_decay: float = 0.0
    max_step_norm: float | None = None
    auto_timestep: bool = False
    step_up: float = 1.10
    step_down: float = 0.6
    min_timestep: float | None = None
    max_timestep: float | None = None
    corr_increase_thre: float = 0.8
    corr_decrease_thre: float = -0.1
    plateau_window: int = 3
    plateau_rtol: float = 0.05
    plateau_growth_lag: int = 1

    def __post_init__(self):
        self.orig_timestep = float(self.timestep)
        if self.min_timestep is None:
            self.min_timestep = max(1e-6, 1e-3 * self.orig_timestep)
        if self.max_timestep is None:
            self.max_timestep = max(self.orig_timestep, 4.0 * self.orig_timestep)
        if self.max_timestep < self.min_timestep:
            self.min_timestep, self.max_timestep = self.max_timestep, self.min_timestep
        self._prev_grad_flat: np.ndarray | None = None
        self._grad_norm_history: list[float] = []
        self._n_plateau_hits: int = 0

    def reset(self):
        self.timestep = float(self.orig_timestep)
        self._prev_grad_flat = None
        self._grad_norm_history = []
        self._n_plateau_hits = 0

    def _metrics_plateauing(self, grad_norm: float) -> bool:
        window = max(0, int(self.plateau_window))
        self._grad_norm_history.append(float(grad_norm))
        keep = window + 1
        if len(self._grad_norm_history) > keep:
            self._grad_norm_history = self._grad_norm_history[-keep:]
        if window == 0:
            return True
        if len(self._grad_norm_history) < keep:
            return False
        start = float(self._grad_norm_history[0])
        end = float(self._grad_norm_history[-1])
        scale = max(abs(start), 1e-12)
        rel_improvement = (start - end) / scale
        return rel_improvement <= float(self.plateau_rtol)

    def _adapt_timestep(self, grads: np.ndarray) -> None:
        g_flat = grads.flatten()
        g_norm = float(np.linalg.norm(g_flat))
        plateauing = self._metrics_plateauing(g_norm)

        if self._prev_grad_flat is not None:
            denom = float(np.linalg.norm(self._prev_grad_flat) * np.linalg.norm(g_flat))
            corr = 0.0
            if denom > 1e-16:
                corr = float(np.dot(self._prev_grad_flat, g_flat) / denom)

            if corr < float(self.corr_decrease_thre):
                self.timestep *= float(self.step_down)
                self._n_plateau_hits = 0
            elif corr > float(self.corr_increase_thre) and plateauing:
                self._n_plateau_hits += 1
                if self._n_plateau_hits >= max(1, int(self.plateau_growth_lag)):
                    self.timestep *= float(self.step_up)
                    self._n_plateau_hits = 0
            else:
                self._n_plateau_hits = 0

        self.timestep = max(float(self.min_timestep), min(float(self.max_timestep), float(self.timestep)))
        self._prev_grad_flat = g_flat.copy()

    def optimize_step(self, chain, chain_gradients):
        grads = np.array(chain_gradients, dtype=float, copy=True)
        converged_mask = np.array([node.converged for node in chain.nodes], dtype=bool)
        if converged_mask.any():
            grads[converged_mask] = 0.0

        if self.weight_decay != 0.0:
            grads = grads + self.weight_decay * chain.coordinates

        if bool(self.auto_timestep):
            self._adapt_timestep(grads=grads)

        step = self.timestep * grads

        step, _ = scale_step_imagewise(step, self.max_step_norm)

        new_chain_coordinates = chain.coordinates - step
        new_nodes = [node.update_coords(new_coords) for node, new_coords in zip(chain.nodes, new_chain_coordinates)]
        return chain.model_copy(update={"nodes": new_nodes, "parameters": chain.parameters})
