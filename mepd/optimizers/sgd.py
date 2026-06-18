from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mepd.optimizers.optimizer import Optimizer, scale_step_imagewise


@dataclass
class SGDOptimizer(Optimizer):
    """Stochastic gradient descent optimizer for path updates."""

    timestep: float = 0.05
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    max_step_norm: float | None = None

    # Removed weight_decay: physically invalid for absolute Cartesian/internal coordinates.

    def __post_init__(self):
        if self.nesterov and self.momentum <= 0.0:
            raise ValueError("Nesterov momentum requires momentum > 0.")
        if self.nesterov and self.dampening != 0.0:
            raise ValueError(
                "Nesterov momentum is incompatible with dampening.")
        self.velocity = None

    def reset(self):
        self.velocity = None

    def optimize_step(self, chain, chain_gradients):
        grads = np.array(chain_gradients, dtype=float, copy=True)
        converged_mask = np.array(
            [node.converged for node in chain.nodes], dtype=bool)

        # 1. Mask gradients for converged nodes
        if converged_mask.any():
            grads[converged_mask] = 0.0

        if self.momentum != 0.0:
            if self.velocity is None or self.velocity.shape != grads.shape:
                self.velocity = np.zeros_like(grads, dtype=float)

            # 2. Mask velocity for converged nodes to prevent ghost drifting
            if converged_mask.any():
                self.velocity[converged_mask] = 0.0

            self.velocity = self.momentum * self.velocity + \
                (1.0 - self.dampening) * grads

            if self.nesterov:
                effective_grad = grads + self.momentum * self.velocity
            else:
                effective_grad = self.velocity
        else:
            effective_grad = grads

        step = self.timestep * effective_grad

        # 3. Step scaling with velocity rescaling (prevents wind-up)
        step, large_steps = scale_step_imagewise(step, self.max_step_norm)
        if self.velocity is not None and large_steps is not None and np.any(large_steps):
            self.velocity, _ = scale_step_imagewise(
                self.velocity,
                self.max_step_norm / max(abs(float(self.timestep)), 1e-16),
            )

        # 4. Final safety catch: ensure step is strictly 0 for converged nodes
        if converged_mask.any():
            step[converged_mask] = 0.0

        new_chain_coordinates = chain.coordinates - step

        # 5. Optional optimization: only update nodes that actually moved
        new_nodes = []
        for i, (node, new_coords) in enumerate(zip(chain.nodes, new_chain_coordinates)):
            if converged_mask[i]:
                new_nodes.append(node)
            else:
                new_nodes.append(node.update_coords(new_coords))

        return chain.model_copy(update={"nodes": new_nodes, "parameters": chain.parameters})
