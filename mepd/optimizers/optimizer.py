from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np


def scale_step_imagewise(step, max_step_norm: float | None):
    """Scale each image displacement to max_step_norm RMS without changing direction."""
    if max_step_norm is None or max_step_norm <= 0.0:
        return step, None

    clipped = np.array(step, dtype=float, copy=True)
    flat = clipped.reshape(clipped.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1)
    norm_caps = max_step_norm * np.sqrt(flat.shape[1])
    large = norms > norm_caps
    if np.any(large):
        scale = (norm_caps / (norms[large] + 1e-16)).reshape(-1, 1)
        flat[large] = flat[large] * scale
        clipped = flat.reshape(clipped.shape)
    return clipped, large


def clip_step_atomwise(step, max_step_norm: float | None):
    """Compatibility wrapper for the image-wise step scaler."""
    return scale_step_imagewise(step, max_step_norm)


@dataclass
class Optimizer(ABC):
    @abstractmethod
    def optimize_step(self):
        ...
