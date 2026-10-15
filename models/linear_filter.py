"""Fixed, source-driven linear temperature diffusion baseline for the rollout harness."""
import numpy as np

from graphs.laplacian import stiffness
from solver.ti64 import K_CONST, CP_CONST, RHO


class LinearFilter:
    """Explicit substeps with fixed k/cp, insulated edges and integrated laser input.

    This is a temperature baseline, not a phase-change solver. Use identical
    k/cp for Model B's fixed branch; week1 has precisely these properties.
    """
    name = "laplacian_filter"

    def __init__(self, conductivity=K_CONST, heat_capacity=CP_CONST):
        if not np.isfinite([conductivity, heat_capacity]).all() or min(conductivity, heat_capacity) <= 0:
            raise ValueError("material constants must be finite and positive")
        self.conductivity, self.heat_capacity = conductivity, heat_capacity
        self._cache = {}

    def step(self, run, T, k, stride):
        dt = float(run.t[k + stride] - run.t[k])
        key = (run.mesh.fingerprint(), dt)
        if key not in self._cache:
            area = run.mesh.area
            matrix = stiffness(run.mesh)
            off = matrix.copy()
            off.setdiag(0)
            bound = 2 * np.asarray(abs(off).sum(axis=1)).ravel() / area
            alpha = self.conductivity / (RHO * self.heat_capacity)
            n = max(1, int(np.ceil(dt * alpha * bound.max() / 0.9)))
            self._cache[key] = (matrix, n)
        matrix, n = self._cache[key]
        state = np.asarray(T, dtype=float).copy()
        source = run.laser(k, stride) / (RHO * self.heat_capacity) / n
        for _ in range(n):
            state += -dt / n * self.conductivity / (RHO * self.heat_capacity) * (matrix @ state) / run.mesh.area + source
        return state
