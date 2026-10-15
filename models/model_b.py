"""Fixed linear diffusion branch plus an injected learned correction.

The shared Model A can be passed as correction; it must return one normalised
dT residual per node. This module owns no training loop or new backbone.
"""
import math
import torch
from torch import nn
from solver.ti64 import K_CONST, CP_CONST, RHO


class LinearDiffusion(nn.Module):
    """Stable explicit substeps of the mass-lumped cotangent operator.

    A Gershgorin bound using absolute weights bounds the largest eigenvalue
    even on meshes with negative cotangents. The source is split uniformly
    across substeps. Constants fix a linear operator, with no learned physics.
    """
    def __init__(self, conductivity=K_CONST, heat_capacity=CP_CONST, max_substeps=10000):
        super().__init__()
        if not math.isfinite(conductivity) or not math.isfinite(heat_capacity) or min(conductivity, heat_capacity) <= 0:
            raise ValueError("material constants must be finite and positive")
        if max_substeps < 1:
            raise ValueError("max_substeps must be positive")
        self.alpha = conductivity / (RHO * heat_capacity)
        self.capacity = RHO * heat_capacity
        self.max_substeps = max_substeps

    def forward(self, graph):
        T = graph.T.reshape(-1)
        area = graph.area.reshape(-1).to(T)
        w = graph.cotangent_weight.to(T)
        dt = torch.as_tensor(graph.dt, dtype=T.dtype, device=T.device).reshape(-1)
        if dt.numel() > 1:
            dt = dt[graph.batch]
        if torch.any(area <= 0) or torch.any(dt <= 0):
            raise ValueError("node areas and step durations must be positive")
        src, dst = graph.edge_index
        rows = torch.zeros_like(T).index_add(0, src, w.abs())
        # lambda_max <= 2 max(sum |w| / area); safety factor 0.9.
        n = max(1, math.ceil(float((2 * self.alpha * dt * rows / area).max().detach()) / 0.9))
        if n > self.max_substeps:
            raise ValueError(f"diffusion needs {n} substeps; check mesh units and dt")
        source = graph.E.reshape(-1).to(T) / self.capacity / n
        state = T
        for _ in range(n):
            flux = torch.zeros_like(T).index_add(0, src, w * (state[dst] - state[src]))
            state = state + self.alpha * dt / n * flux / area + source
        return state - T


class ModelB(nn.Module):
    target = "dT"

    def __init__(self, correction=None, stats=None, width=128, layers=15, hidden_layers=2,
                 conductivity=K_CONST, heat_capacity=CP_CONST, max_substeps=10000):
        super().__init__()
        if stats is None:
            raise ValueError("ModelB needs the training statistics (or checkpoint stats)")
        if correction is None:
            from models.model_a import ModelA
            correction = ModelA(width=width, layers=layers, hidden_layers=hidden_layers)
        self.correction = correction
        self.diffusion = LinearDiffusion(conductivity, heat_capacity, max_substeps)
        target = stats["target"]["dT"]
        if not math.isfinite(target["std"]) or target["std"] <= 0:
            raise ValueError("dT target std must be finite and positive")
        self.register_buffer("target_mean", torch.tensor(float(target["mean"])))
        self.register_buffer("target_std", torch.tensor(float(target["std"])))

    def forward(self, graph):
        fixed = self.diffusion(graph)
        residual = self.correction(graph).reshape(-1)
        if residual.shape != fixed.shape:
            raise ValueError("correction must return one value per node")
        # Residual is in std units with zero physical correction at output 0.
        return (fixed - self.target_mean) / self.target_std + residual
