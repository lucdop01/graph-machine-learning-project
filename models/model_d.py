"""Model D: Exact heat conservation via antisymmetric edge fluxes (plan step 4).

Architecture & physics:
  - Backbone: shared EncodeProcessDecode (MeshGraphNet), extracting node and edge latents after 15 blocks.
  - Edge decoder: predicts scalar energy flux f_e per directed edge in fp32.
  - Antisymmetry: F_ij = 0.5 * (f_ij - f_rev(ij)) using the dataset's rev_edge index, so F_ij = -F_ji bit for bit.
  - Conservation: for each node i, net energy change = V_i * E_i (laser) + sum of F into node i (conduction).
    Total mesh energy changes only by the deposited laser energy (sum of fluxes into the mesh is identically 0).
  - Target: output dH per node, normalized with training target statistics ((dH - mean) / std).
    TorchPredictor converts predicted dH back to temperature via the exact material enthalpy table.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from graphs.dataset import EDGE_FEATURES, NODE_FEATURES
from models.backbone import EncodeProcessDecode, mlp
from solver.ti64 import RHO


class ModelD(nn.Module):
    """Model D predicting enthalpy changes dH with exact internal energy conservation."""

    target = "dH"
    mean: torch.Tensor
    std: torch.Tensor
    rho: torch.Tensor

    def __init__(
        self,
        width: int = 128,
        layers: int = 15,
        hidden_layers: int = 2,
        mean: float = 0.0,
        std: float = 1.0,
        rho: float = RHO,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.hidden_layers = hidden_layers

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32))

        # Shared EncodeProcessDecode backbone
        self.net = EncodeProcessDecode(
            len(NODE_FEATURES), len(EDGE_FEATURES), 1, width, layers, hidden_layers
        )
        # Dedicated edge decoder mapping edge latents to scalar conduction fluxes
        self.edge_decoder = mlp(width, width, 1, hidden_layers=hidden_layers, layer_norm=False)
        last_layer = self.edge_decoder[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.zeros_(last_layer.weight)
            if last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)

    def set_stats(self, mean: float, std: float):
        """Set target normalization statistics from dataset norm_stride<s>.json."""
        self.mean.fill_(float(mean))
        self.std.fill_(float(std))

    def fluxes(self, g: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute raw edge fluxes and antisymmetric fluxes F_ij."""
        _, e = self.net.latents(g.x, g.edge_attr, g.edge_index)
        with torch.autocast(e.device.type, enabled=False):
            f = self.edge_decoder(e.to(self.edge_decoder[0].weight.dtype)).squeeze(-1)
        F = 0.5 * (f - f[g.rev_edge])
        return f, F

    def forward(self, g: Any) -> torch.Tensor:
        """Predict normalised change in specific enthalpy dH per node."""
        _, F = self.fluxes(g)

        # Sum incoming fluxes: dst is g.edge_index[1]
        dst = g.edge_index[1]
        flux_in = torch.zeros(g.x.size(0), dtype=torch.float32, device=g.x.device)
        flux_in.index_add_(0, dst, F)

        # Retrieve node volumes and laser energy
        V = getattr(g, "V", None)
        E = getattr(g, "E", None)
        if V is None:
            V = torch.full((g.x.size(0),), 1.0, dtype=torch.float32, device=g.x.device)
        else:
            V = V.to(torch.float32)

        if E is None:
            E = torch.zeros(g.x.size(0), dtype=torch.float32, device=g.x.device)
        else:
            E = E.to(torch.float32)

        # Total energy change per node = laser energy + incoming conduction fluxes
        # Specific enthalpy change: delta_H = (E * V + flux_in) / (rho * V)
        rho_V = torch.clamp(self.rho * V, min=1e-15)
        delta_H = (E * V + flux_in) / rho_V

        # Output normalised dH for training loss and TorchPredictor
        std = torch.clamp(self.std, min=1e-8)
        return (delta_H - self.mean) / std
