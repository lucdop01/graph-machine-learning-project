"""GCN baseline: Graph Convolutional Network (Kipf & Welling 2017) for melt-pool simulation.

Features & target:
  - Input: node features x (T, phi, E normalised, boundary flag), edge_index
  - Target: normalised one-step change in temperature (dT - mean) / std (same as Model A)
  - Geometry: standard GCN ignores edge features (no metric geometry / anisotropic directions).
    Optionally supports edge_weight (inverse edge lengths) for edge-weighted GCN ablation.

Architecture:
  - Node encoder: MLP(4 -> width) with LayerNorm
  - Processor: stack of GCN blocks with residual connections and LayerNorm32
  - Node decoder: MLP(width -> 1) producing normalised dT
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GCNConv

from graphs.dataset import NODE_FEATURES
from models.backbone import LayerNorm32, mlp


class GCNBlock(nn.Module):
    """One GCN message-passing block with residual connection and LayerNorm."""

    def __init__(self, width: int):
        super().__init__()
        self.conv = GCNConv(width, width, add_self_loops=True, normalize=True)
        self.act = nn.ReLU()
        self.norm = LayerNorm32(width)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_conv = self.act(self.conv(h, edge_index, edge_weight=edge_weight))
        return self.norm(h + h_conv)


class GCNModel(nn.Module):
    """GCN baseline predicting one-step normalised temperature change dT per node."""

    target = "dT"

    def __init__(
        self,
        width: int = 128,
        layers: int = 15,
        hidden_layers: int = 2,
        use_edge_weight: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.use_edge_weight = use_edge_weight

        self.node_encoder = mlp(
            len(NODE_FEATURES), width, width, hidden_layers=hidden_layers, layer_norm=True
        )
        self.blocks = nn.ModuleList([GCNBlock(width) for _ in range(layers)])
        self.node_decoder = mlp(
            width, width, 1, hidden_layers=hidden_layers, layer_norm=False
        )

    def forward(self, g: Any) -> torch.Tensor:
        edge_weight = None
        if self.use_edge_weight and hasattr(g, "edge_attr") and g.edge_attr is not None:
            # edge_attr[:, 2] is edge length (normalised); convert to inverse distance weights
            lengths = torch.clamp(g.edge_attr[:, 2].abs(), min=1e-3)
            edge_weight = 1.0 / lengths

        h = self.node_encoder(g.x)
        for block in self.blocks:
            h = block(h, g.edge_index, edge_weight=edge_weight)

        return self.node_decoder(h).squeeze(-1)
