"""Encode-process-decode backbone (MeshGraphNets, Pfaff et al. 2021), shared by models A, B, C and D (plan step 4).

  encoder    a node MLP on the node features x and an edge MLP on the edge features edge_attr, both to the latent
             width (128)
  processor  a stack of blocks (15). Each block first updates every edge from its own latent and the latents of its
             two nodes, then every node from its own latent and the sum of its incoming (updated) edge latents. Both
             updates are residual:
               e'_ij = e_ij + phi_e(e_ij, h_i, h_j)          i = edge_index[0] (source), j = edge_index[1] (target)
               h'_j  = h_j + phi_v(h_j, sum over edges ij into j of e'_ij)
  decoder    a node MLP from the latent to the outputs (no LayerNorm)
Every MLP has two hidden layers of the latent width with ReLU, and LayerNorm on its output except the decoder. At
width 128 and 15 blocks with 4 node and 3 edge features and one output this has 2,331,905 parameters.
The nodes sum the updated edge latents e', as in the proposal's equations. DeepMind's MeshGraphNets code sums phi_e
before adding the edge residual; the two differ only in that detail.

The sum over incoming edges is a scatter-sum (index_add_). On a GPU it uses atomic additions, so results are not
bit-reproducible from run to run; on a CPU they are.

The first layer of the edge MLP takes the concatenation [e_ij, h_i, h_j], so W [e; h_i; h_j] = W_e e_ij + W_s h_i +
W_t h_j with W = [W_e, W_s, W_t]. W_s h and W_t h are computed once per node and then gathered for every edge, instead
of once per edge (each node has about 6 edges). Same function, same parameters (the slices of one nn.Linear, so
checkpoints are unchanged), about a third less matrix work in the edge MLPs, and no 3 x width input stored per edge.

Mixed precision (training with precision bf16, models/train.py): under torch.autocast the hidden matrix products of the
encoders and the processor run in bf16, and so do the three terms of the split first edge layer and their sum. The
encoders' first layers (on the raw features, so the input temperature is not rounded to bf16; about 3 K steps at 1,000 K
and 6 K near the solidus), LayerNorm, so every MLP output, the residual streams h and e, the sums over incoming edges and
the decoder stay in fp32 (autocast switched off). Without autocast (validation, evaluation) everything is fp32.
A model that adds its own decoder (model D's edge fluxes) must switch autocast off there as well.
"""
import itertools

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm32(nn.LayerNorm):
    """LayerNorm computed in the precision of its parameters (fp32), also under autocast. CUDA autocast already does
    this; CPU autocast would not. The parameters and state_dict are those of nn.LayerNorm."""

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return super().forward(x.to(self.weight.dtype))


def mlp(n_in, width, n_out, hidden_layers=2, layer_norm=True):
    """hidden_layers hidden layers of size width with ReLU, a linear output layer, and LayerNorm on the output."""
    layers, d = [], n_in
    for _ in range(hidden_layers):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, n_out))
    if layer_norm:
        layers.append(LayerNorm32(n_out))
    return nn.Sequential(*layers)


class ProcessorBlock(nn.Module):
    """One message-passing block: edge update, then node update, both residual."""

    def __init__(self, width=128, hidden_layers=2):
        super().__init__()
        self.edge_mlp = mlp(3 * width, width, width, hidden_layers)
        self.node_mlp = mlp(2 * width, width, width, hidden_layers)

    def forward(self, h, e, edge_index):
        src, dst = edge_index[0], edge_index[1]
        first = self.edge_mlp[0]
        d = first.in_features // 3
        w_e, w_s, w_t = first.weight[:, :d], first.weight[:, d:2 * d], first.weight[:, 2 * d:]
        z = F.linear(e, w_e, first.bias) + F.linear(h, w_s)[src] + F.linear(h, w_t)[dst]   # = first([e, h_i, h_j])
        for layer in itertools.islice(self.edge_mlp, 1, None):
            z = layer(z)
        e = e + z
        incoming = torch.zeros_like(h).index_add_(0, dst, e)            # sum of the edge latents at each target
        h = h + self.node_mlp(torch.cat([h, incoming], dim=1))
        return h, e


class EncodeProcessDecode(nn.Module):
    """latents() gives the node and edge latents after the processor (model D decodes the edges itself);
    forward() decodes the nodes, shape (nodes, n_out)."""

    def __init__(self, node_in, edge_in, n_out, width=128, layers=15, hidden_layers=2):
        super().__init__()
        self.node_encoder = mlp(node_in, width, width, hidden_layers)
        self.edge_encoder = mlp(edge_in, width, width, hidden_layers)
        self.blocks = nn.ModuleList(ProcessorBlock(width, hidden_layers) for _ in range(layers))
        self.decoder = mlp(width, width, n_out, hidden_layers, layer_norm=False)

    def latents(self, x, edge_attr, edge_index):
        h, e = self.node_encoder[0], self.edge_encoder[0]
        with torch.autocast(x.device.type, enabled=False):              # the raw features in fp32
            h, e = h(x.to(h.weight.dtype)), e(edge_attr.to(e.weight.dtype))
        for layer in itertools.islice(self.node_encoder, 1, None):
            h = layer(h)
        for layer in itertools.islice(self.edge_encoder, 1, None):
            e = layer(e)
        for block in self.blocks:
            h, e = block(h, e, edge_index)
        return h, e

    def forward(self, x, edge_attr, edge_index):
        h = self.latents(x, edge_attr, edge_index)[0]
        with torch.autocast(h.device.type, enabled=False):             # the decoder in fp32, also in training
            return self.decoder(h.to(self.decoder[0].weight.dtype))
