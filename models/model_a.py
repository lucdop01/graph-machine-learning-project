"""Model A: MeshGraphNet (Pfaff et al. 2021), the advanced baseline without built-in physics (plan step 4).

Input: a GraphSample or a batch of them (graphs/dataset.py): x (T, phi, E normalised, boundary flag), edge_attr
(x_source - x_target and its length, normalised), edge_index. Output: one value per node, the normalised one-step
change in temperature (dT - mean) / std with the training statistics, which is what evaluation/torch_predictor.py
expects for target "dT".
"""
from torch import nn

from graphs.dataset import EDGE_FEATURES, NODE_FEATURES
from models.backbone import EncodeProcessDecode


class ModelA(nn.Module):
    target = "dT"

    def __init__(self, width=128, layers=15, hidden_layers=2, stats=None, stride=None, **kwargs):
        super().__init__()
        self.net = EncodeProcessDecode(len(NODE_FEATURES), len(EDGE_FEATURES), 1, width, layers, hidden_layers)

    def forward(self, g):
        return self.net(g.x, g.edge_attr, g.edge_index).squeeze(-1)
