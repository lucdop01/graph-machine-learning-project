"""Model C: model A with an enthalpy output and the exact material table (plan step 4).

The same backbone, inputs and size as model A (models/model_a.py); only the target changes. The output is the
normalised one-step change in enthalpy (dH - mean) / std with the training statistics, which is what
evaluation/torch_predictor.py expects for target "dH": T_next = T(H(T) + dH) with the run's material table. Latent heat
is then exact: across the melting range (1,878 to 1,928 K) a smooth dH gives the temperature plateau, which model A has
to learn as a sharp drop in dT/dH.
"""
from torch import nn

from graphs.dataset import EDGE_FEATURES, NODE_FEATURES
from models.backbone import EncodeProcessDecode


class ModelC(nn.Module):
    target = "dH"

    def __init__(self, width=128, layers=15, hidden_layers=2):
        super().__init__()
        self.net = EncodeProcessDecode(len(NODE_FEATURES), len(EDGE_FEATURES), 1, width, layers, hidden_layers)

    def forward(self, g):
        return self.net(g.x, g.edge_attr, g.edge_index).squeeze(-1)
