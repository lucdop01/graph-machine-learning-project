"""Node-based data, loss and rollout adapter for the grid U-Net baseline.

Only the U-Net input is rasterized. Targets and the residual temperature update
stay on the original mesh nodes. All temperature values are in K; laser energy
is in J/m^3. The five-frame stride and train statistics match the graph models.
"""

import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from graphs.dataset import check_stats, load_stats
from graphs.grid import get_grid_adapter
from models.unet import DEFAULT_GRID_DX, UNet

STRIDE = 5


class UNetInputBuilder:
    """Build the same three channels from either true or predicted node T."""

    def __init__(self, stats, grid_dx=DEFAULT_GRID_DX):
        if stats.get("stride") != STRIDE:
            raise ValueError(f"U-Net statistics must use stride {STRIDE}")
        self.stats, self.grid_dx = stats, float(grid_dx)
        self.T = stats["node"]["T"]
        self.E = stats["node"]["E"]
        self.dT = stats["target"]["dT"]
        for name, scale in (("T", self.T), ("E", self.E), ("dT", self.dT)):
            if not np.isfinite(scale["std"]) or scale["std"] <= 0:
                raise ValueError(f"{name} normalization needs a positive finite standard deviation")

    def adapter(self, run):
        return get_grid_adapter(run.mesh, self.grid_dx)

    def input_grid(self, run, temperature, k):
        """Return [3, y, x] from current T, current phi(T), and E[k:k+5]."""
        T = np.asarray(temperature, dtype=np.float64)
        if T.shape != (run.mesh.n_nodes,):
            raise ValueError(f"current temperature must have {run.mesh.n_nodes} nodes")
        E = run.laser(k, STRIDE)
        phi = run.phi(T)
        channels = np.stack(((T - self.T["mean"]) / self.T["std"],
                             phi,
                             (E - self.E["mean"]) / self.E["std"]))
        return torch.from_numpy(np.ascontiguousarray(self.adapter(run).nodes_to_grid(channels), dtype=np.float32))

    def target_nodes(self, temperature, next_temperature):
        """Normalized true dT on the original nodes; used only in supervised training."""
        delta = np.asarray(next_temperature, dtype=np.float64) - np.asarray(temperature, dtype=np.float64)
        return torch.from_numpy(np.ascontiguousarray((delta - self.dT["mean"]) / self.dT["std"],
                                                      dtype=np.float32))


class UNetSamples(Dataset):
    """Lazy train samples with the project's existing run split and start frames.

    A minibatch shares one mesh, so its node targets can be stacked. The current
    training split uses the straight seed-0 mesh; evaluation supports every mesh.
    """

    def __init__(self, runs, stats, grid_dx=DEFAULT_GRID_DX):
        if not runs or any(run.attrs["split"] != "train" for run in runs):
            raise ValueError("U-Net training samples must come from train runs only")
        check_stats(stats, STRIDE, runs, "U-Net train runs")
        fingerprints = {run.mesh.fingerprint() for run in runs}
        if len(fingerprints) != 1:
            raise ValueError("a U-Net minibatch needs one mesh; group training runs by mesh")
        self.runs = runs
        self.builder = UNetInputBuilder(stats, grid_dx)
        self.adapter = self.builder.adapter(runs[0])
        self.index = [(i, k) for i, run in enumerate(runs) for k in run.starts(STRIDE)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        run_index, k = self.index[item]
        run = self.runs[run_index]
        T = run.T(k)
        return (self.builder.input_grid(run, T, k),
                self.builder.target_nodes(T, run.T(k + STRIDE)))


def normalized_node_mse(predicted_grid, target_nodes, adapter):
    """MSE of normalized dT after differentiable grid -> original mesh nodes."""
    if predicted_grid.ndim != 4 or predicted_grid.shape[1] != 1:
        raise ValueError("U-Net prediction must have shape [batch, 1, y, x]")
    prediction = adapter.grid_to_nodes(predicted_grid)[:, 0]
    if prediction.shape != target_nodes.shape:
        raise ValueError(f"node targets have shape {tuple(target_nodes.shape)}, expected {tuple(prediction.shape)}")
    return F.mse_loss(prediction, target_nodes)


class UNetPredictor:
    """Shared evaluation interface: step(run, current node T, k, 5) -> next node T."""

    name = "unet"
    stride = STRIDE

    def __init__(self, model, stats, device="cpu"):
        if not isinstance(model, nn.Module) or not hasattr(model, "grid_dx"):
            raise TypeError("predictor needs a U-Net-like torch model with grid_dx")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.builder = UNetInputBuilder(stats, model.grid_dx)
        self.stats = stats

    def reset(self, run):
        check_stats(self.stats, STRIDE, [run], f"run {run.name}")
        self.builder.adapter(run)  # warm the geometry cache for this mesh

    def step(self, run, T, k, stride):
        if stride != STRIDE:
            raise ValueError(f"U-Net expects stride {STRIDE}, got {stride}")
        grid = self.builder.input_grid(run, T, k).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predicted_grid = self.model(grid)
            if predicted_grid.shape != (1, 1, *grid.shape[-2:]):
                raise ValueError(f"U-Net returned {tuple(predicted_grid.shape)}, expected one grid dT channel")
            normalized = self.builder.adapter(run).grid_to_nodes(predicted_grid)[0, 0]
        delta = (normalized.detach().cpu().numpy().astype(np.float64) * self.builder.dT["std"]
                 + self.builder.dT["mean"])
        if not np.isfinite(delta).all():
            raise FloatingPointError(f"U-Net returned a non-finite dT at frame {k} of {run.name}")
        return np.asarray(T, dtype=np.float64) + delta


def save_checkpoint(path, model, stats, epoch, train_loss, val_nrmse, seed):
    """Atomically save the best val-rollout checkpoint and train statistics."""
    if not isinstance(model, UNet):
        raise TypeError("checkpoint expects a UNet architecture")
    payload = dict(kind="UNet-dT", stride=STRIDE, seed=int(seed), epoch=int(epoch),
                   train_loss=float(train_loss), val_nrmse=float(val_nrmse),
                   train_fingerprint=stats["train_fingerprint"], stats=stats,
                   architecture=dict(depth=model.depth, channels=model.channels, grid_dx=model.grid_dx,
                                     norm=model.norm),
                   model_state=model.state_dict())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu"):
    """Load a trained U-Net, rejecting stale or incompatible train statistics."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "UNet-dT" or payload.get("stride") != STRIDE:
        raise ValueError("checkpoint is not a stride-5 U-Net dT fit")
    stats = load_stats(STRIDE)
    if payload["train_fingerprint"] != stats["train_fingerprint"] or payload["stats"] != stats:
        raise ValueError("U-Net checkpoint statistics differ from the current training data")
    model = UNet(**payload["architecture"])
    model.load_state_dict(payload["model_state"])
    return UNetPredictor(model, stats, device), payload
