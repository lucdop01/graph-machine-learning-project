"""Rollouts with a PyTorch Geometric model (plan step 3).

TorchPredictor wraps a model that maps a GraphSample (graphs/dataset.py SampleBuilder, the same features as in
training) to the normalised one-step change of its target, one value per node:
  target "dT"   T_next = T + dT                       (GCN, models A and B)
  target "dH"   T_next = T(H(T) + dH), material table  (models C and D; D computes dH from its edge fluxes inside
                                                        the model)
The change is de-normalised with the training statistics the model was trained with (stats["target"]). The statistics
must fit each run (stride, interpolation, frame interval); load them with graphs.dataset.load_stats, which also checks
them against the current training data, or take them from the model's checkpoint.
Each step builds the input from the predictor's own previous output, so no stored truth reaches the model.
"""
import numpy as np

from graphs.dataset import TARGETS, SampleBuilder, check_stats, check_stride, require_torch

try:
    import torch
except ModuleNotFoundError as e:                     # the error is raised in TorchPredictor (require_torch)
    if e.name != "torch":
        raise


class TorchPredictor:
    """Rollout predictor for a model that outputs the normalised dT or dH per node."""

    def __init__(self, model, stats, stride, target="dT", device="cpu", name=None):
        require_torch("TorchPredictor")
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, not {target!r}")
        self.stats, self.stride, self.target = stats, check_stride(stride), target
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.builder = SampleBuilder(stats, self.stride, self.device)
        self.mean, self.std = float(stats["target"][target]["mean"]), float(stats["target"][target]["std"])
        self.name = name or f"{type(model).__name__}_{target}"

    def reset(self, run):
        check_stats(self.stats, self.stride, [run], f"run {run.name}")

    def step(self, run, T, k, stride):
        if stride != self.stride:
            raise ValueError(f"{self.name} was built for stride {self.stride}, not {stride}")
        g = self.builder.sample(run, k, T)
        with torch.no_grad():
            out = self.model(g)
        z = out.detach().reshape(-1).to("cpu", torch.float64).numpy()
        if z.shape != (run.mesh.n_nodes,):
            raise ValueError(f"{self.name} returned {tuple(out.shape)}, expected one value per node "
                             f"({run.mesh.n_nodes})")
        if not np.isfinite(z).all():                 # checked here: the material table would clip an infinite dH
            raise FloatingPointError(f"{self.name} returned a non-finite output for the step from frame {k} of "
                                     f"{run.name}")
        delta = z * self.std + self.mean
        T = np.asarray(T, dtype=np.float64)
        if self.target == "dT":
            return T + delta
        return run.T_from_H(run.H_from_T(T) + delta)
