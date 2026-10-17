"""U-Net integration checks and a one-example overfit smoke test.

Run: python -m verify.unet_pipeline
Reads the existing train and test preprocessing; does not save a checkpoint.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from evaluation.rollout import evaluate_split, load_size_bins, rollout
from graphs.dataset import RunGraph, load_runs, load_stats
from graphs.mesh import get_mesh
from graphs.preprocess import load_run
from models.train_unet import train_one_epoch
from models.unet import UNet
from models.unet_pipeline import (STRIDE, UNetInputBuilder, UNetPredictor,
                                  UNetSamples, load_checkpoint, normalized_node_mse,
                                  save_checkpoint)


class ConstantDelta(torch.nn.Module):
    """A predictable grid output to audit the residual update and rollout."""

    grid_dx = 5e-6

    def __init__(self, normalized_delta):
        super().__init__()
        self.normalized_delta = float(normalized_delta)

    def forward(self, x):
        return torch.full((len(x), 1, *x.shape[-2:]), self.normalized_delta,
                          device=x.device, dtype=x.dtype)


class GuardedRun:
    """Permit only the initial true T; record model-state phi and laser calls."""

    def __init__(self, source):
        self.source = source
        self.phi_states, self.laser_calls = [], []

    def __getattr__(self, name):
        return getattr(self.source, name)

    def T(self, k):
        if k != 0:
            raise AssertionError(f"rollout read future true temperature at frame {k}")
        return self.source.T(0)

    def phi(self, temperature):
        self.phi_states.append(np.asarray(temperature).copy())
        return self.source.phi(temperature)

    def laser(self, k, stride):
        self.laser_calls.append((k, stride))
        return self.source.laser(k, stride)


def check_data_and_rollout(run, stats):
    builder = UNetInputBuilder(stats)
    k = min(int(0.5 * run.attrs["laser_off"] / run.t[1]), run.last - STRIDE)
    samples = UNetSamples([run], stats)
    grid, target = samples[samples.index.index((0, k))]
    T = run.T(k)
    adapter = builder.adapter(run)
    expected = np.stack(((T - stats["node"]["T"]["mean"]) / stats["node"]["T"]["std"],
                         run.phi(T),
                         (run.laser(k, STRIDE) - stats["node"]["E"]["mean"]) / stats["node"]["E"]["std"]))
    np.testing.assert_allclose(grid.numpy(), adapter.nodes_to_grid(expected), rtol=2e-6, atol=2e-6)
    expected_target = ((run.T(k + STRIDE) - T - stats["target"]["dT"]["mean"])
                       / stats["target"]["dT"]["std"])
    np.testing.assert_allclose(target.numpy(), expected_target, rtol=1e-6, atol=1e-6)
    assert grid.shape == (3, *adapter.shape) and target.shape == (run.mesh.n_nodes,)

    mean, std = stats["target"]["dT"]["mean"], stats["target"]["dT"]["std"]
    predictor = UNetPredictor(ConstantDelta((1.0 - mean) / std), stats)
    hot = 300.0 + 1700.0 * np.exp(-np.sum((run.mesh.pos - [run.mesh.Lx / 2, run.mesh.Ly / 2]) ** 2,
                                           axis=1) / (2 * (45e-6) ** 2))
    np.testing.assert_allclose(predictor.step(run, hot, 0, STRIDE), hot + 1, rtol=0, atol=2e-5)
    guarded = GuardedRun(run)
    predicted = rollout(guarded, predictor, STRIDE)
    starts = list(range(0, run.last // STRIDE * STRIDE, STRIDE))
    assert guarded.laser_calls == [(start, STRIDE) for start in starts]
    assert len(guarded.phi_states) == len(starts)
    for step, temperature in enumerate(guarded.phi_states):
        np.testing.assert_allclose(temperature, run.T(0) + step, rtol=0, atol=2e-4)
    np.testing.assert_allclose(predicted["T"][-1], run.T(0) + len(starts), rtol=0, atol=2e-4)
    return samples, grid, target


def check_overfit(samples, grid, target, stats):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(depth=2, base_channels=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    one = [(grid.unsqueeze(0), target.unsqueeze(0))]
    with torch.no_grad():
        initial = float(normalized_node_mse(model(one[0][0].to(device)), one[0][1].to(device),
                                            samples.adapter).cpu())
    for _ in range(40):
        train_one_epoch(model, one, samples.adapter, optimizer, device)
    with torch.no_grad():
        final = float(normalized_node_mse(model(one[0][0].to(device)), one[0][1].to(device),
                                          samples.adapter).cpu())
    if not final < 0.5 * initial:
        raise AssertionError(f"one-example overfit did not reduce loss enough: {initial:.4g} -> {final:.4g}")
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "unet.pt")
        save_checkpoint(path, model, stats, epoch=40, train_loss=final,
                        val_nrmse=0.5, seed=0)
        loaded, payload = load_checkpoint(path, device)
        assert payload["epoch"] == 40
        with torch.no_grad():
            restored = normalized_node_mse(loaded.model(one[0][0].to(device)), one[0][1].to(device),
                                           samples.adapter)
        np.testing.assert_allclose(float(restored.cpu()), final, rtol=1e-5)
    return initial, final, str(device)


def check_shared_evaluation(stats):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    predictor = UNetPredictor(UNet(depth=2, base_channels=8), stats, device)
    val = load_runs("val", preload=False)[0]
    rows, summary = evaluate_split([val], predictor, STRIDE, load_size_bins())
    assert len(rows) == summary["runs"] == 1
    assert all(key in summary for key in ("nrmse_mean", "iou", "energy", "solidification"))
    for split in ("test_path", "test_fine", "test_large"):
        run = load_runs(split, preload=False)[0]
        predicted = predictor.step(run, run.T(0), 0, STRIDE)
        assert predicted.shape == (run.mesh.n_nodes,) and np.isfinite(predicted).all()
    # The full seed-1 processed split is optional locally. Build just one run
    # in memory from its existing raw solver file to check unseen mesh geometry.
    with open("data/splits.json") as file:
        row = next(row for row in json.load(file) if row["split"] == "test_interp")
    run = RunGraph.from_simulation(load_run(row["file"]), get_mesh("straight", 1))
    predicted = predictor.step(run, run.T(0), 0, STRIDE)
    assert predicted.shape == (run.mesh.n_nodes,) and np.isfinite(predicted).all()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    stats = load_stats(STRIDE)
    train = load_runs("train", preload=False)
    run = max(train, key=lambda r: float(r.attrs["P"]) / float(r.attrs["v"]))
    samples, grid, target = check_data_and_rollout(run, stats)
    initial, final, device = check_overfit(samples, grid, target, stats)
    check_shared_evaluation(stats)
    print(f"U-Net pipeline: {run.name}, stride {STRIDE}, grid {samples.adapter.shape}; "
          f"one-example node MSE {initial:.5g} -> {final:.5g} in 40 steps on {device}")
    print("normalization, laser alignment, residual update, no-future rollout, gradients, "
          "checkpoint and shared evaluation pass")


if __name__ == "__main__":
    main()
