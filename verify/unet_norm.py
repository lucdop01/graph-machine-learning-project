"""Focused GroupNorm / pixel-wise LayerNorm checks: python -m verify.unet_norm."""

import numpy as np
import torch
from torch.nn import functional as F

from graphs.dataset import load_runs, load_stats
from models.unet import PixelwiseLayerNorm, UNet
from models.unet_pipeline import (STRIDE, UNetPredictor, UNetSamples,
                                  normalized_node_mse)


def hotspot_inputs(device="cpu"):
    """Same local input, aligned to the U-Net's pooling lattice, on two plates."""
    inputs = []
    for height, width, cy, cx in ((200, 400, 96, 192), (400, 800, 200, 400)):
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        r2 = ((y - cy) ** 2 + (x - cx) ** 2).float()
        spot = torch.exp(-r2 / (2 * 12.0 ** 2))
        # Normalized T and E plus a plausible bounded melted-fraction channel.
        inputs.append(torch.stack((0.1 + 3.0 * spot, torch.clamp((spot - 0.25) * 2, 0, 1),
                                   -0.05 + 2.0 * spot))[None].to(device))
    return inputs


@torch.no_grad()
def hotspot_difference(model, delta_std):
    model.eval()
    small, large = (model(x) for x in hotspot_inputs(next(model.parameters()).device))
    # Centers are separated by exact multiples of 2**depth. The 33x33 local
    # window is at least 80 pixels from every edge of the normal plate.
    local_small = small[0, 0, 80:113, 176:209]
    local_large = large[0, 0, 184:217, 384:417]
    difference = (local_small - local_large).abs()
    center = float(difference[16, 16].cpu())
    local_mean = float(difference.mean().cpu())
    local_max = float(difference.max().cpu())
    return dict(center_norm=center, local_mean_norm=local_mean, local_max_norm=local_max,
                center_K=center * delta_std, local_mean_K=local_mean * delta_std,
                local_max_K=local_max * delta_std)


@torch.no_grad()
def padding_difference(model, delta_std):
    """Effect of adding one replicated row/column to a 200x400 synthetic plate."""
    model.eval()
    base = hotspot_inputs(next(model.parameters()).device)[0]
    odd = F.pad(base, (0, 1, 0, 1), mode="replicate")
    difference = (model(base) - model(odd)[..., :200, :400]).abs()[0, 0]
    center = difference[80:113, 176:209]
    edge = torch.cat((difference[-16:, :].flatten(), difference[:, -16:].flatten()))
    return dict(center_mean_norm=float(center.mean().cpu()),
                center_max_norm=float(center.max().cpu()),
                edge_mean_norm=float(edge.mean().cpu()),
                edge_max_norm=float(edge.max().cpu()),
                center_mean_K=float(center.mean().cpu()) * delta_std,
                edge_mean_K=float(edge.mean().cpu()) * delta_std)


def check_variant(norm, stats, sample, adapter, run):
    torch.manual_seed(0)
    model = UNet(norm=norm)
    if norm == "pixelwise_layer":
        layers = [m for m in model.modules() if isinstance(m, PixelwiseLayerNorm)]
        assert len(layers) == 2 * (2 * model.depth + 1)
        x = torch.randn(2, 5, 7, 11)
        y = PixelwiseLayerNorm(5)(x)
        assert torch.allclose(y.mean(dim=1), torch.zeros_like(y[:, 0]), atol=1e-6)
        expected_var = x.var(dim=1, unbiased=False) / (x.var(dim=1, unbiased=False) + 1e-5)
        assert torch.allclose(y.var(dim=1, unbiased=False), expected_var, atol=1e-5)
        changed = x.clone()
        changed[:, :, 0, 0] += 10
        assert torch.equal(PixelwiseLayerNorm(5)(changed)[:, :, 1:, 1:], y[:, :, 1:, 1:])
    model.eval()
    initial = hotspot_difference(model, stats["target"]["dT"]["std"])
    grid, target = sample
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    small = UNet(depth=2, base_channels=8, norm=norm).to(device)
    x, target = grid[None].to(device), target[None].to(device)
    optimizer = torch.optim.Adam(small.parameters(), lr=3e-3)
    with torch.no_grad():
        loss_before = float(normalized_node_mse(small(x), target, adapter).cpu())
    for _ in range(80):
        small.train()
        optimizer.zero_grad(set_to_none=True)
        loss = normalized_node_mse(small(x), target, adapter)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())
        optimizer.step()
    small.eval()
    with torch.no_grad():
        loss_after = float(normalized_node_mse(small(x), target, adapter).cpu())
        odd = small(torch.randn(1, 3, 35, 53, device=device))
    assert odd.shape == (1, 1, 35, 53) and torch.isfinite(odd).all()
    assert loss_after < 0.5 * loss_before, (norm, loss_before, loss_after)
    predictor = UNetPredictor(small, stats, device)
    T = predictor.step(run, run.T(0), 0, STRIDE)
    assert T.shape == (run.mesh.n_nodes,) and np.isfinite(T).all()
    return dict(parameters=model.parameter_count, hotspot=initial,
                overfit_initial=loss_before, overfit_final=loss_after)


def main():
    torch.set_num_threads(1)
    stats = load_stats(STRIDE)
    train = load_runs("train", preload=False)
    run = max(train, key=lambda r: float(r.attrs["P"]) / float(r.attrs["v"]))
    samples = UNetSamples([run], stats)
    k = min(int(0.5 * run.attrs["laser_off"] / run.t[1]), run.last - STRIDE)
    sample = samples[samples.index.index((0, k))]
    results = {norm: check_variant(norm, stats, sample, samples.adapter, run)
               for norm in ("group", "pixelwise_layer")}
    assert results["group"]["parameters"] == results["pixelwise_layer"]["parameters"]
    assert results["pixelwise_layer"]["hotspot"]["local_max_norm"] < 1e-4
    for name, values in results.items():
        print(name, values)


if __name__ == "__main__":
    main()
