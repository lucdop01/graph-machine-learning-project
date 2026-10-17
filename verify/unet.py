"""Focused architecture checks: python -m verify.unet."""

import numpy as np
import torch

from models.unet import DEFAULT_GRID_DX, UNet
from solver.ti64 import R_SPOT


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)

    model = UNet().eval()
    assert model.channels == (16, 32, 64, 128)
    assert model.grid_dx == DEFAULT_GRID_DX == 5e-6
    assert model.deepest_receptive_field_pixels == 68
    assert np.isclose(model.deepest_receptive_field_m, 340e-6)
    assert model.deepest_receptive_field_m > 2 * R_SPOT
    with torch.no_grad():
        output = model(torch.randn(1, 3, 200, 400))  # normal plate, 5 um cells
        large = model(torch.randn(1, 3, 400, 800))  # test_large physical plate
    assert output.shape == (1, 1, 200, 400)
    assert large.shape == (1, 1, 400, 800)
    assert torch.isfinite(output).all()
    assert torch.isfinite(large).all()

    # Independent depth and widths, batch > 1, non-divisible dimensions and
    # end-to-end backpropagation through every encoder and decoder block.
    small = UNet(depth=2, channels=(8, 12, 24), grid_dx=2.5e-6)
    x = torch.randn(2, 3, 35, 53, requires_grad=True)
    y = small(x)
    assert y.shape == (2, 1, 35, 53)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())

    # The regression head is signed; there is no final sigmoid or ReLU.
    with torch.no_grad():
        small.head.weight.zero_()
        small.head.bias.fill_(-1)
        assert torch.allclose(small(torch.zeros(1, 3, 35, 53)), torch.full((1, 1, 35, 53), -1.0))

    print(f"U-Net default: {model.parameter_count:,} parameters, "
          f"bottleneck receptive field {model.deepest_receptive_field_pixels} pixels "
          f"({model.deepest_receptive_field_m * 1e6:.0f} um at 5 um/pixel)")
    print("U-Net: output dimensions, padding, forward pass, signed output and gradients pass")


if __name__ == "__main__":
    main()
