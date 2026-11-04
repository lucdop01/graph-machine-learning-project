"""Grid U-Net architecture for one-step temperature-change regression.

Input channels are normalized node-derived T, melted fraction and accumulated
laser energy, rasterized onto a physical grid. Output is one normalized dT
channel. Normalization is applied to hidden features, never raw input channels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_GRID_DX = 5e-6  # metres; passed to graphs.grid.get_grid_adapter by the future pipeline


def _groups(channels):
    """Use multiple channels per GroupNorm group, also with batch size one."""
    return max(g for g in range(1, min(8, channels) + 1)
               if channels % g == 0 and channels // g >= 2)


class PixelwiseLayerNorm(nn.Module):
    """LayerNorm over channels independently at every (batch, y, x) location."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _normalization(kind, channels):
    if kind == "group":
        return nn.GroupNorm(_groups(channels), channels)
    if kind == "pixelwise_layer":
        return PixelwiseLayerNorm(channels)
    raise ValueError("norm must be 'group' or 'pixelwise_layer'")


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with edge-value padding, hidden normalization and ReLU."""

    def __init__(self, in_channels, out_channels, norm="group"):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="replicate"),
            _normalization(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="replicate"),
            _normalization(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """Standard encoder-decoder U-Net, [B, 3, H, W] -> [B, 1, H, W].

    depth counts 2x downsampling stages. If channels is omitted, widths double
    from base_channels; otherwise provide depth + 1 widths including the
    bottleneck. grid_dx records the physical spacing used by the pipeline.
    """

    def __init__(self, depth=3, base_channels=16, channels=None, grid_dx=DEFAULT_GRID_DX,
                 norm="group"):
        super().__init__()
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        if isinstance(base_channels, bool) or not isinstance(base_channels, int) or base_channels < 2:
            raise ValueError("base_channels must be an integer of at least 2")
        if not math.isfinite(grid_dx) or grid_dx <= 0:
            raise ValueError("grid_dx must be finite and positive")
        if norm not in ("group", "pixelwise_layer"):
            raise ValueError("norm must be 'group' or 'pixelwise_layer'")
        widths = tuple(base_channels * 2 ** i for i in range(depth + 1)) if channels is None else tuple(channels)
        if len(widths) != depth + 1 or any(isinstance(c, bool) or not isinstance(c, int) or c < 2 for c in widths):
            raise ValueError("channels must contain depth + 1 integer widths of at least 2")

        self.depth, self.channels, self.grid_dx, self.norm = depth, widths, float(grid_dx), norm
        self.encoder = nn.ModuleList()
        previous = 3
        for width in widths[:-1]:
            self.encoder.append(DoubleConv(previous, width, norm))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(previous, widths[-1], norm)
        self.decoder = nn.ModuleList(DoubleConv(widths[i + 1] + widths[i], widths[i], norm)
                                     for i in reversed(range(depth)))
        self.head = nn.Conv2d(widths[0], 1, kernel_size=1)  # unrestricted signed dT

    @property
    def deepest_receptive_field_pixels(self):
        """The receptive-field width at the bottleneck, before the decoder."""
        field, jump = 1, 1
        for _ in range(self.depth):
            field += 4 * jump  # two 3x3 convolutions
            field += jump      # 2x2 max pool
            jump *= 2
        return field + 4 * jump  # bottleneck's two 3x3 convolutions

    @property
    def deepest_receptive_field_m(self):
        return self.deepest_receptive_field_pixels * self.grid_dx

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError("U-Net expects [batch, 3, height, width] with positive spatial dimensions")
        height, width = x.shape[-2:]
        factor = 2 ** self.depth
        pad_height, pad_width = (-height) % factor, (-width) % factor
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")

        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((skip, x), dim=1))
        return self.head(x)[..., :height, :width]
