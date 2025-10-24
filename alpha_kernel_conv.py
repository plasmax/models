"""Lightweight convolution module using alpha channel as position-dependent kernels.

This module expects an RGBA tensor with shape ``(batch, 4, height, width)``.
The first three channels are treated as RGB values. The alpha channel encodes
spatially-varying kernels arranged as a grid. For every output location, the
module samples a ``kernel_size`` patch from the alpha channel which is used as a
per-location convolution kernel applied uniformly to each color channel.

Example
-------
>>> module = AlphaKernelConv(kernel_size=3, padding=1)
>>> rgba = torch.randn(2, 4, 32, 32)
>>> output = module(rgba)
>>> output.shape
torch.Size([2, 3, 32, 32])
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class AlphaKernelConv(nn.Module):
    """Convolve RGB channels using kernels sampled from the alpha channel.

    Parameters
    ----------
    kernel_size:
        Size of the spatial window sampled from the alpha channel. Can be an
        ``int`` or a tuple ``(height, width)``.
    stride:
        Sampling stride used for both the RGB input and the alpha kernels.
    padding:
        Zero-padding applied prior to sampling. Padding is applied equally to the
        RGB channels and the alpha channel so that the kernels remain aligned.
    """

    def __init__(self, kernel_size: int | Tuple[int, int] = 3,
                 stride: int | Tuple[int, int] = 1,
                 padding: int | Tuple[int, int] = 0) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 4:
            raise ValueError(
                "Expected input of shape (batch, 4, height, width) representing an RGBA tensor."
            )

        rgb = x[:, :3]
        alpha = x[:, 3:4]

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        rgb_unfold = F.unfold(rgb, kernel_size=self.kernel_size,
                              dilation=1, padding=self.padding, stride=self.stride)
        alpha_unfold = F.unfold(alpha, kernel_size=self.kernel_size,
                                dilation=1, padding=self.padding, stride=self.stride)

        if alpha_unfold.size(-1) != rgb_unfold.size(-1):
            raise RuntimeError("Alpha channel sampling does not align with RGB sampling.")

        # Repeat alpha kernels to match the three color channels and weigh RGB patches.
        weights = alpha_unfold.repeat_interleave(3, dim=1)
        weighted_rgb = rgb_unfold * weights

        batch_size = x.size(0)
        spatial_locs = weighted_rgb.size(-1)

        # Sum over the kernel dimension while preserving color channels.
        weighted_rgb = weighted_rgb.view(batch_size, 3, k_h * k_w, spatial_locs)
        aggregated = weighted_rgb.sum(dim=2)

        h_in, w_in = x.shape[-2:]
        h_out = (h_in + 2 * p_h - k_h) // s_h + 1
        w_out = (w_in + 2 * p_w - k_w) // s_w + 1

        return aggregated.view(batch_size, 3, h_out, w_out)
