import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DepthRefinementModule(nn.Module):
    """
    TorchScript-friendly module that performs lightweight test-time optimization.

    The input is expected to be shaped ``(1, 3, H, W)`` with channel order:
    0. predicted dense depth
    1. sparse ground-truth depth
    2. validity mask for the sparse depth

    During ``forward`` a small number of gradient-descent steps are run to
    fit a per-frame scale and bias that align the predicted depth to the sparse
    observations. This keeps the module compatible with TorchScript while
    performing the optimization directly inside ``forward`` (useful when running
    inside a Nuke Inference node where parameters need to adapt on the fly).
    """

    def __init__(self, lr: float = 0.1, optim_steps: int = 4, smooth_factor: float = 0.3):
        super().__init__()
        self.lr = lr
        self.optim_steps = optim_steps
        self.smooth_factor = smooth_factor

        # trainable template parameters that will be cloned and optimized at test time
        self.init_scale = nn.Parameter(torch.tensor(1.0))
        self.init_bias = nn.Parameter(torch.tensor(0.0))

    def _split_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted = x[:, 0:1, :, :]
        sparse = x[:, 1:2, :, :]
        mask = x[:, 2:3, :, :]
        return predicted, sparse, mask

    def _optimize_affine(self, predicted: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a few gradient-descent steps on scale and bias inside forward."""
        device = predicted.device
        dtype = predicted.dtype

        scale = self.init_scale.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
        bias = self.init_bias.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)

        # keep the loop TorchScript friendly by using range() over a statically typed int attribute
        for _ in range(self.optim_steps):
            refined = scale * predicted + bias
            residual = (refined - sparse) * mask
            # Avoid division by zero when the mask is empty.
            loss = residual.abs().sum() / (mask.sum() + 1e-6)
            # torch.autograd.grad expects a sequence of outputs; wrapping ``loss``
            # keeps TorchScript happy while still computing gradients for the two
            # scalar parameters.
            grads = torch.autograd.grad((loss,), (scale, bias), allow_unused=True, retain_graph=False)

            # TorchScript needs ``scale_grad``/``bias_grad`` to have a consistent
            # tensor type (not Optional[Tensor]) for the scalar update math to
            # compile cleanly.
            scale_grad = grads[0] if grads[0] is not None else torch.zeros_like(scale)
            bias_grad = grads[1] if grads[1] is not None else torch.zeros_like(bias)

            scale = scale - self.lr * scale_grad
            bias = bias - self.lr * bias_grad

        return scale, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predicted, sparse, mask = self._split_inputs(x)

        scale, bias = self._optimize_affine(predicted, sparse, mask)
        refined = scale * predicted + bias

        # light spatial smoothing to reduce frame-to-frame jitter
        smoothed = F.avg_pool2d(refined, kernel_size=3, stride=1, padding=1)
        output = (1.0 - self.smooth_factor) * refined + self.smooth_factor * smoothed
        return output


if __name__ == "__main__":
    # Minimal example demonstrating scripting succeeds.
    module = DepthRefinementModule()
    example = torch.randn(1, 3, 16, 16)
    scripted = torch.jit.script(module)
    print(scripted(example).shape)
