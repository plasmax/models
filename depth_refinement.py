"""
Depth Refinement Module with Test-Time Optimization

This module refines predicted depth using sparse ground truth observations.
All optimization happens inside the forward() function, making it suitable
for use in Nuke's Inference node.

Input tensor shape: [1, 3, H, W]
- Channel 0: Predicted depth (may have frame-to-frame jitter)
- Channel 1: Sparse ground truth depth (tracking points or lidar)
- Channel 2: Binary mask indicating valid sparse ground truth locations

Output tensor shape: [1, 1, H, W]
- Refined depth map

TorchScript compatible: Yes
Requires pretrained weights: No (test-time optimization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DepthRefinementModule(torch.nn.Module):
    """
    Test-time optimization module that refines predicted depth using sparse GT.

    The optimization consists of two stages:
    1. Closed-form global affine correction (scale + shift)
    2. Iterative diffusion-based local correction with data fidelity constraints

    This handles:
    - Frame-to-frame jitter in predicted depth (via global affine correction)
    - Extremely sparse ground truth like tracking points (via smooth interpolation)
    - Lidar-like sparse data with holes (via diffusion-based optimization)
    """

    def __init__(
        self,
        num_iterations: int = 50,
        smooth_weight: float = 0.5,
        downsample_factor: int = 4
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.smooth_weight = smooth_weight
        self.downsample_factor = downsample_factor

        # Laplacian kernel for smoothness (diffusion)
        # This is equivalent to averaging with neighbors
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([[[[0.0, 1.0, 0.0],
                           [1.0, -4.0, 1.0],
                           [0.0, 1.0, 0.0]]]])
        )

    def _compute_global_affine(
        self,
        predicted: torch.Tensor,
        sparse_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute closed-form global scale and shift correction.

        Solves: argmin_{a,b} sum_i mask_i * (a * pred_i + b - gt_i)^2

        Returns (scale, shift) tensors.
        """
        mask_sum = mask.sum()

        # Default values if no valid points
        one = torch.ones(1, device=predicted.device, dtype=predicted.dtype)
        zero = torch.zeros(1, device=predicted.device, dtype=predicted.dtype)

        if mask_sum < 2.0:
            return one, zero

        # Compute sums for normal equations
        pred_masked = predicted * mask
        gt_masked = sparse_gt * mask

        sum_p = pred_masked.sum()
        sum_g = gt_masked.sum()
        sum_pp = (predicted * pred_masked).sum()
        sum_pg = (predicted * gt_masked).sum()
        n = mask_sum

        # Solve 2x2 linear system using Cramer's rule
        det = sum_pp * n - sum_p * sum_p

        if det.abs() < 1e-8:
            return one, zero

        scale = (n * sum_pg - sum_p * sum_g) / det
        shift = (sum_pp * sum_g - sum_p * sum_pg) / det

        # Clamp scale to reasonable range to avoid instability
        scale = torch.clamp(scale, 0.1, 10.0)

        return scale, shift

    def _optimize_correction_field(
        self,
        predicted: torch.Tensor,
        sparse_gt: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Iteratively optimize a local correction field using diffusion.

        Uses Jacobi iteration to solve:
        minimize: sum_i mask_i * (pred_i + corr_i - gt_i)^2 + lambda * ||grad(corr)||^2

        This is equivalent to solving a linear system via iterative refinement.
        Works at a downsampled resolution for efficiency, then upsamples.
        """
        B, C, H, W = predicted.shape
        device = predicted.device
        dtype = predicted.dtype

        # Work at lower resolution for efficiency
        ds = self.downsample_factor
        H_ds = H // ds
        W_ds = W // ds

        if H_ds < 8 or W_ds < 8:
            H_ds = H
            W_ds = W
            ds = 1

        # Downsample inputs carefully for sparse data
        if ds > 1:
            pred_ds = F.avg_pool2d(predicted, ds)
            # For sparse GT, we need weighted average (sum of values / sum of mask)
            gt_sum_ds = F.avg_pool2d(sparse_gt, ds) * (ds * ds)  # Undo avg to get sum
            mask_sum_ds = F.avg_pool2d(mask, ds) * (ds * ds)  # Sum of mask in each block
            # Avoid division by zero
            mask_sum_ds_safe = torch.clamp(mask_sum_ds, min=1e-8)
            gt_ds = gt_sum_ds / mask_sum_ds_safe  # Proper weighted average
            mask_ds = (mask_sum_ds > 0.5).to(dtype)
            # Zero out GT where no valid samples
            gt_ds = gt_ds * mask_ds
        else:
            pred_ds = predicted
            gt_ds = sparse_gt
            mask_ds = mask

        # Target correction at observed points (what we need to add to prediction)
        target_correction = gt_ds - pred_ds

        # Initialize correction field to zero
        correction = torch.zeros(1, 1, H_ds, W_ds, device=device, dtype=dtype)

        # Check if we have any valid points
        if mask_ds.sum() < 1.0:
            if ds > 1:
                return F.interpolate(
                    correction, size=(H, W), mode='bilinear', align_corners=False
                )
            return correction

        # Smoothness weight for blending
        smooth_w = self.smooth_weight

        # Jacobi iteration for solving the variational problem:
        # minimize: sum_i mask_i * (corr_i - target_i)^2 + smooth_w * ||grad(corr)||^2
        #
        # The optimal solution satisfies:
        # - At observed points: corr = (target + smooth_w * avg_neighbors) / (1 + smooth_w)
        # - At unobserved points: corr = avg_neighbors (Laplace equation)

        for i in range(self.num_iterations):
            # Compute average of neighbors (diffusion/smoothing)
            padded = F.pad(correction, (1, 1, 1, 1), mode='replicate')

            left = padded[:, :, 1:-1, :-2]
            right = padded[:, :, 1:-1, 2:]
            top = padded[:, :, :-2, 1:-1]
            bottom = padded[:, :, 2:, 1:-1]

            neighbor_avg = (left + right + top + bottom) / 4.0

            # Update rule:
            # - At observed points: blend between target and neighbor average
            # - At unobserved points: just use neighbor average (diffusion)
            observed_update = (target_correction + smooth_w * neighbor_avg) / (1.0 + smooth_w)
            unobserved_update = neighbor_avg

            # Combine based on mask
            correction = mask_ds * observed_update + (1.0 - mask_ds) * unobserved_update

        # Upsample correction to full resolution
        if ds > 1:
            correction = F.interpolate(
                correction, size=(H, W), mode='bilinear', align_corners=False
            )

        return correction

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Refine predicted depth using sparse ground truth via test-time optimization.

        Args:
            input: Tensor of shape [1, 3, H, W] with channels:
                   [predicted_depth, sparse_gt_depth, valid_mask]

        Returns:
            Refined depth tensor of shape [1, 1, H, W]
        """
        # Extract channels
        predicted = input[:, 0:1, :, :]  # [1, 1, H, W]
        sparse_gt = input[:, 1:2, :, :]  # [1, 1, H, W]
        mask = input[:, 2:3, :, :]       # [1, 1, H, W]

        # Binarize mask
        mask = (mask > 0.5).to(input.dtype)

        # Stage 1: Global affine correction (closed-form)
        scale, shift = self._compute_global_affine(predicted, sparse_gt, mask)
        predicted_corrected = scale * predicted + shift

        # Stage 2: Local correction field optimization
        local_correction = self._optimize_correction_field(
            predicted_corrected, sparse_gt, mask
        )

        # Final refined output
        refined = predicted_corrected + local_correction

        return refined


class DepthRefinementSimple(torch.nn.Module):
    """
    Simplified version using only closed-form global affine correction.

    Faster but less accurate for spatially-varying errors.
    Good baseline for comparison.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        predicted = input[:, 0:1, :, :]
        sparse_gt = input[:, 1:2, :, :]
        mask = input[:, 2:3, :, :]

        mask = (mask > 0.5).to(input.dtype)
        mask_sum = mask.sum()

        if mask_sum < 2.0:
            return predicted

        pred_m = predicted * mask
        gt_m = sparse_gt * mask

        sum_p = pred_m.sum()
        sum_g = gt_m.sum()
        sum_pp = (predicted * pred_m).sum()
        sum_pg = (predicted * gt_m).sum()
        n = mask_sum

        det = sum_pp * n - sum_p * sum_p

        if det.abs() < 1e-8:
            return predicted

        scale = (n * sum_pg - sum_p * sum_g) / det
        shift = (sum_pp * sum_g - sum_p * sum_pg) / det

        scale = torch.clamp(scale, 0.1, 10.0)

        return scale * predicted + shift


def create_torchscript_model(
    num_iterations: int = 50,
    smooth_weight: float = 0.5,
    downsample_factor: int = 4
) -> torch.jit.ScriptModule:
    """
    Create and return a TorchScript-compatible model.

    Args:
        num_iterations: Number of diffusion iterations
        smooth_weight: Weight for smoothness (higher = smoother corrections)
        downsample_factor: Factor to downsample correction field (for speed)

    Returns:
        TorchScript module ready for Nuke .cat file creation
    """
    model = DepthRefinementModule(
        num_iterations=num_iterations,
        smooth_weight=smooth_weight,
        downsample_factor=downsample_factor
    )
    model.eval()

    scripted = torch.jit.script(model)

    return scripted


def save_model(filepath: str = "depth_refinement.pt", **kwargs) -> None:
    """Save the TorchScript model to a .pt file for Nuke."""
    scripted = create_torchscript_model(**kwargs)
    scripted.save(filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    print("Testing DepthRefinementModule...")

    B, C, H, W = 1, 3, 256, 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)

    # Ground truth depth
    y_coords = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(1, 1, H, W)
    x_coords = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(1, 1, H, W)
    gt_depth = 0.5 + 0.3 * torch.sin(2 * 3.14159 * x_coords) * torch.cos(2 * 3.14159 * y_coords)

    # Predicted depth with errors
    predicted_depth = 0.8 * gt_depth + 0.1 + 0.02 * torch.randn_like(gt_depth)

    # Sparse mask (~5% of pixels)
    mask = (torch.rand(1, 1, H, W) < 0.05).float()

    # Sparse GT
    sparse_gt = gt_depth * mask

    input_tensor = torch.cat([predicted_depth, sparse_gt, mask], dim=1).to(device)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Device: {device}")
    print(f"Sparse points: {int(mask.sum().item())} / {H*W} ({100*mask.mean().item():.1f}%)")

    model = DepthRefinementModule(num_iterations=100, smooth_weight=0.5)
    model = model.to(device)
    model.eval()

    print("\nRunning forward pass (with test-time optimization)...")
    output = model(input_tensor)

    print(f"Output shape: {output.shape}")

    pred_error = ((predicted_depth.to(device) - gt_depth.to(device)) ** 2).mean().sqrt()
    refined_error = ((output - gt_depth.to(device)) ** 2).mean().sqrt()

    print(f"\nRMSE before refinement: {pred_error.item():.6f}")
    print(f"RMSE after refinement:  {refined_error.item():.6f}")
    print(f"Improvement: {100 * (1 - refined_error/pred_error).item():.1f}%")

    print("\nTesting TorchScript conversion...")
    try:
        scripted = torch.jit.script(model)
        output_scripted = scripted(input_tensor)
        print("TorchScript conversion: SUCCESS")
        print(f"Scripted output matches: {torch.allclose(output, output_scripted, atol=1e-5)}")

        scripted.save("depth_refinement.pt")
        print("Model saved to depth_refinement.pt")
    except Exception as e:
        print(f"TorchScript conversion failed: {e}")

    # Test the simple version too
    print("\nTesting DepthRefinementSimple...")
    simple_model = DepthRefinementSimple()
    simple_model = simple_model.to(device)
    try:
        scripted_simple = torch.jit.script(simple_model)
        output_simple = scripted_simple(input_tensor)
        simple_error = ((output_simple - gt_depth.to(device)) ** 2).mean().sqrt()
        print(f"Simple model RMSE: {simple_error.item():.6f}")
        print("Simple model TorchScript: SUCCESS")
        scripted_simple.save("depth_refinement_simple.pt")
        print("Simple model saved to depth_refinement_simple.pt")
    except Exception as e:
        print(f"Simple model TorchScript failed: {e}")

    print("\nDone!")
