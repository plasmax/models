import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBAKernelConv(nn.Module):
    """
    A simple convolutional module that uses the alpha channel as a grid of kernels.

    The input is an RGBA tensor where:
    - RGB channels (0-2): The image data to be convolved
    - Alpha channel (3): A grid of kernels to sample from

    Args:
        kernel_size (int): Size of the convolution kernel (default: 3)
        stride (int): Stride of the convolution (default: 1)
        padding (int): Padding for the convolution (default: 1)
    """

    def __init__(self, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, rgba_input):
        """
        Args:
            rgba_input: Tensor of shape (B, 4, H, W) where channel 3 is alpha

        Returns:
            Tensor of shape (B, 3, H_out, W_out) - convolved RGB channels
        """
        # Split RGB and alpha channels
        rgb = rgba_input[:, :3, :, :]  # (B, 3, H, W)
        alpha = rgba_input[:, 3:4, :, :]  # (B, 1, H, W)

        # Unfold the alpha channel to get kernel windows
        # This creates a grid where each spatial position has a kernel_size x kernel_size patch
        alpha_unfolded = F.unfold(
            alpha,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )  # (B, kernel_size^2, L) where L is number of output positions

        # Unfold the RGB channels similarly
        rgb_unfolded = F.unfold(
            rgb,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )  # (B, 3*kernel_size^2, L)

        B, _, L = alpha_unfolded.shape

        # Reshape for easier computation
        # alpha_unfolded: (B, kernel_size^2, L)
        # rgb_unfolded: (B, 3, kernel_size^2, L)
        rgb_unfolded = rgb_unfolded.view(B, 3, self.kernel_size**2, L)

        # Apply the alpha kernels to RGB patches
        # For each spatial position, multiply the RGB patch by the corresponding alpha kernel
        output = torch.sum(rgb_unfolded * alpha_unfolded.unsqueeze(1), dim=2)  # (B, 3, L)

        # Calculate output dimensions
        H, W = rgba_input.shape[2:]
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Reshape back to spatial dimensions
        output = output.view(B, 3, H_out, W_out)

        return output


if __name__ == "__main__":
    # Simple test
    print("Testing RGBAKernelConv module...")

    # Create a simple RGBA input
    B, H, W = 2, 8, 8
    rgba_input = torch.randn(B, 4, H, W)

    # Initialize module
    conv = RGBAKernelConv(kernel_size=3, stride=1, padding=1)

    # Forward pass
    output = conv(rgba_input)

    print(f"Input shape: {rgba_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({B}, 3, {H}, {W})")
    print("Test completed successfully!")
