"""Test script to verify TorchScript compatibility of examples from CatFileCreation.md"""
import torch
import torch.nn as nn
from typing import Tuple

# Example 1: Cifar10ResNet (simplified without actual ResNet)
class Cifar10ResNet(torch.nn.Module):
    def __init__(self, userLabel=0):
        super().__init__()
        self.userLabel = userLabel
        # Normalization parameters for ImageNet
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def normalize(self, input, mean, std):
        # type: (Tensor, Tuple[float, float, float], Tuple[float, float, float]) -> Tensor
        """Normalize input tensor using mean and std."""
        mean_tensor = torch.tensor(mean, device=input.device, dtype=input.dtype)
        std_tensor = torch.tensor(std, device=input.device, dtype=input.dtype)

        # Reshape for broadcasting
        mean_tensor = mean_tensor.view(1, 3, 1, 1)
        std_tensor = std_tensor.view(1, 3, 1, 1)

        return (input - mean_tensor) / std_tensor

    def forward(self, input):
        # Clamp input to valid range
        input = torch.clamp(input, min=0.0, max=1.0)

        # Normalize input
        normalized = self.normalize(
            input,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )

        # Simplified output (no actual model)
        output = torch.ones(
            1, 1, input.shape[2], input.shape[3],
            device=input.device,
            dtype=input.dtype
        )

        return output


# Example 2: ColourTransfer (from the doc)
class LinearColourTransfer(torch.nn.Module):
    """Core linear color transfer model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, input):
        # type: (Tensor) -> Tensor
        # input shape: [B, 3, H, W]
        B, C, H, W = input.shape

        # Reshape for linear layer: [B*H*W, 3]
        reshaped = input.permute(0, 2, 3, 1).reshape(-1, 3)

        # Apply linear transformation
        transformed = self.linear(reshaped)

        # Reshape back: [B, 3, H, W]
        output = transformed.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return output


class ColourTransfer(torch.nn.Module):
    """Wrapper model with multiple inputs and mix control."""
    def __init__(self, mixValue=1.0):
        super().__init__()
        self.mixValue = mixValue
        self.colour_transfer = LinearColourTransfer()

        # Normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def normalize(self, input):
        # type: (Tensor) -> Tensor
        mean = self.mean.to(device=input.device, dtype=input.dtype)
        std = self.std.to(device=input.device, dtype=input.dtype)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        return (input - mean) / std

    def denormalize(self, input):
        # type: (Tensor) -> Tensor
        mean = self.mean.to(device=input.device, dtype=input.dtype)
        std = self.std.to(device=input.device, dtype=input.dtype)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        return (input * std) + mean

    def forward(self, input):
        # type: (Tensor) -> Tensor
        # Split combined input into two images
        input_split = torch.split(input, 3, 1)
        img1 = input_split[0]  # Source image
        img2 = input_split[1]  # Target image (for color reference)

        # Clamp inputs
        img1 = torch.clamp(img1, min=0.0, max=1.0)
        img2 = torch.clamp(img2, min=0.0, max=1.0)

        # Normalize both images
        img1_norm = self.normalize(img1)
        img2_norm = self.normalize(img2)

        # Apply color transfer to img1
        transferred = self.colour_transfer(img1_norm)

        # Denormalize
        transferred = self.denormalize(transferred)

        # Mix between original and transferred based on mixValue
        output = (1.0 - self.mixValue) * img1 + self.mixValue * transferred

        # Clamp output
        output = torch.clamp(output, min=0.0, max=1.0)

        return output


if __name__ == "__main__":
    print("Testing Example 1: Cifar10ResNet")
    print("=" * 50)
    try:
        model1 = Cifar10ResNet()
        scripted1 = torch.jit.script(model1)
        print("✓ Example 1 is jit-scriptable!")
    except Exception as e:
        print(f"✗ Example 1 failed:")
        print(f"  {type(e).__name__}: {e}")

    print("\n" + "=" * 50)
    print("Testing Example 2: ColourTransfer")
    print("=" * 50)
    try:
        model2 = ColourTransfer()
        scripted2 = torch.jit.script(model2)
        print("✓ Example 2 is jit-scriptable!")
    except Exception as e:
        print(f"✗ Example 2 failed:")
        print(f"  {type(e).__name__}: {e}")
