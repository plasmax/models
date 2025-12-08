import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledDotProductAttentionModel(nn.Module):
    """Minimal model that exercises torch.nn.functional.scaled_dot_product_attention.

    The model expects a 4D input tensor with batch size 1 and channels-last layout:
    ``(1, in_channels, height, width)``. The internal attention operates on flattened
    spatial tokens using a single attention head.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        # Ensure the module works regardless of the input device/dtype.
        x = self.input_proj(x)

        batch, channels, height, width = x.shape
        seq_len = height * width

        # Flatten spatial dimensions to tokens: (batch, seq_len, channels)
        tokens = x.permute(0, 2, 3, 1).reshape(batch, seq_len, channels)

        # Project to queries, keys, and values.
        q = self.q_proj(tokens).unsqueeze(1)  # (batch, heads=1, seq_len, hidden_dim)
        k = self.k_proj(tokens).unsqueeze(1)
        v = self.v_proj(tokens).unsqueeze(1)

        # Apply scaled dot-product attention.
        attended = F.scaled_dot_product_attention(q, k, v)

        # Collapse the head dimension and reshape back to image layout.
        attended = attended.squeeze(1).reshape(batch, height, width, self.hidden_dim)
        attended = attended.permute(0, 3, 1, 2)

        return self.output_proj(attended)


if __name__ == "__main__":
    model = ScaledDotProductAttentionModel()
    model.eval()

    example_input = torch.randn(1, 3, 8, 8)
    scripted = torch.jit.script(model)
    scripted.save("scaled_dot_product_attention_model.pt")
    print("Saved scaled_dot_product_attention_model.pt")
