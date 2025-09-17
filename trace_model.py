import torch
import torch.nn as nn

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Instantiate the model
model = SimpleNet()

# Create example input, which is required for tracing
# NOTE: Tracing relies on the provided example input to capture the operations.
example_input = torch.randn(1, 10)

# JIT trace the model
traced_model = torch.jit.trace(model, example_input)

# Save the traced model
traced_model.save("traced_model.pt")

print("Successfully traced and saved the model.")
