import torch
import torch.nn as nn

# Define the shape of your tensor
shape = (3, 3)

a = 0.05

b = 0.55
mean = a + torch.rand(1).item() * (b - a)

# Initialize a tensor with truncated normal distribution
tensor = torch.empty(shape)

# Use trunc_normal_ to initialize the tensor
nn.init.trunc_normal_(tensor, mean=mean, std=0.1, a=a, b=b)

# Ensure the sum of the tensor is at least 1
while torch.sum(tensor) < 1:
    nn.init.trunc_normal_(tensor, mean=mean, std=0.01, a=a, b=b)

# Print the initialized tensor
print(tensor)
