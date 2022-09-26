import numpy as np
import torch

y = torch.empty(1, 1, 2, 2, 2)
x = torch.empty(1, 1, 2, 2, 2)
loss = torch.nn.L1Loss()

print(loss(x, y).item())
