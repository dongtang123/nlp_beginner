import torch

a = torch.tensor([1, 1, 2])
b = torch.tensor([1, 0, 0])
print((a == b).sum())
