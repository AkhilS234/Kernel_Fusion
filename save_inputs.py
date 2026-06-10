import torch
import numpy as np

N, dim = 1024, 64
torch.manual_seed(0)
Q = torch.rand(N, dim)
K = torch.rand(N, dim)
V = torch.rand(N, dim)

Q.numpy().tofile("outputs/input_Q.bin")
K.numpy().tofile("outputs/input_K.bin")
V.numpy().tofile("outputs/input_V.bin")