import torch
import torch.nn.functional as F
import numpy as np

N = 1024
dim = 64

cuda_output = np.fromfile("outputs/naive_output.bin", dtype=np.float32)
cuda_output = torch.tensor(cuda_output).reshape(N, dim)

Q = torch.tensor(np.fromfile("outputs/input_Q.bin", dtype=np.float32)).reshape(N, dim)
K = torch.tensor(np.fromfile("outputs/input_K.bin", dtype=np.float32)).reshape(N, dim)
V = torch.tensor(np.fromfile("outputs/input_V.bin", dtype=np.float32)).reshape(N, dim)

S = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)
P = F.softmax(S, dim=-1)
ref_output = torch.matmul(P, V)

match = torch.allclose(ref_output, cuda_output, atol=1e-3)
print(f"outputs match: {match}")
print(f"max difference: {(ref_output - cuda_output).abs().max().item():.6f}")