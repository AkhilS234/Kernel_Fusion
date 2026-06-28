import torch 
import torch.nn.functional as F
import numpy as np
import subprocess

N = 1024
dim = 64

S = torch.randn(N, N, dtype=torch.float32)
S.numpy().tofile("outputs/input_S.bin")

subprocess.run(["./online_softmax"])

cuda_output = torch.tensor(np.fromfile("outputs/output_S.bin", dtype=np.float32)).reshape(N,N)
pytorch_output = torch.softmax(S, dim=1)

max_diff = (cuda_output-pytorch_output).abs().max().item()
assert max_diff < 1e-5, f"FAILED: max diff {max_diff} exceeds tolerance"
print("PASSED")