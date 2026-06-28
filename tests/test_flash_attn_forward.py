import torch 
import numpy as np
import subprocess
import torch.nn.functional as F

N = 1024
dim = 64

Q = torch.randn(N, dim, dtype=torch.float32)
K = torch.randn(N, dim, dtype=torch.float32)
V = torch.randn(N, dim, dtype=torch.float32)

Q.numpy().tofile("outputs/input_Q.bin")
K.numpy().tofile("outputs/input_K.bin")
V.numpy().tofile("outputs/input_V.bin")

subprocess.run(["./build/flash_attn_forward", str(N), str(dim), "1"])

Q_sdpa = Q.unsqueeze(0).unsqueeze(0)  
K_sdpa = K.unsqueeze(0).unsqueeze(0)
V_sdpa = V.unsqueeze(0).unsqueeze(0)

cuda_output = torch.tensor(np.fromfile("outputs/output_S.bin", dtype=np.float32)).reshape(N,dim)
pytorch_output = torch.nn.functional.scaled_dot_product_attention(Q_sdpa, K_sdpa, V_sdpa, attn_mask=None, dropout_p=0.0, is_causal=False).squeeze(0).squeeze(0)

max_diff = (cuda_output-pytorch_output).abs().max().item()

print(f"Max difference: {max_diff:.6f}")
assert max_diff < 1e-3, f"FAILED: max diff {max_diff}"
print("PASSED")