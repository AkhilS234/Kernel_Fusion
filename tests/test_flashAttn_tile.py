import torch
import numpy as np
import subprocess
import torch.nn.functional as F

N = 2048
dim = 128
batch = 2
heads = 8

torch.manual_seed(0)
Q = torch.randn(batch, heads, N, dim, dtype=torch.float32)
K = torch.randn(batch, heads, N, dim, dtype=torch.float32)
V = torch.randn(batch, heads, N, dim, dtype=torch.float32)

Q.numpy().tofile("outputs/input_Q.bin")
K.transpose(-2, -1).contiguous().numpy().tofile("outputs/input_K.bin")
V.numpy().tofile("outputs/input_V.bin")

result = subprocess.run(
    ["./build/flashAttn_tile", str(N), str(dim), str(batch), str(heads)],
    capture_output=True, text=True
)
print("--- binary stdout ---")
print(result.stdout)
if result.returncode != 0:
    print("--- binary stderr ---")
    print(result.stderr)
    raise RuntimeError(f"Binary exited with code {result.returncode}")

ref = F.scaled_dot_product_attention(Q.cuda(), K.cuda(), V.cuda()).cpu()

out = torch.tensor(np.fromfile("outputs/output_S.bin", dtype=np.float32)).reshape(batch, heads, N, dim)

max_diff = (out - ref).abs().max().item()

print(f"Max difference: {max_diff:.6f}")
assert max_diff < 1e-3, f"FAILED: max diff {max_diff}"
print("PASSED")
