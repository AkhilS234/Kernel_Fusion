import numpy as np
import torch
import torch.nn.functional as F
import time


def standard_attention(Q, K, V):
    dim = Q.shape[-1]
    S = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)
    P = F.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    return O

def benchmark(sequence_length, dim_head = 64, batch = 1, heads = 1, n_runs=100):
    Q = torch.randn(batch, heads, sequence_length, dim_head, device='cuda', dtype=torch.float16)
    K = torch.randn(batch, heads, sequence_length, dim_head, device='cuda', dtype=torch.float16)
    V = torch.randn(batch, heads, sequence_length, dim_head, device='cuda', dtype=torch.float16)

    for j in range(10):
        j = standard_attention(Q, K, V)
    torch.cuda.synchronize()

    start = time.perf_counter()

    for i in range(n_runs):
        O = standard_attention(Q, K, V)
    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed_time_ms = ((end - start) * 1000) / 1000

    bytes_accessed = (
    3 * batch * heads * sequence_length * dim_head * 2 +
    2 * batch * heads * sequence_length * sequence_length * 2 +
    batch * heads * sequence_length * dim_head * 2
    )
    bandwidth_tb = (bytes_accessed / (elapsed_time_ms / 1000)) / 1e9

    print(f"Sequence Length: {sequence_length:5d}, Time per run: {elapsed_time_ms:.3f} ms, Bandwidth: {bandwidth_tb:.2f} GB/s, HBM Accessed: {bytes_accessed / 1e9:.4f} GB")


if __name__ == "__main__":
    sequence_lengths = [128, 256, 512, 1024, 2048]
    for seq_len in sequence_lengths:
        benchmark(seq_len)