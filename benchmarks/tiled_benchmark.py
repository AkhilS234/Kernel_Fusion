import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TIME_RE = re.compile(r"Time taken:\s*([\d.]+)\s*ms")

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# flashAttn_tile.cu hard-codes TILE_K=64 and doesn't loop the O/V accumulation
# over multiple head-dim chunks, so it only works correctly at dim=64.
TILE_DIM = 64


def run_binary(cmd, output_file, output_shape, n_warmup=5):
    for _ in range(n_warmup):
        subprocess.run(cmd, capture_output=True, cwd=str(REPO_ROOT))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    match = TIME_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(f"{cmd} failed or printed no timing:\n{result.stdout}\n{result.stderr}")
    ms = float(match.group(1))
    output = torch.tensor(np.fromfile(OUTPUTS_DIR / output_file, dtype=np.float32)).reshape(output_shape)
    return ms, output


def run_tiled(N, dim, n_warmup=5):
    cmd = [str(BUILD_DIR / "flashAttn_tile"), str(N), str(dim)]
    return run_binary(cmd, "output_S.bin", (N, dim), n_warmup=n_warmup)


def run_flash_attn_forward(N, dim, n_warmup=5):
    cmd = [str(BUILD_DIR / "flash_attn_forward"), str(N), str(dim), "1", "1"]
    ms, out = run_binary(cmd, "output_S.bin", (1, 1, N, dim), n_warmup=n_warmup)
    return ms, out.squeeze(0).squeeze(0)


def run_naive(N, dim, n_warmup=1):
    cmd = [str(BUILD_DIR / "naive_attention"), str(N), str(dim), "1", "1"]
    ms, out = run_binary(cmd, "naive_output.bin", (1, 1, N, dim), n_warmup=n_warmup)
    return ms, out.squeeze(0).squeeze(0)


def sdpa_ms(Q, K, V, n_runs=20):
    Q = Q.cuda()
    K = K.cuda()
    V = V.cuda()

    for _ in range(5):
        out = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        out = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_runs

    return elapsed_ms, out.cpu()


def write_inputs(Q, K, V):
    # flashAttn_tile expects Q as (N x dim), K TRANSPOSED as (dim x N)
    # (bShape = {K, N} in the kernel), and V as (N x dim).
    Q.numpy().tofile(OUTPUTS_DIR / "input_Q.bin")
    K.transpose(0, 1).contiguous().numpy().tofile(OUTPUTS_DIR / "input_K.bin")
    V.numpy().tofile(OUTPUTS_DIR / "input_V.bin")


def sweep(seq_lens, run_naive_flag=False, run_wmma_flag=True, naive_n_warmup=1):
    cols = ["N", "dim"]
    if run_naive_flag:
        cols += ["naive(ms)"]
    if run_wmma_flag:
        cols += ["wmma(ms)"]
    cols += ["tiled(ms)", "sdpa(ms)", "tiled/sdpa"]
    header = " ".join(f"{c:>12}" for c in cols)
    print(header)

    for N in seq_lens:
        dim = TILE_DIM
        Q = torch.randn(N, dim, dtype=torch.float32)
        K = torch.randn(N, dim, dtype=torch.float32)
        V = torch.randn(N, dim, dtype=torch.float32)

        write_inputs(Q, K, V)
        tiled_time, _ = run_tiled(N, dim)

        Q_sdpa = Q.unsqueeze(0).unsqueeze(0)
        K_sdpa = K.unsqueeze(0).unsqueeze(0)
        V_sdpa = V.unsqueeze(0).unsqueeze(0)
        sdpa_time, _ = sdpa_ms(Q_sdpa, K_sdpa, V_sdpa)
        tiled_vs_sdpa = tiled_time / sdpa_time

        row = [f"{N:>12}", f"{dim:>12}"]

        if run_naive_flag:
            # naive_attention.cu reads Q/K/V untransposed -- rewrite K in that
            # layout right before calling it, then restore the tiled layout
            # in case another tiled run follows in the same sweep.
            K.numpy().tofile(OUTPUTS_DIR / "input_K.bin")
            naive_time, _ = run_naive(N, dim, n_warmup=naive_n_warmup)
            row.append(f"{naive_time:>12.3f}")
            K.transpose(0, 1).contiguous().numpy().tofile(OUTPUTS_DIR / "input_K.bin")

        if run_wmma_flag:
            K.numpy().tofile(OUTPUTS_DIR / "input_K.bin")
            wmma_time, _ = run_flash_attn_forward(N, dim)
            row.append(f"{wmma_time:>12.3f}")
            K.transpose(0, 1).contiguous().numpy().tofile(OUTPUTS_DIR / "input_K.bin")

        row += [f"{tiled_time:>12.3f}", f"{sdpa_time:>12.3f}", f"{tiled_vs_sdpa:>11.2f}x"]
        print(" ".join(row))


if __name__ == "__main__":
    print("=== tiled flash attention vs sdpa (dim fixed at 64 -- TILE_K limit) ===")
    sweep(
        seq_lens=[1024, 2048, 4096, 8192],
        run_naive_flag=False,
        run_wmma_flag=True,
    )

    print("\n=== showcase: naive vs wmma vs tiled vs sdpa ===")
    sweep(
        seq_lens=[2048],
        run_naive_flag=True,
        run_wmma_flag=True,
        naive_n_warmup=1,
    )
