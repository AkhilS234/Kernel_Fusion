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


def run_tiled(N, dim, batch=1, heads=1, n_warmup=5):
    cmd = [str(BUILD_DIR / "flashAttn_tile"), str(N), str(dim), str(batch), str(heads)]
    return run_binary(cmd, "output_S.bin", (batch, heads, N, dim), n_warmup=n_warmup)


def run_flash_attn_forward(N, dim, batch=1, heads=1, n_warmup=5):
    cmd = [str(BUILD_DIR / "flash_attn_forward"), str(N), str(dim), str(batch), str(heads)]
    return run_binary(cmd, "output_S.bin", (batch, heads, N, dim), n_warmup=n_warmup)


def run_naive(N, dim, batch=1, heads=1, n_warmup=1):
    cmd = [str(BUILD_DIR / "naive_attention"), str(N), str(dim), str(batch), str(heads)]
    return run_binary(cmd, "naive_output.bin", (batch, heads, N, dim), n_warmup=n_warmup)


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


def write_tiled_inputs(Q, K, V):
    Q.numpy().tofile(OUTPUTS_DIR / "input_Q.bin")
    K.transpose(-2, -1).contiguous().numpy().tofile(OUTPUTS_DIR / "input_K.bin")
    V.numpy().tofile(OUTPUTS_DIR / "input_V.bin")


def write_untransposed_inputs(Q, K, V):
    Q.numpy().tofile(OUTPUTS_DIR / "input_Q.bin")
    K.numpy().tofile(OUTPUTS_DIR / "input_K.bin")
    V.numpy().tofile(OUTPUTS_DIR / "input_V.bin")


def sweep(seq_lens, dim=64, batch=1, heads=1, run_naive_flag=False, run_wmma_flag=True, naive_n_warmup=1):
    cols = ["N", "dim", "batch", "heads"]
    if run_naive_flag:
        cols += ["naive(ms)"]
    if run_wmma_flag:
        cols += ["wmma(ms)"]
    cols += ["tiled(ms)", "sdpa(ms)", "tiled/sdpa"]
    header = " ".join(f"{c:>12}" for c in cols)
    print(header)

    for N in seq_lens:
        Q = torch.randn(batch, heads, N, dim, dtype=torch.float32)
        K = torch.randn(batch, heads, N, dim, dtype=torch.float32)
        V = torch.randn(batch, heads, N, dim, dtype=torch.float32)

        write_tiled_inputs(Q, K, V)
        tiled_time, _ = run_tiled(N, dim, batch=batch, heads=heads)

        sdpa_time, _ = sdpa_ms(Q, K, V)
        tiled_vs_sdpa = tiled_time / sdpa_time

        row = [f"{N:>12}", f"{dim:>12}", f"{batch:>12}", f"{heads:>12}"]

        if run_naive_flag:
            write_untransposed_inputs(Q, K, V)
            naive_time, _ = run_naive(N, dim, batch=batch, heads=heads, n_warmup=naive_n_warmup)
            row.append(f"{naive_time:>12.3f}")
            write_tiled_inputs(Q, K, V)

        if run_wmma_flag:
            write_untransposed_inputs(Q, K, V)
            wmma_time, _ = run_flash_attn_forward(N, dim, batch=batch, heads=heads)
            row.append(f"{wmma_time:>12.3f}")
            write_tiled_inputs(Q, K, V)

        row += [f"{tiled_time:>12.3f}", f"{sdpa_time:>12.3f}", f"{tiled_vs_sdpa:>11.2f}x"]
        print(" ".join(row))


if __name__ == "__main__":
    print("=== tiled flash attention vs sdpa (dim=64, single head/batch) ===")
    sweep(
        seq_lens=[1024, 2048, 4096, 8192],
        dim=64,
        run_naive_flag=False,
        run_wmma_flag=True,
    )

    print("\n=== experimental progression: naive -> wmma -> tiled, matching the")
    print("=== original benchmark shape (N=2048, dim=128, heads=8, batch=2) ===")
    sweep(
        seq_lens=[2048],
        dim=128,
        batch=2,
        heads=8,
        run_naive_flag=True,
        run_wmma_flag=True,
        naive_n_warmup=1,
    )
