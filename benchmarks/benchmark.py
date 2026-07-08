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


def run_binary(binary_name, N, dim, batch, num_heads, output_file, n_warmup=5):
    cmd = [str(BUILD_DIR / binary_name), str(N), str(dim), str(batch), str(num_heads)]
    for _ in range(n_warmup):
        subprocess.run(cmd, capture_output=True, cwd=str(REPO_ROOT))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    match = TIME_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(f"{binary_name} failed or printed no timing:\n{result.stdout}\n{result.stderr}")
    ms = float(match.group(1))
    output = torch.tensor(np.fromfile(OUTPUTS_DIR / output_file, dtype=np.float32)).reshape(batch, num_heads, N, dim)
    return ms, output


def sdpa_ms(Q, K, V, n_runs=20):
    # Q/K/V shape: [batch, num_heads, N, dim]
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


def sweep(seq_lens, head_dims, num_heads_list, batches, run_naive=True, naive_n_warmup=5):
    if run_naive:
        header = f"{'N':>6} {'dim':>5} {'heads':>6} {'batch':>6} {'naive(ms)':>10} {'flash(ms)':>10} {'sdpa(ms)':>10} {'naive/flash':>12} {'flash/sdpa':>11}"
    else:
        header = f"{'N':>6} {'dim':>5} {'heads':>6} {'batch':>6} {'flash(ms)':>10} {'sdpa(ms)':>10} {'flash/sdpa':>11}"
    print(header)

    for N in seq_lens:
        for dim in head_dims:
            for num_heads in num_heads_list:
                for batch in batches:
                    Q = torch.randn(batch, num_heads, N, dim, dtype=torch.float32)
                    K = torch.randn(batch, num_heads, N, dim, dtype=torch.float32)
                    V = torch.randn(batch, num_heads, N, dim, dtype=torch.float32)

                    Q.numpy().tofile(OUTPUTS_DIR / "input_Q.bin")
                    K.numpy().tofile(OUTPUTS_DIR / "input_K.bin")
                    V.numpy().tofile(OUTPUTS_DIR / "input_V.bin")

                    flash_time, _ = run_binary("flash_attn_forward", N, dim, batch, num_heads, "output_S.bin")
                    pytorch_time, _ = sdpa_ms(Q, K, V)
                    flash_vs_sdpa = flash_time / pytorch_time

                    if run_naive:
                        naive_time, _ = run_binary("naive_attention", N, dim, batch, num_heads, "naive_output.bin",
                                                   n_warmup=naive_n_warmup)
                        naive_vs_flash = naive_time / flash_time
                        print(f"{N:>6} {dim:>5} {num_heads:>6} {batch:>6} {naive_time:>10.3f} {flash_time:>10.3f} "
                              f"{pytorch_time:>10.3f} {naive_vs_flash:>11.2f}x {flash_vs_sdpa:>10.2f}x")
                    else:
                        print(f"{N:>6} {dim:>5} {num_heads:>6} {batch:>6} {flash_time:>10.3f} "
                              f"{pytorch_time:>10.3f} {flash_vs_sdpa:>10.2f}x")


if __name__ == "__main__":
    # single showcase row: naive + flash + sdpa at the same config for a credible apples-to-apples claim
    # naive is O(N²) so cap N at 2048 to keep warmup reasonable (~1 warmup run)
    print("=== showcase: naive vs flash vs sdpa ===")
    sweep(
        seq_lens=[2048],
        head_dims=[128],
        num_heads_list=[8],
        batches=[2],
        run_naive=True,
        naive_n_warmup=1,
    )

    print("\n=== naive vs flash (scaling sweep, single head) ===")
    sweep(
        seq_lens=[1024, 2048],
        head_dims=[64, 128],
        num_heads_list=[1],
        batches=[1],
        run_naive=True,
    )

    print("\n=== flash vs sdpa (full range) ===")
    sweep(
        seq_lens=[1024, 2048, 4096, 8192],
        head_dims=[64, 128],
        num_heads_list=[1, 8],
        batches=[1, 2],
        run_naive=False,
    )
