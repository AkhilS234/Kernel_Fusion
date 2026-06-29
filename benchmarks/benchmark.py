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


def run_binary(binary_name, N, dim, batch, output_file):
    result = subprocess.run(
        [str(BUILD_DIR / binary_name), str(N), str(dim), str(batch)],
        capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    match = TIME_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(f"{binary_name} failed or printed no timing:\n{result.stdout}\n{result.stderr}")
    ms = float(match.group(1))
    output = torch.tensor(np.fromfile(OUTPUTS_DIR / output_file, dtype=np.float32)).reshape(batch, N, dim)
    return ms, output


def sdpa_ms(Q, K, V, n_runs=20):
    Q = Q.unsqueeze(1).cuda()
    K = K.unsqueeze(1).cuda()
    V = V.unsqueeze(1).cuda()

    for _ in range(5):
        out = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        out = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_runs

    return elapsed_ms, out.squeeze(1).cpu()


def sweep(seq_lens, head_dims, batches):
    header = f"{'N':>6} {'dim':>5} {'batch':>6} {'naive(ms)':>10} {'flash(ms)':>10} {'sdpa(ms)':>10} {'flash_speedup':>14}"
    print(header)

    for N in seq_lens:
        for dim in head_dims:
            for batch in batches:
                Q = torch.randn(batch, N, dim, dtype=torch.float32)
                K = torch.randn(batch, N, dim, dtype=torch.float32)
                V = torch.randn(batch, N, dim, dtype=torch.float32)

                Q.numpy().tofile(OUTPUTS_DIR / "input_Q.bin")
                K.numpy().tofile(OUTPUTS_DIR / "input_K.bin")
                V.numpy().tofile(OUTPUTS_DIR / "input_V.bin")

                naive_time, _ = run_binary("naive_attention", N, dim, batch, "naive_output.bin")
                flash_time, _ = run_binary("flash_attn_forward", N, dim, batch, "output_S.bin")
                pytorch_time, _ = sdpa_ms(Q, K, V)

                speedup = naive_time / flash_time
                print(f"{N:>6} {dim:>5} {batch:>6} {naive_time:>10.3f} {flash_time:>10.3f} "
                      f"{pytorch_time:>10.3f} {speedup:>13.2f}x")


if __name__ == "__main__":
    sweep(
        seq_lens=[1024, 2048, 4096, 8192],
        head_dims=[64, 128],
        batches=[2, 1],
    )
