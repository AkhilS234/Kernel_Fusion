# FlashAttention CUDA Kernels — WMMA Tensor Cores vs. CUDA Tile C++

Two independent CUDA implementations of the FlashAttention forward pass, built to compare a
hand-tuned tensor-core kernel against a compiler-driven tile-abstraction kernel on the same
problem. Verified against PyTorch's `scaled_dot_product_attention` throughout.

## Results

At N=2048, dim=128, heads=8, batch=2 (Ada/L4, compute capability 8.9):

| Implementation | Time | vs. naive | Correctness (max abs diff vs. SDPA) |
|---|---|---|---|
| Naive (unfused, three separate kernels) | 822.9 ms | — | verified |
| WMMA (hand-tuned tensor cores) | 9.45 ms | 87x | 1.9e-4 |
| CUDA Tile C++ (compiler-managed tiling) | 7.59 ms | 108x | 1.4e-4 |
| PyTorch SDPA (reference) | 3.06 ms | 269x | — |

The tile-based kernel ends up ~24% faster than the hand-tuned WMMA kernel at this shape, while
using a fraction of the manual shared-memory/tensor-core bookkeeping — see the writeups below for
why, and where each kernel's remaining bottleneck actually is.

## Two approaches

### `src/flash_attn_forward.cu` — hand-tuned WMMA tensor cores

Every byte of shared memory, every tensor-core fragment, every synchronization point is manually
placed. `nvcuda::wmma` fragment intrinsics drive both matmuls (`Q @ K^T` and `P @ V`); Q/K/V are
FP16, softmax and accumulation stay FP32.

- Warp-cooperative design: each warp owns a private 16-row slice of the query block
  (`NWARPS` warps/block, `Br = NWARPS × 16` total rows). K/V tiles load once per block and are
  shared across warps; softmax and output accumulation are private per warp.
- Full-warp online softmax: all 32 lanes participate via `__shfl_xor_sync`-based reduction
  (16 rows × 2 lanes/row) — an earlier design left half of every warp idle.
- P conversion (float→half) is fused directly into the softmax's exp/sum loop instead of a
  separate pass over the tile.
- Grid-stride row-tiling: `gridDim.y` is sized to the SM count rather than the exact tile count,
  with each block looping over multiple row-tiles internally, avoiding a "tail effect" where an
  uneven last wave leaves some SMs idle.

**Debugging history:**
- **Shared-memory bank conflicts.** `wmma::load_matrix_sync`/`store_matrix_sync` require `ldm` to
  be a multiple of 8 (half buffers) or 4 (float buffers). The float `S`/`O` buffers were reusing
  the half-buffer padding, landing on a worse bank-alignment (gcd=8) than achievable (gcd=4).
  Splitting into separate `PAD_H`/`PAD_F` values dropped load conflicts 78%→17% and store
  conflicts 71%→15%, confirmed with Nsight Compute before/after profiles.
- **Occupancy floor from shared-memory footprint.** Staging all of `Q` plus a full `O` accumulator
  in shared memory capped occupancy at 1 block/SM. Reduced via `NWARPS` tuning.
- **Tail effect.** An uneven grid vs. SM count left some SMs idle on the final wave; fixed with the
  grid-stride scheme above.
- **A silent, unchecked launch failure.** At dim=128, the dynamic shared-memory opt-in call
  (`cudaFuncSetAttribute`) was failing without being checked — the kernel never launched, and the
  "output" was uninitialized device memory compared against a real reference, producing
  implausible benchmark numbers rather than an error. Fixed by checking every CUDA API return
  value explicitly.
- **A reverted optimization attempt.** Moving `Q` out of shared memory into per-warp registers
  improved occupancy further, but introduced a real correctness bug at multi-head/multi-batch
  configurations that surfaced only under `heads>1`/`batch>1` testing. Reverted to the
  pre-register-Q version rather than ship it unverified.

### `src/flashAttn_tile.cu` — NVIDIA CUDA Tile C++ (`cuda::tiles`)

An experimental, CUDA 13.3+ compiler extension (`__tile_global__`, `--enable-tile`, Ampere/`sm_80`
minimum) where the compiler owns tensor-core codegen and register/shared-memory placement
entirely. The kernel is written in terms of whole-tile operations — `ct::mma`, `ct::reduce_max`,
`ct::exp`, broadcasting arithmetic — with no explicit thread or warp indexing anywhere in the
source.

**Debugging history — a different kind of debugging than the WMMA kernel, since the compiler hides
the implementation:**
- **No official documentation for most of the API surface.** `ct::tile` doesn't support
  `operator[]` or `.size()` at all — an early draft assumed array-like access and had to be
  rewritten entirely around tile-level reduction/broadcast primitives once real compiler errors
  (and, eventually, grepping the installed header directly) revealed the actual API.
- **Multiple rounds of online-softmax loop-nesting bugs**, from computing softmax over a
  partially-accumulated `S` (nested one loop level too deep) to a rescale formula that referenced
  out-of-scope variables, before landing on the correct structure.
- **A required, non-negotiable launch configuration**: exactly 1 thread per block — `32` and `64`
  were both rejected outright by the launch validator. Nsight Compute later confirmed this "1"
  is transparently mapped to a real 32-thread hardware warp internally by the compiler.
- **A ~4ms one-time JIT/warmup cost**, initially misread as "the kernel doesn't scale with
  sequence length" because it dominated wall-clock time at small N. Nsight Systems timeline
  profiling isolated it to `cudaLaunchKernel` itself (not GPU execution time); an untimed warmup
  launch before the timed one resolved it.
- **The actual root-cause correctness bug: a missing `1/√head_dim` score scale before softmax.**
  Silent, plausible-looking wrong output that survived several rounds of hypothesis elimination
  (reduction axis, launch config, block indexing) before being found by building a real reference
  kernel comparison against NVIDIA's own TileGym implementation.
- **Extended to `dim > TILE_K`** (originally hard-capped at `dim=64`) via an additional outer loop
  over head-dimension chunks — correct, but recomputes `S`/softmax redundantly per chunk; a known,
  currently-unoptimized cost.
- **Extended to multi-head/batch**, matching the WMMA kernel's grid convention
  (`blockIdx = {row-tile, head, batch}`) and offsetting base pointers by head/batch stride before
  any tile view is built.
- **Nsight Compute found an 85.76KB/block shared-memory footprint capping occupancy at ~8%** —
  diagnosed, but not directly fixable: the compiler decides register vs. shared-memory placement
  for every tile with no exposed control, unlike register-tile/shared-tile libraries such as
  ThunderKittens. The only indirect lever (`TILE_M`/`TILE_N`/`TILE_K`) was tested at a smaller
  size and made things worse once corrected for a correctness bug the smaller tiles had
  introduced (silently incomplete output, caught by re-running the correctness test rather than
  trusting the faster-looking benchmark).

## Build

```
cmake -B build && cmake --build build
```

`flashAttn_tile` additionally requires CUDA 13.3+ and `sm_80`+:
```
nvcc -O3 --enable-tile -std=c++20 -arch=sm_80 -o build/flashAttn_tile src/flashAttn_tile.cu -lm
```

## Run

All three binaries share the same CLI: `N dim batch num_heads`.

```
./build/naive_attention 2048 128 2 8
./build/flash_attn_forward 2048 128 2 8
./build/flashAttn_tile 2048 128 2 8
```

## Test / benchmark

```
python tests/test_flash_attn_forward.py
python tests/test_flashAttn_tile.py
python benchmarks/benchmark.py
python benchmarks/tiled_benchmark.py
```

## Environment

- GPU: Ada Lovelace (L4, compute capability 8.9) for final verified numbers; WMMA/naive kernels
  also verified earlier on Turing (T4, compute capability 7.5).
- CUDA: 13.3 (required for `flashAttn_tile.cu`'s `cuda::tiles`); 12.8 sufficient for the other two.
- Precision: FP16 inputs/compute, FP32 accumulation and output, throughout.
