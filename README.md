# FlashAttention CUDA Kernel — Tensor Core (WMMA) Optimization

A from-scratch CUDA implementation of the FlashAttention forward pass, progressing from a naive O(N²) baseline to a warp-cooperative tensor-core (WMMA) kernel. Built and profiled on a Tesla T4 (Turing, compute capability 7.5).

## Summary

| Version | Description | Duration @ N=2048, dim=128, heads=8, batch=2 |
|---|---|---|
| Naive | O(N²) attention, no fusion | ~1840 ms |
| Flash (scalar) | Fused online-softmax, tiled, FP32 scalar math | ~63–80 ms |
| Flash (WMMA, final) | Tensor-core QK^T + PV, FP16/FP32 mixed precision | **25.20 ms** |

Verified numerically correct against PyTorch's `scaled_dot_product_attention` (max abs difference: 1.4e-4) across single-head and multi-head/multi-batch configurations.

## Architecture (final version)

- **Tensor cores via WMMA**: both QK^T and PV matmuls run through `nvcuda::wmma` fragment intrinsics (16×16×16 tiles), not scalar loops.
- **Mixed precision**: Q/K/V stored and multiplied in FP16; softmax and accumulation kept in FP32 to avoid precision loss.
- **Warp-cooperative design**: each warp owns a private 16-row slice of the query block (`NWARPS` warps per block, `Br = NWARPS × 16` total rows). K/V tiles are loaded once per block and shared across all warps; softmax and output accumulation are private per warp.
- **Full-warp online softmax**: all 32 lanes of a warp participate via `__shfl_xor_sync`-based reduction (16 rows × 2 lanes/row), replacing an earlier design that left half of every warp idle.
- **Fused P conversion**: the float→half cast for the softmax-weighted P matrix is fused directly into the softmax's exp/sum loop, avoiding a separate pass over the tile.
- **Grid-stride row-tiling**: `gridDim.y` is sized to the SM count rather than the exact tile count, with each block looping over multiple row-tiles internally — avoids the "tail effect" where an uneven last wave leaves some SMs idle.

## Optimization history

1. **Register spilling** — `q_register`/`acc` arrays were indexed by a runtime variable, preventing the compiler from allocating real registers (confirmed via `ptxas -v`: 18 registers/thread, heavy local-memory spill). Fixed by templating the kernel on head dimension (`template<int DIM>`), enabling compile-time-constant loop bounds — registers/thread rose to 158 (dim=64, zero spill) and 255 (dim=128, near-hardware-ceiling with minor residual spill).
2. **Grid undersizing** — initial launch config produced only 0.5 "waves" of GPU occupancy (Nsight Compute flagged "Small Grid"). Fixed via `Br`/`Bc` tuning, more than tripling kernel duration (123ms → 41ms) at a fixed input size.
3. **Tensor cores (WMMA)** — replaced scalar dot-product loops with warp-cooperative `wmma::mma_sync` calls for both QK^T and PV.
4. **Shared memory bank conflicts** — `wmma::load_matrix_sync`/`store_matrix_sync` require `ldm` to be a multiple of 8 (half) or 4 (float). Reusing a single padding value for both buffer types left the float S/O buffers at a suboptimal bank-alignment (gcd=8 instead of achievable gcd=4). Splitting into `PAD_H=8` (half buffers) and `PAD_F=4` (float buffers) reduced shared-memory bank conflicts from 78.5%→15.75% (load) and 71.11%→13.73% (store), confirmed via Nsight Compute before/after profiles.
5. **Tail effect** — grid-stride row-tiling, sized to SM count, to avoid an uneven final wave.

## Known remaining bottleneck

At dim=128, the block's shared-memory footprint (Q + K + V + S + P + O tiles, all padded) limits occupancy to **1 block/SM** (Achieved Occupancy ≈ 12–25% depending on run), which caps overall throughput — confirmed via Nsight Compute's Block Limit / Occupancy analysis. An attempted fix (loading Q directly into per-warp registers instead of shared memory, freeing enough space for 2 blocks/SM) was implemented but failed numerical verification (max diff ~0.03–0.04, vs. ~1e-4 baseline) due to a bug in the padded Q memory layout, and was reverted rather than shipped unverified.

**Next steps**, in priority order:
- Debug the register-resident Q layout (likely a stride/offset mismatch between the host-side padded copy and the per-warp fragment load)
- Reduce `Bc` below `WMMA_N=16` with boundary-masking logic, to shrink the K/V tile footprint directly
- Address remaining shared-memory bank conflicts in the softmax/rescale access pattern (row-major access across 16 threads simultaneously)

## Correctness verification

All results are validated against PyTorch's `torch.nn.functional.scaled_dot_product_attention` on the same random Q/K/V inputs, including at the multi-head/multi-batch configuration (N=2048, dim=128, batch=2, heads=8) that surfaced two distinct silent-failure bugs during development (a shared-memory allocation exceeding the 64KB per-block limit on cc7.5, and a padded Q memory layout bug) — both caught specifically because performance numbers looked implausibly good and prompted a correctness re-check rather than being taken at face value.

## Environment

- GPU: Tesla T4 (Turing, compute capability 7.5)
- CUDA: 12.8
- Precision: FP16 inputs/compute, FP32 accumulation and output
