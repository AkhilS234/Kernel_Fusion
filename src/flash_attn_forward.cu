#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

#define WMMA_M   16
#define WMMA_N   16
#define WMMA_K   16
#define Bc       16                // K/V rows per tile = one WMMA N tile
#define PAD      8                 // element padding on every 2D smem row to dodge bank conflicts
#define MAX_DIM  128
#define FULL_MASK 0xffffffffu

/*
 * NWARPS warps (NWARPS*32 threads) per block. Each warp owns a private
 * WMMA_M=16 row-group of Q, so the block covers Br = NWARPS*16 rows.
 * NWARPS is a template parameter (not a fixed constant) because shared
 * memory scales with Br*(dim+PAD); at dim=128, NWARPS=4 (Br=64) overflows
 * the 64KB opt-in limit on cc7.5 (T4), so callers must pick a smaller
 * NWARPS there — see the host dispatch in main().
 *
 * S [16 x Bc] = Q_warp [16 x DIM] * K^T [DIM x Bc]   via WMMA (Tensor Cores)
 * O [16 x DIM] += P [16 x Bc] * V [Bc x DIM]         via WMMA
 *
 * K/V tiles are shared block-wide (loaded once per k_tile, 2 __syncthreads()
 * total per iteration). Everything after that — softmax, P conversion, the O
 * update — is private per warp and only needs __syncwarp(), since each warp
 * writes exclusively to its own S/P/O/m/l shared-memory region.
 *
 * Softmax uses all 32 lanes of a warp for its 16x16 tile: lanes are paired
 * two-per-row (row = lane & 15, half = lane >> 4), each half handling 8
 * columns, combined via __shfl_xor_sync. This replaces the old `tx < Br`
 * pattern that left half of every warp idle.
 */

template <int DIM, int NWARPS>
__global__ void flash_attention(
    const __half *matrix_Q,
    const __half *matrix_K,
    const __half *matrix_V,
    float        *matrix_O,
    int N, int num_heads)
{
    constexpr int Br  = NWARPS * WMMA_M;
    constexpr int QLD = DIM + PAD;
    constexpr int KLD = DIM + PAD;
    constexpr int VLD = DIM + PAD;
    constexpr int OLD = DIM + PAD;
    constexpr int SLD = Bc + PAD;
    constexpr int PLD = Bc + PAD;

    int head_idx  = blockIdx.x;
    int batch_idx = blockIdx.z;
    size_t head_stride  = (size_t)N * DIM;
    size_t batch_stride = (size_t)num_heads * N * DIM;

    const __half *Q  = matrix_Q + batch_idx * batch_stride + head_idx * head_stride;
    const __half *K  = matrix_K + batch_idx * batch_stride + head_idx * head_stride;
    const __half *V  = matrix_V + batch_idx * batch_stride + head_idx * head_stride;
    float        *Og = matrix_O + batch_idx * batch_stride + head_idx * head_stride;

    // Shared memory layout (byte-addressed, manually partitioned):
    //   smem_Q [Br x QLD]            __half   Q tile, loaded once per block
    //   smem_K [Bc x KLD]            __half   K tile, refreshed each outer iteration (block-shared)
    //   smem_V [Bc x VLD]            __half   V tile, refreshed each outer iteration (block-shared)
    //   smem_S [NWARPS x 16 x SLD]   float    raw scores / P weights, private per warp
    //   smem_P [NWARPS x 16 x PLD]   __half   P weights converted for WMMA, private per warp
    //   smem_O [Br x OLD]            float    running output accumulator
    //   smem_m [Br], smem_l [Br]     float    running per-row softmax stats
    extern __shared__ char smem_raw[];
    __half *smem_Q = (__half *)smem_raw;
    __half *smem_K = smem_Q + Br * QLD;
    __half *smem_V = smem_K + Bc * KLD;
    float  *smem_S = (float *)(smem_V + Bc * VLD);
    __half *smem_P = (__half *)(smem_S + NWARPS * WMMA_M * SLD);
    float  *smem_O = (float *)(smem_P + NWARPS * WMMA_M * PLD);
    float  *smem_m = smem_O + Br * OLD;
    float  *smem_l = smem_m + Br;

    int tx            = threadIdx.x;              // 0..127
    int warp_id        = tx / 32;                  // 0..NWARPS-1
    int lane           = tx % 32;                  // 0..31
    int warp_row_base  = warp_id * WMMA_M;          // local row offset within Br
    int q_row_base     = blockIdx.y * Br;

    float  *smem_S_w = smem_S + warp_id * WMMA_M * SLD;
    __half *smem_P_w = smem_P + warp_id * WMMA_M * PLD;
    float  *smem_O_w = smem_O + warp_row_base * OLD;
    float  *smem_m_w = smem_m + warp_row_base;
    float  *smem_l_w = smem_l + warp_row_base;

    // Load Q tile [Br x DIM] into padded smem, cooperatively across the whole block
    for (int idx = tx; idx < Br * DIM; idx += blockDim.x) {
        int row = idx / DIM, col = idx % DIM;
        int g_row = q_row_base + row;
        smem_Q[row * QLD + col] = (g_row < N) ? Q[g_row * DIM + col] : __float2half(0.0f);
    }
    // Init output accumulator and running softmax stats
    for (int idx = tx; idx < Br * DIM; idx += blockDim.x) {
        int row = idx / DIM, col = idx % DIM;
        smem_O[row * OLD + col] = 0.0f;
    }
    if (tx < Br) {
        smem_m[tx] = -INFINITY;
        smem_l[tx] = 0.0f;
    }
    __syncthreads();

    // Outer loop: stream K/V tiles through shared memory
    for (int k_tile = 0; k_tile < N; k_tile += Bc) {

        // Load K/V tiles [Bc x DIM] into padded smem, block-shared across all warps
        for (int idx = tx; idx < Bc * DIM; idx += blockDim.x) {
            int row = idx / DIM, col = idx % DIM;
            int g_row = k_tile + row;
            __half kv_k = (g_row < N) ? K[g_row * DIM + col] : __float2half(0.0f);
            __half kv_v = (g_row < N) ? V[g_row * DIM + col] : __float2half(0.0f);
            smem_K[row * KLD + col] = kv_k;
            smem_V[row * VLD + col] = kv_v;
        }
        __syncthreads();  // K/V load visible to all warps before any warp reads it

        // S [16 x Bc] = Q_warp [16 x DIM] * K^T [DIM x Bc]  via Tensor Cores
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
        wmma::fill_fragment(S_frag, 0.0f);

        for (int k = 0; k < DIM; k += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> Q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> K_frag;
            wmma::load_matrix_sync(Q_frag, smem_Q + warp_row_base * QLD + k, QLD);
            wmma::load_matrix_sync(K_frag, smem_K + k, KLD);
            wmma::mma_sync(S_frag, Q_frag, K_frag, S_frag);
        }

        // Store S fragment to this warp's private smem [16 x Bc] row-major
        wmma::store_matrix_sync(smem_S_w, S_frag, SLD, wmma::mem_row_major);
        __syncwarp();

        // Online softmax, full-warp: lanes paired 2-per-row (16 rows * 2 halves = 32 lanes)
        {
            int row      = lane & 15;
            int half     = lane >> 4;
            int col0     = half * 8;
            float *Srow  = smem_S_w + row * SLD;

            float local_max = -INFINITY;
            for (int c = col0; c < col0 + 8; c++) {
                Srow[c] /= sqrtf((float)DIM);
                local_max = max(local_max, Srow[c]);
            }
            float other_max = __shfl_xor_sync(FULL_MASK, local_max, 16);
            float row_max    = max(local_max, other_max);

            float m_old = smem_m_w[row];
            float m_new = max(m_old, row_max);
            float rescale = expf(m_old - m_new);

            if (half == 0) smem_l_w[row] *= rescale;
            for (int d = half * (DIM / 2); d < half * (DIM / 2) + DIM / 2; d++)
                smem_O_w[row * OLD + d] *= rescale;

            float local_sum = 0.0f;
            for (int c = col0; c < col0 + 8; c++) {
                float p = expf(Srow[c] - m_new);
                Srow[c] = p;
                local_sum += p;
            }
            float other_sum = __shfl_xor_sync(FULL_MASK, local_sum, 16);
            if (half == 0) {
                smem_l_w[row] += local_sum + other_sum;
                smem_m_w[row] = m_new;
            }
        }
        __syncwarp();

        // Convert P (float in smem_S_w) to __half in smem_P_w, full warp
        for (int idx = lane; idx < WMMA_M * Bc; idx += 32) {
            int r = idx / Bc, c = idx % Bc;
            smem_P_w[r * PLD + c] = __float2half(smem_S_w[r * SLD + c]);
        }
        __syncwarp();

        // O [16 x DIM] += P [16 x Bc] * V [Bc x DIM]  via Tensor Cores
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> P_frag;
        wmma::load_matrix_sync(P_frag, smem_P_w, PLD);

        for (int d = 0; d < DIM; d += WMMA_N) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> O_frag;
            wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> V_frag;
            wmma::load_matrix_sync(O_frag, smem_O_w + d, OLD, wmma::mem_row_major);
            wmma::load_matrix_sync(V_frag, smem_V + d, VLD);
            wmma::mma_sync(O_frag, P_frag, V_frag, O_frag);
            wmma::store_matrix_sync(smem_O_w + d, O_frag, OLD, wmma::mem_row_major);
        }
        __syncwarp();
    }

    // Normalize by l and write to global memory — full warp, 16 rows x DIM cols
    for (int idx = lane; idx < WMMA_M * DIM; idx += 32) {
        int r = idx / DIM, c = idx % DIM;
        int g_row = q_row_base + warp_row_base + r;
        if (g_row < N) {
            float l_inv = 1.0f / smem_l_w[r];
            Og[g_row * DIM + c] = smem_O_w[r * OLD + c] * l_inv;
        }
    }
}

// Computes smem size for (DIM, NWARPS), sets the dynamic-smem opt-in, launches,
// and checks every CUDA call so a budget overrun fails loud instead of silently
// (a >64KB request on cc7.5 makes cudaFuncSetAttribute fail, which used to leave
// the kernel launch silently skipped and matrix_O full of uninitialized memory).
template <int DIM, int NWARPS>
bool launch_flash_attention(const __half *dQ, const __half *dK, const __half *dV,
                             float *dO, int N, int num_heads, int batch)
{
    constexpr int Br = NWARPS * WMMA_M;
    size_t smem_bytes = (size_t)Br * (DIM + PAD) * sizeof(__half)              // Q
                      + (size_t)Bc * (DIM + PAD) * sizeof(__half)              // K
                      + (size_t)Bc * (DIM + PAD) * sizeof(__half)              // V
                      + (size_t)NWARPS * WMMA_M * (Bc + PAD) * sizeof(float)   // S
                      + (size_t)NWARPS * WMMA_M * (Bc + PAD) * sizeof(__half)  // P
                      + (size_t)Br * (DIM + PAD) * sizeof(float)               // O
                      + (size_t)Br * 2 * sizeof(float);                        // m, l

    dim3 blockDim(NWARPS * 32);
    dim3 gridDim(num_heads, (N + Br - 1) / Br, batch);

    cudaError_t err = cudaFuncSetAttribute(flash_attention<DIM, NWARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    if (err != cudaSuccess) {
        printf("cudaFuncSetAttribute failed: %s (requested %zu bytes, DIM=%d NWARPS=%d)\n",
               cudaGetErrorString(err), smem_bytes, DIM, NWARPS);
        return false;
    }

    flash_attention<DIM, NWARPS><<<gridDim, blockDim, smem_bytes>>>(dQ, dK, dV, dO, N, num_heads);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("kernel execution failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {

    int N         = argc > 1 ? atoi(argv[1]) : 1024;
    int dim       = argc > 2 ? atoi(argv[2]) : 64;
    int batch     = argc > 3 ? atoi(argv[3]) : 1;
    int num_heads = argc > 4 ? atoi(argv[4]) : 1;

    if (dim > MAX_DIM) {
        printf("Error: dim %d exceeds MAX_DIM %d\n", dim, MAX_DIM);
        return 1;
    }
    if (dim % WMMA_K != 0) {
        printf("Error: dim %d must be a multiple of WMMA_K=%d\n", dim, WMMA_K);
        return 1;
    }

    size_t qkv_elems = (size_t)batch * num_heads * N * dim;

    float *host_query  = new float[qkv_elems];
    float *host_key    = new float[qkv_elems];
    float *host_value  = new float[qkv_elems];
    float *host_output = new float[qkv_elems];

    FILE *fq = fopen("outputs/input_Q.bin", "rb");
    FILE *fk = fopen("outputs/input_K.bin", "rb");
    FILE *fv = fopen("outputs/input_V.bin", "rb");
    if (!fq || !fk || !fv) {
        printf("Error: could not open input files in outputs/\n");
        return 1;
    }
    fread(host_query, sizeof(float), qkv_elems, fq);
    fread(host_key,   sizeof(float), qkv_elems, fk);
    fread(host_value, sizeof(float), qkv_elems, fv);
    fclose(fq); fclose(fk); fclose(fv);

    // Convert FP32 -> FP16 for Q, K, V
    __half *host_query_fp16 = new __half[qkv_elems];
    __half *host_key_fp16   = new __half[qkv_elems];
    __half *host_value_fp16 = new __half[qkv_elems];
    for (size_t i = 0; i < qkv_elems; i++) {
        host_query_fp16[i] = __float2half(host_query[i]);
        host_key_fp16[i]   = __float2half(host_key[i]);
        host_value_fp16[i] = __float2half(host_value[i]);
    }

    __half *d_matrixQ, *d_matrixK, *d_matrixV;
    float  *d_matrixO;
    cudaMalloc(&d_matrixQ, qkv_elems * sizeof(__half));
    cudaMalloc(&d_matrixK, qkv_elems * sizeof(__half));
    cudaMalloc(&d_matrixV, qkv_elems * sizeof(__half));
    cudaMalloc(&d_matrixO, qkv_elems * sizeof(float));

    cudaMemcpy(d_matrixQ, host_query_fp16, qkv_elems * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixK, host_key_fp16,   qkv_elems * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixV, host_value_fp16, qkv_elems * sizeof(__half), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // NWARPS=4 (Br=64) at dim=64 fits well under the 64KB cc7.5 opt-in limit.
    // At dim=128 the same NWARPS would need ~69KB, so drop to NWARPS=2 (Br=32),
    // which lands at ~41KB — comfortably in budget.
    bool ok;
    if (dim == 64) {
        ok = launch_flash_attention<64, 4>(d_matrixQ, d_matrixK, d_matrixV, d_matrixO, N, num_heads, batch);
    } else if (dim == 128) {
        ok = launch_flash_attention<128, 2>(d_matrixQ, d_matrixK, d_matrixV, d_matrixO, N, num_heads, batch);
    } else {
        printf("Error: unsupported dim %d\n", dim);
        return 1;
    }
    if (!ok) {
        cudaFree(d_matrixQ); cudaFree(d_matrixK); cudaFree(d_matrixV); cudaFree(d_matrixO);
        return 1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken: %f ms\n", ms);

    cudaMemcpy(host_output, d_matrixO, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost);

    float bytes_accessed = 3.0f * qkv_elems * sizeof(__half)  // Q, K, V (FP16)
                         + 1.0f * qkv_elems * sizeof(float);  // O       (FP32)
    float bandwidth_gb = (bytes_accessed / (ms / 1000.0f)) / 1e9f;
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gb); 
    printf("HBM accessed: %.4f GB\n", bytes_accessed / 1e9f);

    FILE *fo = fopen("outputs/output_S.bin", "wb");
    fwrite(host_output, sizeof(float), qkv_elems, fo);
    fclose(fo);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_matrixQ); cudaFree(d_matrixK); cudaFree(d_matrixV); cudaFree(d_matrixO);
    delete[] host_query;      delete[] host_key;      delete[] host_value;      delete[] host_output;
    delete[] host_query_fp16; delete[] host_key_fp16; delete[] host_value_fp16;

    return 0;
}
