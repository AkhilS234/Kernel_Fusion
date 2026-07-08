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
#define Br       16    // Q rows per block  = one WMMA M tile
#define Bc       16    // K/V rows per tile = one WMMA N tile
#define MAX_DIM  128

/*
 * One warp (32 threads) per block. Each block owns Br=16 consecutive Q rows.
 *
 * S [Br x Bc] = Q [Br x DIM] * K^T [DIM x Bc]  via WMMA (Tensor Cores)
 *   - Q fragment:  matrix_a, row_major,  stride=DIM
 *   - K fragment:  matrix_b, col_major,  stride=DIM  -> loads K^T implicitly
 *   - Tiled over the DIM dimension in steps of WMMA_K=16
 *
 * O [Br x DIM] += P [Br x Bc] * V [Bc x DIM]  via WMMA
 *   - P fragment:  matrix_a, row_major,  stride=Bc
 *   - V fragment:  matrix_b, row_major,  stride=DIM
 *   - Tiled over the DIM output dimension in steps of WMMA_N=16
 *
 * Online softmax (scalar, threads 0..Br-1) is applied to smem_S between the
 * two WMMA phases; S/P materialization stays in shared memory, never HBM.
 */

template <int DIM>
__global__ void flash_attention(
    const __half *matrix_Q,
    const __half *matrix_K,
    const __half *matrix_V,
    float        *matrix_O,
    int N, int num_heads)
{
    int head_idx  = blockIdx.x;
    int batch_idx = blockIdx.z;
    size_t head_stride  = (size_t)N * DIM;
    size_t batch_stride = (size_t)num_heads * N * DIM;

    const __half *Q  = matrix_Q + batch_idx * batch_stride + head_idx * head_stride;
    const __half *K  = matrix_K + batch_idx * batch_stride + head_idx * head_stride;
    const __half *V  = matrix_V + batch_idx * batch_stride + head_idx * head_stride;
    float        *Og = matrix_O + batch_idx * batch_stride + head_idx * head_stride;

    // Shared memory layout (byte-addressed, manually partitioned):
    //   smem_Q  [Br  x DIM]  __half   Q tile, loaded once per block
    //   smem_K  [Bc  x DIM]  __half   K tile, refreshed each outer iteration
    //   smem_V  [Bc  x DIM]  __half   V tile, refreshed each outer iteration
    //   smem_S  [Br  x Bc ]  float    raw attention scores (overwritten with P weights)
    //   smem_P  [Br  x Bc ]  __half   softmax weights converted for WMMA
    //   smem_O  [Br  x DIM]  float    running output accumulator
    //   smem_m  [Br ]         float    running per-row max
    //   smem_l  [Br ]         float    running per-row softmax denominator
    extern __shared__ char smem_raw[];
    __half *smem_Q = (__half *)smem_raw;
    __half *smem_K = smem_Q + Br * DIM;
    __half *smem_V = smem_K + Bc * DIM;
    float  *smem_S = (float *)(smem_V + Bc * DIM);   // float-aligned: offset=(Br+2Bc)*DIM*2
    __half *smem_P = (__half *)(smem_S + Br * Bc);
    float  *smem_O = (float *)(smem_P + Br * Bc);
    float  *smem_m = smem_O + Br * DIM;
    float  *smem_l = smem_m + Br;

    int tx         = threadIdx.x;
    int q_row_base = blockIdx.y * Br;

    // Load Q tile [Br x DIM]
    for (int idx = tx; idx < Br * DIM; idx += blockDim.x) {
        int row = idx / DIM, col = idx % DIM;
        int g_row = q_row_base + row;
        smem_Q[idx] = (g_row < N) ? Q[g_row * DIM + col] : __float2half(0.0f);
    }
    // Init output accumulator and running softmax stats
    for (int idx = tx; idx < Br * DIM; idx += blockDim.x)
        smem_O[idx] = 0.0f;
    if (tx < Br) {
        smem_m[tx] = -INFINITY;
        smem_l[tx] = 0.0f;
    }
    __syncthreads();

    // Outer loop: stream K/V tiles through shared memory
    for (int k_tile = 0; k_tile < N; k_tile += Bc) {

        // Load K and V tiles [Bc x DIM]
        for (int idx = tx; idx < Bc * DIM; idx += blockDim.x) {
            int row = idx / DIM, col = idx % DIM;
            int g_row = k_tile + row;
            smem_K[idx] = (g_row < N) ? K[g_row * DIM + col] : __float2half(0.0f);
            smem_V[idx] = (g_row < N) ? V[g_row * DIM + col] : __float2half(0.0f);
        }
        __syncthreads();

        // S [Br x Bc] = Q [Br x DIM] * K^T [DIM x Bc]  via Tensor Cores
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
        wmma::fill_fragment(S_frag, 0.0f);

        for (int k = 0; k < DIM; k += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> Q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> K_frag;
            // Q columns k..k+15, row-stride=DIM
            wmma::load_matrix_sync(Q_frag, smem_Q + k, DIM);
            // K col_major + stride=DIM: element (r,c) = smem_K[k + c*DIM + r] = K[c][k+r] = K^T[r][c]
            wmma::load_matrix_sync(K_frag, smem_K + k, DIM);
            wmma::mma_sync(S_frag, Q_frag, K_frag, S_frag);
        }

        // Store S fragment to smem [Br x Bc] row-major
        wmma::store_matrix_sync(smem_S, S_frag, Bc, wmma::mem_row_major);
        __syncthreads();

        // Online softmax — one thread per Q row (threads 0..Br-1)
        if (tx < Br) {
            // Scale and find new row max
            float m_new = smem_m[tx];
            for (int j = 0; j < Bc; j++) {
                smem_S[tx * Bc + j] /= sqrtf((float)DIM);
                m_new = max(m_new, smem_S[tx * Bc + j]);
            }
            // Rescale running O and l by exp(m_old - m_new)
            float rescale = expf(smem_m[tx] - m_new);
            smem_l[tx] *= rescale;
            for (int d = 0; d < DIM; d++)
                smem_O[tx * DIM + d] *= rescale;
            // Compute unnormalized weights P = exp(s - m_new), accumulate into l
            for (int j = 0; j < Bc; j++) {
                float p = expf(smem_S[tx * Bc + j] - m_new);
                smem_S[tx * Bc + j] = p;
                smem_l[tx] += p;
            }
            smem_m[tx] = m_new;
        }
        __syncthreads();

        // Convert P (float in smem_S) to __half in smem_P for WMMA
        for (int idx = tx; idx < Br * Bc; idx += blockDim.x)
            smem_P[idx] = __float2half(smem_S[idx]);
        __syncthreads();

        // O [Br x DIM] += P [Br x Bc] * V [Bc x DIM]  via Tensor Cores
        // Load P once; reuse across all DIM column slices
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> P_frag;
        wmma::load_matrix_sync(P_frag, smem_P, Bc);

        for (int d = 0; d < DIM; d += WMMA_N) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> O_frag;
            wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> V_frag;
            // Load existing O slice to accumulate into it
            wmma::load_matrix_sync(O_frag, smem_O + d, DIM, wmma::mem_row_major);
            // V columns d..d+15, row-stride=DIM
            wmma::load_matrix_sync(V_frag, smem_V + d, DIM);
            wmma::mma_sync(O_frag, P_frag, V_frag, O_frag);
            wmma::store_matrix_sync(smem_O + d, O_frag, DIM, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Normalize by l and write to global memory
    if (tx < Br) {
        int g_row = q_row_base + tx;
        if (g_row < N) {
            float l_inv = 1.0f / smem_l[tx];
            for (int d = 0; d < DIM; d++)
                Og[g_row * DIM + d] = smem_O[tx * DIM + d] * l_inv;
        }
    }
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

    // smem: Q+K+V (fp16) + S+P (float+fp16) + O accumulator (float) + m+l (float)
    size_t smem_bytes = (size_t)(Br + 2*Bc) * dim * sizeof(__half)
                      + (size_t)Br * Bc * (sizeof(float) + sizeof(__half))
                      + (size_t)Br * dim * sizeof(float)
                      + (size_t)Br * 2 * sizeof(float);

    dim3 blockDim(32);   // one warp per block
    dim3 gridDim(num_heads, (N + Br - 1) / Br, batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (dim == 64)
        flash_attention<64><<<gridDim, blockDim, smem_bytes>>>(
            d_matrixQ, d_matrixK, d_matrixV, d_matrixO, N, num_heads);
    else if (dim == 128)
        flash_attention<128><<<gridDim, blockDim, smem_bytes>>>(
            d_matrixQ, d_matrixK, d_matrixV, d_matrixO, N, num_heads);

    cudaGetLastError();
    cudaDeviceSynchronize();

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
