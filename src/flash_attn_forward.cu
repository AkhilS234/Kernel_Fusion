#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 32
#define MAX_DIM 128

#define Br 64
#define Bc 16

template <int DIM>

__global__ void flash_attention(float *matrix_Q, float *matrix_K, float *matrix_V, float *matrix_O, int dim, int N, int num_heads) {

    int head_idx  = blockIdx.x;
    int batch_idx = blockIdx.z;
    size_t head_stride  = (size_t)N * dim;
    size_t batch_stride = (size_t)num_heads * N * dim;
    const float *Q = matrix_Q + batch_idx * batch_stride + head_idx * head_stride;
    const float *K = matrix_K + batch_idx * batch_stride + head_idx * head_stride;
    const float *V = matrix_V + batch_idx * batch_stride + head_idx * head_stride;
    float *matrix_O_b = matrix_O + batch_idx * batch_stride + head_idx * head_stride;

    extern __shared__ float smem[];
    float *tile_K = smem;
    float *tile_V = smem + Bc * dim;

    int q_row = blockIdx.y * Br + threadIdx.x;

    float q_register[DIM];
    if (q_row < N) {
        for (int d = 0; d < DIM; d++) {
            q_register[d] = Q[q_row * dim + d];
        }
    }

    __syncthreads();

    float m_old = -INFINITY;
    float l_old = 0.0f;
    float acc[DIM] = {0.0f};

    for (int k_tile = 0; k_tile < N; k_tile += Bc) {
        
        for (int idx = threadIdx.x; idx < Bc * dim; idx += blockDim.x) {
            int row = idx / dim;
            int col = idx % dim;
            int global_row = k_tile + row;
            tile_K[row * dim + col] = (global_row < N) ? K[global_row * dim + col] : 0.0f;
            tile_V[row * dim + col] = (global_row < N) ? V[global_row * dim + col] : 0.0f;
        }
        __syncthreads();

        int tile_rows = min(Bc, N - k_tile);
        for (int k = 0; k < tile_rows; k++) {
            float val = 0.0f;
            for (int i = 0; i < DIM; i++) {
                val += q_register[i] * tile_K[k * DIM + i];
            }

            val /= sqrtf((float)DIM);

            float m_new = max(m_old, val);
            float l_new = expf(m_old - m_new) * l_old + expf(val - m_new);
            float p = expf(val - m_new);

            for (int d = 0; d < DIM; d++) {
                acc[d] = acc[d] * expf(m_old - m_new) + p * tile_V[k * DIM + d];
            }

            m_old = m_new;
            l_old = l_new;
        }

        __syncthreads();
    }

    if (q_row < N) {
        for (int d = 0; d < DIM; d++) {
            matrix_O_b[q_row * DIM + d] = acc[d] / l_old;
        }
    }
}

int main(int argc, char **argv) {

    int N         = argc > 1 ? atoi(argv[1]) : 1024;
    int dim       = argc > 2 ? atoi(argv[2]) : 64;
    int batch     = argc > 3 ? atoi(argv[3]) : 1;
    int num_heads = argc > 4 ? atoi(argv[4]) : 1;

    if (dim > MAX_DIM) {
        printf("Error: dim %d exceeds MAX_DIM %d supported by this kernel\n", dim, MAX_DIM);
        return 1;
    }

    size_t qkv_elems = (size_t)batch * num_heads * N * dim;

    float *host_query = new float[qkv_elems];
    float *host_key = new float[qkv_elems];
    float *host_value = new float[qkv_elems];
    float *host_output = new float[qkv_elems];

    FILE *fq = fopen("outputs/input_Q.bin", "rb");
    FILE *fk = fopen("outputs/input_K.bin", "rb");
    FILE *fv = fopen("outputs/input_V.bin", "rb");
    if (!fq || !fk || !fv) {
        printf("Error: could not open input files in outputs/\n");
        return 1;
    }
    fread(host_query, sizeof(float), qkv_elems, fq);
    fread(host_key, sizeof(float), qkv_elems, fk);
    fread(host_value, sizeof(float), qkv_elems, fv);
    fclose(fq);
    fclose(fk);
    fclose(fv);

    float *d_matrixQ;
    float *d_matrixK;
    float *d_matrixV;
    float *d_matrixO;

    cudaMalloc(&d_matrixQ, qkv_elems * sizeof(float));
    cudaMalloc(&d_matrixK, qkv_elems * sizeof(float));
    cudaMalloc(&d_matrixV, qkv_elems * sizeof(float));
    cudaMalloc(&d_matrixO, qkv_elems * sizeof(float));

    cudaMemcpy(d_matrixQ, host_query, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixK, host_key, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixV, host_value, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(Br);
    dim3 gridDim(num_heads, (N + Br - 1) / Br, batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    size_t smem_bytes = 2 * Bc * dim * sizeof(float);

    if (dim == 64) flash_attention<64><<<gridDim, blockDim, smem_bytes>>>(d_matrixQ, d_matrixK, d_matrixV, d_matrixO, dim, N, num_heads);
    if (dim == 128) flash_attention<128><<<gridDim, blockDim, smem_bytes>>>(d_matrixQ, d_matrixK, d_matrixV, d_matrixO, dim, N, num_heads);

    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken: %f ms\n", ms);

    cudaMemcpy(host_output, d_matrixO, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost);

    float bytes_accessed = 4.0f * batch * num_heads * N * dim * sizeof(float); // Q, K, V read once, O written once
    float bandwidth_gb = (bytes_accessed / (ms / 1000.0f)) / 1e9f;
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gb);
    printf("HBM accessed: %.4f GB\n", bytes_accessed / 1e9f);

    FILE *fo = fopen("outputs/output_S.bin", "wb");
    fwrite(host_output, sizeof(float), qkv_elems, fo);
    fclose(fo);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_matrixQ);
    cudaFree(d_matrixK);
    cudaFree(d_matrixV);
    cudaFree(d_matrixO);
    delete[] host_query;
    delete[] host_key;
    delete[] host_value;
    delete[] host_output;

    return 0;
}

/*
Implements the fused FlashAttention forward pass. One thread is permanently assigned to one row of Q.
Each thread loads its Q row once, then sweeps through all of K and V — fetched in TILE_SIZE-sized chunks from HBM
into shared memory to minimize memory traffic. Within each tile, the thread sequentially processes every row: computing
the scaled dot product, updating the running softmax statistics (m, l) via the online-softmax merge formula, and
accumulating the weighted V contribution into a private output buffer (acc), rescaling previous contributions whenever
the running max changes. S and P are never materialized to HBM — they exist only transiently in registers. After all of
K/V has been processed, each thread normalizes its accumulator by the final l and writes one row directly to O. Net effect:
replaces the naive kernel's three separate HBM round trips (S, P, O) with a single fused pass that touches HBM only to read
Q/K/V once and write O once.
*/
