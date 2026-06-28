#include <stdio.h>
#include <math.h>

#define TILE_SIZE 64

__global__ void flash_attention(float *Q, float *K, float *V, float *matrix_O, int dim, int N) {

    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_K[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_V[TILE_SIZE][TILE_SIZE];

    int q_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int k_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    if (q_row < N ) {
        for (int d = threadIdx.x;  d < dim; d += TILE_SIZE) {
            tile_Q[threadIdx.y][d] = Q[q_row * dim + d];
        }
    }
    __syncthreads();

    float m_old = -INFINITY;
    float l_old = 0.0f;
    float acc[64] = {0.0f};

    for (int k_tile = 0; k_tile < N; k_tile += TILE_SIZE) {
        int k_row = k_tile + threadIdx.y;
        if (k_row < N) {
            for (int d = threadIdx.x; d < dim; d += TILE_SIZE) {
                tile_K[threadIdx.y][d] = K[k_row * dim + d];
                tile_V[threadIdx.y][d] = V[k_row * dim + d];
            }
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            float val = 0.0f;
            for (int i = 0; i < dim; i++) {
                val += tile_Q[threadIdx.y][i] * tile_K[k][i];
            }

            val /= sqrtf((float)dim);

            float m_new = max(m_old, val);
            float l_new = expf(m_old - m_new) * l_old + expf(val - m_new);
            float p = expf(val - m_new);

            for (int d=0; d < dim; d++) {
                acc[d] = acc[d] * expf(m_old - m_new) + p * tile_V[k][d];
            }

            m_old = m_new;
            l_old = l_new;
        }

        __syncthreads();
    }

    if (q_row < N) {
        for (int d = 0; d < dim; d++) {
            matrix_O[q_row * dim + d] = acc[d] / l_old;
        }
    }
}

int main() {

    int N = 1024;
    int dim = 64;

    float *host_query = new float[N * dim];
    float *host_key = new float[N * dim];
    float *host_value = new float[N * dim];
    float *host_output = new float[N * dim];

    FILE *fq = fopen("outputs/input_Q.bin", "rb");
    FILE *fk = fopen("outputs/input_K.bin", "rb");
    FILE *fv = fopen("outputs/input_V.bin", "rb");
    if (!fq || !fk || !fv) {
        printf("Error: could not open input files in outputs/\n");
        return 1;
    }
    fread(host_query, sizeof(float), N * dim, fq);
    fread(host_key, sizeof(float), N * dim, fk);
    fread(host_value, sizeof(float), N * dim, fv);
    fclose(fq);
    fclose(fk);
    fclose(fv);

    float *d_matrixQ;
    float *d_matrixK;
    float *d_matrixV;

    float *d_matrixS;
    float *d_matrixP;
    float *d_matrixO;

    cudaMalloc(&d_matrixQ, N * dim * sizeof(float));
    cudaMalloc(&d_matrixK, N * dim * sizeof(float));
    cudaMalloc(&d_matrixV, N * dim * sizeof(float));
    cudaMalloc(&d_matrixO, N * dim * sizeof(float));

    cudaMemcpy(d_matrixQ, host_query, N * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixK, host_key, N * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixV, host_value, N * dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32); 
    dim3 gridDim(1, 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    flash_attention<<<gridDim, blockDim>>>(d_matrixQ, d_matrixK, d_matrixV, d_matrixO, dim, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken: %f ms\n", ms);

    cudaMemcpy(host_output, d_matrixO, N * dim * sizeof(float), cudaMemcpyDeviceToHost);
    FILE *fo = fopen("outputs/flash_attn_output.bin", "wb");
    fwrite(host_output, sizeof(float), N * dim, fo);
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