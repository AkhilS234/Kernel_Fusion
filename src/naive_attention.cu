#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

__global__ void attention_scores(float *matrix_Q, float *matrix_K, float *matrix_S, int dim, int N) {

    int batch_idx = blockIdx.z;
    const float *Q = matrix_Q + (size_t)batch_idx * N * dim;
    const float *K = matrix_K + (size_t)batch_idx * N * dim;
    float *S = matrix_S + (size_t)batch_idx * N * N;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    if (row < N && col < N) {
        for (int i = 0; i < dim; i++) {
            sum += Q[row * dim + i] * K[col * dim + i];
        }
        S[row * N + col] = sum / sqrtf((float)dim);
    }

}

__global__ void softmax(float *matrix_S, float *matrix_P, int N) {

    int batch_idx = blockIdx.z;
    float *S = matrix_S + (size_t)batch_idx * N * N;
    float *P = matrix_P + (size_t)batch_idx * N * N;

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N) {
        float max_val = -INFINITY;
        for (int j=0; j < N; j++) {
            max_val = max(max_val, S[row * N + j]);
        }

        float sum_exp = 0.0f;
        for (int j=0; j < N; j++) {
            sum_exp += expf(S[row * N + j] - max_val);
        }

        for (int j=0; j < N; j++) {
            P[row * N + j] = expf(S[row * N + j] - max_val) / sum_exp;
        }
    }
}

__global__ void output(float *matrix_P, float *matrix_V, float *matrix_O, int dim, int N) {

    int batch_idx = blockIdx.z;
    const float *P = matrix_P + (size_t)batch_idx * N * N;
    const float *V = matrix_V + (size_t)batch_idx * N * dim;
    float *O = matrix_O + (size_t)batch_idx * N * dim;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < N && col < dim) {
        for (int i = 0; i < N; i++) {
            sum += P[row * N + i] * V[i * dim + col];
        }
        O[row * dim + col] = sum;
    }
}

int main(int argc, char **argv) {

    int N = argc > 1 ? atoi(argv[1]) : 1024;
    int dim = argc > 2 ? atoi(argv[2]) : 64;
    int batch = argc > 3 ? atoi(argv[3]) : 1;

    size_t qkv_elems = (size_t)batch * N * dim;
    size_t s_elems = (size_t)batch * N * N;

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

    float *device_query, *device_key, *device_value;
    float *device_S, *device_P, *device_O;

    cudaMalloc(&device_query, qkv_elems * sizeof(float));
    cudaMalloc(&device_key, qkv_elems * sizeof(float));
    cudaMalloc(&device_value, qkv_elems * sizeof(float));

    cudaMalloc(&device_S, s_elems * sizeof(float));
    cudaMalloc(&device_P, s_elems * sizeof(float));
    cudaMalloc(&device_O, qkv_elems * sizeof(float));

    cudaMemcpy(device_query, host_query, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_key, host_key, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_value, host_value, qkv_elems * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((N + 31) / 32, (N + 31) / 32, batch);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    attention_scores<<<gridDim, blockDim>>>(device_query, device_key, device_S, dim, N);
    softmax<<<gridDim, blockDim>>>(device_S, device_P, N);
    output<<<gridDim, blockDim>>>(device_P, device_value, device_O, dim, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken: %f ms\n", ms);

    cudaMemcpy(host_output, device_O, qkv_elems * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Output[0][0]: %f\n", host_output[0]);

    float bytes_accessed = (
    3.0f * batch * N * dim * sizeof(float) +
    2.0f * batch * N * N * sizeof(float) +
    1.0f * batch * N * dim * sizeof(float)
    );

    float bandwidth_gb = (bytes_accessed / (ms / 1000.0f)) / 1e9f;
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gb);
    printf("HBM accessed: %.4f GB\n", bytes_accessed / 1e9f);

    FILE *f = fopen("outputs/naive_output.bin", "wb");
    fwrite(host_output, sizeof(float), qkv_elems, f);
    fclose(f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_query);
    cudaFree(device_key);
    cudaFree(device_value);
    cudaFree(device_S);
    cudaFree(device_P);
    cudaFree(device_O);
    delete[] host_query;
    delete[] host_key;
    delete[] host_value;
    delete[] host_output;

    return 0;
}
