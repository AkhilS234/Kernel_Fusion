#include <stdio.h>
#include <math.h>
#include <float.h>



__global__ void attention_scores(float *matrix_Q, float *matrix_K, float *matrix_S, int dim, int N) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += matrix_Q[row * dim + i] * matrix_K[col * dim + i];
        }
        matrix_S[row * N + col] = sum / sqrtf((float)dim);
    }

}

__global__ void softmax(float *matrix_S, float *matrix_P, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N) {
        float max_val = -INFINITY;
        for (int j=0; j < N; j++) {
            max_val = max(max_val, matrix_S[row * N + j]);
        }

        float sum_exp = 0.0f;
        for (int j=0; j < N; j++) {
            sum_exp += expf(matrix_S[row * N + j] - max_val);
        }

        for (int j=0; j < N; j++) {
            matrix_P[row * N + j] = expf(matrix_S[row * N + j] - max_val) / sum_exp;
        }
    }
}

__global__ void output(float *matrix_P, float *matrix_V, float *matrix_O, int dim, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < N && col < dim) {
        for (int i = 0; i < N; i++) {
            sum += matrix_P[row * N + i] * matrix_V[i * dim + col];
        }
        matrix_O[row * dim + col] = sum;
    }
}

int main () {

    int N = 1024;
    int dim = 64;

    float *host_query = new float[N * dim];
    float *host_key = new float[N * dim];
    float *host_value = new float[N * dim];

    float *host_output = new float[N * dim];

    for (int i = 0; i < N * dim; i++) {
        host_query[i] = (float)rand() / RAND_MAX;
        host_key[i]   = (float)rand() / RAND_MAX;
        host_value[i] = (float)rand() / RAND_MAX;
        host_output[i] = (float)rand() / RAND_MAX;
    }

    float *device_query, *device_key, *device_value;
    float *device_S, *device_P, *device_O;

    cudaMalloc(&device_query, N * dim * sizeof(float));
    cudaMalloc(&device_key, N * dim * sizeof(float));
    cudaMalloc(&device_value, N * dim * sizeof(float));

    cudaMalloc(&device_S, N * N * sizeof(float));
    cudaMalloc(&device_P, N * N * sizeof(float));
    cudaMalloc(&device_O, N * dim * sizeof(float));

    cudaMemcpy(device_query, host_query, N * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_key, host_key, N * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_value, host_value, N * dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim(N/32, N/32);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    attention_scores<<<gridDim, blockDim>>>(device_query, device_key, device_S, dim, N);
    softmax<<<gridDim, blockDim>>>(device_S, device_P, N);
    output<<<gridDim, blockDim>>>(device_P, device_value, device_O. dim, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken: %f ms\n", ms);

    cudaMemcpy(host_output, device_O, N * dim * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Output[0][0]: %f\n", host_value[0]);

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
}