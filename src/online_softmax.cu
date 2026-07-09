#include <stdio.h>

__global__ void online_softmax(float *matrix_S, float *matrix_P, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N) {

        float m = -INFINITY;
        float l = 0.0f;

        for (int j = 0; j < N; j++) {
            float val = matrix_S[row * N + j];
            float m_new = max(m, val);
            l = expf(m-m_new) * l + expf(val - m_new);
            m = m_new;
        }

        for (int j = 0; j < N; j++) {
            matrix_P[row * N + j] = expf(matrix_S[row * N + j] - m) / l;
        }
    }
}

int main() {

    int N = 4;
    float host_input[] = {1.0f, 2.0f, 3.0f, 4.0f,
                          1.0f, 1.0f, 1.0f, 1.0f,
                          0.1f, 0.2f, 0.3f, 0.4f,
                          4.0f, 3.0f, 2.0f, 1.0f};

    float host_output[N * N] = {0};

    float *device_input, *device_output;
    cudaMalloc(&device_input, N * N * sizeof(float));
    cudaMalloc(&device_output, N * N * sizeof(float));

    cudaMemcpy(device_input, host_input, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32);
    dim3(gridDim(N+31) / 32);
    online_softmax<<<gridDim, blockDim>>>(device_input, device_output, N);

    cudaMemcpy(host_output, device_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("row %d: ", i);
        for (int j = 0; j < N; j++) {
            printf("%.4f ", host_output[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(device_input);
    cudaFree(device_output);
    return 0;

} 