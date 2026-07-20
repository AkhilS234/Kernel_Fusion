#include "cuda_tile.h"
#include "cuda_fp16.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>


constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;

__tile_global__ void fused_attn_tile(const __half* A, const __half* B, const __half* C, float* D, int M, int N, int K, int batch_size) {
    
    // M is the number of total queries, K is the head dimension, N is the number of keys/values 

    // Query is (M x K), Key is (N x K), Value is (N x K)
    // The output is (M x K) - M rows (one per query) and K columns (head_dim)

    namespace ct = cuda::tiles;
    using namespace ct::literals;

    // TILE_BLOCK is the fixed size unit of work one block does per step, M/N/K determine how many of those units are needed in total

    // a = query, b = key, c = value, d = output

    auto a = ct::assume_aligned(A, 16_ic);
    auto b = ct::assume_aligned(B, 16_ic);
    auto c = ct::assume_aligned(C, 16_ic);
    auto d = ct::assume_aligned(D, 16_ic);

    auto aShape = ct::extents{M, K};
    auto bShape = ct::extents{K, N};
    auto cShape = ct::extents{N, K};
    auto dShape = ct::extents{M, K};

    // Tensor span attaches shape to a pointer to an array that already exists in memory 

    auto aSpan = ct::tensor_span{a, aShape};
    auto bSpan = ct::tensor_span{b, bShape};
    auto cSpan = ct::tensor_span{c, cShape};
    auto dSpan = ct::tensor_span{d, dShape};

    // Partition view defines how that array is defined into TILE_M x TILE_K sized chunks 

    auto aView = ct::partition_view{aSpan, ct::shape<TILE_M, TILE_K>{}};
    auto bView = ct::partition_view{bSpan, ct::shape<TILE_K, TILE_N>{}};
    auto cView = ct::partition_view{cSpan, ct::shape<TILE_N, TILE_K>{}};
    auto dView = ct::partition_view{dSpan, ct::shape<TILE_M, TILE_K>{}};

    // Find block's row and tile indices 
    //pid_m picks a size-defined chunk of rows out of Q, and therefore the same-numbered chunk of rows in the output 

    auto [pid_m, pid_n, dummy] = ct::bid();

    int total_k_blocks = (K + TILE_K - 1) / TILE_K;

    int total_kv_blocks = ((N + batch_size - 1)/batch_size);

    auto O = ct::zeros<ct::tile<float, ct::shape< TILE_M, TILE_K>>>();

    auto global_max = ct::zeros<ct::tile<float, ct::shape<TILE_M>>>();
    auto global_sum = ct::zeros<ct::tile<float, ct::shape<TILE_M>>>();

    auto max_old = ct::zeros<ct::tile<float, ct::shape<TILE_M>>>();
    auto max_new = ct::zeros<ct::tile<float, ct::shape<TILE_M>>>();


    for (int i = 0; i < total_kv_blocks; i++) {

        auto S = ct::zeros<ct::tile<float, ct::shape<TILE_M, TILE_N>>>();
        auto P = ct::zeros<ct::tile<float, ct::shape<TILE_M, TILE_N>>>();

            for (int k_block = 0; k_block < total_k_blocks; k_block++) {

                ct::tile<__half, ct::shape<TILE_M, TILE_K>>a_tile;
                ct::tile<__half, ct::shape<TILE_K, TILE_N>>b_tile;

                // Load_masked ensures that it fills in zero if reading past the array's true bounds 

                a_tile = aView.load_masked(pid_m, k_block);
                b_tile = bView.load_masked(k_block, i);

                S = ct::mma(a_tile, b_tile, S);
            }

            for (int r = 0; r < S.size(); r++) {

                float local_max = global_max[r];
                float expSum = global_sum[r];

                for (int c = 0; c < S[0].size(); c++) {
                    if (S[r][c] > local_max) {
                        local_max = S[r][c];
                    }
                }
                
                max_old[r] = global_max[r];
                float d_old = global_sum[r];
                max_new[r] = max(local_max, max_old[r]);
                float d_new = d_old;

                for (int c = 0; c < S[0].size(); c++) {
                    expSum += std::exp(S[r][c] - max_new[r]);
                }
                d_new = ((d_old * std::exp(max_old[r]-max_new[r])) + expSum);

                global_max[r] = max_new[r];
                global_sum[r] = d_new;
                
            }

            for (int r = 0; r < S.size(); r++) {
                for (int c = 0; c < S[0].size(); c++) {
                    float val = S[r][c];
                    P[r][c] = (std::exp(val-global_max[r])) / global_sum[r];
                }
            }

            ct::tile<__half, ct::shape<TILE_N, TILE_K>>c_tile;
            c_tile = cView.load_masked(i, 0);

            for (int r = 0; r < O.size(); r++) {
                for (int c = 0; c < O[0].size(); c++) {
                    O[r][c] = O[r][c] * std::exp(max_old[r]-max_new[r]);                
                }
            }
            O = ct::mma(P, c_tile, O);
    }
    // Store_masked ensures that write backs only happen to elements that map to genuine positions inside C, no padding
    // Prevents writing past the end of the C buffer 

    dView.store_masked(O, pid_m, pid_n);
}

int main(int argc, char **argv) {

    int N          = argc > 1 ? atoi(argv[1]) : 1024;
    int dim        = argc > 2 ? atoi(argv[2]) : 64;
    int batch_size = argc > 3 ? atoi(argv[3]) : TILE_N;

    int M = N;
    int K = dim;

    if (K % TILE_K != 0) {
        printf("Error: dim %d must be a multiple of TILE_K=%d\n", dim, TILE_K);
        return 1;
    }

    size_t q_elements = (size_t)M * K;
    size_t k_elements = (size_t)K * N;
    size_t v_elements = (size_t)N * K;
    size_t o_elements = (size_t)M * K;

    float *host_output = new float[o_elements];

    __half *host_query = new __half[q_elements];
    __half *host_key   = new __half[k_elements];
    __half *host_value = new __half[v_elements];

    for (size_t i = 0; i < q_elements; i++) host_query[i] = __float2half((float)i);
    for (size_t i = 0; i < k_elements; i++) host_key[i]   = __float2half((float)i);
    for (size_t i = 0; i < v_elements; i++) host_value[i] = __float2half((float)i);

    __half *device_Q, *device_K, *device_V;
    float  *device_O;

    cudaMalloc(&device_Q, q_elements * sizeof(__half));
    cudaMalloc(&device_K, k_elements * sizeof(__half));
    cudaMalloc(&device_V, v_elements * sizeof(__half));
    cudaMalloc(&device_O, o_elements * sizeof(float));

    cudaMemcpy(device_Q, host_query, q_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_K, host_key,   k_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_V, host_value, v_elements * sizeof(__half), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 gridDim((M + TILE_M - 1) / TILE_M, 1, 1);
    fused_attn_tile<<<gridDim, 1>>>(device_Q, device_K, device_V, device_O, M, N, K, batch_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken: %f ms\n", ms);

    cudaMemcpy(host_output, device_O, o_elements * sizeof(float), cudaMemcpyDeviceToHost);

    float bytes_accessed = 3.0f * q_elements * sizeof(__half)
                         + 1.0f * o_elements * sizeof(float);
    float bandwidth_gb = (bytes_accessed / (ms / 1000.0f)) / 1e9f;
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gb);
    printf("HBM accessed: %.4f GB\n", bytes_accessed / 1e9f);

    FILE *fo = fopen("outputs/output_S.bin", "wb");
    fwrite(host_output, sizeof(float), o_elements, fo);
    fclose(fo);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_Q); cudaFree(device_K); cudaFree(device_V); cudaFree(device_O);
    delete[] host_query; delete[] host_key; delete[] host_value; delete[] host_output;

    return 0;
}