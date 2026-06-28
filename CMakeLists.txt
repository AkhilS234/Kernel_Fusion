cmake_minimum_required(VERSION 3.10)
project(MyCUDAProject LANGUAGES CXX CUDA)

add_executable(naive_attention src/naive_attention.cu)
add_executable(flash_attn_forward src/flash_attn_forward.cu)
