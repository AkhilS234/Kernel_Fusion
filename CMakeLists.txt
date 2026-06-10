cmake_minimum_required(VERSION 3.10)
   project(MyCUDAProject LANGUAGES CXX CUDA)

   add_executable(naive_attention naive_attention.cu)