cmake_minimum_required(VERSION 3.24)
project(cublaslt_runner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

add_executable(cublaslt_runner main.cu)

target_link_libraries(cublaslt_runner PRIVATE CUDA::cudart CUDA::cublasLt CUDA::cublas)
