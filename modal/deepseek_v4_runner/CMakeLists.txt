cmake_minimum_required(VERSION 3.18)
project(deepseek_v4_runner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(deepseek_v4_runner main.cu)
set_target_properties(deepseek_v4_runner PROPERTIES CUDA_ARCHITECTURES 80)
target_compile_options(deepseek_v4_runner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
