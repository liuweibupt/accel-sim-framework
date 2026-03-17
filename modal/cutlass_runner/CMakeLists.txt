cmake_minimum_required(VERSION 3.24)
project(cutlass_runner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/cutlass")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS headers not found at ${CUTLASS_DIR}. Run modal/scripts/fetch_cutlass.sh first.")
endif()

add_executable(cutlass_runner main.cu)

target_include_directories(cutlass_runner PRIVATE
  "${CUTLASS_DIR}/include"
  "${CUTLASS_DIR}/tools/util/include"
)

target_link_libraries(cutlass_runner PRIVATE CUDA::cudart)

target_compile_options(cutlass_runner PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
