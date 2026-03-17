#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"

namespace {

void CheckCuda(cudaError_t result, char const* expr) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error for " << expr << ": " << cudaGetErrorString(result) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(expr) CheckCuda((expr), #expr)

enum class DataType {
  kFp16,
  kBf16,
};

bool ParseDataType(std::string const& arg, DataType* dtype) {
  if (arg == "fp16") {
    *dtype = DataType::kFp16;
    return true;
  }
  if (arg == "bf16") {
    *dtype = DataType::kBf16;
    return true;
  }
  return false;
}

template <typename T>
T ToCutlassType(float value);

template <>
cutlass::half_t ToCutlassType<cutlass::half_t>(float value) {
  return cutlass::half_t(value);
}

template <>
cutlass::bfloat16_t ToCutlassType<cutlass::bfloat16_t>(float value) {
  return cutlass::bfloat16_t(value);
}

template <typename InputType>
int RunGemm(int m, int n, int k) {
  using ElementAccumulator = float;
  using ElementCompute = float;

  using Gemm = cutlass::gemm::device::Gemm<
      InputType, cutlass::layout::RowMajor,
      InputType, cutlass::layout::RowMajor,
      ElementCompute, cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80>;

  std::vector<InputType> host_a(static_cast<size_t>(m) * static_cast<size_t>(k));
  std::vector<InputType> host_b(static_cast<size_t>(k) * static_cast<size_t>(n));
  std::vector<float> host_c(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);

  for (size_t i = 0; i < host_a.size(); ++i) {
    float value = static_cast<float>((i % 13) - 6) * 0.125f;
    host_a[i] = ToCutlassType<InputType>(value);
  }
  for (size_t i = 0; i < host_b.size(); ++i) {
    float value = static_cast<float>((i % 7) - 3) * 0.25f;
    host_b[i] = ToCutlassType<InputType>(value);
  }

  InputType* device_a = nullptr;
  InputType* device_b = nullptr;
  float* device_c = nullptr;

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_a), sizeof(InputType) * host_a.size()));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_b), sizeof(InputType) * host_b.size()));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_c), sizeof(float) * host_c.size()));

  CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), sizeof(InputType) * host_a.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), sizeof(InputType) * host_b.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(device_c, 0, sizeof(float) * host_c.size()));

  typename Gemm::Arguments arguments(
      {m, n, k},
      {device_a, k},
      {device_b, n},
      {device_c, n},
      {device_c, n},
      {1.0f, 0.0f});

  Gemm gemm_op;
  cutlass::Status status = gemm_op(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM launch failed: " << cutlassGetStatusString(status) << "\n";
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    return EXIT_FAILURE;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, sizeof(float) * host_c.size(), cudaMemcpyDeviceToHost));

  double checksum = std::accumulate(host_c.begin(), host_c.end(), 0.0);
  std::cout << "GEMM complete: M=" << m << " N=" << n << " K=" << k
            << " checksum=" << checksum << "\n";

  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));

  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  int m = 512;
  int n = 512;
  int k = 512;
  DataType dtype = DataType::kFp16;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dtype" && i + 1 < argc) {
      if (!ParseDataType(argv[++i], &dtype)) {
        std::cerr << "Unsupported dtype. Use fp16 or bf16.\n";
        return EXIT_FAILURE;
      }
    } else if (arg == "--m" && i + 1 < argc) {
      m = std::stoi(argv[++i]);
    } else if (arg == "--n" && i + 1 < argc) {
      n = std::stoi(argv[++i]);
    } else if (arg == "--k" && i + 1 < argc) {
      k = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: cutlass_runner [--dtype fp16|bf16] [--m M] [--n N] [--k K]\n";
      std::cout << "Examples:\n";
      std::cout << "  cutlass_runner --dtype fp16 --m 512 --n 512 --k 512\n";
      std::cout << "  cutlass_runner --dtype bf16 --m 512 --n 12288 --k 12288\n";
      return EXIT_SUCCESS;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return EXIT_FAILURE;
    }
  }

  if (m <= 0 || n <= 0 || k <= 0) {
    std::cerr << "M, N, and K must be positive integers.\n";
    return EXIT_FAILURE;
  }

  if (!((m == 512 && n == 512 && k == 512) || (m == 512 && n == 12288 && k == 12288))) {
    std::cerr << "Currently supported shapes are 512x512x512 and 512x12288x12288.\n";
    return EXIT_FAILURE;
  }

  if (dtype == DataType::kFp16) {
    return RunGemm<cutlass::half_t>(m, n, k);
  }
  return RunGemm<cutlass::bfloat16_t>(m, n, k);
}
