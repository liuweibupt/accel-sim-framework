#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kLlama31M = 2048;
constexpr int kSmallBatchMs[] = {1, 2, 4, 8};
constexpr int kLlamaHidden = 4096;
constexpr int kLlamaKvDim = 1024;
constexpr int kLlamaIntermediate = 14336;
constexpr int kQwen3Intermediate = 12288;

void CheckCuda(cudaError_t result, char const* expr) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error for " << expr << ": " << cudaGetErrorString(result) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

void CheckCublasLt(cublasStatus_t result, char const* expr) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLASLt error for " << expr << ": status=" << static_cast<int>(result) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(expr) CheckCuda((expr), #expr)
#define CHECK_CUBLASLT(expr) CheckCublasLt((expr), #expr)

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

bool IsSmallBatchM(int m) {
  for (int supported_m : kSmallBatchMs) {
    if (m == supported_m) {
      return true;
    }
  }
  return false;
}

bool IsSupportedM(int m) { return IsSmallBatchM(m) || m == kLlama31M; }

bool IsSupportedShape(int m, int n, int k) {
  if (IsSupportedM(m) && n == 12288 && k == 12288) {
    return true;
  }
  if (!IsSupportedM(m)) {
    return false;
  }
  return (n == kLlamaHidden && k == kLlamaHidden) ||
         (n == kLlamaKvDim && k == kLlamaHidden) ||
         (n == kLlamaIntermediate && k == kLlamaHidden) ||
         (n == kLlamaHidden && k == kLlamaIntermediate) ||
         (n == kQwen3Intermediate && k == kLlamaHidden) ||
         (n == kLlamaHidden && k == kQwen3Intermediate);
}

char const* SupportedShapeMessage() {
  return "Supported shapes: GPT-style Mx12288x12288 with M in {1,2,4,8,2048} and "
         "LLaMA-3.1 projection shapes Mx4096x4096, Mx1024x4096, "
         "Mx14336x4096, Mx4096x14336, plus Qwen3 Mx12288x4096 and "
         "Mx4096x12288 with M in {1,2,4,8,2048}.";
}

template <typename T>
__global__ void FillPatternKernel(T* data, size_t count, int mod, int offset, float scale);

template <>
__global__ void FillPatternKernel<__half>(__half* data, size_t count, int mod, int offset, float scale) {
  size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx < count) {
    float value = static_cast<float>(static_cast<int>(idx % static_cast<size_t>(mod)) - offset) * scale;
    data[idx] = __float2half(value);
  }
}

template <>
__global__ void FillPatternKernel<__nv_bfloat16>(__nv_bfloat16* data, size_t count, int mod, int offset, float scale) {
  size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx < count) {
    float value = static_cast<float>(static_cast<int>(idx % static_cast<size_t>(mod)) - offset) * scale;
    data[idx] = __float2bfloat16(value);
  }
}

template <typename InputType>
int RunGemm(int m, int n, int k, cudaDataType_t input_cuda_type) {
  constexpr int kThreadsPerBlock = 256;
  constexpr size_t kWorkspaceBytes = 64ull * 1024ull * 1024ull;

  const size_t a_elements = static_cast<size_t>(m) * static_cast<size_t>(k);
  const size_t b_elements = static_cast<size_t>(k) * static_cast<size_t>(n);
  const size_t c_elements = static_cast<size_t>(m) * static_cast<size_t>(n);

  InputType* device_a = nullptr;
  InputType* device_b = nullptr;
  float* device_c = nullptr;
  void* workspace = nullptr;

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_a), sizeof(InputType) * a_elements));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_b), sizeof(InputType) * b_elements));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_c), sizeof(float) * c_elements));
  CHECK_CUDA(cudaMalloc(&workspace, kWorkspaceBytes));

  int blocks_a = static_cast<int>((a_elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
  int blocks_b = static_cast<int>((b_elements + kThreadsPerBlock - 1) / kThreadsPerBlock);

  FillPatternKernel<InputType><<<blocks_a, kThreadsPerBlock>>>(device_a, a_elements, 13, 6, 0.125f);
  FillPatternKernel<InputType><<<blocks_b, kThreadsPerBlock>>>(device_b, b_elements, 7, 3, 0.25f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemset(device_c, 0, sizeof(float) * c_elements));

  cublasLtHandle_t lt_handle = nullptr;
  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatrixLayout_t d_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  CHECK_CUBLASLT(cublasLtCreate(&lt_handle));
  CHECK_CUBLASLT(cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t op_n = CUBLAS_OP_N;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&a_layout, input_cuda_type, m, k, k));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&b_layout, input_cuda_type, k, n, n));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, m, n, n));
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_32F, m, n, n));

  cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));

  CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
  CHECK_CUBLASLT(
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes,
                                           sizeof(kWorkspaceBytes)));

  constexpr int kMaxAlgos = 16;
  cublasLtMatmulHeuristicResult_t heuristics[kMaxAlgos];
  int returned_results = 0;
  CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(lt_handle, operation_desc, a_layout, b_layout, c_layout, d_layout,
                                                preference, kMaxAlgos, heuristics, &returned_results));

  if (returned_results <= 0) {
    std::cerr << "No cuBLASLt algorithm found for M=" << m << " N=" << n << " K=" << k << "\n";
    return EXIT_FAILURE;
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUBLASLT(cublasLtMatmul(lt_handle, operation_desc, &alpha, device_a, a_layout, device_b, b_layout, &beta,
                                device_c, c_layout, device_c, d_layout, &heuristics[0].algo, workspace,
                                kWorkspaceBytes, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> sample(8, 0.0f);
  CHECK_CUDA(cudaMemcpy(sample.data(), device_c, sizeof(float) * sample.size(), cudaMemcpyDeviceToHost));

  std::cout << "GEMM complete with cuBLASLt: M=" << m << " N=" << n << " K=" << k
            << " sample=[" << sample[0] << ", " << sample[1] << ", " << sample[2] << ", " << sample[3] << "]"
            << "\n";

  CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(a_layout));
  CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(b_layout));
  CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(c_layout));
  CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(d_layout));
  CHECK_CUBLASLT(cublasLtMatmulDescDestroy(operation_desc));
  CHECK_CUBLASLT(cublasLtDestroy(lt_handle));

  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));

  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  int m = 2048;
  int n = 12288;
  int k = 12288;
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
      std::cout << "Usage: cublaslt_runner [--dtype fp16|bf16] [--m M] [--n N] [--k K]\n";
      std::cout << "Example: cublaslt_runner --dtype bf16 --m 2048 --n 4096 --k 4096\n";
      std::cout << SupportedShapeMessage() << "\n";
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

  if (!IsSupportedShape(m, n, k)) {
    std::cerr << SupportedShapeMessage() << "\n";
    return EXIT_FAILURE;
  }

  if (dtype == DataType::kFp16) {
    return RunGemm<__half>(m, n, k, CUDA_R_16F);
  }

  return RunGemm<__nv_bfloat16>(m, n, k, CUDA_R_16BF);
}
