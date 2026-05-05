#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void CheckCuda(cudaError_t result, char const* expr) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error for " << expr << ": " << cudaGetErrorString(result) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(expr) CheckCuda((expr), #expr)

constexpr int kFillThreads = 256;
constexpr int kKernelThreads = 128;
constexpr int kMoeTileElements = 16384;

__global__ void FillHalfWords(uint16_t* data, size_t count, uint32_t seed) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = static_cast<uint16_t>(((idx + seed) * 1103515245ull) >> 16);
  }
}

__global__ void FillInts(int* data, size_t count, int modulo) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = static_cast<int>(idx % static_cast<size_t>(modulo));
  }
}

__global__ void DeepSeekV4IndexKernel(
    int const* __restrict__ csa_index,
    int const* __restrict__ hca_index,
    float* __restrict__ out,
    int batch,
    int csa_entries,
    int hca_entries) {
  int request = blockIdx.x;
  int lane = threadIdx.x;
  if (request >= batch) {
    return;
  }
  float acc = 0.0f;
  for (int entry = lane; entry < csa_entries; entry += blockDim.x) {
    acc += static_cast<float>(csa_index[request * csa_entries + entry] & 0xff) * 0.001f;
  }
  for (int entry = lane; entry < hca_entries; entry += blockDim.x) {
    acc += static_cast<float>(hca_index[request * hca_entries + entry] & 0xff) * 0.002f;
  }
  atomicAdd(&out[request], acc);
}

__global__ void DeepSeekV4SelectedKvKernel(
    uint16_t const* __restrict__ selected_kv_pages,
    int const* __restrict__ selected_page_table,
    float* __restrict__ out,
    int batch,
    int selected_blocks,
    int elements_per_block,
    int samples_per_block) {
  int request = blockIdx.x;
  int block = blockIdx.y;
  int lane = threadIdx.x;
  if (request >= batch || block >= selected_blocks || lane >= samples_per_block) {
    return;
  }
  int physical_block = selected_page_table[request * selected_blocks + block];
  size_t base = static_cast<size_t>(physical_block) * static_cast<size_t>(elements_per_block);
  float acc = 0.0f;
  for (int offset = lane; offset < elements_per_block; offset += blockDim.x) {
    uint16_t value = selected_kv_pages[base + static_cast<size_t>(offset)];
    acc += static_cast<float>(value & 0xffu) * 0.0005f;
  }
  atomicAdd(&out[request], acc);
}

__global__ void DeepSeekV4HcaSummaryKernel(
    uint16_t const* __restrict__ hca_pages,
    float* __restrict__ out,
    int batch,
    int hca_blocks,
    int elements_per_block) {
  int request = blockIdx.x;
  int block = blockIdx.y;
  int lane = threadIdx.x;
  if (request >= batch || block >= hca_blocks) {
    return;
  }
  size_t base = static_cast<size_t>(request * hca_blocks + block) * static_cast<size_t>(elements_per_block);
  float acc = 0.0f;
  for (int offset = lane; offset < elements_per_block; offset += blockDim.x) {
    uint16_t value = hca_pages[base + static_cast<size_t>(offset)];
    acc += static_cast<float>(value & 0x7fu) * 0.0003f;
  }
  atomicAdd(&out[request], acc);
}

__global__ void DeepSeekV4SwaTailKernel(
    uint16_t const* __restrict__ swa_tail,
    float* __restrict__ out,
    int batch,
    int swa_tokens,
    int elements_per_token) {
  int request = blockIdx.x;
  int token = blockIdx.y;
  int lane = threadIdx.x;
  if (request >= batch || token >= swa_tokens) {
    return;
  }
  size_t base = static_cast<size_t>(request * swa_tokens + token) * static_cast<size_t>(elements_per_token);
  float acc = 0.0f;
  for (int offset = lane; offset < elements_per_token; offset += blockDim.x) {
    uint16_t value = swa_tail[base + static_cast<size_t>(offset)];
    acc += static_cast<float>(value & 0x3fu) * 0.0007f;
  }
  atomicAdd(&out[request], acc);
}

__global__ void DeepSeekV4MoeExpertKernel(
    uint16_t const* __restrict__ expert_tiles,
    int const* __restrict__ expert_table,
    float* __restrict__ out,
    int batch,
    int experts_per_request,
    int elements_per_tile) {
  int request = blockIdx.x;
  int expert = blockIdx.y;
  int lane = threadIdx.x;
  if (request >= batch || expert >= experts_per_request) {
    return;
  }
  int physical_expert = expert_table[request * experts_per_request + expert];
  size_t base = static_cast<size_t>(physical_expert) * static_cast<size_t>(elements_per_tile);
  float acc = 0.0f;
  for (int offset = lane; offset < elements_per_tile; offset += blockDim.x) {
    uint16_t value = expert_tiles[base + static_cast<size_t>(offset)];
    acc += static_cast<float>(value & 0x1fu) * 0.0009f;
  }
  atomicAdd(&out[request], acc);
}

struct Options {
  int batch = 4;
  int kvlen = 16384;
  int shared_prefix = 12288;
  int page_tokens = 16;
  int kv_heads = 8;
  int head_dim = 128;
  int csa_group = 4;
  int hca_group = 128;
  int selected_blocks = 512;
  int selected_shared_blocks = 384;
  int swa_tokens = 128;
  int hot_experts = 8;
  int experts_per_request = 2;
  int samples_per_block = 32;
};

int ParseInt(char const* value, char const* name) {
  int parsed = std::stoi(value);
  if (parsed < 0) {
    std::cerr << name << " must be non-negative\n";
    std::exit(EXIT_FAILURE);
  }
  return parsed;
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](char const* name) -> char const* {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (arg == "--batch") options.batch = ParseInt(need_value("--batch"), "batch");
    else if (arg == "--kvlen") options.kvlen = ParseInt(need_value("--kvlen"), "kvlen");
    else if (arg == "--shared-prefix") options.shared_prefix = ParseInt(need_value("--shared-prefix"), "shared-prefix");
    else if (arg == "--page") options.page_tokens = ParseInt(need_value("--page"), "page");
    else if (arg == "--kv-heads") options.kv_heads = ParseInt(need_value("--kv-heads"), "kv-heads");
    else if (arg == "--head-dim") options.head_dim = ParseInt(need_value("--head-dim"), "head-dim");
    else if (arg == "--csa-group") options.csa_group = ParseInt(need_value("--csa-group"), "csa-group");
    else if (arg == "--hca-group") options.hca_group = ParseInt(need_value("--hca-group"), "hca-group");
    else if (arg == "--selected-blocks") options.selected_blocks = ParseInt(need_value("--selected-blocks"), "selected-blocks");
    else if (arg == "--selected-shared-blocks") options.selected_shared_blocks = ParseInt(need_value("--selected-shared-blocks"), "selected-shared-blocks");
    else if (arg == "--swa-tokens") options.swa_tokens = ParseInt(need_value("--swa-tokens"), "swa-tokens");
    else if (arg == "--hot-experts") options.hot_experts = ParseInt(need_value("--hot-experts"), "hot-experts");
    else if (arg == "--experts-per-request") options.experts_per_request = ParseInt(need_value("--experts-per-request"), "experts-per-request");
    else if (arg == "--samples-per-block") options.samples_per_block = ParseInt(need_value("--samples-per-block"), "samples-per-block");
    else if (arg == "--help") {
      std::cout << "Usage: deepseek_v4_runner [--batch N] [--kvlen tokens] [--shared-prefix tokens] "
                << "[--selected-blocks N] [--selected-shared-blocks N] [--swa-tokens N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if (options.batch <= 0 || options.kvlen <= 0 || options.page_tokens <= 0 || options.kv_heads <= 0 ||
      options.head_dim <= 0 || options.csa_group <= 0 || options.hca_group <= 0 || options.selected_blocks <= 0 ||
      options.samples_per_block <= 0 || options.hot_experts <= 0 || options.experts_per_request <= 0) {
    std::cerr << "all positive options must be > 0\n";
    std::exit(EXIT_FAILURE);
  }
  if (options.shared_prefix > options.kvlen) {
    std::cerr << "shared-prefix must be <= kvlen\n";
    std::exit(EXIT_FAILURE);
  }
  if (options.selected_shared_blocks > options.selected_blocks) {
    std::cerr << "selected-shared-blocks must be <= selected-blocks\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

int CeilDiv(int a, int b) { return (a + b - 1) / b; }

template <typename T>
T* DeviceAlloc(size_t count) {
  T* ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&ptr, count * sizeof(T)));
  return ptr;
}

void FillDeviceHalf(uint16_t* ptr, size_t count, uint32_t seed) {
  int blocks = static_cast<int>((count + kFillThreads - 1) / kFillThreads);
  FillHalfWords<<<blocks, kFillThreads>>>(ptr, count, seed);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace

int main(int argc, char** argv) {
  Options options = ParseOptions(argc, argv);
  int elements_per_token = 2 * options.kv_heads * options.head_dim;
  int elements_per_page = elements_per_token * options.page_tokens;
  int csa_entries = CeilDiv(options.kvlen, options.csa_group);
  int hca_entries = CeilDiv(options.kvlen, options.hca_group);
  int hca_blocks = hca_entries;
  int shared_selected = std::min(options.selected_shared_blocks, options.selected_blocks);
  int private_selected = options.selected_blocks - shared_selected;
  int unique_selected_blocks = shared_selected + options.batch * private_selected;
  int unique_expert_tiles = options.hot_experts;

  std::vector<int> selected_table(static_cast<size_t>(options.batch) * options.selected_blocks);
  for (int request = 0; request < options.batch; ++request) {
    for (int block = 0; block < options.selected_blocks; ++block) {
      if (block < shared_selected) {
        selected_table[request * options.selected_blocks + block] = block;
      } else {
        selected_table[request * options.selected_blocks + block] =
            shared_selected + request * private_selected + (block - shared_selected);
      }
    }
  }

  std::vector<int> expert_table(static_cast<size_t>(options.batch) * options.experts_per_request);
  for (int request = 0; request < options.batch; ++request) {
    for (int expert = 0; expert < options.experts_per_request; ++expert) {
      expert_table[request * options.experts_per_request + expert] = (request + expert) % options.hot_experts;
    }
  }

  size_t csa_index_count = static_cast<size_t>(options.batch) * csa_entries;
  size_t hca_index_count = static_cast<size_t>(options.batch) * hca_entries;
  size_t selected_kv_elements = static_cast<size_t>(unique_selected_blocks) * elements_per_page;
  size_t hca_elements = static_cast<size_t>(options.batch) * hca_blocks * elements_per_page;
  size_t swa_elements = static_cast<size_t>(options.batch) * options.swa_tokens * elements_per_token;
  size_t expert_elements = static_cast<size_t>(unique_expert_tiles) * kMoeTileElements;

  int* csa_index = DeviceAlloc<int>(csa_index_count);
  int* hca_index = DeviceAlloc<int>(hca_index_count);
  int* selected_page_table = DeviceAlloc<int>(selected_table.size());
  int* device_expert_table = DeviceAlloc<int>(expert_table.size());
  uint16_t* selected_kv = DeviceAlloc<uint16_t>(selected_kv_elements);
  uint16_t* hca_pages = DeviceAlloc<uint16_t>(hca_elements);
  uint16_t* swa_tail = DeviceAlloc<uint16_t>(swa_elements);
  uint16_t* expert_tiles = DeviceAlloc<uint16_t>(expert_elements);
  float* out = DeviceAlloc<float>(static_cast<size_t>(options.batch));

  int fill_csa_blocks = static_cast<int>((csa_index_count + kFillThreads - 1) / kFillThreads);
  int fill_hca_blocks = static_cast<int>((hca_index_count + kFillThreads - 1) / kFillThreads);
  FillInts<<<fill_csa_blocks, kFillThreads>>>(csa_index, csa_index_count, options.selected_blocks);
  FillInts<<<fill_hca_blocks, kFillThreads>>>(hca_index, hca_index_count, std::max(1, hca_blocks));
  CHECK_CUDA(cudaMemcpy(selected_page_table, selected_table.data(), selected_table.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(device_expert_table, expert_table.data(), expert_table.size() * sizeof(int), cudaMemcpyHostToDevice));
  FillDeviceHalf(selected_kv, selected_kv_elements, 17);
  FillDeviceHalf(hca_pages, hca_elements, 29);
  FillDeviceHalf(swa_tail, swa_elements, 43);
  FillDeviceHalf(expert_tiles, expert_elements, 61);
  CHECK_CUDA(cudaMemset(out, 0, static_cast<size_t>(options.batch) * sizeof(float)));
  CHECK_CUDA(cudaDeviceSynchronize());

  DeepSeekV4IndexKernel<<<options.batch, kKernelThreads>>>(csa_index, hca_index, out, options.batch, csa_entries, hca_entries);
  CHECK_CUDA(cudaGetLastError());
  DeepSeekV4SelectedKvKernel<<<dim3(options.batch, options.selected_blocks, 1), kKernelThreads>>>(
      selected_kv, selected_page_table, out, options.batch, options.selected_blocks, elements_per_page, options.samples_per_block);
  CHECK_CUDA(cudaGetLastError());
  DeepSeekV4HcaSummaryKernel<<<dim3(options.batch, hca_blocks, 1), kKernelThreads>>>(
      hca_pages, out, options.batch, hca_blocks, elements_per_page);
  CHECK_CUDA(cudaGetLastError());
  DeepSeekV4SwaTailKernel<<<dim3(options.batch, options.swa_tokens, 1), kKernelThreads>>>(
      swa_tail, out, options.batch, options.swa_tokens, elements_per_token);
  CHECK_CUDA(cudaGetLastError());
  DeepSeekV4MoeExpertKernel<<<dim3(options.batch, options.experts_per_request, 1), kKernelThreads>>>(
      expert_tiles, device_expert_table, out, options.batch, options.experts_per_request, kMoeTileElements);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> host_out(static_cast<size_t>(options.batch));
  CHECK_CUDA(cudaMemcpy(host_out.data(), out, host_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  size_t index_logical_bytes = (csa_index_count + hca_index_count) * sizeof(int);
  size_t selected_logical_bytes = static_cast<size_t>(options.batch) * options.selected_blocks * elements_per_page * sizeof(uint16_t);
  size_t selected_unique_bytes = selected_kv_elements * sizeof(uint16_t);
  size_t hca_bytes = hca_elements * sizeof(uint16_t);
  size_t swa_bytes = swa_elements * sizeof(uint16_t);
  size_t moe_logical_bytes = static_cast<size_t>(options.batch) * options.experts_per_request * kMoeTileElements * sizeof(uint16_t);
  size_t moe_unique_bytes = expert_elements * sizeof(uint16_t);
  size_t logical_bytes = index_logical_bytes + selected_logical_bytes + hca_bytes + swa_bytes + moe_logical_bytes;
  size_t unique_bytes = index_logical_bytes + selected_unique_bytes + hca_bytes + swa_bytes + moe_unique_bytes;

  std::cout << "DEEPSEEK_V4_TRACE complete batch=" << options.batch << " kvlen=" << options.kvlen
            << " shared_prefix=" << options.shared_prefix << " csa_group=" << options.csa_group
            << " hca_group=" << options.hca_group << " selected_blocks=" << options.selected_blocks
            << " selected_shared_blocks=" << shared_selected << " swa_tokens=" << options.swa_tokens
            << " hot_experts=" << options.hot_experts << " logical_bytes=" << logical_bytes
            << " unique_bytes=" << unique_bytes << " sample=" << host_out[0] << "\n";

  CHECK_CUDA(cudaFree(csa_index));
  CHECK_CUDA(cudaFree(hca_index));
  CHECK_CUDA(cudaFree(selected_page_table));
  CHECK_CUDA(cudaFree(device_expert_table));
  CHECK_CUDA(cudaFree(selected_kv));
  CHECK_CUDA(cudaFree(hca_pages));
  CHECK_CUDA(cudaFree(swa_tail));
  CHECK_CUDA(cudaFree(expert_tiles));
  CHECK_CUDA(cudaFree(out));
  return EXIT_SUCCESS;
}
