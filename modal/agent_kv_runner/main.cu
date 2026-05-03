#include <cuda_runtime.h>

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

__global__ void FillHalfWords(uint16_t* data, size_t count) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = static_cast<uint16_t>((idx * 1315423911ull) >> 16);
  }
}

__global__ void AgentPagedKvDecodeKernel(
    uint16_t const* __restrict__ kv_pages,
    int const* __restrict__ page_table,
    float* __restrict__ out,
    int batch,
    int pages_per_request,
    int elements_per_page,
    int samples_per_page) {
  int request = blockIdx.x;
  int page = blockIdx.y;
  int lane = threadIdx.x;
  if (request >= batch || page >= pages_per_request || lane >= samples_per_page) {
    return;
  }
  int physical_page = page_table[request * pages_per_request + page];
  size_t base = static_cast<size_t>(physical_page) * static_cast<size_t>(elements_per_page);
  float acc = 0.0f;
  for (int offset = lane; offset < elements_per_page; offset += blockDim.x) {
    uint16_t value = kv_pages[base + static_cast<size_t>(offset)];
    acc += static_cast<float>(value & 0xffu) * 0.001f;
  }
  atomicAdd(&out[request], acc);
}

struct Options {
  int batch = 4;
  int kvlen = 16384;
  int shared_prefix = 12288;
  int page = 16;
  int kv_heads = 8;
  int head_dim = 128;
  int dtype_bytes = 2;
  int samples_per_page = 32;
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
    else if (arg == "--page") options.page = ParseInt(need_value("--page"), "page");
    else if (arg == "--kv-heads") options.kv_heads = ParseInt(need_value("--kv-heads"), "kv-heads");
    else if (arg == "--head-dim") options.head_dim = ParseInt(need_value("--head-dim"), "head-dim");
    else if (arg == "--samples-per-page") options.samples_per_page = ParseInt(need_value("--samples-per-page"), "samples-per-page");
    else if (arg == "--help") {
      std::cout << "Usage: agent_kv_runner [--batch N] [--kvlen tokens] [--shared-prefix tokens] "
                << "[--page tokens] [--kv-heads N] [--head-dim D] [--samples-per-page N]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if (options.batch <= 0 || options.kvlen <= 0 || options.page <= 0 || options.kv_heads <= 0 ||
      options.head_dim <= 0 || options.samples_per_page <= 0) {
    std::cerr << "batch, kvlen, page, kv-heads, head-dim, and samples-per-page must be positive\n";
    std::exit(EXIT_FAILURE);
  }
  if (options.shared_prefix > options.kvlen) {
    std::cerr << "shared-prefix must be <= kvlen\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

int CeilDiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

int main(int argc, char** argv) {
  Options options = ParseOptions(argc, argv);
  int pages_per_request = CeilDiv(options.kvlen, options.page);
  int shared_pages = CeilDiv(options.shared_prefix, options.page);
  int private_pages = pages_per_request - shared_pages;
  int unique_pages = shared_pages + options.batch * private_pages;
  int elements_per_page = 2 * options.page * options.kv_heads * options.head_dim;

  std::vector<int> page_table(static_cast<size_t>(options.batch) * pages_per_request);
  for (int request = 0; request < options.batch; ++request) {
    for (int page = 0; page < pages_per_request; ++page) {
      if (page < shared_pages) {
        page_table[request * pages_per_request + page] = page;
      } else {
        int private_page = page - shared_pages;
        page_table[request * pages_per_request + page] = shared_pages + request * private_pages + private_page;
      }
    }
  }

  size_t kv_elements = static_cast<size_t>(unique_pages) * elements_per_page;
  uint16_t* kv_pages = nullptr;
  int* device_page_table = nullptr;
  float* out = nullptr;
  CHECK_CUDA(cudaMalloc(&kv_pages, kv_elements * sizeof(uint16_t)));
  CHECK_CUDA(cudaMalloc(&device_page_table, page_table.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&out, static_cast<size_t>(options.batch) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(device_page_table, page_table.data(), page_table.size() * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(out, 0, static_cast<size_t>(options.batch) * sizeof(float)));

  int fill_threads = 256;
  int fill_blocks = static_cast<int>((kv_elements + fill_threads - 1) / fill_threads);
  FillHalfWords<<<fill_blocks, fill_threads>>>(kv_pages, kv_elements);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 grid(options.batch, pages_per_request, 1);
  int block_threads = 128;
  AgentPagedKvDecodeKernel<<<grid, block_threads>>>(kv_pages, device_page_table, out, options.batch, pages_per_request,
                                                   elements_per_page, options.samples_per_page);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> host_out(static_cast<size_t>(options.batch));
  CHECK_CUDA(cudaMemcpy(host_out.data(), out, host_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  size_t logical_bytes = static_cast<size_t>(options.batch) * pages_per_request * elements_per_page * sizeof(uint16_t);
  size_t unique_bytes = static_cast<size_t>(unique_pages) * elements_per_page * sizeof(uint16_t);
  std::cout << "AGENT_KV complete batch=" << options.batch << " kvlen=" << options.kvlen
            << " shared_prefix=" << options.shared_prefix << " pages_per_request=" << pages_per_request
            << " shared_pages=" << shared_pages << " private_pages=" << private_pages
            << " unique_pages=" << unique_pages << " logical_bytes=" << logical_bytes
            << " unique_bytes=" << unique_bytes << " sample=" << host_out[0] << "\n";

  CHECK_CUDA(cudaFree(kv_pages));
  CHECK_CUDA(cudaFree(device_page_table));
  CHECK_CUDA(cudaFree(out));
  return EXIT_SUCCESS;
}
