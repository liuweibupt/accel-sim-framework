#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_loader.h"
#include "cuda_api_object.h"
#include "gpgpu_context.h"

// These weak compatibility shims exist only to let the standalone
// trace-driven accel-sim.out link against the already-built simulator objects
// when libcuda/cuda_runtime_api.cc is temporarily unavailable in the local
// build environment. If the full libcuda path is built successfully, its
// strong definitions override these shims automatically.

void __attribute__((weak)) register_ptx_function(const char *name,
                                                 function_info *impl) {
  (void)name;
  (void)impl;
}

_cuda_device_id *__attribute__((weak)) gpgpu_context::GPGPUSim_Init() {
  _cuda_device_id *the_device = the_gpgpusim->the_cude_device;
  if (!the_device) {
    gpgpu_sim *the_gpu = gpgpu_ptx_sim_init_perf();

    cudaDeviceProp *prop = (cudaDeviceProp *)calloc(sizeof(cudaDeviceProp), 1);
    snprintf(prop->name, 256, "GPGPU-Sim_v%s", g_gpgpusim_version_string);
    prop->major = the_gpu->compute_capability_major();
    prop->minor = the_gpu->compute_capability_minor();
    prop->totalGlobalMem = 0x80000000;
    prop->memPitch = 0;
    if (prop->major >= 2) {
      prop->maxThreadsPerBlock = 1024;
      prop->maxThreadsDim[0] = 1024;
      prop->maxThreadsDim[1] = 1024;
    } else {
      prop->maxThreadsPerBlock = 512;
      prop->maxThreadsDim[0] = 512;
      prop->maxThreadsDim[1] = 512;
    }

    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 0x40000000;
    prop->maxGridSize[1] = 0x40000000;
    prop->maxGridSize[2] = 0x40000000;
    prop->totalConstMem = 0x40000000;
    prop->textureAlignment = 0;
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
#if (CUDART_VERSION > 5050)
    prop->regsPerMultiprocessor = the_gpu->num_registers_per_core();
    prop->sharedMemPerMultiprocessor = the_gpu->shared_mem_size();
#endif
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
    prop->regsPerBlock = the_gpu->num_registers_per_block();
    prop->warpSize = the_gpu->wrp_size();
#if (CUDART_VERSION >= 2010)
    prop->multiProcessorCount = the_gpu->get_config().num_shader();
#endif
#if (CUDART_VERSION >= 4000)
    prop->maxThreadsPerMultiProcessor = the_gpu->threads_per_core();
#endif
    the_gpu->set_prop(prop);
    the_gpgpusim->the_cude_device = new _cuda_device_id(the_gpu);
    the_device = the_gpgpusim->the_cude_device;
  }
  start_sim_thread(1);
  return the_device;
}

gpgpu_context *__attribute__((weak)) GPGPU_Context() {
  static gpgpu_context *gpgpu_ctx = NULL;
  if (gpgpu_ctx == NULL) {
    gpgpu_ctx = new gpgpu_context();
  }
  return gpgpu_ctx;
}

static CUctx_st *trace_driven_compat_context(gpgpu_context *ctx) {
  CUctx_st *the_context = ctx->the_gpgpusim->the_context;
  if (the_context == NULL) {
    _cuda_device_id *the_gpu = ctx->GPGPUSim_Init();
    ctx->the_gpgpusim->the_context = new CUctx_st(the_gpu);
    the_context = ctx->the_gpgpusim->the_context;
  }
  return the_context;
}

void __attribute__((weak)) ptxinfo_data::ptxinfo_addinfo() {
  CUctx_st *context = trace_driven_compat_context(gpgpu_ctx);
  if (!get_ptxinfo_kname()) {
    print_ptxinfo();
    context->add_ptxinfo(get_ptxinfo());
    clear_ptxinfo();
    return;
  }
  if (!strcmp("__cuda_dummy_entry__", get_ptxinfo_kname())) {
    clear_ptxinfo();
    return;
  }
  print_ptxinfo();
  context->add_ptxinfo(get_ptxinfo_kname(), get_ptxinfo());
  clear_ptxinfo();
}
