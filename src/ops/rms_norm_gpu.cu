#include "ctranslate2/ops/rms_norm.h"

#ifdef CT2_USE_HIP
#include <hipcub/hipcub.hpp>
#include <hipcub/block/block_reduce.hpp>
#define cub hipcub
#else
#include <cub/block/block_reduce.cuh>
#endif

#include "cuda/helpers.h"
#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    constexpr dim_t num_threads = 512;

    template <typename In, typename Out>
    __global__ void rms_norm_kernel(const In* input,
                                    const In* gamma,
                                    Out* output,
                                    cuda::index_t depth,
                                    float epsilon,
                                    bool use_residual) {
      typedef cub::BlockReduce<float, num_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ float s_inv_rms;

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      float sum_squares = 0;
      for (cuda::index_t i = threadIdx.x; i < depth; i += blockDim.x)
        sum_squares += float(input[i]) * float(input[i]);
      sum_squares = BlockReduce(temp_storage).Sum(sum_squares);

      if (threadIdx.x == 0)
        s_inv_rms = rsqrtf(sum_squares / depth + epsilon);

      __syncthreads();

      for (cuda::index_t i = threadIdx.x; i < depth; i += blockDim.x)
        if (use_residual)
          output[i] = float(input[i]) * s_inv_rms * (1 + float(gamma[i]));
        else
          output[i] = float(input[i]) * s_inv_rms * float(gamma[i]);
    }

    template <Device D, typename In, typename Out>
    void RMSNorm::compute(const StorageView& gamma,
                          const StorageView& input,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      rms_norm_kernel<<<batch_size, num_threads, 0, cuda::get_cuda_stream()>>>(
        cuda::device_cast(input.data<In>()),
        cuda::device_cast(gamma.data<In>()),
        cuda::device_cast(output.data<Out>()),
        depth,
        _epsilon,
        _use_residual);
    }

#define DECLARE_IMPL(In, Out)                                                 \
    template void RMSNorm::compute<Device::CUDA, In, Out>(const StorageView&, \
                                                          const StorageView&, \
                                                          StorageView&) const;

    DECLARE_IMPL(float, float)
    DECLARE_IMPL(float16_t, float16_t)
    DECLARE_IMPL(bfloat16_t, bfloat16_t)
    DECLARE_IMPL(float, float8_t)
    DECLARE_IMPL(float16_t, float8_t)
    DECLARE_IMPL(bfloat16_t, float8_t)
    DECLARE_IMPL(float, bfloat8_t)
    DECLARE_IMPL(float16_t, bfloat8_t)
    DECLARE_IMPL(bfloat16_t, bfloat8_t)

  }
}
