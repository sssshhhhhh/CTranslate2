#include "ctranslate2/ops/layer_norm.h"

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

    constexpr dim_t num_threads = 256;

    template <typename In, typename Out>
    __global__ void layer_norm_kernel(const In* input,
                                      Out* output,
                                      const In* beta,
                                      const In* gamma,
                                      cuda::index_t axis_size,
                                      const float epsilon,
                                      const float scale) {
      typedef cub::BlockReduce<float, num_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage m_temp_storage;
      __shared__ typename BlockReduce::TempStorage v_temp_storage;
      __shared__ float s_mean;
      __shared__ float s_var;

      input += blockIdx.x * axis_size;
      output += blockIdx.x * axis_size;

      float sum1 = 0;
      float sum2 = 0;
      for (cuda::index_t i = threadIdx.x; i < axis_size; i += blockDim.x) {
        const float v = float(input[i]);
        sum1 += v;
        sum2 += v * v;
      }
      sum1 = BlockReduce(m_temp_storage).Sum(sum1);
      sum2 = BlockReduce(v_temp_storage).Sum(sum2);
      if (threadIdx.x == 0) {
        const float r_size = 1.f / float(axis_size);
        sum1 *= r_size;
        sum2 = fmaxf(sum2 * r_size - sum1 * sum1, 0.f);
        s_mean = sum1;
        s_var = rsqrtf(sum2 + epsilon);
      }

      __syncthreads();

      for (cuda::index_t i = threadIdx.x; i < axis_size; i += blockDim.x) {
        const float gamma_v = gamma == nullptr ? 1.f : float(gamma[i]);
        const float beta_v = beta == nullptr ? 0.f : float(beta[i]);
        output[i] = ((float(input[i]) - s_mean) * s_var * gamma_v + beta_v) * scale;
      }
    }

    template <typename In, typename Out>
    __global__ void layer_norm_axis_kernel(const In* input,
                                           Out* output,
                                           const In* beta,
                                           const In* gamma,
                                           cuda::index_t axis_size,
                                           cuda::index_t inner_size,
                                           const float epsilon,
                                           const float scale) {
      typedef cub::BlockReduce<float, num_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage m_temp_storage;
      __shared__ typename BlockReduce::TempStorage v_temp_storage;
      __shared__ float s_mean;
      __shared__ float s_var;

      const cuda::index_t feature_idx = blockIdx.x % inner_size;
      const cuda::index_t offset = blockIdx.x / inner_size * axis_size * inner_size + feature_idx;
      input += offset;
      output += offset;

      float sum1 = 0;
      float sum2 = 0;
      for (cuda::index_t i = threadIdx.x; i < axis_size; i += blockDim.x) {
        const float v = float(input[i * inner_size]);
        sum1 += v;
        sum2 += v * v;
      }
      sum1 = BlockReduce(m_temp_storage).Sum(sum1);
      sum2 = BlockReduce(v_temp_storage).Sum(sum2);
      if (threadIdx.x == 0) {
        const float r_size = 1.f / float(axis_size);
        sum1 *= r_size;
        sum2 = fmaxf(sum2 * r_size - sum1 * sum1, 0.f);
        s_mean = sum1;
        s_var = rsqrtf(sum2 + epsilon);
      }

      __syncthreads();

      const float gamma_v = gamma == nullptr ? 1.f : float(gamma[feature_idx]);
      const float beta_v = beta == nullptr ? 0.f : float(beta[feature_idx]);
      for (cuda::index_t i = threadIdx.x; i < axis_size; i += blockDim.x)
        output[i * inner_size] = ((float(input[i * inner_size]) - s_mean)
                                  * s_var * gamma_v + beta_v) * scale;
    }

    template <Device D, typename In, typename Out>
    void LayerNorm::compute(const StorageView* beta,
                            const StorageView* gamma,
                            const StorageView& input,
                            const dim_t axis,
                            const dim_t outer_size,
                            const dim_t axis_size,
                            const dim_t inner_size,
                            StorageView& output,
                            const float scale) const {
      if (axis == input.rank() - 1) {
        layer_norm_kernel<<<outer_size, num_threads, 0, cuda::get_cuda_stream()>>>(
          cuda::device_cast(input.data<In>()),
          cuda::device_cast(output.data<Out>()),
          beta ? cuda::device_cast(beta->data<In>()) : nullptr,
          gamma ? cuda::device_cast(gamma->data<In>()) : nullptr,
          axis_size, _epsilon, scale);
      } else {
        const dim_t blocks = std::min(outer_size * inner_size, cuda::max_blocks);
        layer_norm_axis_kernel<<<blocks, num_threads, 0, cuda::get_cuda_stream()>>>(
          cuda::device_cast(input.data<In>()),
          cuda::device_cast(output.data<Out>()),
          beta ? cuda::device_cast(beta->data<In>()) : nullptr,
          gamma ? cuda::device_cast(gamma->data<In>()) : nullptr,
          axis_size, inner_size, _epsilon, scale);
      }
    }

#define DECLARE_IMPL(In, Out)                                           \
    template void                                                       \
    LayerNorm::compute<Device::CUDA, In, Out>(const StorageView* beta,  \
                                              const StorageView* gamma, \
                                              const StorageView& input, \
                                              const dim_t axis,         \
                                              const dim_t outer_size,   \
                                              const dim_t axis_size,    \
                                              const dim_t inner_size,   \
                                              StorageView& output,      \
                                              const float scale) const;

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
