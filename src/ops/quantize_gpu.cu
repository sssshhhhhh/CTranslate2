#include "ctranslate2/ops/quantize.h"

#ifdef CT2_USE_HIP
#include <hipcub/block/block_reduce.hpp>
#define cub hipcub
#else
#include <cub/block/block_reduce.cuh>
#endif

#include "cuda/helpers.h"

namespace ctranslate2 {
  namespace ops {

    constexpr dim_t num_threads = 256;

    template <typename T>
    struct absolute_maximum_func {
      __device__ __forceinline__ T operator()(T a, T b) const {
        return fmaxf(fabsf(a), fabsf(b));
      }
    };

#if CUDA_CAN_USE_HALF
    template<>
    struct absolute_maximum_func<__half> {
      __device__ __forceinline__ __half operator()(__half a, __half b) const {
        a = __habs(a);
        b = __habs(b);
        return a > b ? a : b;
      }
    };
#endif

#if CUDA_CAN_USE_BF16_MATH
    template<>
    struct absolute_maximum_func<__nv_bfloat16> {
      __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __hmax(__habs(a), __habs(b));
      }
    };
#endif

    struct rescale_func {
      __device__ __forceinline__ rescale_func(float scale)
        : _scale(scale) {
      }
      __device__ __forceinline__ float operator()(float v) const {
        return v * _scale;
      }
    private:
      const float _scale;
    };

    struct rescale_and_round_func {
      __device__ __forceinline__ rescale_and_round_func(float scale)
        : _scale(scale) {
      }
      __device__ __forceinline__ float operator()(float v) const {
        return nearbyintf(v * _scale);
      }
    private:
      const float _scale;
    };

    template <typename T>
    __global__ void quantize_kernel(const T* input,
                                    cuda::index_t depth,
                                    float* scales,
                                    int8_t* output,
                                    bool round_before_cast) {
      typedef cub::BlockReduce<T, num_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ float scale;

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      T thread_max = cuda::ilp_reduce(input, depth, absolute_maximum_func<T>(), T(0.f));
      float max = BlockReduce(temp_storage).Reduce(thread_max, cuda::maximum<T>());
      if (threadIdx.x == 0) {
        scale = max != 0.f ? 127.f / max : 1.f;
        scales[blockIdx.x] = scale;
      }

      __syncthreads();

      if (round_before_cast)
        cuda::apply_epilogue(input, depth, rescale_and_round_func(scale), output);
      else
        cuda::apply_epilogue(input, depth, rescale_func(scale), output);
    }

    template <typename InT, typename OutT>
    __global__ void quantize_lowp_kernel(const InT* input,
                                         cuda::index_t depth,
                                         float* scales,
                                         OutT* output) {
      typedef cub::BlockReduce<InT, num_threads> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      __shared__ float r_scale;

      input += blockIdx.x * depth;
      output += blockIdx.x * depth;

      InT thread_max = cuda::ilp_reduce(input, depth, absolute_maximum_func<InT>(), InT(0.f));
      float max = BlockReduce(temp_storage).Reduce(thread_max, cuda::maximum<InT>());
      if (threadIdx.x == 0) {
        constexpr float fmax = 448;
        const float scale = max != 0.f ? max / fmax : 1.f;
        scales[blockIdx.x] = scale;
        r_scale = 1.f / scale;
      }

      __syncthreads();

      cuda::apply_epilogue(input, depth, rescale_func(r_scale), output);
    }

    template <Device D, typename InT, typename OutT>
    void Quantize::quantize(const StorageView& input,
                            StorageView& output,
                            StorageView& scale,
                            const ScaleType) const {
      if (_shift_to_uint8)
        throw std::invalid_argument("Shift to uin8_t is not defined on CUDA");

      const dim_t batch_size = scale.size();
      const dim_t depth = input.dim(-1);

      if constexpr (std::is_same_v<OutT, int8_t>) {
        quantize_kernel<<<batch_size, num_threads, 0, cuda::get_cuda_stream()>>>(
          cuda::device_cast<InT>(input.data<InT>()),
          depth,
          scale.data<float>(),
          cuda::device_cast<OutT>(output.data<OutT>()),
          _round_before_cast);
      } else {
        quantize_lowp_kernel<<<batch_size, num_threads, 0, cuda::get_cuda_stream()>>>(
          cuda::device_cast<InT>(input.data<InT>()),
          depth,
          scale.data<float>(),
          cuda::device_cast<OutT>(output.data<OutT>()));
      }
    }

#define DECLARE_IMPL(T)                                                       \
    template void                                                             \
    Quantize::quantize<Device::CUDA, T, int8_t>(const StorageView&,           \
                                                StorageView&,                 \
                                                StorageView&,                 \
                                                const ScaleType) const;       \
    template void                                                             \
    Quantize::quantize<Device::CUDA, T, float8_t>(const StorageView&,         \
                                                  StorageView&,               \
                                                  StorageView&,               \
                                                  const ScaleType) const;     \
    template void                                                             \
    Quantize::quantize<Device::CUDA, T, bfloat8_t>(const StorageView&,        \
                                                   StorageView&,              \
                                                   StorageView&,              \
                                                   const ScaleType) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
