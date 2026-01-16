#include "ctranslate2/ops/layer_norm.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    LayerNorm::LayerNorm(const dim_t axis, const float epsilon)
      : _axis(axis)
      , _epsilon(epsilon)
    {
    }

    void LayerNorm::operator()(const StorageView& beta,
                               const StorageView& gamma,
                               const StorageView& input,
                               StorageView& output) const {
      operator()(&beta, &gamma, input, output);
    }

    void LayerNorm::operator()(StorageView& input) const {
      operator()(input, input);
    }

    void LayerNorm::operator()(const StorageView& input, StorageView& output) const {
      operator()(nullptr, nullptr, input, output);
    }

#define LOWP_CASE(T)                                                                    \
      case DataTypeToEnum<T>::value: {                                                  \
        if (output.device() == Device::CPU)                                             \
          throw std::invalid_argument("LayerNorm only supports FP32 on CPU");           \
        switch (input.dtype()) {                                                        \
        case DataType::FLOAT32:                                                         \
          compute<Device::CUDA, float, T>(beta, gamma, input, axis, outer_size,         \
                                          axis_size, inner_size, output);               \
          break;                                                                        \
        case DataType::FLOAT16:                                                         \
          compute<Device::CUDA, float16_t, T>(beta, gamma, input, axis, outer_size,     \
                                              axis_size, inner_size, output);           \
          break;                                                                        \
        case DataType::BFLOAT16:                                                        \
          compute<Device::CUDA, bfloat16_t, T>(beta, gamma, input, axis, outer_size,    \
                                          axis_size, inner_size, output);               \
          break;                                                                        \
        default:                                                                        \
          throw std::invalid_argument("LayerNorm unsupported input type "               \
                                      + dtype_name(input.dtype()));                     \
        }                                                                               \
        break;                                                                          \
      }

    void LayerNorm::operator()(const StorageView* beta,
                               const StorageView* gamma,
                               const StorageView& input,
                               StorageView& output) const {
      PROFILE("LayerNorm");
      output.resize_as(input);

      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t axis_size = input.dim(axis);

      dim_t inner_size = 1;
      dim_t outer_size = 1;
      for (dim_t i = 0; i < axis; ++i)
        outer_size *= input.dim(i);
      for (dim_t i = axis + 1; i < input.rank(); ++i)
        inner_size *= input.dim(i);

      switch (output.dtype()) {
      case DataType::FLOAT32:
        DEVICE_DISPATCH(output.device(), (compute<D, float, float>(beta, gamma, input, axis, outer_size,
                                                  axis_size, inner_size, output)));
        break;
      case DataType::FLOAT16:
        if (output.device() == Device::CPU)
          throw std::invalid_argument("LayerNorm only supports FP32 on CPU");
        compute<Device::CUDA, float16_t, float16_t>(beta, gamma, input, axis, outer_size,
                                                    axis_size, inner_size, output);
        break;
      case DataType::BFLOAT16:
        if (output.device() == Device::CPU)
          throw std::invalid_argument("LayerNorm only supports FP32 on CPU");
        compute<Device::CUDA, bfloat16_t, bfloat16_t>(beta, gamma, input, axis, outer_size,
                                                      axis_size, inner_size, output);
        break;
      LOWP_CASE(float8_t)
      LOWP_CASE(bfloat8_t)
      default:
        throw std::invalid_argument("LayerNorm only supports float types" + dtype_name(output.dtype()) + dtype_name(input.dtype()));
      }
    }

  }
}
