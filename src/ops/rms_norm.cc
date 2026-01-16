#include "ctranslate2/ops/rms_norm.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    RMSNorm::RMSNorm(const float epsilon, const bool use_residual)
      : _epsilon(epsilon)
      , _use_residual(use_residual)
    {
    }

#define LOWP_CASE(T)                                                          \
      case DataTypeToEnum<T>::value: {                                        \
        if (output.device() == Device::CPU)                                   \
          throw std::invalid_argument("RMSNorm only supports FP32 on CPU");   \
        switch (input.dtype()) {                                              \
        case DataType::FLOAT32:                                               \
          compute<Device::CUDA, float, T>(gamma, input, output);              \
          break;                                                              \
        case DataType::FLOAT16:                                               \
          compute<Device::CUDA, float16_t, T>(gamma, input, output);          \
          break;                                                              \
        case DataType::BFLOAT16:                                              \
          compute<Device::CUDA, bfloat16_t, T>(gamma, input, output);         \
          break;                                                              \
        default:                                                              \
          throw std::invalid_argument("RMSNorm unsupported input type "       \
                                      + dtype_name(input.dtype()));           \
        }                                                                     \
        break;                                                                \
      }

    void RMSNorm::operator()(const StorageView& gamma,
                             const StorageView& input,
                             StorageView& output) const {
      PROFILE("RMSNorm");

      output.resize_as(input);

      switch (output.dtype()) {
      case DataType::FLOAT32:
        DEVICE_DISPATCH(output.device(), (compute<D, float, float>(gamma, input, output)));
        break;
      case DataType::FLOAT16:
        if (output.device() == Device::CPU)
          throw std::invalid_argument("RMSNorm only supports FP32 on CPU");
        compute<Device::CUDA, float16_t, float16_t>(gamma, input, output);
        break;
      case DataType::BFLOAT16:
        if (output.device() == Device::CPU)
          throw std::invalid_argument("RMSNorm only supports FP32 on CPU");
        compute<Device::CUDA, bfloat16_t, bfloat16_t>(gamma, input, output);
        break;
      LOWP_CASE(float8_t)
      LOWP_CASE(bfloat8_t)
      default:
        throw std::invalid_argument("RMSNorm only supports float types");
      }
    }

  }
}
