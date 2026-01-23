#include "ctranslate2/ops/rms_norm.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    RMSNorm::RMSNorm(const float epsilon, const bool use_residual)
      : _epsilon(epsilon)
      , _use_residual(use_residual)
    {
    }

#define LOWP_CASE(Out)                                                                  \
  case DataTypeToEnum<Out>::value: {                                                    \
    if (output.device() == Device::CPU)                                                 \
      throw std::invalid_argument("RMSNorm only supports FP32 on CPU");                 \
    FLOAT_DISPATCH("RMSNorm", input.dtype(),                                            \
                   (compute<Device::CUDA, T, Out>(gamma, input, output, scale)));       \
    break;                                                                              \
  }

    void RMSNorm::operator()(const StorageView& gamma,
                             const StorageView& input,
                             StorageView& output,
                             const float scale) const {
      PROFILE("RMSNorm");

      output.resize_as(input);

      switch (output.dtype()) {
      case DataType::FLOAT32:
        DEVICE_DISPATCH(output.device(), (compute<D, float, float>(gamma, input, output, scale)));
        break;
      case DataType::FLOAT16:
        if (output.device() == Device::CPU)
          throw std::invalid_argument("RMSNorm only supports FP32 on CPU");
        compute<Device::CUDA, float16_t, float16_t>(gamma, input, output, scale);
        break;
      case DataType::BFLOAT16:
        if (output.device() == Device::CPU)
          throw std::invalid_argument("RMSNorm only supports FP32 on CPU");
        compute<Device::CUDA, bfloat16_t, bfloat16_t>(gamma, input, output, scale);
        break;
      LOWP_CASE(float8_t)
      LOWP_CASE(bfloat8_t)
      default:
        throw std::invalid_argument("RMSNorm only supports float types");
      }
    }

  }
}
