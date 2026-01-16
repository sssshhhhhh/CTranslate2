#include "ctranslate2/ops/dequantize.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Dequantize::Dequantize(const ActivationType* activation_type)
      : _activation_type(activation_type)
    {
    }

    void Dequantize::operator()(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      PROFILE("Dequantize");
      output.resize_as(input);

      switch (input.dtype()) {
      case DataType::INT16: {
        if (input.device() != Device::CPU)
          throw std::invalid_argument("INT16 dequantization is only supported on CPU");
        if (!scale.is_scalar())
          throw std::invalid_argument("INT16 quantization scale should be a scalar value");
        dequantize<Device::CPU, int16_t, float>(input, scale, output);
        break;
      }

      case DataType::INT8: {
        const dim_t batch_size = input.size() / input.dim(-1);
        if (scale.size() != batch_size)
          throw std::invalid_argument("INT8 dequantization expects per-batch scales");

        DEVICE_AND_FLOAT_DISPATCH("Dequantize", output.device(), output.dtype(),
                                  (dequantize<D, int8_t, T>(input, scale, output)));

        break;
      }

      case DataType::FLOAT8:
      case DataType::BFLOAT8: {
        if (input.device() != Device::CPU)
          throw std::invalid_argument("FP8 dequantization is only supported on CPU");
        const dim_t batch_size = input.size() / input.dim(-1);
        if (!scale.is_scalar() && scale.size() != batch_size)
          throw std::invalid_argument("FP8 dequantization expects scalar or per-batch scales");

        if (output.dtype() == DataType::FLOAT8)
          dequantize<Device::CPU, float8_t, float>(input, scale, output);
        else
          dequantize<Device::CPU, bfloat8_t, float>(input, scale, output);
        break;
      }

      default:
        throw std::invalid_argument("Dequantize: invalid quantized type " + dtype_name(input.dtype())
                                    + ", expected int8, int16, float8, or bfloat8");
      }
    }

    void Dequantize::operator()(const StorageView& c,
                                const StorageView& a_scale,
                                const StorageView& b_scale,
                                const bool transpose_a,
                                const bool transpose_b,
                                StorageView& y,
                                const StorageView* bias) const {
      PROFILE("DequantizeGemmOutput");
      y.resize_as(c);

      DEVICE_AND_FLOAT_DISPATCH(
        "DequantizeGemmOutput", y.device(), y.dtype(),
        (dequantize_gemm_output<D, T>(c, a_scale, b_scale, transpose_a, transpose_b, bias, y)));
    }

  }
}
