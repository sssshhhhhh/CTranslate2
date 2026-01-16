#include "ctranslate2/ops/gemm.h"

#include "ctranslate2/ops/bias_add.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // act(x + bias + residual)
    void apply_bias_and_activation(StorageView& x,
                                   const StorageView* bias,
                                   const ActivationType* activation_type,
                                   const StorageView* residual,
                                   const dim_t axis) {
      if (bias) {
        const BiasAdd bias_add_op(activation_type, axis);
        bias_add_op(x, *bias, x, residual);
      } else {
        if (residual)
          Add()(*residual, x, x);
        if (activation_type)
          get_activation_op(*activation_type)(x, x);
      }
    }


    Gemm::Gemm(float alpha,
               float beta,
               bool trans_a,
               bool trans_b,
               bool a_is_packed,
               bool b_is_packed,
               const ActivationType* activation_type)
      : _alpha(alpha)
      , _beta(beta)
      , _trans_a(trans_a)
      , _trans_b(trans_b)
      , _a_is_packed(a_is_packed)
      , _b_is_packed(b_is_packed)
      , _activation_type(activation_type)
    {
    }


#define LOWP_CASE(T)                                                                    \
      case DataTypeToEnum<T>::value: {                                                  \
        if (a.device() != Device::CUDA)                                                 \
          throw std::invalid_argument("Low precision gemm is only supported on GPU");   \
        const StorageView a_lowp = a.to(DataTypeToEnum<T>::value);                      \
        switch (c.dtype()) {                                                            \
        case DataType::FLOAT32:                                                         \
          compute<Device::CUDA, T, float, float>(a_lowp, b, c, nullptr, bias,           \
                                                 residual, scale_a, scale_b);           \
          break;                                                                        \
        case DataType::FLOAT16:                                                         \
          compute<Device::CUDA, T, float16_t, float16_t>(a_lowp, b, c, nullptr, bias,   \
                                              residual, scale_a, scale_b);              \
          break;                                                                        \
        case DataType::BFLOAT16:                                                        \
          compute<Device::CUDA, T, bfloat16_t, bfloat16_t>(a_lowp, b, c, nullptr, bias, \
                                               residual, scale_a, scale_b);             \
          break;                                                                        \
        case DataTypeToEnum<T>::value:                                                  \
          switch (bias ? bias->dtype() : DataType::FLOAT32) {                           \
          case DataType::FLOAT32:                                                       \
            compute<Device::CUDA, T, T, float>(a_lowp, b, c, nullptr, bias,             \
                                               residual, scale_a, scale_b);             \
            break;                                                                      \
          case DataType::FLOAT16:                                                       \
            compute<Device::CUDA, T, T, float16_t>(a_lowp, b, c, nullptr, bias,         \
                                                   residual, scale_a, scale_b);         \
            break;                                                                      \
          case DataType::BFLOAT16:                                                      \
            compute<Device::CUDA, T, T, bfloat16_t>(a_lowp, b, c, nullptr, bias,        \
                                                    residual, scale_a, scale_b);        \
            break;                                                                      \
          default:                                                                      \
            throw std::invalid_argument("Gemm unsupported bias type "                   \
                                        + dtype_name(bias->dtype()));                   \
            break;                                                                      \
          }                                                                             \
          break;                                                                        \
        default:                                                                        \
          throw std::invalid_argument("Gemm unsupported output type "                   \
                                      + dtype_name(c.dtype()));                         \
        }                                                                               \
        break;                                                                          \
      }

    void Gemm::operator()(const StorageView& a,
                          const StorageView& b,
                          StorageView& c,
                          const StorageView* a_shift_compensation,
                          const StorageView* bias,
                          const StorageView* residual,
                          const StorageView* scale_a,
                          const StorageView* scale_b) const {
      PROFILE("Gemm");

      switch (b.dtype()) {
      case DataType::INT8:
        DEVICE_DISPATCH(a.device(),
                        (compute<D, int8_t, int32_t, int32_t>(a, b, c, a_shift_compensation, bias, residual)));
        break;

      case DataType::INT16:
        if (a.device() != Device::CPU)
          throw std::invalid_argument("INT16 GEMM is only supported on CPU");
        compute<Device::CPU, int16_t, int32_t, int32_t>(a, b, c, a_shift_compensation, bias, residual);
        break;

      case DataType::FLOAT32:
      case DataType::FLOAT16:
      case DataType::BFLOAT16: {
        DEVICE_AND_FLOAT_DISPATCH("Gemm", a.device(), a.dtype(),
                                  (compute<D, T, T, T>(a, b, c, a_shift_compensation, bias, residual)));
        break;
      }

      LOWP_CASE(float8_t)
      LOWP_CASE(bfloat8_t)

      default:
        throw std::invalid_argument("Gemm: unsupported input type " + dtype_name(a.dtype()));
      }
    }

    template <typename T>
    static void pack_b(const StorageView& b,
                       const bool transpose,
                       const dim_t k,
                       const dim_t n,
                       const float alpha,
                       StorageView& packed) {
      const T* src = b.data<T>();
      const dim_t pack_bytes = primitives<Device::CPU>::gemm_pack_b(src,
                                                                    transpose,
                                                                    k, n,
                                                                    alpha);

      if (pack_bytes == 0)  // Packed Gemm is not supported.
        throw std::runtime_error("Packed GEMM APIs are not supported by this GEMM backend");

      const dim_t pack_size = pack_bytes / sizeof (T);
      const dim_t b_size = b.size();

      // We want the packed storage to have the same shape as the original weight
      // so that operators can query its shape, but also have enough space to store
      // the packed data.
      packed.reserve(std::max(b_size, pack_size));
      packed.resize_as(b);

      primitives<Device::CPU>::gemm_pack_b(src,
                                           transpose,
                                           k, n,
                                           alpha,
                                           packed.data<T>());
    }

    StorageView Gemm::pack_b_input(const StorageView& b,
                                   const bool transpose,
                                   const dim_t k,
                                   const dim_t n,
                                   const float alpha) {
      if (b.device() != Device::CPU)
        throw std::invalid_argument("Packed GEMM APIs are only defined on CPU");

      DataType dtype = b.dtype();
      StorageView packed(dtype);

      switch (dtype) {
      case DataType::FLOAT32:
        pack_b<float>(b, transpose, k, n, alpha, packed);
        break;
      case DataType::INT16:
        pack_b<int16_t>(b, transpose, k, n, alpha, packed);
        break;
      case DataType::INT8:
        pack_b<int8_t>(b, transpose, k, n, alpha, packed);
        break;
      default:
        throw std::invalid_argument("Cannot pack GEMM input of type " + dtype_name(dtype));
        break;
      }

      return packed;
    }

    StorageView Gemm::compensate_u8_input(const StorageView& b,
                                          const bool transpose,
                                          const dim_t k,
                                          const dim_t n,
                                          const float alpha) {
      if (b.device() != Device::CPU && b.dtype() != DataType::INT8)
        throw std::invalid_argument("Unsigned input compensation is only defined for "
                                    "INT8 GEMM on CPU");

      StorageView compensation({n}, DataType::INT32);
      primitives<Device::CPU>::compute_u8_compensation(b.data<int8_t>(),
                                                       transpose,
                                                       k, n,
                                                       alpha,
                                                       compensation.data<int32_t>());
      return compensation;
    }

  }
}
