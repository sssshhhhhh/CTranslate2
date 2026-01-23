#include "ctranslate2/ops/gemm.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename In, typename Out, typename Bias>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       StorageView& c,
                       const StorageView* a_shift_compensation,
                       const StorageView* bias,
                       const StorageView* residual,
                       const StorageView*,
                       const StorageView*,
                       const float) const {
      const dim_t k = a.dim(_trans_a ? -2 : -1);
      const dim_t n = b.dim(_trans_b ? -2 : -1);
      const dim_t m = a.size() / k;  // Collapse leading dimensions.
      const dim_t lda = _trans_a ? m : k;
      const dim_t ldb = _trans_b ? k : n;
      const dim_t ldc = n;

      {
        Shape output_shape(a.shape());
        output_shape[output_shape.size() - 2] = a.dim(_trans_a ? -1 : -2); // m
        output_shape[output_shape.size() - 1] = n;
        c.resize(std::move(output_shape));
      }

      primitives<D>::gemm(_a_is_packed, _b_is_packed,
                          _trans_a, _trans_b,
                          m, n, k,
                          _alpha,
                          a.data<In>(), lda,
                          b.data<In>(), ldb,
                          _beta,
                          c.data<Out>(), ldc,
                          a_shift_compensation ? a_shift_compensation->data<Out>() : nullptr);

      apply_bias_and_activation(c, bias, _activation_type, residual);
    }

#define DECLARE_IMPL(In, Out)                                                           \
    template void                                                                       \
    Gemm::compute<Device::CPU, In, Out, Out>(const StorageView& a,                      \
                                             const StorageView& b,                      \
                                             StorageView& c,                            \
                                             const StorageView* a_shift_compensation,   \
                                             const StorageView* bias,                   \
                                             const StorageView* residual,               \
                                             const StorageView* scale_a,                \
                                             const StorageView* scale_b,                \
                                             const float scale_c) const;

    DECLARE_IMPL(int8_t, int32_t)
    DECLARE_IMPL(int16_t, int32_t)
    DECLARE_IMPL(float, float)

  }
}
