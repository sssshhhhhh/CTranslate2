#include "ctranslate2/ops/gemm.h"

#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename In, typename Out>
    void Gemm::compute(const StorageView& a,
                       const StorageView& b,
                       StorageView& c,
                       const dim_t m,
                       const dim_t n,
                       const dim_t k,
                       const dim_t lda,
                       const dim_t ldb,
                       const dim_t ldc,
                       const StorageView* a_shift_compensation,
                       const StorageView* bias,
                       const StorageView* residual) const {
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

#define DECLARE_IMPL(In, Out)                                                       \
    template void                                                                   \
    Gemm::compute<Device::CPU, In, Out>(const StorageView& a,                       \
                                        const StorageView& b,                       \
                                        StorageView& c,                             \
                                        const dim_t m,                              \
                                        const dim_t n,                              \
                                        const dim_t k,                              \
                                        const dim_t lda,                            \
                                        const dim_t ldb,                            \
                                        const dim_t ldc,                            \
                                        const StorageView* a_shift_compensation,    \
                                        const StorageView* bias,                    \
                                        const StorageView* residual) const;

    DECLARE_IMPL(int8_t, int32_t)
    DECLARE_IMPL(int16_t, int32_t)
    DECLARE_IMPL(float, float)

  }
}
