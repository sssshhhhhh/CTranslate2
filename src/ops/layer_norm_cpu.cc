#include "ctranslate2/ops/layer_norm.h"

#include "cpu/kernels.h"

namespace ctranslate2 {
  namespace ops {

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
      if (scale != 1.f)
        throw std::invalid_argument("LayerNorm scale not supported on CPU");

      if (axis == input.rank() - 1 && beta && gamma) {
        CPU_ISA_DISPATCH((cpu::layer_norm<ISA>(input.data<In>(),
                                               gamma->data<In>(),
                                               beta->data<In>(),
                                               output.data<Out>(),
                                               outer_size,
                                               axis_size,
                                               _epsilon)));
      } else {
        CPU_ISA_DISPATCH((cpu::layer_norm_axis<ISA>(input.data<In>(),
                                                    gamma ? gamma->data<In>() : nullptr,
                                                    beta ? beta->data<In>() : nullptr,
                                                    output.data<Out>(),
                                                    outer_size,
                                                    axis_size,
                                                    inner_size,
                                                    _epsilon)));
      }
    }

#define DECLARE_IMPL(T)                                             \
    template void                                                   \
    LayerNorm::compute<Device::CPU, T, T>(const StorageView* beta,  \
                                          const StorageView* gamma, \
                                          const StorageView& input, \
                                          const dim_t axis,         \
                                          const dim_t outer_size,   \
                                          const dim_t axis_size,    \
                                          const dim_t inner_size,   \
                                          StorageView& output,      \
                                          const float scale) const;

    DECLARE_IMPL(float)

  }
}
