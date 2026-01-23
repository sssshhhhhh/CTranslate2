#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class LayerNorm : public Op {
    public:
      LayerNorm(const dim_t axis = -1, const float epsilon = 1e-5);

      void operator()(const StorageView& beta,
                      const StorageView& gamma,
                      const StorageView& input,
                      StorageView& output,
                      const float scale = 1.f) const;

      void operator()(StorageView& input, const float scale = 1.f) const;
      void operator()(const StorageView& input,
                      StorageView& output,
                      const float scale = 1.f) const;

    private:
      void operator()(const StorageView* beta,
                      const StorageView* gamma,
                      const StorageView& input,
                      StorageView& output,
                      const float scale) const;

      template <Device D, typename In, typename Out>
      void compute(const StorageView* beta,
                   const StorageView* gamma,
                   const StorageView& input,
                   const dim_t axis,
                   const dim_t outer_size,
                   const dim_t axis_size,
                   const dim_t inner_size,
                   StorageView& output,
                   const float scale) const;

      const dim_t _axis;
      const float _epsilon;
    };

  }
}
