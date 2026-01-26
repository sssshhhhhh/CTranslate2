#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Concat : public Op {
    public:
      Concat(int axis, int padding = 0);
      void operator()(const std::vector<const StorageView*>& inputs,
                      StorageView& output) const;

    private:
      const dim_t _axis;
      const dim_t _padding;

      template <Device D, typename T>
      void compute(const std::vector<const StorageView*>& inputs, StorageView& output) const;
    };

  }
}
