#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Insert : public Op {
    public:
      Insert(int axis, int index);
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      const dim_t _axis;
      const dim_t _index;

      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output) const;
    };

  }
}
