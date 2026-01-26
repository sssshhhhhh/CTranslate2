#include "ctranslate2/ops/insert.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Insert::Insert(int axis, int index)
      : _axis(axis)
      , _index(index) {
    }

    void Insert::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("Insert");
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      Shape shape(output.shape());
      shape[axis] = input.dim(axis);
      if (shape != input.shape())
        throw std::runtime_error("Insert dims don't match");
      if (output.dim(axis) < _index + input.dim(axis))
        throw std::runtime_error("Insert output too small");

      DEVICE_AND_TYPE_DISPATCH(output.device(), output.dtype(), (compute<D, T>(input, output)));
    }

  }
}
