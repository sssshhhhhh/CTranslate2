#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Mul : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        c.resize_as(a);
        if (is_lowp_type(c.dtype())
            && !std::is_same_v<T, float8_t>
            && !std::is_same_v<T, bfloat8_t>) {
          if (a.device() != Device::CUDA)
            throw std::invalid_argument("Low precision Mul output is only supported on GPU");
          switch (c.dtype()) {
            case DataType::FLOAT8:
              primitives<Device::CUDA>::mul(a.data<T>(), b.data<T>(), c.data<float8_t>(), c.size());
              break;
            case DataType::BFLOAT8:
              primitives<Device::CUDA>::mul(a.data<T>(), b.data<T>(), c.data<bfloat8_t>(), c.size());
              break;
            default:
              throw std::invalid_argument("Mul invalid output type " + dtype_name(c.dtype()));
          }
        } else if (b.is_scalar()) {
          primitives<D>::mul(b.data<T>()[0], a.data<T>(), c.data<T>(), c.size());
        } else {
          primitives<D>::mul(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
        }
      }
    };

  }
}
