#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bit_cast.h"

namespace ctranslate2 {

  // Adapted from https://github.com/uxlfoundation/oneDNN/blob/v3.10.2/src/common/bfloat16.hpp

  class bfloat16_t {
  public:
    constexpr bfloat16_t(uint16_t bits, bool) : _bits(bits) {}

    bfloat16_t() = default;
    bfloat16_t(float f) {
      *this = f;
    }

    bfloat16_t& operator=(float f) {
      auto iraw = bit_cast<std::array<uint16_t, 2>>(f);
      switch (std::fpclassify(f)) {
      case FP_SUBNORMAL:
      case FP_ZERO:
        // sign preserving zero (denormal go to zero)
        _bits = iraw[1];
        _bits &= 0x8000;
        break;
      case FP_INFINITE:
        _bits = iraw[1];
        break;
      case FP_NAN:
        // truncate and set MSB of the mantissa force QNAN
        _bits = iraw[1];
        _bits |= 1 << 6;
        break;
      case FP_NORMAL:
        // round to nearest even and truncate
        const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
        const uint32_t int_raw = bit_cast<uint32_t>(f) + rounding_bias;
        iraw = bit_cast<std::array<uint16_t, 2>>(int_raw);
        _bits = iraw[1];
        break;
      }

      return *this;
    }

    operator float() const {
      std::array<uint16_t, 2> iraw = {{0, _bits}};
      return bit_cast<float>(iraw);
    }

  private:
    uint16_t _bits;
  };

}
