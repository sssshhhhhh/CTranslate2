#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "bit_cast.h"

namespace ctranslate2 {

  // Adapted from https://github.com/uxlfoundation/oneDNN/blob/v3.10.2/src/common/float16.hpp

  class float16_t {
  public:
    uint16_t _bits;

    constexpr float16_t(uint16_t bits, bool) : _bits(bits) {}

    float16_t() = default;
    float16_t(float f) {
      *this = f;
    }

    float16_t& operator=(float f) {
      uint32_t i = bit_cast<uint32_t>(f);
      uint32_t s = i >> 31;
      uint32_t e = (i >> 23) & 0xFF;
      uint32_t m = i & 0x7FFFFF;

      uint32_t ss = s;
      uint32_t mm = m >> 13;
      uint32_t r = m & 0x1FFF;
      uint32_t ee = 0;
      int32_t eee = static_cast<int32_t>((e - 127) + 15);

      if (e == 0) {
        // Denormal/zero floats all become zero.
        ee = 0;
        mm = 0;
      } else if (e == 0xFF) {
        // Preserve inf/nan, but set quiet bit for nan (snan->qnan).
        ee = 0x1F;
        if (m != 0) mm |= 0x200;
      } else if (eee > 0 && eee < 0x1F) {
        // Normal range. Perform round to even on mantissa.
        ee = static_cast<uint32_t>(eee);
        if (r > (0x1000 - (mm & 1))) {
          // Round up.
          mm++;
          if (mm == 0x400) {
            // Rounds up to next dyad (or inf).
            mm = 0;
            ee++;
          }
        }
      } else if (eee >= 0x1F) {
        // Overflow.
        ee = 0x1F;
        mm = 0;
      } else {
        // Underflow.
        float ff = fabsf(f) + 0.5f;
        uint32_t ii = bit_cast<uint32_t>(ff);
        ee = 0;
        mm = ii & 0x7FF;
      }

      this->_bits = static_cast<uint16_t>((ss << 15) | (ee << 10) | mm);
      return *this;
    }

    operator float() const {
      uint32_t ss = _bits >> 15;
      uint32_t ee = (_bits >> 10) & 0x1F;
      uint32_t mm = _bits & 0x3FF;

      uint32_t s = ss;
      uint32_t eee = ee - 15 + 127;
      uint32_t m = mm << 13;
      uint32_t e;

      if (ee == 0) {
        if (mm == 0)
          e = 0;
        else {
          // Half denormal -> float normal
          return (ss ? -1 : 1) * std::scalbn((float)mm, -24);
        }
      } else if (ee == 0x1F) {
        // inf/nan
        e = 0xFF;
        // set quiet bit for nan (snan->qnan)
        if (m != 0) m |= 0x400000;
      } else
        e = eee;

      uint32_t f = (s << 31) | (e << 23) | m;

      return bit_cast<float>(f);
    }
  };

}
