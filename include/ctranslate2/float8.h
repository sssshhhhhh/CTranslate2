#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <half_float/half.hpp>

#include "bit_cast.h"

namespace ctranslate2 {

  // Adapted from https://github.com/uxlfoundation/oneDNN/blob/v3.10.2/src/common/float8.hpp

  using float16_t = half_float::half;
  class float8_t;

  // e5m2
  class bfloat8_t {
  public:
    constexpr bfloat8_t(uint8_t bits, bool) : _bits(bits) {}

    bfloat8_t() = default;
    bfloat8_t(float f) {
      *this = f;
    }
    bfloat8_t(float16_t f) {
      *this = f;
    }

    operator float16_t() const {
      uint16_t snan_mask = 0x7d;
      uint16_t qnan_qbit = 0x02;
      const bool is_snan = (_bits & snan_mask) == snan_mask;
      const uint8_t raw = is_snan ? _bits | qnan_qbit : _bits;
      std::array<uint8_t, 2> iraw = {{0, raw}};
      auto f16 = bit_cast<float16_t>(iraw);
      return f16;
    }

    operator float() const {
      float16_t f16 = *this;
      return static_cast<float>(f16);
    }

    bfloat8_t& operator=(float16_t f) {
      // we just need to apply rounding
      uint16_t fraw = bit_cast<uint16_t>(f);
      uint16_t naninf_mask = 0x7c00;

      bool is_special = (fraw & naninf_mask) == naninf_mask;
      bool is_nan = is_special && (fraw & 0x03ff); // one of the lsb is non zero

      // we always set quiet bit for NaN
      if (is_nan) {
        _bits = (fraw >> 8) | 0x02;
        return *this;
      }

      // if infinity, we just return it as is
      if (is_special) {
        _bits = fraw >> 8;
        return *this;
      }

      // otherwise we just round and return
      int16_t rounding_nudge = 0x007f + ((fraw & 0x0100) >> 8);
      fraw = fraw + rounding_nudge;
      _bits = fraw >> 8;
      return *this;
    }

    bfloat8_t& operator=(float f) {
      float16_t f16 = static_cast<float16_t>(f);
      bfloat8_t f8 = f16;
      _bits = f8._bits;
      return *this;
    }

    bfloat8_t& operator=(float8_t f);

    private:
      uint8_t _bits;
      friend class float8_t;
  };

  // e4m3
  class float8_t {
  public:
    constexpr float8_t(uint8_t bits, bool) : _bits(bits) {}

    float8_t() = default;
    float8_t(float f) {
      *this = f;
    }
    float8_t(float16_t f) {
      *this = f;
    }

    operator float16_t() const {
      const uint16_t s8 = (_bits & 0x80) >> 7;
      const uint16_t e8 = (_bits & 0x78) >> 3;
      const uint16_t m8 = (_bits & 0x7);
      uint16_t s16 = s8;
      uint16_t e16 = e8 + 8; // 8 = 15 - 7 = f16_bias - f8_e4m3_bias
      uint16_t m16 = m8;

      // Need to convert f8_e4m3 denormal into f16 normal.
      if (e8 == 0 && m8 != 0) {
        uint16_t count = 2;
        count = m8 > 0x1 ? 1 : count;
        count = m8 > 0x3 ? 0 : count;
        e16 -= count;
        m16 = (m16 << (count + 1)) & 0x7;
      } else if (e8 == 0 && m8 == 0) {
        // set correct exponent for zero
        e16 = 0;
      } else if (e8 == 0xf && m8 == 0x7) {
        // set correct exponent and mantissa for NaN input
        e16 = 0x1f;
        m16 = 0x4; // Real Indefinite (a qNaN)
      }
      s16 <<= 15;
      e16 <<= 10;
      m16 <<= 7;

      const uint16_t u16 = s16 | e16 | m16;
      return bit_cast<float16_t>(u16);
    }

    operator float() const {
      float16_t f16 = *this;
      return static_cast<float>(f16);
    }

    float8_t& operator=(float16_t f) {
      // Here the idea is to add a large constant to the float16_t to force the
      // proper rounding to f8_e4m3 accuracy.
      uint16_t fraw = bit_cast<uint16_t>(f);

      // first we extract the sign and make the input positive
      uint8_t s8 = (fraw & 0x8000) >> 8;
      fraw = fraw & 0x7fff;

      // we filter out overflow, nan
      // Note: values in [448;464] round to 448, which is representable
      // So we overflow above 464
      if (fraw > 0x5f40) {
        _bits = s8 | 0x7f;
        return *this;
      }
      // we filter out underflow when f <= 2^-10
      if (fraw <= 0x1400) {
        _bits = s8;
        return *this;
      }

      // Compute the rounding shifter by taking its exponent + 7.
      // Lucky us, it does not overflow as fraw <= 448.
      // This allows to discard 7 bits of mantissa during addition,
      // leaving the remaining 3 bits perfectly rounded.
      uint16_t sraw = (fraw & 0x7c00) + 0x1c00;
      // e8 = e16 - e16_bias + e8_bias = e16 - 15 + 7
      // e8 will be denorm if e8 <= 0 or e16 + 7 < 16
      constexpr uint16_t exp_threshold = 0x4000; // raw bits of exponent = 16
      const bool is_denorm = sraw < exp_threshold;
      float16_t shifter = bit_cast<float16_t>(is_denorm ? exp_threshold : sraw);

      float16_t rounded = bit_cast<float16_t>(fraw) + shifter;
      // separate line to force line above to round to f16
      rounded = rounded - shifter;

      uint16_t rraw = bit_cast<uint16_t>(rounded);
      int e8 = ((rraw & 0x7c00) >> 10) - 8;
      uint8_t m8 = (rraw & 0x03ff) >> 7;

      // we need to make the implicit f32 mantissa bit explicit for
      // denorm f8_e4m3
      if (is_denorm) {
        m8 = (m8 | 0x00000008) >> (-e8 + 1);
        e8 = 0;
      }

      _bits = s8 | (e8 << 3) | m8;
      return *this;
    }

    float8_t& operator=(float f) {
      float16_t f16 = static_cast<float16_t>(f);
      float8_t f8 = f16;
      _bits = f8._bits;
      return *this;
    }

    float8_t& operator=(bfloat8_t f) {
      float16_t f16 = f;
      float8_t f8 = f16;
      _bits = f8._bits;
      return *this;
    }

    private:
      uint8_t _bits;
      friend class bfloat8_t;
  };

  inline bfloat8_t& bfloat8_t::operator=(float8_t f) {
    float16_t f16 = f;
    bfloat8_t f8 = f16;
    _bits = f8._bits;
    return *this;
  }

}
