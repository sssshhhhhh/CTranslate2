#pragma once

namespace ctranslate2 {

  template <typename T, typename U>
  inline T bit_cast(const U &u) {
    T t;
    std::memcpy(&t, &u, sizeof(U));
    return t;
  }
}
