#pragma once
#include <arm_neon.h>

namespace nncase::ntt {
template <> struct native_vector_type<float, 32> {
    using type = float32x4_t[8];
};

template <> struct native_vector_type<float, 4> {
    using type = float32x4_t;
};
} // namespace nncase::ntt