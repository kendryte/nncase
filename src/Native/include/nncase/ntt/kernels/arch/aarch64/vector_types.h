#pragma once
#include <arm_neon.h>

namespace nncase::ntt {
template <> struct native_vector_type<float, 32> {
    using type = float32x4_t[8];
};

template <> struct native_vector_type<float, 8> {
    using type = float32x4x2_t;
    static type from_element(const float &f) {
        return type{vdupq_n_f32(f), vdupq_n_f32(f)};
    }
};

template <> struct native_vector_type<float, 4> {
    using type = float32x4_t;
    static type from_element(const float &f) { return vdupq_n_f32(f); }
};
} // namespace nncase::ntt