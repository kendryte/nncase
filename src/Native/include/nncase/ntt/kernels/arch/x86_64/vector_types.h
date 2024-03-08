#pragma once
#include <immintrin.h>
namespace nncase::ntt {
template <> struct native_vector_type<float, 4> {
    using type = __m128;
    static type from_element(const float &f) { return _mm_setr_ps(f, f, f, f); }
};

template <> struct native_vector_type<float, 8> {
    using type = __m256;
};
} // namespace nncase::ntt