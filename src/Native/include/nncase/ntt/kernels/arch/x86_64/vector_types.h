#pragma once
#include <immintrin.h>
template <> struct native_vector_type<float, 4> {
    using type = __m128;
};

template <> struct native_vector_type<float, 8> {
    using type = __m256;
};
