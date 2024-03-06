#include <immintrin.h>
template <> struct simd_type<float, 4> {
    using type = __m128;
};

template <> struct simd_type<float, 8> {
    using type = __m256;
};
