#include <arm_neon.h>
template <> struct simd_type<float, 4> {
    using type = float32x4_t;
};

template <> struct simd_type<float, 2> {
    using type = float32x2_t;
};

