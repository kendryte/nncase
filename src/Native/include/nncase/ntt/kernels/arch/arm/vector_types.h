#include <arm_neon.h>

template <> struct native_vector_type<float, 32> {
    using type = float32x4_t[8];
};

template <> struct native_vector_type<float, 4> {
    using type = float32x4_t;
};
