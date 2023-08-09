#pragma once
#include "thread_pool.h"
#include <functional>
#include <math.h>
#include <nncase/runtime/cpu/compiler_defs.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

BEGIN_NS_NNCASE_RT_MODULE(cpu)

typedef struct nncase_method_table {
    // float unary
    float (*float_unary_abs)(float);
    float (*float_unary_acos)(float);
    float (*float_unary_acosh)(float);
    float (*float_unary_asin)(float);
    float (*float_unary_asinh)(float);
    float (*float_unary_ceil)(float);
    float (*float_unary_cos)(float);
    float (*float_unary_cosh)(float);
    float (*float_unary_exp)(float);
    float (*float_unary_floor)(float);
    float (*float_unary_log)(float);
    float (*float_unary_logical_not)(float);
    float (*float_unary_neg)(float);
    float (*float_unary_round)(float);
    float (*float_unary_rsqrt)(float);
    float (*float_unary_sign)(float);
    float (*float_unary_sin)(float);
    float (*float_unary_sinh)(float);
    float (*float_unary_sqrt)(float);
    float (*float_unary_square)(float);
    float (*float_unary_tanh)(float);
    // float bianry
    float (*float_binary_add)(float, float);
    float (*float_binary_sub)(float, float);
    float (*float_binary_mul)(float, float);
    float (*float_binary_div)(float, float);
    float (*float_binary_min)(float, float);
    float (*float_binary_max)(float, float);
    float (*float_binary_pow)(float, float);
    float (*float_binary_logical_and)(float, float);
    float (*float_binary_mod)(float, float);
    // int32 bianry
    int32_t (*int32_binary_add)(int32_t, int32_t);
    int32_t (*int32_binary_sub)(int32_t, int32_t);
    int32_t (*int32_binary_mul)(int32_t, int32_t);
    int32_t (*int32_binary_div)(int32_t, int32_t);
    int32_t (*int32_binary_min)(int32_t, int32_t);
    int32_t (*int32_binary_max)(int32_t, int32_t);
    int32_t (*int32_binary_pow)(int32_t, int32_t);
    int32_t (*int32_binary_logical_and)(int32_t, int32_t);
    int32_t (*int32_binary_mod)(int32_t, int32_t);
    // int64 bianry
    int64_t (*int64_binary_add)(int64_t, int64_t);
    int64_t (*int64_binary_sub)(int64_t, int64_t);
    int64_t (*int64_binary_mul)(int64_t, int64_t);
    int64_t (*int64_binary_div)(int64_t, int64_t);
    int64_t (*int64_binary_min)(int64_t, int64_t);
    int64_t (*int64_binary_max)(int64_t, int64_t);
    int64_t (*int64_binary_pow)(int64_t, int64_t);
    int64_t (*int64_binary_logical_and)(int64_t, int64_t);
    int64_t (*int64_binary_mod)(int64_t, int64_t);
    // bool binary
    bool (*bool_binary_and)(bool, bool);
    bool (*bool_binary_or)(bool, bool);
    bool (*bool_binary_xor)(bool, bool);

    // multi-thread
    void *(*thread_start)(void *(*callable)(void *), void *user, size_t user_size);
    void *(*thread_end)();
} nncase_mt_t;

typedef struct buffer {
    void *vaddr;
    size_t paddr;
    uint32_t *shape;
    uint32_t *stride;
    uint32_t rank;
} buffer_t;

inline float float_unary_logical_not(float x) { return !x; }
inline float float_unary_neg(float x) { return std::negate<float>()(x); }
inline float float_unary_rsqrt(float x) { return 1.f / sqrtf(x); }
inline float float_unary_sign(float x) { return (0.f < x) - (x < 0.f); }
inline float float_unary_square(float x) { return x * x; }

inline float float_binary_add(float x, float y) { return x + y; }
inline float float_binary_sub(float x, float y) { return x - y; }
inline float float_binary_mul(float x, float y) { return x * y; }
inline float float_binary_div(float x, float y) { return x / y; }
inline float float_binary_min(float x, float y) { return std::min(x, y); }
inline float float_binary_max(float x, float y) { return std::max(x, y); }
inline float float_binary_pow(float x, float y) { return powf(x, y); }
inline float float_binary_logical_and(float x, float y) { return x && y; }
inline float float_binary_mod(float x, float y) { return fmod(x, y); }

inline int32_t int32_binary_add(int32_t x, int32_t y) { return x + y; }
inline int32_t int32_binary_sub(int32_t x, int32_t y) { return x - y; }
inline int32_t int32_binary_mul(int32_t x, int32_t y) { return x * y; }
inline int32_t int32_binary_div(int32_t x, int32_t y) { return x / y; }
inline int32_t int32_binary_min(int32_t x, int32_t y) { return std::min(x, y); }
inline int32_t int32_binary_max(int32_t x, int32_t y) { return std::max(x, y); }
#if defined(__APPLE__)
inline int32_t int32_binary_pow(int32_t x, int32_t y) {
    return (int32_t)pow(x, y);
}
#else
inline int32_t int32_binary_pow(int32_t x, int32_t y) { return std::pow(x, y); }
#endif
inline int32_t int32_binary_logical_and(int32_t x, int32_t y) { return x && y; }
inline int32_t int32_binary_mod(int32_t x, int32_t y) { return x % y; }

inline int64_t int64_binary_add(int64_t x, int64_t y) { return x + y; }
inline int64_t int64_binary_sub(int64_t x, int64_t y) { return x - y; }
inline int64_t int64_binary_mul(int64_t x, int64_t y) { return x * y; }
inline int64_t int64_binary_div(int64_t x, int64_t y) { return x / y; }
inline int64_t int64_binary_min(int64_t x, int64_t y) { return std::min(x, y); }
inline int64_t int64_binary_max(int64_t x, int64_t y) { return std::max(x, y); }
#if defined(__APPLE__)
inline int64_t int64_binary_pow(int64_t x, int64_t y) {
    return (int64_t)pow(x, y);
}
#else
inline int64_t int64_binary_pow(int64_t x, int64_t y) { return std::pow(x, y); }
#endif
inline int64_t int64_binary_logical_and(int64_t x, int64_t y) { return x && y; }
inline int64_t int64_binary_mod(int64_t x, int64_t y) { return x % y; }

inline bool bool_binary_logical_and(bool x, bool y) { return x && y; }
inline bool bool_binary_logical_or(bool x, bool y) { return x || y; }
inline bool bool_binary_logical_xor(bool x, bool y) { return x ^ y; }

[[maybe_unused]] static nncase_mt_t nncase_mt{fabsf,
                                              acosf,
                                              acoshf,
                                              asinf,
                                              asinhf,
                                              ceilf,
                                              cosf,
                                              coshf,
                                              expf,
                                              floorf,
                                              logf,
                                              float_unary_logical_not,
                                              float_unary_neg,
                                              roundf,
                                              float_unary_rsqrt,
                                              float_unary_sign,
                                              sinf,
                                              sinhf,
                                              sqrtf,
                                              float_unary_square,
                                              tanhf,
                                              float_binary_add,
                                              float_binary_sub,
                                              float_binary_mul,
                                              float_binary_div,
                                              float_binary_min,
                                              float_binary_max,
                                              float_binary_pow,
                                              float_binary_logical_and,
                                              float_binary_mod,
                                              int32_binary_add,
                                              int32_binary_sub,
                                              int32_binary_mul,
                                              int32_binary_div,
                                              int32_binary_min,
                                              int32_binary_max,
                                              int32_binary_pow,
                                              int32_binary_logical_and,
                                              int32_binary_mod,
                                              int64_binary_add,
                                              int64_binary_sub,
                                              int64_binary_mul,
                                              int64_binary_div,
                                              int64_binary_min,
                                              int64_binary_max,
                                              int64_binary_pow,
                                              int64_binary_logical_and,
                                              int64_binary_mod,
                                              bool_binary_logical_and,
                                              bool_binary_logical_or,
                                              bool_binary_logical_xor};

END_NS_NNCASE_RT_MODULE