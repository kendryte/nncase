#pragma once
#include <functional>
#include <math.h>
#include <nncase/runtime/cpu/compiler_defs.h>
#include <stddef.h>
#include <stdint.h>

BEGIN_NS_NNCASE_RT_MODULE(cpu)

typedef struct nncase_method_table {
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

[[maybe_unused]] static nncase_mt_t nncase_mt = {
    .float_unary_abs = fabsf,
    .float_unary_acos = acosf,
    .float_unary_acosh = acoshf,
    .float_unary_asin = asinf,
    .float_unary_asinh = asinhf,
    .float_unary_ceil = ceilf,
    .float_unary_cos = cosf,
    .float_unary_cosh = coshf,
    .float_unary_exp = expf,
    .float_unary_floor = floorf,
    .float_unary_log = logf,
    .float_unary_logical_not = &float_unary_logical_not,
    .float_unary_neg = &float_unary_neg,
    .float_unary_round = roundf,
    .float_unary_rsqrt = &float_unary_rsqrt,
    .float_unary_sign = &float_unary_sign,
    .float_unary_sin = sinf,
    .float_unary_sinh = sinhf,
    .float_unary_sqrt = sqrtf,
    .float_unary_square = &float_unary_square,
    .float_unary_tanh = tanhf};

END_NS_NNCASE_RT_MODULE