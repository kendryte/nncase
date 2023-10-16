#pragma once
#include "hardware_context.h"
#include "method_table_def.h"
#include "runtime_utils.h"
#include "thread_pool.h"
#include <functional>
#include <math.h>
#include <nncase/runtime/cpu/compiler_defs.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

BEGIN_NS_NNCASE_RT_MODULE(cpu)

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

#ifdef __riscv_vector
inline void matmul_unit_impl(const float *input_a, const float *input_b,
                             float *output, size_t size_m, size_t size_k,
                             size_t size_n, size_t lda, size_t ldb,
                             size_t ldc) {
    size_t vl;
    for (size_t m = 0; m < size_m; ++m) {
        const float *b_n_ptr = input_b;
        float *c_n_ptr = output;
        for (size_t c_n_count = size_n; c_n_count; c_n_count -= vl) {
            vl = vsetvl_e32m1(c_n_count);
            const float *a_k_ptr = input_a;
            const float *b_k_ptr = b_n_ptr;
            vfloat32m1_t acc = vle32_v_f32m1(c_n_ptr, vl);
            for (size_t k = 0; k < size_k; ++k) {
                vfloat32m1_t b_n_data = vle32_v_f32m1(b_k_ptr, vl);
                acc = vfmacc_vf_f32m1(acc, *a_k_ptr, b_n_data, vl);
                b_k_ptr += ldb;
                a_k_ptr++;
            }
            vse32_v_f32m1(c_n_ptr, acc, vl);
            c_n_ptr += vl;
            b_n_ptr += vl;
        }
        input_a += lda;
        output += ldc;
    }
}
#else
inline void
matmul_unit_impl([[maybe_unused]] const float *input_a,
                 [[maybe_unused]] const float *input_b,
                 [[maybe_unused]] float *output, [[maybe_unused]] size_t size_m,
                 [[maybe_unused]] size_t size_k, [[maybe_unused]] size_t size_n,
                 [[maybe_unused]] size_t lda, [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] size_t ldc) {
    throw std::runtime_error("not supported");
}
#endif

[[maybe_unused]] static nncase_mt_t nncase_mt = {
    fabsf,
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
    erff,
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
    bool_binary_logical_xor,
    thread_pool::thread_start,
    thread_pool::thread_end,
    matmul_unit_impl,
};

inline void *rt_malloc(size_t size) { return (void *)new uint8_t[size](); }

[[maybe_unused]] static runtime_util_mt runtime_util = {
    printf,      rt_malloc, free,   create_thread,
    join_thread, rt_assert, memcpy, memset};

[[maybe_unused]] static hardware_context_mt hw_ctx_mt = {
    hardware_context::lock_block,   hardware_context::mark_block_visit,
    hardware_context::unlock_block, hardware_context::wait_block_sync,
    hardware_context::lock_all,     hardware_context::mark_all_visit,
    hardware_context::unlock_all,   hardware_context::wait_all_sync,
    hardware_context::init};

END_NS_NNCASE_RT_MODULE