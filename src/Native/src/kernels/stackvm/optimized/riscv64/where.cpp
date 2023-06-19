
/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../../reference/ref_ops.h"
#include "../opt_ops.h"
#include <iostream>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;
using namespace nncase::runtime::stackvm;

namespace {
#if __riscv_vector

void ternary_vec(const uint8_t *input_a, const int input_a_len,
                 const float *input_b, const int input_b_len,
                 const float *input_c, const int input_c_len, float *out,
                 int out_len) {
    (void)input_c_len;
    __asm volatile(

        "div a4, %[dst_len], %[mask_len];"
        // "mv a2, %[c];"
        "mv a3, %[dst];"
        "addi a6, x0, 1;"

        "beq %[c_len], %[b_len], BC_IS_VECTOR%=;"
        "beq %[c_len], a6, C_IS_SCLAR%=;"
        // "beq %[mask_len], %[b_len], B_IS_VECTOR%=;"
        ///////////////////////////////////////////
        "flw ft0, (%[b]);"
        "mv a2, %[c];"
        "TERNARY_RVV_B%=:;"

        "mv a0, %[mask_len];"
        "mv a1, %[mask];"

        "XXXXXX_B%=:"
        "vsetvli t0, a0, e32, m8;"
        "vle8.v v8, (a1);"
        "vle32.v v16,(a2);"
        "vsetvli t0, a0, e8, m8;"
        "vmsne.vx v0, v8, x0;"
        "vsetvli t0, a0, e32, m8;"
        "vfmerge.vfm v8, v16, ft0, v0;"
        "vse32.v v8, (a3);"

        "slli t1, t0, 2;"
        "sub a0, a0, t0; "
        "add a1, a1, t0;"
        "add a2, a2, t1;"
        "add a3, a3, t1;"
        "bnez a0, XXXXXX_B%=;"

        "addi a4, a4, -1;"
        "bnez a4, TERNARY_RVV_B%=;"
        "j END%=;"

        //////////// len c == 1

        "C_IS_SCLAR%=:"
        "mv a5, %[b];"
        "flw ft0, (%[c]);"
        "TERNARY_RVV_C%=:;"

        "mv a0, %[mask_len];"
        "mv a1, %[mask];"

        "XXXXXX_C%=:"
        "vsetvli t0, a0, e32, m8;"
        "vle8.v v8, (a1);"
        "vle32.v v16,(a5);"
        "vsetvli t0, a0, e8, m8;"
        "vmseq.vx v0, v8, x0;"
        "vsetvli t0, a0, e32, m8;"
        "vfmerge.vfm v8, v16, ft0, v0;"
        "vse32.v v8, (a3);"

        "slli t1, t0, 2;"
        "sub a0, a0, t0; "
        "add a1, a1, t0;"
        "add a5, a5, t1;"
        "add a3, a3, t1;"
        "bnez a0, XXXXXX_C%=;"

        "addi a4, a4, -1;"
        "bnez a4, TERNARY_RVV_C%=;"
        "j END%=;"

        //////////////////////////////////////
        "BC_IS_VECTOR%=:;"

        "mv a5, %[b];"
        "mv a2, %[c];"

        "TERNARY_RVV2%=:;"
        "mv a0, %[mask_len];"
        "mv a1, %[mask];"

        "XXXXXX2%=:"
        "vsetvli t0, a0, e32, m8;"
        "vle8.v v8, (a1);"
        "vle32.v v16,(a2);"
        "vle32.v v24, (a5);"
        "vsetvli t0, a0, e8, m8;"
        "vmsne.vx v0, v8, x0;"
        "vsetvli t0, a0, e32, m8;"
        "vmerge.vvm v8, v16, v24, v0;"
        "vse32.v v8, (a3);"

        "slli t1, t0, 2;"
        "sub a0, a0, t0; "
        "add a1, a1, t0;"
        "add a2, a2, t1;"
        "add a3, a3, t1;"
        "add a5, a5, t1;"
        "bnez a0, XXXXXX2%=;"

        "addi a4, a4, -1;"
        "bnez a4, TERNARY_RVV2%=;"

        "END%=:;"
        :
        : [mask] "r"(input_a), [mask_len] "r"(input_a_len), [b] "r"(input_b),
          [b_len] "r"(input_b_len), [c] "r"(input_c), [c_len] "r"(input_c_len),
          [dst] "r"(out), [dst_len] "r"(out_len)
        : "t0", "t1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "ft0", "v0",
          "v8", "v16", "v24");
}

int tenary_impl(const uint8_t *input_cond, const float *input_b,
                const float *input_c, float *output,
                gsl::span<size_t> in_cond_shape, gsl::span<size_t> in_b_shape,
                gsl::span<size_t> in_c_shape, gsl::span<size_t> out_shape) {
    int len_cond = (int)compute_size(in_cond_shape);
    int len_b = (int)compute_size(in_b_shape);
    int len_c = (int)compute_size(in_c_shape);
    int len_out = (int)compute_size(out_shape);

    if (len_cond == len_b && len_cond == len_c) {
        ternary_vec(input_cond, len_cond, input_b, len_b, input_c, len_c,
                    output, len_out);
    } else if (in_b_shape.empty() || in_c_shape.empty()) {
        ternary_vec(input_cond, len_cond, input_b, len_b, input_c, len_c,
                    output, len_out);
    } else {
        return -1;
    }
    return 0;
}

#endif
} // namespace

result<void> nncase::kernels::stackvm::optimized::where(
    datatype_t dt, const bool *cond, const gsl::byte *x, const gsl::byte *y,
    gsl::byte *output, gsl::span<size_t> cond_shape, gsl::span<size_t> x_shape,
    gsl::span<size_t> y_shape, gsl::span<size_t> out_shape,
    const strides_t &cond_strides, const strides_t &x_strides,
    const strides_t &y_strides, const strides_t &out_strides) {

#if __riscv_vector
    // 这里做一步转换，明确下 cond 数据类型， c++ 中的 sizeof(bool) == 1，对于
    // sizeof(bool) != 1 的情况结果未定义。
    assert(sizeof(bool) == 1);
    const uint8_t *cond_pointer = (const uint8_t *)cond;
#define WHERE_IMPL(_ty)                                                        \
    auto *input_x = IN_CAST(_ty, x);                                           \
    auto *input_y = IN_CAST(_ty, y);                                           \
    auto *out = OUT_CAST(_ty, output);                                         \
    tenary_impl(cond_pointer, input_x, input_y, out, cond_shape, x_shape,      \
                y_shape, out_shape);

    try_var(typecode, to_typecode(dt));
    switch (typecode) {
    case dt_float32: {
        WHERE_IMPL(float);
        return ok();
    }
    default:;
    }
#endif

    return reference::where(dt, cond, x, y, output, cond_shape, x_shape,
                            y_shape, out_shape, cond_strides, x_strides,
                            y_strides, out_strides);
}
