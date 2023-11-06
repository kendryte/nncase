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

#include <iostream>
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

#if __riscv_vector

void ternary_vec(const float *input_a, int input_a_len, const float *input_b, int input_b_len, const float *input_c,
    [[maybe_unused]] int input_c_len, float *out, int out_len)
{
    __asm volatile(

        "div a4, %[dst_len], %[mask_len];"
        "mv a2, %[c];"
        "mv a3, %[dst];"

        "beq %[mask_len], %[b_len], B_IS_VECTOR%=;"

        "flw ft0, (%[b]);"

        "TERNARY_RVV%=:;"

        "mv a0, %[mask_len];"
        "mv a1, %[mask];"

        "XXXXXX%=:"
        "vsetvli t0, a0, e32, m8;"
        "vle32.v v8, (a1);"
        "vle32.v v16,(a2);"
        "vmsne.vx v0, v8, x0;"
        "vfmerge.vfm v8, v16, ft0, v0;"
        "vse32.v v8, (a3);"

        "slli t1, t0, 2;"
        "sub a0, a0, t0; "
        "add a1, a1, t1;"
        "add a2, a2, t1;"
        "add a3, a3, t1;"
        "bnez a0, XXXXXX%=;"

        "addi a4, a4, -1;"
        "bnez a4, TERNARY_RVV%=;"
        "j END%=;"

        //////////////////////////////////////
        "B_IS_VECTOR%=:;"
        "TERNARY_RVV2%=:;"

        "mv a0, %[mask_len];"
        "mv a1, %[mask];"
        "mv a5, %[b];"

        "XXXXXX2%=:"
        "vsetvli t0, a0, e32, m8;"
        "vle32.v v8, (a1);"
        "vle32.v v16,(a2);"
        "vle32.v v24, (a5);"
        "vmsne.vx v0, v8, x0;"
        "vmerge.vvm v8, v16, v24, v0;"
        "vse32.v v8, (a3);"

        "slli t1, t0, 2;"
        "sub a0, a0, t0; "
        "add a1, a1, t1;"
        "add a2, a2, t1;"
        "add a3, a3, t1;"
        "add a5, a5, t1;"
        "bnez a0, XXXXXX2%=;"

        "addi a4, a4, -1;"
        "bnez a4, TERNARY_RVV2%=;"

        "END%=:;"

        :
        : [mask] "r"(input_a), [mask_len] "r"(input_a_len), [b] "r"(input_b), [b_len] "r"(input_b_len), [c] "r"(input_c), [dst] "r"(out), [dst_len] "r"(out_len)
        : "t0", "t1", "a0", "a1", "a2", "a3", "a4", "a5", "ft0", "v0", "v8", "v16", "v24");
}

result<void> tenary_impl(const float *input_a, const float *input_b, const float *input_c, float *output,
    const runtime_shape_t &in_a_shape, [[maybe_unused]] const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    [[maybe_unused]] const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape, [[maybe_unused]] const runtime_shape_t &in_c_strides,
    [[maybe_unused]] const runtime_shape_t &out_strides)
{

    int len_a = 1;
    for (int i = 0; i < (int)in_a_shape.size(); ++i)
    {
        len_a *= in_a_shape[i];
    }
    int len_b = 1;
    for (int i = 0; i < (int)in_b_shape.size(); ++i)
    {
        len_b *= in_b_shape[i];
    }
    int len_c = 1;
    for (int i = 0; i < (int)in_c_shape.size(); ++i)
    {
        len_c *= in_c_shape[i];
    }
    const auto out_shape = kernels::detail::get_binary_output_shape(kernels::detail::get_binary_output_shape(in_a_shape, in_b_shape), in_c_shape);
    int len_out = 1;
    for (int i = 0; i < (int)out_shape.size(); ++i)
    {
        len_out *= out_shape[i];
    }
    ternary_vec(input_a, len_a, input_b, len_b, input_c, len_c, output, len_out);
    return ok();
}

#endif

template <>
result<void> optimized::ternary<float>(const float *input_a, const float *input_b, const float *input_c, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape, const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept
{
#if __riscv_vector
    return tenary_impl(input_a, input_b, input_c, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, in_c_shape, in_c_strides, out_strides);
#else
    return cpu::reference::ternary(input_a, input_b, input_c, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, in_c_shape, in_c_strides, out_strides);
#endif
}

template result<void> optimized::ternary<int64_t>(const float *input_a, const int64_t *input_b, const int64_t *input_c, int64_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape, const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept;

template result<void> optimized::ternary<int32_t>(const float *input_a, const int32_t *input_b, const int32_t *input_c, int32_t *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape, const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> optimized::ternary(const float *input_a, const T *input_b, const T *input_c, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &in_c_shape, const runtime_shape_t &in_c_strides,
    const runtime_shape_t &out_strides) noexcept
{
    return cpu::reference::ternary(input_a, input_b, input_c, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, in_c_shape, in_c_strides, out_strides);
}
