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
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

#if __riscv_vector
static void reduce_block(int dim, int block, const float *input, float *out,
                         int gap) {
    __asm volatile(
        "vsetvli t0, %[block], e32, m8;"
        "mv a1, %[input];"
        "mv a3, %[gap];"
        "mv a0, %[dim];"
        "fcvt.s.w ft0, a0;"
        "slli a3, a3, 2;"
        "vmv.v.x v8, x0;"
        "reduce_block%=:;"
        "vle32.v v16, (a1);"
        "vfadd.vv v8, v16, v8;"
        "add a1,a1,a3;"
        "addi a0, a0, -1;"
        "bnez a0, reduce_block%=;"
        "vfdiv.vf v8, v8, ft0;"
        "vse32.v v8, ( %[out]);" ::[dim] "r"(dim),
        [block] "r"(block), [input] "r"(input), [out] "r"(out), [gap] "r"(gap)
        : "t0", "a0", "a1", "a3", "ft0", "v8", "v16");
}

static void reduce_mean(const float *input, float *out, int dim, int n) {
    __asm volatile(

        "mv a1, %[input];"
        "mv a2,  %[out]; "
        "mv a3, %[n];"
        "fcvt.s.w ft0, %[dim];"
        "reduce_mean_n_cycle%=:;"
        "mv a0, %[dim];"
        "vsetvli t0, a0, e32, m8;"
        "vmv.s.x v16, x0;"
        "reduce_mean2%=:;"
        "vsetvli t0, a0, e32, m8;"
        "vle32.v v8, (a1);"
        "slli t1,t0, 2;"
        "sub a0,a0,t0;"
        "add a1, a1, t1;"
        "vfredusum.vs v16,v8,v16;"
        "bnez a0, reduce_mean2%=;"
        "vfmv.f.s ft1, v16;"
        "fdiv.s ft1,ft1,ft0;"
        "fsw ft1, (a2);"
        "addi a2, a2, 4;"
        "addi a3,a3, -1;"
        "bnez a3, reduce_mean_n_cycle%=;" ::[dim] "r"(dim),
        [n] "r"(n), [input] "r"(input), [out] "r"(out)
        : "t0", "t1", "a0", "a1", "a2", "a3", "ft0", "ft1", "v8", "v16");
}

static void reduce_mean_s(int32_t c, int32_t dim, const float *input,
                          float *out, int32_t gap) {
#define BLOCK_N 32
    while (c--) {
        const float *tmp_input = input;
        for (int j = 0; j < gap / BLOCK_N; ++j) {
            reduce_block(dim, BLOCK_N, tmp_input, out, gap);
            tmp_input += BLOCK_N;
            out += BLOCK_N;
        }
        int left_number = gap & (BLOCK_N - 1);
        if (left_number) {
            reduce_block(dim, left_number, tmp_input, out, gap);
            out += left_number;
        }
        input += dim * gap;
    }
}

static int compute_size_by_index(gsl::span<const size_t> input, int start_index,
                                 int end_index) {
    int init_value = 1;
    for (int i = start_index; i < end_index; ++i) {
        init_value *= input[i];
    }
    return init_value;
}

static int get_parameter(gsl::span<const size_t> in_shape,
                         gsl::span<const size_t> axis, gsl::span<int> out) {
    int min_index = axis[0];
    int max_index = axis[0];
    for (int i = 1; i < (int)axis.size(); ++i) {
        int value = axis[i];
        if (value < min_index)
            min_index = value;
        else if (value > max_index)
            max_index = value;
    }
    int _sum1 = (max_index + min_index) * (max_index - min_index + 1) >> 1;
    int _sum2 = axis[0];
    for (int i = 1; i < (int)axis.size(); ++i) {
        _sum2 += axis[i];
    }
    if (_sum2 != _sum1) {
        return 1;
    }
    out[0] = compute_size_by_index(in_shape, min_index, max_index + 1);
    out[1] = compute_size_by_index(in_shape, 0, min_index);
    out[2] = compute_size_by_index(in_shape, max_index + 1, in_shape.size());
    return 0;
}
#endif

result<void> optimized::reduce(
    typecode_t typecode, nncase::runtime::stackvm::reduce_op_t op,
    const gsl::byte *init_value, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> axis,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    bool keep_dims, kernel_context &context) noexcept {
#if __riscv_vector
    do {
        if (op == reduce_op_t::mean && typecode == dt_float32) {
            int parameters[3];
            int ret = get_parameter(in_shape, axis, parameters);
            if (ret) {
                break;
            }
            auto input_data = IN_CAST(float, input);
            auto out_data = OUT_CAST(float, output);
            int gap = parameters[2];
            if (gap == 1) {
                reduce_mean(input_data, out_data, parameters[0], parameters[1]);
            } else {
                reduce_mean_s(parameters[1], parameters[0], input_data,
                              out_data, gap);
            }
            return ok();
        }
    } while (0);
#endif

    return stackvm::reference::reduce(typecode, op, init_value, input, output,
                                      in_shape, axis, in_strides, out_strides,
                                      keep_dims, context);
}
