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

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

#if __riscv_vector
#include "utils.h"
#include <math.h>
#include <riscv_vector.h>

static float get_max_value(int n, const float *x) {
    float max_value;
    __asm volatile("mv a0, %[avl];"
                   "mv a1, %[input_ptr];"
                   "vsetvli t0, a0, e32, m8;"
                   "vfmv.s.f v16, %3;"
                   "loop_rvv_max_index%=:;"
                   "vsetvli t0, a0, e32, m8;"
                   "vle32.v v8, (a1);"
                   "slli t1, t0, 2;"
                   "sub a0, a0, t0;"
                   "add a1, a1, t1;"
                   "vfredmax.vs v16,v8,v16;"
                   /////////////////
                   "bnez a0, loop_rvv_max_index%=;"
                   "vfmv.f.s %[value_index], v16;"
                   : [value_index] "=f"(max_value)
                   : [avl] "r"(n), [input_ptr] "r"(x), "f"(-FLT_MAX)
                   : "t0", "t1", "t2", "t3", "t5", "a0", "a1", "fa0");
    return max_value;
}

void log_softmax_step1(int n, const float *x, float *y) {

    float max_value = get_max_value(n, x);
    size_t vl;
    const float *src_bak = x;
    int nn = n;
    float sum = 0.0f;
    for (; n > 0; n -= vl) {
        vl = vsetvl_e32m8(n);
        vfloat32m8_t _p = vle32_v_f32m8(src_bak, vl);
        vfloat32m1_t _sum = vfmv_s_f_f32m1(vundefined_f32m1(), sum, vl);

        _p = vfsub_vf_f32m8(_p, max_value, vl);
        _p = exp_ps(_p, vl);

        _sum = vfredusum_vs_f32m8_f32m1(_sum, _p, /* scalar*/ _sum, vl);

        sum = vfmv_f_s_f32m1_f32(_sum);
        src_bak += vl;
    }
    sum = logf(sum) + max_value;
    for (; nn > 0; nn -= vl) {
        vl = vsetvl_e32m8(nn);
        vfloat32m8_t _p = vle32_v_f32m8(x, vl);
        _p = vfsub_vf_f32m8(_p, sum, vl);
        vse32_v_f32m8(y, _p, vl);
        y += vl;
        x += vl;
    }
}

void log_softmax_block(int len, int real_blocksize, const float *x, float *dx,
                       int data_stride) {
    size_t vl;
    vl = vsetvl_e32m8(real_blocksize);
    assert((int)vl == real_blocksize);
    vfloat32m8_t max_sf = vfmv_v_f_f32m8(-FLT_MAX, vl);
    vfloat32m8_t sum_sf = vfmv_v_f_f32m8(0.0f, vl);
    for (int32_t i = 0; i < len; i++) {
        vfloat32m8_t v_in = vle32_v_f32m8(x + i * data_stride, vl);
        max_sf = vfmax_vv_f32m8(max_sf, v_in, vl);
    }
    for (int32_t i = 0; i < len; i++) {
        vfloat32m8_t v_in = vle32_v_f32m8(x + i * data_stride, vl);
        vfloat32m8_t vsub_data = vfsub_vv_f32m8(v_in, max_sf, vl);
        vsub_data = exp_ps(vsub_data, vl);
        sum_sf = vfadd_vv_f32m8(sum_sf, vsub_data, vl);
    }

    sum_sf = log_ps(sum_sf, vl);

    for (int32_t i = 0; i < len; i++) {
        vfloat32m8_t v_in = vle32_v_f32m8(x + i * data_stride, vl);
        vfloat32m8_t vsub_data = vfsub_vv_f32m8(v_in, sum_sf, vl);
        vsub_data = vfsub_vv_f32m8(vsub_data, max_sf, vl);
        vse32_v_f32m8(dx + i * data_stride, vsub_data, vl);
    }
}

#define BLOCK_LOGSOFTMAX 32
void log_softmax_step_not1(int32_t len, const float *x, float *dx, int step) {
    for (int j = 0; j < step / BLOCK_LOGSOFTMAX; ++j) {
        log_softmax_block(len, BLOCK_LOGSOFTMAX, x, dx, step);
        x += BLOCK_LOGSOFTMAX;
        dx += BLOCK_LOGSOFTMAX;
    }
    int left_number = step & (BLOCK_LOGSOFTMAX - 1);
    if (left_number) {
        log_softmax_block(len, left_number, x, dx, step);
    }
}

static void log_softmax_impl(const float *input, float *output,
                             std::span<const size_t> in_shape, int axis) {
    size_t ndim = in_shape.size();
    size_t positive_axis = axis < 0 ? ndim + axis : axis;
    size_t axis_dim = in_shape[positive_axis];

    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++) {
        out_side *= in_shape[i];
    }

    size_t in_side = 1;
    for (size_t i = positive_axis + 1; i < ndim; i++) {
        in_side *= in_shape[i];
    }

    if (positive_axis == (ndim - 1)) {
        const float *ptr_input = input;
        float *ptr_output = output;
        for (size_t i = 0; i < out_side; i++) {
            int n = axis_dim;
            const float *ptr_input_vl = ptr_input;
            float *ptr_output_vl = ptr_output;
            log_softmax_step1(n, ptr_input_vl, ptr_output_vl);
            ptr_input += axis_dim;
            ptr_output += axis_dim;
        }
    } else {
        const float *ptr_input = input;
        float *ptr_output = output;
        for (size_t i = 0; i < out_side; i++) {
            int n = axis_dim;
            const float *ptr_input_vl = ptr_input;
            float *ptr_output_vl = ptr_output;
            log_softmax_step_not1(n, ptr_input_vl, ptr_output_vl, in_side);
            ptr_input += axis_dim * in_side;
            ptr_output += axis_dim * in_side;
        }
    }
}
#endif

#define IN_CAST(_ty, _name) reinterpret_cast<const _ty *>(_name)
#define OUT_CAST(_ty, _name) reinterpret_cast<_ty *>(_name)

// template result<void> optimized::log_softmax<float>(
//     const float *input, float *output, std::span<const size_t> in_shape,
//     [[maybe_unused]] std::span<const size_t> in_strides,
//     [[maybe_unused]] std::span<const size_t> out_strides,
//     int32_t axis) noexcept;

// template <typename T>
result<void>
optimized::log_softmax([[maybe_unused]] typecode_t typecode,
                       const std::byte *input, std::byte *output,
                       std::span<const size_t> in_shape,
                       [[maybe_unused]] std::span<const size_t> in_strides,
                       [[maybe_unused]] std::span<const size_t> out_strides,
                       int32_t axis) noexcept {
    result<void> ret_value = ok();
#if __riscv_vector
    log_softmax_impl(IN_CAST(float, input), OUT_CAST(float, output), in_shape,
                     axis);
#else
    ret_value = reference::softmax(typecode, input, output, in_shape,
                                   in_strides, out_strides, axis, 1.f, true);
#endif
    return ret_value;
}
