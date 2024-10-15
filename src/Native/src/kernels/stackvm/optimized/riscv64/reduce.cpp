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
#include <riscv_vector.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

// #if __riscv_vector

result<void> reduce_max_impl(const float *in, float *out, size_t outter_size, size_t inner_size, size_t reduce_size){
    for (size_t i = 0; i < outter_size; ++i) {
        size_t outer_offset = i * reduce_size * inner_size;
        for (size_t j = 0; j < inner_size; ++j) {
            size_t base_index = outer_offset + j;
            float max_val = in[base_index];  // 初始化最大值为第一个元素
            size_t remaining = reduce_size;

            // set vlen and convert scaler to vector
            if(0)
            {
                size_t vl = vsetvl_e32m1(remaining);
                vfloat32m1_t v_max = vfmv_v_f_f32m1(max_val, vl);

                // process full registers data.
                while (remaining / vl > 0) {
                    vfloat32m1_t v_in = vle32_v_f32m1(&in[base_index], vl);
                    v_max = vfmax_vv_f32m1(v_max, v_in, vl);

                    remaining -= vl;
                    base_index += vl;
                }
                vfloat32m1_t reduced_max_ =
                    vfredmax_vs_f32m1_f32m1(v_max, v_max, v_max, vl);
                max_val = vfmv_f_s_f32m1_f32(reduced_max_);

                // process the remaining elements
                if (remaining > 0) {
                    vl = vsetvl_e32m1(remaining);
                    v_max = vfmv_v_f_f32m1(max_val, vl);
                    vfloat32m1_t v_in = vle32_v_f32m1(&in[base_index], vl);
                    v_max = vfmax_vv_f32m1(v_max, v_in, vl);
                    reduced_max_ =
                        vfredmax_vs_f32m1_f32m1(v_max, v_max, v_max, vl);
                    max_val = vfmv_f_s_f32m1_f32(reduced_max_);
                }
            }
            else
            {
                // set vlen and convert scaler to vector
                size_t vl = vsetvl_e32m4(remaining);
                vfloat32m4_t v_max = vfmv_v_f_f32m4(max_val, vl);

                // process full registers data.
                while (remaining / vl > 0) {
                    vfloat32m4_t v_in = vle32_v_f32m4(&in[base_index], vl);
                    v_max = vfmax_vv_f32m4(v_max, v_in, vl);

                    remaining -= vl;
                    base_index += vl;
                }
                vfloat32m1_t reduced_max_ = vfredmax_vs_f32m4_f32m1(
                    vundefined_f32m1(), v_max, vfmv_v_f_f32m1(max_val, vl), vl);
                max_val = vfmv_f_s_f32m1_f32(reduced_max_);

                // process the remaining elements

                // 处理剩余的元素
                if (remaining > 0) {
                    vl = vsetvl_e32m4(remaining);
                    v_max = vfmv_v_f_f32m4(max_val, vl);
                    vfloat32m4_t v_in = vle32_v_f32m4(&in[base_index], vl);
                    v_max = vfmax_vv_f32m4(v_max, v_in, vl);
                    reduced_max_ = vfredmax_vs_f32m4_f32m1(
                        vundefined_f32m1(), v_max, vfmv_v_f_f32m1(max_val, vl),
                        vl);
                    max_val = vfmv_f_s_f32m1_f32(reduced_max_);
                }
            }
            out[i * inner_size + j] = max_val;  // 存储结果
        }
    }
    return ok();
}

result<void> reduce_min_impl(const float *in, float *out, size_t outter_size, size_t inner_size, size_t reduce_size){
    for (size_t i = 0; i < outter_size; ++i) {
        size_t outer_offset = i * reduce_size * inner_size;
        for (size_t j = 0; j < inner_size; ++j) {
            size_t base_index = outer_offset + j;
            float min_val = in[base_index];  // 初始化最大值为第一个元素
            size_t remaining = reduce_size;

            // set vlen and convert scaler to vector
            size_t vl = vsetvl_e32m1(remaining);
            vfloat32m1_t v_min = vfmv_v_f_f32m1(min_val, vl);

            // process full registers data.
            while (remaining / vl > 0) {
                vfloat32m1_t v_in = vle32_v_f32m1(&in[base_index], vl);
                v_min = vfmin_vv_f32m1(v_min, v_in, vl);

                remaining -= vl;
                base_index += vl;
            }
            vfloat32m1_t reduced_min_ = vfredmin_vs_f32m1_f32m1(v_min, v_min, v_min, vl);
            min_val = vfmv_f_s_f32m1_f32(reduced_min_);

            // process the remaining elements
            vl = vsetvl_e32m1(remaining);
            v_min = vfmv_v_f_f32m1(min_val, vl);
            vfloat32m1_t v_in = vle32_v_f32m1(&in[base_index], vl);
            v_min = vfmin_vv_f32m1(v_min, v_in, vl);
            reduced_min_ = vfredmin_vs_f32m1_f32m1(v_min, v_min, v_min, vl);
            min_val = vfmv_f_s_f32m1_f32(reduced_min_);

            out[i * inner_size + j] = min_val;  // 存储结果
        }
    }
    return ok();
}
// #endif

result<void> optimized::reduce(
    NNCASE_UNUSED typecode_t typecode, nncase::runtime::stackvm::reduce_op_t op,
    NNCASE_UNUSED  const gsl::byte *init_value, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> axis,
    NNCASE_UNUSED  gsl::span<const size_t> in_strides, NNCASE_UNUSED gsl::span<const size_t> out_strides,
    NNCASE_UNUSED  bool keep_dims, NNCASE_UNUSED  kernel_context &context) noexcept {
#if __riscv_vector
    // The type of axis is 'size_t'. It is real axis.
    // 计算inner_size、outter_size
    size_t inner_size = 1, outter_size = 1;
    size_t reduce_size = in_shape[axis[0]];

    for (size_t i = 0; i < axis[0]; i++) { outter_size *= in_shape[i]; }

    for (size_t i = axis[0]+1; i < in_shape.size(); i++) { inner_size *= in_shape[i]; }

    const float* in = reinterpret_cast<const float*>(input);
    float* out = reinterpret_cast<float*>(output);
    if (axis.size() == 1)
    {
        switch(op)
        {
        case reduce_op_t::max:
            return reduce_max_impl(in, out, outter_size, inner_size, reduce_size);
        case reduce_op_t::min:
            return reduce_min_impl(in, out, outter_size, inner_size, reduce_size);
            break;
        case reduce_op_t::sum:
        case reduce_op_t::mean:
        case reduce_op_t::prod:
        default:
            break;
        }
    }

#endif
    return stackvm::reference::reduce(typecode, op, init_value, input, output,
                                      in_shape, axis, in_strides, out_strides,
                                      keep_dims, context);
}
