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

#if __riscv_vector

result<void> reduce_max_impl(const float *in, float *out, size_t outter_size,
                             size_t inner_size, size_t reduce_size) {
    for (size_t i = 0; i < outter_size; ++i) {
        size_t outer_offset = i * reduce_size * inner_size;
        for (size_t j = 0; j < inner_size; ++j) {
            size_t base_index = outer_offset + j;
            float max_val = in[base_index];
            size_t remaining = reduce_size;

            // set vlen and convert scaler to vector
            if (0) // m1
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
            } else // m4
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
            out[i * inner_size + j] = max_val;
        }
    }
    return ok();
}

result<void> reduce_min_impl(const float *in, float *out, size_t outter_size,
                             size_t inner_size, size_t reduce_size) {
    for (size_t i = 0; i < outter_size; ++i) {
        size_t outer_offset = i * reduce_size * inner_size;
        for (size_t j = 0; j < inner_size; ++j) {
            size_t base_index = outer_offset + j;
            float min_val = in[base_index];
            size_t remaining = reduce_size;

            // set vlen and convert scaler to vector
            size_t vl = vsetvl_e32m4(remaining);
            vfloat32m4_t v_min = vfmv_v_f_f32m4(min_val, vl);

            // process full registers data.
            while (remaining / vl > 0) {
                vfloat32m4_t v_in = vle32_v_f32m4(&in[base_index], vl);
                v_min = vfmin_vv_f32m4(v_min, v_in, vl);

                remaining -= vl;
                base_index += vl;
            }
            vfloat32m1_t reduced_min_ = vfredmin_vs_f32m4_f32m1(
                vundefined_f32m1(), v_min, vfmv_v_f_f32m1(min_val, vl), vl);
            min_val = vfmv_f_s_f32m1_f32(reduced_min_);

            // process the remaining elements
            if (remaining > 0) {
                vl = vsetvl_e32m4(remaining);
                v_min = vfmv_v_f_f32m4(min_val, vl);
                vfloat32m4_t v_in = vle32_v_f32m4(&in[base_index], vl);
                v_min = vfmin_vv_f32m4(v_min, v_in, vl);
                reduced_min_ = vfredmin_vs_f32m4_f32m1(
                    vundefined_f32m1(), v_min, vfmv_v_f_f32m1(min_val, vl), vl);
                min_val = vfmv_f_s_f32m1_f32(reduced_min_);
            }

            out[i * inner_size + j] = min_val;
        }
    }
    return ok();
}

result<void> reduce_sum_impl(const float *in, float *out, size_t outter_size,
                             size_t inner_size, size_t reduce_size) {
    for (size_t i = 0; i < outter_size; ++i) {
        size_t outer_offset = i * reduce_size * inner_size;
        for (size_t j = 0; j < inner_size; ++j) {
            size_t base_index = outer_offset + j;
            float sum = 0.0f;
            size_t remaining = reduce_size;

            // set vlen and convert scaler to vector
            size_t vl = vsetvl_e32m4(remaining);
            vfloat32m4_t v_sum = vfmv_v_f_f32m4(0.0f, vl);

            // process full registers data.
            while (remaining / vl > 0) {
                vfloat32m4_t v_in = vle32_v_f32m4(&in[base_index], vl);
                v_sum = vfadd_vv_f32m4(v_sum, v_in, vl);

                remaining -= vl;
                base_index += vl;
            }
            vfloat32m1_t reduced_sum_ = vfredosum_vs_f32m4_f32m1(
                vundefined_f32m1(), v_sum, vfmv_v_f_f32m1(0.0f, vl), vl);
            sum += vfmv_f_s_f32m1_f32(reduced_sum_);

            // process the remaining elements
            if (remaining > 0) {
                vl = vsetvl_e32m4(remaining);
                v_sum = vfmv_v_f_f32m4(0.0f, vl);
                vfloat32m4_t v_in = vle32_v_f32m4(&in[base_index], vl);
                v_sum = vfadd_vv_f32m4(v_sum, v_in, vl);
                reduced_sum_ = vfredosum_vs_f32m4_f32m1(
                    vundefined_f32m1(), v_sum, vfmv_v_f_f32m1(0.0f, vl), vl);
                sum += vfmv_f_s_f32m1_f32(reduced_sum_);
            }

            out[i * inner_size + j] = sum;
        }
    }
    return ok();
}

result<void> reduce_prod_impl(const float *in, float *out, size_t outter_size,
                              size_t inner_size, size_t reduce_size) {
    for (size_t i = 0; i < outter_size; ++i) {
        size_t outer_offset = i * reduce_size * inner_size;
        for (size_t j = 0; j < inner_size; ++j) {
            size_t base_index = outer_offset + j;
            float acc = 1.0f;
            size_t remaining = reduce_size;

            // set vlen and convert scaler to vector
            size_t vl = vsetvl_e32m4(remaining);
            vfloat32m4_t v_acc = vfmv_v_f_f32m4(1.0f, vl);

            // process full registers data.
            while (remaining / vl > 0) {
                vfloat32m4_t v_in = vle32_v_f32m4(&in[base_index], vl);
                v_acc = vfmul_vv_f32m4(v_acc, v_in, vl);

                remaining -= vl;
                base_index += vl;
            }
            for (size_t i = 0; i < vl; i++) {
                acc *= vfmv_f_s_f32m4_f32(
                    vslidedown_vx_f32m4(vundefined_f32m4(), v_acc, i, vl));
            }

            // process the remaining elements
            if (remaining > 0) {
                vl = vsetvl_e32m4(remaining);
                v_acc = vfmv_v_f_f32m4(1.0f, vl);
                vfloat32m4_t v_in = vle32_v_f32m4(&in[base_index], vl);
                v_acc = vfmul_vv_f32m4(v_acc, v_in, vl);
                for (size_t i = 0; i < vl; i++) {
                    acc *= vfmv_f_s_f32m4_f32(
                        vslidedown_vx_f32m4(vundefined_f32m4(), v_acc, i, vl));
                }
            }

            out[i * inner_size + j] = acc;
        }
    }
    return ok();
}

#endif

result<void> optimized::reduce(
    NNCASE_UNUSED typecode_t typecode, nncase::runtime::stackvm::reduce_op_t op,
    NNCASE_UNUSED const gsl::byte *init_value, const gsl::byte *input,
    gsl::byte *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> axis,
    NNCASE_UNUSED gsl::span<const size_t> in_strides,
    NNCASE_UNUSED gsl::span<const size_t> out_strides,
    NNCASE_UNUSED bool keep_dims,
    NNCASE_UNUSED kernel_context &context) noexcept {
#if __riscv_vector
    // The type of axis is 'size_t'. It is real axis.
    // 计算inner_size、outter_size
    size_t inner_size = 1, outter_size = 1;
    size_t reduce_size = in_shape[axis[0]];

    for (size_t i = 0; i < axis[0]; i++) {
        outter_size *= in_shape[i];
    }

    for (size_t i = axis[0] + 1; i < in_shape.size(); i++) {
        inner_size *= in_shape[i];
    }

    const float *in = reinterpret_cast<const float *>(input);
    float *out = reinterpret_cast<float *>(output);
    if (axis.size() == 1 && axis[0] == in_shape.size() - 1) {
        switch (op) {
        case reduce_op_t::max:
            return reduce_max_impl(in, out, outter_size, inner_size,
                                   reduce_size);
        case reduce_op_t::min:
            return reduce_min_impl(in, out, outter_size, inner_size,
                                   reduce_size);
        case reduce_op_t::sum:
            return reduce_sum_impl(in, out, outter_size, inner_size,
                                   reduce_size);
        case reduce_op_t::mean:
            reduce_sum_impl(in, out, outter_size, inner_size, reduce_size)
                .unwrap();
            for (size_t i = 0; i < outter_size; i++) {
                out[i] *= 1.0f / reduce_size;
            }
            return ok();
        case reduce_op_t::prod:
            return reduce_prod_impl(in, out, outter_size, inner_size,
                                    reduce_size);
        default:
            break;
        }
    }
    // TODO: implement non-last axis reduce
    // TODO: implement multi-axis reduce

#endif
    return stackvm::reference::reduce(typecode, op, init_value, input, output,
                                      in_shape, axis, in_strides, out_strides,
                                      keep_dims, context);
}
