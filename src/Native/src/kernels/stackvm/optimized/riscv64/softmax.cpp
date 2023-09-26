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
#if __riscv_vector
#include "utils.h"
#include <riscv_vector.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

namespace {
#if __riscv_vector

static __inline __attribute__((__always_inline__)) vfloat32m8_t
exp_ps2(vfloat32m8_t _p, size_t vl) {
    _p = vfmax_vf_f32m8(_p, -88.0f, vl);
    _p = vfmul_vf_f32m8(_p, 12102203.0f, vl);
    _p = vfadd_vf_f32m8(_p, 1065414017, vl);

    vint32m8_t p2 = vfcvt_x_f_v_i32m8(_p, vl);
    _p = vreinterpret_v_i32m8_f32m8(p2);
    return _p;
}

vfloat32m8_t exp_ps2_opt(vfloat32m8_t _p, const float c0, const float c1,
                         const float c2, size_t vl) {
    _p = vfmax_vf_f32m8(_p, c0, vl);
    _p = vfmadd_vf_f32m8(_p, c1, vfmv_v_f_f32m8(c2, vl), vl);

    vint32m8_t p2 = vfcvt_x_f_v_i32m8(_p, vl);
    _p = vreinterpret_v_i32m8_f32m8(p2);
    return _p;
}

template <typename T>
result<void> optimized_softmax_impl_opt(const T *input, T *output,
                                        gsl::span<const size_t> in_shape,
                                        int32_t axis, float beta) noexcept {
    size_t ndim = in_shape.size();
    size_t positive_axis = axis < 0 ? ndim + axis : axis;
    size_t axis_dim = in_shape[positive_axis];

    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++)
        out_side *= in_shape[i];

    size_t in_side = 1;
    for (size_t i = positive_axis + 1; i < ndim; i++)
        in_side *= in_shape[i];
    float c0 = -88.0f * beta;
    float c1 = 12102203.0f * beta;
    float c2 = 1065414017.0f * beta;

    // axis == -1
    if (positive_axis == (ndim - 1)) {
        const float *ptr_input = input;
        float *ptr_output = output;
        for (size_t i = 0; i < out_side; i++) {
            auto n = axis_dim;
            const float *ptr_input_vl = ptr_input;
            float *ptr_output_vl = ptr_output;

            // max
            float max = std::numeric_limits<float>::lowest();
            auto s = vfmv_v_f_f32m1(max, vsetvl_e32m8(n));
            while (n) {
                auto vl = vsetvl_e32m8(n);
                auto v = vle32_v_f32m8(ptr_input_vl, vl);
                s = vfredmax_vs_f32m8_f32m1(s, v, s, vl);
                ptr_input_vl += vl;
                n -= vl;
            }
            max = vfmv_f_s_f32m1_f32(s);

            // exp((x - max) * beta) and sum(exp)
            float sum = 0.f;
            ptr_input_vl = ptr_input;
            n = axis_dim;
            s = vfmv_v_f_f32m1(sum, vsetvl_e32m8(n));
            while (n) {
                auto vl = vsetvl_e32m8(n);
                auto v_in = vle32_v_f32m8(ptr_input_vl, vl);
                auto v_out =
                    exp_ps2_opt(vfsub_vf_f32m8(v_in, max, vl), c0, c1, c2, vl);
                s = vfredusum_vs_f32m8_f32m1(s, v_out, s, vl);

                vse32_v_f32m8(ptr_output_vl, v_out, vl);
                ptr_input_vl += vl;
                ptr_output_vl += vl;
                n -= vl;
            }
            sum = vfmv_f_s_f32m1_f32(s);

            // div
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            n = axis_dim;
            sum = 1.0f / sum;
            while (n) {
                auto vl = vsetvl_e32m8(n);
                auto v_out = vle32_v_f32m8(ptr_output_vl, vl);
                v_out = vfmul_vf_f32m8(v_out, sum, vl);

                vse32_v_f32m8(ptr_output_vl, v_out, vl);
                ptr_output_vl += vl;
                n -= vl;
            }

            ptr_input += axis_dim;
            ptr_output += axis_dim;
        }
    } else {
        dims_t axes{positive_axis};
        auto reduced_shape =
            kernels::detail::get_reduced_shape(in_shape, axes, true);
        auto reduced_size = compute_size(reduced_shape);
        std::vector<float> max(reduced_size,
                               std::numeric_limits<float>::lowest());
        std::vector<float> sum(reduced_size, 0.f);

        for (size_t i = 0; i < out_side; i++) {
            const float *ptr_input = input + i * axis_dim * in_side;
            const float *ptr_input_vl = ptr_input;

            float *ptr_output = output + i * axis_dim * in_side;
            float *ptr_output_vl = ptr_output;

            float *ptr_max = max.data() + i * in_side;
            float *ptr_max_vl = ptr_max;

            float *ptr_sum = sum.data() + i * in_side;
            float *ptr_sum_vl = ptr_sum;

            // max
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_max_vl = ptr_max;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e32m8(n);
                    auto v_in = vle32_v_f32m8(ptr_input_vl, vl);
                    auto v_max = vle32_v_f32m8(ptr_max_vl, vl);

                    v_max = vfmax_vv_f32m8(v_in, v_max, vl);
                    vse32_v_f32m8(ptr_max_vl, v_max, vl);

                    ptr_input_vl += vl;
                    ptr_max_vl += vl;
                    n -= vl;
                }
            }

            // exp((x - max) * beta) and sum(exp)
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_max_vl = ptr_max;
                ptr_sum_vl = ptr_sum;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e32m8(n);
                    auto v_in = vle32_v_f32m8(ptr_input_vl, vl);
                    auto v_max = vle32_v_f32m8(ptr_max_vl, vl);
                    auto v_sum = vle32_v_f32m8(ptr_sum_vl, vl);

                    auto v_out =
                        exp_ps(vfmul_vf_f32m8(vfsub_vv_f32m8(v_in, v_max, vl),
                                              beta, vl),
                               vl);
                    vse32_v_f32m8(ptr_output_vl, v_out, vl);

                    v_sum = vfadd_vv_f32m8(v_sum, v_out, vl);
                    vse32_v_f32m8(ptr_sum_vl, v_sum, vl);

                    ptr_input_vl += vl;
                    ptr_output_vl += vl;
                    ptr_max_vl += vl;
                    ptr_sum_vl += vl;
                    n -= vl;
                }
            }

            // div
            ptr_output_vl = ptr_output;
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_sum_vl = ptr_sum;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e32m8(n);
                    auto v_out = vle32_v_f32m8(ptr_output_vl, vl);
                    auto v_sum = vle32_v_f32m8(ptr_sum_vl, vl);

                    v_out = vfdiv_vv_f32m8(v_out, v_sum, vl);
                    vse32_v_f32m8(ptr_output_vl, v_out, vl);

                    ptr_output_vl += vl;
                    ptr_sum_vl += vl;
                    n -= vl;
                }
            }
        }
    }
    return ok();
}

template <typename T>
result<void> optimized_softmax_impl(const T *input, T *output,
                                    gsl::span<const size_t> in_shape,
                                    int32_t axis, float beta) noexcept {
    size_t ndim = in_shape.size();
    size_t positive_axis = axis < 0 ? ndim + axis : axis;
    size_t axis_dim = in_shape[positive_axis];

    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++)
        out_side *= in_shape[i];

    size_t in_side = 1;
    for (size_t i = positive_axis + 1; i < ndim; i++)
        in_side *= in_shape[i];

    // axis == -1
    if (positive_axis == (ndim - 1)) {
        const float *ptr_input = input;
        float *ptr_output = output;

        for (size_t i = 0; i < out_side; i++) {
            auto n = axis_dim;
            const float *ptr_input_vl = ptr_input;
            float *ptr_output_vl = ptr_output;

            // max
            float max = std::numeric_limits<float>::lowest();
            while (n) {
                auto vl = vsetvl_e32m8(n);
                auto v = vle32_v_f32m8(ptr_input_vl, vl);
                auto s = vfmv_s_f_f32m1(vundefined_f32m1(), max, vl);

                s = vfredmax_vs_f32m8_f32m1(s, v, s, vl);
                max = vfmv_f_s_f32m1_f32(s);
                ptr_input_vl += vl;
                n -= vl;
            }

            // exp((x - max) * beta) and sum(exp)
            float sum = 0.f;
            ptr_input_vl = ptr_input;
            n = axis_dim;
            while (n) {
                auto vl = vsetvl_e32m8(n);
                auto v_in = vle32_v_f32m8(ptr_input_vl, vl);
                auto s = vfmv_s_f_f32m1(vundefined_f32m1(), sum, vl);

                auto v_out = exp_ps(
                    vfmul_vf_f32m8(vfsub_vf_f32m8(v_in, max, vl), beta, vl),
                    vl);
                s = vfredosum_vs_f32m8_f32m1(s, v_out, s, vl);

                vse32_v_f32m8(ptr_output_vl, v_out, vl);
                sum = vfmv_f_s_f32m1_f32(s);
                ptr_input_vl += vl;
                ptr_output_vl += vl;
                n -= vl;
            }

            // div
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            n = axis_dim;
            sum = 1.0f / sum;
            while (n) {
                auto vl = vsetvl_e32m8(n);
                auto v_out = vle32_v_f32m8(ptr_output_vl, vl);
                v_out = vfmul_vf_f32m8(v_out, sum, vl);
                vse32_v_f32m8(ptr_output_vl, v_out, vl);
                ptr_output_vl += vl;
                n -= vl;
            }

            ptr_input += axis_dim;
            ptr_output += axis_dim;
        }
    } else {
        dims_t axes{positive_axis};
        auto reduced_shape =
            kernels::detail::get_reduced_shape(in_shape, axes, true);
        auto reduced_size = compute_size(reduced_shape);
        std::vector<float> max(reduced_size,
                               std::numeric_limits<float>::lowest());
        std::vector<float> sum(reduced_size, 0.f);

        for (size_t i = 0; i < out_side; i++) {
            const float *ptr_input = input + i * axis_dim * in_side;
            const float *ptr_input_vl = ptr_input;

            float *ptr_output = output + i * axis_dim * in_side;
            float *ptr_output_vl = ptr_output;

            float *ptr_max = max.data() + i * in_side;
            float *ptr_max_vl = ptr_max;

            float *ptr_sum = sum.data() + i * in_side;
            float *ptr_sum_vl = ptr_sum;

            // max
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_max_vl = ptr_max;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e32m8(n);
                    auto v_in = vle32_v_f32m8(ptr_input_vl, vl);
                    auto v_max = vle32_v_f32m8(ptr_max_vl, vl);

                    v_max = vfmax_vv_f32m8(v_in, v_max, vl);
                    vse32_v_f32m8(ptr_max_vl, v_max, vl);

                    ptr_input_vl += vl;
                    ptr_max_vl += vl;
                    n -= vl;
                }
            }

            // exp((x - max) * beta) and sum(exp)
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_max_vl = ptr_max;
                ptr_sum_vl = ptr_sum;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e32m8(n);
                    auto v_in = vle32_v_f32m8(ptr_input_vl, vl);
                    auto v_max = vle32_v_f32m8(ptr_max_vl, vl);
                    auto v_sum = vle32_v_f32m8(ptr_sum_vl, vl);

                    auto v_out =
                        exp_ps(vfmul_vf_f32m8(vfsub_vv_f32m8(v_in, v_max, vl),
                                              beta, vl),
                               vl);
                    vse32_v_f32m8(ptr_output_vl, v_out, vl);

                    v_sum = vfadd_vv_f32m8(v_sum, v_out, vl);
                    vse32_v_f32m8(ptr_sum_vl, v_sum, vl);

                    ptr_input_vl += vl;
                    ptr_output_vl += vl;
                    ptr_max_vl += vl;
                    ptr_sum_vl += vl;
                    n -= vl;
                }
            }

            // div
            ptr_output_vl = ptr_output;
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_sum_vl = ptr_sum;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e32m8(n);
                    auto v_out = vle32_v_f32m8(ptr_output_vl, vl);
                    auto v_sum = vle32_v_f32m8(ptr_sum_vl, vl);

                    v_out = vfdiv_vv_f32m8(v_out, v_sum, vl);
                    vse32_v_f32m8(ptr_output_vl, v_out, vl);

                    ptr_output_vl += vl;
                    ptr_sum_vl += vl;
                    n -= vl;
                }
            }
        }
    }
    return ok();
}
#endif
} // namespace

#define IN_CAST(_ty, _name) reinterpret_cast<const _ty *>(_name)
#define OUT_CAST(_ty, _name) reinterpret_cast<_ty *>(_name)

// template result<void> optimized::softmax<float>(
//     const float *input, float *output, gsl::span<const size_t> in_shape,
//     gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
//     int32_t axis, float beta) noexcept;

// template <typename T>
result<void> optimized::softmax([[maybe_unused]] typecode_t typecode,
                                const gsl::byte *input, gsl::byte *output,
                                gsl::span<const size_t> in_shape,
                                gsl::span<const size_t> in_strides,
                                gsl::span<const size_t> out_strides,
                                int32_t axis, float beta) noexcept {
#if __riscv_vector
    return optimized_softmax_impl(
        IN_CAST(float, input), OUT_CAST(float, output), in_shape, axis, beta);
//    TYPE_SELECT_SOFTMAX(typecode, SOFTMAX_IMPL);
#endif
    return stackvm::reference::softmax(typecode, input, output, in_shape,
                                       in_strides, out_strides, axis, beta);
}