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
            float max = *ptr_input_vl;
            {
                size_t vl = vsetvl_e32m4(n);
                vfloat32m4_t s = vfmv_v_f_f32m4(max, vl);
                while (n / vl > 0) {
                    vfloat32m4_t v = vle32_v_f32m4(ptr_input_vl, vl);
                    s = vfmax_vv_f32m4(s, v, vl);

                    n -= vl;
                    ptr_input_vl += vl;
                }

                vfloat32m1_t reduced_max_ = vfredmax_vs_f32m4_f32m1(
                    vundefined_f32m1(), s, vfmv_v_f_f32m1(max, vl), vl);
                max = vfmv_f_s_f32m1_f32(reduced_max_);

                if (n > 0) {
                    vl = vsetvl_e32m4(n);
                    s = vfmv_v_f_f32m4(max, vl);
                    vfloat32m4_t v = vle32_v_f32m4(ptr_input_vl, vl);
                    s = vfmax_vv_f32m4(s, v, vl);
                    reduced_max_ = vfredmax_vs_f32m4_f32m1(
                        vundefined_f32m1(), s, vfmv_v_f_f32m1(max, vl), vl);
                    max = vfmv_f_s_f32m1_f32(reduced_max_);
                }
            }

            // exp((x - max) * beta) and sum(exp)
            float sum = 0.f;
            ptr_input_vl = ptr_input;
            n = axis_dim;
            {
                auto vl = vsetvl_e32m4(n);
                auto s = vfmv_v_f_f32m4(0.0f, vl);
                while (n / vl > 0) {

                    auto v_in = vle32_v_f32m4(ptr_input_vl, vl);
                    auto v_out = exp_ps(
                        vfmul_vf_f32m4(vfsub_vf_f32m4(v_in, max, vl), beta, vl),
                        vl);
                    s = vfadd_vv_f32m4(s, v_out, vl);
                    vse32_v_f32m4(ptr_output_vl, v_out, vl);

                    ptr_input_vl += vl;
                    ptr_output_vl += vl;
                    n -= vl;
                }
                vfloat32m1_t reduce_sum_ = vfredosum_vs_f32m4_f32m1(
                    vundefined_f32m1(), s, vfmv_v_f_f32m1(0.0f, vl), vl);
                sum += vfmv_f_s_f32m1_f32(reduce_sum_);

                if (n > 0) {
                    vl = vsetvl_e32m4(n);
                    auto v_in = vle32_v_f32m4(ptr_input_vl, vl);
                    auto v_out = exp_ps(
                        vfmul_vf_f32m4(vfsub_vf_f32m4(v_in, max, vl), beta, vl),
                        vl);
                    reduce_sum_ =
                        vfredosum_vs_f32m4_f32m1(vundefined_f32m1(), v_out,
                                                 vfmv_v_f_f32m1(0.0f, vl), vl);

                    vse32_v_f32m4(ptr_output_vl, v_out, vl);
                    sum += vfmv_f_s_f32m1_f32(reduce_sum_);
                }
            }
            // div
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            n = axis_dim;
            sum = 1.0f / sum;
            {
                auto vl = vsetvl_e32m4(n);
                while (n / vl > 0) {
                    auto v_out = vle32_v_f32m4(ptr_output_vl, vl);
                    v_out = vfmul_vf_f32m4(v_out, sum, vl);
                    vse32_v_f32m4(ptr_output_vl, v_out, vl);
                    ptr_output_vl += vl;
                    n -= vl;
                }
                if (n > 0) {
                    vl = vsetvl_e32m4(n);
                    auto v_out = vle32_v_f32m4(ptr_output_vl, vl);
                    v_out = vfmul_vf_f32m4(v_out, sum, vl);
                    vse32_v_f32m4(ptr_output_vl, v_out, vl);
                }
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
result<void> optimized_softmax_half_impl(const T *input, T *output,
                                         gsl::span<const size_t> in_shape,
                                         int32_t axis,
                                         __float16_t beta) noexcept {
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
        const __float16_t *ptr_input = input;
        __float16_t *ptr_output = output;

        for (size_t i = 0; i < out_side; i++) {
            auto n = axis_dim;
            const __float16_t *ptr_input_vl = ptr_input;
            __float16_t *ptr_output_vl = ptr_output;

            // max
            __float16_t max = std::numeric_limits<__float16_t>::lowest();

            auto vl = vsetvl_e16m4(n);
            vfloat16m4_t v_max = vfmv_v_f_f16m4(max, vl);
            while (n / vl > 0) {
                vfloat16m4_t v_in = vle16_v_f16m4(ptr_input_vl, vl);
                v_max = vfmax_vv_f16m4(v_max, v_in, vl);

                n -= vl;
                ptr_input_vl += vl;
            }
            vfloat16m1_t reduced_max_ = vfredmax_vs_f16m4_f16m1(
                vundefined_f16m1(), v_max, vfmv_v_f_f16m1(max, vl), vl);
            max = vfmv_f_s_f16m1_f16(reduced_max_);

            if (n > 0) {
                vl = vsetvl_e16m4(n);
                v_max = vfmv_v_f_f16m4(max, vl);
                vfloat16m4_t v_in = vle16_v_f16m4(ptr_input_vl, vl);
                v_max = vfmax_vv_f16m4(v_max, v_in, vl);
                reduced_max_ = vfredmax_vs_f16m4_f16m1(
                    vundefined_f16m1(), v_max, vfmv_v_f_f16m1(max, vl), vl);
                max = vfmv_f_s_f16m1_f16(reduced_max_);
            }

            // exp((x - max) * beta) and sum(exp)
            __float16_t sum = (__float16_t)0.f;
            ptr_input_vl = ptr_input;
            n = axis_dim;
            vl = vsetvl_e16m4(n);
            vfloat16m4_t v_sum = vfmv_v_f_f16m4(sum, vl);
            while (n / vl > 0) {
                auto v_in = vle16_v_f16m4(ptr_input_vl, vl);
                auto v_out = exp_ph(
                    vfmul_vf_f16m4(vfsub_vf_f16m4(v_in, max, vl), beta, vl),
                    vl);
                v_sum = vfadd_vv_f16m4(v_sum, v_in, vl);
                vse16_v_f16m4(ptr_output_vl, v_out, vl);
                ptr_input_vl += vl;
                ptr_output_vl += vl;
                n -= vl;
            }

            vfloat16m1_t reduced_sum_ = vfredosum_vs_f16m4_f16m1(
                vundefined_f16m1(), v_sum, vfmv_v_f_f16m1(0, vl), vl);
            sum = vfmv_f_s_f16m1_f16(reduced_sum_);

            if (n > 0) {
                vl = vsetvl_e16m4(n);
                auto v_in = vle16_v_f16m4(ptr_input_vl, vl);
                auto s = vfmv_v_f_f16m1(0, vl);

                auto v_out = exp_ph(
                    vfmul_vf_f16m4(vfsub_vf_f16m4(v_in, max, vl), beta, vl),
                    vl);
                s = vfredosum_vs_f16m4_f16m1(s, v_out, s, vl);
                vse16_v_f16m4(ptr_output_vl, v_out, vl);
                sum += vfmv_f_s_f16m1_f16(s);
            }

            // div
            ptr_input_vl = ptr_input;
            ptr_output_vl = ptr_output;
            n = axis_dim;
            sum = (__float16_t)1.0f / sum;
            while (n) {
                auto vl = vsetvl_e16m4(n);

                auto v_out = vle16_v_f16m4(ptr_output_vl, vl);
                v_out = vfmul_vf_f16m4(v_out, sum, vl);
                vse16_v_f16m4(ptr_output_vl, v_out, vl);
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
        std::vector<__float16_t> max(
            reduced_size, std::numeric_limits<__float16_t>::lowest());
        std::vector<__float16_t> sum(reduced_size, (__float16_t)0.f);

        for (size_t i = 0; i < out_side; i++) {
            const __float16_t *ptr_input = input + i * axis_dim * in_side;
            const __float16_t *ptr_input_vl = ptr_input;

            __float16_t *ptr_output = output + i * axis_dim * in_side;
            __float16_t *ptr_output_vl = ptr_output;

            __float16_t *ptr_max = max.data() + i * in_side;
            __float16_t *ptr_max_vl = ptr_max;

            __float16_t *ptr_sum = sum.data() + i * in_side;
            __float16_t *ptr_sum_vl = ptr_sum;

            // max
            for (size_t j = 0; j < axis_dim; j++) {
                ptr_max_vl = ptr_max;
                auto n = in_side;
                while (n) {
                    auto vl = vsetvl_e16m4(n);

                    auto v_in = vle16_v_f16m4(ptr_input_vl, vl);
                    auto v_max = vle16_v_f16m4(ptr_max_vl, vl);

                    v_max = vfmax_vv_f16m4(v_in, v_max, vl);
                    vse16_v_f16m4(ptr_max_vl, v_max, vl);

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
                    auto vl = vsetvl_e16m4(n);

                    auto v_in = vle16_v_f16m4(ptr_input_vl, vl);
                    auto v_max = vle16_v_f16m4(ptr_max_vl, vl);
                    auto v_sum = vle16_v_f16m4(ptr_sum_vl, vl);

                    auto v_out =
                        exp_ph(vfmul_vf_f16m4(vfsub_vv_f16m4(v_in, v_max, vl),
                                              beta, vl),
                               vl);
                    vse16_v_f16m4(ptr_output_vl, v_out, vl);

                    v_sum = vfadd_vv_f16m4(v_sum, v_out, vl);
                    vse16_v_f16m4(ptr_sum_vl, v_sum, vl);

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
                    auto vl = vsetvl_e16m4(n);

                    auto v_out = vle16_v_f16m4(ptr_output_vl, vl);
                    auto v_sum = vle16_v_f16m4(ptr_sum_vl, vl);

                    v_out = vfdiv_vv_f16m4(v_out, v_sum, vl);
                    vse16_v_f16m4(ptr_output_vl, v_out, vl);

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
    if (typecode == typecode_t::dt_float16) {
        return optimized_softmax_half_impl(IN_CAST(__float16_t, input),
                                           OUT_CAST(__float16_t, output),
                                           in_shape, axis, __float16_t(beta));
    }
    return optimized_softmax_impl(
        IN_CAST(float, input), OUT_CAST(float, output), in_shape, axis, beta);
//    TYPE_SELECT_SOFTMAX(typecode, SOFTMAX_IMPL);
#endif
    return stackvm::reference::softmax(typecode, input, output, in_shape,
                                       in_strides, out_strides, axis, beta);
}