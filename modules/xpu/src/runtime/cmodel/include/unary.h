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

#include "../../gsl-lite.hpp"
#include <apply.h>
#include <runtime_utils.h>

using namespace nncase::runtime::xpu;
namespace kernels {

namespace {
template <class T, class TOp>
void unary_impl(TOp &&op, const T *input, T *output,
                gsl::span<const size_t> input_strides,
                gsl::span<const size_t> out_shape,
                gsl::span<const size_t> out_strides) noexcept {
    apply(out_shape, [&](gsl::span<const size_t> index) -> void {
        const auto v = input[offset(input_strides, index)];
        output[offset(out_strides, index)] = (T)op(v);
    });
}

#define UNARY_IMPL_OP(op, callable)                                            \
    case unary_op_t::op:                                                       \
        return unary_impl(callable, input, output, input_strides, out_shape,   \
                          out_strides)

template <class T>
void unary_impl(unary_op_t op, const T *input, T *output,
                gsl::span<const size_t> input_strides,
                gsl::span<const size_t> out_shape,
                gsl::span<const size_t> out_strides) noexcept {
    if (std::is_same_v<T, float>) {
        switch (op) {
            UNARY_IMPL_OP(abs, nncase_mt->float_unary_abs);
            UNARY_IMPL_OP(acos, nncase_mt->float_unary_cos);
            UNARY_IMPL_OP(acosh, nncase_mt->float_unary_cosh);
            UNARY_IMPL_OP(asin, nncase_mt->float_unary_asin);
            UNARY_IMPL_OP(asinh, nncase_mt->float_unary_asinh);
            UNARY_IMPL_OP(ceil, nncase_mt->float_unary_ceil);
            UNARY_IMPL_OP(cos, nncase_mt->float_unary_cos);
            UNARY_IMPL_OP(cosh, nncase_mt->float_unary_cosh);
            UNARY_IMPL_OP(exp, nncase_mt->float_unary_exp);
            UNARY_IMPL_OP(floor, nncase_mt->float_unary_floor);
            UNARY_IMPL_OP(log, nncase_mt->float_unary_log);
            UNARY_IMPL_OP(logical_not, nncase_mt->float_unary_logical_not);
            UNARY_IMPL_OP(neg, nncase_mt->float_unary_neg);
            UNARY_IMPL_OP(round, nncase_mt->float_unary_round);
            UNARY_IMPL_OP(rsqrt, nncase_mt->float_unary_rsqrt);
            UNARY_IMPL_OP(sign, nncase_mt->float_unary_sign);
            UNARY_IMPL_OP(sin, nncase_mt->float_unary_sin);
            UNARY_IMPL_OP(sinh, nncase_mt->float_unary_sinh);
            UNARY_IMPL_OP(sqrt, nncase_mt->float_unary_sqrt);
            UNARY_IMPL_OP(square, nncase_mt->float_unary_square);
            UNARY_IMPL_OP(tanh, nncase_mt->float_unary_tanh);
            UNARY_IMPL_OP(swish, [](float x) {
                return x / (1 + nncase_mt->float_unary_exp(-x));
            });
            UNARY_IMPL_OP(gelu, [](float x) {
                return 0.5f * (x) *
                       (1.f + nncase_mt->float_unary_erf(
                                  x / nncase_mt->float_unary_sqrt(2.f)));
            });
        default:
            runtime_util->rt_assert(false, (char *)"Unsupported Unary Op!");
        }
    } else {
        runtime_util->rt_assert(false, (char *)"Unsupported Unary Type!");
    }
} // namespace

} // namespace

template <class T>
void unary(unary_op_t op, const T *input, T *output,
           gsl::span<const size_t> in_strides,
           gsl::span<const size_t> out_shape,
           gsl::span<const size_t> out_strides) noexcept {
    unary_impl(op, input, output, in_strides, out_shape, out_strides);
}

void swishb(const float *input, float *output, float beta,
            gsl::span<const size_t> input_strides,
            gsl::span<const size_t> out_shape,
            gsl::span<const size_t> out_strides) noexcept {
    unary_impl(
        [&beta](float x) {
            return x * (1 / (1 + nncase_mt->float_unary_exp(-(x * beta))));
        },
        input, output, input_strides, out_shape, out_strides);
}
} // namespace kernels