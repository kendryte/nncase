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

#include <apply.h>
#include <gsl/gsl-lite.hpp>
#include <runtime_utils.h>

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
    switch (op) {
        UNARY_IMPL_OP(abs, fabsf);
        UNARY_IMPL_OP(acos, acosf);
        UNARY_IMPL_OP(acosh, acoshf);
        UNARY_IMPL_OP(asin, asinf);
        UNARY_IMPL_OP(asinh, asinhf);
        UNARY_IMPL_OP(ceil, ceilf);
        UNARY_IMPL_OP(cos, cosf);
        UNARY_IMPL_OP(cosh, coshf);
        UNARY_IMPL_OP(exp, expf);
        UNARY_IMPL_OP(floor, floorf);
        UNARY_IMPL_OP(log, logf);
        UNARY_IMPL_OP(logical_not, [](float v) { return !v; });
        UNARY_IMPL_OP(neg, std::negate<float>());
        UNARY_IMPL_OP(round, [](float v) { return round(v); });
        UNARY_IMPL_OP(rsqrt, [](float v) { return 1.f / sqrtf(v); });
        UNARY_IMPL_OP(sign, [](float v) { return (0.f < v) - (v < 0.f); });
        UNARY_IMPL_OP(sin, sinf);
        UNARY_IMPL_OP(sinh, sinhf);
        UNARY_IMPL_OP(sqrt, sqrtf);
        UNARY_IMPL_OP(square, [](float v) { return v * v; });
        UNARY_IMPL_OP(tanh, tanhf);
    default:
        return;
    }
}

} // namespace

template <class T>
void unary(unary_op_t op, const T *input, T *output,
           gsl::span<const size_t> in_strides,
           gsl::span<const size_t> out_shape,
           gsl::span<const size_t> out_strides) noexcept {
    unary_impl(op, input, output, in_strides, out_shape, out_strides);
}

} // namespace kernels