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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class TOp>
result<void> unary_impl(TOp &&op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto v = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = op(v);
        return ok();
    });
}
}

#define UNARY_IMPL(op, funct) \
    case op:                  \
        return unary_impl(funct, input, output, shape, in_strides, out_strides, context)

result<void> reference::unary(unary_op_t op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    switch (op)
    {
        UNARY_IMPL(unary_abs, fabsf);
        UNARY_IMPL(unary_acos, acosf);
        UNARY_IMPL(unary_asin, asinf);
        UNARY_IMPL(unary_ceil, ceilf);
        UNARY_IMPL(unary_cos, cosf);
        UNARY_IMPL(unary_exp, expf);
        UNARY_IMPL(unary_floor, floorf);
        UNARY_IMPL(unary_log, logf);
        UNARY_IMPL(unary_logical_not, [](float v) { return !v; });
        UNARY_IMPL(unary_neg, std::negate<float>());
        UNARY_IMPL(unary_round, rintf);
        UNARY_IMPL(unary_rsqrt, [](float v) { return 1.f / sqrtf(v); });
        UNARY_IMPL(unary_sign, [](float v) { return (0.f < v) - (v < 0.f); });
        UNARY_IMPL(unary_sin, sinf);
        UNARY_IMPL(unary_sqrt, sqrtf);
        UNARY_IMPL(unary_square, [](float v) { return v * v; });
        UNARY_IMPL(unary_tanh, tanhf);
        UNARY_IMPL(unary_erf, erff);
    default:
        return err(std::errc::not_supported);
    }
}
