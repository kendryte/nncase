/* Copyright 2019-2020 Canaan Inc.
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
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class TOp>
result<void> binary_impl(TOp &&op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides, value_range<float> fused_activation) noexcept
{
    const auto out_shape = kernels::detail::get_binary_output_shape(in_a_shape, in_b_shape);
    return apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto in_a_index = kernels::detail::get_reduced_offset(index, in_a_shape);
        const auto in_b_index = kernels::detail::get_reduced_offset(index, in_b_shape);
        const auto a = input_a[offset(in_a_strides, in_a_index)];
        const auto b = input_b[offset(in_b_strides, in_b_index)];
        output[offset(out_strides, index)] = kernels::detail::apply_activation(op(a, b), fused_activation);
        return ok();
    });
}
}

#define BINARY_IMPL(op, funct) \
    case op:                   \
        return binary_impl(funct, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_strides, fused_activation)

result<void> reference::binary(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides, value_range<float> fused_activation) noexcept
{
    switch (op)
    {
        BINARY_IMPL(binary_add, std::plus<float>());
        BINARY_IMPL(binary_sub, std::minus<float>());
        BINARY_IMPL(binary_mul, std::multiplies<float>());
        BINARY_IMPL(binary_div, std::divides<float>());
        BINARY_IMPL(binary_min, [](float a, float b) { return std::min(a, b); });
        BINARY_IMPL(binary_max, [](float a, float b) { return std::max(a, b); });
        BINARY_IMPL(binary_pow, powf);
    default:
        return err(std::errc::not_supported);
    }
}
