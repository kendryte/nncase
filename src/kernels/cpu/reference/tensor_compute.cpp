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

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class TOp>
result<void> binary_impl(TOp &&op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_b_shape, const runtime_shape_t &out_shape,
    runtime_shape_t off_prefix, runtime_shape_t::const_iterator off_begin, runtime_shape_t::const_iterator off_end, value_range<float> fused_activation) noexcept
{
    const auto head = *off_begin++;
    off_prefix.push_back(0);
    if (off_begin == off_end)
    {
        for (size_t i = 0; i < head; i++)
        {
            off_prefix.back() = i;
            const auto in_a_off = kernels::details::get_reduced_offset(off_prefix, in_a_shape);
            const auto in_b_off = kernels::details::get_reduced_offset(off_prefix, in_b_shape);
            const auto a = input_a[offset(in_a_shape, in_a_off)];
            const auto b = input_b[offset(in_b_shape, in_b_off)];
            output[offset(out_shape, off_prefix)] = kernels::details::apply_activation(op(a, b), fused_activation);
        }
    }
    else
    {
        for (size_t i = 0; i < head; i++)
        {
            off_prefix.back() = i;
            try_(binary_impl(std::forward<TOp>(op), input_a, input_b, output, in_a_shape, in_b_shape, out_shape, off_prefix,
                off_begin, off_end, fused_activation));
        }
    }

    return ok();
}

template <class TOp>
result<void> binary_impl(TOp &&op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_b_shape, value_range<float> fused_activation) noexcept
{
    const auto out_shape = kernels::details::get_binary_output_shape(in_a_shape, in_b_shape);
    return binary_impl(std::forward<TOp>(op), input_a, input_b, output, in_a_shape, in_b_shape, out_shape, runtime_shape_t(),
        out_shape.cbegin(), out_shape.cend(), fused_activation);
}
}

#define BINARY_IMPL(op, funct) \
    case op:                   \
        return binary_impl(funct, input_a, input_b, output, in_a_shape, in_b_shape, fused_activation)

result<void> reference::binary(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_b_shape, value_range<float> fused_activation) noexcept
{
    switch (op)
    {
        BINARY_IMPL(binary_add, std::plus<float>());
        BINARY_IMPL(binary_sub, std::minus<float>());
        BINARY_IMPL(binary_mul, std::multiplies<float>());
        BINARY_IMPL(binary_div, std::divides<float>());
        BINARY_IMPL(binary_min, [](float a, float b) { return std::min(a, b); });
        BINARY_IMPL(binary_max, [](float a, float b) { return std::max(a, b); });
        BINARY_IMPL(binary_pow, [](float a, float b) { return std::pow(a, b); });
    default:
        return err(std::errc::not_supported);
    }
}
