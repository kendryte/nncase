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
#include <iostream>
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
template <class TOp, class T>
result<void> compare_impl(TOp &&op, const T *input_a, const T *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept
{
    return apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto in_a_index = kernels::detail::get_reduced_offset(index, in_a_shape);
        const auto in_b_index = kernels::detail::get_reduced_offset(index, in_b_shape);
        const auto a = input_a[offset(in_a_strides, in_a_index)];
        const auto b = input_b[offset(in_b_strides, in_b_index)];
        output[offset(out_strides, index)] = static_cast<bool>(op(a, b));
        return ok();
    });
}
}

#define COMPARE_IMPL(op, funct) \
    case op:                    \
        return compare_impl(funct, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_shape, out_strides)

template result<void> reference::compare<uint8_t>(compare_op_t op, const uint8_t *input_a, const uint8_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept;

template result<void> reference::compare<float>(compare_op_t op, const float *input_a, const float *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept;

template result<void> reference::compare<int32_t>(compare_op_t op, const int32_t *input_a, const int32_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept;

template result<void> reference::compare<int64_t>(compare_op_t op, const int64_t *input_a, const int64_t *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> reference::compare(compare_op_t op, const T *input_a, const T *input_b, bool *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides,
    const runtime_shape_t &in_b_shape, const runtime_shape_t &in_b_strides,
    const runtime_shape_t &out_shape, const runtime_shape_t &out_strides) noexcept
{
    switch (op)
    {
        COMPARE_IMPL(compare_equal, std::equal_to<T>());
        COMPARE_IMPL(compare_not_equal, std::not_equal_to<T>());
        COMPARE_IMPL(compare_greater, std::greater<T>());
        COMPARE_IMPL(compare_greater_equal, std::greater_equal<T>());
        COMPARE_IMPL(compare_less, std::less<T>());
        COMPARE_IMPL(compare_less_equal, std::less_equal<T>());
    default:
        std::cerr << "Unsupported compare op: " + compare_op_to_string(op) << std::endl;
        return err(std::errc::not_supported);
    }
}