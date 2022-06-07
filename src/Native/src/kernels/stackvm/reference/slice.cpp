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
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/util.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class T>
result<void> slice_impl(const T *input, T *output, const dims_t &in_shape,
    const strides_t &in_strides, const strides_t &out_strides, const axes_t &begins, const axes_t &ends, const axes_t &strides,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(in_shape, [&](const dims_t &index) -> result<void> {
        dims_t out_index(index.size());
        for (size_t i = 0; i < index.size(); i++)
        {
            const auto stride = strides[i];
            if (stride > 0)
            {
                if (index[i] < begins[i] || index[i] >= static_cast<size_t>(ends[i]))
                    return ok();
            }
            else
            {
                if (static_cast<int32_t>(index[i]) <= ends[i] || index[i] > begins[i])
                    return ok();
            }

            auto out_div = div((int32_t)(index[i] - begins[i]), strides[i]);
            if (out_div.rem)
                return ok();
            out_index[i] = (size_t)out_div.quot;
        }

        output[offset(out_strides, out_index)] = input[offset(in_strides, index)];
        return ok();
    });
}
}

#define SLICE_IMPL(size, type) \
    case size:                 \
        return slice_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, in_strides, out_strides, begins, ends, strides, context)

result<void> slice_impl(datatype_t type, const gsl::byte *input, gsl::byte *output, const dims_t &in_shape,
    const strides_t &in_strides, const strides_t &out_strides, const axes_t &begins, const axes_t &ends, const axes_t &strides,
    kernel_context &context) noexcept
{
    switch (runtime::get_bytes(type))
    {
        SLICE_IMPL(1, uint8_t);
        SLICE_IMPL(2, uint16_t);
        SLICE_IMPL(4, uint32_t);
        SLICE_IMPL(8, uint64_t);
    default:
        return err(std::errc::not_supported);
    }
}

dims_t infer_shape(const dims_t& in_shape, const axes_t& begins, const axes_t& ends,
                   const axes_t& strides, const axes_t& axes) {
    auto new_shape = in_shape;
    for (int i = 0; i < axes.size(); ++i) {
        auto axis = positive_index(axes[i], in_shape.size());
        auto begin = begins[i];
        auto end = ends[i];
        auto stride = strides[i];
        auto old = in_shape[axis];
        begin = begin >= 0 ? begin : old + begin;
        end = end >= 0 ? end : old + begin;
        //stride = stride >= 0 ? stride : -stride;
        new_shape[axis] = (end - begin) / stride;
    }
    return new_shape;
}

result<value_t> nncase::kernels::stackvm::slice(value_t input, value_t begins,
                                                value_t ends, value_t axes,
                                                value_t strides, value_t output,
                                                kernel_context &context) {
    try_input(in_mem, input);
    try_axes(begins_value, begins);
    try_axes(ends_value, ends);
    try_axes(axes_value, axes);
    try_axes(strides_value, strides);
    auto out_shape = infer_shape(input_tensor->shape(), begins_value, ends_value, strides_value, axes_value);
    try_output(out_mem, output, input_tensor->dtype(), out_shape);

    try_(slice_impl(input_tensor->dtype(), in_mem, out_mem,
                    input_tensor->shape(), input_tensor->strides(),
                    output_tensor->strides(), begins_value, ends_value,
                    strides_value, context));

    return ok(output);
}