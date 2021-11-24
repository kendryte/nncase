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
template <class T>
result<void> slice_impl(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_axis_t &ends, const runtime_axis_t &strides,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        runtime_shape_t out_index(index.size());
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

result<void> reference::slice(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_axis_t &ends, const runtime_axis_t &strides,
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
