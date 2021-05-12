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
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

namespace
{
template <class T>
result<void> slice_continuous_impl(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, NNCASE_UNUSED const runtime_axis_t &strides,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    // if strides
    size_t elemsize = sizeof(T);
    auto *out_ptr = output;
    auto dims = in_shape.size();
    if (dims == 1)
    {
        auto copy_size = (ends[0] - begins[0]) * elemsize;
        memcpy(out_ptr, input + begins[0], copy_size);
    }
    else if (dims == 2)
    {
        for (size_t i = begins[0]; i < ends[0]; ++i)
        {
            const auto distance = ends[1] - begins[1];
            auto copy_size = distance * elemsize;
            const auto *in_ptr = input + i * in_strides[0] + begins[1];
            memcpy(out_ptr, in_ptr, copy_size);
            out_ptr += distance;
        }
    }
    else if (dims == 3)
    {
        for (size_t i = begins[0]; i < ends[0]; ++i)
        {
            for (size_t j = begins[1]; j < ends[1]; ++j)
            {
                const auto distance = ends[2] - begins[2];
                auto copy_size = distance * elemsize;
                const auto *in_ptr = input + i * in_strides[0] + j * in_strides[1] + in_strides[2];
                memcpy(out_ptr, in_ptr, copy_size);
                out_ptr += distance;
            }
        }
    }
    else if (dims == 4)
    {
        for (size_t i = begins[0]; i < ends[0]; ++i)
        {
            for (size_t j = begins[1]; j < ends[1]; ++j)
            {
                for (size_t k = begins[2]; k < ends[2]; ++k)
                {
                    const auto distance = ends[3] - begins[3];
                    auto copy_size = distance * elemsize;
                    const auto *in_ptr = input + i * in_strides[0] + j * in_strides[1] + k * in_strides[2] + in_strides[3];
                    memcpy(out_ptr, in_ptr, copy_size);
                    out_ptr += distance;
                }
            }
        }
    }
    return ok();
}

template <class T>
result<void> slice_impl(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, NNCASE_UNUSED const runtime_axis_t &strides,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    // if strides
    size_t elemsize = sizeof(T);
    auto *out_ptr = output;
    auto dims = in_shape.size();
    runtime_shape_t in_index(dims);
    runtime_shape_t out_index(dims);
    if (dims == 1)
    {
        auto copy_size = (ends[0] - begins[0]) * elemsize;
        memcpy(out_ptr, input + begins[0], copy_size);
    }
    else if (dims == 2)
    {
        for (size_t i = begins[0]; i < ends[0]; ++i)
        {
            in_index[0] = i;
            in_index[1] = begins[1];
            const auto distance = ends[1] - begins[1];
            auto copy_size = distance * elemsize;
            const auto *in_ptr = input + offset(in_strides, in_index);
            out_ptr = output + offset(out_strides, out_index);
            memcpy(out_ptr, in_ptr, copy_size);
            ++out_index[0];
        }
    }
    else if (dims == 3)
    {
        for (size_t i = begins[0]; i < ends[0]; ++i)
        {
            in_index[0] = i;
            out_index[1] = 0;
            for (size_t j = begins[1]; j < ends[1]; ++j)
            {
                in_index[1] = j;
                in_index[2] = begins[2];
                const auto distance = ends[2] - begins[2];
                auto copy_size = distance * elemsize;
                const auto *in_ptr = input + offset(in_strides, in_index);
                out_ptr = output + offset(out_strides, out_index);
                memcpy(out_ptr, in_ptr, copy_size);
                ++out_index[1];
            }
            ++out_index[0];
        }
    }
    else if (dims == 4)
    {
        for (size_t i = begins[0]; i < ends[0]; ++i)
        {
            in_index[0] = i;
            out_index[1] = 0;
            for (size_t j = begins[1]; j < ends[1]; ++j)
            {
                in_index[1] = j;
                out_index[2] = 0;
                for (size_t k = begins[2]; k < ends[2]; ++k)
                {
                    in_index[2] = k;
                    in_index[3] = begins[3];
                    const auto distance = ends[3] - begins[3];
                    auto copy_size = distance * elemsize;
                    const auto *in_ptr = input + offset(in_strides, in_index);
                    out_ptr = output + offset(out_strides, out_index);
                    memcpy(out_ptr, in_ptr, copy_size);
                    ++out_index[2];
                }
                ++out_index[1];
            }
            ++out_index[0];
        }
    }
    return ok();
}
}

#define SLICE_IMPL(size, type) \
    case size:                 \
        return slice_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, in_strides, out_strides, begins, ends, strides, context)

result<void> optimized::slice(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, const runtime_axis_t &strides,
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
