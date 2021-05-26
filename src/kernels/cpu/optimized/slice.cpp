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
result<void> slice_contiguous_impl(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, NNCASE_UNUSED const runtime_axis_t &strides,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    size_t elemsize = sizeof(T);
    auto *out_ptr = output;
    auto dims = in_shape.size();
    runtime_shape_t in_index(dims);

    auto line_copy = [&]() {
        auto d = dims - 1;
        const auto distance = ends[d] - begins[d];
        const auto copy_size = distance * elemsize;
        const auto *in_ptr = input + offset(in_strides, in_index);
        memcpy(out_ptr, in_ptr, copy_size);
        out_ptr += distance;
    };

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
            line_copy();
        }
    }
    else if (dims == 3)
    {
        for (size_t i = begins[0]; i < ends[0]; ++i)
        {
            in_index[0] = i;
            for (size_t j = begins[1]; j < ends[1]; ++j)
            {
                in_index[1] = j;
                in_index[2] = begins[2];
                line_copy();
            }
        }
    }
    else if (dims == 4)
    {
        for (size_t i = begins[0]; i < ends[0]; ++i)
        {
            in_index[0] = i;
            for (size_t j = begins[1]; j < ends[1]; ++j)
            {
                in_index[1] = j;
                for (size_t k = begins[2]; k < ends[2]; ++k)
                {
                    in_index[2] = k;
                    in_index[3] = begins[3];
                    line_copy();
                }
            }
        }
    }
    return ok();
}

template <class T, class Callable>
result<void> _slice_impl(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, 
    const runtime_axis_t &strides, Callable &&line_copy,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    // if strides
    auto dims = in_shape.size();
    runtime_shape_t in_index(dims);
    runtime_shape_t out_index(dims);
    if (dims == 1)
    {
        in_index[0] = begins[0];
        line_copy(input, output, in_strides, out_strides, begins, ends, in_index, out_index, dims, strides);
    }
    else if (dims == 2)
    {
        for (size_t i = begins[0]; i < ends[0]; i += strides[0])
        {
            in_index[0] = i;
            line_copy(input, output, in_strides, out_strides, begins, ends, in_index, out_index, dims, strides);
            ++out_index[0];
        }
    }
    else if (dims == 3)
    {
        for (size_t i = begins[0]; i < ends[0]; i += strides[0])
        {
            in_index[0] = i;
            out_index[1] = 0;
            for (size_t j = begins[1]; j < ends[1]; j += strides[1])
            {
                in_index[1] = j;
                line_copy(input, output, in_strides, out_strides, begins, ends, in_index, out_index, dims, strides);
                ++out_index[1];
            }
            ++out_index[0];
        }
    }
    else if (dims == 4)
    {
        for (size_t i = begins[0]; i < ends[0]; i += strides[0])
        {
            in_index[0] = i;
            out_index[1] = 0;
            for (size_t j = begins[1]; j < ends[1]; j += strides[1])
            {
                in_index[1] = j;
                out_index[2] = 0;
                for (size_t k = begins[2]; k < ends[2]; k += strides[2])
                {
                    in_index[2] = k;
                    line_copy(input, output, in_strides, out_strides, begins, ends, in_index, out_index, dims, strides);
                    ++out_index[2];
                }
                ++out_index[1];
            }
            ++out_index[0];
        }
    }
    return ok();
}

template <class T>
result<void> slice_linecopy_impl(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, const runtime_axis_t &strides,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    return _slice_impl(input, output, in_shape, in_strides, out_strides, begins, ends, strides, 
        [](const T *input, T *output,
            const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
            const runtime_shape_t &begins, const runtime_shape_t &ends,
            runtime_shape_t in_index, runtime_shape_t out_index, size_t dims, NNCASE_UNUSED const runtime_axis_t &strides) {
            --dims;
            in_index[dims] = begins[dims];
            const auto distance = ends[dims] - begins[dims];
            auto copy_size = distance * sizeof(T);
            const auto *in_ptr = input + offset(in_strides, in_index);
            auto *out_ptr = output + offset(out_strides, out_index);
            memcpy(out_ptr, in_ptr, copy_size);
        },
        context);
}

template <class T>
result<void> slice_strides_impl(const T *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, const runtime_axis_t &strides,
    NNCASE_UNUSED kernel_context &context) noexcept
{
    // return _slice_impl(input, output, in_shape, in_strides, out_strides, begins, ends, strides, _strides_linecopy<T>, context);
    return _slice_impl(
        input, output, in_shape, in_strides, out_strides, begins, ends, strides, 
        [](const T *input, T *output, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, 
            const runtime_shape_t &begins, const runtime_shape_t &ends, runtime_shape_t& in_index, runtime_shape_t& out_index, 
            size_t dims, const runtime_axis_t &strides) {
            --dims;
            for (size_t i = begins[dims]; i < ends[dims]; i += strides[dims])
            {
                in_index[dims] = i;
                output[offset(out_strides, out_index)] = input[offset(in_strides, in_index)];
                ++out_index[dims];
            }
            out_index[dims] = 0;
        },
        context);
}
}

#define SLICE_LINECOPY_IMPL(size, type) \
    case size:                 \
        return slice_linecopy_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, in_strides, out_strides, begins, ends, strides, context)

#define SLICE_CONTIGUOUS_IMPL(size, type) \
    case size:                 \
        return slice_contiguous_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, in_strides, out_strides, begins, ends, strides, context)

#define SLICE_STRIDES_IMPL(size, type) \
    case size:                            \
        return slice_strides_impl(reinterpret_cast<const type *>(input), reinterpret_cast<type *>(output), in_shape, in_strides, out_strides, begins, ends, strides, context)

result<void> optimized::slice(datatype_t type, const gsl::byte *input, gsl::byte *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, const runtime_shape_t &begins, const runtime_shape_t &ends, const runtime_axis_t &strides,
    kernel_context &context) noexcept
{
    auto dims = begins.size();
    runtime_shape_t out_shape(dims);
    for (size_t i = 0; i < dims; ++i)
    {
        out_shape[i] = ends[i] - begins[i];
    }
    
    for (size_t i = 0; i < dims; ++i)
    {
        if (strides[i] != 1)
        {
            // only last dims' stride is not 1
            if (strides[dims - 1] == 1)
            {
                TYPE_IMPL_SELECT(type, SLICE_LINECOPY_IMPL);
            }
            else
            {
                TYPE_IMPL_SELECT(type, SLICE_STRIDES_IMPL);
            }
        }
    }
    if (is_contiguous(in_shape, in_strides) && is_contiguous(out_shape, out_strides))
    {
        // all of strides are 1 and contiguous
        TYPE_IMPL_SELECT(type, SLICE_CONTIGUOUS_IMPL);
    }
    else 
    {
        // summary memory is not continous, but line is contiguous
        TYPE_IMPL_SELECT(type, SLICE_LINECOPY_IMPL);
    }
}
