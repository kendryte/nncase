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
result<void> concat_continuous_impl(gsl::span<const gsl::byte *const> inputs, T *output, const runtime_shape_t &out_shape,
    gsl::span<const runtime_shape_t> &in_strides, NNCASE_UNUSED const runtime_shape_t &out_strides, size_t axis, const runtime_shape_t &concat_dims, NNCASE_UNUSED kernel_context &context) noexcept
{
    runtime_shape_t in_shape(out_shape), in_index(out_shape.size());
    size_t elemsize = sizeof(T);
    auto subsize = std::accumulate(in_shape.begin() + (axis + 1), in_shape.end(), 1, std::multiplies<int>());
    auto *out_ptr = output;
    auto line_copy = [&](size_t n) {
        const auto size = concat_dims[n] * subsize;
        const auto in_offset = offset(in_strides[n], in_index);
        auto *in_ptr = reinterpret_cast<const T *>(inputs[n]) + in_offset;
        memcpy(out_ptr, in_ptr, size * elemsize);
        out_ptr += size;
    };
    if (axis == 0)
    {
        for (size_t n = 0; n < inputs.size(); ++n)
        {
            const auto size = concat_dims[n] * subsize;
            memcpy(out_ptr, inputs[n], size * elemsize);
            out_ptr += size;
        }
    }
    else if (axis == 1)
    {
        for (size_t height = 0; height < in_shape[0]; ++height)
        {
            in_index[0] = height;
            for (size_t n = 0; n < inputs.size(); ++n)
            {
                line_copy(n);
            }
        }
    }
   else if (axis == 2)
    {
        for (size_t channel = 0; channel < in_shape[0]; ++channel)
        {
            in_index[0] = channel;
            for (size_t height = 0; height < in_shape[1]; ++height)
            {
                in_index[1] = height;
                for (size_t n = 0; n < inputs.size(); ++n)
                {
                    line_copy(n);
                }
            }
        }
    }
    else if (axis == 3)
    {
        for (size_t batch = 0; batch < in_shape[0]; ++batch)
        {
            in_index[0] = batch;
            for (size_t channel = 0; channel < in_shape[1]; ++channel)
            {
                in_index[1] = channel;
                for (size_t height = 0; height < in_shape[2]; ++height)
                {
                    in_index[2] = height;
                    for (size_t n = 0; n < inputs.size(); ++n)
                    {
                        line_copy(n);
                    }
                }
            }
        }
    }
    return ok();
}

template <class T>
result<void> concat_impl(gsl::span<const gsl::byte *const> inputs, T *output, const runtime_shape_t &out_shape,
    gsl::span<const runtime_shape_t> &in_strides, const runtime_shape_t &out_strides, size_t axis, const runtime_shape_t &concat_dims, NNCASE_UNUSED kernel_context &context) noexcept
{
    runtime_shape_t in_shape(out_shape);
    size_t elemsize = sizeof(T);
    auto *out_ptr = output;
    auto dims = in_strides[0].size();
    runtime_shape_t out_index(dims);
    runtime_shape_t in_index(dims);
    auto line_copy = [&](size_t width, size_t n) {
        out_ptr = output + offset(out_strides, out_index);
        const auto *in_ptr = reinterpret_cast<const T *>(inputs[n]) + offset(in_strides[n], in_index);
        memcpy(out_ptr, in_ptr, width * elemsize);
    };

    if (dims == 1)
    {
        for (size_t n = 0; n < inputs.size(); ++n)
        {
            const auto width = concat_dims[n];
            out_ptr = output + out_index[0] * out_strides[0];
            memcpy(out_ptr, inputs[n], width * elemsize);
            out_index[0] += width;
        }
    }
    else if (dims == 2)
    {
        if (axis == 0)
        {
            const auto width = out_shape[1];
            for (size_t n = 0; n < inputs.size(); ++n)
            {
                for (size_t height = 0; height < concat_dims[n]; ++height)
                {
                    in_index[0] = height;
                    line_copy(width, n);
                    ++out_index[0];
                }
            }
        }
        else
        {
            for (size_t height = 0; height < in_shape[0]; ++height)
            {
                in_index[0] = height;
                out_index[0] = height;
                out_index[1] = 0;
                // set per input value in same row
                // so process once need move col(out_index[1])
                for (size_t n = 0; n < inputs.size(); ++n)
                {
                    const auto width = concat_dims[n];
                    line_copy(width, n);
                    out_index[1] += width;
                }
            }
        }
    }
    else if (dims == 3)
    {
        if (axis == 0)
        {
            const auto width = out_shape[2];
            for (size_t n = 0; n < inputs.size(); ++n)
            {
                for (size_t channel = 0; channel < concat_dims[n]; ++channel)
                {
                    in_index[0] = channel;
                    for (size_t height = 0; height < in_shape[1]; ++height)
                    {
                        in_index[1] = height;
                        out_index[1] = height;
                        line_copy(width, n);
                    }
                    ++out_index[0];
                }
            }
        }
        else if (axis == 1)
        {
            const auto width = in_shape[2];

            for (size_t channel = 0; channel < in_shape[0]; ++channel)
            {
                in_index[0] = channel;
                out_index[0] = channel;
                out_index[1] = 0;
                for (size_t n = 0; n < inputs.size(); ++n)
                {
                    for (size_t height = 0; height < concat_dims[n]; ++height)
                    {
                        in_index[1] = height;
                        line_copy(width, n);
                        ++out_index[1];
                    }
                }
            }
        }
        else
        {
            for (size_t channel = 0; channel < in_shape[0]; ++channel)
            {
                in_index[0] = channel;
                out_index[0] = channel;
                for (size_t height = 0; height < in_shape[1]; ++height)
                {
                    in_index[1] = height;
                    out_index[1] = height;
                    out_index[2] = 0;
                    for (size_t n = 0; n < inputs.size(); ++n)
                    {
                        const auto width = concat_dims[n];
                        line_copy(width, n);
                        out_index[2] += width;
                    }
                }
            }
        }
    }
    else if (dims == 4)
    {
        if (axis == 0)
        {
            const auto width = out_shape[3];
            for (size_t n = 0; n < inputs.size(); ++n)
            {
                for (size_t batch = 0; batch < concat_dims[n]; ++batch)
                {
                    in_index[0] = batch;
                    for (size_t channel = 0; channel < in_shape[1]; ++channel)
                    {
                        in_index[1] = channel;
                        out_index[1] = channel;
                        for (size_t height = 0; height < in_shape[2]; ++height)
                        {
                            in_index[2] = height;
                            out_index[2] = height;
                            line_copy(width, n);
                        }
                    }
                    ++out_index[0];
                }
            }
        }
        else if (axis == 1)
        {
            const auto width = out_shape[3];
            for (size_t batch = 0; batch < in_shape[0]; ++batch)
            {
                in_index[0] = batch;
                out_index[0] = batch;
                out_index[1] = 0;
                for (size_t n = 0; n < inputs.size(); ++n)
                {
                    for (size_t channel = 0; channel < concat_dims[n]; ++channel)
                    {
                        in_index[1] = channel;
                        for (size_t height = 0; height < in_shape[2]; ++height)
                        {
                            in_index[2] = height;
                            out_index[2] = height;                            
                            line_copy(width, n);
                        }
                        ++out_index[1];
                    }
                }
            }
        }
        else if (axis == 2)
        {
            const auto width = out_shape[3];
            for (size_t batch = 0; batch < in_shape[0]; ++batch)
            {
                in_index[0] = batch;
                out_index[0] = batch;
                for (size_t channel = 0; channel < in_shape[1]; ++channel)
                {
                    in_index[1] = channel;
                    out_index[1] = channel;
                    out_index[2] = 0;
                    for (size_t n = 0; n < inputs.size(); ++n)
                    {
                        for (size_t height = 0; height < concat_dims[n]; ++height)
                        {
                            in_index[2] = height;
                            line_copy(width, n);
                            ++out_index[2];
                        }
                    }
                }
            }
        }
        else
        {
            for (size_t batch = 0; batch < in_shape[0]; ++batch)
            {
                in_index[0] = batch;
                out_index[0] = batch;
                for (size_t channel = 0; channel < in_shape[1]; ++channel)
                {
                    in_index[1] = channel;
                    out_index[1] = channel;
                    for (size_t height = 0; height < in_shape[2]; ++height)
                    {
                        in_index[2] = height;
                        out_index[2] = height;
                        out_index[3] = 0;
                        for (size_t n = 0; n < inputs.size(); ++n)
                        {
                            const auto width = concat_dims[n];
                            line_copy(width, n);
                            out_index[3] += width;
                        }
                    }
                }
            }
        }
    }
    return ok();
}
}

#define CONCAT_IMPL(size, type) \
    case size:                  \
        return concat_impl(inputs, reinterpret_cast<type *>(output), out_shape, in_strides, out_strides, axis, concat_dims, context)

#define CONCAT_CONTINUOUS_IMPL(size, type) \
    case size:                  \
        return concat_continuous_impl(inputs, reinterpret_cast<type *>(output), out_shape, in_strides, out_strides, axis, concat_dims, context)

result<void> optimized::concat(datatype_t type, gsl::span<const gsl::byte *const> inputs, gsl::byte *output, const runtime_shape_t &out_shape,
    gsl::span<const runtime_shape_t> in_strides, const runtime_shape_t &out_strides, size_t axis, const runtime_shape_t &concat_dims, kernel_context &context) noexcept
{
    runtime_shape_t in_shape(out_shape);
    if (!is_continuous(out_shape, out_strides))
    {
        TYPE_IMPL_SELECT(type, CONCAT_IMPL);
    }
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto tmp = in_shape[i];
        in_shape[axis] = concat_dims[i];
        if (!is_continuous(in_shape, in_strides[i]))
        {
            TYPE_IMPL_SELECT(type, CONCAT_IMPL);
        }
        in_shape[i] = tmp;
    }
    TYPE_IMPL_SELECT(type, CONCAT_CONTINUOUS_IMPL);
}
