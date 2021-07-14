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
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <string>
#include <sstream>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

namespace
{
// if constexpr can be used in C++17
// but k210 runtime only support C++14
template<class T>
void memset_(T *output, size_t output_size, T off_value)
{
    std::fill_n(output, output_size, off_value);
}

template<>
void memset_(int32_t *output, size_t output_size, int32_t off_value)
{
    memset(output, off_value, output_size);
}

template <class T>
result<void> onehot_impl(const int32_t *indices, T *output, const runtime_shape_t &indices_shape, const runtime_shape_t &out_shape,
    NNCASE_UNUSED const runtime_shape_t &out_strides, NNCASE_UNUSED size_t depth, T off_value, T on_value,
    size_t axis, NNCASE_UNUSED kernel_context &context)
{
    auto output_size = compute_size(out_shape);
    memset_(output, output_size, off_value);
    auto indices_size = compute_size(indices_shape);

    size_t out_size = std::accumulate(indices_shape.begin(), indices_shape.begin() + axis, 1, std::multiplies<size_t> {});
    size_t inner_size = std::accumulate(indices_shape.begin() + axis, indices_shape.end(), 1, std::multiplies<size_t> {});
    auto indices_dims = indices_shape.size();
    auto onehot_dims = indices_dims - axis;

    if (onehot_dims == 0)
    {
        // a depth is a line, set a value per line
        // next line
        for (size_t i = 0; i < indices_size; ++i)
        {
            if (*indices >= 0)
            {
                output[*indices] = on_value;
            }
            ++indices;
            output += depth;
        }
    }
    else if (onehot_dims == 1)
    {
        const auto x_size = indices_shape[indices_shape.size() - 1];
        // next indices_inner_size
        for (size_t i = 0; i < out_size; ++i)
        {
            for (size_t x = 0; x < x_size; ++x)
            {
                if (*indices >= 0)
                {
                    output[(*indices) * inner_size + x] = on_value;
                }
                ++indices;
            }
            output += inner_size * depth;
        }
    }
    else if (onehot_dims == 2)
    {
        const auto y_size = indices_shape[indices_shape.size() - 2];
        const auto x_size = indices_shape[indices_shape.size() - 1];
        for (size_t i = 0; i < out_size; ++i)
        {
            for (size_t y = 0; y < y_size; ++y)
            {
                for (size_t x = 0; x < x_size; ++x)
                {
                    if (*indices >= 0)
                    {
                        output[(*indices) * inner_size + y * x_size + x] = on_value;
                    }
                    ++indices;
                }
            }
            output += inner_size * depth;
        }
    }
    else if (onehot_dims == 3)
    {
        const auto c_size = indices_shape[indices_shape.size() - 3];
        const auto y_size = indices_shape[indices_shape.size() - 2];
        const auto x_size = indices_shape[indices_shape.size() - 1];
        const auto y_block_size = out_strides[out_strides.size() - 2];
        const auto c_block_size = out_strides[out_strides.size() - 3];
        for (size_t i = 0; i < out_size; ++i)
        {
            for (size_t c = 0; c < c_size; ++c)
            {
                for (size_t y = 0; y < y_size; ++y)
                {
                    for (size_t x = 0; x < x_size; ++x)
                    {
                        if (*indices >= 0)
                        {
                            output[(*indices) * inner_size + c * c_block_size + y * y_block_size + x] = on_value;
                        }
                        ++indices;
                    }
                }
            }
            output += inner_size * depth;
        }
    }
    else
    {
        return err(std::errc::result_out_of_range);
    }
    return ok();
}
}

#define ONEHOT_IMPL(size, type)                                                                              \
    case size:                                                                                               \
        return onehot_impl(indices, reinterpret_cast<type *>(output), indices_shape, out_shape, out_strides, \
            *reinterpret_cast<size_t *>(depth), *reinterpret_cast<type *>(off_value), *reinterpret_cast<type *>(on_value), axis, context);

result<void> optimized::onehot(datatype_t type, const int32_t *indices, gsl::byte *output, const runtime_shape_t &indices_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &out_strides, gsl::byte *depth, gsl::byte *off_value, gsl::byte *on_value, size_t axis,
    kernel_context &context) noexcept
{
    TYPE_IMPL_SELECT(type, ONEHOT_IMPL);
}
