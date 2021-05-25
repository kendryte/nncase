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
result<void> copy_continuous_impl(const T *src, T *dest, const runtime_shape_t &shape, NNCASE_UNUSED const runtime_shape_t &src_strides,
    NNCASE_UNUSED const runtime_shape_t &dest_strides, NNCASE_UNUSED kernel_context &context) noexcept
{
    memcpy(dest, src, compute_size(shape) * sizeof(T));
    return ok();
}

template <class T, class Callable>
result<void> _copy_impl(NNCASE_UNUSED const T *src, NNCASE_UNUSED T *dest, const runtime_shape_t &shape, NNCASE_UNUSED const runtime_shape_t &src_strides,
    NNCASE_UNUSED const runtime_shape_t &dest_strides, size_t dims_offset, Callable &&line_copy, NNCASE_UNUSED kernel_context &context) noexcept
{
    runtime_shape_t src_index(shape.size());
    const auto width = std::accumulate(shape.begin() + dims_offset, shape.end(), 1, std::multiplies<int>());
    if (dims_offset == 1)
    {
        for (size_t i = 0; i < shape[0]; ++i)
        {
            src_index[0] = i;
            line_copy(src_index, width);
        }
    }
    else if (dims_offset == 2)
    {
        for (size_t i = 0; i < shape[0]; ++i)
        {
            src_index[0] = i;
            for (size_t j = 0; j < shape[1]; ++j)
            {
                src_index[1] = j;
                line_copy(src_index, width);
            }
        }
    }
    else if (dims_offset == 3)
    {
        for (size_t i = 0; i < shape[0]; ++i)
        {
            src_index[0] = i;
            for (size_t j = 0; j < shape[1]; ++j)
            {
                src_index[1] = j;
                for (size_t k = 0; k < shape[2]; ++k)
                {
                    src_index[2] = k;
                    line_copy(src_index, width);
                }
            }
        }
    }
    return ok();
}

template <class T>
result<void> copy_dest_continuous_impl(const T *src, T *dest, const runtime_shape_t &shape, NNCASE_UNUSED const runtime_shape_t &src_strides,
    NNCASE_UNUSED const runtime_shape_t &dest_strides, size_t dims_offset, NNCASE_UNUSED kernel_context &context) noexcept
{
    auto *dest_ptr = dest;
    return _copy_impl(
        src, dest, shape, src_strides, dest_strides, dims_offset,
        // dest_ptr is pointer reference
        [&](const runtime_shape_t &src_index, auto width) {
            const auto size = width * sizeof(T);
            const auto src_ptr = src + offset(src_strides, src_index);
            memcpy(dest_ptr, src_ptr, size);
            dest_ptr += width;
        },
        context);
}

template <class T>
result<void> copy_src_continuous_impl(const T *src, T *dest, NNCASE_UNUSED const runtime_shape_t &shape, NNCASE_UNUSED const runtime_shape_t &src_strides,
    NNCASE_UNUSED const runtime_shape_t &dest_strides, size_t dims_offset, NNCASE_UNUSED kernel_context &context) noexcept
{
    auto *src_ptr = src;
    return _copy_impl(
        src, dest, shape, src_strides, dest_strides, dims_offset,
        // dest_ptr is pointer reference
        [&](const runtime_shape_t &dest_index, auto width) {
            const auto size = width * sizeof(T);
            const auto dest_ptr = dest + offset(dest_strides, dest_index);
            memcpy(dest_ptr, src_ptr, size);
            src_ptr += width;
        },
        context);
}
}

int find_last_not_continuous_index(const runtime_shape_t &strides, const runtime_shape_t &default_strides)
{
    for (int i = strides.size() - 1; i >= 0; --i)
    {
        if (strides[i] != default_strides[i])
        {
            return i + 1;
        }
    }
    return -1;
}

#define COPY_CONTINUOUS_IMPL(size, type) \
    case size:                           \
        return copy_continuous_impl(reinterpret_cast<const type *>(src), reinterpret_cast<type *>(dest), shape, src_strides, dest_strides, context)

#define COPY_DEST_CONTINUOUS_IMPL(size, type) \
    case size:                                \
        return copy_dest_continuous_impl(reinterpret_cast<const type *>(src), reinterpret_cast<type *>(dest), shape, src_strides, dest_strides, dims_offset, context)

#define COPY_SRC_CONTINUOUS_IMPL(size, type) \
    case size:                               \
        return copy_src_continuous_impl(reinterpret_cast<const type *>(src), reinterpret_cast<type *>(dest), shape, src_strides, dest_strides, dims_offset, context)

result<void> optimized::copy(datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const runtime_shape_t &shape, const runtime_shape_t &src_strides, const runtime_shape_t &dest_strides, 
    size_t dims_offset, size_t impl_select, kernel_context &context) noexcept
{
    switch (impl_select)
    {
    case 0:
        TYPE_IMPL_SELECT(type, COPY_CONTINUOUS_IMPL);
    case 1:
        TYPE_IMPL_SELECT(type, COPY_SRC_CONTINUOUS_IMPL);
    case 2:
        TYPE_IMPL_SELECT(type, COPY_DEST_CONTINUOUS_IMPL);
    }
    return ok();
}
