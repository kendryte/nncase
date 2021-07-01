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

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

namespace
{
template <class T>
result<void> copy_contiguous_impl(const T *src, T *dest, const runtime_shape_t &shape, NNCASE_UNUSED const runtime_shape_t &src_strides,
    NNCASE_UNUSED const runtime_shape_t &dest_strides) noexcept
{
    memcpy(dest, src, compute_size(shape) * sizeof(T));
    return ok();
}

template <size_t Dims, size_t CurDim = 0, class Callable = DefaultCallable>
void _dim_copy(NNCASE_UNUSED const runtime_shape_t &shape,
    Callable &&line_copy, runtime_shape_t &not_contiguous_index,
    std::false_type) noexcept
{
    line_copy(not_contiguous_index);
}

template <size_t Dims, size_t CurDim = 0, class Callable = DefaultCallable>
void _dim_copy(const runtime_shape_t &shape, Callable &&line_copy,
    runtime_shape_t &not_contiguous_index, std::true_type) noexcept
{
    for (size_t i = 0; i < shape[CurDim]; ++i)
    {
        not_contiguous_index[CurDim] = i;
        _dim_copy<Dims, CurDim + 1>(shape, std::forward<Callable>(line_copy), not_contiguous_index,
            is_not_equal<Dims, CurDim + 1>);
    }
}

template <class Callable>
result<void> _copy_impl(const runtime_shape_t &shape, int dims_offset, Callable &&line_copy) noexcept
{
    runtime_shape_t not_contiguous_index(shape.size());
    if (dims_offset == 1)
    {
        _dim_copy<1>(shape, line_copy, not_contiguous_index, std::true_type {});
    }
    else if (dims_offset == 2)
    {
        _dim_copy<2>(shape, line_copy, not_contiguous_index, std::true_type {});
    }
    else if (dims_offset == 3)
    {
        _dim_copy<3>(shape, line_copy, not_contiguous_index, std::true_type {});
    }
    else if (dims_offset == 4)
    {
        _dim_copy<4>(shape, line_copy, not_contiguous_index, std::true_type {});
    }
    else
    {
        assert(false);
    }
    return ok();
}

template <class T>
result<void> copy_dest_contiguous_impl(const T *src, T *dest, const runtime_shape_t &shape, const runtime_shape_t &src_strides,
    NNCASE_UNUSED const runtime_shape_t &dest_strides, int dims_offset) noexcept
{
    const auto width = std::accumulate(shape.begin() + dims_offset, shape.end(), 1, std::multiplies<size_t>());
    auto *dest_ptr = dest;
    return _copy_impl(
        shape, dims_offset,
        // src_ptr is pointer reference
        [&, width](const runtime_shape_t &src_index)
        {
            const auto size = width * sizeof(T);
            const auto src_ptr = src + offset(src_strides, src_index);
            memcpy(dest_ptr, src_ptr, size);
            dest_ptr += width;
        });
}

template <class T>
result<void> copy_src_contiguous_impl(const T *src, T *dest, const runtime_shape_t &shape, NNCASE_UNUSED const runtime_shape_t &src_strides,
    const runtime_shape_t &dest_strides, int dims_offset) noexcept
{
    const auto width = std::accumulate(shape.begin() + dims_offset, shape.end(), 1, std::multiplies<size_t>());
    auto *src_ptr = src;
    return _copy_impl(
        shape, dims_offset,
        // dest_ptr is pointer reference
        [&, width](const runtime_shape_t &dest_index)
        {
            const auto size = width * sizeof(T);
            const auto dest_ptr = dest + offset(dest_strides, dest_index);
            memcpy(dest_ptr, src_ptr, size);
            src_ptr += width;
        });
}
}

#define COPY_CONTIGUOUS_IMPL(size, type) \
    case size:                           \
        return copy_contiguous_impl(reinterpret_cast<const type *>(src), reinterpret_cast<type *>(dest), shape, src_strides, dest_strides)

#define COPY_DEST_CONTIGUOUS_IMPL(size, type) \
    case size:                                \
        return copy_dest_contiguous_impl(reinterpret_cast<const type *>(src), reinterpret_cast<type *>(dest), shape, src_strides, dest_strides, dims_offset)

#define COPY_SRC_CONTIGUOUS_IMPL(size, type) \
    case size:                               \
        return copy_src_contiguous_impl(reinterpret_cast<const type *>(src), reinterpret_cast<type *>(dest), shape, src_strides, dest_strides, dims_offset)

result<void> optimized::copy(datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const runtime_shape_t &shape, const runtime_shape_t &src_strides, const runtime_shape_t &dest_strides,
    int dims_offset, copy_impl_select impl_select, NNCASE_UNUSED kernel_context &context) noexcept
{
    switch (impl_select)
    {
    case copy_impl_select::all_contiguous:
        TYPE_IMPL_SELECT(type, COPY_CONTIGUOUS_IMPL);
    case copy_impl_select::src_contiguous:
        TYPE_IMPL_SELECT(type, COPY_SRC_CONTIGUOUS_IMPL);
    case copy_impl_select::dest_contiguous:
        TYPE_IMPL_SELECT(type, COPY_DEST_CONTIGUOUS_IMPL);
    }
    return ok();
}
