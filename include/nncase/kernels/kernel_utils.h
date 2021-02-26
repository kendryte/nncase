/* Copyright 2020 Canaan Inc.
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
#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <nncase/runtime/datatypes.h>
#include <numeric>
#include <xtensor/xstrides.hpp>

#ifdef __GNUC__
#define CXX_RESTRICT __restrict__
#elif _MSC_VER
#define CXX_RESTRICT __restrict
#else
#define CXX_RESTRICT
#endif

BEGIN_NS_NNCASE_KERNELS

template <class TShape>
size_t offset(const TShape &strides, const TShape &index)
{
    assert(strides.size() == index.size());
    return xt::element_offset<size_t>(strides, index.begin(), index.end());
}

template <class TShape>
TShape reshape_linear_index(const TShape &new_shape, size_t index)
{
    TShape new_index(new_shape.size());
    size_t i = new_shape.size() - 1;
    for (auto it = new_shape.rbegin(); it != new_shape.rend(); ++it)
    {
        new_index[i--] = index % *it;
        index /= *it;
    }

    return new_index;
}

template <class TShape>
size_t linear_index(const TShape &shape, const TShape &index)
{
    assert(index.size() == shape.size());
    size_t new_index = index[0];
    for (size_t i = 1; i < shape.size(); i++)
        new_index = new_index * shape[i] + index[i];
    return new_index;
}

namespace detail
{
inline size_t get_windowed_output_size(size_t size, int32_t filter, int32_t stride, int32_t dilation, const padding &padding)
{
    auto effective_filter_size = (filter - 1) * dilation + 1;
    return (size_t)((int32_t)size + padding.before + padding.after - effective_filter_size + stride) / stride;
}

inline runtime_shape_t get_binary_output_shape(const runtime_shape_t &input_a_shape, const runtime_shape_t &input_b_shape)
{
    runtime_shape_t out_shape;

    const auto dest_dims = (int32_t)std::max(input_a_shape.size(), input_b_shape.size());
    const auto in_a_ext = dest_dims - (int32_t)input_a_shape.size();
    const auto in_b_ext = dest_dims - (int32_t)input_b_shape.size();

    for (int32_t i = 0; i < dest_dims; i++)
    {
        const auto in_a_dim = i - (int32_t)in_a_ext;
        const auto in_b_dim = i - (int32_t)in_b_ext;

        const auto in_a = in_a_dim < 0 ? 1 : input_a_shape[in_a_dim];
        const auto in_b = in_b_dim < 0 ? 1 : input_b_shape[in_b_dim];
        if (in_a == in_b)
            out_shape.push_back(in_a);
        else if (in_a == 1)
            out_shape.push_back(in_b);
        else if (in_b == 1)
            out_shape.push_back(in_a);
        else
            assert(!"inputs are not compatible to broadcast");
    }

    return out_shape;
}

template <class TShape>
size_t compute_size(const TShape &shape)
{
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<void>());
}

template <class T>
inline T clamp(T value, T min, T max)
{
    return std::max(std::min(value, max), min);
}

template <class T>
inline T apply_activation(T value, value_range<T> activation)
{
    return clamp(value, activation.min, activation.max);
}

template <class TShape>
TShape get_reduced_offset(const TShape &in_offset, const TShape &reduced_shape)
{
    TShape off(reduced_shape.size());
    const auto dims_ext = in_offset.size() - reduced_shape.size();
    for (size_t i = 0; i < reduced_shape.size(); i++)
    {
        if (in_offset[i + dims_ext] >= reduced_shape[i])
            off[i] = 0;
        else
            off[i] = in_offset[i + dims_ext];
    }

    return off;
}

template <class TShape>
TShape get_reduced_shape(const TShape &in_shape, const TShape &axis, bool keep_dims)
{
    TShape shape;
    shape.reserve(in_shape.size() - keep_dims ? 0 : axis.size());
    for (size_t i = 0; i < in_shape.size(); i++)
    {
        if (std::find(axis.begin(), axis.end(), i) == axis.end())
        {
            shape.push_back(in_shape[i]);
        }
        else
        {
            if (keep_dims)
                shape.push_back(1);
        }
    }

    if (shape.empty())
        shape.push_back(1);
    return shape;
}

template <class TShape>
size_t get_reduce_block_size(const TShape &in_shape, const TShape &axis)
{
    size_t size = 1;
    for (size_t i = 0; i < in_shape.size(); i++)
    {
        if (std::find(axis.begin(), axis.end(), i) != axis.end())
        {
            size *= in_shape[i];
        }
    }

    return size;
}

template <class TShape>
TShape get_reduced_offset(const TShape &in_offset, const TShape &axis, bool keep_dims)
{
    TShape off;
    off.reserve(in_offset.size() - keep_dims ? 0 : axis.size());
    for (size_t i = 0; i < in_offset.size(); i++)
    {
        if (std::find(axis.begin(), axis.end(), i) == axis.end())
        {
            off.push_back(in_offset[i]);
        }
        else
        {
            if (keep_dims)
                off.push_back(0);
        }
    }

    if (off.empty())
        off.push_back(0);
    return off;
}

template <class T, class TRange>
struct default_ptr_getter
{
    T *operator()(const TRange &range) const noexcept { return range; }
};

template <int32_t Bits>
int32_t to_signed(uint32_t value)
{
    auto mask = uint32_t(1) << (Bits - 1);
    if (Bits != 32 && (value & mask) != 0)
    {
        auto sign = 0xFFFFFFFF << Bits;
        return (int)(value | sign);
    }

    return (int32_t)value;
}

template <int32_t Bits>
int64_t to_signed(uint64_t value)
{
    auto mask = uint64_t(1) << (Bits - 1);
    if ((value & mask) != 0)
    {
        auto sign = 0xFFFFFFFFFFFFFFFF << Bits;
        return (int64_t)(value | sign);
    }

    return (int64_t)value;
}
}
END_NS_NNCASE_KERNELS
