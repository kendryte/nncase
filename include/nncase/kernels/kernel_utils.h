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

#ifdef __GNUC__
#define CXX_RESTRICT __restrict__
#elif _MSC_VER
#define CXX_RESTRICT __restrict
#else
#define CXX_RESTRICT
#endif

BEGIN_NS_NNCASE_KERNELS

template <class TShape>
size_t offset(const TShape &shape, const TShape &index)
{
    assert(shape.size() == index.size());
    size_t result = index[0];
    for (size_t i = 1; i < index.size(); i++)
        result = result * shape[i] + index[i];
    return result;
}

namespace details
{
inline size_t get_windowed_output_size(size_t size, int32_t filter, int32_t stride, int32_t dilation, const padding &padding)
{
    auto effective_filter_size = (filter - 1) * dilation + 1;
    return (size_t)((int32_t)size + padding.before + padding.after - effective_filter_size + stride) / stride;
}

template <class TShape>
size_t compute_size(const TShape &shape)
{
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::plus<void>());
}

template <class T>
inline T apply_activation(T value, value_range<T> activation)
{
    return std::clamp(value, activation.min, activation.max);
}

template <class TShape>
inline TShape get_reduced_offset(const TShape &in_offset, const TShape &reduced_shape)
{
    TShape off;
    for (size_t i = 0; i < in_offset.size(); i++)
    {
        if (in_offset[i] >= reduced_shape[i])
            off[i] = 0;
        else
            off[i] = in_offset[i];
    }

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
