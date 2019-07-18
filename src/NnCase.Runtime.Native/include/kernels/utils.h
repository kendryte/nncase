#pragma once
#include <cassert>
#include <cstddef>
#include <datatypes.h>
#include <algorithm>

namespace nncase
{
namespace kernels
{
    inline size_t offset(const runtime_shape_t &shape, const runtime_shape_t &index)
    {
        return ((index[0] * shape[1] + index[1]) * shape[2] + index[2]) * shape[3] + index[3];
    }

    namespace details
    {
        inline int32_t get_windowed_output_size(int32_t size, int32_t filter, int32_t stride, int32_t dilation, const padding &padding)
        {
            auto effective_filter_size = (filter - 1) * dilation + 1;
            return (size + padding.before + padding.after - effective_filter_size + stride) / stride;
        }

        inline size_t compute_size(const runtime_shape_t &shape)
        {
            return size_t(shape[0]) * shape[1] * shape[2] * shape[3];
        }

        template <class T>
        inline T apply_activation(T value, value_range<T> activation)
        {
            return std::clamp(value, activation.min, activation.max);
        }

        inline runtime_shape_t get_reduced_offset(const runtime_shape_t &in_offset, const runtime_shape_t &reduced_shape)
        {
            runtime_shape_t off;
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
    }
}
}
