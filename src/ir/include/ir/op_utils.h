/* Copyright 2019 Canaan Inc.
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
#include "node.h"
#include <runtime/runtime_op_utility.h>
#include <xtensor/xtensor.hpp>

namespace nncase
{
namespace ir
{
    inline shape_t get_transposed_shape(const shape_t &input_shape, const axis_t &perm)
    {
        shape_t new_shape(input_shape.size());
        for (size_t i = 0; i < new_shape.size(); i++)
            new_shape[i] = input_shape[perm[i]];
        return new_shape;
    }

    inline size_t get_windowed_output_size(int32_t size, int32_t filter, int32_t stride, int32_t dilation, bool same)
    {
        auto effective_filter_size = (filter - 1) * dilation + 1;
        if (same)
            return (size_t(size) + stride - 1) / stride;
        else
            return (size_t(size) - effective_filter_size + stride) / stride;
    }

    inline padding get_windowed_padding(int32_t input_size, int32_t output_size, int32_t filter, int32_t stride, int32_t dilation)
    {
        auto effective_filter_size = (filter - 1) * dilation + 1;
        int padding = std::max(0, (output_size - 1) * stride + effective_filter_size - input_size);
        return { padding / 2, padding - padding / 2 };
    }

    inline padding get_windowed_padding(int32_t input_size, int32_t filter, int32_t stride, int32_t dilation, bool same)
    {
        auto output_size = get_windowed_output_size(input_size, filter, stride, dilation, same);
        return get_windowed_padding(input_size, (int32_t)output_size, filter, stride, dilation);
    }

    inline size_t get_bytes(datatype_t type, const shape_t &shape)
    {
        return xt::compute_size(shape) * runtime::get_bytes(type);
    }

    inline axis_t normalize_axis(const shape_t &input_shape, const axis_t &axis)
    {
        axis_t new_axis = axis;
        for (auto &a : new_axis)
        {
            if (a < 0)
                a = input_shape.size() + a;
        }

        return new_axis;
    }

    inline axis_t normalize_reduce_axis(const shape_t &input_shape, const axis_t &axis)
    {
        axis_t new_axis = normalize_axis(input_shape, axis);
        std::sort(new_axis.begin(), new_axis.end());
        return new_axis;
    }

    inline shape_t get_reduced_shape(const shape_t &input_shape, const axis_t &axis, bool keep_dims)
    {
        if (!std::is_sorted(axis.begin(), axis.end()))
            throw std::invalid_argument("axis must be sorted");

        shape_t shape;
        for (size_t i = 0; i < input_shape.size(); i++)
        {
            if (std::find(axis.begin(), axis.end(), i) == axis.end())
                shape.push_back(input_shape[i]);
            else if (keep_dims)
                shape.push_back(1);
        }

        return shape;
    }

    inline shape_t normalize_reshape(const shape_t &in_shape, const axis_t &new_shape)
    {
        shape_t result(new_shape.size());

        size_t shape_size = 1;
        std::optional<size_t> non_det_id;
        for (size_t i = 0; i < new_shape.size(); i++)
        {
            auto v = new_shape[i];
            if (v == -1)
            {
                if (non_det_id)
                    throw std::runtime_error("Reshap can only have 1 non-determined dimension at most");
                non_det_id = i;
            }
            else
            {
                shape_size *= v;
                result[i] = (size_t)new_shape[i];
            }
        }

        if (non_det_id)
            result[*non_det_id] = xt::compute_size(in_shape) / shape_size;
        return result;
    }

    inline shape_t get_binary_output_shape(const shape_t &input_a_shape, const shape_t &input_b_shape)
    {
        shape_t out_shape;

        const auto dest_dims = std::max(input_a_shape.size(), input_b_shape.size());
        const auto in_a_ext = dest_dims - input_a_shape.size();
        const auto in_b_ext = dest_dims - input_b_shape.size();

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
                throw std::invalid_argument("inputs are not compatible to broadcast");
        }

        return out_shape;
    }

    inline shape_t get_concated_shape(xtl::span<shape_t> input_shapes, int32_t axis)
    {
        if (input_shapes.empty())
            throw std::invalid_argument("there must be at least one input");

        auto shape = input_shapes[0];

        for (size_t i = 1; i < input_shapes.size(); i++)
        {
            auto &in = input_shapes[i];
            if (in.size() != shape.size())
                throw std::invalid_argument("inputs must have same ranks");

            for (size_t j = 0; j < shape.size(); j++)
            {
                if (j == axis)
                {
                    shape[j] += in[j];
                }
                else if (in[j] != shape[j])
                {
                    throw std::invalid_argument("inputs are not compatible to concat");
                }
            }
        }

        return shape;
    }

    template <class T>
    runtime_shape_t to(const xt::dynamic_shape<T> &in_shape)
    {
        assert(in_shape.size() <= 4);

        runtime_shape_t r_in_shape;
        const auto in_ext = 4 - in_shape.size();

        for (int i = 0; i < in_ext; i++)
            r_in_shape[i] = 1;
        for (size_t i = in_ext; i < 4; i++)
            r_in_shape[i] = int32_t(in_shape[i - in_ext]);
        return r_in_shape;
    }

    inline runtime_paddings_t to(const xt::svector<padding> &paddings)
    {
        assert(paddings.size() <= 4);

        runtime_paddings_t r_paddings;
        const auto in_ext = 4 - paddings.size();

        for (int i = 0; i < in_ext; i++)
            r_paddings[i] = padding::zero();
        for (size_t i = in_ext; i < 4; i++)
            r_paddings[i] = paddings[i - in_ext];
        return r_paddings;
    }

    inline void extend_transpose_shape(const shape_t &in_shape, const axis_t &perm, runtime_shape_t &r_in_shape, runtime_shape_t &r_perm)
    {
        assert(perm.size() <= 4);

        const auto in_ext = 4 - in_shape.size();
        const auto perm_ext = 4 - perm.size();
        r_in_shape = to(in_shape);

        for (size_t i = 0; i < perm_ext; i++)
            r_perm[i] = int32_t(i);
        for (size_t i = 0; i < perm.size(); i++)
            r_perm[i + perm_ext] = int32_t(perm[i] + in_ext);
    }

    inline void get_concat_params(const shape_t &out_shape, size_t elem_size, int32_t axis, uint64_t &inner_size, uint64_t &outer_size)
    {
        inner_size = elem_size;
        outer_size = 1;

        for (size_t i = 0; i < out_shape.size(); i++)
        {
            if (i > axis)
                inner_size *= out_shape[i];
            else if (i < axis)
                outer_size *= out_shape[i];
        }
    }

    inline shape_t get_padded_shape(const shape_t &in_shape, const xt::svector<padding> &paddings)
    {
        auto new_shape = in_shape;
        for (size_t i = 0; i < in_shape.size(); i++)
            new_shape[i] = size_t(int32_t(new_shape[i]) + paddings[i].sum());
        return new_shape;
    }

    inline shape_t get_resize_image_shape(const shape_t &in_shape, const std::array<int32_t, 2> &new_size)
    {
        auto new_shape = in_shape;
        auto r = new_shape.rbegin();
        *r++ = new_size[1];
        *r++ = new_size[0];
        return new_shape;
    }

    inline axis_t normalize_strided_slice_begin(const shape_t &in_shape, const axis_t &begin, const axis_t &strides, int32_t begin_mask)
    {
        axis_t new_shape(strides.size());
        for (size_t i = 0; i < new_shape.size(); i++)
        {
            auto stride = strides[i];
            assert(stride);
            new_shape[i] = (begin_mask & (1 << i)) != 0
                ? stride > 0 ? 0 : in_shape[i] - 1
                : begin[i];
        }

        return new_shape;
    }

    inline axis_t normalize_strided_slice_end(const shape_t &in_shape, const axis_t &begin, const axis_t &end, const axis_t &strides, int32_t end_mask, int32_t shrink_axis_mask)
    {
        axis_t new_shape(strides.size());
        for (size_t i = 0; i < new_shape.size(); i++)
        {
            auto stride = strides[i];
            auto begin_val = begin[i];
            auto end_val = (end_mask & (1 << i)) != 0
                ? stride > 0 ? in_shape[i] : -1
                : end[i];
            auto shrink = shrink_axis_mask & (1 << i);
            if (shrink)
                end_val = begin_val + 1;
            new_shape[i] = end_val;
        }

        return new_shape;
    }

    inline shape_t get_strided_slice_output_shape(const axis_t &begin, const axis_t &end, const axis_t &strides, int32_t ellipsis_mask, int32_t new_axis_mask, int32_t shrink_axis_mask)
    {
        if (ellipsis_mask)
            NNCASE_THROW(std::invalid_argument, "Non-zero ellipsis_mask is not supported");
        if (new_axis_mask)
            NNCASE_THROW(std::invalid_argument, "Non-zero new_axis_mask is not supported");

        shape_t new_shape;
        for (size_t i = 0; i < strides.size(); i++)
        {
            auto stride = strides[i];
            auto begin_val = begin[i];
            auto end_val = end[i];
            auto shrink = (shrink_axis_mask & (1 << i)) != 0;
            auto dim = (int)std::ceil((end_val - begin_val) / (float)stride);
            if (!shrink)
                new_shape.push_back(dim);
        }

        return new_shape;
    }
}
}
