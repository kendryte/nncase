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
#pragma once
#include "node.h"
#include <span>
#include <xtensor/xtensor.hpp>

namespace nncase::ir
{
inline shape_t get_transposed_shape(const shape_t &input_shape, const axis_t &perm)
{
    shape_t new_shape(input_shape.size());
    for (size_t i = 0; i < new_shape.size(); i++)
        new_shape[i] = input_shape[perm[i]];
    return new_shape;
}

inline size_t get_windowed_output_size(int32_t size, int32_t filter, int32_t stride, int32_t dilation, bool same, bool ceil_mode = false)
{
    auto effective_filter_size = (filter - 1) * dilation + 1;
    if (same)
        return (size_t(size) + stride - 1) / stride;
    else
    {
        if (!ceil_mode)
            return (size_t(size) - effective_filter_size + stride) / stride;
        else
        {
            return static_cast<int>(ceil(static_cast<float>(size_t(size) - effective_filter_size + stride) / stride));
        }
    }
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

inline constexpr size_t get_bytes(datatype_t type)
{
    switch (type)
    {
#define DEFINE_DATATYPE(id, t, name, value) \
    case (dt_##id):                         \
        return sizeof(t);
#include <nncase/runtime/datatypes.def>
#undef DEFINE_DATATYPE
    default:
        throw std::invalid_argument("Invalid datatype");
    }
}

inline size_t get_bytes(datatype_t type, const shape_t &shape)
{
    return xt::compute_size(shape) * get_bytes(type);
}

template <class shape_type, class strides_type>
inline void compute_strides(const shape_type &shape, strides_type &strides)
{
    using strides_value_type = typename std::decay_t<strides_type>::value_type;
    strides_value_type data_size = 1;
    for (std::size_t i = shape.size(); i != 0; --i)
    {
        strides[i - 1] = data_size;
        data_size = strides[i - 1] * static_cast<strides_value_type>(shape[i - 1]);
    }
}

inline nncase::ir::shape_t to_strides(const nncase::ir::shape_t &shape)
{
    nncase::ir::shape_t strides(shape.size());
    compute_strides(shape, strides);
    return strides;
}

inline int32_t normalize_axis(const shape_t &input_shape, int32_t axis)
{
    return axis < 0 ? (int32_t)input_shape.size() + axis : axis;
}

inline axis_t normalize_axis(const shape_t &input_shape, const axis_t &axis)
{
    axis_t new_axis = axis;
    for (auto &a : new_axis)
    {
        if (a < 0)
            a = (int32_t)input_shape.size() + a;
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

    if (shape.empty())
        shape.push_back(1);

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
                throw std::runtime_error("Reshape can only have 1 non-determined dimension at most");
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
            throw std::invalid_argument("inputs are not compatible to broadcast");
    }

    return out_shape;
}

inline std::vector<shape_t> get_input_shapes(std::span<input_connector *const> inputs)
{
    std::vector<shape_t> shapes;
    shapes.reserve(inputs.size());
    for (auto in : inputs)
        shapes.emplace_back(in->shape());
    return shapes;
}

inline shape_t get_concated_shape(std::span<shape_t> input_shapes, size_t axis)
{
    if (input_shapes.empty())
        throw std::invalid_argument("there must be at least one input");

    auto concated_shape = input_shapes[0];

    for (size_t i = 1; i < input_shapes.size(); i++)
    {
        auto &cur_shape = input_shapes[i];
        if (concated_shape.size() != cur_shape.size())
            throw std::invalid_argument("inputs must have same ranks");

        for (size_t j = 0; j < concated_shape.size(); j++)
        {
            if (j == axis)
            {
                concated_shape[j] += cur_shape[j];
            }
            else if (cur_shape[j] != concated_shape[j])
            {
                throw std::invalid_argument("inputs are not compatible to concat");
            }
        }
    }

    return concated_shape;
}

inline void get_concat_params(const shape_t &out_shape, size_t elem_size, size_t axis, uint64_t &inner_size, uint64_t &outer_size)
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
        new_shape[i] = size_t(int32_t(new_shape[i]) + paddings[i].sum() + (new_shape[i] - 1) * paddings[i].interior);
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
            ? stride > 0 ? 0 : (int32_t)in_shape[i]
            : (begin[i] >= 0 ? begin[i] : (int32_t)in_shape[i] + begin[i]);
    }

    return new_shape;
}

inline axis_t normalize_strided_slice_end(const shape_t &in_shape, [[maybe_unused]] const axis_t &begin, const axis_t &end, const axis_t &strides, int32_t end_mask, int32_t shrink_axis_mask)
{
    axis_t new_shape(strides.size());
    for (size_t i = 0; i < new_shape.size(); i++)
    {
        auto stride = strides[i];
        int32_t end_val = (end_mask & (1 << i)) != 0
            ? stride > 0 ? (int32_t)in_shape[i] : -1
            : (shrink_axis_mask & (1 << i)) == 0 ? (end[i] >= 0 ? end[i] : in_shape[i] + end[i] + 1)
                                                 : begin[i] + 1;
        new_shape[i] = (int32_t)end_val;
    }

    return new_shape;
}

inline shape_t get_strided_slice_output_shape(const axis_t &begin, const axis_t &end, const axis_t &strides, int32_t ellipsis_mask, int32_t new_axis_mask, int32_t shrink_axis_mask)
{
    if (ellipsis_mask)
        throw std::invalid_argument("Non-zero ellipsis_mask is not supported");
    if (new_axis_mask)
        throw std::invalid_argument("Non-zero new_axis_mask is not supported");

    shape_t new_shape;
    for (size_t i = 0; i < strides.size(); i++)
    {
        auto stride = strides[i];
        auto begin_val = begin[i];
        auto end_val = end[i];
        auto dim = (shrink_axis_mask & (1 << i)) == 0
            ? (int)std::ceil(((float)abs(end_val - begin_val) / (float)abs(stride)))
            : 1;
        new_shape.push_back(dim);
    }

    return new_shape.size() ? new_shape : shape_t { 1 };
}

inline shape_t get_matmul_output_shape(const shape_t &input_a_shape, const shape_t &input_b_shape)
{
    shape_t b_shape = input_b_shape;
    b_shape[b_shape.size() - 2] = input_a_shape[input_a_shape.size() - 2];
    b_shape[b_shape.size() - 1] = input_a_shape[input_a_shape.size() - 1];
    shape_t out_shape;

    const auto dest_dims = (int32_t)std::max(input_a_shape.size(), b_shape.size());
    const auto in_a_ext = dest_dims - (int32_t)input_a_shape.size();
    const auto in_b_ext = dest_dims - (int32_t)b_shape.size();

    for (int32_t i = 0; i < dest_dims; i++)
    {
        const auto in_a_dim = i - (int32_t)in_a_ext;
        const auto in_b_dim = i - (int32_t)in_b_ext;

        const auto in_a = in_a_dim < 0 ? 1 : input_a_shape[in_a_dim];
        const auto in_b = in_b_dim < 0 ? 1 : b_shape[in_b_dim];
        if (in_a == in_b)
            out_shape.push_back(in_a);
        else if (in_a == 1)
            out_shape.push_back(in_b);
        else if (in_b == 1)
            out_shape.push_back(in_a);
        else
            throw std::invalid_argument("inputs are not compatible to broadcast");
    }

    out_shape[out_shape.size() - 2] = input_a_shape[input_a_shape.size() - 2];
    out_shape[out_shape.size() - 1] = input_b_shape.back();

    return out_shape;
}

inline bool is_copy_slice(const axis_t &strides)
{
    return std::all_of(strides.begin(), strides.end(), [](int32_t stride) { return stride == 1; });
}

inline bool is_simple_slice(const axis_t &begin, const axis_t &end, const axis_t &strides, const shape_t &input_shape)
{
    if (!is_copy_slice(strides))
        return false;

    bool is_simple_slice = true;
    bool allow_not_equal = true;
    for (size_t i = 0; i < begin.size(); i++)
    {
        if (begin[i] != 0
            || end[i] != input_shape[i])
        {
            if (allow_not_equal)
            {
                allow_not_equal = false;
            }
            else
            {
                is_simple_slice = false;
                break;
            }
        }
        else if (input_shape[i] != 1)
        {
            allow_not_equal = false;
        }
    }

    return is_simple_slice;
}

inline shape_t get_instancenorm_const_shape(const shape_t &in_shape)
{
    shape_t const_shape(in_shape.size() - 1, 1);
    const_shape[0] = in_shape[1];
    return const_shape;
}

inline bool is_axis0_squeeze_or_expand_dim_bitcast(const shape_t &in_shape, const shape_t &out_shape)
{
    auto in_begin = std::find_if_not(in_shape.begin(), in_shape.end(), [](size_t dim) { return dim == 1; });
    auto out_begin = std::find_if_not(out_shape.begin(), out_shape.end(), [](size_t dim) { return dim == 1; });
    return std::distance(in_begin, in_shape.end()) == std::distance(out_begin, out_shape.end())
        && std::equal(in_begin, in_shape.end(), out_begin);
}

template <class U, class T>
std::span<U> as_span(const std::span<T> &src) noexcept
{
    assert(src.size_bytes() % sizeof(U) == 0);
    return std::span<U>(reinterpret_cast<U *>(src.data()), src.size_bytes() / sizeof(U));
}
}

namespace xt
{
inline nncase::ir::shape_t operator+(const nncase::ir::shape_t &lhs, const nncase::ir::shape_t &rhs)
{
    using namespace nncase::ir;

    if (lhs.size() != rhs.size())
        throw std::invalid_argument("Shape's rank mismatch");
    shape_t ret = lhs;
    for (size_t i = 0; i < lhs.size(); i++)
        ret[i] += rhs[i];
    return ret;
}

inline nncase::ir::shape_t &operator+=(nncase::ir::shape_t &lhs, const nncase::ir::shape_t &rhs)
{
    using namespace nncase::ir;

    if (lhs.size() != rhs.size())
        throw std::invalid_argument("Shape's rank mismatch");
    for (size_t i = 0; i < lhs.size(); i++)
        lhs[i] += rhs[i];
    return lhs;
}
}
