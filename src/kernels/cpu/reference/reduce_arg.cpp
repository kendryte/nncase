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
#include <limits>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <unordered_map>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class TReducer, class TOutput>
result<void> reduce_arg_impl(TReducer &&reducer, float init_value,
    const float *input, TOutput *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &out_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axes, bool keep_dims, bool select_last_idx, NNCASE_UNUSED kernel_context &context) noexcept
{
    const float epsilon = 0.000001f;

    // init with init_value
    std::unique_ptr<float[]> ptr(new float[compute_size(out_shape)]);
    try_(apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        ptr[offset(out_strides, index)] = init_value;
        return ok();
    }));

    // collect all min/max indices
    std::unordered_map<size_t, std::vector<TOutput>> out_map;
    try_(apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto src = input[offset(in_strides, index)];
        auto out_idx = offset(out_strides, kernels::detail::get_reduced_offset(index, axes, keep_dims));
        auto &dst = ptr[out_idx];
        auto ret = reducer(src, dst);
        if (ret)
        {
            out_map[out_idx].clear();
            out_map[out_idx].push_back(index[axes[0]]);
            dst = src;
        }
        else if (fabs(src - dst) < epsilon)
        {
            out_map[out_idx].push_back(index[axes[0]]);
        }
        return ok();
    }));

    // update min/max idx
    try_(apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        auto out_idx = offset(out_strides, index);
        output[out_idx] = select_last_idx ? out_map[out_idx].back() : out_map[out_idx].front();
        return ok();
    }));
    return ok();
}
}

template NNCASE_API result<void> reference::reduce_arg<int32_t>(reduce_arg_op_t op, const float *input, int32_t *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axis, bool keep_dims, bool select_last_idx, kernel_context &context) noexcept;

template NNCASE_API result<void> reference::reduce_arg<int64_t>(reduce_arg_op_t op, const float *input, int64_t *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axis, bool keep_dims, bool select_last_idx, kernel_context &context) noexcept;

template <typename T>
result<void> reference::reduce_arg(reduce_arg_op_t op, const float *input, T *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axes, bool keep_dims, bool select_last_idx, kernel_context &context) noexcept
{
    auto out_shape = kernels::detail::get_reduced_shape(in_shape, axes, keep_dims);
    switch (op)
    {
    case reduce_arg_min:
        return reduce_arg_impl([](float a, float b) { return a < b; }, std::numeric_limits<float>::max(),
            input, output, in_shape, out_shape, in_strides, out_strides, axes, keep_dims, select_last_idx, context);
    case reduce_arg_max:
        return reduce_arg_impl([](float a, float b) { return a > b; }, std::numeric_limits<float>::lowest(),
            input, output, in_shape, out_shape, in_strides, out_strides, axes, keep_dims, select_last_idx, context);
    default:
        return err(std::errc::not_supported);
    }
}
