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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class T>
struct identity
{
    T operator()(const T &src) const noexcept
    {
        return src;
    }
};

template <class T, class TReducer, class TPostProcess>
result<void> reduce_impl(TReducer &&reducer, TPostProcess &&post_process, T init_value, const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides, bool keep_dims, NNCASE_UNUSED kernel_context &context) noexcept
{
    T *tmp_out;
    if ((intptr_t)input == (intptr_t)output)
    {
        int out_len = compute_size(out_shape);
        tmp_out = (T *)malloc(out_len * sizeof(T));
    }
    else
    {
        tmp_out = output;
    }
    try_(apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        tmp_out[offset(out_strides, index)] = init_value;
        return ok();
    }));
    try_(apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto v = input[offset(in_strides, index)];
        const auto out_index = kernels::detail::get_reduced_offset(index, axis, keep_dims);
        auto &dest = tmp_out[offset(out_strides, out_index)];
        dest = reducer(dest, v);
        return ok();
    }));
    try_(apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        if ((intptr_t)input == (intptr_t)output)
        {
            auto &src = tmp_out[offset(out_strides, index)];
            auto &dest = output[offset(out_strides, index)];
            dest = post_process(src);
        }
        else
        {
            auto &dest = tmp_out[offset(out_strides, index)];
            dest = post_process(dest);
        }
        return ok();
    }));
    return ok();
}
}

#define REDUCE_IMPL(op, reducer, post_process) \
    case op:                                   \
        return reduce_impl(reducer, post_process, init_value, input, output, in_shape, axis, in_strides, out_shape, out_strides, keep_dims, context)

#define REDUCE_IMPL_NO_POST(op, reducer) \
    case op:                             \
        return reduce_impl(reducer, identity<T>(), init_value, input, output, in_shape, axis, in_strides, out_shape, out_strides, keep_dims, context)

template result<void> reference::reduce<float>(reduce_op_t op, float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept;

template result<void> reference::reduce<int32_t>(reduce_op_t op, int32_t init_value, const int32_t *input, int32_t *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept;

template result<void> reference::reduce<int64_t>(reduce_op_t op, int64_t init_value, const int64_t *input, int64_t *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept;

template <typename T>
result<void> reference::reduce(reduce_op_t op, T init_value, const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims, kernel_context &context) noexcept
{
    auto out_shape = kernels::detail::get_reduced_shape(in_shape, axis, keep_dims);
    switch (op)
    {
        REDUCE_IMPL(reduce_mean, std::plus<T>(), [block_size = (T)kernels::detail::get_reduce_block_size(in_shape, axis)](T v) { return v / block_size; });
        REDUCE_IMPL_NO_POST(reduce_sum, std::plus<T>());
    case reduce_min:
        init_value = std::numeric_limits<T>::max();
        return reduce_impl([](T a, T b) { return std::min(a, b); }, identity<T>(), init_value, input, output, in_shape, axis, in_strides,
            out_shape, out_strides, keep_dims, context);
        break;
    case reduce_max:
        init_value = std::numeric_limits<T>::lowest();
        return reduce_impl([](T a, T b) { return std::max(a, b); }, identity<T>(), init_value, input, output, in_shape, axis, in_strides,
            out_shape, out_strides, keep_dims, context);
        break;
    default:
        return err(std::errc::not_supported);
    }
}
