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
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/util.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
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

template <class TReducer, class TPostProcess>
result<void> reduce_impl(TReducer &&reducer, TPostProcess &&post_process, float init_value, const float *input, float *output, const dims_t &in_shape, const dims_t &axis,
    const strides_t &in_strides, const dims_t &out_shape, const strides_t &out_strides, bool keep_dims, NNCASE_UNUSED kernel_context &context) noexcept
{
    try_(apply(out_shape, [&](const dims_t &index) -> result<void> {
        output[offset(out_strides, index)] = init_value;
        return ok();
    }));
    try_(apply(in_shape, [&](const dims_t &index) -> result<void> {
        const auto v = input[offset(in_strides, index)];
        const auto out_index = kernels::detail::get_reduced_offset(index, axis, keep_dims);
        auto &dest = output[offset(out_strides, out_index)];
        dest = reducer(dest, v);
        return ok();
    }));
    try_(apply(out_shape, [&](const dims_t &index) -> result<void> {
        auto &dest = output[offset(out_strides, index)];
        dest = post_process(dest);
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
        return reduce_impl(reducer, identity<float>(), init_value, input, output, in_shape, axis, in_strides, out_shape, out_strides, keep_dims, context)

template <typename T>
result<void> reduce_prod(const T *input, T *output, const dims_t &in_shape,
                         const strides_t &in_strides, const strides_t &out_strides,
                         const dims_t &axes, bool keep_dims) noexcept
{
    auto out_shape = kernels::detail::get_reduced_shape(in_shape, axes, keep_dims);

    // init with init_value
    try_(reference::apply(out_shape, [&](const dims_t &index) -> result<void> {
        output[offset(out_strides, index)] = 1;
        return ok();
    }));

    try_(apply(in_shape, [&](const dims_t &index) -> result<void> {
        const auto src = input[offset(in_strides, index)];
        auto out_idx = offset(out_strides, kernels::detail::get_reduced_offset(index, axes, keep_dims));
        auto &dst = output[out_idx];
        dst *= src;
        return ok();
    }));

    return ok();
}

template NNCASE_API result<void> reduce_prod<float>(const float *input, float *output, const dims_t &in_shape,
                                                    const strides_t &in_strides, const strides_t &out_strides,
                                                    const dims_t &axis, bool keep_dims) noexcept;

result<void> reduce_impl(reduce_op_t op, float init_value, const float *input, float *output, const dims_t &in_shape, const dims_t &axis,
    const strides_t &in_strides, const strides_t &out_strides, bool keep_dims, kernel_context &context) noexcept
{
    auto out_shape = kernels::detail::get_reduced_shape(in_shape, axis, keep_dims);
    switch (op)
    {
        REDUCE_IMPL(reduce_op_t::mean, std::plus<float>(), [block_size = (float)kernels::detail::get_reduce_block_size(in_shape, axis)](float v) { return v / block_size; });
        REDUCE_IMPL_NO_POST(reduce_op_t::min, [](float a, float b) { return std::min(a, b); });
        REDUCE_IMPL_NO_POST(reduce_op_t::max, [](float a, float b) { return std::max(a, b); });
        REDUCE_IMPL_NO_POST(reduce_op_t::sum, std::plus<float>());
    case reduce_op_t::prod:
        return reduce_prod(input, output, in_shape, in_strides, out_strides, axis, keep_dims);
    default:
        return err(std::errc::not_supported);
    }
}

dims_t infer_shape(const dims_t& in_shape, const dims_t& axes, bool keep_dims) {
    auto tmp_shape = in_shape;
    for (int i = 0; i < axes.size(); ++i) {
        auto d = keep_dims ? 1 : 0;
        tmp_shape[axes[positive_index(i, in_shape.size())]] = d;
    }
    auto new_shape = dims_t();
    for (auto d : tmp_shape) {
        if(d > 0) {
            new_shape.push_back(d);
        }
    }
    return new_shape;
}

result<value_t> nncase::kernels::stackvm::reduce(
    reduce_op_t reduce_op, value_t input, value_t axis, value_t init_value,
    value_t keep_dims, value_t output, kernel_context &context) {
    try_f32_input(in_mem, input);
    try_axis(axis_value, axis, input_tensor->shape().size());
    try_to_scalar(keep_dims_value, keep_dims, bool);
    try_to_scalar(init_v, init_value, float);
    auto out_shape = infer_shape(input_tensor->shape(), axis_value, keep_dims_value);
    try_f32_output(out_mem, output, input_tensor->dtype(), out_shape);

    try_(reduce_impl(reduce_op, init_v, in_mem, out_mem,
                     input_tensor->shape(), axis_value,
                     input_tensor->strides(), output_tensor->strides(),
                     keep_dims_value, context));
    return ok(output);
}
