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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <xtensor/xadapt.hpp>

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

template <class TOp>
result<void> binary_impl(TOp &&op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides, value_range<float> fused_activation) noexcept
{
    const auto out_shape = kernels::detail::get_binary_output_shape(in_a_shape, in_b_shape);
    return apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto in_a_index = kernels::detail::get_reduced_offset(index, in_a_shape);
        const auto in_b_index = kernels::detail::get_reduced_offset(index, in_b_shape);
        const auto a = input_a[offset(in_a_strides, in_a_index)];
        const auto b = input_b[offset(in_b_strides, in_b_index)];
        output[offset(out_strides, index)] = kernels::detail::apply_activation(op(a, b), fused_activation);
        return ok();
    });
}

template <class TOp>
result<void> unary_impl(TOp &&op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept
{
    return apply(shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto v = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = op(v);
        return ok();
    });
}

template <class TReducer, class TPostProcess>
result<void> reduce_impl(TReducer &&reducer, TPostProcess &&post_process, float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides, bool keep_dims) noexcept
{
    try_(apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        output[offset(out_strides, index)] = init_value;
        return ok();
    }));
    try_(apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto v = input[offset(in_strides, index)];
        const auto out_index = kernels::detail::get_reduced_offset(index, axis, keep_dims);
        auto &dest = output[offset(out_strides, out_index)];
        dest = reducer(dest, v);
        return ok();
    }));
    try_(apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        auto &dest = output[offset(out_strides, index)];
        dest = post_process(dest);
        return ok();
    }));
    return ok();
}
}

result<void> reference::copy(datatype_t type, const gsl::byte *src, gsl::byte *dest,
    const runtime_shape_t &shape, const runtime_shape_t &src_strides, const runtime_shape_t &dest_strides) noexcept
{
    auto src_view = xt::adapt(src, runtime::get_bytes(type, src_strides), xt::no_ownership(),
        runtime::convert_shape_type(shape, type, dt_uint8), runtime::convert_strides_type(src_strides, type, dt_uint8));
    auto dest_view = xt::adapt(dest, runtime::get_bytes(type, dest_strides), xt::no_ownership(),
        runtime::convert_shape_type(shape, type, dt_uint8), runtime::convert_strides_type(dest_strides, type, dt_uint8));
    std::copy(src_view.begin(), src_view.end(), dest_view.begin());
    return ok();
}

#define BINARY_IMPL(op, funct) \
    case op:                   \
        return binary_impl(funct, input_a, input_b, output, in_a_shape, in_a_strides, in_b_shape, in_b_strides, out_strides, fused_activation)

result<void> reference::binary(binary_op_t op, const float *input_a, const float *input_b, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_strides, value_range<float> fused_activation) noexcept
{
    switch (op)
    {
        BINARY_IMPL(binary_add, std::plus<float>());
        BINARY_IMPL(binary_sub, std::minus<float>());
        BINARY_IMPL(binary_mul, std::multiplies<float>());
        BINARY_IMPL(binary_div, std::divides<float>());
        BINARY_IMPL(binary_min, [](float a, float b) { return std::min(a, b); });
        BINARY_IMPL(binary_max, [](float a, float b) { return std::max(a, b); });
        BINARY_IMPL(binary_pow, powf);
    default:
        return err(std::errc::not_supported);
    }
}

#define UNARY_IMPL(op, funct) \
    case op:                  \
        return unary_impl(funct, input, output, shape, in_strides, out_strides)

result<void> reference::unary(unary_op_t op, const float *input, float *output, const runtime_shape_t &shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept
{
    switch (op)
    {
        UNARY_IMPL(unary_abs, fabsf);
        UNARY_IMPL(unary_ceil, ceilf);
        UNARY_IMPL(unary_cos, cosf);
        UNARY_IMPL(unary_exp, expf);
        UNARY_IMPL(unary_floor, floorf);
        UNARY_IMPL(unary_log, logf);
        UNARY_IMPL(unary_neg, std::negate<float>());
        UNARY_IMPL(unary_round, roundf);
        UNARY_IMPL(unary_rsqrt, [](float v) { return 1.f / sqrtf(v); });
        UNARY_IMPL(unary_sin, sinf);
        UNARY_IMPL(unary_sqrt, sqrtf);
        UNARY_IMPL(unary_square, [](float v) { return v * v; });
        UNARY_IMPL(unary_tanh, tanhf);
    default:
        return err(std::errc::not_supported);
    }
}

#define REDUCE_IMPL(op, reducer, post_process) \
    case op:                                   \
        return reduce_impl(reducer, post_process, init_value, input, output, in_shape, axis, in_strides, out_shape, out_strides, keep_dims)

#define REDUCE_IMPL_NO_POST(op, reducer) \
    case op:                             \
        return reduce_impl(reducer, identity<float>(), init_value, input, output, in_shape, axis, in_strides, out_shape, out_strides, keep_dims)

result<void> reference::reduce(reduce_op_t op, float init_value, const float *input, float *output, const runtime_shape_t &in_shape, const runtime_shape_t &axis,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, bool keep_dims) noexcept
{
    auto out_shape = kernels::detail::get_reduced_shape(in_shape, axis, keep_dims);
    switch (op)
    {
        REDUCE_IMPL(reduce_mean, std::plus<float>(), [block_size = (float)xt::compute_size(axis)](float v) { return v / block_size; });
        REDUCE_IMPL_NO_POST(reduce_min, [](float a, float b) { return std::min(a, b); });
        REDUCE_IMPL_NO_POST(reduce_max, [](float a, float b) { return std::max(a, b); });
        REDUCE_IMPL_NO_POST(reduce_sum, std::plus<float>());
    default:
        return err(std::errc::not_supported);
    }
}
