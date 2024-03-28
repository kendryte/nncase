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
#include "kernel_template.h"
#include "ref_ops.h"
#include <iostream>
#include <nncase/kernels/kernel_context.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::kernels;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels::stackvm::reference;
using namespace nncase::kernels::stackvm;

namespace {
template <typename T>
result<void> instance_norm_impl(const T *input, const T *scale, const T *bias,
                                const T *input_mean, const T *input_var,
                                T *output, std::span<const size_t> in_shape,
                                std::span<const size_t> in_strides,
                                std::span<const size_t> out_strides,
                                float epsilon) {
    return apply(in_shape, [&](std::span<const size_t> index) -> result<void> {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        auto c = index[1];
#pragma GCC diagnostic pop
        auto offi = index[0] * in_shape[1] + index[1];
        auto off = offset(in_strides, index);
        const auto x = input[off];
        output[offset(out_strides, index)] =
            scale[c] * (x - static_cast<T>(input_mean[offi])) /
                static_cast<T>(
                    std::sqrt(static_cast<float>(input_var[offi]) + epsilon)) +
            bias[c];
        return ok();
    });
}
} // namespace

template <typename T>
result<void>
instance_norm_impl2(typecode_t type, const T *input, const T *scale,
                    const T *bias, T *output, std::span<const size_t> in_shape,
                    std::span<const size_t> in_strides,
                    std::span<const size_t> out_strides, float epsilon) {
    auto axes = dims_t{};
    for (size_t i = 2; i < in_shape.size(); ++i) {
        axes.push_back(i);
    }
    auto in_size = runtime::compute_size(in_shape);
    auto channels = in_shape[0] * in_shape[1];
    auto mean = std::make_unique<T[]>(channels);
    auto var = std::make_unique<T[]>(channels);
    auto square_output = std::make_unique<T[]>(in_size);
    auto sub_output = std::make_unique<T[]>(in_size);

    // square and get var
    T init_value = 0;
    auto init_value_addr = IN_CAST(std::byte, &init_value);
    auto tmp_out_strides = strides_t{in_shape[1], 1};
    auto tmp_out_shape = strides_t{in_shape[0], in_shape[1]};
    for (auto i = 0; i < in_shape.size() - 2; ++i) {
        tmp_out_shape.push_back(1);
        tmp_out_strides.push_back(1);
    }
    auto run_reduce = [&](auto &&input, auto &&output, auto &&in_shape,
                          auto &&in_strides) -> result<void> {
        try_(nncase::kernels::stackvm::reference::reduce(
            type, reduce_op_t::mean, init_value_addr, IN_CAST(std::byte, input),
            OUT_CAST(std::byte, output), in_shape, axes, in_strides,
            tmp_out_strides, true, kernels::default_kernel_context()));
        return ok();
    };
    // mean -> reduce_mean(input)
    try_(run_reduce(input, mean.get(), in_shape, in_strides));
    // x - mean
    auto sub_out_shape =
        kernels::detail::get_binary_output_shape(in_shape, tmp_out_shape);
    auto sub_out_strides = runtime::get_default_strides(sub_out_shape);
    try_(nncase::kernels::stackvm::reference::binary(
        type, runtime::stackvm::binary_op_t::sub, IN_CAST(std::byte, input),
        IN_CAST(std::byte, mean.get()), OUT_CAST(std::byte, sub_output.get()),
        in_shape, in_strides, tmp_out_shape, tmp_out_strides, sub_out_shape,
        sub_out_strides));
    try_(nncase::kernels::stackvm::reference::unary(
        type, unary_op_t::square, IN_CAST(std::byte, sub_output.get()),
        OUT_CAST(std::byte, square_output.get()), sub_out_shape,
        sub_out_strides, sub_out_shape, sub_out_strides,
        kernels::default_kernel_context()));
    // var = reduce_mean(square(input - mean))
    try_(run_reduce(square_output.get(), var.get(), sub_out_shape,
                    sub_out_strides));
    try_(instance_norm_impl(input, scale, bias, mean.get(), var.get(), output,
                            in_shape, in_strides, out_strides, epsilon));
    return ok();
}

#define INSTANCE_NORM_IMPL(type)                                               \
    return instance_norm_impl2(typecode, IN_CAST(type, input),                 \
                               IN_CAST(type, scale), IN_CAST(type, bias),      \
                               OUT_CAST(type, output), in_shape, in_strides,   \
                               out_strides, epsilon);

#define TYPE_SELECT_INSTANCE_NORM(_typecode, _impl)                            \
    switch (_typecode) {                                                       \
    case dt_float32:                                                           \
        _impl(float);                                                          \
    case dt_float16:                                                           \
        _impl(half);                                                           \
    case dt_bfloat16:                                                          \
        _impl(bfloat16);                                                       \
    case dt_int8:                                                              \
        _impl(int8_t);                                                         \
    case dt_int16:                                                             \
        _impl(int16_t);                                                        \
    case dt_int32:                                                             \
        _impl(int32_t);                                                        \
    case dt_int64:                                                             \
        _impl(int64_t);                                                        \
    case dt_uint8:                                                             \
        _impl(uint8_t);                                                        \
    case dt_uint16:                                                            \
        _impl(uint16_t);                                                       \
    case dt_uint32:                                                            \
        _impl(uint32_t);                                                       \
    case dt_uint64:                                                            \
        _impl(uint64_t);                                                       \
    case dt_float64:                                                           \
        _impl(double);                                                         \
    default:                                                                   \
        return err(std::errc::not_supported);                                  \
    }

result<void> nncase::kernels::stackvm::reference::instance_norm(
    typecode_t typecode, const std::byte *input, const std::byte *scale,
    const std::byte *bias, std::byte *output, std::span<const size_t> in_shape,
    std::span<const size_t> in_strides, std::span<const size_t> out_strides,
    float epsilon) {
    TYPE_SELECT_INSTANCE_NORM(typecode, INSTANCE_NORM_IMPL);
}
