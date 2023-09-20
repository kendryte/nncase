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
                                const float *input_mean, const float *input_var,
                                T *output, gsl::span<const size_t> in_shape,
                                gsl::span<const size_t> in_strides,
                                gsl::span<const size_t> out_strides,
                                float epsilon) {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto c = index[1];
        auto offi = index[0] * in_shape[1] + index[1];
        auto off = offset(in_strides, index);
        const auto x = input[off];
        output[offset(out_strides, index)] =
            scale[c] * (x - input_mean[offi]) /
                std::sqrt(input_var[offi] + epsilon) +
            bias[c];
        return ok();
    });
}
} // namespace

template <typename T>
result<void> instance_norm_impl2(const T *input, const T *scale, const T *bias,
                                 T *output, gsl::span<const size_t> in_shape,
                                 gsl::span<const size_t> in_strides,
                                 gsl::span<const size_t> out_strides,
                                 float epsilon) {
    auto axes = dims_t{};
    for (size_t i = 2; i < in_shape.size(); ++i) {
        axes.push_back(i);
    }
    auto in_size = runtime::compute_size(in_shape);
    auto channels = in_shape[0] * in_shape[1];
    auto mean = std::make_unique<float[]>(channels);
    auto var = std::make_unique<float[]>(channels);
    auto square_output = std::make_unique<float[]>(in_size);
    auto sub_output = std::make_unique<float[]>(in_size);

    // square and get var
    auto init_value = 0.f;
    auto init_value_addr = IN_CAST(gsl::byte, &init_value);
    auto tmp_out_strides = strides_t{in_shape[1], 1};
    auto tmp_out_shape = strides_t{in_shape[0], in_shape[1]};
    for (auto i = 0; i < in_shape.size() - 2; ++i) {
        tmp_out_shape.push_back(1);
        tmp_out_strides.push_back(1);
    }
    auto run_reduce = [&](auto &&input, auto &&output, auto &&in_shape,
                          auto &&in_strides) -> result<void> {
        try_(nncase::kernels::stackvm::reference::reduce(
            dt_float32, reduce_op_t::mean, init_value_addr,
            IN_CAST(gsl::byte, input), OUT_CAST(gsl::byte, output), in_shape,
            axes, in_strides, tmp_out_strides, true,
            kernels::default_kernel_context()));
        return ok();
    };
    // mean -> reduce_mean(input)
    try_(run_reduce(input, mean.get(), in_shape, in_strides));
    // x - mean
    auto sub_out_shape =
        kernels::detail::get_binary_output_shape(in_shape, tmp_out_shape);
    auto sub_out_strides = runtime::get_default_strides(sub_out_shape);
    try_(nncase::kernels::stackvm::reference::binary(
        dt_float32, runtime::stackvm::binary_op_t::sub,
        IN_CAST(gsl::byte, input), IN_CAST(gsl::byte, mean.get()),
        OUT_CAST(gsl::byte, sub_output.get()), in_shape, in_strides,
        tmp_out_shape, tmp_out_strides, sub_out_shape, sub_out_strides));
    try_(nncase::kernels::stackvm::reference::unary(
        dt_float32, unary_op_t::square, IN_CAST(gsl::byte, sub_output.get()),
        OUT_CAST(gsl::byte, square_output.get()), sub_out_shape,
        sub_out_strides, sub_out_shape, sub_out_strides,
        kernels::default_kernel_context()));
    // var = reduce_mean(square(input - mean))
    try_(run_reduce(square_output.get(), var.get(), sub_out_shape,
                    sub_out_strides));
    try_(instance_norm_impl(input, scale, bias, mean.get(), var.get(), output,
                            in_shape, in_strides, out_strides, epsilon));
    return ok();
}

result<void> nncase::kernels::stackvm::reference::instance_norm(
    const float *input, const float *scale, const float *bias, float *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, float epsilon) {
    return instance_norm_impl2(input, scale, bias, output, in_shape, in_strides,
                               out_strides, epsilon);
}
