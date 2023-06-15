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
#include "ref_ops.h"
#include <cmath>
#include <iostream>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm::reference;
using namespace nncase::kernels::stackvm;

namespace {
// log_softmax(x) = log(exp(x - reduce_max(x)) / reduce_sum(exp(x -
// reduce_max(x))))
template <typename T>
result<void>
log_softmax_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                 gsl::span<const size_t> in_strides,
                 gsl::span<const size_t> out_strides, int64_t axis) noexcept {
    size_t positive_axis = axis < 0 ? in_shape.size() + axis : axis;
    dims_t axes{positive_axis};

    auto reduced_shape =
        kernels::detail::get_reduced_shape(in_shape, axes, true);
    auto reduced_strides = get_default_strides(reduced_shape);
    auto reduced_size = compute_size(reduced_shape);
    std::vector<T> tmp(reduced_size, std::numeric_limits<T>::lowest());

    // reduce_max
    try_(apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto in_idx = offset(in_strides, index);
        const auto in = input[in_idx];

        const auto out_index =
            kernels::detail::get_reduced_offset(index, axes, true);
        auto out_idx = offset(reduced_strides, out_index);
        auto &out = tmp[out_idx];

        out = std::max(in, out);
        return ok();
    }));

    // x - reduce_max
    try_(apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto in_idx = offset(in_strides, index);
        const auto in = input[in_idx];

        const auto out_index =
            kernels::detail::get_reduced_offset(index, axes, true);
        auto max_idx = offset(reduced_strides, out_index);

        auto out_idx = offset(out_strides, index);
        output[out_idx] = (in - tmp[max_idx]);

        return ok();
    }));

    // exp(x - reduce_max) and sum
    tmp.assign(tmp.size(), static_cast<T>(0));
    try_(apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto in_idx = offset(out_strides, index);
        const auto in = output[in_idx];

        const auto out_index =
            kernels::detail::get_reduced_offset(index, axes, true);
        auto out_idx = offset(reduced_strides, out_index);
        output[in_idx] = expf(in);
        tmp[out_idx] += output[in_idx];

        return ok();
    }));

    // div and log
    try_(apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        const auto in_index =
            kernels::detail::get_reduced_offset(index, axes, true);
        auto in_idx = offset(reduced_strides, in_index);
        auto in = tmp[in_idx];

        auto out_idx = offset(out_strides, index);
        auto &out = output[out_idx];
        out /= in;
        out = std::log(out);
        return ok();
    }));

    return ok();
}
} // namespace

result<void> nncase::kernels::stackvm::reference::log_softmax(
    const float *input, float *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    int32_t axis) noexcept {
    return log_softmax_impl(input, output, in_shape, in_strides, out_strides,
                            axis);
}