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
#include <cmath>
#include <runtime_utils.h>
#include <type_traits>

using namespace nncase::runtime::cpu;

namespace kernels {

namespace {
// softmax(x) = exp(x - reduce_max(x)) / reduce_sum(exp(x - reduce_max(x)))
template <typename T>
void softmax_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
                  gsl::span<const size_t> in_strides,
                  gsl::span<const size_t> out_strides, int64_t axis, float beta,
                  bool needLog = false) noexcept {
    size_t positive_axis = axis < 0 ? in_shape.size() + axis : axis;
    dims_t axes{positive_axis};

    auto reduced_shape = get_reduced_shape(in_shape, axes, true);
    auto reduced_strides = get_default_strides(reduced_shape);
    auto reduced_size = compute_size(reduced_shape);
    auto tmp = (T *)runtime_util->malloc(reduced_size * sizeof(T));
    for (size_t i = 0; i < reduced_size; i++) {
        tmp[i] = std::numeric_limits<T>().lowest();
    }

    // reduce_max
    (apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        auto in_idx = offset(in_strides, index);
        const auto in = input[in_idx];

        const auto out_index = get_reduced_offset(index, axes, true);
        auto out_idx = offset(reduced_strides, out_index);
        auto &out = tmp[out_idx];

        out = in > out ? in : out;
    }));

    // x - reduce_max
    (apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        auto in_idx = offset(in_strides, index);
        const auto in = input[in_idx];

        const auto out_index = get_reduced_offset(index, axes, true);
        auto max_idx = offset(reduced_strides, out_index);

        auto out_idx = offset(out_strides, index);
        output[out_idx] = (in - tmp[max_idx]) * beta;
    }));

    // exp(x - reduce_max) and sum
    for (size_t i = 0; i < reduced_size; i++) {
        tmp[i] = static_cast<T>(0);
    }
    (apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        auto in_idx = offset(out_strides, index);
        const auto in = output[in_idx];

        const auto out_index = get_reduced_offset(index, axes, true);
        auto out_idx = offset(reduced_strides, out_index);
        if (std::is_same_v<T, float>) {
            output[in_idx] = nncase_mt->float_unary_exp(in);
        } else {
            runtime_util->rt_assert(false,
                                   (char *)"Not supported Type in softmax!");
        }

        tmp[out_idx] += output[in_idx];
    }));

    // div
    (apply(in_shape, [&](gsl::span<const size_t> index) -> void {
        const auto in_index = get_reduced_offset(index, axes, true);
        auto in_idx = offset(reduced_strides, in_index);
        auto in = tmp[in_idx];

        auto out_idx = offset(out_strides, index);
        auto &out = output[out_idx];
        out /= in;
        if (needLog) {
            if (std::is_same_v<T, float>) {
                out = nncase_mt->float_unary_log(out);
            } else {
                runtime_util->rt_assert(
                    false, (char *)"Not supported Type in softmax!");
            }
        }
    }));

    runtime_util->free(tmp);
}
} // namespace
void softmax(const float *input, float *output,
             gsl::span<const size_t> in_shape,
             gsl::span<const size_t> in_strides,
             gsl::span<const size_t> out_strides, int64_t axis, float beta,
             bool needLog) noexcept {
    return softmax_impl(input, output, in_shape, in_strides, out_strides, axis,
                        beta, needLog);
}
} // namespace kernels