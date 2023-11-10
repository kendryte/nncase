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
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm::reference;
using namespace nncase::kernels::stackvm;

namespace {
// softmax(x) = exp(x - reduce_max(x)) / reduce_sum(exp(x - reduce_max(x)))
template <typename T>
result<void> softmax_impl(const T *input, T *output,
                          gsl::span<const size_t> in_shape,
                          NNCASE_UNUSED gsl::span<const size_t> in_strides,
                          NNCASE_UNUSED gsl::span<const size_t> out_strides, int64_t axis,
                          float beta, bool needLog = false) noexcept {
    size_t positive_axis = axis < 0 ? in_shape.size() + axis : axis;
    dims_t axes{positive_axis};

    size_t reduced_size = 1;
    for (size_t i = positive_axis; i < in_shape.size(); i++)
    {
        reduced_size *= in_shape[i];
    }
    auto out_size = compute_size(in_shape) / reduced_size;
    std::vector<T> tmp(reduced_size, std::numeric_limits<T>::lowest());

    for (size_t i = 0; i < out_size; i++)
    {
        auto p_input = input + i * reduced_size;
        auto p_output = output + i * reduced_size;

        // reduce_max
        auto max_value = *p_input;
        for(size_t j = 0; j < reduced_size; j++)
        {
            max_value = std::max(max_value, p_input[j]);
        }

        // x - reduce_max
        for(size_t j = 0; j < reduced_size; j++)
        {
            p_output[j] = static_cast<T>((static_cast<float>(p_input[j]) -
                                          static_cast<float>(max_value)) *
                                         beta);
        }

        // exp(x-reduce_max) and sum
        T sum = 0;
        for(size_t j = 0; j < reduced_size; j++)
        {
            p_output[j] = static_cast<T>(expf(static_cast<float>(p_output[j])));
            sum += p_output[j];
        }

        for(size_t j = 0; j < reduced_size; j++)
        {
            p_output[j] /= sum;
            if (needLog)
            {
                p_output[j] = static_cast<T>(std::log(static_cast<float>(p_output[j])));
            }
        }
    }

    //     // reduce_max
    //     try_(
    //         apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
    //             auto in_idx = offset(in_strides, index);
    //             const auto in = input[in_idx];

    //             const auto out_index =
    //                 kernels::detail::get_reduced_offset(index, axes, true);
    //             auto out_idx = offset(reduced_strides, out_index);
    //             auto &out = tmp[out_idx];

    //             out = std::max(in, out);
    //             return ok();
    //         }));

    // // x - reduce_max
    // try_(apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
    //     auto in_idx = offset(in_strides, index);
    //     const auto in = input[in_idx];

    //     const auto out_index =
    //         kernels::detail::get_reduced_offset(index, axes, true);
    //     auto max_idx = offset(reduced_strides, out_index);

    //     auto out_idx = offset(out_strides, index);
    //     output[out_idx] =
    //         static_cast<T>(static_cast<float>(in - tmp[max_idx]) * beta);

    //     return ok();
    // }));

    // // exp(x - reduce_max) and sum
    // tmp.assign(tmp.size(), static_cast<T>(0));
    // try_(apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
    //     auto in_idx = offset(out_strides, index);
    //     const auto in = output[in_idx];

    //     const auto out_index =
    //         kernels::detail::get_reduced_offset(index, axes, true);
    //     auto out_idx = offset(reduced_strides, out_index);
    //     output[in_idx] = static_cast<T>(expf(static_cast<float>(in)));
    //     tmp[out_idx] += static_cast<T>(output[in_idx]);

    //     return ok();
    // }));

    // // div
    // try_(apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
    //     const auto in_index =
    //         kernels::detail::get_reduced_offset(index, axes, true);
    //     auto in_idx = offset(reduced_strides, in_index);
    //     auto in = tmp[in_idx];

    //     auto out_idx = offset(out_strides, index);
    //     auto &out = output[out_idx];
    //     out /= in;
    //     if (needLog) {
    //         out = static_cast<T>(std::log(static_cast<float>(out)));
    //     }
    //     return ok();
    // }));

    return ok();
}

#define SOFTMAX_IMPL(type)                                                     \
    return softmax_impl(IN_CAST(type, input), OUT_CAST(type, output),          \
                        in_shape, in_strides, out_strides, axis, beta,         \
                        needLog);

#define TYPE_SELECT_SOFTMAX(_typecode, _impl)                                  \
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

} // namespace

result<void> nncase::kernels::stackvm::reference::softmax(
    typecode_t typecode, const gsl::byte *input, gsl::byte *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, int64_t axis, float beta,
    bool needLog) noexcept {
    TYPE_SELECT_SOFTMAX(typecode, SOFTMAX_IMPL);
}
