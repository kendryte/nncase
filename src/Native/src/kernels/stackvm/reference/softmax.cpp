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

    if(positive_axis == in_shape.size()-1)
    {
        size_t reduced_size = in_shape[positive_axis];
        auto out_size = compute_size(in_shape) / reduced_size;
        std::vector<T> tmp(reduced_size, std::numeric_limits<T>::lowest());

        for (size_t i = 0; i < out_size; i++)
        {
            auto in_ = input + i * reduced_size;
            auto out_ = output + i * reduced_size;

            // reduce_max
            auto max_value = *in_;
            for(size_t j = 0; j < reduced_size; j++)
            {
                max_value = std::max(max_value, in_[j]);
            }

            // (x - reduce_max) * beta
            for(size_t j = 0; j < reduced_size; j++)
            {
                out_[j] = static_cast<T>((static_cast<float>(in_[j]) - static_cast<float>(max_value)) * beta);
            }

            // exp((x - reduce_max) * beta) and sum
            T sum = 0;
            for(size_t j = 0; j < reduced_size; j++)
            {
                out_[j] = static_cast<T>(expf(static_cast<float>(out_[j])));
                sum += out_[j];
            }

            // div
            for(size_t j = 0; j < reduced_size; j++)
            {
                out_[j] /= sum;
                if (needLog)
                {
                    out_[j] = static_cast<T>(std::log(static_cast<float>(out_[j])));
                }
            }
        }
    }
    else
    {
        size_t axis_size = in_shape[positive_axis];
        size_t reduced_size = 1;
        for (size_t i = positive_axis+1; i < in_shape.size(); i++)
        {
            reduced_size *= in_shape[i];
        }
        auto out_size = compute_size(in_shape) / reduced_size / axis_size;
        std::vector<T> axis_sum(reduced_size, static_cast<T>(0));
        std::vector<T> max_value(reduced_size, std::numeric_limits<T>::lowest());

        for (size_t i = 0; i < out_size; i++)
        {
            auto in_ = input + i * reduced_size * axis_size;
            auto out_ = output + i * reduced_size * axis_size;

            // reduce_max
            for (size_t k = 0; k < axis_size; k++)
            {
                auto in_k = in_ + k * reduced_size;
                for (size_t j = 0; j < reduced_size; j++)
                {
                    max_value[j] = std::max(max_value[j], in_k[j]);
                }
            }

            // exp((x - reduce_max) * beta) and sum
            for (size_t k = 0; k < axis_size; k++)
            {
                auto in_k = in_ + k * reduced_size;
                auto out_k = out_ + k * reduced_size;
                for (size_t j = 0; j < reduced_size; j++)
                {
                    out_k[j] = static_cast<T>(expf((static_cast<float>(in_k[j]) - static_cast<float>(max_value[j])) * beta));
                    axis_sum[j] += out_k[j];
                }
            }

            // div
            for (size_t k = 0; k < axis_size; k++)
            {
                auto out_k = out_ + k * reduced_size;
                for (size_t j = 0; j < reduced_size; j++)
                {
                    out_k[j] /= axis_sum[j];
                    if(needLog)
                        out_k[j] = static_cast<T>(std::log(static_cast<float>((out_k[j]))));
                }
            }
        }
    }


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
