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
#include <iomanip>
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
result<void>
softmax_impl(const T *input, T *output, gsl::span<const size_t> in_shape,
             NNCASE_UNUSED gsl::span<const size_t> in_strides,
             NNCASE_UNUSED gsl::span<const size_t> out_strides, int64_t axis,
             float beta, bool needLog = false) noexcept {
    size_t positive_axis = axis < 0 ? in_shape.size() + axis : axis;

    if (positive_axis == in_shape.size() - 1) {
        size_t reduced_size = in_shape[positive_axis];
        auto out_size = compute_size(in_shape) / reduced_size;
        std::vector<T> tmp(reduced_size, std::numeric_limits<T>::lowest());

        for (size_t i = 0; i < out_size; i++) {
            auto in_ = input + i * reduced_size;
            auto out_ = output + i * reduced_size;

            // reduce_max
            auto max_value = *in_;
            for (size_t j = 0; j < reduced_size; j++) {
                max_value = std::max(max_value, in_[j]);
            }

            // Debug: 输出 max 结果
            std::cout << "[REF DEBUG] Batch " << i << " - Max: " << std::fixed << std::setprecision(6) << (float)max_value << std::endl;

            // (x - reduce_max) * beta
            for (size_t j = 0; j < reduced_size; j++) {
                out_[j] = static_cast<T>((static_cast<float>(in_[j]) -
                                          static_cast<float>(max_value)) *
                                         beta);
            }

            // Debug: 输出前几个 sub 结果
            std::cout << "[REF DEBUG] Batch " << i << " - Sub results (first 8): ";
            for (size_t debug_idx = 0; debug_idx < std::min(reduced_size, (size_t)8); debug_idx++) {
                std::cout << std::fixed << std::setprecision(6) << (float)out_[debug_idx] << " ";
            }
            std::cout << std::endl;

            // exp((x - reduce_max) * beta) and sum
            T sum = 0;
            for (size_t j = 0; j < reduced_size; j++) {
                out_[j] = static_cast<T>(expf(static_cast<float>(out_[j])));
                sum += out_[j];
            }

            // Debug: 输出前几个 exp 结果
            std::cout << "[REF DEBUG] Batch " << i << " - Exp results (first 8): ";
            for (size_t debug_idx = 0; debug_idx < std::min(reduced_size, (size_t)8); debug_idx++) {
                std::cout << std::fixed << std::setprecision(6) << (float)out_[debug_idx] << " ";
            }
            std::cout << std::endl;

            // Debug: 输出 sum 结果
            std::cout << "[REF DEBUG] Batch " << i << " - Sum: " << std::fixed << std::setprecision(6) << (float)sum << std::endl;

            // div
            T inv_sum = static_cast<T>(1.0f / static_cast<float>(sum));

            // Debug: 输出 1/sum 结果
            std::cout << "[REF DEBUG] Batch " << i << " - 1/Sum: " << std::fixed << std::setprecision(6) << (float)inv_sum << std::endl;

            for (size_t j = 0; j < reduced_size; j++) {
                out_[j] /= sum;
                if (needLog) {
                    out_[j] =
                        static_cast<T>(std::log(static_cast<float>(out_[j])));
                }
            }

            // Debug: 输出前几个最终结果
            std::cout << "[REF DEBUG] Batch " << i << " - Final results (first 8): ";
            for (size_t debug_idx = 0; debug_idx < std::min(reduced_size, (size_t)8); debug_idx++) {
                std::cout << std::fixed << std::setprecision(6) << (float)out_[debug_idx] << " ";
            }
            std::cout << std::endl;

            // Debug: 验证概率和
            T prob_sum = 0;
            for (size_t debug_idx = 0; debug_idx < reduced_size; debug_idx++) {
                prob_sum += out_[debug_idx];
            }
            std::cout << "[REF DEBUG] Batch " << i << " - Probability sum: " << std::fixed << std::setprecision(6) << (float)prob_sum << std::endl;
            std::cout << "REF ----------------------------------------" << std::endl;
        }
    } else {
        size_t axis_size = in_shape[positive_axis];
        size_t reduced_size = 1;
        for (size_t i = positive_axis + 1; i < in_shape.size(); i++) {
            reduced_size *= in_shape[i];
        }
        auto out_size = compute_size(in_shape) / reduced_size / axis_size;

        for (size_t i = 0; i < out_size; i++) {
            std::vector<T> axis_sum(reduced_size, static_cast<T>(0));
            std::vector<T> max_value(reduced_size,
                                     std::numeric_limits<T>::lowest());
            auto in_ = input + i * reduced_size * axis_size;
            auto out_ = output + i * reduced_size * axis_size;

            // reduce_max
            for (size_t k = 0; k < axis_size; k++) {
                auto in_k = in_ + k * reduced_size;
                for (size_t j = 0; j < reduced_size; j++) {
                    max_value[j] = std::max(max_value[j], in_k[j]);
                }
            }

            // Debug: 输出 max 结果 (只显示前几个元素)
            std::cout << "[REF DEBUG] Batch " << i << " - Max (first 8): ";
            for (size_t debug_idx = 0; debug_idx < std::min(reduced_size, (size_t)8); debug_idx++) {
                std::cout << std::fixed << std::setprecision(6) << (float)max_value[debug_idx] << " ";
            }
            std::cout << std::endl;

            // exp((x - reduce_max) * beta) and sum
            for (size_t k = 0; k < axis_size; k++) {
                auto in_k = in_ + k * reduced_size;
                auto out_k = out_ + k * reduced_size;

                // Debug: 输出第一个轴元素的 sub 结果
                if (k == 0) {
                    std::cout << "[REF DEBUG] Batch " << i << " - Sub results axis[0] (first 8): ";
                    for (size_t debug_idx = 0; debug_idx < std::min(reduced_size, (size_t)8); debug_idx++) {
                        float sub_val = (static_cast<float>(in_k[debug_idx]) - static_cast<float>(max_value[debug_idx])) * beta;
                        std::cout << std::fixed << std::setprecision(6) << sub_val << " ";
                    }
                    std::cout << std::endl;
                }

                for (size_t j = 0; j < reduced_size; j++) {
                    out_k[j] =
                        static_cast<T>(expf((static_cast<float>(in_k[j]) -
                                             static_cast<float>(max_value[j])) *
                                            beta));
                    axis_sum[j] += out_k[j];
                }

                // Debug: 输出第一个轴元素的 exp 结果
                if (k == 0) {
                    std::cout << "[REF DEBUG] Batch " << i << " - Exp results axis[0] (first 8): ";
                    for (size_t debug_idx = 0; debug_idx < std::min(reduced_size, (size_t)8); debug_idx++) {
                        std::cout << std::fixed << std::setprecision(6) << (float)out_k[debug_idx] << " ";
                    }
                    std::cout << std::endl;
                }
            }

            // Debug: 输出 sum 结果
            std::cout << "[REF DEBUG] Batch " << i << " - Sum (first 8): ";
            for (size_t debug_idx = 0; debug_idx < std::min(reduced_size, (size_t)8); debug_idx++) {
                std::cout << std::fixed << std::setprecision(6) << (float)axis_sum[debug_idx] << " ";
            }
            std::cout << std::endl;

            // div
            for (size_t k = 0; k < axis_size; k++) {
                auto out_k = out_ + k * reduced_size;

                // Debug: 输出第一个轴元素的 1/sum 结果
                if (k == 0) {
                    std::cout << "[REF DEBUG] Batch " << i << " - 1/Sum axis[0] (first 8): ";
                    for (size_t debug_idx = 0; debug_idx < std::min(reduced_size, (size_t)8); debug_idx++) {
                        float inv_sum = 1.0f / static_cast<float>(axis_sum[debug_idx]);
                        std::cout << std::fixed << std::setprecision(6) << inv_sum << " ";
                    }
                    std::cout << std::endl;
                }

                for (size_t j = 0; j < reduced_size; j++) {
                    out_k[j] /= axis_sum[j];
                    if (needLog)
                        out_k[j] = static_cast<T>(
                            std::log(static_cast<float>((out_k[j]))));
                }

                // Debug: 输出第一个轴元素的最终结果
                if (k == 0) {
                    std::cout << "[REF DEBUG] Batch " << i << " - Final results axis[0] (first 8): ";
                    for (size_t debug_idx = 0; debug_idx < std::min(reduced_size, (size_t)8); debug_idx++) {
                        std::cout << std::fixed << std::setprecision(6) << (float)out_k[debug_idx] << " ";
                    }
                    std::cout << std::endl;
                }
            }

            // Debug: 验证第一个轴元素的概率和
            T prob_sum = 0;
            auto out_0 = out_;
            for (size_t debug_idx = 0; debug_idx < reduced_size; debug_idx++) {
                prob_sum += out_0[debug_idx];
            }
            std::cout << "[REF DEBUG] Batch " << i << " - Probability sum axis[0]: " << std::fixed << std::setprecision(6) << (float)prob_sum << std::endl;
            std::cout << "REF ----------------------------------------" << std::endl;
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
