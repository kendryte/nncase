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
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::kernels::stackvm;

template <class T>
static void layernorm_impl(int inner_size, const T *src, const T *scale,
                           const T *bias, float epsilon, T *dst,
                           bool use_mean) {
    T mean1 = 0;
    if (use_mean) {
        for (auto i = 0; i < inner_size; i++)
            mean1 += src[i] / inner_size;
    }

    std::vector<T> sub(inner_size, 0);
    for (auto i = 0; i < inner_size; i++)
        sub[i] = src[i] - mean1;

    std::vector<T> pow(inner_size, 0);
    for (auto i = 0; i < inner_size; i++)
        pow[i] = sub[i] * sub[i];

    T mean2 = 0;
    for (auto i = 0; i < inner_size; i++)
        mean2 += pow[i] / inner_size;

    T add = mean2 + static_cast<T>(epsilon);
    T sqrt = std::sqrt(add);

    std::vector<T> div(inner_size, 0);
    for (auto i = 0; i < inner_size; i++)
        div[i] = sub[i] / sqrt;

    for (auto i = 0; i < inner_size; i++)
        dst[i] = div[i] * scale[i] + bias[i];
}

template <class T>
result<void> layer_norm_impl2(const T *input, T *output, const T *scale,
                              const T *bias, std::span<const size_t> in_shape,
                              int32_t axis, float epsilon, bool use_mean) {

    int ndim = in_shape.size();
    int positive_axis = axis < 0 ? ndim + axis : axis;
    int axis_dim = 1; // in_shape[positive_axis];

    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++)
        out_side *= in_shape[i];

    for (size_t i = positive_axis; i < ndim; i++) {
        axis_dim *= in_shape[i];
    }

    for (size_t i = 0; i < out_side; i++) {
        layernorm_impl(axis_dim, input, scale, bias, epsilon, output, use_mean);
        input += axis_dim;
        output += axis_dim;
    }
    return ok();
}

#define LAYER_NORM_IMPL(type)                                                  \
    return layer_norm_impl2(IN_CAST(type, input), OUT_CAST(type, output),      \
                            IN_CAST(type, scale), IN_CAST(type, bias),         \
                            in_shape, axis, epsilon, use_mean)

#define TYPE_SELECT_LAYER_NORM(_typecode, _impl)                               \
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

result<void> nncase::kernels::stackvm::reference::layer_norm(
    typecode_t typecode, const std::byte *input, std::byte *output,
    const std::byte *scale, const std::byte *bias,
    std::span<const size_t> in_shape, int32_t axis, float epsilon,
    bool use_mean) {
    TYPE_SELECT_LAYER_NORM(typecode, LAYER_NORM_IMPL);
}
