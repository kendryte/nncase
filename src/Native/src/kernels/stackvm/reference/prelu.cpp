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
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

template <typename T>
result<void> prelu_impl(const T *input, const T *slope_mem, T *output,
                        std::span<const size_t> in_shape,
                        std::span<const size_t> input_strides,
                        std::span<const size_t> slope_shape,
                        std::span<const size_t> slope_strides,
                        std::span<const size_t> out_shape,
                        std::span<const size_t> out_strides,
                        NNCASE_UNUSED kernel_context &context) {
    return apply(out_shape, [&](std::span<const size_t> index) -> result<void> {
        const auto in_index =
            kernels::detail::get_reduced_offset(index, in_shape);
        const auto slope_index =
            kernels::detail::get_reduced_offset(index, slope_shape);
        const auto slope = slope_mem[offset(slope_strides, slope_index)];
        const auto x = input[offset(input_strides, in_index)];
        output[offset(out_strides, index)] =
            x < static_cast<T>(0) ? slope * x : x;
        return ok();
    });
}

#define PRELU_IMPL(type)                                                       \
    return prelu_impl(IN_CAST(type, input), IN_CAST(type, slope_mem),          \
                      OUT_CAST(type, output), in_shape, input_strides,         \
                      slope_shape, slope_strides, out_shape, out_strides,      \
                      context);

#define TYPE_SELECT_PRELU(_typecode, _impl)                                    \
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

result<void> nncase::kernels::stackvm::reference::prelu(
    typecode_t typecode, const std::byte *input, const std::byte *slope_mem,
    std::byte *output, std::span<const size_t> in_shape,
    std::span<const size_t> input_strides, std::span<const size_t> slope_shape,
    std::span<const size_t> slope_strides, std::span<const size_t> out_shape,
    std::span<const size_t> out_strides,
    NNCASE_UNUSED kernel_context &context) {
    TYPE_SELECT_PRELU(typecode, PRELU_IMPL);
}