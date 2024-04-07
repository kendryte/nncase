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
#include <nncase/kernels/apply.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;

namespace {
#define SCALAR_CAST_IMPL(f)                                                    \
    if (is_scalar(in_shape)) {                                                 \
        output[0] = f(input[0]);                                               \
        return ok();                                                           \
    }

template <class TInput, class TOutput>
result<void> cast_impl(const TInput *input, TOutput *output,
                       std::span<const size_t> in_shape,
                       std::span<const size_t> in_strides,
                       std::span<const size_t> out_strides,
                       NNCASE_UNUSED kernel_context &context) noexcept {
    SCALAR_CAST_IMPL(static_cast<TOutput>);
    return apply(in_shape, [&](std::span<const size_t> index) -> result<void> {
        auto value = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = static_cast<TOutput>(value);
        return ok();
    });
}

result<void> cast_f32_to_bf16_impl(
    const float *input, bfloat16 *output, std::span<const size_t> in_shape,
    std::span<const size_t> in_strides, std::span<const size_t> out_strides,
    NNCASE_UNUSED kernel_context &context) noexcept {
    SCALAR_CAST_IMPL(bfloat16::round_to_bfloat16);
    return apply(in_shape, [&](std::span<const size_t> index) -> result<void> {
        auto value = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = bfloat16::round_to_bfloat16(value);
        return ok();
    });
}

result<void> cast_f32_to_fp16_impl(
    const float *input, half *output, std::span<const size_t> in_shape,
    std::span<const size_t> in_strides, std::span<const size_t> out_strides,
    NNCASE_UNUSED kernel_context &context) noexcept {
    SCALAR_CAST_IMPL(half::round_to_half);
    return apply(in_shape, [&](std::span<const size_t> index) -> result<void> {
        auto value = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = half::round_to_half(value);
        return ok();
    });
}
} // namespace

#define CAST_IMPL_LV2(input_t, output_t)                                       \
    if (cmp_type<output_t>(out_type)) {                                        \
        if (contiguous) {                                                      \
            return cast_impl(reinterpret_cast<const input_t *>(input),         \
                             reinterpret_cast<output_t *>(output), in_shape,   \
                             in_strides, out_strides, context);                \
        } else {                                                               \
            return cast_impl(reinterpret_cast<const input_t *>(input),         \
                             reinterpret_cast<output_t *>(output), in_shape,   \
                             in_strides, out_strides, context);                \
        }                                                                      \
    }

#define CAST_IMPL_LV1(input_t)                                                 \
    if (cmp_type<input_t>(in_type)) {                                          \
        CAST_IMPL_LV2(input_t, bool);                                          \
        CAST_IMPL_LV2(input_t, uint8_t);                                       \
        CAST_IMPL_LV2(input_t, uint16_t);                                      \
        CAST_IMPL_LV2(input_t, uint32_t);                                      \
        CAST_IMPL_LV2(input_t, uint64_t);                                      \
        CAST_IMPL_LV2(input_t, int8_t);                                        \
        CAST_IMPL_LV2(input_t, int16_t);                                       \
        CAST_IMPL_LV2(input_t, int32_t);                                       \
        CAST_IMPL_LV2(input_t, int64_t);                                       \
        CAST_IMPL_LV2(input_t, float);                                         \
    }

result<void> cast_impl(datatype_t in_type, datatype_t out_type,
                       const std::byte *input, std::byte *output,
                       std::span<const size_t> in_shape,
                       std::span<const size_t> in_strides,
                       std::span<const size_t> out_strides,
                       kernel_context &context) noexcept {
    if (cmp_dt(in_type, dt_float32) && cmp_dt(out_type, dt_bfloat16))
        return cast_f32_to_bf16_impl(reinterpret_cast<const float *>(input),
                                     reinterpret_cast<bfloat16 *>(output),
                                     in_shape, in_strides, out_strides,
                                     context);
    if (cmp_dt(in_type, dt_float32) && cmp_dt(out_type, dt_float16))
        return cast_f32_to_fp16_impl(reinterpret_cast<const float *>(input),
                                     reinterpret_cast<half *>(output), in_shape,
                                     in_strides, out_strides, context);
    bool contiguous = is_contiguous(in_shape, in_strides);
    CAST_IMPL_LV1(bool);
    CAST_IMPL_LV1(uint8_t);
    CAST_IMPL_LV1(uint16_t);
    CAST_IMPL_LV1(uint32_t);
    CAST_IMPL_LV1(uint64_t);
    CAST_IMPL_LV1(int8_t);
    CAST_IMPL_LV1(int16_t);
    CAST_IMPL_LV1(int32_t);
    CAST_IMPL_LV1(int64_t);
    CAST_IMPL_LV1(bfloat16);
    CAST_IMPL_LV1(half);
    CAST_IMPL_LV1(float);
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::cast(
    typecode_t new_type, runtime::stackvm::cast_mode_t cast_mode, value_t input,
    value_t output, kernel_context &context) {
    if (cast_mode != runtime::stackvm::cast_mode_t::kdefault)
        return err(std::errc::not_supported);
    try_input(input_mem, input);
    try_output(out_mem, output, new_type, input_tensor->shape());
    try_(cast_impl(input_tensor->dtype(), new_type, input_mem, out_mem,
                   input_tensor->shape(), input_tensor->strides(),
                   output_tensor->strides(), context));
    return ok(output);
}
