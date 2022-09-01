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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class TInput, class TOutput>
result<void> convert_impl(const TInput *input, TOutput *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        auto value = input[offset(in_strides, index)];
        if (to_datatype<TOutput>() == dt_bfloat16)
            output[offset(out_strides, index)] = bfloat16::round_to_bfloat16(static_cast<float>(value));
        else
            output[offset(out_strides, index)] = static_cast<TOutput>(value);
        return ok();
    });
}

result<void> convert_f32_to_bf16_impl(const float *input, bfloat16 *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        auto value = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = bfloat16::round_to_bfloat16(value);
        return ok();
    });
}

result<void> convert_f32_to_fp16_impl(const float *input, half *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        auto value = input[offset(in_strides, index)];
        output[offset(out_strides, index)] = half::round_to_half(value);
        return ok();
    });
}
}

#define CONVERT_IMPL_LV2(input_t, output_t)  \
    if (out_type == to_datatype<output_t>()) \
    return convert_impl(reinterpret_cast<const input_t *>(input), reinterpret_cast<output_t *>(output), in_shape, in_strides, out_strides, context)

#define CONVERT_IMPL_LV1(input_t)            \
    if (in_type == to_datatype<input_t>())   \
    {                                        \
        CONVERT_IMPL_LV2(input_t, uint8_t);  \
        CONVERT_IMPL_LV2(input_t, uint16_t); \
        CONVERT_IMPL_LV2(input_t, uint32_t); \
        CONVERT_IMPL_LV2(input_t, uint64_t); \
        CONVERT_IMPL_LV2(input_t, int8_t);   \
        CONVERT_IMPL_LV2(input_t, int16_t);  \
        CONVERT_IMPL_LV2(input_t, int32_t);  \
        CONVERT_IMPL_LV2(input_t, int64_t);  \
        CONVERT_IMPL_LV2(input_t, float);    \
        CONVERT_IMPL_LV2(input_t, bfloat16); \
    }

result<void> reference::convert(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, kernel_context &context) noexcept
{
    if (in_type == dt_float32 && out_type == dt_bfloat16)
        return convert_f32_to_bf16_impl(reinterpret_cast<const float *>(input), reinterpret_cast<bfloat16 *>(output),
            in_shape, in_strides, out_strides, context);
    if (in_type == dt_float32 && out_type == dt_float16)
        return convert_f32_to_fp16_impl(reinterpret_cast<const float *>(input), reinterpret_cast<half *>(output),
            in_shape, in_strides, out_strides, context);
    CONVERT_IMPL_LV1(uint8_t);
    CONVERT_IMPL_LV1(uint16_t);
    CONVERT_IMPL_LV1(uint32_t);
    CONVERT_IMPL_LV1(uint64_t);
    CONVERT_IMPL_LV1(int8_t);
    CONVERT_IMPL_LV1(int16_t);
    CONVERT_IMPL_LV1(int32_t);
    CONVERT_IMPL_LV1(int64_t);
    CONVERT_IMPL_LV1(bfloat16);
    CONVERT_IMPL_LV1(half);
    CONVERT_IMPL_LV1(float);
    return err(std::errc::not_supported);
}
