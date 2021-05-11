/* Copyright 2019-2020 Canaan Inc.
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
#include <xtensor/xadapt.hpp>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class TQint, class TFloat>
result<void> dequantize_impl(const TQint *input, TFloat *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias, NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        auto value = (float)input[offset(in_strides, index)];
        value = value * scale + bias;
        output[offset(out_strides, index)] = (TFloat)value;
        return ok();
    });
}
}

#define DEQUANTIZE_IMPL(qint_t, float_t)                                        \
    if (in_type == to_datatype<qint_t>() && out_type == to_datatype<float_t>()) \
    return dequantize_impl(reinterpret_cast<const qint_t *>(input), reinterpret_cast<float_t *>(output), in_shape, in_strides, out_strides, scale, bias, context)

result<void> reference::dequantize(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const runtime_shape_t &in_shape, const runtime_shape_t &in_strides, const runtime_shape_t &out_strides, float scale, float bias, kernel_context &context) noexcept
{
    DEQUANTIZE_IMPL(uint8_t, float);
    return err(std::errc::not_supported);
}
