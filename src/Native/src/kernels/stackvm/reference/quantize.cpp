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
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/util.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

namespace
{
template <class TFloat, class TQint>
result<void> quantize_impl(const TFloat *input, TQint *output, const dims_t &in_shape,
    const strides_t &in_strides, const strides_t &out_strides, float scale, float bias, NNCASE_UNUSED kernel_context &context) noexcept
{
    return apply(in_shape, [&](const dims_t &index) -> result<void> {
        auto value = (float)input[offset(in_strides, index)];
        value = value * scale + bias;
        auto qvalue = (int32_t)lrintf(value);
        qvalue = kernels::detail::clamp(qvalue, (int32_t)std::numeric_limits<TQint>::lowest(), (int32_t)std::numeric_limits<TQint>::max());
        output[offset(out_strides, index)] = (TQint)qvalue;
        return ok();
    });
}
}

#define QUANTIZE_IMPL(float_t, qint_t)                                          \
    if (cmp_type<float_t>(in_type) && cmp_type<qint_t>(out_type)) \
    return quantize_impl(reinterpret_cast<const float_t *>(input), reinterpret_cast<qint_t *>(output), in_shape, in_strides, out_strides, scale, bias, context)

result<void> quantize_impl(datatype_t in_type, datatype_t out_type, const gsl::byte *input, gsl::byte *output,
    const dims_t &in_shape, const strides_t &in_strides, const strides_t &out_strides, float scale, float bias, kernel_context &context) noexcept
{
    QUANTIZE_IMPL(float, uint8_t);
    QUANTIZE_IMPL(float, int8_t);
    return err(std::errc::not_supported);
}

result<value_t> nncase::kernels::stackvm::quantize(typecode_t target_type,
                                                   value_t input,
                                                   value_t quant_param,
                                                   value_t output,
                                                   kernel_context &context) {
    try_input(input_mem, input);
    try_output(out_mem, output, target_type, input_tensor->shape());
    try_input_with_value_type(qp, quant_param, quant_param_t);

    try_(quantize_impl(input_tensor->dtype(), output_tensor->dtype(),
                         input_mem,out_mem,
                         input_tensor->shape(), input_tensor->strides(),
                         output_tensor->strides(),
                         qp->scale, (float)qp->zero_point,
                         context));
    return ok(output);
}
