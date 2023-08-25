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

namespace {
template <class TQint, class TFloat>
result<void> dequantize_impl(const TQint *input, TFloat *output,
                             gsl::span<const size_t> in_shape,
                             gsl::span<const size_t> in_strides,
                             gsl::span<const size_t> out_strides, float scale,
                             float bias,
                             NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto value = (float)input[offset(in_strides, index)];
        value = (value - bias) * scale;
        output[offset(out_strides, index)] = (TFloat)value;
        return ok();
    });
}
} // namespace

#define DEQUANTIZE_IMPL(qint_t, float_t)                                       \
    if (cmp_type<qint_t>(in_type) && cmp_type<float_t>(out_type))              \
    return dequantize_impl(reinterpret_cast<const qint_t *>(input),            \
                           reinterpret_cast<float_t *>(output), in_shape,      \
                           in_strides, out_strides, scale, bias, context)

result<void> nncase::kernels::stackvm::reference::dequantize(
    datatype_t in_type, datatype_t out_type, const gsl::byte *input,
    gsl::byte *output, gsl::span<const size_t> in_shape,
    gsl::span<const size_t> in_strides, gsl::span<const size_t> out_strides,
    float scale, float bias, kernel_context &context) noexcept {
    DEQUANTIZE_IMPL(uint8_t, float);
    DEQUANTIZE_IMPL(int8_t, float);
    DEQUANTIZE_IMPL(int16_t, float);
    return err(std::errc::not_supported);
}