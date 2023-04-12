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
#include "opt_ops.h"
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::stackvm;
using namespace nncase::kernels::stackvm::optimized;

namespace impl {
#if __riscv
template <class TQ>
void riscv_quantize(const float *CXX_RESTRICT input, TQ *CXX_RESTRICT output,
                    size_t count, float scale, float bias) {

    for (size_t i = 0; i < count / 2; i++) {
        auto in1 = input[i * 2];
        auto in2 = input[i * 2 + 1];
        in1 = in1 / scale + bias;
        in2 = in2 / scale + bias;
        int32_t out1, out2;
        asm volatile("fcvt.w.s %0, %1, rne" : "=r"(out1) : "f"(in1));
        asm volatile("fcvt.w.s %0, %1, rne" : "=r"(out2) : "f"(in2));

        output[i * 2] = kernels::detail::clamp(
            out1, (int32_t)std::numeric_limits<TQ>::lowest(),
            (int32_t)std::numeric_limits<TQ>::max());
        output[i * 2 + 1] = kernels::detail::clamp(
            out2, (int32_t)std::numeric_limits<TQ>::lowest(),
            (int32_t)std::numeric_limits<TQ>::max());
    }

    if (count % 2) {
        auto in = (int32_t)roundf(input[count - 1] / scale + bias);
        output[count - 1] = kernels::detail::clamp(
            in, (int32_t)std::numeric_limits<TQ>::lowest(),
            (int32_t)std::numeric_limits<TQ>::max());
    }
}
#endif

template <class TQ>
result<void> quantize(const float *CXX_RESTRICT input, TQ *CXX_RESTRICT output,
                      size_t count, float scale, float bias) {
#if __riscv
    riscv_quantize(input, output, count, scale, bias);
#else
    for (size_t i = 0; i < count; i++) {
        auto qvalue = (int32_t)std::nearbyintf(input[i] / scale + bias);
        output[i] = (TQ)kernels::detail::clamp(
            qvalue, (int32_t)std::numeric_limits<TQ>::lowest(),
            (int32_t)std::numeric_limits<TQ>::max());
    }
#endif
    return ok();
}

} // namespace impl

#define QUANTIZE_IMPL(float_t, qint_t)                                         \
    if (cmp_type<float_t>(in_type) && cmp_type<qint_t>(out_type)) {            \
        return impl::quantize(reinterpret_cast<const float_t *>(input),        \
                              reinterpret_cast<qint_t *>(output),              \
                              compute_size(in_shape), scale, bias);            \
    }

result<void> optimized::quantize(
    datatype_t in_type, datatype_t out_type, const gsl::byte *input,
    gsl::byte *output, const dims_t &in_shape,
    NNCASE_UNUSED const dims_t &in_strides,
    NNCASE_UNUSED const dims_t &out_strides, float scale, float bias,
    NNCASE_UNUSED kernel_context &context) noexcept {
    QUANTIZE_IMPL(float, uint8_t)
    QUANTIZE_IMPL(float, int8_t)
    return err(std::errc::not_supported);
}