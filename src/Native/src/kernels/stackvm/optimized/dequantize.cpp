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

template <class TQ>
void riscv_dequantize(const TQ *CXX_RESTRICT input, float *CXX_RESTRICT output,
                      size_t count, float scale, float bias) {
    for (size_t i = 0; i < count / 2; i++) {
        // hand written pipeline for in order CPU
        auto in1_q = input[i * 2];
        auto in2_q = input[i * 2 + 1];
        auto in1 = (float)in1_q;
        auto in2 = (float)in2_q;
        auto out1 = (in1 - bias) * scale;
        auto out2 = (in2 - bias) * scale;

        output[i * 2] = out1;
        output[i * 2 + 1] = out2;
    }

    if (count % 2)
        output[count - 1] = (input[count - 1] - bias) * scale;
}

template <class TQint>
result<void> dequantize(const TQint *CXX_RESTRICT input,
                        float *CXX_RESTRICT output, size_t count, float scale,
                        float bias) {
#if __riscv
    riscv_dequantize(input, output, count, scale, bias);
#else
    for (size_t i = 0; i < count; i++) {
        output[i] = (input[i] - bias) * scale;
    }
#endif
    return ok();
}
} // namespace impl

#define DEQUANTIZE_IMPL(qint_t, float_t)                                       \
    if (cmp_type<qint_t>(in_type) && cmp_type<float_t>(out_type)) {            \
        return impl::dequantize(reinterpret_cast<const qint_t *>(input),       \
                                reinterpret_cast<float_t *>(output),           \
                                compute_size(in_shape), scale, bias);          \
    }

result<void> optimized::dequantize(
    datatype_t in_type, datatype_t out_type, const gsl::byte *input,
    gsl::byte *output, const dims_t &in_shape,
    NNCASE_UNUSED const dims_t &in_strides,
    NNCASE_UNUSED const dims_t &out_strides, float scale, float bias,
    NNCASE_UNUSED kernel_context &context) noexcept {
    DEQUANTIZE_IMPL(uint8_t, float)
    DEQUANTIZE_IMPL(int8_t, float)
    return err(std::errc::not_supported);
}
