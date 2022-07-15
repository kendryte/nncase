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
#include "../shape_infer.h"
#include <iostream>
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_context.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/ref_ops.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu::reference;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels::stackvm::reference;

namespace {
result<void> lrn_impl(const float *input, float alpha, float beta, float bias,
                      float *output, const dims_t &in_shape,
                      const strides_t &in_strides,
                      const strides_t &out_strides) {
    return apply(in_shape, [&](const dims_t &index) -> result<void> {
        auto off = offset(in_strides, index);
        const auto x = input[off];
        output[offset(out_strides, index)] =
            x / std::pow(x * alpha + bias, beta);
        return ok();
    });
}
} // namespace

result<void> nncase::kernels::stackvm::reference::lrn(
    const float *input, float alpha, float beta, float bias, int size,
    float *output, const dims_t &in_shape, const strides_t &in_strides,
    const strides_t &out_strides) {
    return ok();
}