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

result<void> nncase::kernels::stackvm::reference::batchnorm(
    const float *input, const float *scale, const float *bias,
    const float *input_mean, const float *input_var, float *output,
    gsl::span<const size_t> in_shape, gsl::span<const size_t> in_strides,
    gsl::span<const size_t> out_strides, float epsilon) {
    return apply(in_shape, [&](gsl::span<const size_t> index) -> result<void> {
        auto c = index[1];
        const auto x = input[offset(in_strides, index)];
        output[offset(out_strides, index)] =
            (x - input_mean[c]) / std::sqrt(input_var[c] + epsilon) * scale[c] +
            bias[c];
        return ok();
    });
}