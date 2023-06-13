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

result<void> nncase::kernels::stackvm::reference::layer_norm(
    const float *input, float *output, const float *scale, const float *bias,
    gsl::span<const size_t> in_shape, int32_t axis, float epsilon) {
    auto outer_size = 1;
    auto inner_size = 1;
    for (auto i = 0; i < axis; i++)
        outer_size *= in_shape[i];
    for (auto i = axis; i < static_cast<int>(in_shape.size()); i++)
        inner_size *= in_shape[i];

    for (int32_t batch = 0; batch < outer_size; batch++) {
        auto src = input + batch * inner_size;
        auto dest = output + batch * inner_size;

        float mean1 = 0.f;
        for (auto i = 0; i < inner_size; i++)
            mean1 += src[i] / inner_size;

        std::vector<float> sub(inner_size, 0.f);
        for (auto i = 0; i < inner_size; i++)
            sub[i] = src[i] - mean1;

        std::vector<float> pow(inner_size, 0.f);
        for (auto i = 0; i < inner_size; i++)
            pow[i] = sub[i] * sub[i];

        float mean2 = 0.f;
        for (auto i = 0; i < inner_size; i++)
            mean2 += pow[i] / inner_size;

        float add = mean2 + epsilon;
        float sqrt = std::sqrt(add);

        std::vector<float> div(inner_size, 0.f);
        for (auto i = 0; i < inner_size; i++)
            div[i] = sub[i] / sqrt;

        for (auto i = 0; i < inner_size; i++)
            dest[i] = div[i] * scale[i] + bias[i];
    }

    return ok();
}