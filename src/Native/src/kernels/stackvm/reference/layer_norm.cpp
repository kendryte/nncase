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

static void layernorm_impl(int inner_size, const float *src, const float *scale,
                           const float *bias, float epsilon, float *dst) {
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
        dst[i] = div[i] * scale[i] + bias[i];
}

result<void> nncase::kernels::stackvm::reference::layer_norm(
    const float *input, float *output, const float *scale, const float *bias,
    gsl::span<const size_t> in_shape, int32_t axis, float epsilon) {

    int ndim = in_shape.size();
    int positive_axis = axis < 0 ? ndim + axis : axis;
    int axis_dim = 1; // in_shape[positive_axis];

    size_t out_side = 1;
    for (size_t i = 0; i < positive_axis; i++)
        out_side *= in_shape[i];

    for (size_t i = positive_axis; i < ndim; i++) {
        axis_dim *= in_shape[i];
    }

    for (size_t i = 0; i < out_side; i++) {
        layernorm_impl(axis_dim, input, scale, bias, epsilon, output);
        input += axis_dim;
        output += axis_dim;
    }
    return ok();
}