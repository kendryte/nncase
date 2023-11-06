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
#include <cmath>
#include <iostream>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template result<void> reference::instancenorm<float>(const float *input, float *output, float *scale, float *bias, const runtime_shape_t &in_shape, float epsilon) noexcept;

template <typename T>
result<void> reference::instancenorm(const T *input, T *output, T *scale, T *bias, const runtime_shape_t &in_shape, float epsilon) noexcept
{
    auto outer_size = static_cast<int>(in_shape[0]);
    auto inner_size = 1;
    for (size_t i = 2; i < in_shape.size(); i++)
        inner_size *= static_cast<int>(in_shape[i]);
    for (auto batch = 0; batch < outer_size; batch++)
    {
        for (size_t c = 0; c < in_shape[1]; c++)
        {
            auto src = input + batch * inner_size * in_shape[1] + c * inner_size;
            auto dest = output + batch * inner_size * in_shape[1] + c * inner_size;

            double mean1 = 0.f;
            for (auto i = 0; i < inner_size; i++)
                mean1 += src[i] / inner_size;

            std::vector<double> sub(inner_size, 0.f);
            for (auto i = 0; i < inner_size; i++)
                sub[i] = (src[i] - mean1);

            std::vector<double> pow(inner_size, 0.f);
            for (auto i = 0; i < inner_size; i++)
                pow[i] = sub[i] * sub[i];

            double mean2 = 0.f;
            for (auto i = 0; i < inner_size; i++)
                mean2 += pow[i] / inner_size;

            double add = mean2 + epsilon;
            double sqrt = std::sqrt(add);

            for (auto i = 0; i < inner_size; i++)
                dest[i] = sub[i] * scale[c] / sqrt + bias[c];
        }
    }

    return ok();
}