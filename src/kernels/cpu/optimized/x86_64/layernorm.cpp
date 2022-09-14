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

#include <iostream>
#include <nncase/kernels/cpu/optimized/tensor_compute.h>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::optimized;

template result<void> optimized::layernorm<float>(const float *input, float *output, float *scale, float *bias, const runtime_shape_t &in_shape, int32_t axis, float epsilon) noexcept;

template <typename T>
result<void> optimized::layernorm(const T *input, T *output, T *scale, T *bias, const runtime_shape_t &in_shape, int32_t axis, float epsilon) noexcept
{
    return cpu::reference::layernorm(input, output, scale, bias, in_shape, axis, epsilon);
}