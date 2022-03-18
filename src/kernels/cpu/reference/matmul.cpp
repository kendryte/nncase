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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template result<void> reference::matmul<float>(const float *input_a, const float *input_b, const float *bias, float *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation) noexcept;

template <typename T>
result<void> reference::matmul(const T *input_a, const T *input_b, const T *bias, T *output,
    const runtime_shape_t &in_a_shape, const runtime_shape_t &in_a_strides, const runtime_shape_t &in_b_shape,
    const runtime_shape_t &in_b_strides, const runtime_shape_t &out_shape, const runtime_shape_t &out_strides,
    value_range<float> fused_activation) noexcept
{
    (void)in_a_strides;
    (void)in_b_strides;
    (void)out_shape;
    (void)out_strides;
    int32_t a_rows = static_cast<int32_t>(in_a_shape[0]);
    int32_t a_cols = static_cast<int32_t>(in_a_shape[1]);
    int32_t b_cols = static_cast<int32_t>(in_b_shape[1]);

    for (int32_t oy = 0; oy < a_rows; oy++)
    {
        for (int32_t ox = 0; ox < b_cols; ox++)
        {
            float value = bias[ox];

            for (int32_t i = 0; i < a_cols; i++)
            {
                const auto a = input_a[oy * a_cols + i];
                const auto b = input_b[i * b_cols + ox];
                value += a * b;
            }

            output[oy * b_cols + ox] = nncase::kernels::detail::apply_activation(value, fused_activation);
        }
    }

    return ok();
}
