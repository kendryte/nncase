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
#include <chrono>
#include <iostream>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template result<void> reference::compress<float>(const float *input, const uint8_t *condition, float *output, const runtime_shape_t &input_shape, const runtime_shape_t &condition_shape, const int axis) noexcept;

template <class T>
result<void> reference::compress(const T *input, const uint8_t *condition, T *output, const runtime_shape_t &input_shape, const runtime_shape_t &condition_shape, const int axis) noexcept
{
    if (axis == (int)input_shape.size())
    {
        for (auto i = 0; i < (int)condition_shape[0]; i++)
        {
            if ((float)*(condition + i) == 0)
            {
                continue;
            }
            *output++ = input[i];
        }
    }
    else
    {
        int select_slice = 1;
        for (auto i = axis + 1; i < (int)input_shape.size(); i++)
        {
            select_slice *= input_shape[i];
        }
        for (auto j = 0; j < (int)kernels::detail::compute_size(input_shape); j++)
        {
            auto i = j % (select_slice * input_shape[axis]);
            auto cond_index = i / select_slice;
            if (select_slice != 1 && (cond_index >= condition_shape[0] || condition[cond_index] == 0))
                continue;
            if (select_slice == 1 && (i % input_shape[axis] >= condition_shape[0] || condition[cond_index % input_shape[axis] % condition_shape[0]] == 0))
                continue;
            *output++ = input[j];
        }
    }
    return ok();
}