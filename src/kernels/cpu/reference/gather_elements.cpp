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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>
using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;
using namespace std;

template result<void> reference::gather_elements(const float *input, const int64_t *indices, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &indices_shape, const int axis) noexcept;

template <typename TI, typename TK>
result<void> reference::gather_elements(const TI *input, const TK *indices, TI *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &indices_shape, const int axis) noexcept
{
    // indices_shape == output_shape
    // out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
    // out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
    // out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
    std::vector<int> per_axis_size(indices_shape.size(), 1);
    std::vector<int> input_per_axis_size(indices_shape.size(), 1);

    // compute size per axis
    for (int idx = indices_shape.size() - 2; idx >= 0; idx--)
    {
        per_axis_size[idx] = indices_shape[idx + 1] * per_axis_size[idx + 1];
        input_per_axis_size[idx] = in_shape[idx + 1] * input_per_axis_size[idx + 1];
    }

    for (size_t i = 0; i < compute_size(indices_shape); i++)
    {
        std::vector<int> index;
        get_gather_index(per_axis_size, index, i, axis, 0);

        // compute indices offset to update index
        int indice_index = 0;
        for (size_t t = 0; t < index.size(); t++)
        {
            indice_index += per_axis_size[t] * index[t];
        }
        // process index value if negative value
        index[axis] = indices[indice_index] < 0 ? indices[indice_index] + in_shape[axis] : indices[indice_index];

        // compute input offset
        int input_index = 0;
        for (size_t t = 0; t < index.size(); t++)
        {
            input_index += input_per_axis_size[t] * index[t];
        }
        output[i] = input[input_index];
    }

    return ok();
}
