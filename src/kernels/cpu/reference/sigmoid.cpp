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
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template result<void> reference::sigmoid<float>(const float *input, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides) noexcept;

template <typename T>
result<void> reference::sigmoid(const T *input, T *output, const runtime_shape_t &in_shape, const runtime_shape_t &in_strides,
    const runtime_shape_t &out_strides) noexcept
{
    return apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto in_index = kernels::detail::get_reduced_offset(index, in_shape);
        auto src_idx = offset(in_strides, in_index);
        auto dst_idx = offset(out_strides, in_index);
        output[dst_idx] = 1 / (1 + exp(-input[src_idx]));
        return ok();
    });
}