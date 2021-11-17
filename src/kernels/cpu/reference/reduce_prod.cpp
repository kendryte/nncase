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
#include <limits>
#include <nncase/kernels/cpu/reference/tensor_compute.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu;
using namespace nncase::kernels::cpu::reference;

template NNCASE_API result<void> reference::reduce_prod<float>(const float *input, float *output, const runtime_shape_t &in_shape,
    const runtime_shape_t &in_strides, const runtime_shape_t &out_strides,
    const runtime_shape_t &axis, bool keep_dims, NNCASE_UNUSED kernel_context &context) noexcept;

template <typename T>
result<void> reference::reduce_prod(NNCASE_UNUSED const T *input, NNCASE_UNUSED T *output, NNCASE_UNUSED const runtime_shape_t &in_shape,
    NNCASE_UNUSED const runtime_shape_t &in_strides, NNCASE_UNUSED const runtime_shape_t &out_strides,
    NNCASE_UNUSED const runtime_shape_t &axes, NNCASE_UNUSED bool keep_dims, NNCASE_UNUSED kernel_context &context) noexcept
{
    auto out_shape = kernels::detail::get_reduced_shape(in_shape, axes, keep_dims);

    // init with init_value
    try_(reference::apply(out_shape, [&](const runtime_shape_t &index) -> result<void> {
        output[offset(out_strides, index)] = 1;
        return ok();
    }));

    try_(apply(in_shape, [&](const runtime_shape_t &index) -> result<void> {
        const auto src = input[offset(in_strides, index)];
        auto out_idx = offset(out_strides, kernels::detail::get_reduced_offset(index, axes, keep_dims));
        auto &dst = output[out_idx];
        dst *= src;
        return ok();
    }));

    return ok();
}
