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
#include <nncase/kernels/cpu/reference/runtime_types.h>
#include <nncase/kernels/kernel_utils.h>
#include <nncase/kernels/stackvm/tensor_ops.h>
#include <nncase/runtime/util.h>
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <iostream>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;
using namespace nncase::kernels::cpu::reference;
using namespace nncase::kernels::stackvm;

template <class T>
result<void> sigmoid_impl(const T *input, T *output,
                         const dims_t &in_shape, const strides_t &input_strides,
                         const dims_t &out_shape, const strides_t &out_strides,
                         NNCASE_UNUSED kernel_context &context) noexcept {
    return apply(out_shape, [&](const dims_t &index) -> result<void> {
        const auto in_index = kernels::detail::get_reduced_offset(index, in_shape);
        auto src_idx = offset(input_strides, in_index);
        auto dst_idx = offset(out_strides, in_index);
        output[dst_idx] = 1 / (1 + exp(-input[src_idx]));
        return ok();
    });
}

template result<void> sigmoid_impl<float>(const float *input, float *output,
                                          const dims_t &input_shape, const strides_t &input_strides,
                                          const dims_t &out_shape, const strides_t &out_strides,
                                          NNCASE_UNUSED kernel_context &context);


result<value_t> nncase::kernels::stackvm::sigmoid(value_t input, value_t output,
                                                  kernel_context &context) {
    try_f32_input(input_mem, input);
    auto dtype = input_tensor->dtype();
    try_var(typecode, to_typecode(dtype));
    try_f32_output(out_mem, output, dtype, input_tensor->shape());
    try_(sigmoid_impl(input_mem,
                        out_mem,
                        input_tensor->shape(), input_tensor->strides(),
                        output_tensor->shape(), output_tensor->strides(), context));
    return ok(output);
}
