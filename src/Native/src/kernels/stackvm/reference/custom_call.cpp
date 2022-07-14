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
#include <nncase/runtime/allocator.h>
#include <nncase/runtime/host_buffer.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/util.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;
using namespace nncase::kernels;

result<value_t> nncase::kernels::stackvm::custom_call(
    NNCASE_UNUSED const std::string &registered_name,
    NNCASE_UNUSED gsl::span<const gsl::byte> fields,
    NNCASE_UNUSED const std::vector<value_t> &inputs,
    NNCASE_UNUSED value_t output, NNCASE_UNUSED kernel_context &context) {
    // try_input(input_mem, input);
    // try_output(out_mem, output, target_type, input_tensor->shape());
    // try_input_with_value_type(deq_param, dequant_param, quant_param_t);

    // try_(dequantize_impl(input_tensor->dtype(), output_tensor->dtype(),
    //                      input_mem, out_mem, input_tensor->shape(),
    //                      input_tensor->strides(), output_tensor->strides(),
    //                      deq_param->scale, (float)deq_param->zero_point,
    //                      context));
    return ok(output);
}