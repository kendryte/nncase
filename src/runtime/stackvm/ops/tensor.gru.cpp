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
#include "../runtime_function.h"
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_function::visit(const tensor_gru_op_t &op) noexcept
{
    try_var(output_h, pop_addr());
    try_var(output, pop_addr());
    try_var(initial_h, pop_addr());
    try_var(b, pop_addr());
    try_var(r, pop_addr());
    try_var(w, pop_addr());
    try_var(input, pop_addr());

    try_var(in_shape, module().shape_reg(op.input_shape_src));
    try_var(w_shape, module().shape_reg(op.w_shape_src));

    return kernels::gru(reinterpret_cast<const float *>(input), reinterpret_cast<const float *>(w),
        reinterpret_cast<const float *>(r), reinterpret_cast<const float *>(b),
        reinterpret_cast<float *>(initial_h), reinterpret_cast<float *>(output),
        reinterpret_cast<float *>(output_h), in_shape, w_shape, op.direction, op.linear_before_reset);
}