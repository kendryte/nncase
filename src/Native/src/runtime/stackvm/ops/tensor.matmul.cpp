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

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_function::visit(const tensor_matmul_op_t &op) noexcept
{
    try_var(output, pop_addr());
    try_var(bias, pop_addr());
    try_var(input_b, pop_addr());
    try_var(input_a, pop_addr());

    try_var(in_shape_a, module().shape_reg(op.rshape_src1));
    try_var(in_shape_b, module().shape_reg(op.rshape_src2));

    return kernels::matmul(reinterpret_cast<const float *>(input_a), reinterpret_cast<const float *>(input_b),
        reinterpret_cast<const float *>(bias), reinterpret_cast<float *>(output), in_shape_a, in_shape_b,
        { op.fused_clamp_low, op.fused_clamp_high });
}
