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

result<void> stackvm_runtime_function::visit(const tensor_quantize_op_t &op) noexcept
{
    try_var(bias, stack_.pop());
    try_var(scale, stack_.pop());
    try_var(output, pop_addr());
    try_var(input, pop_addr());

    try_var(shape, module().shape_reg(op.rshape_src));
    try_var(in_strides, module().shape_reg(op.rstride_src));
    try_var(out_strides, module().shape_reg(op.rstride_dest));

    return kernels::quantize(op.in_datatype, op.dst_datatype, reinterpret_cast<const gsl::byte *>(input),
        reinterpret_cast<gsl::byte *>(output), shape, in_strides, out_strides, scale.as_r4(), bias.as_r4(), module().kernel_context());
}
