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
#include <iostream>
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/debug.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_function::visit(const tensor_reduce_op_t &op) noexcept
{
    try_var(init_value, stack_.pop());
    try_var(output, pop_addr());
    try_var(input, pop_addr());
    try_var(in_shape, module().shape_reg(op.rshape_src));
    try_var(axis, module().shape_reg(op.rshape_axis));
    try_var(in_strides, module().shape_reg(op.rstride_src));
    try_var(out_strides, module().shape_reg(op.rstride_dest));

    switch (op.datatype)
    {
    case dt_float32:
        return kernels::reduce(op.reduce_op, init_value.as_r4(), reinterpret_cast<const float *>(input),
            reinterpret_cast<float *>(output), in_shape, axis, in_strides, out_strides, op.keep_dims, module().kernel_context());
        break;
    case dt_int32:
        return kernels::reduce(op.reduce_op, init_value.as_i4(), reinterpret_cast<const int32_t *>(input),
            reinterpret_cast<int32_t *>(output), in_shape, axis, in_strides, out_strides, op.keep_dims, module().kernel_context());
        break;
    case dt_int64:
        return kernels::reduce(op.reduce_op, init_value.as_i8(), reinterpret_cast<const int64_t *>(input),
            reinterpret_cast<int64_t *>(output), in_shape, axis, in_strides, out_strides, op.keep_dims, module().kernel_context());
        break;
    default:
        std::cerr << "unsupported dtype for reduce: " + std::string(datatype_names(op.datatype)) << std::endl;
        return err(std::errc::invalid_argument);
    }
}
