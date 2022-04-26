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

result<void> stackvm_runtime_function::visit(const tensor_call_op_t &op) noexcept
{
    try_var(mod, module().interp().find_module_by_id(op.module_id));
    try_var(func, mod->find_function_by_id(op.function_id));

    auto create_tensor = [&]() -> result<runtime_tensor> {
        try_var(rstrides, stack_.pop());
        try_var(strides, module().shape_reg(rstrides.as_u4()));
        try_var(rshape, stack_.pop());
        try_var(shape, module().shape_reg(rshape.as_u4()));
        try_var(e_datatype, stack_.pop());
        try_var(addr, pop_addr());

        auto datatype = (datatype_t)e_datatype.as_u1();
        return this->create_tensor(addr, datatype, shape, strides);
    };

    for (uint8_t i = 0; i < op.num_dst; i++)
    {
        try_var(tensor, create_tensor());
        try_(func->output_tensor((size_t)op.num_dst - i - 1, tensor));
    }

    for (uint8_t i = 0; i < op.num_src; i++)
    {
        try_var(tensor, create_tensor());
        try_(func->input_tensor((size_t)op.num_src - i - 1, tensor));
    }

    return func->invoke();
}
