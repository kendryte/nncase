/* Copyright 2019-2020 Canaan Inc.
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
#include "../runtime_module.h"
#include <nncase/kernels/tensor_compute.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_module::visit(const tensor_call_op_t &op) noexcept
{
    try_var(mod, interp().find_module_by_id(op.module_id));

    auto create_tensor = [&]() -> result<runtime_tensor> {
        try_var(rstrides, stack_.pop());
        auto &strides = shape_regs_[rstrides.as_u4()];
        try_var(rshape, stack_.pop());
        auto &shape = shape_regs_[rshape.as_u4()];
        try_var(e_datatype, stack_.pop());
        try_var(addr, pop_addr());

        auto datatype = (datatype_t)e_datatype.as_u1();
        auto size = runtime::compute_size(shape, strides) * get_bytes(datatype);
        try_var(tensor, host_runtime_tensor::create(datatype, shape, strides, { reinterpret_cast<gsl::byte *>(addr), size }, false));
        return ok(tensor);
    };

    for (uint8_t i = 0; i < op.num_dst; i++)
    {
        try_var(tensor, create_tensor());
        try_(mod->output_tensor(op.num_dst - i - 1, tensor));
    }

    for (uint8_t i = 0; i < op.num_src; i++)
    {
        try_var(tensor, create_tensor());
        try_(mod->input_tensor(op.num_src - i - 1, tensor));
    }

    return mod->run();
}
