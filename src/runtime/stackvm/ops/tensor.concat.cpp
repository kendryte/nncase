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

result<void> stackvm_runtime_module::visit(const tensor_concat_op_t &op) noexcept
{
    try_var(output, pop_addr());
    itlib::small_vector<const gsl::byte *, 4> inputs(op.num_src);
    std::vector<runtime_shape_t> in_strides(op.num_src);
    for (size_t i = 0; i < inputs.size(); i++)
    {
        try_var(rstrides, stack_.pop());
        auto &strides = shape_regs_[rstrides.as_u4()];
        try_var(input, pop_addr());
        inputs[inputs.size() - i - 1] = reinterpret_cast<const gsl::byte *>(input);
        in_strides[inputs.size() - i - 1] = strides;
    }

    auto &out_shape = shape_regs_[op.rshape_dest];
    auto &out_strides = shape_regs_[op.rstride_dest];
    auto &concat_dims = shape_regs_[op.rshape_dims];
    return kernels::concat(op.datatype, inputs, reinterpret_cast<gsl::byte *>(output), out_shape,
        in_strides, out_strides, op.axis, concat_dims);
}
