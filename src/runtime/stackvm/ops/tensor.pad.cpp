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
#include "../runtime_module.h"
#include <nncase/kernels/tensor_compute.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_module::visit(const tensor_pad_op_t &op) noexcept
{
    try_var(pad_value, pop_scalar(op.datatype));
    try_var(output, pop_addr());
    try_var(input, pop_addr());
    auto &shape = shape_regs_[op.rshape_src];
    auto &in_strides = shape_regs_[op.rstride_src];
    auto &out_strides = shape_regs_[op.rstride_dest];
    auto &paddings = paddings_regs_[op.rpaddings];

    return kernels::pad(op.datatype, reinterpret_cast<const gsl::byte *>(input), reinterpret_cast<gsl::byte *>(output), shape, in_strides, out_strides, paddings, op.pad_mode, pad_value, kernel_context());
}
