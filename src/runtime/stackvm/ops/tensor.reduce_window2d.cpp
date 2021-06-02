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
#include <nncase/kernels/reduce_window.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_module::visit(const tensor_reduce_window2d_op_t &op) noexcept
{
    try_var(padding_w, pop_padding());
    try_var(padding_h, pop_padding());
    try_var(output, pop_addr());
    try_var(init_value, stack_.pop());
    try_var(input, pop_addr());
    auto &in_shape = shape_regs_[op.rshape_src];
    auto &in_strides = shape_regs_[op.rstride_src];
    auto &out_strides = shape_regs_[op.rstride_dest];

    if (op.datatype != dt_float32)
        return err(nncase_errc::datatype_mismatch);
    return kernels::reduce_window2d(op.reduce_op, reinterpret_cast<const float *>(input), init_value.as_r4(),
        reinterpret_cast<float *>(output), in_shape, in_strides, out_strides, padding_h, padding_w, op.filter_h, op.filter_w,
        op.stride_h, op.stride_w, op.dilation_h, op.dilation_w, { op.fused_clamp_low, op.fused_clamp_high }, kernel_context());
}
