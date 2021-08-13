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
#include <nncase/kernels/convolution.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::stackvm;

result<void> stackvm_runtime_function::visit(const tensor_conv2d_op_t &op) noexcept
{
    try_var(padding_w, pop_padding());
    try_var(padding_h, pop_padding());
    try_var(output, pop_addr());
    try_var(bias, pop_addr());
    try_var(weights, pop_addr());
    try_var(input, pop_addr());
    try_var(in_shape, module().shape_reg(op.rshape_src));
    try_var(in_strides, module().shape_reg(op.rstride_src));
    try_var(w_shape, module().shape_reg(op.rshape_kernel));
    try_var(w_strides, module().shape_reg(op.rstride_kernel));
    try_var(bias_strides, module().shape_reg(op.rstride_bias));
    try_var(out_strides, module().shape_reg(op.rstride_dest));

    if (op.datatype != dt_float32)
        return err(nncase_errc::datatype_mismatch);
    return kernels::conv2d(reinterpret_cast<const float *>(input), reinterpret_cast<const float *>(weights),
        reinterpret_cast<const float *>(bias), reinterpret_cast<float *>(output), in_shape, in_strides, w_shape, w_strides, bias_strides, out_strides,
        padding_h, padding_w, op.groups, op.stride_h, op.stride_w, op.dilation_h, op.dilation_w, { op.fused_clamp_low, op.fused_clamp_high }, module().kernel_context());
}
