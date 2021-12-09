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

result<void> stackvm_runtime_function::visit(const tensor_topk_op_t &op) noexcept
{
    try_var(output_b, pop_addr());
    try_var(output_a, pop_addr());
    try_var(input, pop_addr());
    try_var(in_shape, module().shape_reg(op.rshape_src));
    try_var(in_strides, module().shape_reg(op.rstride_src));
    try_var(out_a_shape, module().shape_reg(op.rshape_dest1));
    try_var(out_a_strides, module().shape_reg(op.rstride_dest1));
    try_var(out_b_shape, module().shape_reg(op.rshape_dest2));
    try_var(out_b_strides, module().shape_reg(op.rstride_dest2));

    switch (op.datatype)
    {
    case dt_float32:
        return kernels::topk(reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output_a), reinterpret_cast<int64_t *>(output_b),
            in_shape, in_strides, out_a_shape, out_a_strides, out_b_shape, out_b_strides, op.k, op.axis, op.largest, op.sorted);
        break;
    default:
        std::cerr << "unsupported dtype for ternary: " + std::string(datatype_names(op.datatype));
        return err(std::errc::invalid_argument);
    }
}
