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

result<void> stackvm_runtime_function::visit(const tensor_cumsum_op_t &op) noexcept
{
    try_var(output, pop_addr());
    try_var(input, pop_addr());
    try_var(in_shape, module().shape_reg(op.rshape_src));

    switch (op.datatype)
    {
    case dt_float32:
        return kernels::cumsum(reinterpret_cast<const float *>(input), reinterpret_cast<float *>(output),
            in_shape, op.axis, op.exclusive, op.reverse);
        break;
    case dt_int32:
        return kernels::cumsum(reinterpret_cast<const int32_t *>(input), reinterpret_cast<int32_t *>(output),
            in_shape, op.axis, op.exclusive, op.reverse);
        break;
    default:
        std::cerr << "unsupported dtype for cumsum: " + std::string(datatype_names(op.datatype));
        return err(std::errc::invalid_argument);
    }
}
