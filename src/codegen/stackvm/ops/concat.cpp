/* Copyright 2020 Canaan Inc.
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
#include "../module_builder.h"

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::stackvm;
using namespace nncase::ir;

void stackvm_module_builder::emit(concat &node, stackvm_op_builder &builder)
{
    uint8_t rshape = 0;
    for (auto in : node.inputs())
    {
        auto &input = allocation(*in);
        builder.lea_buffer(input);
        builder.stshape(rshape, input.strides);
        builder.ldc_i4_(rshape++);
    }

    auto &output = allocation(node.output());
    builder.lea_buffer(output);

    const auto rshape_after_in = rshape;
    ir::shape_t concat_dims { node.concat_dims().begin(), node.concat_dims().end() };

    builder.stshape(rshape_after_in, output.shape);
    builder.stshape(rshape_after_in + 1, output.strides);
    builder.stshape(rshape_after_in + 2, concat_dims);
    builder.tensor_concat_(node.output().type(), (uint8_t)node.inputs().size(), (uint8_t)node.axis(), rshape_after_in + 2,
        rshape_after_in, rshape_after_in + 1);
}
