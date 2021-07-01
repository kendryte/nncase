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
#include "../module_builder.h"

using namespace nncase;
using namespace nncase::codegen;
using namespace nncase::codegen::stackvm;
using namespace nncase::ir;

void stackvm_module_builder::emit(call &node, stackvm_op_builder &builder)
{
    auto target_id = module_id(&node.target());

    uint8_t rshape = 0;
    for (auto in : node.inputs())
    {
        auto &input = allocation(*in);
        builder.lea_buffer(input);
        builder.ldc_i4_((uint8_t)input.type);
        builder.stshape(rshape, input.shape);
        builder.ldc_i4_(rshape++);
        builder.stshape(rshape, input.strides);
        builder.ldc_i4_(rshape++);
    }

    for (auto out : node.outputs())
    {
        auto &output = allocation(*out);
        builder.lea_buffer(output);
        builder.ldc_i4_((uint8_t)output.type);
        builder.stshape(rshape, output.shape);
        builder.ldc_i4_(rshape++);
        builder.stshape(rshape, output.strides);
        builder.ldc_i4_(rshape++);
    }

    builder.tensor_call_(target_id, (uint8_t)node.inputs().size(), (uint8_t)node.outputs().size());
}
