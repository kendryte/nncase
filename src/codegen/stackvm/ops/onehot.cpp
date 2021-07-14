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

void stackvm_module_builder::emit(onehot &node, stackvm_op_builder &builder)
{
    auto &indices = allocation(node.indices());
    auto &depth = allocation(node.depth());
    auto &on_value = allocation(node.on_value());
    auto &off_value = allocation(node.off_value());
    auto &output = allocation(node.output());
    builder.lea_buffer(indices);
    builder.lea_buffer(depth);
    builder.lea_buffer(on_value);
    builder.lea_buffer(off_value);
    builder.lea_buffer(output);

    builder.stshape(0, indices.shape);
    builder.stshape(1, output.shape);
    builder.stshape(2, output.strides);
    builder.tensor_onehot_(node.depth().type(), 0, 1, 2, node.axis());
}
