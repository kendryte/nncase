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

void stackvm_module_builder::emit(topk &node, stackvm_op_builder &builder)
{
    auto &input = allocation(node.input());
    auto &output_a = allocation(node.output_a());
    auto &output_b = allocation(node.output_b());

    builder.lea_buffer(input);
    builder.lea_buffer(output_a);
    builder.lea_buffer(output_b);

    builder.stshape(0, input.shape);
    builder.stshape(1, input.strides);
    builder.stshape(2, output_a.shape);
    builder.stshape(3, output_a.strides);
    builder.stshape(4, output_b.shape);
    builder.stshape(5, output_b.strides);

    builder.tensor_topk_(node.input().type(), 0, 1, 2, 3, 4, 5, node.k(), node.axis(), node.largest(), node.sorted());
}
