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

void stackvm_module_builder::emit(matmul &node, stackvm_op_builder &builder)
{
    auto &input_a = allocation(node.input_a());
    auto &input_b = allocation(node.input_b());
    auto &bias = allocation(node.bias());
    auto &output = allocation(node.output());
    builder.lea_buffer(input_a);
    builder.lea_buffer(input_b);
    builder.lea_buffer(bias);
    builder.lea_buffer(output);

    builder.stshape(0, input_a.shape);
    builder.stshape(1, input_a.strides);
    builder.stshape(2, input_b.shape);
    builder.stshape(3, input_b.strides);
    builder.stshape(4, output.shape);
    builder.stshape(5, output.strides);

    builder.tensor_matmul_(0, 1, 2, 3, 4, 5, node.fused_activation().min, node.fused_activation().max);
}
