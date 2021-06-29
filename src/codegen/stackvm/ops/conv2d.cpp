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

void stackvm_module_builder::emit(conv2d &node, stackvm_op_builder &builder)
{
    auto &input = allocation(node.input());
    auto &weights = allocation(node.weights());
    auto &bias = allocation(node.bias());
    auto &output = allocation(node.output());
    builder.lea_buffer(input);
    builder.lea_buffer(weights);
    builder.lea_buffer(bias);
    builder.lea_buffer(output);
    builder.ldpadding(node.padding_h());
    builder.ldpadding(node.padding_w());

    builder.stshape(0, input.shape);
    builder.stshape(1, input.strides);
    builder.stshape(2, weights.shape);
    builder.stshape(3, weights.strides);
    builder.stshape(4, bias.strides);
    builder.stshape(5, output.strides);
    builder.tensor_conv2d_(node.input().type(), 0, 1, 2, 3, 4, 5, (uint16_t)node.groups(), (uint16_t)node.stride_h(), (uint16_t)node.stride_w(),
        (uint16_t)node.dilation_h(), (uint16_t)node.dilation_w(), node.fused_activation().min, node.fused_activation().max);
}
