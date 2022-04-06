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
#include <nncase/ir/op_utils.h>
#include <nncase/ir/ops/batch_to_space.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/space_to_batch.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fix_tflite_error_shape.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fix_dilated_conv_transform::on_try_match(node &node, transform_context &context)
{
    if (auto s2b = node_cast<space_to_batch>(node))
    {
        if (auto conv = try_get_direct_child<conv2d>(*s2b))
        {
            if (auto b2s = try_get_direct_child<batch_to_space>(*conv))
            {
                if (s2b->output().shape()[0] != s2b->input().shape()[0])
                    return false;

                context.matched_nodes.emplace_back(s2b);
                context.matched_nodes.emplace_back(conv);
                context.matched_nodes.emplace_back(b2s);

                context.inputs.emplace_back(&s2b->input());
                context.inputs.emplace_back(&conv->weights());
                context.inputs.emplace_back(&conv->bias());
                context.outputs.emplace_back(&b2s->output());

                return true;
            }
        }
    }

    return false;
}

void fix_dilated_conv_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &weights = *context.inputs[1]->connection();
    auto &bias = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_s2b = static_cast<space_to_batch &>(*context.matched_nodes[0]);
    auto &old_conv = static_cast<conv2d &>(*context.matched_nodes[1]);
    auto &old_b2s = static_cast<batch_to_space &>(*context.matched_nodes[2]);

    auto new_s2b = context.graph.emplace<space_to_batch>(old_s2b.input().type(), old_s2b.input().shape(),
        old_s2b.block_size_h(), old_s2b.block_size_w(),
        old_s2b.padding_h(), old_s2b.padding_w(),
        old_s2b.pad_value(),
        old_s2b.block_size_h(), old_s2b.block_size_w());

    auto new_conv = context.graph.emplace<conv2d>(new_s2b->output().shape(), old_conv.input_at(1).shape(), old_conv.groups(), old_conv.padding_h(), old_conv.padding_w(),
        old_conv.stride_h(), old_conv.stride_w(), old_conv.dilation_h(), old_conv.dilation_w(), old_conv.fused_activation());

    auto new_b2s = context.graph.emplace<batch_to_space>(old_b2s.input().type(), new_conv->output().shape(),
        old_b2s.block_size_h(), old_b2s.block_size_w(),
        old_b2s.strides(), old_b2s.begin(),
        old_b2s.end(),
        old_b2s.crop_h(), old_b2s.crop_w(),
        old_b2s.block_size_h(), old_b2s.block_size_w());

    new_s2b->name(old_s2b.name());
    new_conv->name(old_conv.name());
    new_b2s->name(old_b2s.name());

    new_s2b->input().connect(output);
    new_conv->input().connect(new_s2b->output());
    new_conv->weights().connect(weights);
    new_conv->bias().connect(bias);
    new_b2s->input().connect(new_conv->output());

    for (auto &in : dup(inputs))
        in->connect(new_b2s->output());
}