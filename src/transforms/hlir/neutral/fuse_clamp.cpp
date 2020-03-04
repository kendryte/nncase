/* Copyright 2019-2020 Canaan Inc.
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
#include <hlir/ops/binary.h>
#include <hlir/ops/clamp.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/conv2d.h>
#include <hlir/transforms/neutral/fuse_clamp.h>
#include <hlir/visitor.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

namespace
{
value_range<float> combine(value_range<float> act, constant &c_low, constant &c_high)
{
    auto low = *reinterpret_cast<const float *>(c_low.data().data());
    auto high = *reinterpret_cast<const float *>(c_high.data().data());
    act.min = std::max(act.min, low);
    act.max = std::max(act.max, high);
    return act;
}
}

bool fuse_clamp_conv2d_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_clamp)
    {
        auto &p = static_cast<clamp &>(node);
        if (auto low = try_get_direct_parent<constant>(p, 1))
        {
            if (auto high = try_get_direct_parent<constant>(p, 2))
            {
                if (auto conv = try_get_direct_parent<conv2d>(p, 0))
                {
                    if (xt::compute_size(low->output().shape()) == 1
                        && xt::compute_size(high->output().shape()) == 1)
                    {
                        context.inputs.emplace_back(&conv->input());
                        context.outputs.emplace_back(&p.output());

                        context.matched_nodes.emplace_back(&p);
                        context.matched_nodes.emplace_back(low);
                        context.matched_nodes.emplace_back(high);
                        context.matched_nodes.emplace_back(conv);
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void fuse_clamp_conv2d_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_low = static_cast<constant &>(*context.matched_nodes[1]);
    auto &old_high = static_cast<constant &>(*context.matched_nodes[2]);
    auto &old_conv = static_cast<conv2d &>(*context.matched_nodes[3]);

    auto act = combine(old_conv.fused_activation(), old_low, old_high);

    auto conv = context.graph.emplace<conv2d>(output.shape(), old_conv.weights(), old_conv.bias(), old_conv.groups(),
        old_conv.padding_h(), old_conv.padding_w(), old_conv.stride_h(), old_conv.stride_w(), old_conv.dilation_h(), old_conv.dilation_w(),
        act);
    conv->name(old_conv.name());

    conv->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(conv->output());
}

bool fuse_clamp_binary_transform::on_try_match(node &node, transform_context &context)
{
    if (node.runtime_opcode() == op_clamp)
    {
        auto &p = static_cast<clamp &>(node);
        if (auto low = try_get_direct_parent<constant>(p, 1))
        {
            if (auto high = try_get_direct_parent<constant>(p, 2))
            {
                if (auto b = try_get_direct_parent<binary>(p, 0))
                {
                    if (xt::compute_size(low->output().shape()) == 1
                        && xt::compute_size(high->output().shape()) == 1)
                    {
                        context.inputs.emplace_back(&b->input_a());
                        context.inputs.emplace_back(&b->input_b());
                        context.outputs.emplace_back(&p.output());

                        context.matched_nodes.emplace_back(&p);
                        context.matched_nodes.emplace_back(low);
                        context.matched_nodes.emplace_back(high);
                        context.matched_nodes.emplace_back(b);
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void fuse_clamp_binary_transform::process(transform_context &context)
{
    auto &output_a = *context.inputs[0]->connection();
    auto &output_b = *context.inputs[1]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_low = static_cast<constant &>(*context.matched_nodes[1]);
    auto &old_high = static_cast<constant &>(*context.matched_nodes[2]);
    auto &old_b = static_cast<binary &>(*context.matched_nodes[3]);

    auto act = combine(old_b.fused_activation(), old_low, old_high);

    auto b = context.graph.emplace<binary>(old_b.binary_op(), output_a.shape(), output_b.shape(), act);
    b->name(old_b.name());

    b->input_a().connect(output_a);
    b->input_b().connect(output_b);
    for (auto &in : dup(inputs))
        in->connect(b->output());
}
