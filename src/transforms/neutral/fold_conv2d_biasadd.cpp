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
#include <nncase/ir/ops/binary.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_conv2d_biasadd.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_conv2d_biasadd_transform::on_try_match(node &node, transform_context &context)
{
    conv2d *conv = nullptr;
    binary *add = nullptr;
    constant *bias_constant = nullptr;
    constant *add_constant = nullptr;
    if ((add = node_cast<binary>(node))
        && (add->binary_op() == binary_add)
        && (((conv = try_get_direct_parent<conv2d>(*add, 0)) && (add_constant = try_get_direct_parent<constant>(*add, 1))) || ((conv = try_get_direct_parent<conv2d>(*add, 1)) && (add_constant = try_get_direct_parent<constant>(*add, 0))))
        && (conv->fused_activation() == value_range<float>::full())
        && (bias_constant = node_cast<constant>(conv->bias().connection()->owner()))
        && (bias_constant->data().size() == add_constant->data().size()))
    {
        context.inputs.emplace_back(&conv->input());
        context.inputs.emplace_back(&conv->weights());
        context.outputs.emplace_back(&add->output());

        context.matched_nodes.emplace_back(conv);
        context.matched_nodes.emplace_back(bias_constant);
        context.matched_nodes.emplace_back(add);
        context.matched_nodes.emplace_back(add_constant);
        return true;
    }

    return false;
}

void fold_conv2d_biasadd_transform::process(transform_context &context)
{
    auto old_conv = node_cast<conv2d>(*context.matched_nodes[0]);
    auto old_bias = node_cast<constant>(*context.matched_nodes[1]);
    const float *p = reinterpret_cast<const float *>(old_bias->data().data());

    auto add = node_cast<binary>(*context.matched_nodes[2]);
    auto add_bias = node_cast<constant>(*context.matched_nodes[3]);
    const float *q = reinterpret_cast<const float *>(add_bias->data().data());

    // update bias
    size_t channels = old_conv->output_channels();
    std::vector<float> bias(channels, 0.f);
    for (size_t i = 0; i < channels; i++)
    {
        bias[i] = p[i] + q[i];
    }

    // create new bias
    auto new_bias = context.graph.emplace<constant>(dt_float32, shape_t{ channels }, std::span(bias));
    new_bias->name(old_bias->name());

    // update act
    auto act = old_conv->fused_activation();
    auto add_act = add->fused_activation();
    act.min = std::max(act.min, add_act.min);
    act.max = std::min(act.max, add_act.max);

    // create new conv2d
    auto new_conv = context.graph.emplace<conv2d>(old_conv->input().shape(), old_conv->weights().shape(), old_conv->groups(), old_conv->padding_h(), old_conv->padding_w(), old_conv->stride_h(), old_conv->stride_w(), old_conv->dilation_h(), old_conv->dilation_w(), act);

    new_conv->input().connect(*context.inputs[0]->connection());
    new_conv->weights().connect(*context.inputs[1]->connection());
    new_conv->bias().connect(new_bias->output());
    new_conv->name(old_conv->name());

    auto inputs = context.outputs[0]->connections();
    for (auto &in : dup(inputs))
        in->connect(new_conv->output());
}