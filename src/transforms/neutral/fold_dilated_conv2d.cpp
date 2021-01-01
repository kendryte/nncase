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
#include <nncase/ir/ops/batch_to_space.h>
#include <nncase/ir/ops/bitcast.h>
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/dequantize.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/quantize.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/ops/space_to_batch.h>
#include <nncase/ir/ops/transpose.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/fold_dilated_conv2d.h>
#include <nncase/transforms/pass.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

bool fold_dilated_conv2d::on_try_match(node &node, transform_context &context)
{
    conv2d *conv;
    space_to_batch *s2b;
    batch_to_space *b2s;

    if ((conv = node_cast<conv2d>(node))
        && (s2b = try_get_direct_parent<space_to_batch>(*conv, 0))
        && (b2s = try_get_direct_child<batch_to_space>(*conv)))
    {
        context.inputs.emplace_back(&s2b->input());
        context.inputs.emplace_back(&conv->input_at(1));
        context.inputs.emplace_back(&conv->input_at(2));
        context.outputs.emplace_back(&b2s->output());

        context.matched_nodes.emplace_back(s2b);
        context.matched_nodes.emplace_back(conv);
        context.matched_nodes.emplace_back(b2s);

        return true;
    }
    return false;
}

void fold_dilated_conv2d::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &weights = *context.inputs[1]->connection();
    auto &bias = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &s2b = static_cast<space_to_batch &>(*context.matched_nodes[0]);
    auto &b2s = static_cast<batch_to_space &>(*context.matched_nodes[2]);
    auto &conv = static_cast<conv2d &>(*context.matched_nodes[1]);
    int32_t padded_if_h = s2b.padding_h().sum() + s2b.input().shape()[2];
    int32_t padded_if_w = s2b.padding_w().sum() + s2b.input().shape()[3];
    int32_t dilation_h = s2b.block_size_h();
    int32_t dilation_w = s2b.block_size_w();
    int32_t weight_h = weights.shape()[2];
    int32_t weight_w = weights.shape()[3];
    int32_t out_h = b2s.output().shape()[2] + b2s.crop_h()[0] + b2s.crop_h()[1];
    int32_t out_w = b2s.output().shape()[3] + b2s.crop_w()[0] + b2s.crop_w()[1];

    int32_t stride_h = (padded_if_h - dilation_h * (weight_h - 1) - 1) / (out_h - 1);
    int32_t stride_w = (padded_if_w - dilation_w * (weight_w - 1) - 1) / (out_w - 1);

    auto begin = b2s.begin();
    auto end = b2s.end();
    [[maybe_unused]] xt::svector<padding> slice_paddings(4, padding::zero());
    // for (size_t i = 0; i < slice_paddings.size(); i++)
    slice_paddings[0] = { -begin[0], end[0] - (int32_t)b2s.output().shape()[0] };
    slice_paddings[1] = { -begin[3], end[3] - (int32_t)b2s.output().shape()[1] };
    slice_paddings[2] = { -begin[1], end[1] - (int32_t)b2s.output().shape()[2] };
    slice_paddings[3] = { -begin[2], end[2] - (int32_t)b2s.output().shape()[3] };

    // update s2b_pad padding value according to b2s_slice padding.
    [[maybe_unused]] xt::svector<padding> new_s2b_pad_paddings(4, padding::zero());
    new_s2b_pad_paddings[0] = padding { 0, 0 };
    // slice_paddings is minus value(crop) actually, so need to use "+" next.
    new_s2b_pad_paddings[1] = padding { 0, 0 };
    new_s2b_pad_paddings[2] = { s2b.padding_h().before + stride_h * slice_paddings[2].before - b2s.crop_h()[0], s2b.padding_h().after + stride_h * slice_paddings[2].after - b2s.crop_h()[1] };
    new_s2b_pad_paddings[3] = { s2b.padding_w().before + stride_w * slice_paddings[3].before - b2s.crop_w()[0], s2b.padding_w().after + stride_w * slice_paddings[3].after - b2s.crop_w()[1] };

    auto dilated_conv = context.graph.emplace<conv2d>(s2b.input().shape(), conv.weights().shape(), conv.groups(), new_s2b_pad_paddings[2], new_s2b_pad_paddings[3], stride_h, stride_w, dilation_h, dilation_w, conv.fused_activation());

    dilated_conv->input().connect(output);
    dilated_conv->weights().connect(weights);
    dilated_conv->bias().connect(bias);
    output.connect(dilated_conv->input());

    for (auto &in : dup(inputs))
        in->connect(dilated_conv->output());
}