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
#include <nncase/ir/ops/constant.h>
#include <nncase/ir/ops/conv2d.h>
#include <nncase/ir/ops/pad.h>
#include <nncase/ir/ops/slice.h>
#include <nncase/ir/visitor.h>
#include <nncase/transforms/neutral/pad_conv.h>

using namespace nncase;
using namespace nncase::ir;
using namespace nncase::ir::transforms;

int gcd(int x, int y)
{
    return y ? gcd(y, x % y) : x;
}

bool pad_conv_transform::on_try_match(node &node, transform_context &context)
{
    conv2d *conv;
    if ((conv = node_cast<conv2d>(node)))
    {
        if ((conv->input().shape()[2] < 4 || conv->input().shape()[3] < 4)
            && ((conv->weights().shape()[2] == 1 && conv->weights().shape()[3] == 1)
                || (conv->weights().shape()[2] == 3 && conv->weights().shape()[3] == 3)))
        {
            context.inputs.emplace_back(&conv->input());
            context.inputs.emplace_back(&conv->weights());
            context.inputs.emplace_back(&conv->bias());
            context.outputs.emplace_back(&conv->output());

            context.matched_nodes.emplace_back(conv);
            return true;
        }
    }

    return false;
}

void pad_conv_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto &weights = *context.inputs[1]->connection();
    auto &bias = *context.inputs[2]->connection();
    auto inputs = context.outputs[0]->connections();
    auto &conv = static_cast<conv2d &>(*context.matched_nodes[0]);

    xt::svector<padding> data_pad_padding;
    data_pad_padding.push_back(padding { 0, 0 });
    data_pad_padding.push_back(padding { 0, 0 });
    for (int i = 2; i < 4; i++)
    {
        if (output.shape()[i] < 4)
        {
            int pad_before = 4 - output.shape()[i];
            data_pad_padding.push_back(padding { 0, pad_before });
        }
        else
        {
            data_pad_padding.push_back(padding { 0, 0 });
        }
    }
    auto data_pad = context.graph.emplace<pad>(output.type(), output.shape(), data_pad_padding, pad_mode_t::pad_constant, (scalar)0);
    data_pad->name(conv.name() + "(pre_pad)");
    auto new_conv = context.graph.emplace<conv2d>(data_pad->output().shape(), weights.shape(),
        conv.groups(), conv.padding_h(), conv.padding_w(), conv.stride_h(), conv.stride_w(),
        conv.dilation_h(), conv.dilation_w(), conv.fused_activation());
    new_conv->name(conv.name());
    new_conv->input().connect(data_pad->output());
    new_conv->weights().connect(weights);
    new_conv->bias().connect(bias);
    axis_t data_slice_begin, data_slice_after;
    for (int i = 0; i < data_pad_padding.size(); i++)
    {
        data_slice_begin.push_back(0);
        data_slice_after.push_back(conv.output().shape()[i]);
    }
    auto data_slice = context.graph.emplace<slice>(new_conv->output().type(), new_conv->output().shape(), data_slice_begin, data_slice_after);
    data_slice->name(conv.name() + "(post_slice)");
    data_slice->input().connect(new_conv->output());

    data_pad->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(data_slice->output());
}