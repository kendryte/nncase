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
#include <hlir/ops/concat.h>
#include <hlir/ops/constant.h>
#include <hlir/ops/dequantize.h>
#include <hlir/ops/pad.h>
#include <hlir/ops/reduce.h>
#include <hlir/ops/reshape.h>
#include <hlir/ops/resize_image.h>
#include <hlir/ops/strided_slice.h>
#include <hlir/ops/transpose.h>
#include <hlir/transforms/neutral/dequantize_motion.h>
#include <hlir/visitor.h>

using namespace nncase;
using namespace nncase::hlir;
using namespace nncase::hlir::transforms;

bool dequantize_pad_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto p = try_get_direct_child<pad>(*deq))
        {
            context.matched_nodes.emplace_back(deq);
            context.matched_nodes.emplace_back(p);

            context.inputs.emplace_back(&deq->input());
            context.outputs.emplace_back(&p->output());

            return true;
        }
    }

    return false;
}

void dequantize_pad_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);
    auto &old_p = static_cast<pad &>(*context.matched_nodes[1]);
    auto q_param = old_deq.quant_param();
    auto pad_value = std::clamp((int32_t)std::round(old_p.pad_value().as<float>() * q_param.scale + q_param.zero_point), 0, 255);

    auto p = context.graph.emplace<pad>(dt_uint8, old_deq.input().shape(), old_p.paddings(), (uint8_t)pad_value);
    p->name(old_p.name());
    auto deq = context.graph.emplace<dequantize>(p->output().shape(), q_param);
    deq->name(old_deq.name());
    deq->input().connect(p->output());

    p->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}

bool dequantize_transpose_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto tp = try_get_direct_child<transpose>(*deq))
        {
            context.matched_nodes.emplace_back(deq);
            context.matched_nodes.emplace_back(tp);

            context.inputs.emplace_back(&deq->input());
            context.outputs.emplace_back(&tp->output());

            return true;
        }
    }

    return false;
}

void dequantize_transpose_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);
    auto &old_tp = static_cast<transpose &>(*context.matched_nodes[1]);

    auto tp = context.graph.emplace<transpose>(dt_uint8, old_deq.input().shape(), old_tp.perm());
    tp->name(old_tp.name());
    auto deq = context.graph.emplace<dequantize>(tp->output().shape(), old_deq.quant_param());
    deq->name(old_deq.name());
    deq->input().connect(tp->output());

    tp->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}

bool dequantize_strided_slice_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto slice = try_get_direct_child<strided_slice>(*deq))
        {
            context.matched_nodes.emplace_back(deq);
            context.matched_nodes.emplace_back(slice);

            context.inputs.emplace_back(&deq->input());
            context.outputs.emplace_back(&slice->output());

            return true;
        }
    }

    return false;
}

void dequantize_strided_slice_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);
    auto &old_slice = static_cast<strided_slice &>(*context.matched_nodes[1]);

    auto slice = context.graph.emplace<strided_slice>(dt_uint8, old_deq.input().shape(), old_slice.begin(), old_slice.end(), old_slice.strides(),
        old_slice.begin_mask(), old_slice.end_mask(), old_slice.ellipsis_mask(), old_slice.new_axis_mask(), old_slice.shrink_axis_mask());
    slice->name(old_slice.name());
    auto deq = context.graph.emplace<dequantize>(slice->output().shape(), old_deq.quant_param());
    deq->name(old_deq.name());
    deq->input().connect(slice->output());

    slice->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}

bool dequantize_resize_image_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto resize = try_get_direct_child<resize_image>(*deq))
        {
            context.matched_nodes.emplace_back(deq);
            context.matched_nodes.emplace_back(resize);

            context.inputs.emplace_back(&deq->input());
            context.outputs.emplace_back(&resize->output());

            return true;
        }
    }

    return false;
}

void dequantize_resize_image_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);
    auto &old_resize = static_cast<resize_image &>(*context.matched_nodes[1]);

    auto resize = context.graph.emplace<resize_image>(dt_uint8, old_resize.mode(), old_deq.input().shape(), old_resize.new_size(), old_resize.align_corners());
    resize->name(old_resize.name());
    auto deq = context.graph.emplace<dequantize>(resize->output().shape(), old_deq.quant_param());
    deq->name(old_deq.name());
    deq->input().connect(resize->output());

    resize->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}

bool dequantize_reshape_motion_transform::on_try_match(node &node, transform_context &context)
{
    if (auto deq = node_cast<dequantize>(node))
    {
        if (auto r = try_get_direct_child<reshape>(*deq))
        {
            context.matched_nodes.emplace_back(deq);
            context.matched_nodes.emplace_back(r);

            context.inputs.emplace_back(&deq->input());
            context.outputs.emplace_back(&r->output());

            return true;
        }
    }

    return false;
}

void dequantize_reshape_motion_transform::process(transform_context &context)
{
    auto &output = *context.inputs[0]->connection();
    auto inputs = context.outputs[0]->connections();

    auto &old_deq = static_cast<dequantize &>(*context.matched_nodes[0]);
    auto &old_r = static_cast<reshape &>(*context.matched_nodes[1]);

    auto r = context.graph.emplace<reshape>(dt_uint8, old_deq.input().shape(), old_r.new_shape());
    r->name(old_r.name());
    auto deq = context.graph.emplace<dequantize>(r->output().shape(), old_deq.quant_param());
    deq->name(old_deq.name());
    deq->input().connect(r->output());

    r->input().connect(output);
    for (auto &in : dup(inputs))
        in->connect(deq->output());
}
